"""Hessian spectroscopy: the reward curvature spectrum through matrix-vector products.

Because a reward is scalar, its Hessian at any site is ``d x d`` and reachable through
Hessian-vector products without ever materializing the matrix. This module holds the numerics:
Lanczos with full reorthogonalization for the top-``k`` eigenvalues, stochastic
Lanczos quadrature (SLQ) for the spectral density when the tail matters, the participation ratio
that turns a spectrum into the effective dimension ``d_eff`` capacity theory needs (A9 / A16),
and the near-zero-curvature ``flat_subspace``.

The numerics are written against a single abstraction, a symmetric linear operator exposed only
through ``matvec``. That is what makes them provable on CPU with no model: pass ``matvec = lambda v:
M @ v`` for a known SPD matrix ``M`` and Lanczos must recover ``M``'s eigenvalues. The same
abstraction wraps ``Runtime.hvp`` so the identical code runs on a real model,
seed-averaged over inputs, with the fp32 accumulation the reward head needs under a bf16 trunk.

Importing this module pulls no torch; torch is imported lazily only inside the HVP operator, at call
time on a real model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
from scipy.linalg import eigh_tridiagonal

from reward_lens.core.errors import NumericsError
from reward_lens.core.evidence import Evidence, make_evidence, register_payload
from reward_lens.core.provenance import Provenance
from reward_lens.core.types import GaugeStatus, Site, SubjectRef

# Marks the dual-use hack-generation surface (RK8). Symbols carrying this are excluded from the
# curated public exports and must be imported explicitly by name.
SENSITIVE = "sensitive:dual-use"

_LANCZOS_BREAKDOWN = 1e-12


# ---------------------------------------------------------------------------
# The symmetric operator abstraction
# ---------------------------------------------------------------------------


class SymmetricOperator:
    """A symmetric linear operator exposed only through matrix-vector products.

    ``matvec`` applies the operator to a single vector; ``to_dense`` reconstructs the full matrix by
    probing with the standard basis (``dim`` matvecs, fp32 accumulate) for the interior-eigenvalue
    work ``flat_subspace`` needs. This is the seam between the pure Lanczos numerics and whatever
    produced the operator: a known matrix in a test, or ``Runtime.hvp`` on a real reward model.
    """

    def __init__(self, matvec: Callable[[np.ndarray], np.ndarray], dim: int):
        self._matvec = matvec
        self.dim = int(dim)

    def matvec(self, v: np.ndarray) -> np.ndarray:
        out = np.asarray(self._matvec(np.ascontiguousarray(v, dtype=np.float64)), dtype=np.float64)
        return out.ravel()

    def to_dense(self) -> np.ndarray:
        """Materialize the ``dim x dim`` matrix by probing the standard basis, then symmetrize.

        Costs ``dim`` matrix-vector products. Cheap for the CPU tests and for offline capacity work
        on a reward head (``d_model`` a few thousand); the spectrum path below never calls this, so
        the top-``k`` spectrum stays materialization-free and within the 8B memory budget.
        """
        m = np.empty((self.dim, self.dim), dtype=np.float64)
        e = np.zeros(self.dim, dtype=np.float64)
        for i in range(self.dim):
            e[i] = 1.0
            m[:, i] = self.matvec(e)
            e[i] = 0.0
        return 0.5 * (m + m.T)


def _runtime_operator(
    runtime: Any,
    batch: Any,
    scalar_fn: Any,
    site: Site,
    dim: int,
) -> SymmetricOperator:
    """Wrap ``Runtime.hvp`` as a `SymmetricOperator`.

    Each ``matvec`` issues one HVP: the double-backprop of the readout scalar at ``site`` applied to
    a single probe vector, averaged over the batch (seed-averaged over inputs).
    torch is imported here and nowhere else in the module, so the spectrum numerics stay torch-free
    until they run on a real model.
    """
    import torch

    def matvec(v: np.ndarray) -> np.ndarray:
        vt = torch.as_tensor(np.ascontiguousarray(v, dtype=np.float32))
        out = runtime.hvp(batch, scalar_fn, at=site, vecs=vt)
        out = torch.as_tensor(out).detach().to("cpu", torch.float64)
        if out.ndim > 1:
            out = out.reshape(out.shape[0], -1).mean(dim=0) if out.shape[0] != dim else out.ravel()
        return out.numpy().ravel()

    return SymmetricOperator(matvec, dim)


def _resolve_operator(
    source: Any,
    site: Site | None,
    dim: int | None,
    scalar_fn: Any,
    batch: Any,
) -> SymmetricOperator:
    """Build a `SymmetricOperator` from whatever the caller passed.

    Accepts, in priority order: an existing `SymmetricOperator`; a plain ``matvec`` callable (the
    synthetic-operator test path, ``dim`` required); a `Runtime` (detected by an ``hvp`` method); or
    a `RewardSignal` (detected by a ``runtime`` attribute). The last two need ``site``, ``batch``,
    and ``scalar_fn`` to define the reward Hessian at that site.
    """
    if isinstance(source, SymmetricOperator):
        return source
    if hasattr(source, "hvp") and not hasattr(source, "runtime"):
        if dim is None or site is None or batch is None or scalar_fn is None:
            raise NumericsError("runtime HVP path needs dim, site, batch, and scalar_fn")
        return _runtime_operator(source, batch, scalar_fn, site, dim)
    if hasattr(source, "runtime"):
        if dim is None or site is None or batch is None or scalar_fn is None:
            raise NumericsError("signal HVP path needs dim, site, batch, and scalar_fn")
        return _runtime_operator(source.runtime, batch, scalar_fn, site, dim)
    if callable(source):
        if dim is None:
            raise NumericsError("matvec operator needs an explicit dim")
        return SymmetricOperator(source, dim)
    raise NumericsError(
        "source must be a SymmetricOperator, a matvec callable, a Runtime (has .hvp), or a "
        f"RewardSignal (has .runtime); got {type(source).__name__}"
    )


# ---------------------------------------------------------------------------
# Lanczos and SLQ (pure numerics)
# ---------------------------------------------------------------------------


def lanczos(
    matvec: Callable[[np.ndarray], np.ndarray] | SymmetricOperator,
    dim: int,
    k: int,
    *,
    n_iter: int | None = None,
    reorth: bool = True,
    seed: int = 0,
    return_vectors: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Lanczos estimate of the ``k`` largest-magnitude eigenvalues of a symmetric operator.

    Full reorthogonalization at every step: the small-``k`` regime is exactly where the classic
    Lanczos loss of orthogonality produces spurious "ghost" eigenvalues, so we pay
    the ``O(m^2 d)`` reorthogonalization to keep the Ritz values clean. Runs ``n_iter`` iterations
    (default ``min(dim, max(2k + 20, 40))``), forms the tridiagonal ``T``, and returns its extreme
    Ritz values sorted by descending magnitude. All accumulation is float64 for a stable tridiagonal
    solve even when the operator returns fp32 (the real HVP case).

    Returns ``(ritz_values, ritz_vectors)`` with ``ritz_vectors`` the ``dim x k`` Ritz vectors when
    ``return_vectors`` is set, else None.
    """
    op = matvec if isinstance(matvec, SymmetricOperator) else SymmetricOperator(matvec, dim)
    if op.dim != dim:
        raise NumericsError(f"operator dim {op.dim} disagrees with declared dim {dim}")
    if k < 1 or k > dim:
        raise NumericsError(f"require 1 <= k <= dim; got k={k}, dim={dim}")
    m = n_iter or min(dim, max(2 * k + 20, 40))
    m = min(m, dim)

    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v)

    vecs = np.empty((m, dim), dtype=np.float64)
    alphas = np.empty(m, dtype=np.float64)
    betas = np.empty(m - 1, dtype=np.float64)

    vecs[0] = v
    w = op.matvec(v)
    alphas[0] = float(v @ w)
    w = w - alphas[0] * v
    used = 1
    for j in range(1, m):
        if reorth:  # full reorthogonalization against the accumulated Krylov basis
            w = w - vecs[:j].T @ (vecs[:j] @ w)
        beta = float(np.linalg.norm(w))
        if beta < _LANCZOS_BREAKDOWN:  # invariant subspace found: stop cleanly
            break
        betas[j - 1] = beta
        v = w / beta
        vecs[j] = v
        w = op.matvec(v)
        alphas[j] = float(v @ w)
        w = w - alphas[j] * v - beta * vecs[j - 1]
        used = j + 1

    a = alphas[:used]
    b = betas[: used - 1]
    if used == 1:
        theta = a.copy()
        s = np.array([[1.0]])
    else:
        theta, s = eigh_tridiagonal(a, b)

    order = np.argsort(np.abs(theta))[::-1][:k]
    ritz_values = theta[order].astype(np.float64)
    ritz_vectors = None
    if return_vectors:
        ritz_vectors = (vecs[:used].T @ s[:, order]).astype(np.float64)
    return ritz_values, ritz_vectors


def slq_density(
    matvec: Callable[[np.ndarray], np.ndarray] | SymmetricOperator,
    dim: int,
    *,
    n_probes: int = 16,
    m_lanczos: int = 40,
    n_grid: int = 128,
    sigma: float | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic Lanczos quadrature estimate of the spectral density.

    Each Rademacher probe produces an ``m_lanczos``-point Gaussian quadrature rule (Ritz values as
    nodes, squared first eigenvector components as weights) for the operator's spectral measure; the
    estimates are averaged over probes and smoothed onto a grid with a Gaussian kernel of bandwidth
    ``sigma``. This is the tool for the heavy-tail diagnostics where the top-``k`` values miss the
    bulk of the mass. Returns ``(grid, density)`` with ``density`` integrating to ``dim``.
    """
    op = matvec if isinstance(matvec, SymmetricOperator) else SymmetricOperator(matvec, dim)
    rng = np.random.default_rng(seed)
    all_nodes: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    for _ in range(n_probes):
        v = rng.integers(0, 2, size=dim).astype(np.float64) * 2 - 1  # Rademacher probe
        v /= np.linalg.norm(v)
        a, b, used = _lanczos_tridiag(op, v, min(m_lanczos, dim))
        if used == 1:
            theta = a[:1]
            tau = np.array([1.0])
        else:
            theta, s = eigh_tridiagonal(a[:used], b[: used - 1])
            tau = s[0, :] ** 2
        all_nodes.append(theta)
        all_weights.append(tau)

    nodes = np.concatenate(all_nodes)
    weights = np.concatenate(all_weights) / n_probes
    lo, hi = float(nodes.min()), float(nodes.max())
    pad = 0.05 * (hi - lo + 1e-9)
    grid = np.linspace(lo - pad, hi + pad, n_grid)
    if sigma is None:
        sigma = max((hi - lo) / max(n_grid, 1) * 2.0, 1e-6)
    diff = (grid[:, None] - nodes[None, :]) / sigma
    kernel = np.exp(-0.5 * diff**2) / (sigma * np.sqrt(2 * np.pi))
    density = (kernel @ weights) * dim
    return grid, density


def _lanczos_tridiag(
    op: SymmetricOperator, v0: np.ndarray, m: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Bare Lanczos tridiagonalization from a given start vector, with full reorthogonalization."""
    dim = op.dim
    vecs = np.empty((m, dim), dtype=np.float64)
    alphas = np.zeros(m, dtype=np.float64)
    betas = np.zeros(max(m - 1, 1), dtype=np.float64)
    v = v0 / np.linalg.norm(v0)
    vecs[0] = v
    w = op.matvec(v)
    alphas[0] = float(v @ w)
    w = w - alphas[0] * v
    used = 1
    for j in range(1, m):
        w = w - vecs[:j].T @ (vecs[:j] @ w)
        beta = float(np.linalg.norm(w))
        if beta < _LANCZOS_BREAKDOWN:
            break
        betas[j - 1] = beta
        v = w / beta
        vecs[j] = v
        w = op.matvec(v)
        alphas[j] = float(v @ w)
        w = w - alphas[j] * v - beta * vecs[j - 1]
        used = j + 1
    return alphas, betas, used


# ---------------------------------------------------------------------------
# participation ratio (d_eff)
# ---------------------------------------------------------------------------


def participation_ratio(spectrum: Any) -> float:
    """Participation ratio of a curvature spectrum: the effective dimension ``d_eff``.

    ``PR = (sum |lambda_i|)^2 / sum lambda_i^2`` (A9 / A16). It counts how many eigenvalues
    carry the curvature: ``n`` equal eigenvalues give ``PR = n``, a single dominant eigenvalue gives
    ``PR = 1``. Magnitudes are used so a mixed-sign (indefinite) Hessian still reports the effective
    number of curved directions. Accepts a `SpectrumResult`, a raw array, or any 1-D sequence.
    """
    if isinstance(spectrum, SpectrumResult):
        lam = np.abs(np.asarray(spectrum.eigenvalues, dtype=np.float64))
    else:
        lam = np.abs(np.asarray(spectrum, dtype=np.float64).ravel())
    s2 = float(np.sum(lam**2))
    if s2 <= 0:
        return 0.0
    return float(np.sum(lam) ** 2 / s2)


# ---------------------------------------------------------------------------
# Evidence payloads
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class SpectrumResult:
    """The payload of `hessian_spectrum`.

    ``eigenvalues`` are the top-``k`` Ritz values (descending magnitude); ``participation_ratio`` is
    the effective dimension over those; ``spectral_density`` holds the SLQ grid and density when the
    tail was requested, else None. ``tail`` summarizes the extremes and the trace estimate the
    density integrates to.
    """

    eigenvalues: np.ndarray
    method: str
    k: int
    dim: int
    participation_ratio: float
    n_starts: int
    converged: bool
    tail: dict[str, float]
    spectral_density: dict[str, Any] | None = None


@register_payload
@dataclass
class SubspaceResult:
    """The payload of `flat_subspace`.

    ``basis`` is the ``d x m`` orthonormal set of near-zero-curvature directions (``|lambda| < tol``)
    and ``curvatures`` their eigenvalues. These are the flat directions along which the reward is
    locally indifferent, the gauge-adjacent subspace capacity theory and surgery both consume.
    """

    basis: np.ndarray
    curvatures: np.ndarray
    tol: float
    dim: int
    n_flat: int


# ---------------------------------------------------------------------------
# hessian_spectrum / flat_subspace
# ---------------------------------------------------------------------------


def hessian_spectrum(
    source: Any,
    view: Any | None = None,
    site: Site | None = None,
    *,
    k: int = 64,
    method: Literal["lanczos", "slq"] = "lanczos",
    dim: int | None = None,
    scalar_fn: Any | None = None,
    batch: Any | None = None,
    n_starts: int = 1,
    density: bool = False,
    seed: int = 0,
    subject: SubjectRef | None = None,
    provenance: Provenance | None = None,
) -> Evidence[SpectrumResult]:
    """Top-``k`` reward-Hessian spectrum at a site, materialization-free via HVPs.

    ``source`` is a `RewardSignal`, a `Runtime`, a `SymmetricOperator`, or a plain ``matvec``
    callable (the synthetic-operator test path); the reward Hessian is the
    double-backprop of the readout scalar at ``site``, so nothing larger than ``dim`` probe vectors
    is ever held, which is what keeps the spectrum within the 8B memory budget. Lanczos with full
    reorthogonalization gives the extreme eigenvalues; ``n_starts > 1`` seed-averages the Ritz
    values across random starts for robustness; ``density=True`` adds an SLQ spectral-density
    estimate for the tail.

    The spectrum is INVARIANT (a property of one signal at one site, not a cross-signal comparison),
    so no frame is required.
    """
    op = _resolve_operator(source, site, dim, scalar_fn, batch)
    d = op.dim
    k = min(k, d)

    if method not in ("lanczos", "slq"):
        raise NumericsError(f"unknown method '{method}'; use 'lanczos' or 'slq'")

    starts = max(1, n_starts)
    stacked = np.empty((starts, k), dtype=np.float64)
    for s in range(starts):
        vals, _ = lanczos(op, d, k, seed=seed + s)
        stacked[s] = vals
    eigenvalues = stacked.mean(axis=0)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[order]
    seed_spread = float(np.mean(np.std(stacked, axis=0))) if starts > 1 else 0.0

    dens = None
    if density or method == "slq":
        grid, dvals = slq_density(op, d, seed=seed)
        dens = {"grid": grid, "density": dvals}

    tail = {
        "lambda_max": float(eigenvalues[0]),
        "lambda_min_topk": float(eigenvalues[-1]),
        "abs_max": float(np.max(np.abs(eigenvalues))),
        "trace_topk": float(np.sum(eigenvalues)),
    }
    value = SpectrumResult(
        eigenvalues=eigenvalues.astype(np.float32),
        method=method,
        k=int(k),
        dim=int(d),
        participation_ratio=participation_ratio(eigenvalues),
        n_starts=starts,
        converged=bool(np.all(np.isfinite(eigenvalues))),
        tail=tail,
        spectral_density=dens,
    )
    from reward_lens.core.evidence import Uncertainty

    subj = subject or SubjectRef(extra={"site": str(site) if site else None})
    return make_evidence(
        observable="geometry.hessian_spectrum",
        observable_version="1",
        subject=subj,
        value=value,
        uncertainty=Uncertainty(seed_spread=seed_spread or None, method="lanczos-seed-average"),
        gauge=GaugeStatus.INVARIANT,
        provenance=provenance,
    )


def flat_subspace(
    source: Any,
    view: Any | None = None,
    site: Site | None = None,
    *,
    tol: float,
    dim: int | None = None,
    scalar_fn: Any | None = None,
    batch: Any | None = None,
    subject: SubjectRef | None = None,
    provenance: Provenance | None = None,
) -> Evidence[SubspaceResult]:
    """The near-zero-curvature subspace of the reward Hessian at a site.

    The flat directions are interior eigenvalues (near zero), which Lanczos cannot isolate reliably,
    so this materializes the operator by probing (``dim`` HVPs, fp32 accumulate) and takes a dense
    symmetric eigendecomposition, returning the eigenvectors with ``|lambda| < tol``. That is exact
    and cheap for a reward head's ``d_model`` and is honest about its cost; the top-``k`` spectrum
    path above never materializes. The subspace is INVARIANT, so no frame is required.
    """
    op = _resolve_operator(source, site, dim, scalar_fn, batch)
    dense = op.to_dense()
    evals, evecs = np.linalg.eigh(dense)
    mask = np.abs(evals) < tol
    basis = np.ascontiguousarray(evecs[:, mask].astype(np.float32))
    curv = evals[mask].astype(np.float32)
    value = SubspaceResult(
        basis=basis,
        curvatures=curv,
        tol=float(tol),
        dim=int(op.dim),
        n_flat=int(basis.shape[1]),
    )
    subj = subject or SubjectRef(extra={"site": str(site) if site else None})
    return make_evidence(
        observable="geometry.flat_subspace",
        observable_version="1",
        subject=subj,
        value=value,
        gauge=GaugeStatus.INVARIANT,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# gradient_ascent_probe (SENSITIVE, dual-use; excluded from public exports)
# ---------------------------------------------------------------------------


@dataclass
class AscentTrace:
    """The trace of a gradient-ascent hack probe. SENSITIVE / dual-use.

    ``reward_curve`` is the reward proxy at each step; ``excited_directions`` are the top internal
    directions the ascent moved along (left singular vectors of the accumulated updates), the
    defensive payload that feeds surgery targets and forecast validation; ``tokens`` holds the
    discretized token ids when an embedding table was supplied. This artifact is marked sensitive and
    excluded from public artifact exports by default (RK8).
    """

    reward_curve: np.ndarray
    excited_directions: np.ndarray
    final_embeddings: np.ndarray
    tokens: list[int] = field(default_factory=list)
    steps: int = 0
    sensitivity: str = SENSITIVE


def gradient_ascent_probe(
    grad_fn: Callable[[np.ndarray], np.ndarray],
    seed_embeddings: np.ndarray,
    steps: int,
    *,
    constraint: float = 1.0,
    lr: float = 0.1,
    reward_fn: Callable[[np.ndarray], float] | None = None,
    embedding_matrix: np.ndarray | None = None,
    top_directions: int = 8,
) -> Evidence[AscentTrace]:
    """The RL-free hack generator: ascend the reward gradient in embedding space.

    SENSITIVE / dual-use (RK8). Ascends ``grad_fn`` (the reward gradient in embedding space, wired to
    ``Runtime.grad``) under a norm-ball ``constraint``, records which internal directions the ascent
    excites, and, when an ``embedding_matrix`` is given, discretizes back to tokens by nearest
    neighbour (GCG-style). Its defensive purpose is to feed surgery targets, forecast validation, and
    the hydra experiment; its output is marked sensitive and is not in the curated public exports.
    This is deliberately a thin, model-agnostic core against a gradient callable; the full
    token-space search is wired at the loop layer where the signal's embedding table lives.
    """
    emb = np.array(seed_embeddings, dtype=np.float64)
    origin = emb.copy()
    curve = np.empty(steps, dtype=np.float64)
    updates = []
    for t in range(steps):
        g = np.asarray(grad_fn(emb), dtype=np.float64)
        updates.append(g.ravel())
        emb = emb + lr * g
        delta = emb - origin  # project back onto the norm ball around the seed
        norm = np.linalg.norm(delta)
        if norm > constraint:
            emb = origin + delta * (constraint / norm)
        curve[t] = float(reward_fn(emb)) if reward_fn is not None else float(np.linalg.norm(g))

    upd = np.asarray(updates, dtype=np.float64)
    if upd.size and upd.shape[0] >= 1:
        _, _, vh = np.linalg.svd(upd, full_matrices=False)
        excited = vh[: min(top_directions, vh.shape[0])].astype(np.float32)
    else:
        excited = np.zeros((0, emb.reshape(-1).shape[0]), dtype=np.float32)

    tokens: list[int] = []
    if embedding_matrix is not None:
        table = np.asarray(embedding_matrix, dtype=np.float64)
        seq = emb.reshape(-1, table.shape[1]) if emb.ndim > 1 else emb.reshape(1, -1)
        for row in seq:
            tokens.append(int(np.argmin(np.linalg.norm(table - row, axis=1))))

    value = AscentTrace(
        reward_curve=curve,
        excited_directions=excited,
        final_embeddings=emb.astype(np.float32),
        tokens=tokens,
        steps=int(steps),
    )
    return make_evidence(
        observable="geometry.gradient_ascent_probe",
        observable_version="1",
        subject=SubjectRef(extra={"sensitivity": SENSITIVE}),
        value=value,
        gauge=GaugeStatus.INVARIANT,
        provenance=provenance_with_sensitive(),
    )


def provenance_with_sensitive() -> Provenance:
    """Provenance stamped so the store and cards know this artifact is sensitive/dual-use (RK8)."""
    return Provenance(extra={"sensitivity": SENSITIVE, "exclude_from_public_exports": True})


__all__ = [
    "SymmetricOperator",
    "lanczos",
    "slq_density",
    "participation_ratio",
    "hessian_spectrum",
    "flat_subspace",
    "SpectrumResult",
    "SubspaceResult",
    # gradient_ascent_probe and AscentTrace are intentionally NOT exported (SENSITIVE, RK8).
]

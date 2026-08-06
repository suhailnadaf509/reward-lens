"""LEACE affine concept erasure as an Intervention.

LEACE (Least-squares Concept Erasure; Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff,
Biderman 2023, "LEACE: Perfect linear concept erasure in closed form") fits, in closed form, the
affine map that provably removes *all* linear information about a concept from a feature while
making the least-squares-minimal edit. It is the defensive surgery the design calls for: given
capture matrices ``X`` (features at a site) and ``Z`` (concept labels or a concept subspace), the
eraser ``r(x) = x - P (x - mu)`` guarantees ``Cov(r(X), Z) = 0`` and, among all affine maps with
that guarantee, moves the feature the least in the covariance metric.

The closed form is a whitening argument, and getting the order of the whitening right is the whole
game. With ``Sigma = Cov(X)`` and its symmetric PSD square root ``W = Sigma^{1/2}``, whiten the
feature (``u = W^+ (x - mu)``, isotropic covariance), erase in the isotropic space by orthogonally
projecting out the directions that carry concept covariance there (the column space of
``W^+ Sigma_XZ``), then unwhiten. That yields

    P = W  Pi  W^+ ,   Pi = orthogonal projection onto col(W^+ Sigma_XZ),   W = Sigma^{1/2}.

The projection lives *between* the whitening and its inverse, which is what makes the edit minimal
under the ``Sigma`` metric rather than the raw Euclidean one; the transpose-ordered arrangement
``W^+ Pi W`` also erases the concept but is not least-squares-minimal, and the moment-level tests
pin the minimal form to a hand-solved optimum so the distinction cannot silently regress.

Three surfaces are exposed. Chosen-layers and all-layers erasure mount the affine
map ``r`` at the residual sites the caller names, through the runtime's single hook path, via
:class:`LeaceErasure`. Head-only erasure instead projects the reward direction ``w_r`` against the
concept subspace with :meth:`Eraser.apply_to_readout`, which reproduces the erased-feature reward up
to the immaterial per-prompt constant (the reward gauge already quotients out that shift).

An eraser is persisted with the ids of the data it was fit on (design rule R8: directions and
erasers always carry their training-data provenance), so an eraser can never be applied while its
provenance is lost. The certificate that certifies an eraser lives in ``certify.py``; an eraser
without a certificate is EXPLORATORY by gate 1, which ``certify.eraser_evidence``
makes explicit.

This module imports no torch at module scope; the mount hook imports it lazily, so fitting and
persisting an eraser on captured numpy matrices needs no torch at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from reward_lens.core.types import Site, content_hash, hash_bytes
from reward_lens.interventions.base import CompiledIntervention, MountHook

if TYPE_CHECKING:
    import torch

    from reward_lens.signals.base import RewardSignal

# Relative tolerance for treating an eigenvalue of Sigma (or a singular value of the whitened
# concept covariance) as numerical zero, as a fraction of the largest. The whitening inverts only
# the directions the data actually spans; the rest are left alone rather than amplified.
_RCOND = 1e-10


# ---------------------------------------------------------------------------
# The closed form
# ---------------------------------------------------------------------------


def _symmetric_sqrt_and_pinv(sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(W, W_pinv)`` with ``W = Sigma^{1/2}`` and ``W_pinv = (Sigma^{1/2})^+``.

    ``sigma`` is a symmetric PSD covariance. The symmetric square root is formed from the
    eigendecomposition; eigenvalues that are numerically zero or slightly negative (a rounding
    artifact of a PSD matrix) are clamped to zero and left out of the inverse, so ``W_pinv``
    inverts only the subspace the features actually span. Both matrices are symmetric.
    """
    sigma = 0.5 * (sigma + sigma.T)  # symmetrize away rounding asymmetry before eigh
    evals, evecs = np.linalg.eigh(sigma)
    evals = np.clip(evals, 0.0, None)
    sqrt_evals = np.sqrt(evals)
    cutoff = _RCOND * float(sqrt_evals.max()) if sqrt_evals.size else 0.0
    inv_sqrt = np.where(
        sqrt_evals > cutoff, np.divide(1.0, sqrt_evals, where=sqrt_evals > cutoff), 0.0
    )
    w = (evecs * sqrt_evals) @ evecs.T
    w_pinv = (evecs * inv_sqrt) @ evecs.T
    return w, w_pinv


def _orthonormal_range(m: np.ndarray) -> np.ndarray:
    """An orthonormal basis for the column space of ``m`` (the significant left singular vectors).

    Singular values below ``_RCOND`` of the largest are treated as zero, so a rank-deficient or
    all-zero concept covariance (no linear concept present) yields an empty basis and hence the
    identity eraser rather than a spurious projection.
    """
    if m.size == 0 or not np.any(np.abs(m) > 0):
        return np.zeros((m.shape[0], 0), dtype=np.float64)
    u, s, _ = np.linalg.svd(m, full_matrices=False)
    if s.size == 0:
        return np.zeros((m.shape[0], 0), dtype=np.float64)
    rank = int(np.sum(s > _RCOND * float(s[0])))
    return np.ascontiguousarray(u[:, :rank])


def leace_matrix(sigma: np.ndarray, sigma_xz: np.ndarray) -> tuple[np.ndarray, int]:
    """The LEACE projection ``P = W Pi W^+`` from second moments (the closed form's core).

    ``sigma`` is ``Cov(X)`` (``d x d``); ``sigma_xz`` is ``Cov(X, Z)`` (``d x k``). Returns
    ``(P, rank)`` where ``rank`` is the dimension of the erased concept subspace. The eraser built
    from ``P`` satisfies ``(I - P) sigma_xz = 0`` (perfect linear erasure) exactly, and ``P`` is an
    idempotent projection, both to machine precision, because they are algebraic identities of this
    construction rather than statistical claims. This function takes moments, not data, so the
    same closed form is testable against analytically specified ``sigma`` and ``sigma_xz`` with no
    sampling error in the way.
    """
    w, w_pinv = _symmetric_sqrt_and_pinv(sigma)
    whitened_concept = w_pinv @ sigma_xz  # concept covariance directions in the isotropic space
    basis = _orthonormal_range(whitened_concept)  # d x r
    if basis.shape[1] == 0:
        return np.zeros_like(sigma), 0
    pi = basis @ basis.T
    p = w @ pi @ w_pinv
    return p, basis.shape[1]


# ---------------------------------------------------------------------------
# The fitted eraser (a persisted artifact, R8)
# ---------------------------------------------------------------------------


@dataclass
class Eraser:
    """A fitted LEACE affine eraser, persisted with its fit-data provenance (R8).

    ``P`` is the ``d x d`` projection and ``mu`` the ``d`` fit-data mean, so the erasure is the
    affine map ``r(x) = x - P (x - mu)``. ``rank`` is the dimension of the erased concept subspace.
    ``fit_data_id`` and ``concept_id`` are the ids of the feature matrix and the concept the eraser
    was fit on; carrying them on the artifact is design rule R8, and it is what lets a certificate
    (``certify.py``) name the exact data an eraser was trained on. ``sites`` records the residual
    sites the eraser is intended for, informational metadata for the head-only and chosen-layers
    surfaces.

    The matrices are kept in float64: the eraser's defining properties (idempotence, exact linear
    erasure, minimal edit) are proven to ~1e-6 or tighter, and float32 storage would put that
    tolerance out of reach for no memory saving that matters at a single site.
    """

    P: np.ndarray
    mu: np.ndarray
    rank: int
    dim: int
    method: str = "leace"
    fit_data_id: str | None = None
    concept_id: str | None = None
    sites: tuple[Site, ...] = ()

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Erase a feature matrix ``X`` (``n x d``, rows are samples): ``r(X) = X - (X - mu) P^T``.

        Accepts a single vector ``(d,)`` or a batch ``(n, d)``; returns the same shape in float64.
        This is the numpy path used to fit certificates and to prove the guarantee on captured
        matrices; the torch mount hook in :class:`LeaceErasure` applies the identical map on-device.
        """
        arr = np.asarray(x, dtype=np.float64)
        single = arr.ndim == 1
        mat = arr[None, :] if single else arr
        out = mat - (mat - self.mu) @ self.P.T
        return out[0] if single else out

    def apply_to_readout(self, w_r: np.ndarray) -> np.ndarray:
        """Project a reward direction ``w_r`` against the concept subspace (the head-only surface).

        The reward read from erased features, ``w_r . r(x)``, equals ``w_r' . x`` for the projected
        readout ``w_r' = (I - P)^T w_r = w_r - P^T w_r``, up to a per-sample-constant offset the
        reward gauge already quotients out (per-prompt shifts). So editing the reward direction
        once reproduces feature erasure everywhere without mounting a hook, which is the head-only
        erasure surface. Returns a float64 vector ``(d,)``.
        """
        w = np.asarray(w_r, dtype=np.float64)
        return w - self.P.T @ w

    def fingerprint(self) -> str:
        """A stable content id for the eraser, folding in its fit-data provenance (R8).

        Derived from the projection and mean bytes together with the fit-data and concept ids, so
        two erasers fit from the same data and concept share an id and an eraser can never be
        confused with one trained on different data. This is the cache-key and provenance component
        the Intervention protocol requires.
        """
        p_digest = hash_bytes(np.ascontiguousarray(self.P, dtype=np.float64).tobytes(), "erase")
        mu_digest = hash_bytes(np.ascontiguousarray(self.mu, dtype=np.float64).tobytes(), "erase")
        return content_hash(
            {
                "method": self.method,
                "dim": int(self.dim),
                "rank": int(self.rank),
                "P": p_digest,
                "mu": mu_digest,
                "fit_data_id": self.fit_data_id,
                "concept_id": self.concept_id,
                "sites": [str(s) for s in self.sites],
            },
            "iv",
        )


def fit_leace(
    X: np.ndarray,
    Z: np.ndarray,
    *,
    fit_data_id: str | None = None,
    concept_id: str | None = None,
    sites: tuple[Site, ...] = (),
) -> Eraser:
    """Fit the LEACE eraser from captured features ``X`` and a concept ``Z``.

    ``X`` is ``(n, d)`` (features at a site). ``Z`` is the concept: a length-``n`` vector of labels
    for a single binary or scalar concept, or an ``(n, k)`` matrix whose columns span a
    ``k``-dimensional concept subspace (one-hot class indicators, or several concepts at once). The
    returned :class:`Eraser` removes *all* linear information about every direction in
    ``span(Cov(X, Z))``.

    Only second moments enter, so the fit is a closed form with no optimization loop: the feature
    covariance ``Sigma`` and the cross-covariance ``Sigma_XZ`` are formed, and
    :func:`leace_matrix` returns the projection. The fit-data and concept ids are stored on the
    eraser (R8); pass them from the capture provenance so the persisted eraser names its training
    data.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be (n, d); got shape {X.shape}")
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    if Z.shape[0] != X.shape[0]:
        raise ValueError(f"X and Z must share the sample axis; got {X.shape[0]} vs {Z.shape[0]}")

    n, d = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu
    Zc = Z - Z.mean(axis=0)
    sigma = (Xc.T @ Xc) / n
    sigma_xz = (Xc.T @ Zc) / n
    p, rank = leace_matrix(sigma, sigma_xz)
    return Eraser(
        P=p,
        mu=mu,
        rank=int(rank),
        dim=int(d),
        fit_data_id=fit_data_id,
        concept_id=concept_id,
        sites=tuple(sites),
    )


# ---------------------------------------------------------------------------
# The Intervention wrapper (chosen-layers / all-layers surfaces)
# ---------------------------------------------------------------------------


@dataclass
class LeaceErasure:
    """Mount a fitted :class:`Eraser` at one or more sites as an Intervention.

    ``sites`` selects the surface: a single residual site for a targeted edit, several named
    residual sites for the chosen-layers surface, or every residual site the caller enumerates for
    the all-layers surface. Compilation resolves the eraser into one affine mount hook per site;
    the hooks install through the runtime's single mounting path, exactly as patches and steers do,
    so an erased forward pass is measured by any Observable unchanged and the erasure fingerprint is
    carried into the intervened Evidence's SubjectRef. The head-only surface is not a hook; it is
    :meth:`Eraser.apply_to_readout` applied to the reward direction.

    This wraps the frozen :class:`~reward_lens.interventions.base.Intervention` protocol: the map is
    signal-independent (it is an affine function of the activation), so ``compile`` ignores the
    signal and the sites are architecture-resolved by the runner's SiteMap at mount time, the same
    contract ``patch.py`` follows.
    """

    eraser: Eraser
    sites: tuple[Site, ...]
    id: str = "leace_erasure"

    def fingerprint(self) -> str:
        """The cache-key and provenance component: the eraser's id plus the mounted sites."""
        return content_hash(
            {
                "kind": "leace_erasure",
                "eraser": self.eraser.fingerprint(),
                "sites": [str(s) for s in self.sites],
            },
            "iv",
        )

    def _hook(self) -> MountHook:
        p = np.ascontiguousarray(self.eraser.P, dtype=np.float64)
        mu = np.ascontiguousarray(self.eraser.mu, dtype=np.float64)

        def apply(hidden: "torch.Tensor", _ctx: dict) -> "torch.Tensor":
            import torch

            pt = torch.as_tensor(p, dtype=hidden.dtype, device=hidden.device)
            mut = torch.as_tensor(mu, dtype=hidden.dtype, device=hidden.device)
            # r(x) = x - (x - mu) P^T, broadcasting mu over the leading (batch, position) axes.
            return hidden - (hidden - mut) @ pt.T

        return apply

    def compile(self, signal: "RewardSignal | None" = None) -> CompiledIntervention:
        """Resolve into one affine mount hook per site (the Intervention protocol)."""
        del signal  # the affine map is signal-independent; sites are resolved by the runner
        hook = self._hook()
        return CompiledIntervention(
            fingerprint=self.fingerprint(),
            mounts={site: hook for site in self.sites},
            meta={
                "kind": "leace_erasure",
                "rank": int(self.eraser.rank),
                "sites": [str(s) for s in self.sites],
                "fit_data_id": self.eraser.fit_data_id,
                "concept_id": self.eraser.concept_id,
            },
        )


__all__ = [
    "Eraser",
    "fit_leace",
    "leace_matrix",
    "LeaceErasure",
]

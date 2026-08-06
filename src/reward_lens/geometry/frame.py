"""Frames: the per-(site, corpus) whitening artifact that fixes the reward gauge.

Reward models trained on preference comparisons are identified only up to per-prompt shifts,
positive affine transforms, and directions unconstrained on the data distribution (invariant I3). A
raw-coordinate comparison of two reward vectors conflates a coordinate change with a functional
change; that is exactly the E19 ``cos = 0.005`` failure. The
`Frame` is the object that fixes the gauge: it holds the reference distribution's mean and the
shrinkage-regularized square roots of its activation covariance, so a reward direction can be
whitened into a canonical coordinate system shared across signals (see ``canonical.py``).

A frame is defined per (site, corpus): whitening is meaningful only relative to where in the
network it is read and which reference distribution it is read on. Frames are shared across
studies, so the Atlas can publish standard frames and external users compare against the same
reference. The frame carries a content fingerprint (its `FrameID`) derived from structural
content, not raw float payloads, so two fits of the same distribution at the same site land on
the same id.

Numerics discipline: frames refuse fp16 inputs. Anything entering a covariance or a whitening must
be fp32, because the small eigenvalues that define the null subspace are precisely where fp16
rounding destroys the signal. ``fit_frame`` raises rather than
silently upcasting, so a caller who passed fp16 activations learns immediately.

This module is pure numpy/scipy/scikit-learn and imports no torch; a frame is fully testable on
CPU with a synthetic activation matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reward_lens.core.errors import NumericsError
from reward_lens.core.evidence import Evidence, make_evidence, register_payload
from reward_lens.core.provenance import Provenance
from reward_lens.core.types import (
    DatasetID,
    FrameID,
    GaugeStatus,
    Site,
    SubjectRef,
    content_hash,
)

# A shrunk covariance eigenvalue below this absolute floor would make sigma_inv_sqrt blow up;
# the Ledoit-Wolf shrinkage keeps eigenvalues away from zero, but we clamp defensively.
_EIG_FLOOR = 1e-12

# Default: a sample-covariance eigenvalue below this fraction of the top eigenvalue marks a
# data-null direction (a direction the reference distribution barely varies along, hence a gauge
# direction whose reward loading is unconstrained by preferences). Tunable per fit.
_DEFAULT_NULL_VAR_RATIO = 1e-3

# When margins are supplied, a candidate null direction is rejected if the reward margin varies
# along it more than this fraction of the maximal per-direction margin gradient. This is the
# "small-|d margin|" refinement.
_DEFAULT_NULL_GRAD_RATIO = 5e-2


def _dtype_str(x: Any) -> str:
    """Return the dtype of a numpy array or torch tensor as a string, without importing torch."""
    return str(getattr(x, "dtype", None))


def _refuse_fp16(x: Any, name: str) -> None:
    """Raise if ``x`` is a half-precision array/tensor (frames refuse fp16)."""
    dt = _dtype_str(x)
    if "float16" in dt or "bfloat16" in dt or "half" in dt:
        raise NumericsError(
            f"{name} is {dt}; frames refuse half precision. The small covariance eigenvalues that "
            "define the null subspace are exactly where fp16 rounding destroys the signal. Pass "
            "fp32 activations."
        )


def _to_fp32(x: Any, name: str) -> np.ndarray:
    """Convert a numpy array or torch tensor to a contiguous fp32 numpy array, refusing fp16."""
    _refuse_fp16(x, name)
    if hasattr(x, "detach"):  # torch tensor, detected by duck typing to stay torch-free
        x = x.detach().to("cpu").numpy()
    arr = np.asarray(x)
    if arr.dtype == np.float16:
        _refuse_fp16(arr, name)
    return np.ascontiguousarray(arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# The Frame
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """The gauge-fixing whitening artifact for one (site, corpus).

    ``mean`` is the reference distribution's activation mean; ``sigma_sqrt`` and ``sigma_inv_sqrt``
    are the symmetric square roots of the shrinkage-regularized activation covariance. The canonical
    form of a reward direction is ``Sigma^{1/2} w`` normalized (``canonical.py``), which weights
    directions by how much the reference distribution varies along them and kills directions it does
    not vary along at all. ``null_basis`` holds those data-null / gauge directions explicitly (an
    orthonormal ``d x m`` matrix, or None when none were estimated) so canonicalization can quotient
    them exactly rather than relying on the shrinkage floor. ``estimator_meta`` records the fit
    provenance: sample count, shrinkage intensity, eigenspectrum summary, and the fp32 assertion.

    All arrays are fp32. The frame is frozen and carries a content-derived `FrameID` so it is a
    stable, shareable, per-(site, corpus) artifact.
    """

    id: FrameID
    site: Site
    corpus: DatasetID | None
    mean: np.ndarray
    sigma_sqrt: np.ndarray
    sigma_inv_sqrt: np.ndarray
    null_basis: np.ndarray | None
    estimator_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def d(self) -> int:
        """Ambient activation dimension the frame whitens in."""
        return int(self.mean.shape[0])

    @property
    def null_dim(self) -> int:
        """Number of estimated data-null / gauge directions (0 when ``null_basis`` is None)."""
        return 0 if self.null_basis is None else int(self.null_basis.shape[1])


@register_payload
@dataclass
class FrameArtifact:
    """The serializable, content-addressed payload form of a `Frame`.

    A `Frame` holds a live `Site` object; this artifact holds its canonical dict plus the fp32
    arrays, so it round-trips exactly through the evidence store's value codec and can be persisted
    and shared. ``frame_evidence`` wraps one of these in an `Evidence` so a fitted frame is a
    first-class, provenance-carrying store citizen.
    """

    frame_id: str
    site: dict[str, Any]
    corpus: str | None
    mean: np.ndarray
    sigma_sqrt: np.ndarray
    sigma_inv_sqrt: np.ndarray
    null_basis: np.ndarray | None
    estimator_meta: dict[str, Any]


# ---------------------------------------------------------------------------
# Ledoit-Wolf shrinkage covariance
# ---------------------------------------------------------------------------


def _ledoit_wolf(x_centered: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage covariance of centered data, toward the scaled identity.

    Prefers ``sklearn.covariance.LedoitWolf`` when available (the reference implementation) and
    falls back to the closed-form estimator of Ledoit & Wolf (2004) otherwise, so the frame is
    computable without scikit-learn. The shrinkage target is ``mu * I`` with ``mu`` the mean
    eigenvalue; the returned intensity ``lambda`` is the optimal convex weight on the target. The
    shrinkage is what keeps the covariance invertible (hence ``sigma_inv_sqrt`` finite) even when
    ``n < d`` or the distribution has a genuine null subspace.

    Returns ``(covariance, shrinkage)`` with covariance a ``d x d`` fp32 SPD matrix.
    """
    n, d = x_centered.shape
    try:
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf(assume_centered=True).fit(x_centered.astype(np.float64))
        return lw.covariance_.astype(np.float32), float(lw.shrinkage_)
    except Exception:
        pass

    # Closed-form fallback (Ledoit & Wolf 2004, "A well-conditioned estimator...").
    x = x_centered.astype(np.float64)
    s = (x.T @ x) / n  # MLE sample covariance (biased); the shrinkage target is built from it
    mu = float(np.trace(s) / d)
    delta_sq = float(np.sum((s - mu * np.eye(d)) ** 2))
    # beta^2: the average squared Frobenius fluctuation of the per-sample rank-one covariances.
    beta_sq = 0.0
    for k in range(n):
        xk = x[k][:, None]
        beta_sq += float(np.sum((xk @ xk.T - s) ** 2))
    beta_sq /= n * n
    beta_sq = min(beta_sq, delta_sq)
    shrinkage = 0.0 if delta_sq <= 0 else float(beta_sq / delta_sq)
    cov = shrinkage * mu * np.eye(d) + (1.0 - shrinkage) * s
    return cov.astype(np.float32), shrinkage


def _symmetric_roots(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric square root and inverse square root of an SPD matrix via eigendecomposition.

    Returns ``(sigma_sqrt, sigma_inv_sqrt, eigenvalues)`` with eigenvalues ascending. Eigenvalues
    are clamped at ``_EIG_FLOOR`` so a numerically tiny (but positive after shrinkage) direction
    cannot make the inverse square root blow up.
    """
    cov = 0.5 * (cov + cov.T)  # enforce exact symmetry before eigh
    evals, evecs = np.linalg.eigh(cov.astype(np.float64))
    safe = np.clip(evals, _EIG_FLOOR, None)
    sqrt = (evecs * np.sqrt(safe)) @ evecs.T
    inv_sqrt = (evecs * (1.0 / np.sqrt(safe))) @ evecs.T
    return sqrt.astype(np.float32), inv_sqrt.astype(np.float32), evals


def _estimate_null_basis(
    x_centered: np.ndarray,
    margins: np.ndarray | None,
    null_var_ratio: float,
    null_grad_ratio: float,
) -> tuple[np.ndarray | None, int]:
    """Estimate the data-null / gauge subspace.

    The gauge directions are those the reference distribution barely varies along: a reward's
    loading on such a direction is unconstrained by any on-distribution preference, so it is pure
    gauge. Operationally these are the smallest-variance eigen-directions of the *sample* covariance
    (computed pre-shrinkage, because shrinkage lifts the small eigenvalues off the floor and would
    otherwise hide them). When per-item ``margins`` are supplied we refine the estimate further,
    keeping only candidate directions along which the reward margin does not vary
    on the data distribution. The on-distribution margin sensitivity along direction ``u_j`` is the
    covariance ``|Cov(margin, H u_j)|`` rather than a regression coefficient: a covariance reads ~0
    on a direction the data does not move along (so it is robust to the tiny variances that would
    make a regression coefficient explode), which is exactly the "small-|d margin|" subspace.

    Returns ``(null_basis, null_dim)`` with ``null_basis`` an orthonormal ``d x m`` matrix (columns
    are gauge directions) or None when no direction qualifies.
    """
    n, d = x_centered.shape
    s = (x_centered.T @ x_centered) / max(n, 1)
    evals, evecs = np.linalg.eigh(0.5 * (s + s.T))
    top = float(evals[-1]) if evals.size else 0.0
    if top <= 0:
        return None, 0
    var_mask = evals < null_var_ratio * top

    if margins is not None and np.any(var_mask):
        m = np.asarray(margins, dtype=np.float64).ravel()
        if m.size == n and np.std(m) > 0:
            m_c = m - m.mean()
            proj = x_centered.astype(np.float64) @ evecs  # projection onto each eigen-direction
            # On-distribution margin sensitivity: |Cov(margin, H u_j)|, ~0 on data-null directions.
            cov_margin = np.abs((proj * m_c[:, None]).mean(axis=0))
            max_cov = float(cov_margin.max()) if cov_margin.size else 0.0
            if max_cov > 0:
                grad_mask = cov_margin < null_grad_ratio * max_cov
                var_mask = var_mask & grad_mask

    if not np.any(var_mask):
        return None, 0
    basis = evecs[:, var_mask].astype(np.float32)
    return np.ascontiguousarray(basis), int(basis.shape[1])


# ---------------------------------------------------------------------------
# fit_frame
# ---------------------------------------------------------------------------


def fit_frame(
    activations: Any,
    margins: Any | None = None,
    *,
    site: Site | None = None,
    corpus: DatasetID | None = None,
    shrinkage: str = "lw",
    null_var_ratio: float = _DEFAULT_NULL_VAR_RATIO,
    null_grad_ratio: float = _DEFAULT_NULL_GRAD_RATIO,
) -> Frame:
    """Fit a `Frame` from a reference activation matrix.

    ``activations`` is an ``n x d`` fp32 matrix of reference-distribution activations read at
    ``site`` (fp16 is refused). ``margins``, when supplied, are the per-item reward margins used to
    refine the null-subspace estimate to directions of near-zero margin gradient. ``shrinkage``
    selects the covariance estimator; only Ledoit-Wolf (``"lw"``) is implemented, and ``"none"`` is
    accepted for the raw sample covariance (which is only well conditioned when ``n`` comfortably
    exceeds ``d``).

    The fit computes the mean, the shrinkage covariance, its symmetric square roots, and the
    null basis, then derives the content `FrameID` from structural content (site, corpus, counts,
    rounded eigenspectrum summary) so identical fits share an id across platforms.
    """
    x = _to_fp32(activations, "activations")
    if x.ndim != 2:
        raise NumericsError(f"activations must be 2-D (n x d); got shape {x.shape}")
    n, d = x.shape
    if n < 2:
        raise NumericsError(f"need at least 2 rows to fit a frame; got n={n}")

    m = None
    if margins is not None:
        m = _to_fp32(margins, "margins").ravel()
        if m.size != n:
            raise NumericsError(f"margins length {m.size} does not match n={n}")

    site = site or Site(layer=-1, point="resid_post")
    mean = x.mean(axis=0)
    x_centered = x - mean

    if shrinkage == "lw":
        cov, lam = _ledoit_wolf(x_centered)
    elif shrinkage == "none":
        cov, lam = ((x_centered.T @ x_centered) / n).astype(np.float32), 0.0
    else:
        raise NumericsError(f"unknown shrinkage '{shrinkage}'; use 'lw' or 'none'")

    sigma_sqrt, sigma_inv_sqrt, eig = _symmetric_roots(cov)
    null_basis, null_dim = _estimate_null_basis(x_centered, m, null_var_ratio, null_grad_ratio)

    eig_desc = np.sort(eig)[::-1]
    participation = (
        float(eig_desc.sum() ** 2 / np.sum(eig_desc**2)) if np.any(eig_desc > 0) else 0.0
    )
    eig_summary = {
        "top": [round(float(v), 6) for v in eig_desc[:5]],
        "bottom": [round(float(v), 9) for v in eig_desc[-5:]],
        "participation_ratio": round(participation, 4),
        "condition_number": round(float(eig_desc[0] / max(eig_desc[-1], _EIG_FLOOR)), 3),
    }
    estimator_meta: dict[str, Any] = {
        "n": int(n),
        "d": int(d),
        "shrinkage": round(float(lam), 8),
        "shrinkage_kind": shrinkage,
        "eig_summary": eig_summary,
        "null_dim": null_dim,
        "null_var_ratio": null_var_ratio,
        "null_grad_ratio": null_grad_ratio if m is not None else None,
        "fp32_asserted": True,
    }

    id_material = {
        "site": site.__canonical__(),
        "corpus": corpus,
        "n": int(n),
        "d": int(d),
        "shrinkage": round(float(lam), 6),
        "eig_top": eig_summary["top"],
        "null_dim": null_dim,
    }
    frame_id = FrameID(content_hash(id_material, "frame"))

    return Frame(
        id=frame_id,
        site=site,
        corpus=corpus,
        mean=mean,
        sigma_sqrt=sigma_sqrt,
        sigma_inv_sqrt=sigma_inv_sqrt,
        null_basis=null_basis,
        estimator_meta=estimator_meta,
    )


def frame_artifact(frame: Frame) -> FrameArtifact:
    """Convert a live `Frame` into its serializable `FrameArtifact` payload form."""
    return FrameArtifact(
        frame_id=str(frame.id),
        site=frame.site.__canonical__(),
        corpus=frame.corpus,
        mean=frame.mean,
        sigma_sqrt=frame.sigma_sqrt,
        sigma_inv_sqrt=frame.sigma_inv_sqrt,
        null_basis=frame.null_basis,
        estimator_meta=frame.estimator_meta,
    )


def frame_evidence(
    frame: Frame,
    *,
    provenance: Provenance | None = None,
) -> Evidence[FrameArtifact]:
    """Wrap a fitted `Frame` in an `Evidence` so it is a persistable, provenance-carrying artifact.

    A frame is INVARIANT machinery (it defines the gauge rather than being compared under one) and
    carries no calibration, so its Evidence is EXPLORATORY. The subject records the frame id and its
    corpus so the store can key the shared artifact.
    """
    subject = SubjectRef(
        dataset=frame.corpus,
        frame=frame.id,
        extra={"site": str(frame.site), "d": frame.d, "null_dim": frame.null_dim},
    )
    return make_evidence(
        observable="geometry.frame",
        observable_version="1",
        subject=subject,
        value=frame_artifact(frame),
        gauge=GaugeStatus.INVARIANT,
        provenance=provenance,
    )


__all__ = [
    "Frame",
    "FrameArtifact",
    "fit_frame",
    "frame_artifact",
    "frame_evidence",
]

"""Canonicalization and the effective angle: comparing reward directions in a shared frame.

This is the executable form of invariant I3 (A13). Two reward directions are only
comparable across signals once both are expressed in a shared `Frame`. The canonical form whitens
by the reference-distribution covariance square root and normalizes:

    w_tilde = Sigma^{1/2} w / || Sigma^{1/2} w ||

so the canonical cosine ``cos(w_tilde_a, w_tilde_b)`` equals the on-distribution correlation of the
two reward readouts, ``Corr(w_a . h, w_b . h)`` for ``h`` drawn from the frame corpus. That is the
quantity STARC (2309.15257) canonicalizes, and it is invariant to exactly the reward gauge group:
positive affine reparameterization of the reward (normalization removes the scale) and the addition
of data-null directions to the reward vector (``Sigma^{1/2}`` annihilates them, and, when the frame
carries a ``null_basis``, we quotient them explicitly so the invariance is exact rather than merely
down-weighted by the shrinkage floor).

`effective_angle` reports that canonical cosine with a bootstrap confidence interval and a
STARC-style behavioural regret bound: the canonical distance upper-bounds worst-case preference
disagreement on the frame corpus, and we report that disagreement empirically by searching corpus
response pairs for the ones the two rewards order oppositely (A13, faithful_to
"STARC 2309.15257 / partial identifiability 2411.15951"). The comparison is COVARIANT and takes a
frame argument with no default; it calls ``require_frame_for_comparison`` so gate 2 cannot be
skipped.

Pure numpy/scipy, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from reward_lens.core.errors import NumericsError
from reward_lens.core.evidence import Evidence, Uncertainty, make_evidence, register_payload
from reward_lens.core.gates import require_frame_for_comparison
from reward_lens.core.provenance import Provenance
from reward_lens.core.types import GaugeStatus, SubjectRef
from reward_lens.geometry.frame import Frame, _to_fp32

_FAITHFUL_TO = "STARC 2309.15257 / partial identifiability 2411.15951"


def canonicalize(w: Any, frame: Frame) -> np.ndarray:
    """Canonical form of a reward direction in a frame: ``Sigma^{1/2} w`` normalized.

    Whitening by ``Sigma^{1/2}`` weights each direction by how much the reference distribution
    varies along it, so directions the distribution does not vary along contribute nothing to the
    canonical direction. When the frame carries an estimated ``null_basis``
    the gauge components of ``w`` are projected out first, which makes canonicalization exactly
    invariant to adding data-null directions to ``w`` (rather than leaving the small residual the
    Ledoit-Wolf shrinkage floor would otherwise pass through). The result is a unit fp32 vector.

    Raises `NumericsError` if ``w`` has no on-distribution component (``Sigma^{1/2} w`` is zero),
    which means the reward direction lives entirely in the gauge subspace and has no canonical form.
    """
    w = _to_fp32(w, "w").ravel()
    if w.shape[0] != frame.d:
        raise NumericsError(f"reward direction has length {w.shape[0]}, frame is {frame.d}-dim")
    w64 = w.astype(np.float64)

    if frame.null_basis is not None:
        n = frame.null_basis.astype(np.float64)
        w64 = w64 - n @ (n.T @ w64)  # remove gauge components: they are unconstrained on data

    whitened = frame.sigma_sqrt.astype(np.float64) @ w64
    norm = float(np.linalg.norm(whitened))
    if norm <= _norm_floor(frame):
        raise NumericsError(
            "reward direction has no on-distribution component (Sigma^{1/2} w is ~0): it lives in "
            "the frame's null subspace and has no canonical direction. This is a pure gauge vector."
        )
    return (whitened / norm).astype(np.float32)


def _norm_floor(frame: Frame) -> float:
    """A scale-aware floor below which a whitened norm is treated as numerically zero."""
    top = 1.0
    summary = frame.estimator_meta.get("eig_summary", {})
    tops = summary.get("top") if isinstance(summary, dict) else None
    if tops:
        top = max(float(tops[0]), 1e-12)
    return 1e-6 * np.sqrt(top)


def _raw_cos(a: np.ndarray, b: np.ndarray) -> float:
    """Ordinary cosine of two raw (un-whitened) direction vectors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


@register_payload
@dataclass
class AngleResult:
    """The payload of `effective_angle` (A13).

    ``canonical_cos`` is the gauge-fixed alignment (on-distribution reward correlation);
    ``raw_cos`` is the un-whitened cosine kept for contrast (the number that made E19's
    cross-model comparison meaningless). ``starc_distance`` is the canonical distance
    ``sqrt(2(1 - canonical_cos))`` between the unit whitened directions, the STARC quantity that
    bounds worst-case regret. ``regret_bound`` is the behavioural realization of that bound:
    the empirical fraction of frame-corpus response pairs the two rewards order oppositely, found
    by pair search (None-safe when no corpus is supplied). ``worst_case_margin`` is the largest
    reward_a preference margin that reward_b reverses, the single worst pair.
    """

    canonical_cos: float
    raw_cos: float
    starc_distance: float
    regret_bound: float
    worst_case_margin: float
    ci_low: float
    ci_high: float
    n_boot: int
    n_corpus: int
    faithful_to: str = _FAITHFUL_TO


def _pairwise_regret(
    activations: np.ndarray, w_a: np.ndarray, w_b: np.ndarray
) -> tuple[float, float]:
    """Empirical worst-case preference disagreement between two rewards on a corpus.

    For rewards ``r_a = A w_a`` and ``r_b = A w_b`` over corpus rows, this searches all ordered
    response pairs and returns ``(disagreement_fraction, worst_case_margin)``: the fraction of
    pairs the two rewards order oppositely (the empirical regret the STARC distance bounds), and
    the largest normalized reward_a margin among the pairs reward_b reverses (the single worst
    case). Ties on either reward are excluded from the disagreement count. Fully vectorized over
    the ``n x n`` pair grid; the caller subsamples the corpus when ``n`` is large.
    """
    ra = activations @ w_a
    rb = activations @ w_b
    da = ra[:, None] - ra[None, :]  # reward_a preference margin for pair (i, j)
    db = rb[:, None] - rb[None, :]
    iu = np.triu_indices(activations.shape[0], k=1)
    da = da[iu]
    db = db[iu]
    both_nonzero = (da != 0) & (db != 0)
    if not np.any(both_nonzero):
        return 0.0, 0.0
    da = da[both_nonzero]
    db = db[both_nonzero]
    discordant = np.sign(da) != np.sign(db)
    frac = float(np.mean(discordant))
    scale = float(np.max(np.abs(da))) or 1.0
    worst = float(np.max(np.abs(da[discordant])) / scale) if np.any(discordant) else 0.0
    return frac, worst


def _bootstrap_canonical_cos(
    activations: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
    frame: Frame,
    n_boot: int,
    ci_level: float,
    seed: int,
) -> tuple[float, float]:
    """Percentile CI for the canonical cosine by resampling corpus rows and refitting the frame.

    The uncertainty in the canonical cosine comes from estimating the whitening covariance from a
    finite corpus, so a proper interval refits the frame on each bootstrap resample of the corpus
    rows and recomputes the canonical cosine. The replicates are already frame-resample draws, so
    the percentile bounds are direct quantiles of them (the percentile bootstrap reduces to exactly
    this); non-finite replicates from a degenerate resample are dropped.
    """
    from reward_lens.geometry.frame import fit_frame

    rng = np.random.default_rng(seed)
    n = activations.shape[0]
    reps = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            fb = fit_frame(
                activations[idx],
                site=frame.site,
                corpus=frame.corpus,
                null_var_ratio=frame.estimator_meta.get("null_var_ratio", 1e-3),
            )
            ca = canonicalize(w_a, fb)
            cb = canonicalize(w_b, fb)
            reps[b] = float(np.dot(ca, cb))
        except NumericsError:
            reps[b] = np.nan
    reps = reps[np.isfinite(reps)]
    if reps.size < 2:
        return float("nan"), float("nan")
    alpha = (1.0 - ci_level) / 2.0
    return float(np.quantile(reps, alpha)), float(np.quantile(reps, 1.0 - alpha))


def effective_angle(
    w_a: Any,
    w_b: Any,
    frame: Frame,
    *,
    n_boot: int = 1000,
    activations_for_bound: Any | None = None,
    ci_level: float = 0.95,
    max_pairs_corpus: int = 512,
    seed: int = 0,
    subject: SubjectRef | None = None,
    provenance: Provenance | None = None,
) -> Evidence[AngleResult]:
    """Canonical angle between two reward directions with a CI and a STARC regret bound.

    A13. Reports ``cos(w_tilde_a, w_tilde_b)`` in the shared ``frame`` (the gauge-fixed alignment),
    the raw cosine for contrast, a bootstrap CI obtained by refitting the frame on corpus resamples
    (when ``activations_for_bound`` is supplied), and a behavioural regret bound: the empirical
    worst-case preference disagreement on the frame corpus, which the canonical (STARC) distance
    upper-bounds.

    The quantity is COVARIANT and the ``frame`` argument has no default; this calls
    ``require_frame_for_comparison`` so a frameless cross-signal comparison raises (gate 2, I3).
    ``faithful_to`` is "STARC 2309.15257 / partial identifiability 2411.15951".
    """
    require_frame_for_comparison(GaugeStatus.COVARIANT, frame.id)

    wa = _to_fp32(w_a, "w_a").ravel()
    wb = _to_fp32(w_b, "w_b").ravel()
    ca = canonicalize(wa, frame)
    cb = canonicalize(wb, frame)
    canonical_cos = float(np.dot(ca, cb))
    raw_cos = _raw_cos(wa.astype(np.float64), wb.astype(np.float64))
    starc_distance = float(np.sqrt(max(0.0, 2.0 * (1.0 - canonical_cos))))

    regret_bound = float("nan")
    worst_case = float("nan")
    ci_low, ci_high = float("nan"), float("nan")
    n_corpus = 0
    method = "none"

    if activations_for_bound is not None:
        acts = _to_fp32(activations_for_bound, "activations_for_bound")
        if acts.ndim != 2 or acts.shape[1] != frame.d:
            raise NumericsError(
                f"activations_for_bound must be (n x {frame.d}); got shape {acts.shape}"
            )
        n_corpus = int(acts.shape[0])
        # Center on the frame mean so the reward margins are read in the frame's coordinates.
        centered = acts - frame.mean
        sub = centered
        if n_corpus > max_pairs_corpus:
            rng = np.random.default_rng(seed)
            sel = rng.choice(n_corpus, size=max_pairs_corpus, replace=False)
            sub = centered[sel]
        regret_bound, worst_case = _pairwise_regret(
            sub.astype(np.float64), wa.astype(np.float64), wb.astype(np.float64)
        )
        if n_boot > 0:
            ci_low, ci_high = _bootstrap_canonical_cos(acts, wa, wb, frame, n_boot, ci_level, seed)
            method = "bootstrap-frame-resample"

    value = AngleResult(
        canonical_cos=canonical_cos,
        raw_cos=raw_cos,
        starc_distance=starc_distance,
        regret_bound=regret_bound,
        worst_case_margin=worst_case,
        ci_low=ci_low,
        ci_high=ci_high,
        n_boot=int(n_boot) if activations_for_bound is not None else 0,
        n_corpus=n_corpus,
    )
    subj = subject or SubjectRef(frame=frame.id)
    uncertainty = Uncertainty(
        ci_low=ci_low if np.isfinite(ci_low) else None,
        ci_high=ci_high if np.isfinite(ci_high) else None,
        ci_level=ci_level if np.isfinite(ci_low) else None,
        n=n_corpus or None,
        method=method,
    )
    return make_evidence(
        observable="geometry.effective_angle",
        observable_version="1",
        subject=subj,
        value=value,
        uncertainty=uncertainty,
        gauge=GaugeStatus.COVARIANT,
        provenance=provenance,
    )


__all__ = [
    "canonicalize",
    "effective_angle",
    "AngleResult",
]

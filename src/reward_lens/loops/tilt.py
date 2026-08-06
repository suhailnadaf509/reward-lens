"""SNIS emulator of the tilted policy family and the susceptibility spectrum.

Optimizing a policy against a reward model pulls it toward the exponentially tilted family
``pi_lambda proportional to pi_0 * exp(lambda * r)`` (the maximum-entropy / KL-regularized RL
solution). The point of this module is that the whole family, and the feature-level consequences
of optimizing against the grader, can be previewed from a single bank of base-policy samples with
zero gradient updates, by self-normalized importance sampling (SNIS): reweight the base samples by
``w_i = exp(lambda * r_i)`` and read off tilted expectations.

Two things come out of that.

The susceptibility spectrum is the initial per-feature drift under optimization. For an exponential
family the derivative of a tilted expectation at ``lambda = 0`` is a base-policy covariance,
``d/dlambda E_lambda[f_i] |_0 = Cov_0(f_i, r) = chi_i`` (Appendix A12, the fluctuation-dissipation
identity). So ``chi_i`` predicts which feature drifts and how fast before any RL, and it is a
covariance of two scalar readouts, hence gauge-invariant. The ``f = r`` diagonal
``chi = Var_0(r)`` is exactly Razin's teacher variance (Appendix A3), the zeroth-order
term of the same law. Predicted hack modes are the features with ``chi_i > 0`` while
``Cov_0(f_i, gold) <= 0``.

The tilt is only a faithful emulator of practical optimization while the importance weights have
not degenerated. Two things break it: pushing ``lambda`` past roughly half the critical pressure
``lambda_c`` (Appendix A5; beyond ``lambda_c`` the reward's moment generating function diverges for
a heavy tail and there is no tilted optimum at all, Appendix A4), and the effective sample size of
the SNIS weights collapsing so a handful of samples carry the estimate. This module refuses in both
cases rather than returning a confident extrapolation from three samples, and says which guard
fired and why. The validity ceiling ``lambda_c / 2`` is the stated safe range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from reward_lens.core.errors import RewardLensError
from reward_lens.core.evidence import Evidence, Uncertainty, make_evidence, register_payload
from reward_lens.core.provenance import Provenance
from reward_lens.core.types import GaugeStatus, SubjectRef

if TYPE_CHECKING:
    from reward_lens.core.types import EvidenceID


class ESSGuardError(RewardLensError):
    """Raised when a tilt is requested outside its regime of validity.

    Two triggers: ``lambda`` beyond ``lambda_c / 2`` (the tilt stops emulating practical
    optimization; past ``lambda_c`` there is no tilted optimum for a heavy tail, Appendix A4/A5), or
    the SNIS effective sample size collapsing below ``min_ess_frac`` (a few samples dominate the
    reweighting, so the estimate is not trustworthy). The message names the offending ``lambda``,
    the ceiling, and which guard fired.
    """


def susceptibility(
    scores: Sequence[float] | np.ndarray,
    features: np.ndarray,
    feature_names: Sequence[str] | None = None,
    *,
    subject: SubjectRef | None = None,
    parents: Sequence["EvidenceID"] = (),
) -> Evidence["SusceptibilitySpectrum"]:
    """The susceptibility spectrum ``chi_i = Cov_0(f_i, r)`` from base-policy samples (Appendix A12).

    ``scores`` is the base-policy reward per sample (shape ``(N,)``); ``features`` is the per-sample
    scalar activation of each feature (shape ``(N, k)``). Returns ``chi_i = Cov_0(f_i, r)``, the
    predicted initial drift of feature ``i`` under optimization, computed as a population covariance
    ``mean(f_i r) - mean(f_i) mean(r)`` (the covariance is an expectation under ``pi_0``, so the
    population form, ddof = 0, is the estimator of the theory object). The ``f = r`` diagonal
    ``Var_0(r)`` is reported as ``teacher_variance`` (Appendix A3).

    Gauge is INVARIANT: a covariance of two scalar readouts does not depend on the activation
    basis. Trust is EXPLORATORY until a planted-chi organism scorecard calibrates it (gate 1);
    that calibration is a study, not this function's job.
    """
    r = np.asarray(scores, dtype=np.float64).ravel()
    f = np.asarray(features, dtype=np.float64)
    if f.ndim == 1:
        f = f[:, None]
    if f.shape[0] != r.size:
        raise ValueError(
            f"features has {f.shape[0]} rows but scores has {r.size}; they must be per-sample aligned"
        )
    n = r.size
    if n < 2:
        raise ValueError(f"susceptibility needs at least 2 samples; got {n}")
    r_c = r - r.mean()
    f_c = f - f.mean(axis=0, keepdims=True)
    chi = (f_c * r_c[:, None]).mean(axis=0)
    teacher_var = float((r_c * r_c).mean())
    names = (
        list(feature_names) if feature_names is not None else [f"f{i}" for i in range(f.shape[1])]
    )
    if len(names) != chi.size:
        raise ValueError(f"got {len(names)} feature names for {chi.size} features")
    payload = SusceptibilitySpectrum(
        chi=chi, teacher_variance=teacher_var, feature_names=names, n_samples=n
    )
    return make_evidence(
        observable="loops.tilt.susceptibility",
        observable_version="1",
        subject=subject or SubjectRef(),
        value=payload,
        uncertainty=Uncertainty(n=n, method="none"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(parents=tuple(parents)),
    )


@register_payload
@dataclass
class SusceptibilitySpectrum:
    """The predicted initial per-feature drift under optimization (Appendix A12).

    ``chi[i] = Cov_0(f_i, r)`` predicts the sign and speed of feature ``i``'s drift before any RL.
    ``teacher_variance = Var_0(r)`` is the ``f = r`` diagonal (Appendix A3). ``predicted_hack_modes``
    (filled by ``flag_hack_modes`` when a gold covariance is supplied) are the features with
    ``chi_i > 0`` and ``Cov_0(f_i, gold) <= 0``.
    """

    chi: np.ndarray
    teacher_variance: float
    feature_names: list[str]
    n_samples: int


def flag_hack_modes(
    spectrum: SusceptibilitySpectrum, gold_covariance: Sequence[float] | np.ndarray
) -> list[str]:
    """Names of the predicted hack modes: ``chi_i > 0`` while ``Cov_0(f_i, gold) <= 0`` (A12).

    ``gold_covariance[i] = Cov_0(f_i, gold)`` for a reference/gold signal. A feature the proxy loves
    (positive susceptibility) but the gold objective does not is exactly the direction optimization
    exploits, which is what the forecasting science (S12) pre-registers.
    """
    gold = np.asarray(gold_covariance, dtype=np.float64).ravel()
    if gold.size != spectrum.chi.size:
        raise ValueError(
            f"gold_covariance has {gold.size} entries for {spectrum.chi.size} features"
        )
    return [
        name for name, c, g in zip(spectrum.feature_names, spectrum.chi, gold) if c > 0 and g <= 0
    ]


def critical_lambda_from_tail(
    scores: Sequence[float] | np.ndarray, tail_quantile: float = 0.9
) -> float:
    """Estimate the critical pressure ``lambda_c`` from the reward's right-tail scale (A4/A5).

    Fits an exponential right tail by peaks-over-threshold: the scale ``tau`` is the mean excess of
    the scores above their ``tail_quantile`` quantile, and ``lambda_c = 1 / tau`` (Appendix A5, an
    exponential tail with scale ``tau`` has critical pressure ``1 / tau``). This is the cheap
    scalar the tilt guard needs; the full Hill / peaks-over-threshold tail index lives in the tail
    index observable (Appendix A4), which this does not replace. Returns ``inf`` when the excess is
    degenerate (a bounded tail imposes no ceiling on lambda).
    """
    r = np.asarray(scores, dtype=np.float64).ravel()
    if r.size < 8:
        raise ValueError(f"tail estimation needs a reasonable bank; got {r.size} samples")
    threshold = float(np.quantile(r, tail_quantile))
    excess = r[r > threshold] - threshold
    if excess.size < 2:
        return float("inf")
    tau = float(excess.mean())
    if tau <= 0:
        return float("inf")
    return 1.0 / tau


def _snis_weights(scores: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Self-normalized importance weights for tilt ``lambda`` and their effective sample size.

    ``w_i proportional to exp(lambda r_i)``, stabilized by subtracting the max exponent before
    exponentiating. Returns the normalized weights and ``ESS = 1 / sum_i w_i^2``, which is the
    number of samples effectively carrying the reweighted estimate.
    """
    z = lam * scores
    z = z - z.max()
    w = np.exp(z)
    w_sum = w.sum()
    wn = w / w_sum
    ess = 1.0 / float(np.sum(wn**2))
    return wn, ess


def tilt_sweep(
    scores: Sequence[float] | np.ndarray,
    features: np.ndarray,
    lambdas: Sequence[float],
    lambda_c: float,
    *,
    feature_names: Sequence[str] | None = None,
    min_ess_frac: float = 0.05,
    subject: SubjectRef | None = None,
    parents: Sequence["EvidenceID"] = (),
) -> Evidence["TiltPrediction"]:
    """Emulate the tilted family across a ``lambda`` grid by SNIS, with the validity guards.

    For each ``lambda`` in ``lambdas`` the tilted feature means ``E_lambda[f_i]`` and reward mean
    ``E_lambda[r]`` are estimated by reweighting the base-policy bank, with no gradient updates. The
    susceptibility ``chi_i = Cov_0(f_i, r)`` (the ``lambda -> 0`` slope) is attached so the initial
    drift the sweep traces can be checked against the closed-form prediction.

    Refuses, via ``ESSGuardError``, when any requested ``|lambda|`` exceeds ``lambda_c / 2`` (past
    that the tilt no longer emulates practical optimization; past ``lambda_c`` there is no tilted
    optimum for a heavy tail, Appendix A4), or when the SNIS effective sample size at some
    valid ``lambda`` falls below ``min_ess_frac`` of the bank. Returning a confident number from a
    handful of dominating samples is the failure this guard exists to prevent.

    Gauge is INVARIANT (tilted means of scalar readouts and their base covariance are gauge-free).
    """
    r = np.asarray(scores, dtype=np.float64).ravel()
    f = np.asarray(features, dtype=np.float64)
    if f.ndim == 1:
        f = f[:, None]
    if f.shape[0] != r.size:
        raise ValueError(
            f"features has {f.shape[0]} rows but scores has {r.size}; they must be per-sample aligned"
        )
    n = r.size
    lam_arr = np.asarray(list(lambdas), dtype=np.float64)
    half = 0.5 * float(lambda_c)

    beyond = lam_arr[np.abs(lam_arr) > half + 1e-12]
    if beyond.size:
        raise ESSGuardError(
            f"tilt refused: lambda(s) {beyond.tolist()} exceed lambda_c/2 = {half:.4g} "
            f"(lambda_c = {lambda_c:.4g}). Beyond half the critical pressure the exponential tilt "
            f"stops emulating practical optimization, and past lambda_c the reward's MGF diverges "
            f"for a heavy tail so there is no tilted optimum at all (Appendix A4/A5). "
            f"Re-run within [{-half:.4g}, {half:.4g}], or use loops.bon for the far frontier."
        )

    feature_means = np.empty((lam_arr.size, f.shape[1]), dtype=np.float64)
    reward_mean = np.empty(lam_arr.size, dtype=np.float64)
    ess = np.empty(lam_arr.size, dtype=np.float64)
    for i, lam in enumerate(lam_arr):
        wn, ess_i = _snis_weights(r, float(lam))
        feature_means[i] = wn @ f
        reward_mean[i] = float(wn @ r)
        ess[i] = ess_i
    ess_frac = ess / n

    low = np.where(ess_frac < min_ess_frac)[0]
    if low.size:
        worst = int(low[np.argmin(ess_frac[low])])
        raise ESSGuardError(
            f"tilt refused: at lambda = {lam_arr[worst]:.4g} the SNIS effective sample size is "
            f"{ess[worst]:.1f} of {n} ({ess_frac[worst]:.1%}), below the min_ess_frac "
            f"{min_ess_frac:.1%} floor. The importance weights have collapsed onto a few samples, "
            f"so the reweighted estimate is not trustworthy. Draw a larger bank or "
            f"reduce lambda."
        )

    chi_ev = susceptibility(r, f, feature_names)
    spectrum: SusceptibilitySpectrum = chi_ev.value
    payload = TiltPrediction(
        lambdas=lam_arr,
        feature_means=feature_means,
        reward_mean=reward_mean,
        ess=ess,
        ess_frac=ess_frac,
        chi=spectrum.chi,
        teacher_variance=spectrum.teacher_variance,
        lambda_c=float(lambda_c),
        feature_names=spectrum.feature_names,
        n_samples=n,
    )
    return make_evidence(
        observable="loops.tilt.sweep",
        observable_version="1",
        subject=subject or SubjectRef(),
        value=payload,
        uncertainty=Uncertainty(n=n, method="snis"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(parents=tuple(parents)),
    )


@register_payload
@dataclass
class TiltPrediction:
    """The SNIS-emulated tilted family across a ``lambda`` grid.

    ``feature_means[i]`` is ``E_lambda[f]`` at ``lambdas[i]``; ``reward_mean[i]`` is
    ``E_lambda[r]``; ``ess`` / ``ess_frac`` record how much of the bank carried each estimate.
    ``chi`` is the ``lambda -> 0`` slope ``Cov_0(f_i, r)`` (Appendix A12) and ``teacher_variance``
    the ``f = r`` diagonal (Appendix A3). ``lambda_c`` is the critical pressure the guard used.
    """

    lambdas: np.ndarray
    feature_means: np.ndarray
    reward_mean: np.ndarray
    ess: np.ndarray
    ess_frac: np.ndarray
    chi: np.ndarray
    teacher_variance: float
    lambda_c: float
    feature_names: list[str]
    n_samples: int


__all__ = [
    "susceptibility",
    "SusceptibilitySpectrum",
    "flag_hack_modes",
    "critical_lambda_from_tail",
    "tilt_sweep",
    "TiltPrediction",
    "ESSGuardError",
]

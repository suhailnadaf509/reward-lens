"""A4 Tail index: the right-tail exponent of the reward.

Formal definition, A4. A Hill / peaks-over-threshold estimate of the right-tail exponent of
``r`` under the base policy ``π_0`` (aggregate), and of each feature's contribution to ``r``
(per-feature). This is the reward-thermodynamics diagnostic (faithful_to Kwa 2407.14503): the shape of
the reward's upper tail decides whether optimization has a finite safe budget.

  - A polynomial (heavy) right tail makes the moment generating function ``E_0[exp(λ r)]`` diverge for
    every ``λ > 0``, so the tilted family ``π_λ ∝ π_0 exp(λ r)`` has no optimum: catastrophic Goodhart,
    critical pressure ``λ_c = 0``.
  - An exponential tail with scale ``τ`` (``P(r > u) ∼ exp(-u/τ)``) makes the MGF finite for ``λ < 1/τ``
    and divergent at ``λ = 1/τ``, so the critical pressure is ``λ_c = 1/τ``, the ceiling A5 turns into a
    K-nats budget.

Deviations from A4: the generalized-Pareto shape and scale are fit by the method of moments over
peaks above a quantile threshold, and the polynomial index is cross-checked with the Hill estimator;
both are standard extreme-value estimators, and the choice is stated here rather than left implicit.
The per-feature tail is computed on the feature values as the observable proxy for each feature's
contribution to ``r``; the attribution-weighted contribution is the production refinement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.types import Access, AccessMatrix, Capability, Component, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.indices._support import (
    ANY_SUBSTRATE,
    GRADER_STUDY_PHASES,
    MEASURED_BY,
    FeatureBank,
    final_activations,
    load_default_bank,
    reward_scores,
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence

# A generalized-Pareto shape within this band of zero is read as an exponential tail.
_EXP_SHAPE_TOL = 0.05


def hill_tail_index(x: np.ndarray, k: int | None = None) -> tuple[float, float]:
    """Hill estimate of the extreme-value index and polynomial tail index of a right tail (A4).

    For the top ``k`` order statistics the Hill estimator of the extreme-value index is
    ``γ = (1/k) Σ_{i<k} [ln x_(n-i) - ln x_(n-k)]``; a heavy (Frechet) tail has ``γ > 0`` and
    polynomial index ``α = 1/γ`` (``P(X > x) ∼ x^{-α}``), while an exponential (Gumbel) tail drives
    ``γ`` toward ``0`` and ``α`` toward ``+∞``. ``k`` defaults to ``⌊n/10⌋`` (a standard upper-tail
    fraction). Requires positive data, since the estimator is on log order statistics; callers shift a
    signed reward so its exceedances are positive before estimating. Returns ``(gamma, alpha)``.
    """
    xs = np.sort(np.asarray(x, dtype=np.float64).ravel())
    n = xs.size
    if k is None:
        k = max(2, n // 10)
    k = int(min(k, n - 1))
    if k < 2 or xs[n - k - 1] <= 0:
        return float("nan"), float("nan")
    top = np.log(xs[n - k :])
    gamma = float(np.mean(top) - np.log(xs[n - k - 1]))
    alpha = 1.0 / gamma if gamma > 0 else float("inf")
    return gamma, alpha


def _gpd_moment_fit(exceedances: np.ndarray) -> tuple[float, float]:
    """Method-of-moments generalized-Pareto ``(shape ξ, scale σ)`` for peaks over a threshold.

    For exceedances with mean ``m`` and variance ``s²`` the moment estimators are
    ``ξ = ½(1 - m²/s²)`` and ``σ = ½ m (m²/s² + 1)``. An exponential tail has ``m = s`` so ``ξ → 0`` and
    ``σ → m`` (the mean excess), which is the identity the exponential-scale test relies on; a
    polynomial tail has ``ξ = 1/α > 0``.
    """
    y = np.asarray(exceedances, dtype=np.float64).ravel()
    m = float(np.mean(y))
    s2 = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
    if s2 <= 0:
        return 0.0, m
    ratio = m * m / s2
    xi = 0.5 * (1.0 - ratio)
    sigma = 0.5 * m * (ratio + 1.0)
    return float(xi), float(sigma)


def tail_estimate(
    r: np.ndarray,
    *,
    quantile: float = 0.9,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Peaks-over-threshold tail estimate of a reward sample, with the critical pressure (A4).

    Sets a threshold at the ``quantile`` of ``r`` (or an explicit ``threshold``), fits a generalized
    Pareto to the exceedances by moments, and classifies the tail. An exponential tail (``|ξ| <
    tol``) reports its scale ``τ = σ`` and critical pressure ``λ_c = 1/τ``; a heavy tail (``ξ > tol``)
    reports its polynomial index ``α = 1/ξ`` and ``λ_c = 0`` (catastrophic: the MGF diverges for all
    positive ``λ``). The Hill index is included as an independent cross-check of the heavy-tail case.
    Returns a dict of the shape, scale, regime, ``λ_c``, ``τ``, and diagnostics.
    """
    x = np.asarray(r, dtype=np.float64).ravel()
    u = float(threshold) if threshold is not None else float(np.quantile(x, quantile))
    exceed = x[x > u] - u
    n_exceed = int(exceed.size)
    if n_exceed < 3:
        return {
            "regime": "undetermined",
            "shape_xi": float("nan"),
            "scale_sigma": float("nan"),
            "tau": float("nan"),
            "lambda_c": float("nan"),
            "alpha": float("nan"),
            "threshold": u,
            "n_exceed": n_exceed,
        }

    xi, sigma = _gpd_moment_fit(exceed)
    _, hill_alpha = hill_tail_index(x - x.min() + 1e-9)

    if xi > _EXP_SHAPE_TOL:
        regime = "polynomial"
        tau = float("inf")
        lambda_c = 0.0
        alpha = 1.0 / xi
    elif xi < -_EXP_SHAPE_TOL:
        regime = "bounded"  # negative shape: a finite right endpoint, MGF finite for all lambda
        tau = float(sigma)
        lambda_c = float("inf")
        alpha = float("inf")
    else:
        regime = "exponential"
        tau = float(sigma)
        lambda_c = 1.0 / tau if tau > 0 else float("inf")
        alpha = float("inf")

    return {
        "regime": regime,
        "shape_xi": float(xi),
        "scale_sigma": float(sigma),
        "tau": tau,
        "lambda_c": lambda_c,
        "alpha": alpha,
        "hill_alpha": float(hill_alpha),
        "threshold": u,
        "n_exceed": n_exceed,
    }


def _bootstrap_shape_ci(
    r: np.ndarray, quantile: float, seed: int, n: int, ci: float
) -> tuple[float, float]:
    """Bootstrap CI on the generalized-Pareto shape ``ξ`` so the tail regime carries uncertainty.

    Resamples the reward with replacement, refits ``ξ`` over the same quantile threshold each time, and
    returns the ``ci`` percentile interval. A shape interval straddling zero is the honest signal that
    the exponential-vs-polynomial call is not yet resolved by the sample.
    """
    x = np.asarray(r, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    shapes: list[float] = []
    for _ in range(n):
        boot = x[rng.integers(0, x.size, size=x.size)]
        u = float(np.quantile(boot, quantile))
        exceed = boot[boot > u] - u
        if exceed.size >= 3:
            xi, _ = _gpd_moment_fit(exceed)
            shapes.append(xi)
    if len(shapes) < 10:
        return float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    arr = np.asarray(shapes)
    return float(np.quantile(arr, alpha)), float(np.quantile(arr, 1.0 - alpha))


class TailIndex(BaseObservable):
    """A4 right-tail exponent of the reward, aggregate and per-feature.

    Requires scores (and activations plus a linear readout for the per-feature tails). Scores the base
    policy samples, estimates the aggregate tail with a bootstrap CI on the shape, and, when a feature
    bank is available, estimates each feature's tail as the proxy for its contribution to ``r``. Gauge
    is INVARIANT: the tail regime and the dimensionless shape are scale-free properties of one signal,
    though ``τ`` and ``λ_c`` carry reward-scale units, which is noted as a deviation.

    What it cannot do. A tail exponent is an extrapolation past the largest score anyone has seen,
    fitted on the peaks above a quantile the caller picks, and the answer moves with that quantile:
    the threshold is a modelling choice the payload records but does not defend. The bootstrap
    interval on the shape covers sampling variation and not threshold choice, so it is narrower than
    the real uncertainty. The regime call, polynomial against exponential, is the part optimisation
    budgets are built on and it is the part a few hundred samples decide least well. The per-feature
    tails are on the feature values, which stand in for each feature's contribution to the reward
    and are not that contribution.
    """

    name = "TailIndex"
    version = "1.0"
    capabilities = Capability.SCORES
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A4"
    deviations = (
        "tau and lambda_c carry reward-scale units (the shape and regime are scale-free)",
        "per-feature tail is on the feature values as the proxy for each feature's contribution to r",
    )

    # -- the observable declarations ---------------------------------------
    quantity = "grader.tail_index"
    #: Scores come from the grader on the view. The per-feature enrichment additionally captures
    #: activations, and it is skipped rather than refused when the signal does not offer them.
    requires: AccessMatrix = {Component.GRADER: Access.QUERY}
    substrates = ANY_SUBSTRATE
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset(
            {RegimeCondition.STATIONARY_GRADER, RegimeCondition.GROUP_NONDEGENERATE}
        ),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: The generalized-Pareto shape is a location-scale invariant of the distribution, so it does
    #: not move under ``r -> a*r + b``. The scale ``tau`` and the critical pressure ``lambda_c`` do
    #: move, which is recorded in the deviations and is why the shape is what this declaration is
    #: about. LIGHT_TAILED is deliberately absent from the envelope: this is the instrument that
    #: measures it, and requiring it would be circular.
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = ("baseline.gaussian_reference", "baseline.exponential_reference")
    rung = 0

    def __init__(
        self,
        feature_bank: FeatureBank | None = None,
        *,
        quantile: float = 0.9,
        bootstrap: int = 500,
        ci: float = 0.95,
        seed: int = 0,
    ) -> None:
        self.feature_bank = feature_bank
        self.quantile = float(quantile)
        self.bootstrap = int(bootstrap)
        self.ci = float(ci)
        self.seed = int(seed)

    def measure(self, ctx: Context) -> "Evidence":
        signal = ctx.signal
        reward = reward_scores(signal, ctx.view, ctx.readout)
        agg = tail_estimate(reward, quantile=self.quantile)
        lo, hi = _bootstrap_shape_ci(reward, self.quantile, self.seed, self.bootstrap, self.ci)
        agg["shape_ci_low"] = lo
        agg["shape_ci_high"] = hi

        per_feature: dict[str, Any] = {}
        bank = self.feature_bank
        if bank is None and (signal.caps & Capability.ACTIVATIONS):
            bank = load_default_bank(signal)
        if bank is not None and (signal.caps & Capability.ACTIVATIONS):
            try:
                acts = final_activations(signal, ctx.view, readout=ctx.readout)
                features = bank.featurize(acts)
                names = list(getattr(bank, "names", tuple())) or [
                    f"f{i}" for i in range(features.shape[1])
                ]
                for i, nm in enumerate(names):
                    per_feature[nm] = tail_estimate(features[:, i], quantile=self.quantile)
            except Exception:  # noqa: BLE001 - per-feature is a best-effort enrichment
                per_feature = {}

        payload = {
            "aggregate": agg,
            "per_feature": per_feature,
            "n_items": int(reward.size),
        }
        return ctx.emit(
            payload,
            uncertainty=Uncertainty(
                ci_low=lo, ci_high=hi, ci_level=self.ci, n=int(reward.size), method="bootstrap"
            ),
        )


__all__ = ["hill_tail_index", "tail_estimate", "TailIndex"]

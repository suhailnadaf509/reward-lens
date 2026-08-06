"""The closure test: is `Var(ρ)` accounted for by `Σ u_i²`, and what it means when it is not.

Four verdicts, not two, because "the budget does not account for the residual" has three different
causes and only one of them is a discovery:

``closed``
    The interval on `Var(ρ) / Σ u_i²` contains 1. The ledger balances and the instrument is
    characterised.
``unmodelled``
    The ratio is above 1 with its interval clear of it, **and every one of the nine terms was
    computed**. There is a contribution the itemisation does not name, and finding that is a
    result.
``incomplete``
    The ratio is above 1 and terms are missing. The budget is a lower bound, so the excess is
    unattributed rather than unmodelled. Naming this separately is the whole point: the flattering
    reading of an open budget is that you have found something, and it is usually that you did not
    measure something.
``over``
    The ratio is below 1 with its interval clear of it. The budget claims more scatter than the
    residual has, which means a term is double-counted or a bound is loose. It is a defect in the
    budget rather than in the run and it is reported rather than clipped.

The interval is a cluster bootstrap over **step pairs**. The `k` features of one step share one
batch of rollouts and one optimiser update, so they are one observation, and the same five-cluster
floor `measure.ledger.explained` derives for its own interval applies here for the same reason: a
bootstrap over `K` clusters resolving a tail of mass `(1-ci)/2` needs at least `2/(1-ci)` distinct
resamples, which at 95% puts the floor at `K = 5`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from reward_lens.measure.reconcile.books import StepReconciliation
from reward_lens.measure.reconcile.residual import FeatureBudget

Verdict = Literal["closed", "unmodelled", "incomplete", "over", "undetermined"]

#: Below this many step pairs the interval is declined rather than reported. Derived in
#: `measure.ledger.explained._bootstrap_lambda` and repeated here rather than re-derived.
MIN_CLUSTERS: int = 5


@dataclass(frozen=True)
class FeatureClosure:
    """One feature's verdict, with the numbers that produced it rather than only the word."""

    feature: str
    ratio: float
    ci_low: float
    ci_high: float
    ci_level: float
    var_residual: float
    accounted: float
    n_steps: int
    n_missing: int
    verdict: Verdict
    dominant: str
    coverage_k: float
    effective_dof: float | None
    #: The rollout-level spread of this feature over the window, so the floor below has a scale.
    feature_sd: float = float("nan")
    #: The root-mean-square first-order prediction, which is what the closure test exists to
    #: arbitrate. When this sits far below `detectable_u`, a closed budget says the test had no
    #: power at the scale of the thing it was testing.
    predicted_rms: float = float("nan")
    detail: str = ""

    @property
    def closed(self) -> bool:
        return self.verdict == "closed"

    @property
    def detectable_u(self) -> float:
        """The smallest extra term this test would have separated from the budget it has.

        `sqrt((ci_high − 1) · Σu²)`: a contribution below it composes into the quadrature sum
        without moving the ratio outside its interval, so the test cannot see it. This is the
        closure test's own limit of detection and it belongs beside every verdict of `closed`,
        because "the budget accounts for the residual" and "the budget is too coarse to tell"
        produce the same word and are not the same finding.
        """
        if not (np.isfinite(self.ci_high) and np.isfinite(self.accounted)):
            return float("nan")
        return float(np.sqrt(max(self.ci_high - 1.0, 0.0) * self.accounted))

    @property
    def powered_at_prediction(self) -> bool:
        """Whether the test could have seen a term the size of the prediction it is arbitrating."""
        floor, predicted = self.detectable_u, self.predicted_rms
        return bool(np.isfinite(floor) and np.isfinite(predicted) and predicted >= floor)

    def render(self) -> str:
        dof = f"{self.effective_dof:.1f}" if self.effective_dof is not None else "none"
        head = (
            f"{self.feature:<20} Var(rho) {self.var_residual:.5g} against sum u^2 "
            f"{self.accounted:.5g}: ratio {self.ratio:.4g} "
            f"[{self.ci_low:.4g}, {self.ci_high:.4g}] at {self.ci_level:.0%}  -> {self.verdict}"
            f"  (dominant {self.dominant}, k {self.coverage_k:.3g}, nu_eff {dof}, "
            f"{self.n_missing} term(s) not computed)"
        )
        if self.verdict == "closed" and not self.powered_at_prediction:
            head += (
                f"\n        floor: this test separates an extra term only above "
                f"{self.detectable_u:.4g}, and the first-order prediction it arbitrates is "
                f"{self.predicted_rms:.4g}. The budget closes and it could not have failed at the "
                f"scale of the thing it is testing."
            )
        return head


@dataclass(frozen=True)
class ClosureResult:
    """The closure test over every feature of one run, and the run-level verdict.

    ``verdict`` is the whole run's, and it is the **worst** of the per-feature verdicts rather than
    a vote: a budget that closes on two features and leaves a third unaccounted has not closed, and
    reporting the majority would hide exactly the feature worth looking at.
    """

    run_id: str
    features: tuple[FeatureClosure, ...]
    verdict: Verdict
    n_steps: int
    n_features: int
    detail: str = ""

    @property
    def closed(self) -> bool:
        return self.verdict == "closed"

    def render(self) -> str:
        lines = [
            f"budget closure on {self.run_id}: {self.verdict.upper()} over {self.n_features} "
            f"feature(s) and {self.n_steps} step pairs",
            *(f"    {f.render()}" for f in self.features),
        ]
        if self.detail:
            lines.append(f"    {self.detail}")
        return "\n".join(lines)


#: Worst first. `undetermined` outranks the rest because a verdict that could not be reached is not
#: evidence that the budget closed, and an aggregate that treated it as neutral would say it was.
_SEVERITY: tuple[Verdict, ...] = ("undetermined", "incomplete", "unmodelled", "over", "closed")


def _bootstrap_ratio(
    residuals: np.ndarray, accounted: float, *, n_bootstrap: int, ci: float, seed: int
) -> tuple[float, float]:
    """Percentile interval on `Var(ρ) / Σ u_i²`, resampling whole step pairs.

    Only the numerator is resampled. `Σ u_i²` is a budget composed from window-level statistics and
    resampling it alongside would double-count the same sampling variation: the terms are already
    root-mean-squared over the same steps the numerator is computed on.
    """
    n = residuals.size
    if n < MIN_CLUSTERS or n_bootstrap <= 0 or not np.isfinite(accounted) or accounted <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_bootstrap, n))
    values = np.var(residuals[draws], axis=1, ddof=1) / accounted
    finite = values[np.isfinite(values)]
    if finite.size < 10:
        return float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(finite, alpha)), float(np.quantile(finite, 1.0 - alpha))


def _verdict_of(ratio: float, low: float, high: float, n_missing: int) -> tuple[Verdict, str]:
    if not np.isfinite(ratio):
        return "undetermined", "the ratio is not finite, so no verdict was reached"
    if not (np.isfinite(low) and np.isfinite(high)):
        return (
            "undetermined",
            f"fewer than {MIN_CLUSTERS} step pairs, so the interval on the ratio is declined and "
            f"a point ratio of {ratio:.4g} is not a verdict",
        )
    if low <= 1.0 <= high:
        return "closed", "the interval on the ratio contains 1"
    if high < 1.0:
        return (
            "over",
            "the budget composes to more scatter than the residual has, which is a defect in the "
            "budget rather than a property of the run. Two mechanisms produce it on a short "
            "window and both are worth checking before looking for a third. `u_MC` is a "
            "root-mean-square over steps of a per-step standard error, and the root-mean-square of "
            "a noisy quantity overstates it, so a window of few steps at a small group size "
            "inflates that term. And consecutive step pairs share a step, which makes the residual "
            "negatively autocorrelated and biases the sample variance across overlapping pairs "
            "downward by roughly 2/n. Widen the window, or compare against a variance estimated "
            "from non-overlapping pairs.",
        )
    if n_missing:
        return (
            "incomplete",
            f"the ratio is above 1 and {n_missing} of the nine terms were not computed, so the "
            f"budget is a lower bound and the excess is unattributed rather than unmodelled",
        )
    return (
        "unmodelled",
        "every itemised term was computed and the residual variance still exceeds them, "
        "so there is a contribution the itemisation does not name",
    )


def closure_of(
    budgets: Sequence[FeatureBudget],
    reconciliations: Sequence[StepReconciliation],
    *,
    run_id: str = "",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> ClosureResult:
    """The closure test per feature and for the run, with the interval that decides each verdict."""
    if not budgets or not reconciliations:
        return ClosureResult(
            run_id=run_id,
            features=(),
            verdict="undetermined",
            n_steps=0,
            n_features=0,
            detail="no reconciled step pair, so there is no residual to budget",
        )
    out: list[FeatureClosure] = []
    for budget in budgets:
        residuals = np.asarray(
            [rec.row(budget.feature).residual for rec in reconciliations], dtype=np.float64
        )
        residuals = residuals[np.isfinite(residuals)]
        low, high = _bootstrap_ratio(
            residuals, budget.accounted, n_bootstrap=n_bootstrap, ci=ci, seed=seed
        )
        verdict, detail = _verdict_of(budget.ratio, low, high, len(budget.missing))
        dominant = budget.budget.dominant
        predictions = np.asarray(
            [rec.row(budget.feature).delta_z_pred for rec in reconciliations], dtype=np.float64
        )
        predictions = predictions[np.isfinite(predictions)]
        observed = np.asarray(
            [rec.row(budget.feature).delta_z_obs for rec in reconciliations], dtype=np.float64
        )
        observed = observed[np.isfinite(observed)]
        out.append(
            FeatureClosure(
                feature_sd=float(np.std(observed, ddof=1)) if observed.size > 1 else float("nan"),
                predicted_rms=(
                    float(np.sqrt(np.mean(predictions**2))) if predictions.size else float("nan")
                ),
                feature=budget.feature,
                ratio=budget.ratio,
                ci_low=low,
                ci_high=high,
                ci_level=ci,
                var_residual=budget.var_residual,
                accounted=budget.accounted,
                n_steps=budget.n_steps,
                n_missing=len(budget.missing),
                verdict=verdict,
                dominant=dominant.name if dominant is not None else "none",
                coverage_k=budget.budget.coverage_factor,
                effective_dof=budget.budget.effective_dof(),
                detail=detail,
            )
        )
    worst = min(out, key=lambda f: _SEVERITY.index(f.verdict))
    return ClosureResult(
        run_id=run_id,
        features=tuple(out),
        verdict=worst.verdict,
        n_steps=len(reconciliations),
        n_features=len(out),
        detail=f"run verdict is the worst feature's: {worst.feature}, because {worst.detail}",
    )


__all__ = [
    "MIN_CLUSTERS",
    "ClosureResult",
    "FeatureClosure",
    "Verdict",
    "closure_of",
]

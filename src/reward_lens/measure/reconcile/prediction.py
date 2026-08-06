"""The two registered predictions this package resolves, and the analyses that resolve them.

**P9** ("the reconciliation residual closes against its itemised budget") and **P10** ("the Lande
slope is not zero") were frozen at `c746e9f`, before either instrument existed.
That ordering is the whole point: a prediction written after the evidence is a description.
`studies.freeze.freeze` hashes the spec and records the git sha, so an edit after the fact produces
a visibly different study version rather than a quiet correction.

Both specs are written out in full below and their hashes are asserted in the acceptance test, so a
later edit fails a test rather than passing quietly.

**Two diagnostics here were added after the freeze and are marked as not preregistered.** Running
the frozen analyses on the real records showed that both registered rules can be satisfied by a
measurement that had no power to fail, and a resolution that does not say so is a resolution that
claims more than it earned. `METRIC_CLOSURE_POWER` is the closure test's detection floor divided by
the first-order prediction it arbitrates: above 1, a verdict of `closed` means the budget is too
coarse to have failed. `METRIC_LANDE_DEGENERATE` is 1 when `G` was the covariance bound, which
makes `η G β = η S` and the Lande fit F2's `η_eff` rescaled rather than a test of Lande's equation.
Both are reported beside the registered metrics and never in place of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from reward_lens.measure.reconcile.closure import ClosureResult
from reward_lens.measure.reconcile.lande import LandeFit
from reward_lens.studies.freeze import FrozenStudy, freeze
from reward_lens.studies.spec import (
    Hypothesis,
    KillCriterion,
    Prediction,
    StudySpec,
    SubjectQuery,
)

# ---------------------------------------------------------------------------
# P9: the closure test
# ---------------------------------------------------------------------------

METRIC_CLOSURE_RATIO = "closure_ratio_worst_feature"
METRIC_CLOSURE_CI_LOW = "closure_ratio_ci_low"
METRIC_CLOSURE_CI_HIGH = "closure_ratio_ci_high"
METRIC_MISSING_TERMS = "n_terms_not_computed"
METRIC_RUNS_CLOSED = "n_runs_closed"
#: The registered rule of P9 is "`Var(rho)` accounted for by `sum u_i^2` within its stated
#: interval", which is a statement about an interval and not about a point. Registered
#: as the indicator so the comparison is mechanical: 1 when the interval on the ratio contains 1.
METRIC_INTERVAL_CONTAINS_ONE = "closure_interval_contains_one"

#: Not preregistered. The closure test's own limit of detection over the prediction it arbitrates.
#: Above 1 the test could not have failed at the scale of the thing it was testing.
METRIC_CLOSURE_POWER = "detectable_u_over_predicted_rms"

BUDGET_CLOSURE_SPEC = StudySpec(
    id="f4-budget-closure",
    title="Does the reconciliation residual close against its itemised budget?",
    science="S01-selection",
    hypotheses=(
        Hypothesis(
            id="H1",
            statement=(
                "On a real training record, the variance of the reconciliation residual "
                "rho = Delta z_obs - eta G C^-1 S is accounted for by the quadrature sum of the "
                "nine uncertainty contributions section 3.1.5 itemises. Reported as the ratio "
                "Var(rho) / sum u_i^2 with a cluster bootstrap interval over step pairs; the "
                "books close when that interval contains 1."
            ),
            prediction=Prediction(
                metric=METRIC_INTERVAL_CONTAINS_ONE,
                comparator="==",
                threshold=1.0,
                rationale=(
                    "The residual is a budget with named terms rather than noise. If the "
                    "itemisation is complete the observed scatter is the composition of the "
                    "terms, and if it is not there is a contribution nobody has named. Either "
                    "outcome is publishable, which is what makes this worth registering: a "
                    "closed budget characterises the instrument and an open one names an "
                    "unmodelled term."
                ),
            ),
            scoreboard_row="F4",
        ),
        Hypothesis(
            id="H2",
            statement=(
                "Whether the budget closes or not, every term that was not computed is named. A "
                "budget with missing terms is a lower bound, so an excess of Var(rho) over it is "
                "unattributed rather than unmodelled, and the verdict distinguishes the two."
            ),
            prediction=Prediction(
                metric=METRIC_MISSING_TERMS,
                comparator=">=",
                threshold=0.0,
                rationale=(
                    "Registered as an accounting obligation rather than as a directional claim. "
                    "The failure this guards against is reporting an open budget as a discovery "
                    "when the arithmetic is short a term the record never carried."
                ),
            ),
            scoreboard_row="F4",
        ),
    ),
    analysis="reward_lens.measure.reconcile.prediction.score_closure",
    subjects=SubjectQuery(
        datasets=(
            "tests/fixtures/grpo_run/short",
            "tests/fixtures/grpo_run/long",
        ),
        extra={
            "featuriser": "measure.ledger.features.SurfaceFeatures",
            "covariance_operator": "within_group",
            "ridge": 0.0,
            "ci": 0.95,
            "n_bootstrap": 1000,
        },
    ),
    kill_criteria=(
        KillCriterion(
            id="K1",
            metric=METRIC_RUNS_CLOSED,
            comparator="<",
            threshold=1.0,
            description=(
                "The budget closes on no run tested. The catalogue's kill for F4 is that the "
                "first-order model is wrong, and that is the result rather than a failure of the "
                "instrument."
            ),
        ),
    ),
    version=1,
    notes=(
        "Frozen before the instrument existed. The nine terms are the itemised uncertainty table "
        "and the composition, the Welch-Satterthwaite effective degrees of freedom and the "
        "coverage factor are core.budget's. The metric names a quantity rather than an "
        "implementation, so a better estimator of any single term does not change what was "
        "predicted."
    ),
)


def freeze_closure(repo_dir: str | None = None, frozen_at: str | None = None) -> FrozenStudy:
    """Freeze P9. The StudyID stamps readings taken under it REGISTERED."""
    return freeze(BUDGET_CLOSURE_SPEC, repo_dir=repo_dir, frozen_at=frozen_at)


@dataclass(frozen=True)
class ClosureResolution:
    """What P9 resolves to across the runs tested, including "it does not"."""

    n_runs: int
    n_closed: int
    worst_ratio: float
    worst_ci_low: float
    worst_ci_high: float
    worst_feature: str
    worst_run: str
    n_missing: int
    power_ratio: float
    resolved: bool
    verdicts: tuple[tuple[str, str], ...] = ()
    detail: str = ""

    @property
    def interval_contains_one(self) -> bool:
        """P9's registered rule on the worst feature: does the interval on the ratio contain 1?"""
        return bool(
            np.isfinite(self.worst_ci_low)
            and np.isfinite(self.worst_ci_high)
            and self.worst_ci_low <= 1.0 <= self.worst_ci_high
        )

    def metrics(self) -> dict[str, float]:
        return {
            METRIC_INTERVAL_CONTAINS_ONE: 1.0 if self.interval_contains_one else 0.0,
            METRIC_CLOSURE_RATIO: self.worst_ratio,
            METRIC_CLOSURE_CI_LOW: self.worst_ci_low,
            METRIC_CLOSURE_CI_HIGH: self.worst_ci_high,
            METRIC_MISSING_TERMS: float(self.n_missing),
            METRIC_RUNS_CLOSED: float(self.n_closed),
            METRIC_CLOSURE_POWER: self.power_ratio,
        }

    def render(self) -> str:
        lines = [
            f"P9: {self.n_closed} of {self.n_runs} run(s) closed. Worst feature "
            f"{self.worst_feature} on {self.worst_run}: ratio {self.worst_ratio:.4g} "
            f"[{self.worst_ci_low:.4g}, {self.worst_ci_high:.4g}], "
            f"{self.n_missing} term(s) not computed",
            *(f"    {run_id}: {verdict}" for run_id, verdict in self.verdicts),
        ]
        if np.isfinite(self.power_ratio) and self.power_ratio > 1.0:
            lines.append(
                f"    not preregistered: the closure test's detection floor is "
                f"{self.power_ratio:.4g} times the first-order prediction it arbitrates, so a "
                f"verdict of `closed` here could not have failed at the scale being tested."
            )
        if self.detail:
            lines.append(f"    {self.detail}")
        return "\n".join(lines)


def score_closure(results: Sequence[ClosureResult]) -> ClosureResolution:
    """Resolve P9 across one or more runs. Returns unresolved rather than a number.

    The worst feature across all runs decides, on the same argument `closure_of` uses within a run:
    a budget that closes on five features and leaves a sixth unaccounted has not closed, and a
    majority vote would hide the feature worth looking at.
    """
    features = [(result, f) for result in results for f in result.features]
    if not features:
        return ClosureResolution(
            n_runs=len(results),
            n_closed=0,
            worst_ratio=float("nan"),
            worst_ci_low=float("nan"),
            worst_ci_high=float("nan"),
            worst_feature="",
            worst_run="",
            n_missing=0,
            power_ratio=float("nan"),
            resolved=False,
            detail="no run produced a reconciled feature, so there is no residual to budget",
        )
    worst_result, worst = max(
        features,
        key=lambda pair: abs(pair[1].ratio - 1.0) if np.isfinite(pair[1].ratio) else float("inf"),
    )
    floors = [
        f.detectable_u / f.predicted_rms
        for _, f in features
        if np.isfinite(f.detectable_u) and np.isfinite(f.predicted_rms) and f.predicted_rms > 0
    ]
    return ClosureResolution(
        n_runs=len(results),
        n_closed=sum(1 for r in results if r.closed),
        worst_ratio=worst.ratio,
        worst_ci_low=worst.ci_low,
        worst_ci_high=worst.ci_high,
        worst_feature=worst.feature,
        worst_run=worst_result.run_id,
        n_missing=worst.n_missing,
        power_ratio=float(np.median(floors)) if floors else float("nan"),
        resolved=bool(np.isfinite(worst.ratio) and np.isfinite(worst.ci_low)),
        verdicts=tuple((r.run_id, r.verdict) for r in results),
    )


# ---------------------------------------------------------------------------
# P10: the Lande slope
# ---------------------------------------------------------------------------

METRIC_LANDE_SLOPE = "lande_slope"
METRIC_LANDE_CI_LOW = "lande_slope_ci_low"
METRIC_LANDE_CI_HIGH = "lande_slope_ci_high"
METRIC_LANDE_NULL_P = "permuted_step_null_p"
METRIC_LANDE_R2 = "lande_uncentred_r_squared"

#: Not preregistered. 1 when `G` was the covariance bound, which makes `eta*G*beta = eta*S` and
#: this fit F2's `eta_eff` rescaled. A slope measured there is not evidence about Lande's equation
#: whichever way it comes out, and the registered rule cannot see the difference.
METRIC_LANDE_DEGENERATE = "g_is_covariance_bound"

LANDE_SLOPE_SPEC = StudySpec(
    id="f6-lande-slope",
    title="Is the Lande slope different from zero on a real training run?",
    science="S01-selection",
    hypotheses=(
        Hypothesis(
            id="H1",
            statement=(
                "Regressing observed Delta z on eta G beta across a window of a real training run, "
                "through the origin and with each feature scaled by its own pooled spread, gives a "
                "slope whose bootstrap interval excludes zero. Lande's equation is the "
                "load-bearing assumption of the whole Level 1 story and a slope near zero retires "
                "it."
            ),
            prediction=Prediction(
                metric=METRIC_LANDE_SLOPE,
                comparator="!=",
                threshold=0.0,
                ci_excludes=0.0,
                rationale=(
                    "Delta z = eta G beta is derived from the natural-gradient step rather than "
                    "transplanted, so a slope indistinguishable from zero would say the "
                    "derivation does not describe what a real optimiser does to real "
                    "behavioural features. That is a publishable result about how policy "
                    "optimisation differs from natural selection, which is why it is registered "
                    "in the direction that can fail."
                ),
            ),
            scoreboard_row="F6",
        ),
        Hypothesis(
            id="H2",
            statement=(
                "The slope beats a permuted-step null, which pairs each step's Delta z with "
                "another step's predicted response. The identity claims a within-step "
                "correspondence and the permutation is the exact null for that claim."
            ),
            prediction=Prediction(
                metric=METRIC_LANDE_NULL_P,
                comparator="<",
                threshold=0.05,
                rationale=(
                    "A slope can be large and mean nothing when the regressor and the response "
                    "share a common scale across steps. Permuting the step index destroys the "
                    "within-step pairing and keeps everything else, so a slope that survives it is "
                    "a slope about the step rather than about the window."
                ),
            ),
            scoreboard_row="F6",
        ),
    ),
    analysis="reward_lens.measure.reconcile.prediction.score_lande",
    subjects=SubjectQuery(
        datasets=("tests/fixtures/grpo_run/long",),
        extra={
            "featuriser": "measure.ledger.features.SurfaceFeatures",
            "fit": "through-origin OLS, features scaled by sd(f), clustered on step pairs",
            "ci": 0.95,
            "n_bootstrap": 1000,
            "n_null_draws": 1000,
        },
    ),
    kill_criteria=(
        KillCriterion(
            id="K1",
            metric=METRIC_LANDE_R2,
            comparator="<",
            threshold=1e-3,
            description=(
                "The fit explains none of the variance it is fitted to, so the slope is a line "
                "through a cloud and its value is not a statement about Lande's equation. "
                "Reported rather than suppressed, because an uninformative fit on a real run is "
                "itself a fact about the subject."
            ),
        ),
    ),
    version=1,
    notes=(
        "Frozen before the instrument existed. The metric names the regression and not the "
        "estimator of G, so substituting a rung-2 Fisher G for a rung-0 one does not change what "
        "was predicted. It does change whether the regression has independent content, which is "
        "why the resolution reports the estimator alongside the slope."
    ),
)


def freeze_lande(repo_dir: str | None = None, frozen_at: str | None = None) -> FrozenStudy:
    """Freeze P10. The StudyID stamps readings taken under it REGISTERED."""
    return freeze(LANDE_SLOPE_SPEC, repo_dir=repo_dir, frozen_at=frozen_at)


@dataclass(frozen=True)
class LandeResolution:
    """What P10 resolves to on one run, including "it does not resolve here"."""

    slope: float
    ci_low: float
    ci_high: float
    r_squared: float
    null_p: float
    null_median: float
    n_steps: int
    excludes_zero: bool
    is_degenerate: bool
    g_rung: int
    g_method: str
    resolved: bool
    detail: str = ""

    def metrics(self) -> dict[str, float]:
        return {
            METRIC_LANDE_SLOPE: self.slope,
            METRIC_LANDE_CI_LOW: self.ci_low,
            METRIC_LANDE_CI_HIGH: self.ci_high,
            METRIC_LANDE_NULL_P: self.null_p,
            METRIC_LANDE_R2: self.r_squared,
            METRIC_LANDE_DEGENERATE: 1.0 if self.is_degenerate else 0.0,
        }

    def render(self) -> str:
        lines = [
            f"P10: slope {self.slope:.6g} [{self.ci_low:.6g}, {self.ci_high:.6g}] over "
            f"{self.n_steps} step pairs, uncentred R^2 {self.r_squared:.5g}, permuted-step null "
            f"p = {self.null_p:.4f} (median {self.null_median:.6g}); interval excludes zero: "
            f"{self.excludes_zero}",
            f"    G at rung {self.g_rung} ({self.g_method})",
        ]
        if self.is_degenerate:
            lines.append(
                "    not preregistered: G is the covariance bound, so eta*G*beta reduces to eta*S "
                "and this is F2's eta_eff rescaled rather than a test of Lande's equation. The "
                "registered rule cannot see that, so the resolution is reported as unresolved "
                "whichever way the interval falls."
            )
        if self.detail:
            lines.append(f"    {self.detail}")
        return "\n".join(lines)


def score_lande(fit: LandeFit, null_draws: np.ndarray) -> LandeResolution:
    """Resolve P10 on one run against its permuted-step null.

    A degenerate fit resolves to `resolved=False` regardless of where the interval falls. The
    registered rule is "the interval excludes zero" and a covariance-bound `G` can satisfy or fail
    that rule for reasons that have nothing to do with Lande's equation, so reporting either
    outcome as a resolution would be claiming a result the analysis cannot support.
    """
    finite = null_draws[np.isfinite(null_draws)] if null_draws.size else null_draws
    if finite.size:
        exceed = int(np.count_nonzero(np.abs(finite) >= abs(fit.slope)))
        p_value = float((exceed + 1) / (finite.size + 1))
        median = float(np.median(finite))
    else:
        p_value, median = float("nan"), float("nan")
    return LandeResolution(
        slope=fit.slope,
        ci_low=fit.ci_low,
        ci_high=fit.ci_high,
        r_squared=fit.r_squared,
        null_p=p_value,
        null_median=median,
        n_steps=fit.n_steps,
        excludes_zero=fit.excludes_zero,
        is_degenerate=fit.is_degenerate,
        g_rung=fit.g_rung,
        g_method=fit.g_method,
        resolved=bool(
            not fit.is_degenerate and np.isfinite(fit.ci_low) and np.isfinite(fit.ci_high)
        ),
        detail=(
            ""
            if not fit.is_degenerate
            else "the independent test needs a G that is not C, which is the rung-2 Fisher solve "
            "at POLICY: BACKWARD."
        ),
    )


__all__ = [
    "BUDGET_CLOSURE_SPEC",
    "LANDE_SLOPE_SPEC",
    "METRIC_CLOSURE_CI_HIGH",
    "METRIC_CLOSURE_CI_LOW",
    "METRIC_CLOSURE_POWER",
    "METRIC_CLOSURE_RATIO",
    "METRIC_LANDE_CI_HIGH",
    "METRIC_LANDE_CI_LOW",
    "METRIC_LANDE_DEGENERATE",
    "METRIC_LANDE_NULL_P",
    "METRIC_LANDE_R2",
    "METRIC_LANDE_SLOPE",
    "METRIC_INTERVAL_CONTAINS_ONE",
    "METRIC_MISSING_TERMS",
    "METRIC_RUNS_CLOSED",
    "ClosureResolution",
    "LandeResolution",
    "freeze_closure",
    "freeze_lande",
    "score_closure",
    "score_lande",
]

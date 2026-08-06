"""H3, rate-extrapolated hysteresis: telling genuine irreversibility apart from lag.

A cusp model predicts that a system pushed past a fold does not come back when the parameter is
lowered again, and reward parameters are annealed monotonically and never back, so nobody runs the
reverse sweep. Running it produces a loop, and here is the problem the catalogue names in its own
baseline: **a raw loop area at one sweep rate is confounded with lag.** Any system driven faster
than it relaxes traces a loop, whether or not it has two stable states, because on the way up it has
not finished rising and on the way down it has not finished falling. A loop is not bistability.

What separates them is the limit. Genuine hysteresis is a property of the landscape and survives an
infinitely slow sweep; lag is a property of the driving and vanishes with it. So the quantity worth
reporting is the **rate-extrapolated loop area**, `A(v)` fitted over at least three sweep rates and
evaluated at `v = 0`:

    A(v) = A0 + c * v ** alpha

`A0` is `run.hysteresis_area` at rung 1 and it is the number the verdict is taken on. The raw area at
a single rate is the same quantity at rung 0, and it is registered as such and scored as the
baseline, because it is what the literature reports and it is the number this instrument exists to
qualify.

**The exponent is the awkward part and it is handled by not pretending.** Dynamic hysteresis in
driven bistable systems has a power-law rate dependence whose exponent is model-dependent, with two
thirds and one half both derived for particular classes. Fitting an exponent from three points is
not an estimate. So `alpha` is fixed at 1.0 by default, which is the first-order expansion of any
smooth `A(v)` about zero and is the assumption that needs the fewest rates; it can be fitted when
five or more rates are supplied, and the reading carries which was done. The fixed-exponent fit is
biased in a knowable direction when the truth is sublinear: a concave `A(v)` fitted with a straight
line **over-predicts** the intercept, so the default errs toward finding hysteresis. That is the
wrong direction for a claim and the right direction for a kill criterion, and it is stated on the
reading rather than buried.

**The interval is what makes the kill condition evaluable.** The catalogue's kill condition for H3 is
"if the extrapolated area is zero", and a point estimate cannot fire it. Every rate should therefore
be run at several seeds; the replicate spread at each rate becomes the weight in a weighted least
squares and the intercept's standard error falls out of it. With one sweep per rate the residual
spread is used instead and the degrees of freedom drop to the number of rates minus the number of
fitted parameters, which at three rates and a fixed exponent is one, and a one-degree-of-freedom
interval carries a t-multiplier of 12.7. That is an honest interval and it is a wide one, and it is
the reason the runbook asks for seeds rather than for more rates.

This module fits areas. Producing them is the compute, and `loops.anneal.run_hysteresis` is the
protocol runner this composes: it sweeps a control parameter up and back and integrates the enclosed
area, and `sweep_areas` calls it once per rate rather than reimplementing a shoelace integral.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
from scipy import stats
from scipy.optimize import least_squares

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import make_evidence, register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.provenance import Provenance
from reward_lens.core.quantity import (
    BiasStatement,
    CostModel,
    EstimatorEntry,
    register_estimator,
)
from reward_lens.core.reading import (
    Reading,
    Refusal,
    RefusalReason,
    bounded_refusal,
    refuse_incomplete,
)
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    SubjectRef,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context, run
from reward_lens.measure.rate.regime import MEASURED_BY

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


@dataclass(frozen=True)
class HysteresisCriteria:
    """Every number a verdict here is compared against, in one place, with where it came from."""

    #: Sweep rates needed. **Chosen: 3**, which is the catalogue's own `access_min` and is the
    #: smallest number that leaves a degree of freedom after fitting an intercept and a slope.
    min_rates: int = 3

    #: Rates needed before the exponent is fitted rather than fixed. **Chosen: 5.** Three
    #: parameters from four points leaves one degree of freedom and an exponent estimated from it
    #: is a number with no interval worth printing.
    min_rates_for_exponent: int = 5

    #: The exponent used when it is not fitted. **Chosen: 1.0**, the first-order term of any smooth
    #: A(v) about zero.
    fixed_exponent: float = 1.0

    #: Coverage of the intercept interval.
    ci_level: float = 0.95

    #: Smallest ratio between the fastest and slowest rate. **Chosen: 4.0.** Extrapolating to zero
    #: from three rates within a factor of two of each other is extrapolating a long way past the
    #: data, and the intercept is then a property of the fit rather than of the system.
    min_rate_span: float = 4.0


# ---------------------------------------------------------------------------
# The sweeps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepArea:
    """One sweep rate's loop area, with the spread across seeds if there was more than one.

    `rate` is the control parameter's change per settling opportunity, which is what "sweep rate"
    means for a protocol that settles the system at each schedule point: halving the number of
    points across a fixed range doubles the rate.
    """

    rate: float
    area: float
    sd: float = float("nan")
    n_seeds: int = 1
    label: str = ""

    @classmethod
    def from_seeds(cls, rate: float, areas: Sequence[float], *, label: str = "") -> "SweepArea":
        a = np.asarray([float(v) for v in areas], dtype=np.float64)
        a = a[np.isfinite(a)]
        return cls(
            rate=float(rate),
            area=float(np.mean(a)) if a.size else float("nan"),
            sd=float(np.std(a, ddof=1)) if a.size > 1 else float("nan"),
            n_seeds=int(a.size),
            label=label,
        )


def sweep_areas(
    responder: Callable[[float, float], float],
    *,
    lam0: float,
    lam1: float,
    n_points: Sequence[int],
    init_state: float = -1.0,
    seeds: Sequence[int] = (0,),
    perturb: float = 0.0,
) -> tuple[SweepArea, ...]:
    """Run the up-and-back protocol at several sweep rates and collect the loop areas.

    Composes `loops.anneal.run_hysteresis`, which is the shipped protocol runner: it folds the
    responder over the up leg carrying state, continues down from where the up leg ended, and
    integrates the enclosed area by the shoelace formula. Nothing here reimplements that.

    The sweep rate is varied by varying `n_points` over a fixed `lam0` to `lam1` range, which is the
    same lever the two-run rate test uses: the responder gets one settling opportunity per schedule
    point, so fewer points over the same range is a faster sweep. `rate` is reported as
    ``(lam1 - lam0) / (n_points - 1)``.

    `seeds` and `perturb` exist so that a deterministic responder can be run at several seeds: with
    `perturb` above zero each seed jitters the initial state, which is what gives the replicate
    spread the weighted fit needs. A responder that is stochastic in its own right can ignore both
    and be called once per seed.
    """
    from reward_lens.loops.anneal import linear_schedule, run_hysteresis

    out: list[SweepArea] = []
    for n in n_points:
        areas: list[float] = []
        for s in seeds:
            rng = np.random.default_rng(int(s))
            start = float(init_state) + (rng.normal(0.0, perturb) if perturb > 0 else 0.0)
            up = linear_schedule(float(lam0), float(lam1), int(n))
            ev = run_hysteresis(responder, up, init_state=start)
            areas.append(float(ev.value.loop_area))
        out.append(
            SweepArea.from_seeds(
                (float(lam1) - float(lam0)) / float(max(int(n) - 1, 1)),
                areas,
                label=f"{int(n)} points",
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# The extrapolation
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class RateExtrapolatedArea:
    """The loop area extrapolated to zero sweep rate, with the interval the kill condition needs.

    `area_zero` is `run.hysteresis_area` at rung 1: the limit of the loop area as the sweep rate
    goes to zero. `genuine` is True when the interval on it excludes zero, which is the statement
    that the irreversibility survives an arbitrarily slow sweep and is therefore a property of the
    landscape rather than of the driving.

    `raw_area_fastest` and `raw_area_slowest` are the rung-0 numbers, the raw areas at one rate. The
    distance between `raw_area_slowest` and `area_zero` is how much of a careful single-rate sweep's
    loop was still lag, and it is the number that says whether this instrument was worth the extra
    sweeps.
    """

    area_zero: float
    ci: tuple[float, float]
    slope: float
    exponent: float
    exponent_fitted: bool
    dof: int
    r2: float
    #: Observed scatter about the fit divided by the scatter the seed replicates predict. Above one
    #: means the model is missing something the replicates cannot see, and the interval has been
    #: widened by this factor. NaN when the fit was unweighted, where there is nothing to compare.
    birge: float
    n_rates: int
    n_seeds_min: int
    rate_span: float
    sweeps: tuple[SweepArea, ...]
    weighted: bool

    @property
    def genuine(self) -> bool:
        """Whether the extrapolated area is distinguishable from zero."""
        return math.isfinite(self.ci[0]) and self.ci[0] > 0.0

    @property
    def raw_area_fastest(self) -> float:
        return max(self.sweeps, key=lambda s: s.rate).area

    @property
    def raw_area_slowest(self) -> float:
        return min(self.sweeps, key=lambda s: s.rate).area

    def says(self) -> str:
        if self.genuine:
            return (
                f"The loop area extrapolated to zero sweep rate is {self.area_zero:.4g} "
                f"[{self.ci[0]:.4g}, {self.ci[1]:.4g}], so the irreversibility is genuine rather "
                f"than lag."
            )
        return (
            f"The loop area extrapolated to zero sweep rate is {self.area_zero:.4g} "
            f"[{self.ci[0]:.4g}, {self.ci[1]:.4g}], which does not exclude zero. The loop measured "
            f"at any one rate is consistent with lag, and there is no evidence here of a second "
            f"stable state."
        )

    def render(self) -> str:
        how = (
            f"exponent fitted at {self.exponent:.3g}"
            if self.exponent_fitted
            else f"exponent fixed at {self.exponent:.3g}"
        )
        if not self.weighted:
            weights = (
                "unweighted, because no rate was run at more than one seed, so the interval rests "
                "on the residual spread across rates alone"
            )
        elif math.isfinite(self.birge) and self.birge > 1.0:
            weights = (
                f"weighted by the seed spread at each rate (at least {self.n_seeds_min} seeds "
                f"each), with the interval widened {self.birge:.3g}-fold because the scatter about "
                f"the fit is that much larger than the seeds predict: the rate model is missing "
                f"something the replicates cannot see"
            )
        else:
            weights = (
                f"weighted by the seed spread at each rate (at least {self.n_seeds_min} seeds "
                f"each), which accounts for the scatter about the fit"
            )
        bias = (
            ""
            if self.exponent_fitted
            else (
                " A fixed linear exponent over-predicts the intercept when the true rate "
                "dependence is sublinear, so this estimate errs toward finding hysteresis."
            )
        )
        return (
            f"{self.says()}\n"
            f"    Fitted over {self.n_rates} rates spanning a factor of {self.rate_span:.3g}, "
            f"{how}, {self.dof} residual degrees of freedom, r-squared {self.r2:.4f}, {weights}."
            f"{bias}\n"
            f"    The raw area is {self.raw_area_fastest:.4g} at the fastest rate and "
            f"{self.raw_area_slowest:.4g} at the slowest; a single-rate sweep would have reported "
            f"one of those, and {self.raw_area_slowest - self.area_zero:+.4g} of the slowest one "
            f"is lag."
        )


def _fit_area(
    rates: np.ndarray,
    areas: np.ndarray,
    weights: np.ndarray | None,
    exponent: float | None,
    criteria: HysteresisCriteria,
) -> tuple[float, float, float, bool, int, float, np.ndarray]:
    """Fit ``A = A0 + c * v ** alpha``. Returns A0, c, alpha, fitted, dof, r2, residuals."""
    w = np.ones_like(areas) if weights is None else weights
    if exponent is not None:
        design = np.column_stack([np.ones_like(rates), rates**exponent])
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(design * sw[:, None], areas * sw, rcond=None)
        fitted_values = design @ coef
        k, alpha, is_fitted = 2, float(exponent), False
    else:

        def residual(p: np.ndarray) -> np.ndarray:
            return np.sqrt(w) * (p[0] + p[1] * rates ** max(p[2], 1e-3) - areas)

        guess = np.array([float(np.min(areas)), 1.0, criteria.fixed_exponent])
        out = least_squares(
            residual,
            guess,
            bounds=(np.array([-np.inf, -np.inf, 0.05]), np.array([np.inf, np.inf, 4.0])),
            max_nfev=4000,
        )
        coef = out.x[:2]
        alpha = float(out.x[2])
        fitted_values = coef[0] + coef[1] * rates**alpha
        k, is_fitted = 3, True

    resid = areas - fitted_values
    dof = int(areas.size - k)
    ss_tot = float(np.sum((areas - areas.mean()) ** 2))
    r2 = float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else float("nan")
    return float(coef[0]), float(coef[1]), alpha, is_fitted, dof, r2, resid


def rate_extrapolated_area(
    sweeps: Sequence[SweepArea],
    *,
    criteria: HysteresisCriteria | None = None,
    instrument: str = "RateExtrapolatedHysteresis",
) -> "RateExtrapolatedArea | Refusal":
    """The loop area at zero sweep rate, or the reason these sweeps cannot support one.

    Four ways this refuses:

    `RECORD_INCOMPLETE` with fewer than `min_rates` rates. The catalogue's own access line says
    three sweep rates and the arithmetic agrees: two rates and two parameters leave no residual and
    the fit passes through both points whatever they are.

    `ENVELOPE_VIOLATED` when the rates span less than `min_rate_span`. Extrapolating to zero from
    rates within a factor of two of each other puts the answer far outside the data.

    `ABOVE_LOD_BELOW_LOQ`, carrying the raw area at the slowest rate as an upper bound, when the fit
    has no residual degrees of freedom. The point estimate exists and no interval does, the kill
    condition is a statement about an interval, and the slowest raw area bounds the extrapolated one
    from above whenever the area increases with rate.

    `BELOW_LOD` when every measured area is itself indistinguishable from zero. There is no loop to
    extrapolate, which is a stronger and cleaner negative than an intercept at zero.
    """
    criteria = criteria or HysteresisCriteria()
    live = [s for s in sweeps if math.isfinite(s.rate) and math.isfinite(s.area) and s.rate > 0]
    if len(live) < criteria.min_rates:
        return refuse_incomplete(
            instrument,
            field=f"loop areas at at least {criteria.min_rates} distinct sweep rates",
            subject=f"this protocol run ({len(live)} usable)",
            remedy=(
                f"run the up-and-back sweep at {criteria.min_rates} rates spanning at least a "
                f"factor of {criteria.min_rate_span:.0f}, holding the parameter range fixed and "
                f"changing only how many steps the sweep takes to cross it. Two rates fit two "
                f"parameters exactly and the intercept from them is not an estimate."
            ),
            n=len(live),
            floor=criteria.min_rates,
        )

    rates = np.array([s.rate for s in live], dtype=np.float64)
    areas = np.array([s.area for s in live], dtype=np.float64)
    span = float(rates.max() / rates.min())
    if span < criteria.min_rate_span:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ENVELOPE_VIOLATED,
            detail=(
                f"the {len(live)} sweep rates span a factor of {span:.3g}, from {rates.min():.4g} "
                f"to {rates.max():.4g}, below the {criteria.min_rate_span:.3g} this extrapolation "
                f"needs"
            ),
            remedy=(
                f"add a sweep at least {criteria.min_rate_span:.0f} times slower than the fastest "
                f"one. The slow sweep is the expensive one and it is the one carrying the "
                f"information: the intercept is being read off the end of the fitted line and a "
                f"short lever makes it a property of the fit."
            ),
            statistics={"rate_span": span, "min_span": criteria.min_rate_span},
        )

    sds = np.array([s.sd for s in live], dtype=np.float64)
    n_seeds = np.array([s.n_seeds for s in live], dtype=np.int64)
    weighted = bool(np.all(np.isfinite(sds)) and np.all(sds > 0) and np.all(n_seeds > 1))
    weights = 1.0 / (sds**2 / n_seeds) if weighted else None

    if weighted and float(np.max(np.abs(areas) / (sds / np.sqrt(n_seeds)))) < 2.0:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.BELOW_LOD,
            detail=(
                f"no sweep rate produced a loop area distinguishable from zero; the largest is "
                f"{float(np.max(np.abs(areas))):.4g} against a seed standard error of "
                f"{float(sds[int(np.argmax(np.abs(areas)))] / math.sqrt(n_seeds[int(np.argmax(np.abs(areas)))])):.4g}"
            ),
            remedy=(
                "check that the reverse leg was actually run and that the responder carries state "
                "between schedule points. A protocol that re-initialises the system at each point "
                "retraces its own path exactly and has no loop by construction, which looks the "
                "same as a system with no hysteresis and is not."
            ),
            statistics={"max_area": float(np.max(np.abs(areas))), "n_rates": len(live)},
        )

    use_fitted_exponent = len(live) >= criteria.min_rates_for_exponent
    a0, slope, alpha, is_fitted, dof, r2, resid = _fit_area(
        rates, areas, weights, None if use_fitted_exponent else criteria.fixed_exponent, criteria
    )

    if dof < 1:
        bound = make_evidence(
            observable=instrument,
            observable_version="1.0",
            subject=SubjectRef(extra={"protocol": "up-and-back sweep"}),
            value=float(areas[int(np.argmin(rates))]),
            gauge=GaugeStatus.INVARIANT,
            provenance=Provenance(),
            quantity="run.hysteresis_area",
        )
        return bounded_refusal(
            instrument,
            RefusalReason.ABOVE_LOD_BELOW_LOQ,
            detail=(
                f"the fit over {len(live)} rates leaves {dof} residual degrees of freedom, so the "
                f"intercept {a0:.4g} has no interval and the kill condition, which is a statement "
                f"about an interval, cannot be evaluated"
            ),
            remedy=(
                "run each rate at three or more seeds so the replicate spread weights the fit and "
                "supplies the interval, or add a fourth rate. Seeds are the cheaper of the two: "
                "the slowest sweep dominates the cost and adding rates adds another one."
            ),
            bound=bound,
            intercept=a0,
            n_rates=len(live),
            dof=dof,
        )

    # The intercept's standard error, and the correction that stops the replicate spread from
    # deciding it on its own.
    #
    # A weighted fit's textbook covariance assumes the supplied variances are the whole of the
    # scatter, and here they are usually not: seeds of a deterministic sweep agree to nine decimal
    # places while the fit's residuals sit three orders of magnitude above that, because the model
    # A(v) = A0 + c v^alpha is an approximation and the replicates cannot see model error. Taking
    # the textbook interval in that case produces an interval of width zero around a fit artifact,
    # which is what the first version of this function did on a single-well responder that has no
    # bistability at all: it reported an extrapolated area of 1.5e-4 with an interval excluding
    # zero, and called lag genuine.
    #
    # `birge` is the ratio of the observed scatter to the scatter the replicates predict, and
    # scaling the covariance by it when it exceeds one is the standard metrological treatment of
    # exactly this situation. It inflates and never deflates: a Birge ratio below one would mean
    # the fit explains more than the replicates permit, which is not a licence to narrow anything.
    design = (
        np.column_stack([np.ones_like(rates), rates**alpha])
        if not is_fitted
        else np.column_stack(
            [np.ones_like(rates), rates**alpha, slope * np.log(rates) * rates**alpha]
        )
    )
    w = np.ones_like(areas) if weights is None else weights
    xtwx = design.T @ (design * w[:, None])
    cov = np.linalg.pinv(xtwx)
    chi2_per_dof = float(np.sum(w * resid**2) / dof)
    birge = math.sqrt(max(chi2_per_dof, 0.0)) if weighted else float("nan")
    scale = max(chi2_per_dof, 1.0) if weighted else chi2_per_dof
    se = float(math.sqrt(max(cov[0, 0] * scale, 0.0)))
    t = float(stats.t.ppf(0.5 + criteria.ci_level / 2.0, dof))
    return RateExtrapolatedArea(
        area_zero=a0,
        ci=(a0 - t * se, a0 + t * se),
        slope=slope,
        exponent=alpha,
        exponent_fitted=is_fitted,
        dof=dof,
        r2=r2,
        birge=birge,
        n_rates=len(live),
        n_seeds_min=int(n_seeds.min()),
        rate_span=span,
        sweeps=tuple(live),
        weighted=weighted,
    )


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

#: H3 cannot require `QUASI_STATIC`. The whole design is a set of sweeps at deliberately different
#: rates, most of which are not quasi-static on purpose, and the extrapolation is how the
#: quasi-static answer is reached without ever running a quasi-static sweep. `STATIONARY_GRADER` is
#: required and refused on: if the grader moved between the up leg and the down leg, the two
#: branches are branches of two different landscapes and the area between them is not a loop.
HYSTERESIS_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
    measured_by=MEASURED_BY,
    on_violation="refuse",
)

_HYSTERESIS_ACCESS: AccessMatrix = {
    Component.OPTIMIZER: Access.CONTROL,
    Component.RECORD: Access.RECORD,
}

#: The catalogue names one baseline, "the raw loop area at one rate, which is confounded with lag".
#: Both ends of the rate range are scored, because they are different numbers and a reader who ran
#: one careful slow sweep and a reader who ran one cheap fast sweep get different answers.
HYSTERESIS_BASELINES = (
    "baseline.raw_area_fastest_rate",
    "baseline.raw_area_slowest_rate",
)


class RateExtrapolatedHysteresis(BaseObservable):
    """H3. The loop area at zero sweep rate, which is the part of a loop that is not lag.

    Reads loop areas that have already been measured. `sweep_areas` produces them from a responder,
    and on a real training loop the responder is an arm of the loop rather than a function, which is
    the compute this package is gated on.

    What it cannot do. It extrapolates, so the answer is a statement about a rate nobody ran, and
    the fit's shape is an assumption: with fewer than five rates the exponent is fixed at one and a
    truly sublinear rate dependence makes the intercept too large. It also cannot tell a system with
    two stable states apart from one with a slow mode it never resolves, because both leave a
    nonzero intercept when the slowest sweep is still faster than the slow mode; the only cure for
    that is a slower sweep, and `rate_span` on the reading says how far the lever reached.
    """

    name = "RateExtrapolatedHysteresis"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to: str | None = "H3"
    deviations: tuple[str, ...] = (
        "the area is 'fitted from at least three rates' with no functional form stated. The "
        "form here is A0 + c v^alpha with alpha fixed at 1 below five rates, which is the "
        "first-order expansion about zero rather than any of the model-specific exponents in the "
        "dynamic-hysteresis literature.",
        "the interval is widened by the Birge ratio when the seed replicates under-explain the "
        "scatter about the fit. Nothing asks for this; it was added after a "
        "single-well responder with no bistability produced an interval of width zero around a fit "
        "artifact and was called genuine.",
    )

    quantity = "run.hysteresis_area"
    requires: AccessMatrix = _HYSTERESIS_ACCESS
    substrates = frozenset(Substrate)
    phases = frozenset({Phase.POST_RUN})
    envelope = HYSTERESIS_ENVELOPE
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = HYSTERESIS_BASELINES
    rung = 1

    def __init__(
        self,
        sweeps: Sequence[SweepArea],
        *,
        criteria: HysteresisCriteria | None = None,
    ) -> None:
        self.sweeps = tuple(sweeps)
        self.criteria = criteria or HysteresisCriteria()
        self._computed: RateExtrapolatedArea | None = None

    def compute(self) -> "RateExtrapolatedArea | Refusal":
        return rate_extrapolated_area(self.sweeps, criteria=self.criteria, instrument=self.name)

    def estimate(self, ctx: Context) -> Reading:
        """Preflight, compute, refuse or emit. Never a bare number, never a silent zero."""
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        out = self.compute()
        if isinstance(out, Refusal):
            return out
        self._computed = out
        try:
            return run(self, ctx)
        finally:
            self._computed = None

    def measure(self, ctx: Context) -> "Evidence":
        out = self._computed if self._computed is not None else self.compute()
        if isinstance(out, Refusal):
            raise ValueError(
                f"{self.name}.measure was called on sweeps that decline to produce Evidence: "
                f"{out.reason.name}. Call `estimate`, which returns the refusal as a value with "
                f"its remedy."
            )
        return ctx.emit(out, baselines=self.baseline_scores(out))

    def baseline_scores(self, reading: RateExtrapolatedArea) -> dict[str, float]:
        """The raw areas, which are what a single-rate sweep reports.

        These are the catalogue's named baseline and they are not a formality: the whole claim of
        this instrument is that they contain lag, so the distance between each of them and
        `area_zero` is the size of the error a single-rate sweep would have made on this system.
        """
        return {
            "baseline.raw_area_fastest_rate": float(reading.raw_area_fastest),
            "baseline.raw_area_slowest_rate": float(reading.raw_area_slowest),
        }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _register() -> None:
    """Two rungs, which is what `spec/QUANTITIES.yaml` gives `run.hysteresis_area`.

    Rung 0 is the raw area at one rate and it is exactly the catalogue's baseline: it is cheap, it
    is what gets published, and it is confounded with lag. Rung 1 is the extrapolation. Registering
    the confounded estimator as a rung rather than leaving it out is the honest arrangement, because
    a reader who can afford one sweep should be told what the one sweep gives them and what it
    costs them, not told the quantity is unavailable.
    """
    register_estimator(
        EstimatorEntry(
            quantity="run.hysteresis_area",
            impl="run.hysteresis_area.raw_single_rate",
            requires=_HYSTERESIS_ACCESS,
            envelope=HYSTERESIS_ENVELOPE,
            rung=0,
            bias=BiasStatement(
                direction="upward",
                why=(
                    "the raw loop area at any finite sweep rate is the rate-independent area plus "
                    "the lag, and the lag is non-negative because a system that has not finished "
                    "responding trails the branch it is on in both directions. Measured on a "
                    "single-well responder with no bistability at all: the area runs 0.0514 at the "
                    "fastest of four rates down to 0.0063 at the slowest, against a "
                    "rate-extrapolated 0.00015 whose interval contains zero. All of it is lag."
                ),
            ),
            cost=CostModel(note="one up-and-back sweep. The cheap rung, and the confounded one"),
            phases=frozenset({Phase.POST_RUN}),
            run=None,
        )
    )
    register_estimator(
        EstimatorEntry(
            quantity="run.hysteresis_area",
            impl="run.hysteresis_area.rate_extrapolated",
            requires=_HYSTERESIS_ACCESS,
            envelope=HYSTERESIS_ENVELOPE,
            rung=1,
            bias=BiasStatement(
                direction="upward",
                why=(
                    "the lag term is removed by extrapolation, and what is left is the shape "
                    "assumption. Below five rates the exponent is fixed at 1, and a straight line "
                    "through a concave A(v) crosses the axis above the true intercept, so a "
                    "sublinear rate dependence makes this too large. Upward is the direction that "
                    "makes the kill condition harder to fire, which is the right way round for a "
                    "kill condition and the wrong way round for a claim."
                ),
            ),
            cost=CostModel(
                note=(
                    "at least three up-and-back sweeps spanning a factor of four in rate, each at "
                    "three or more seeds. The slowest sweep dominates; studies/w6_rate prices it"
                )
            ),
            phases=frozenset({Phase.POST_RUN}),
            run=None,
        )
    )


_register()


__all__ = [
    "HYSTERESIS_BASELINES",
    "HYSTERESIS_ENVELOPE",
    "HysteresisCriteria",
    "RateExtrapolatedArea",
    "RateExtrapolatedHysteresis",
    "SweepArea",
    "rate_extrapolated_area",
    "sweep_areas",
]

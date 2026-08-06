"""H1 rung 0, the adiabaticity number: is this run being driven faster than it relaxes?

    Ad = tau_relax * |d log lambda / dt|

`tau_relax` is how many optimizer steps the system takes to return after a nudge. `lambda` is
whichever schedule parameter is annealing. `Ad` far below 1 is quasi-static, and Level 0
extrapolation and the whole critical-slowing-down early-warning toolbox are licensed on it. `Ad` of
order 1 or more is fast driving, and none of that is licensed, because a system that never catches
up with its driver has no equilibrium to extrapolate from. This is a `RegimeCondition` measured
per step, and it gates every instrument that assumes equilibrium.

**Rung 0 only.** The relaxation time here comes from the early lag-1 autoregression of a recorded
series, which needs a record and nothing else. Rung 1 is perturb-and-hold, which is the measurement
this quantity is defined by and which needs compute and a counterfactual arm; it is registered as a
rung with its access and its cost and it is not built here.

**Why this exists when `rate/regime.py` already computes one.** That module fits the same lag-1
coefficient by ordinary least squares and its own docstring says the estimator is biased low by
roughly ``(1 + 3 phi) / n``, that detrending costs about as much again, and that a short relaxation
time makes `Ad` small and `QUASI_STATIC` pass. So the cheap rung errs toward *licensing*, which is
the wrong direction for a safety check, and it recorded that rather than fixing it because
correcting it is a decision about what the library estimates. This module takes that decision. Two
things change and both push the same way:

The bias is estimated by parametric bootstrap and removed, so the coefficient is not systematically
short. The bootstrap is used rather than the closed-form correction because the estimator being
corrected is the whole pipeline, detrending included, and no textbook formula covers that pipeline.
Measured on planted first-order series, 200 replicates per cell at 50 points: the least-squares
coefficient is low by 0.035, 0.056, 0.083 and 0.124 at true coefficients of 0, 0.3, 0.6 and 0.85,
and the corrected one is off by +0.004, +0.007, +0.006 and -0.010. It is not free. Root mean squared
error improves from 0.148 to 0.137 at 0.6 and worsens from 0.147 to 0.156 at 0, which is the usual
trade and is worth taking here because the bias is the part that points one way on every run.

**The verdict is taken on the upper end of the interval, not on the point.** `Ad` licenses an
assumption, so `QUASI_STATIC` holding has to mean the assumption is established rather than merely
not contradicted. A point estimate that lands below the threshold with an interval crossing it has
established nothing. That interval is a basic bootstrap interval and it under-covers at small
samples: measured over 200 replicates, a nominal 95 percent interval on the relaxation time covers
the planted value 86 to 93 percent of the time at 20 points and 93 to 98 percent at 120. Under-
coverage makes the upper bound optimistic, so the licence is slightly easier to get than 95 percent
says, and the fix is more early steps rather than a wider nominal level.

The two rung-0 estimators therefore disagree by construction, and that disagreement is published as
a `Transfer` rather than reconciled: `tau_transfer` returns the chain term, which is
M11's whole argument applied to two estimators at one rung instead of two rungs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import SubjectRef, make_evidence, register_payload
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
from reward_lens.core.reference import Transfer, ladder_disagreement
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context, run
from reward_lens.measure.rate.regime import (
    MEASURED_BY,
    RegimeInputs,
    RegimeThresholds,
    Window,
    _linear_detrend,
    _relaxation_series,
)
from reward_lens.measure.rate.transition import window_steps
from reward_lens.record.schema import Run, Step

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence

#: The axis every quantity in this module is expressed per. Kept as a value rather than as a
#: comment because it is what `adiabaticity_number` refuses on: a relaxation time in optimizer
#: steps multiplied by a driving rate per epoch is a number in no unit at all, and it is exactly the
#: shape of error the `units` group exists to catch.
STEP_AXIS = "optimizer_step"


# ---------------------------------------------------------------------------
# The numbers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RelaxationFloors:
    """Sample-size floors that decide measured against not measured, not verdicts.

    The two that matter are inherited from `RegimeFloors` so the two estimators of `tau_relax` are
    fitted on the same window and their difference is a difference of method rather than a
    difference of data. That is what makes `tau_transfer` a transfer term and not an artifact.
    """

    #: Points needed for the early lag-1 fit. `RegimeFloors.ar_min_points` is 10 and this matches.
    min_points: int = 10
    #: How many steps from the start of the run count as early. `RegimeFloors.ar_early_steps` is 50
    #: and this matches. The relaxation time should be estimated before the transition,
    #: because a fit run over a transition measures the transition.
    early_steps: int = 50
    #: Parametric-bootstrap replicates for the bias and the interval. **Chosen: 400**, which puts
    #: the Monte Carlo error on the bias well below the bias itself at these sample sizes and costs
    #: a few milliseconds.
    n_boot: int = 400
    #: Coverage of the interval on the coefficient, and so of the interval on the relaxation time.
    ci_level: float = 0.95


# ---------------------------------------------------------------------------
# The relaxation time
# ---------------------------------------------------------------------------


def _phi_ols(x: np.ndarray) -> float:
    """The lag-1 coefficient of the detrended series, by ordinary least squares.

    Exactly the regime estimator, reusing its detrend so the two cannot drift apart. Detrending
    first is not optional: a training run's mean reward climbs, an autoregression fitted to a
    climbing series returns a coefficient near one whatever the dynamics are, and a coefficient
    near one turns into an unbounded relaxation time.
    """
    r = _linear_detrend(x)
    denominator = float(np.dot(r[:-1], r[:-1]))
    if denominator <= 0:
        return float("nan")
    return float(np.dot(r[:-1], r[1:]) / denominator)


def tau_of(phi: float) -> float:
    """``phi**k = exp(-k / tau)``, so ``tau = -1 / ln phi``, with both ends defined.

    Total on purpose, because the two ends are real answers rather than failures. A coefficient at
    or below zero is a series with no memory from one step to the next, which relaxes inside one
    step: the relaxation time is zero and `Ad` is zero, and that is the fastest-relaxing case
    rather than an undefined one. A coefficient at or above one is a series that does not return,
    whose relaxation time is unbounded and whose `Ad` is unbounded with it.

    The regime estimator returns no time at all in both cases and says so, which is right for a
    point estimate on
    one number. It is wrong for an interval, because an interval whose lower end is a
    non-relaxing series still has an upper end, and the upper end is the one a safety check needs.
    """
    if not math.isfinite(phi) or phi <= 0.0:
        return 0.0
    if phi >= 1.0:
        return float("inf")
    return -1.0 / math.log(phi)


@register_payload
@dataclass(frozen=True)
class RelaxationTime:
    """`tau_relax` in optimizer steps, with the bias that was removed and the interval that is left.

    `phi_ols` is what the regime estimator reports and `phi` is what this module reports; `bias` is
    the difference
    and it is measured rather than assumed. The interval is a basic bootstrap interval on the
    coefficient mapped through `tau_of`, which is monotone, so the ends stay the ends.
    """

    tau: float
    tau_low: float
    tau_high: float
    phi: float
    phi_ols: float
    phi_ci: tuple[float, float]
    #: The bootstrap estimate of ``E[phi_hat] - phi``, which is negative for this estimator. `phi`
    #: is ``phi_ols`` minus this.
    bias: float
    n: int
    series: str
    axis: str
    method: str
    note: str = ""

    @property
    def identified(self) -> bool:
        """Whether the upper end of the relaxation time is finite, which is what a bound needs."""
        return math.isfinite(self.tau_high)

    def render(self) -> str:
        return (
            f"tau_relax = {self.tau:.4g} steps [{self.tau_low:.4g}, {self.tau_high:.4g}] from the "
            f"early lag-1 coefficient of {self.series!r} over {self.n} steps: {self.phi_ols:.4g} "
            f"by least squares, {self.phi:.4g} after removing a measured bias of {self.bias:+.4g}"
        )


def relaxation_time(
    series: Sequence[float],
    *,
    name: str = "series",
    floors: RelaxationFloors | None = None,
    instrument: str = "Adiabaticity",
    seed: int = 0,
) -> "RelaxationTime | Refusal":
    """`tau_relax` at rung 0, bias-corrected, with an interval.

    The estimator is: detrend the early window linearly, take the lag-1 least-squares coefficient,
    then correct it by parametric bootstrap. The bootstrap simulates first-order autoregressive
    paths at the fitted coefficient, of the same length, through the identical detrend-and-fit
    pipeline, and the mean of the simulated coefficients minus the fitted one is the bias. The
    basic bootstrap interval falls out of the same simulation.

    The bootstrap is used instead of the closed-form ``-(1 + 3 phi) / n`` because that formula is
    for a coefficient fitted after removing a mean, and this pipeline removes a mean and a trend.
    Simulating the pipeline that is actually run is cheaper than finding the right formula for it
    and is right by construction.

    Two things this cannot fix, and both belong wherever the number is used. The correction is for
    the bias of the estimator under a first-order autoregressive model, so a series whose memory is
    not first-order is being summarised rather than measured, and the summary is a single
    time constant fitted to whatever the real spectrum is. And the early window is early by step
    index rather than by anything measured: if the run's transition begins inside the first fifty
    steps, this fits the transition. `TransitionWidth` is what tells you whether it did.
    """
    floors = floors or RelaxationFloors()
    x = np.asarray([float(v) for v in series], dtype=float)
    x = x[np.isfinite(x)]
    if x.size < floors.min_points:
        return refuse_incomplete(
            instrument,
            field=f"at least {floors.min_points} finite points in the early window",
            subject=f"the series {name!r} ({x.size} recorded)",
            remedy=(
                f"record {name!r} on every step of at least the first {floors.min_points} steps "
                f"of the run. A relaxation time is a property of how the series returns to its "
                f"own level, and nothing that short shows a return."
            ),
            n=int(x.size),
            floor=floors.min_points,
        )

    phi_ols = _phi_ols(x)
    if not math.isfinite(phi_ols):
        return refuse_incomplete(
            instrument,
            field="any variation about its own trend",
            subject=f"the series {name!r} over {x.size} steps, which is exactly linear, and so",
            remedy=(
                "fit the relaxation time to a series that fluctuates. A series lying exactly on "
                "its own trend line has no residual, so there is no autocorrelation in it to read "
                "a time off, and the honest reading is that this record does not identify one."
            ),
            n=int(x.size),
        )

    resid = _linear_detrend(x)
    innovation = float(np.std(resid)) or 1.0
    rng = np.random.default_rng(seed)
    n = int(x.size)
    simulated = np.empty(floors.n_boot, dtype=float)
    # A first-order path at the fitted coefficient, put through the same detrend-and-fit pipeline.
    # Vectorised over replicates would need a scan; the loop is 400 iterations of a 50-point
    # recursion and costs single-digit milliseconds.
    for b in range(floors.n_boot):
        e = rng.normal(0.0, innovation, n)
        path = np.empty(n)
        path[0] = e[0]
        for i in range(1, n):
            path[i] = phi_ols * path[i - 1] + e[i]
        simulated[b] = _phi_ols(path)
    simulated = simulated[np.isfinite(simulated)]
    if simulated.size < 20:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.BELOW_LOD,
            detail=(
                f"only {simulated.size} of {floors.n_boot} bootstrap replicates produced a "
                f"coefficient at all, so the bias of this estimator on this window is not "
                f"measurable and the uncorrected {phi_ols:.4g} would be reported as though it "
                f"were unbiased"
            ),
            remedy=(
                "report the relaxation time from the perturb-and-hold rung instead, or widen the "
                "early window so the fit has more residual to work with. The uncorrected "
                "coefficient is available on the regime reading and it is biased toward "
                "licensing, which is the direction that matters here."
            ),
            statistics={"n": n, "phi_ols": phi_ols, "replicates": int(simulated.size)},
        )

    bias = float(simulated.mean() - phi_ols)
    phi = float(phi_ols - bias)
    alpha = (1.0 - floors.ci_level) / 2.0
    # The basic (reverse percentile) interval, which bias-corrects by the same reflection that
    # produces `phi`, so the point estimate cannot sit outside its own interval.
    lo = float(2.0 * phi_ols - np.quantile(simulated, 1.0 - alpha))
    hi = float(2.0 * phi_ols - np.quantile(simulated, alpha))
    return RelaxationTime(
        tau=tau_of(phi),
        tau_low=tau_of(lo),
        tau_high=tau_of(hi),
        phi=phi,
        phi_ols=phi_ols,
        phi_ci=(lo, hi),
        bias=bias,
        n=n,
        series=name,
        axis=STEP_AXIS,
        method="early lag-1 least squares, parametric-bootstrap bias correction",
        note=(
            ""
            if hi < 1.0
            else (
                "the upper end of the coefficient interval is at or above one, so the upper end of "
                "the relaxation time is unbounded and no upper bound on Ad follows from it"
            )
        ),
    )


# ---------------------------------------------------------------------------
# The driving rate
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class DriveRate:
    """``|d log lambda / dt|`` across one pair of consecutive recorded steps.

    Per step rather than per window, because this condition is measured per step and the two are
    genuinely different. A window maximum is one number for forty steps and it
    hides which step it came from; on a schedule that decays linearly to zero the log derivative is
    small everywhere and unbounded at the last step, so the window maximum says "fast driving" about
    a run that was quasi-static for all but its final step.
    """

    from_step: int
    to_step: int
    rate: float
    parameter: str
    axis: str = STEP_AXIS


def drive_rates(steps: Sequence[Step], key: str | None = None) -> tuple[DriveRate, ...]:
    """One `DriveRate` per consecutive step pair: the fastest-moving schedule parameter and its rate.

    The maximum over parameters unless the caller names one, because the condition has to hold for
    whatever is actually driving the system and the fastest-moving parameter is the one that breaks
    it first. A parameter at or below zero is skipped rather than clamped: a coefficient annealed to
    exactly zero has an unbounded log derivative, which is a real statement about the schedule and
    not a number that belongs in a maximum.

    A pair with no usable parameter contributes nothing, so an empty result means the record
    carries no schedule this window can differentiate rather than that the schedule is flat. Those
    two are different and the caller has to tell them apart; `Adiabaticity.compute` does.
    """
    out: list[DriveRate] = []
    for left, right in zip(steps, steps[1:]):
        dt = float(right.index - left.index)
        if dt <= 0:
            continue
        keys = [key] if key is not None else sorted(set(left.schedule) & set(right.schedule))
        best_rate, best_key = -1.0, ""
        for k in keys:
            a, b = left.schedule.get(k), right.schedule.get(k)
            if a is None or b is None or a <= 0 or b <= 0:
                continue
            rate = abs(math.log(b) - math.log(a)) / dt
            if rate > best_rate:
                best_rate, best_key = rate, k
        if best_key:
            out.append(
                DriveRate(
                    from_step=int(left.index),
                    to_step=int(right.index),
                    rate=float(best_rate),
                    parameter=best_key,
                )
            )
    return tuple(out)


def adiabaticity_number(tau: RelaxationTime, rate: DriveRate) -> "float | Refusal":
    """``Ad = tau_relax * |d log lambda / dt|``, or the refusal that says the two are not on one axis.

    This is the `units` group's assertion for H1 and it is not ceremony. `Ad` is dimensionless only
    because a time in optimizer steps multiplies a rate per optimizer step; the same arithmetic on a
    time in steps and a rate per epoch, or per sample, or per wall-clock second, produces a number
    with no unit that looks exactly like a valid one. Converting is not available here either,
    because the factor is a property of the run rather than of the units.
    """
    if tau.axis != rate.axis:
        return Refusal(
            instrument="Adiabaticity",
            reason=RefusalReason.UNIT_MISMATCH,
            detail=(
                f"the relaxation time is measured per {tau.axis} and the driving rate per "
                f"{rate.axis}, so their product is not dimensionless and is not an adiabaticity "
                f"number"
            ),
            remedy=(
                f"express both on one axis before multiplying. Converting between them needs the "
                f"number of {tau.axis}s per {rate.axis}, which is a property of this run's "
                f"batching and schedule rather than of the units, so it has to be supplied and "
                f"cannot be inferred here."
            ),
            statistics={"tau_axis": tau.axis, "rate_axis": rate.axis},
        )
    return float(tau.tau * rate.rate)


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class StepAdiabaticity:
    """`Ad` at one step, with the interval the relaxation time puts on it."""

    from_step: int
    to_step: int
    rate: float
    parameter: str
    ad: float
    ad_low: float
    ad_high: float


@register_payload
@dataclass(frozen=True)
class AdiabaticityReading:
    """`Ad` per step over a window, and the `QUASI_STATIC` verdict the upper bound supports.

    `holds` is taken on `ad_high`, the upper end of the interval at the worst step, and that is the
    decision this instrument exists to take differently from the regime estimator. Reading `Ad`
    licenses treating the run as quasi-static, so it holds when the whole interval is below the
    threshold, fails when the whole interval is above it, and is `None` when the interval straddles
    it. A point estimate below a threshold with an interval crossing it has established nothing, and
    the rule is that unknown is not a pass.
    """

    per_step: tuple[StepAdiabaticity, ...]
    #: The worst step's point estimate, and the ends of its interval.
    ad: float
    ad_low: float
    ad_high: float
    worst_step: int
    parameter: str
    threshold: float
    holds: bool | None
    tau: RelaxationTime
    #: True when every recorded schedule parameter is constant across the window, in which case
    #: `Ad` is exactly zero whatever the relaxation time is.
    flat_schedule: bool
    window: tuple[int, int]

    @property
    def median_ad(self) -> float:
        """The typical step's `Ad`, which is the number the worst step hides.

        Reported because the two differ by two orders of magnitude on a schedule that decays
        linearly to zero, and a reader who sees only the maximum will conclude the run was driven
        hard throughout when it was driven hard once.
        """
        if not self.per_step:
            return float("nan")
        return float(np.median([s.ad for s in self.per_step]))

    def says(self) -> str:
        state = {True: "holds", False: "fails", None: "cannot be decided"}[self.holds]
        if self.flat_schedule:
            return (
                f"every recorded schedule parameter is constant over steps {self.window[0]} to "
                f"{self.window[1] - 1}, so the driver is not moving, Ad is zero whatever "
                f"tau_relax is, and QUASI_STATIC {state}."
            )
        return (
            f"Ad = {self.ad:.4g} [{self.ad_low:.4g}, {self.ad_high:.4g}] at its worst step "
            f"({self.worst_step}, driven by {self.parameter!r}), against a threshold of "
            f"{self.threshold:.4g}: QUASI_STATIC {state} on the upper end of the interval. The "
            f"median step is {self.median_ad:.4g}."
        )

    def render(self) -> str:
        return f"{self.says()}\n    {self.tau.render()}"


def adiabaticity(
    run: Run,
    *,
    window: Window | None = None,
    schedule_parameter: str | None = None,
    relaxation_series: Literal["group_mean", "entropy", "kl_to_previous"] = "group_mean",
    thresholds: RegimeThresholds | None = None,
    floors: RelaxationFloors | None = None,
    instrument: str = "Adiabaticity",
    seed: int = 0,
) -> "AdiabaticityReading | Refusal":
    """`Ad` per step over one window of one run, from the record alone.

    The relaxation time is fitted on the run's own early window, not on the requested one, because
    it should be estimated before the transition and a window chosen for another reason has
    no claim to be early. The driving rate is per step over the requested window.

    Returns a `Refusal` when the record carries no schedule this window can differentiate, and a
    bounded refusal when the relaxation time is unbounded above, because an `Ad` with no upper bound
    licenses nothing and the bound is the honest part of the answer.
    """
    thresholds = thresholds or RegimeThresholds()
    floors = floors or RelaxationFloors()
    indices = run.steps.indices
    if not indices:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.VOID,
            detail=f"run {run.id} records no steps at all, so it has neither a schedule nor a series",
            remedy=(
                "read a record with steps in it. A run with nothing recorded is void rather than "
                "quasi-static, and reporting Ad = 0 for it would be the second of those."
            ),
            statistics={"recorded": 0},
        )
    lo, hi = window if window is not None else (min(indices), max(indices) + 1)
    steps = window_steps(run, (lo, hi))
    if len(steps) < 2:
        return refuse_incomplete(
            instrument,
            field="two consecutive recorded steps",
            subject=f"the window [{lo}, {hi}) of run {run.id}, which holds {len(steps)}, and so",
            remedy=(
                "ask for a window of at least two recorded steps. Ad is built on a derivative of "
                "the schedule and a window of one step has none; widen the window rather than "
                "reading a rate of zero into it."
            ),
            n=len(steps),
        )

    rates = drive_rates(steps, schedule_parameter)
    if not rates:
        named = f" named {schedule_parameter!r}" if schedule_parameter else ""
        return refuse_incomplete(
            instrument,
            field=f"a positive schedule parameter{named} on consecutive steps",
            subject=f"the {len(steps)} steps of window [{lo}, {hi}) in run {run.id}, which carry no",
            remedy=(
                "record the annealing parameter on every step through `reward_lens.tap`, or name "
                "one that this record carries. An empty schedule is not a flat one: a run that "
                "never wrote down what it was annealing cannot be shown to have been annealing it "
                "slowly."
            ),
            n=len(steps),
        )

    flat = all(r.rate == 0.0 for r in rates)
    early_hi = min(indices) + floors.early_steps
    early = window_steps(run, (min(indices), early_hi))
    tau = relaxation_time(
        _relaxation_series(early, relaxation_series),
        name=relaxation_series,
        floors=floors,
        instrument=instrument,
        seed=seed,
    )
    if isinstance(tau, Refusal):
        if flat:
            # The one case where the relaxation time does not matter: an exact zero times anything
            # is zero. Refusing here would refuse the only unambiguously quasi-static run there is.
            tau = RelaxationTime(
                tau=float("nan"),
                tau_low=float("nan"),
                tau_high=float("nan"),
                phi=float("nan"),
                phi_ols=float("nan"),
                phi_ci=(float("nan"), float("nan")),
                bias=float("nan"),
                n=len(early),
                series=relaxation_series,
                axis=STEP_AXIS,
                method="not fitted; the schedule is flat so Ad is zero whatever it is",
                note=tau.detail,
            )
        else:
            return tau

    per_step: list[StepAdiabaticity] = []
    for r in rates:
        point = 0.0 if flat else float(tau.tau * r.rate)
        low = 0.0 if flat else float(tau.tau_low * r.rate)
        high = 0.0 if flat else float(tau.tau_high * r.rate)
        per_step.append(
            StepAdiabaticity(
                from_step=r.from_step,
                to_step=r.to_step,
                rate=r.rate,
                parameter=r.parameter,
                ad=point,
                ad_low=low,
                ad_high=high,
            )
        )
    worst = max(per_step, key=lambda s: (s.ad_high, s.ad))
    limit = thresholds.ad_max
    if worst.ad_high <= limit:
        holds: bool | None = True
    elif worst.ad_low > limit:
        holds = False
    else:
        holds = None

    reading = AdiabaticityReading(
        per_step=tuple(per_step),
        ad=worst.ad,
        ad_low=worst.ad_low,
        ad_high=worst.ad_high,
        worst_step=worst.to_step,
        parameter=worst.parameter,
        threshold=limit,
        holds=holds,
        tau=tau,
        flat_schedule=flat,
        window=(lo, hi),
    )
    if not flat and not math.isfinite(worst.ad_high):
        return bounded_refusal(
            instrument,
            RefusalReason.ABOVE_LOD_BELOW_LOQ,
            detail=(
                f"the early lag-1 coefficient's interval reaches {tau.phi_ci[1]:.4g}, at or above "
                f"one, so the relaxation time has no upper bound and neither does Ad. The schedule "
                f"is moving at up to {worst.rate:.4g} per step in log units, so what is known is "
                f"that Ad is at least {worst.ad_low:.4g}"
            ),
            remedy=(
                "measure tau_relax by perturb-and-hold and pass it to `measure_regime` as "
                "`RegimeInputs.tau_relax`, which is rung 1 of this ladder. A non-stationary early "
                "fit is the case the cheap rung cannot answer, and treating it as fast driving "
                "would be as much a guess as treating it as slow."
            ),
            bound=make_evidence(
                observable=instrument,
                observable_version="1.0",
                subject=SubjectRef(readout=relaxation_series),
                value=reading,
                provenance=Provenance(),
                quantity="run.adiabaticity",
            ),
            ad_low=worst.ad_low,
            phi_ci_high=tau.phi_ci[1],
            rate=worst.rate,
        )
    return reading


# ---------------------------------------------------------------------------
# The handoff to H5, and the disagreement with it
# ---------------------------------------------------------------------------


def regime_inputs(
    reading: AdiabaticityReading,
    *,
    bound: Literal["upper", "point"] = "upper",
    base: RegimeInputs | None = None,
) -> RegimeInputs:
    """The `RegimeInputs` that let `measure_regime` answer `QUASI_STATIC` from this reading.

    `rate/regime.py` is closed and needs no change for this: it already takes a caller-supplied
    `tau_relax` and skips its own fit when one arrives. So the handoff is two passes. Measure the
    regime once to see what the record supports, run this instrument, then measure it again with
    the result::

        ad = adiabaticity(run)
        reading = measure_regime(run, inputs=regime_inputs(ad))

    ``bound`` defaults to `upper` and that default is the argument. The regime estimator compares a
    point estimate against the threshold, and its own point estimate is biased toward licensing;
    handing it the upper end of this interval makes its verdict mean "the assumption is established"
    rather than "the assumption was not contradicted". Passing `point` gives the bias-corrected
    point estimate instead, which is still strictly less licensing than what it fits for itself.

    A flat schedule passes no relaxation time at all, and that is right rather than a gap. When
    every recorded parameter is constant the driver is not moving, so `Ad` is zero whatever the
    relaxation time is, and `_measure_quasi_static` returns True on the rate before it looks for a
    time. This instrument does not fit one in that case either, so there is nothing to hand over
    and `tau_relax=None` is the honest value.

    One thing this cannot carry across, and it is worth naming because it shows up in the reading:
    `RegimeInputs` has no field for where a supplied `tau_relax` came from, so the condition's
    detail will say "tau_relax supplied by the caller" and not which estimator or which end of
    which interval. The full reading is on this object.
    """
    tau = reading.tau.tau_high if bound == "upper" else reading.tau.tau
    if reading.flat_schedule and not math.isfinite(tau):
        return replace(base or RegimeInputs(), tau_relax=None)
    if not math.isfinite(tau):
        raise ValueError(
            f"this reading's tau_relax {bound} end is {tau!r}, which `measure_regime` would "
            f"multiply into an Ad of the same kind. `adiabaticity` returns a bounded refusal "
            f"rather than a reading in that case, so reaching here means one was built by hand."
        )
    return replace(
        base or RegimeInputs(),
        tau_relax=float(tau),
        relaxation_series=reading.tau.series,  # type: ignore[arg-type]
    )


def tau_transfer(reading: AdiabaticityReading) -> Transfer:
    """The two rung-0 relaxation times, differenced, as a chain term.

    The regime estimator and this one read the same series over the same early window and differ only
    in whether the bias of the lag-1 fit is removed, so their difference is not sampling noise: it
    is what the uncorrected estimator costs, measured on this run. `ladder_disagreement` is the
    kernel's one-call form and this is a thin wrapper over it, deliberately, because M11 already
    owns the general case and a second implementation of it is a second thing to keep right.

    That estimator returns nothing at all when its coefficient is at or below zero, and on the
    200-step GRPO record that is what it does. The transfer is then between no number and a number,
    which is reported as a disagreement of the whole of this module's estimate with the note saying
    which side was absent. That is the honest reading: an estimator that declines is not an
    estimator that agrees.
    """
    tau = reading.tau
    cheap = tau_of(tau.phi_ols) if math.isfinite(tau.phi_ols) and tau.phi_ols > 0 else None
    why = (
        ""
        if cheap is not None
        else (
            f" The uncorrected coefficient is {tau.phi_ols:.4g}, at or below zero, so "
            f"rate/regime.py reports no relaxation time at all and QUASI_STATIC comes back "
            f"undetermined; the cheap side of this transfer is an absence rather than a zero."
        )
    )
    return ladder_disagreement(
        0.0 if cheap is None else cheap,
        tau.tau,
        from_level="working_method",
        to_level="reference_method",
        n=tau.n,
        method=(
            "two rung-0 estimators of run.tau_relax on the same early window: the uncorrected "
            "lag-1 least-squares fit of rate/regime.py against the bootstrap bias-corrected fit."
            + why
        ),
    )


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

#: H1 cannot require `QUASI_STATIC`, because it measures it. It can and does require
#: `STATIONARY_GRADER`: the relaxation time is fitted to a scored series over the early window, and
#: if the grader moved during that window the fit measures the grader moving rather than the policy
#: returning. Downgrade rather than refuse, because the fit is still a real fit of the series that
#: was recorded and what it loses outside the envelope is the right to be read as a property of the
#: policy. The worked case for `downgrade` has this shape.
ADIABATICITY_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
    measured_by=MEASURED_BY,
    on_violation="downgrade",
)

_ADIABATICITY_ACCESS: AccessMatrix = {Component.RECORD: Access.RECORD}

#: Rung 1's access. Perturb-and-hold nudges the policy and holds the schedule while the observable
#: returns, so it needs to write to the policy and to stand up an arm of the loop with the schedule
#: pinned. Neither is on the RECORD ladder.
_PERTURB_ACCESS: AccessMatrix = {
    Component.RECORD: Access.RECORD,
    Component.POLICY: Access.MUTATE,
    Component.OPTIMIZER: Access.CONTROL,
}

#: The catalogue gives H1 one baseline, "assume quasi-static, which is what everyone does", and
#: `spec/CATALOGUE.yaml` carries it as two entries because the merge split it at the comma. The
#: second entry here is not that fragment: it is the other reflex, which is to assume the system
#: relaxes in one step, and it is a
#: real number on every run rather than a restatement of the first.
ADIABATICITY_BASELINES = (
    "baseline.assume_quasi_static",
    "baseline.unit_relaxation",
)


class Adiabaticity(BaseObservable):
    """H1 rung 0. `Ad` per step from a record, and the `QUASI_STATIC` verdict its bound supports.

    Reads a record and nothing else, which is what makes it usable in front of the instruments it
    gates: a check that had to spend GPU time to find out whether an assumption held would not get
    run.

    What it cannot do. The relaxation time is fitted rather than measured, and the quantity is
    defined by the measurement it is not doing: perturb the policy, hold the schedule, count
    the steps until the observable returns. A fitted lag-1 time and a perturb-and-hold time agree
    only if the system's memory really is first-order, which nobody has checked on a language
    policy, so this rung reports a first-order summary of whatever the real dynamics are. That is
    rung 1 and it is registered with its access and its cost so a capability report can price it.

    It also cannot see a driver that is not in the record. `Ad` is built from the schedule the run
    wrote down, so a run annealing something it did not log looks flat, and the instrument refuses
    on an absent schedule rather than reading zero into it.
    """

    name = "Adiabaticity"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to: str | None = "H1"
    deviations: tuple[str, ...] = (
        "tau_relax is defined by perturb-and-hold. This is rung 0, the early lag-1 fit, "
        "which is the other named route to it and which is weaker in every way but cost.",
        "Ad is normally read against a threshold as a point. The verdict here is taken on the "
        "upper end of the interval on tau_relax, because the condition licenses an assumption and "
        "a point below a threshold with an interval crossing it has not established one.",
    )

    quantity = "run.adiabaticity"
    requires: AccessMatrix = _ADIABATICITY_ACCESS
    substrates = frozenset(Substrate)
    #: Not PRE_RUN: there is no schedule derivative before the run. Not DEPLOYED: only the artifact
    #: survives there and this reads the process that produced it.
    phases = frozenset({Phase.IN_RUN, Phase.POST_RUN})
    envelope = ADIABATICITY_ENVELOPE
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = ADIABATICITY_BASELINES
    rung = 0

    def __init__(
        self,
        run: Run,
        *,
        window: Window | None = None,
        schedule_parameter: str | None = None,
        relaxation_series: Literal["group_mean", "entropy", "kl_to_previous"] = "group_mean",
        thresholds: RegimeThresholds | None = None,
        floors: RelaxationFloors | None = None,
        seed: int = 0,
    ) -> None:
        self.run = run
        self.window = window
        self.schedule_parameter = schedule_parameter
        self.relaxation_series = relaxation_series
        self.thresholds = thresholds or RegimeThresholds()
        self.floors = floors or RelaxationFloors()
        self.seed = seed
        self._computed: AdiabaticityReading | None = None

    def compute(self) -> "AdiabaticityReading | Refusal":
        return adiabaticity(
            self.run,
            window=self.window,
            schedule_parameter=self.schedule_parameter,
            relaxation_series=self.relaxation_series,
            thresholds=self.thresholds,
            floors=self.floors,
            instrument=self.name,
            seed=self.seed,
        )

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
                f"{self.name}.measure was called on a window that declines to produce Evidence: "
                f"{out.reason.name}. Call `estimate`, which returns the refusal as a value with "
                f"its remedy."
            )
        return ctx.emit(out, baselines=self.baseline_scores(out))

    def baseline_scores(self, reading: AdiabaticityReading) -> dict[str, float]:
        """What the two dumb comparators say, scored in the reading's own unit.

        `baseline.assume_quasi_static` is what everybody does, which is to proceed as though `Ad`
        were zero. Its score is exactly zero and the whole of `ad_high` is the distance between it
        and the measurement, which is the number that says whether this instrument was worth
        running on this record.

        `baseline.unit_relaxation` is the other reflex: assume the system relaxes in one optimizer
        step, so `Ad` is the driving rate itself. It needs no fit at all, and comparing the two is
        what makes the relaxation time's contribution visible rather than buried in a product.
        """
        rate = max((s.rate for s in reading.per_step), default=0.0)
        return {
            "baseline.assume_quasi_static": 0.0,
            "baseline.unit_relaxation": float(rate),
        }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _register() -> None:
    """Two rungs each for `run.tau_relax` and `run.adiabaticity`. One is built and one is priced.

    Registering rung 1 with `run=None` is what makes `reward-lens capabilities` able to say "the
    better answer exists, here is what it costs and what access it needs" instead of silently
    offering only the cheap one. The whole argument for separating quantities from
    estimators is that a rung nobody has built is a research target rather than a gap.
    """
    register_estimator(
        EstimatorEntry(
            quantity="run.tau_relax",
            impl="run.tau_relax.ar1_bias_corrected",
            requires=_ADIABATICITY_ACCESS,
            envelope=ADIABATICITY_ENVELOPE,
            rung=0,
            bias=BiasStatement(
                direction="unknown",
                why=(
                    "the first-order bias of the lag-1 fit is removed by bootstrap, so what is "
                    "left is the model error: a system whose memory is not first-order is being "
                    "summarised by one time constant and there is no general direction to that. "
                    "The uncorrected estimator in rate/regime.py is biased downward, which is "
                    "toward licensing, and `tau_transfer` publishes the difference."
                ),
            ),
            cost=CostModel(note="one pass over the early window plus 400 simulated paths; free"),
            phases=frozenset({Phase.IN_RUN, Phase.POST_RUN}),
            run=None,
        )
    )
    register_estimator(
        EstimatorEntry(
            quantity="run.tau_relax",
            impl="run.tau_relax.perturb_and_hold",
            requires=_PERTURB_ACCESS,
            envelope=ADIABATICITY_ENVELOPE,
            rung=1,
            bias=BiasStatement(
                direction="unknown",
                why=(
                    "not measured. This is the definition of the quantity rather than an "
                    "estimator of it, so its bias is the bias of the protocol: how large a nudge "
                    "counts as small, and whether the return is exponential at all. Both are "
                    "answered by running it at three magnitudes."
                ),
            ),
            cost=CostModel(
                note="one held arm per perturbation, run until the observable returns, at three "
                "magnitudes to check that the response is linear. Not built."
            ),
            phases=frozenset({Phase.IN_RUN}),
            run=None,
        )
    )
    register_estimator(
        EstimatorEntry(
            quantity="run.adiabaticity",
            impl="run.adiabaticity.record",
            requires=_ADIABATICITY_ACCESS,
            envelope=ADIABATICITY_ENVELOPE,
            rung=0,
            bias=BiasStatement(
                direction="downward",
                why=(
                    "Ad is the relaxation time times the driving rate, the rate is read exactly "
                    "off the recorded schedule, and the relaxation time at this rung is a "
                    "first-order fit. A run annealing a parameter it did not log contributes "
                    "nothing to the rate, so every unrecorded driver makes Ad smaller than the "
                    "truth. The verdict is taken on the upper end of the interval to keep that "
                    "from turning into a licence."
                ),
            ),
            cost=CostModel(note="one pass over the window's steps; no grader calls, no GPU"),
            phases=frozenset({Phase.IN_RUN, Phase.POST_RUN}),
            run=None,
        )
    )
    register_estimator(
        EstimatorEntry(
            quantity="run.adiabaticity",
            impl="run.adiabaticity.perturb_and_hold",
            requires=_PERTURB_ACCESS,
            envelope=ADIABATICITY_ENVELOPE,
            rung=1,
            bias=BiasStatement(
                direction="unknown",
                why="inherits the rung-1 relaxation time's; the driving rate is exact either way.",
            ),
            cost=CostModel(
                note="the rung-1 relaxation time plus one pass over the schedule. Not built."
            ),
            phases=frozenset({Phase.IN_RUN}),
            run=None,
        )
    )


_register()


__all__ = [
    "ADIABATICITY_BASELINES",
    "ADIABATICITY_ENVELOPE",
    "STEP_AXIS",
    "Adiabaticity",
    "AdiabaticityReading",
    "DriveRate",
    "RelaxationFloors",
    "RelaxationTime",
    "StepAdiabaticity",
    "adiabaticity",
    "adiabaticity_number",
    "drive_rates",
    "regime_inputs",
    "relaxation_time",
    "tau_of",
    "tau_transfer",
]

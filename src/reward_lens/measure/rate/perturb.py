"""H1 rung 1: `tau_relax` by perturb-and-hold, which is the measurement the quantity is defined by.

The relaxation time is defined as a protocol and not as a fit: nudge the policy, hold the
schedule where it is, and count the steps the observable takes to come back. Rung 0, in
`adiabaticity.py`, reads a lag-1 coefficient off a series that was never perturbed and is free. This
is the one that costs a held arm per episode, and it is the one the definition names.

**Holding the schedule is the whole protocol and it is the part that is easy to skip.** If `lambda`
keeps annealing during the recovery, the observable is chasing a moving equilibrium and what comes
back is a mixture of the relaxation time and the drift. `held_lambda_span` is on every episode and
the estimator refuses when it is not small, because there is no way to unmix them afterwards.

**The dangerous failure is a hold that ends before the recovery does**, and what it does to the fit
was measured here rather than assumed. Planted single exponentials at a time constant of 20 steps,
300 replicates per cell, three-parameter fit with the asymptote free:

| hold / tau | median fitted tau at amplitude/noise 3 | at 5 | at 10 |
|---|---|---|---|
| 0.5 | 3.99 | 11.77 | 18.81 |
| 1.0 | 15.98 | 16.51 | 20.85 |
| 2.0 | 22.94 | 20.51 | 20.21 |
| 3.0 | 21.15 | 20.29 | 20.06 |
| 5.0 | 20.16 | 19.70 | 20.09 |

Two things fall out and neither is the simple story. **At three time constants of hold the estimator
is unbiased at every signal-to-noise ratio tested**, which is where `min_hold_multiple` is set and it
is set there because of this table. And below that the estimator does not fail in one direction: the
median runs **down**, badly at low signal-to-noise, while the geometric mean runs **up**, because a
minority of fits run away toward an unbounded time constant on a window over which an exponential
looks like a line. So a short hold produces a number that is unstable rather than merely wrong, and
its median error is toward a fast system, which is toward licensing the equilibrium assumption.

The refusal for that case therefore does **not** report the fit as a bound, because the fit is not
one. It reports `hold_steps / min_hold_multiple` as a lower bound, which comes from the fit-free
observation that the observable had not returned to its pre-perturbation level by the end of a hold
of that length.

**Three magnitudes, because the protocol assumes a linear response and nobody has checked it.** A
relaxation time is a property of the system only if the return rate does not depend on how hard it
was pushed. `magnitude_dependence` is the Kendall correlation between the perturbation size and the
fitted time constant, and it is read against **its p-value and not against a threshold on the
statistic**. At three episodes Kendall's tau takes only the values ±1 and ±1/3, a perfectly monotone
ordering arises by chance one time in three, and the smallest two-sided p-value available is 0.333:
**three episodes cannot establish a magnitude dependence at all.** A threshold on the statistic would
have declared a third of linear systems nonlinear, which is what the first version of this module
did before it was run on one. `magnitude_test_powered` says whether the test could have fired, and a
reading where it could not says so rather than reporting a passed check.

This module fits recoveries. It does not perturb anything: perturbing a policy needs `MUTATE` on it
and `CONTROL` on the optimizer, which is the compute this package is gated on, and
`studies/w6_rate/RUNBOOK.md` is the protocol that produces the episodes.

**The registered rung-1 entry for `run.tau_relax` lives in `adiabaticity.py` with `run=None`**, and
this module deliberately does not re-register it: the registry refuses to redefine a live name to a
different value, and reaching across a package boundary to mutate another module's entry at import
time is worse than an entry that is one field out of date. `PERTURB_AND_HOLD_IMPL` names the entry
this implements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import kendalltau

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import make_evidence, register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.provenance import Provenance
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
    SubjectRef,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context, run
from reward_lens.measure.rate.adiabaticity import STEP_AXIS
from reward_lens.measure.rate.regime import MEASURED_BY

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence

#: The registry id this module implements. The entry is registered in `adiabaticity.py` at rung 1
#: with `run=None`; see the module docstring for why it is not re-registered here.
PERTURB_AND_HOLD_IMPL = "run.tau_relax.perturb_and_hold"


@dataclass(frozen=True)
class PerturbCriteria:
    """Every number a verdict here is compared against, in one place, with where it came from."""

    #: Steps of hold needed per fitted time constant before the fit is quantitative.
    #: **Chosen: 3.0.** Three time constants is 95 percent of the recovery, which is the point past
    #: which the exponential's curvature is in the data rather than being extrapolated from its
    #: first bend. At one time constant a fit is essentially unidentified and biased short.
    min_hold_multiple: float = 3.0

    #: Points needed inside one hold. **Chosen: 8.** Three parameters, and a floor below which the
    #: fit is interpolation.
    min_hold_points: int = 8

    #: Episodes needed. **Chosen: 3**, which is the smallest number that can show a magnitude
    #: dependence at all and matches the catalogue's own "at three magnitudes" costing.
    min_episodes: int = 3

    #: The perturbation has to displace the observable by at least this many pre-perturbation
    #: standard deviations. **Chosen: 3.0.** Below it the decay being fitted is the noise.
    min_displacement_sds: float = 3.0

    #: Fraction of its own range `lambda` may move during a hold before the hold is not a hold.
    #: **Chosen: 0.02.**
    max_lambda_drift: float = 0.02

    #: Level the magnitude-dependence test is read at. **Chosen: 0.05.** It is a p-value and not a
    #: threshold on Kendall's tau, because at three episodes the statistic reaches 1.0 by chance one
    #: time in three and a threshold on it would call a third of linear systems nonlinear.
    magnitude_alpha: float = 0.05

    #: How many pre-perturbation standard deviations the tail of the hold may sit away from the
    #: pre-perturbation level and still count as returned. **Chosen: 1.0.** Fit-free on purpose:
    #: whether the observable came back is the one thing about a truncated episode that can be
    #: established without the fit that truncation destabilises.
    returned_within_sds: float = 1.0

    #: Bootstrap replicates for the pooled interval. **Chosen: 2000**; the resampling is over
    #: episodes and costs nothing.
    n_boot: int = 2000

    ci_level: float = 0.95


# ---------------------------------------------------------------------------
# The episode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recovery:
    """One perturb-and-hold episode: what the observable did after the nudge, with the hold's terms.

    `values` is the observable at each step of the hold, starting at the first step after the
    perturbation. `pre_level` and `pre_sd` are the level and the spread over the steps immediately
    before it, and they are what the return is measured against: a recovery has to come back to
    something, and the pre-perturbation level is the only defensible target.

    `held_lambda_span` is the fraction of its own range the schedule parameter moved during the
    hold. Zero is a hold; anything much above it is a slow anneal with a bump in it.
    """

    label: str
    values: np.ndarray
    pre_level: float
    pre_sd: float
    magnitude: float
    held_lambda_span: float = 0.0

    @classmethod
    def from_series(
        cls,
        label: str,
        values: Sequence[float] | np.ndarray,
        *,
        pre: Sequence[float] | np.ndarray,
        held_lambda_span: float = 0.0,
    ) -> "Recovery":
        """Build an episode from the hold's observations and the steps that preceded the nudge."""
        v = np.asarray([float(x) for x in values], dtype=np.float64).ravel()
        v = v[np.isfinite(v)]
        p = np.asarray([float(x) for x in pre], dtype=np.float64).ravel()
        p = p[np.isfinite(p)]
        level = float(np.mean(p)) if p.size else float("nan")
        sd = float(np.std(p, ddof=1)) if p.size > 1 else float("nan")
        magnitude = float(v[0] - level) if v.size and math.isfinite(level) else float("nan")
        return cls(
            label=label,
            values=v,
            pre_level=level,
            pre_sd=sd,
            magnitude=magnitude,
            held_lambda_span=float(held_lambda_span),
        )


@register_payload
@dataclass(frozen=True)
class EpisodeFit:
    """One episode's fitted time constant, with what the hold could and could not support.

    `returned_to_level` is fit-free and it is the field that decides whether this episode measured
    anything: the tail of the hold sat within `returned_within_sds` pre-perturbation standard
    deviations of the pre-perturbation level. `hold_in_taus` is fit-based and is the second check.
    They are separate because the fit is the thing truncation destabilises, so an adequacy check
    computed only from the fit would be the least trustworthy number deciding the most.
    """

    label: str
    tau: float
    asymptote: float
    amplitude: float
    hold_steps: int
    hold_in_taus: float
    residual_rmse: float
    returned_to_level: bool
    tail_offset_sds: float
    converged: bool
    min_hold_multiple: float = 3.0
    note: str = ""

    @property
    def hold_adequate(self) -> bool:
        """Whether the hold covered enough time constants for the fit to be identified."""
        return math.isfinite(self.hold_in_taus) and self.hold_in_taus >= self.min_hold_multiple

    @property
    def quantitative(self) -> bool:
        """Converged, positive, came back, and held long enough. All four, because each can fail."""
        return bool(
            self.converged
            and math.isfinite(self.tau)
            and self.tau > 0
            and self.returned_to_level
            and self.hold_adequate
        )

    def render(self) -> str:
        if not math.isfinite(self.tau):
            return f"{self.label}: no time constant fitted ({self.note})"
        if not self.returned_to_level:
            tail = (
                f" (the hold ended with the observable still {self.tail_offset_sds:.1f} "
                f"pre-perturbation standard deviations from its level)"
            )
        elif not self.hold_adequate:
            tail = " (the hold covered fewer than three time constants, so the fit is unstable)"
        else:
            tail = ""
        return (
            f"{self.label}: tau = {self.tau:.4g} steps from a hold of {self.hold_steps} steps, "
            f"which is {self.hold_in_taus:.2f} time constants{tail}"
        )


def _fit_exponential(y: np.ndarray) -> tuple[float, float, float, bool, float]:
    """Fit ``y(s) = asymptote + amplitude * exp(-s / tau)``. Returns tau, asymptote, amp, ok, rmse.

    Three parameters by nonlinear least squares from a data-derived start. The asymptote is fitted
    rather than pinned to the pre-perturbation level on purpose: a system that returns to a
    different level has not relaxed, it has moved, and fitting the asymptote is what makes that
    visible instead of forcing it into the time constant.
    """
    n = y.size
    s = np.arange(n, dtype=np.float64)
    a0 = float(np.mean(y[max(1, n - max(3, n // 4)) :]))
    amp0 = float(y[0] - a0)
    tau0 = max(1.0, n / 4.0)
    if amp0 == 0.0:
        amp0 = float(np.std(y)) or 1.0

    def residual(p: np.ndarray) -> np.ndarray:
        return p[0] + p[1] * np.exp(-s / max(p[2], 1e-9)) - y

    try:
        out = least_squares(
            residual,
            np.array([a0, amp0, tau0]),
            bounds=(
                np.array([-np.inf, -np.inf, 1e-6]),
                np.array([np.inf, np.inf, 10.0 * max(n, 2)]),
            ),
            max_nfev=2000,
        )
    except (ValueError, RuntimeError):
        return float("nan"), float("nan"), float("nan"), False, float("nan")
    asym, amp, tau = (float(v) for v in out.x)
    rmse = float(np.sqrt(np.mean(residual(out.x) ** 2)))
    return tau, asym, amp, bool(out.success), rmse


def fit_recovery(episode: Recovery, *, criteria: PerturbCriteria | None = None) -> EpisodeFit:
    """One episode's time constant, plus the fit-free check on whether it came back at all."""
    criteria = criteria or PerturbCriteria()
    y = episode.values
    n = int(y.size)

    # Fit-free first, because it is the check that survives a bad fit. The tail is the last
    # quarter of the hold, which is long enough to average the noise down and short enough that a
    # slow return has not had time to hide in it.
    tail = y[max(1, n - max(2, n // 4)) :] if n else y
    sd = episode.pre_sd if math.isfinite(episode.pre_sd) and episode.pre_sd > 0 else float("nan")
    offset = (
        float(abs(float(np.mean(tail)) - episode.pre_level) / sd)
        if tail.size and math.isfinite(sd) and math.isfinite(episode.pre_level)
        else float("nan")
    )
    came_back = bool(math.isfinite(offset) and offset <= criteria.returned_within_sds)

    if n < criteria.min_hold_points:
        return EpisodeFit(
            label=episode.label,
            tau=float("nan"),
            asymptote=float("nan"),
            amplitude=float("nan"),
            hold_steps=n,
            hold_in_taus=float("nan"),
            residual_rmse=float("nan"),
            returned_to_level=came_back,
            tail_offset_sds=offset,
            converged=False,
            min_hold_multiple=criteria.min_hold_multiple,
            note=(
                f"the hold is {n} steps and a three-parameter exponential needs at least "
                f"{criteria.min_hold_points}"
            ),
        )

    tau, asym, amp, ok, rmse = _fit_exponential(y)
    hold_in_taus = float(n / tau) if math.isfinite(tau) and tau > 0 else float("nan")
    note = ""
    if not came_back:
        note = (
            f"the observable was still {offset:.1f} pre-perturbation standard deviations from its "
            f"level at the end of the hold, so this episode establishes a lower bound on the "
            f"return time and not a value"
        )
    elif math.isfinite(hold_in_taus) and hold_in_taus < criteria.min_hold_multiple:
        note = (
            f"the hold covers {hold_in_taus:.2f} time constants against a floor of "
            f"{criteria.min_hold_multiple:.2g}, below which this fit was measured to be unstable "
            f"rather than merely imprecise"
        )
    return EpisodeFit(
        label=episode.label,
        tau=float(tau),
        asymptote=float(asym),
        amplitude=float(amp),
        hold_steps=n,
        hold_in_taus=hold_in_taus,
        residual_rmse=float(rmse),
        returned_to_level=came_back,
        tail_offset_sds=offset,
        converged=bool(ok),
        min_hold_multiple=criteria.min_hold_multiple,
        note=note,
    )


# ---------------------------------------------------------------------------
# The pooled reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class PerturbRelaxation:
    """`tau_relax` at rung 1: the protocol's own answer, pooled over episodes.

    `tau` is the geometric mean over episodes, because a time constant is positive and the thing
    that is symmetric about it is its logarithm; an arithmetic mean of 2 and 50 steps is 26 and its
    geometric mean is 10, and 10 is the one that behaves like a time constant.

    `magnitude_dependence` is the check that decides whether one number was the right thing to
    report, and `magnitude_test_powered` is the check on that check. A relaxation time is a property
    of the system only under a linear response, and at three episodes the test for it cannot reach
    significance at any level below 0.333, so a linear-response call from three episodes is an
    absence of evidence and the reading says which it is.

    **The interval under-covers at small episode counts and the direction matters.** It is a
    percentile bootstrap over episodes, so it estimates the between-episode spread from the episodes
    themselves, and with five of them that estimate is short. Measured on planted single
    exponentials at a time constant of 8 steps, 100 replicate protocols per cell: a nominal 95
    percent interval covers the planted value **84 percent** of the time at five episodes, 90
    percent at eight and 93 percent at twelve, at mean relative widths of 0.107, 0.090 and 0.078.
    An interval that is too narrow makes `Ad` look better determined than it is, and a
    better-determined `Ad` below its threshold is easier to license on, which is the same direction
    of concern the rung-0 estimator already carries. The fix is more episodes rather than a wider
    nominal level, and eight episodes cost 60 percent more held steps than five.
    """

    tau: float
    tau_low: float
    tau_high: float
    episodes: tuple[EpisodeFit, ...]
    magnitude_dependence: float
    magnitude_p: float
    magnitude_test_powered: bool
    linear_response: bool
    n_episodes: int
    n_quantitative: int
    axis: str = STEP_AXIS
    method: str = "perturb-and-hold, three-parameter exponential per episode, pooled in log"
    note: str = ""

    @property
    def identified(self) -> bool:
        return math.isfinite(self.tau) and self.tau > 0

    def render(self) -> str:
        if not self.linear_response:
            tail = (
                f" The fitted time constant tracks the perturbation magnitude at Kendall tau "
                f"{self.magnitude_dependence:+.3f}, p = {self.magnitude_p:.3g} over "
                f"{self.n_episodes} episodes, so the response is not linear over the range probed "
                f"and this is a summary of a nonlinear return rather than a measurement of a rate."
            )
        elif not self.magnitude_test_powered:
            tail = (
                f" The linear-response check could not have fired: with {self.n_episodes} episodes "
                f"the smallest p-value Kendall's tau can take is {self.magnitude_p:.3g}, so the "
                f"linearity this number assumes is untested rather than established. Four episodes "
                f"reach 0.083 and five reach 0.017."
            )
        else:
            tail = (
                f" The fitted time constant does not track the perturbation magnitude (Kendall tau "
                f"{self.magnitude_dependence:+.3f}, p = {self.magnitude_p:.3g}), so the response is "
                f"linear over the range probed."
            )
        return (
            f"tau_relax = {self.tau:.4g} steps [{self.tau_low:.4g}, {self.tau_high:.4g}] by "
            f"perturb-and-hold over {self.n_quantitative} of {self.n_episodes} episodes whose hold "
            f"outlasted the recovery.{tail}"
        )


def _kendall_p_floor(n: int) -> float:
    """The smallest two-sided p-value Kendall's tau can take at `n` observations.

    A perfectly monotone ordering is one of `n!` equally likely permutations under the null, and the
    two-sided p-value of the extreme is `2 / n!`. So three episodes floor at 0.333, four at 0.083,
    five at 0.017 and six at 0.003, which is the whole reason `magnitude_test_powered` exists: a
    three-episode protocol cannot establish a linear response at any conventional level, however the
    numbers come out. Computed rather than tabulated, and checked against `scipy.stats.kendalltau`
    on the perfectly monotone case in the acceptance test.
    """
    if n < 2:
        return float("nan")
    return min(1.0, 2.0 / float(math.factorial(n)))


def relaxation_time_from_hold(
    episodes: Sequence[Recovery],
    *,
    criteria: PerturbCriteria | None = None,
    instrument: str = "PerturbAndHold",
    seed: int = 0,
) -> "PerturbRelaxation | Refusal":
    """`tau_relax` at rung 1, or the reason the episodes cannot support one.

    Five ways this refuses, and each names a defect in the protocol rather than in the system:

    `RECORD_INCOMPLETE` with fewer than `min_episodes` episodes. Three is the floor because it is
    the smallest number that can show a magnitude dependence, and a relaxation time with no linearity
    check attached is a number whose meaning has not been established.

    `ENVELOPE_VIOLATED` when the schedule kept moving during a hold. The recovery then chases a
    moving equilibrium and no fit separates the two afterwards.

    `BELOW_LOD` when the perturbation did not displace the observable past the pre-perturbation
    noise. There is a decay to fit and it is the noise's.

    `ABOVE_LOD_BELOW_LOQ`, carrying the fitted value as a **lower bound**, when every hold ended
    before its own recovery did. This is the important one: a truncated exponential fits a shorter
    time constant well, so this case produces a plausible number and returning it would be biased
    toward calling the run quasi-static. The bound is honest and the point estimate is not.

    `RECORD_INCOMPLETE` when no episode's fit converged at all.
    """
    criteria = criteria or PerturbCriteria()
    n = len(episodes)
    if n < criteria.min_episodes:
        return refuse_incomplete(
            instrument,
            field=f"at least {criteria.min_episodes} perturb-and-hold episodes",
            subject=f"this protocol run ({n} supplied)",
            remedy=(
                f"run {criteria.min_episodes} episodes at three perturbation magnitudes spanning a "
                f"factor of about four. Fewer than that gives a time constant with no way to check "
                f"that it is a property of the system rather than of how hard it was pushed."
            ),
            n=n,
            floor=criteria.min_episodes,
        )

    drifted = [e for e in episodes if e.held_lambda_span > criteria.max_lambda_drift]
    if drifted:
        worst = max(drifted, key=lambda e: e.held_lambda_span)
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ENVELOPE_VIOLATED,
            detail=(
                f"{len(drifted)} of {n} holds let lambda keep moving; the worst, {worst.label!r}, "
                f"moved {worst.held_lambda_span:.3f} of its range against a ceiling of "
                f"{criteria.max_lambda_drift:.3f}"
            ),
            remedy=(
                "freeze the schedule for the whole hold: pin every annealed coefficient at its "
                "value at the moment of the perturbation and release it only after the observable "
                "has returned. A recovery measured while lambda anneals is the relaxation time and "
                "the drift added together, and nothing downstream can separate them."
            ),
            statistics={
                "n_drifted": len(drifted),
                "worst_span": worst.held_lambda_span,
                "ceiling": criteria.max_lambda_drift,
            },
        )

    too_small = [
        e
        for e in episodes
        if math.isfinite(e.pre_sd)
        and e.pre_sd > 0
        and abs(e.magnitude) < criteria.min_displacement_sds * e.pre_sd
    ]
    if len(too_small) == n:
        worst = max(episodes, key=lambda e: abs(e.magnitude) / (e.pre_sd or 1.0))
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.BELOW_LOD,
            detail=(
                f"every perturbation is inside the pre-perturbation noise; the largest, "
                f"{worst.label!r}, displaced the observable by "
                f"{abs(worst.magnitude) / (worst.pre_sd or float('nan')):.2f} standard deviations "
                f"against a floor of {criteria.min_displacement_sds:.2g}"
            ),
            remedy=(
                "increase the nudge until the observable moves at least three pre-perturbation "
                "standard deviations, or average more rollouts per step so the pre-perturbation "
                "spread is smaller. What is being fitted at this displacement is the decay of the "
                "sampling noise."
            ),
            statistics={
                "n_episodes": n,
                "floor_sds": criteria.min_displacement_sds,
            },
        )

    fits = tuple(fit_recovery(e, criteria=criteria) for e in episodes)
    usable = [f for f in fits if f.quantitative]
    converged = [f for f in fits if f.converged and math.isfinite(f.tau) and f.tau > 0]

    if not converged:
        return refuse_incomplete(
            instrument,
            field="an exponential recovery in any episode",
            subject=f"{n} episodes, none of whose fits converged, and so",
            remedy=(
                "check that the observable was recorded during the hold and that the nudge moved "
                "it in a single direction. A recovery that is not monotone toward its own level is "
                "not a relaxation and no time constant describes it."
            ),
            n=n,
        )

    ids = {id(f) for f in converged}
    mags = np.array(
        [abs(e.magnitude) for e, f in zip(episodes, fits) if id(f) in ids], dtype=np.float64
    )
    taus_all = np.array([f.tau for f in converged], dtype=np.float64)
    if mags.size >= 3 and float(np.ptp(mags)) > 0:
        result = kendalltau(mags, taus_all)
        dependence, p_dep = float(result.statistic), float(result.pvalue)
    else:
        dependence, p_dep = float("nan"), float("nan")
    # The smallest two-sided p-value Kendall's tau can reach at this many episodes. Below the level
    # the test is read at, the check cannot fire and a passed check would be an absence of data.
    p_floor = _kendall_p_floor(int(mags.size))
    powered = bool(math.isfinite(p_floor) and p_floor <= criteria.magnitude_alpha)
    linear = not (math.isfinite(p_dep) and p_dep <= criteria.magnitude_alpha)

    if not usable:
        # No episode both came back and held long enough. The fit is not a bound in either
        # direction here: measured on planted exponentials, a hold under three time constants gives
        # a median that runs low and a geometric mean that runs high. What is established is
        # fit-free, from the longest hold that did not see a return: the return took longer than
        # that hold, so the time constant exceeds the hold divided by the three-time-constant
        # convention for having returned.
        longest = max(f.hold_steps for f in fits)
        lower = float(longest) / criteria.min_hold_multiple
        worst = max(converged, key=lambda f: f.hold_in_taus if math.isfinite(f.hold_in_taus) else 0)
        bound = make_evidence(
            observable=instrument,
            observable_version="1.0",
            subject=SubjectRef(extra={"protocol": "perturb-and-hold"}),
            value=lower,
            gauge=GaugeStatus.INVARIANT,
            provenance=Provenance(),
            quantity="run.tau_relax",
        )
        return bounded_refusal(
            instrument,
            RefusalReason.ABOVE_LOD_BELOW_LOQ,
            detail=(
                f"none of {n} holds both returned to its pre-perturbation level and lasted "
                f"{criteria.min_hold_multiple:.2g} fitted time constants; the best, "
                f"{worst.label!r}, covered {worst.hold_in_taus:.2f}. Below that the fit was "
                f"measured to be unstable rather than biased in a known direction, so the fitted "
                f"value is not reported. What is established without a fit is that the observable "
                f"had not returned within {longest} steps, so tau_relax is at least "
                f"{lower:.4g} steps"
            ),
            remedy=(
                f"end the hold on the observable returning to within one pre-perturbation standard "
                f"deviation of its level, not on a step count. If the hold has to be fixed in "
                f"advance, triple it to {3 * longest} steps and re-run; the bound says the return "
                f"takes longer than {longest} steps and says nothing about how much longer, so a "
                f"small increment is as likely to truncate again. Truncation's median error is "
                f"toward a short relaxation time, a short relaxation time makes Ad small, and a "
                f"small Ad licenses the equilibrium assumption, so this is the direction of error "
                f"that costs something."
            ),
            bound=bound,
            n_episodes=n,
            lower_bound=lower,
            longest_hold=longest,
            hold_in_taus=[f.hold_in_taus for f in converged],
        )

    taus = np.array([f.tau for f in usable], dtype=np.float64)
    logs = np.log(taus)
    point = float(np.exp(np.mean(logs)))
    if taus.size >= 2:
        rng = np.random.default_rng(seed)
        draws = np.exp(
            np.mean(rng.choice(logs, size=(criteria.n_boot, logs.size), replace=True), axis=1)
        )
        lo = float(np.quantile(draws, (1.0 - criteria.ci_level) / 2.0))
        hi = float(np.quantile(draws, 1.0 - (1.0 - criteria.ci_level) / 2.0))
        note = ""
    else:
        lo, hi = point, float("inf")
        note = (
            "one episode outlasted its recovery, so there is no between-episode spread and the "
            "upper end of the interval is unbounded"
        )

    return PerturbRelaxation(
        tau=point,
        tau_low=lo,
        tau_high=hi,
        episodes=fits,
        magnitude_dependence=dependence,
        magnitude_p=p_dep if math.isfinite(p_dep) else p_floor,
        magnitude_test_powered=powered,
        linear_response=linear,
        n_episodes=n,
        n_quantitative=len(usable),
        note=note,
    )


def rung_transfer(rung0_tau: float, rung1: PerturbRelaxation) -> Transfer:
    """The free rung against the protocol rung, as a chain term.

    This is what the estimator ladder is for and it is the case M11 was written about: one quantity,
    two rungs, the same run, and the disagreement published rather than reconciled. Rung 0's fit
    assumes the memory is first-order and rung 1 does not, so their difference is the cost of that
    assumption measured on this system instead of argued about.
    """
    return ladder_disagreement(
        float(rung0_tau),
        float(rung1.tau),
        from_level="working_method",
        to_level="reference_method",
        n=rung1.n_quantitative,
        method=(
            "two rungs of run.tau_relax on one run: the bias-corrected early lag-1 fit of "
            "rate/adiabaticity.py against the perturb-and-hold protocol. The cheap "
            "rung summarises whatever the real memory is with one first-order time constant; the "
            "expensive one measures the return directly and needs no such assumption."
        ),
    )


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

#: Same envelope as rung 0, and for the same reason: this cannot require `QUASI_STATIC`, because it
#: measures the quantity that decides it. `STATIONARY_GRADER` is required and refused on rather than
#: downgraded, because a grader that moved during a hold moved the target the recovery is measured
#: against, and unlike the passive rung-0 fit there is no weaker reading left to downgrade to.
PERTURB_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
    measured_by=MEASURED_BY,
    on_violation="refuse",
)

#: Rung 1's access, as `adiabaticity.py` already registered it: write to the policy to perturb it,
#: control the optimizer to pin the schedule, and read the record back.
_PERTURB_ACCESS: AccessMatrix = {
    Component.RECORD: Access.RECORD,
    Component.POLICY: Access.MUTATE,
    Component.OPTIMIZER: Access.CONTROL,
}

#: H1's catalogue baseline is "assume quasi-static, which is what everyone does". At this rung the
#: two comparators are the reflex and the free rung: a reader deciding whether to buy the held arms
#: wants to know what the free estimator would have said on the same run.
PERTURB_BASELINES = (
    "baseline.assume_quasi_static",
    "baseline.unit_relaxation",
)


class PerturbAndHold(BaseObservable):
    """H1 rung 1. The relaxation time from held recoveries, which is the quantity's definition.

    What it cannot do. It measures the return of one observable, so it reports the relaxation time
    of whatever channel was watched; a policy with a fast reward and a slow internal representation
    has two relaxation times and this instrument returns the one it was pointed at. It assumes the
    return is exponential, which is the same first-order assumption rung 0 makes, and it can at
    least see the assumption failing: a non-exponential return shows up as a residual root mean
    square well above the pre-perturbation noise, which is on every episode fit. And it needs the
    schedule genuinely held, which is a property of the protocol rather than of the analysis, so
    `held_lambda_span` has to be recorded honestly by whoever ran it.
    """

    name = "PerturbAndHold"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to: str | None = "H1"
    deviations: tuple[str, ...] = (
        "the definition says 'count the steps the observable takes to return'. A count needs a "
        "threshold for having returned; this fits a three-parameter exponential and reports its "
        "time constant, which uses the whole recovery rather than the first crossing of an "
        "arbitrary band.",
        "the pooled value is a geometric rather than an arithmetic mean over episodes. Nothing "
        "fixes a pooling convention, and an arithmetic mean of time constants is "
        "dominated by the longest episode.",
    )

    quantity = "run.tau_relax"
    requires: AccessMatrix = _PERTURB_ACCESS
    substrates = frozenset(Substrate)
    #: IN_RUN only. Perturb-and-hold is something done to a live loop; a finished artifact cannot
    #: be nudged and held.
    phases = frozenset({Phase.IN_RUN})
    envelope = PERTURB_ENVELOPE
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = PERTURB_BASELINES
    rung = 1

    def __init__(
        self,
        episodes: Sequence[Recovery],
        *,
        criteria: PerturbCriteria | None = None,
        drive_rate: float | None = None,
        seed: int = 0,
    ) -> None:
        self.episodes = tuple(episodes)
        self.criteria = criteria or PerturbCriteria()
        self.drive_rate = drive_rate
        self.seed = seed
        self._computed: PerturbRelaxation | None = None

    def compute(self) -> "PerturbRelaxation | Refusal":
        return relaxation_time_from_hold(
            self.episodes, criteria=self.criteria, instrument=self.name, seed=self.seed
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
                f"{self.name}.measure was called on episodes that decline to produce Evidence: "
                f"{out.reason.name}. Call `estimate`, which returns the refusal as a value with "
                f"its remedy."
            )
        return ctx.emit(out, baselines=self.baseline_scores(out))

    def baseline_scores(self, reading: PerturbRelaxation) -> dict[str, float]:
        """The two reflexes, scored in the adiabaticity number this time constant feeds.

        Both are scored as `Ad` rather than as a time, because the time constant on its own is not
        a thing anybody assumes: what people assume is that the run is quasi-static, which is `Ad`
        of zero, or that the system relaxes in one step, which is `Ad` equal to the driving rate.
        With no driving rate supplied there is nothing to multiply and both come back as zero with
        the reading's own value beside them.
        """
        rate = float(self.drive_rate or 0.0)
        return {
            "baseline.assume_quasi_static": 0.0,
            "baseline.unit_relaxation": rate,
        }


__all__ = [
    "PERTURB_AND_HOLD_IMPL",
    "PERTURB_BASELINES",
    "PERTURB_ENVELOPE",
    "EpisodeFit",
    "PerturbAndHold",
    "PerturbCriteria",
    "PerturbRelaxation",
    "Recovery",
    "fit_recovery",
    "relaxation_time_from_hold",
    "rung_transfer",
]

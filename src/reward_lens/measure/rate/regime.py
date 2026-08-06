"""H5, the regime reading: the twelve regime conditions measured from a record.

Access, phase and substrate all fail loudly. Regime fails quietly, and this module is where the
quiet failure is made to speak. Every instrument in the library declares an `EnvelopeSpec` naming
the conditions its estimator needs to be *true* rather than merely available; the reading this
module produces is what those declarations are checked against. That is why it is a mechanism and
not a finding.

**Three states per condition, and the third one is the work.** `holds=True` and `holds=False` are
easy. `holds=None` means the determination could not be made, and getting those right matters more
than getting the easy ones right, because a `False` that means "I could not tell" is the exact
defect this library exists to prevent. Several of the twelve are `None` on most real records today:
`LINEAR_RESPONSE` needs a Lambda a record does not carry, `ABOVE_LOD` is a property of a reading
rather than of a run, and `STATIONARY_GRADER` needs either a check standard or a weight schedule
that most converters do not carry. Each of those returns `None` with a sentence saying what would
settle it.

**What this cannot do, stated up front.**

The reading is a *window* statistic wearing a per-step name. The conditions are defined "measured
per step", and four of the twelve are statements about change (`QUASI_STATIC`, `STATIONARY_GRADER`,
`EXOGENOUS_CURRICULUM`, and the schedule half of anything else), so on a window of one step they
have nothing to compare against and return `None` rather than a vacuous pass. Widen the window with
`context=` and they become answerable. A one-step window that reported `STATIONARY_GRADER` true
because nothing can change inside one step would be the worst kind of wrong: technically defensible
and useless.

Three of the twelve are one-sided tests and say so in their detail. `EXOGENOUS_CURRICULUM` can
detect a curriculum responding to the score and cannot prove one is not; `NO_COMPACTION` on a
record whose converter never emits `CompactionEvent` cannot tell an absence from a silence, and
returns `None` when it cannot rule that out; `LIGHT_TAILED` at the exceedance counts a training
record supplies is a flag rather than an estimate, and a defensible tail estimate needs roughly
1,570 exceedances against the thirty this module will settle for.

Every threshold a verdict is compared against lives in `RegimeThresholds` and every sample-size
floor that decides measured-against-indeterminate lives in `RegimeFloors`. Two of the thresholds
come with the conditions themselves and are cited at the field; the rest are this module's defaults
and are marked as such, because a threshold in an envelope is not a decision an implementation gets
to make quietly.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import numpy as np

from reward_lens.core.budget import LimitOfDetection
from reward_lens.core.envelope import (
    ConditionReading,
    EnvelopeSpec,
    RegimeCondition,
    RegimeReading,
)
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import (
    BiasStatement,
    CostModel,
    EstimatorEntry,
    register_estimator,
)
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
    content_hash,
)
from reward_lens.measure.base import BaseObservable, Context, run
from reward_lens.record.compaction import events_in_window, summarise
from reward_lens.record.provenance import policy_mixture
from reward_lens.record.schema import Group, Run, Step, Trajectory, encode_regime_reading
from reward_lens.record.turns import Turn, mask_policy_signature

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence

#: Half-open ``[lo, hi)`` over step indices, matching `StepStream.slice` and `core.types.Span`.
Window = tuple[int, int]

#: The quantity every instrument's `measured_by` should name for each condition. Seven of the
#: twelve resolve to an id `spec/QUANTITIES.yaml` already carries. Five do not, and they are
#: proposed under the existing `run.` prefix rather than invented under a new one:
#: `run.max_staleness`, `run.curriculum_response`, `run.compaction_events`, `run.importance_ess`
#: and `run.mask_signature` have no catalogue row, so an instrument requiring `NEAR_POLICY` today
#: has no registered quantity to point `measured_by` at.
#:
#: That gap is silent rather than loud, which is why it is named here. `EnvelopeSpec.__post_init__`
#: checks that every condition in `requires` appears in `measured_by` and does not check that the
#: value is a registered quantity, so an envelope naming an id nobody registered constructs
#: cleanly and reads as rigour.
MEASURED_BY: Mapping[RegimeCondition, str] = {
    RegimeCondition.QUASI_STATIC: "run.adiabaticity",
    RegimeCondition.LINEAR_RESPONSE: "selection.explained_fraction",
    RegimeCondition.GROUP_NONDEGENERATE: "estimator.degenerate_fraction",
    RegimeCondition.NEAR_POLICY: "run.max_staleness",
    RegimeCondition.STATIONARY_GRADER: "monitor.check_standard_drift",
    RegimeCondition.EXOGENOUS_CURRICULUM: "run.curriculum_response",
    RegimeCondition.NO_COMPACTION: "run.compaction_events",
    RegimeCondition.ABOVE_LOD: "substrate.lod",
    RegimeCondition.ESS_ADEQUATE: "run.importance_ess",
    RegimeCondition.LIGHT_TAILED: "frontier.tail_index",
    RegimeCondition.SCALAR_REPRESENTABLE: "grader.curl_mass",
    RegimeCondition.MASK_STABLE: "run.mask_signature",
    # A property of a measurement design rather than of a training run, so nothing in a bare
    # record determines it and `measure_regime` reports it unknown rather than as a pass.
    # It is here because this mapping is asserted to cover every member: a condition with no
    # measuring quantity is one an envelope can require and nobody can check.
    RegimeCondition.DESIGN_CROSSED: "grader.design_balance",
}

#: Which of the twelve `MEASURED_BY` targets `spec/QUANTITIES.yaml` does not carry. Empty today,
#: and kept as data rather than deleted so the assertion stays a live check on an empty set
#: instead of a test nobody wrote. `EnvelopeSpec.__post_init__` now enforces that every id here
#: resolves, which is the guard this constant was standing in for.
UNREGISTERED_MEASURED_BY: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# The numbers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeThresholds:
    """Every number a verdict is compared against, in one place, with where it came from.

    Two of these come with the conditions themselves and are used as stated. The other seven are
    this module's defaults, chosen for the reasons given at each field. A default is not a decision;
    it is a placeholder that behaves like one until somebody says otherwise, which is why they are
    gathered here rather than scattered through the estimators.
    """

    #: Ad = tau_relax * |d log lambda / dt|. The condition as stated: "Ad much less than 1 is
    #: quasi-static and Level 0 extrapolation is licensed. Ad of order 1 or more is fast driving
    #: and it is not." The sentence gives two numbers and only one of them is a boundary: 1.0 is
    #: where the licence stops. The stricter reading of "much less than 1" argues for 0.1, which
    #: would refuse a band nothing explicitly refuses. **Chosen: 1.0.**
    ad_max: float = 1.0

    #: The fraction of groups that may be degenerate. **Stated** by the capability report, which
    #: prints "GROUP_NONDEGENERATE ok degenerate group fraction 0.04 (threshold 0.20)". Note the
    #: direction: the statistic is the *degenerate* fraction and the condition holds when it is
    #: below the threshold, which is the opposite orientation from the condition's name.
    degenerate_group_fraction_max: float = 0.20

    #: Optimizer steps behind the current policy. **Stated** by the capability report, which prints
    #: "NEAR_POLICY ok max staleness 2 steps (threshold 8)".
    max_staleness_steps: int = 8

    #: Lambda, F2's selection-explained fraction, below which the first-order expansion is not
    #: carrying the movement. Nothing states a boundary for it.
    #: **Chosen: 0.5**, on the argument that below a half the term you expanded explains less of
    #: what moved than everything you dropped. Lambda itself is F2's and is not computed here.
    explained_fraction_min: float = 0.5

    #: How far a check standard may move before the grader counts as non-stationary, in the
    #: standard's own units. A check standard is a probe set whose job is to be
    #: invariant, so any movement in it is instrument drift by construction, and the tolerance that
    #: follows from that definition is zero. **Chosen: 0.0**, which is strict and which a run
    #: should override with its standard's own reproducibility limit rather than with a guess.
    check_standard_tolerance: float = 0.0

    #: Two-sided significance for the curriculum-response test. The threshold reported beside the
    #: statistic is the critical correlation this alpha implies at the observed counts, so the
    #: number in the reading is derived rather than picked. **Chosen: 0.05.**
    curriculum_alpha: float = 0.05

    #: Kish effective sample size as a fraction of the token count, below which the importance
    #: weights have degenerated. **Chosen: 0.5**, the sequential-Monte-Carlo resampling convention.
    #: It is a convention and not a theorem, and it is the one number here with the least support.
    ess_fraction_min: float = 0.5

    #: The Hill estimate above which the tail stops being light. The one cited measurement is "a
    #: Hill estimate around 0.20 on an open reward model, consistent with light-tailed error", and
    #: the worked capability report prints "Hill gamma-hat = 0.19 [0.12, 0.27]. Light-tailed
    #: assumption holds", so whatever the bound is, 0.20 has to pass it. **Chosen: 0.25.** Strictly
    #: the moment generating function fails to exist for any positive shape at all, so this is a
    #: bound on the estimate rather than on the truth, and the interval on a training record's
    #: exceedance count is wide enough to cross it.
    hill_max: float = 0.25

    #: Curl mass, the share of comparison energy no scalar can express. No bound is stated
    #: anywhere. **Chosen: 0.05.** The one measured value in the project is 0.214 of
    #: intransitive mass, which is a substantial finding, so the bound has to sit well below it.
    curl_mass_max: float = 0.05


@dataclass(frozen=True)
class RegimeFloors:
    """Sample-size floors that decide measured against indeterminate, not verdicts.

    These are a different kind of number from `RegimeThresholds` and are kept apart on purpose.
    A threshold decides `True` against `False`; a floor decides whether an answer is available at
    all. Setting a floor too low does not make a condition fail, it makes it pass or fail on noise,
    which is worse, so each floor here is the point below which this module says `None` instead.
    """

    #: Points needed for the early AR(1) fit that supplies tau_relax at rung 0.
    ar_min_points: int = 10
    #: How many steps from the start of the run count as "early" for that fit. The relaxation time
    #: should be estimated before the transition, and a fit run over the transition
    #: measures the transition.
    ar_early_steps: int = 50
    #: Exceedances needed before a Hill estimate is reported at all. A defensible estimate needs
    #: roughly 1,570; this is a regime flag rather than a tail claim, and
    #: thirty is where the estimator stops being pure noise. Anything computed here between the
    #: two numbers is a flag, and the detail says so.
    hill_min_exceedances: int = 30
    #: Tasks needed on each side of the curriculum test before the normal approximation is used.
    curriculum_min_per_arm: int = 5
    #: Tokens needed before an importance-weight ESS is reported.
    ess_min_tokens: int = 32


@dataclass(frozen=True)
class RegimeInputs:
    """Measurements this module does not make, supplied by whoever does make them.

    Four of the twelve conditions depend on a quantity that lives in another package or in another
    run. Rather than guess them, the reading takes them here and returns `None` for the condition
    when they are absent. The distinction that matters is that an absent input produces a named
    `None` and never a default value, because a default value would be a measurement nobody took.
    """

    #: The relaxation time in optimizer steps. Rung 1 measures this by perturb and
    #: hold; supplying it here skips the AR(1) fit and the reading says which path produced it.
    tau_relax: float | None = None
    #: Which key of `Step.schedule` is the annealing parameter. None means every positive key is
    #: considered and the fastest-moving one is reported, which is the conservative reading: the
    #: condition has to hold for the driver, and the driver is whichever parameter moves fastest.
    schedule_parameter: str | None = None
    #: Which recorded series the AR(1) relaxation time is fitted to.
    relaxation_series: Literal["group_mean", "entropy", "kl_to_previous"] = "group_mean"
    #: Lambda, the selection-explained fraction. F2 computes it; this module does not.
    explained_fraction: float | None = None
    #: The substrate's disagreement with itself, for the below-LOD check.
    lod: LimitOfDetection | None = None
    #: The effect being tested against that limit. `ABOVE_LOD` is a property of a reading and not
    #: of a run, so both of these have to arrive together or the condition stays `None`.
    effect: float | None = None
    #: Grader or rubric weights per step, from a converter that could see the schedule. The direct
    #: observation of what `STATIONARY_GRADER` is about; a check standard is the indirect proxy.
    grader_weights: Mapping[int, Mapping[str, float]] | None = None
    #: Whether the converter that wrote this record emits `CompactionEvent` at all. None means
    #: infer, which can only ever be inferred one way. See `_measure_no_compaction`.
    compaction_recorded: bool | None = None
    #: An explicit sample for the tail estimate, when the quantity whose tail matters is not the
    #: advantage. Overrides everything the record would supply.
    tail_sample: Sequence[float] | None = None
    #: Whether the importance-weight ESS is computed over tokens or over sequences. Tokens is the
    #: default because the clipped objective weights tokens; sequences degenerate by construction
    #: on any long generation and the detail says so.
    ess_level: Literal["token", "sequence"] = "token"


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

_NAN = float("nan")


def _provenance(condition: RegimeCondition, material: Mapping[str, Any]) -> str:
    """A content-derived id for one condition's determination, so the check is auditable.

    `ConditionReading.provenance` has the type `EvidenceID` for the reason "so the
    check is itself auditable". There is no Evidence row at the point a condition is decided, so
    this is a content hash over the material that decided it: the same inputs give the same id, and
    two readings that disagree have visibly different material rather than an unexplained
    difference. When `RunRegime` emits, the whole reading travels in the Evidence value, which is
    what makes the id resolvable to something a reader can open.
    """
    return content_hash({"condition": condition.name, **dict(material)}, "regime")


def _reading(
    condition: RegimeCondition,
    holds: bool | None,
    statistic: float,
    threshold: float,
    detail: str,
    material: Mapping[str, Any],
) -> ConditionReading:
    return ConditionReading(
        condition=condition,
        holds=holds,
        statistic=statistic,
        threshold=threshold,
        provenance=_provenance(condition, material),
        detail=detail,
    )


def _undetermined(
    condition: RegimeCondition, detail: str, material: Mapping[str, Any] | None = None
) -> ConditionReading:
    """The third state, with the sentence that says what would settle it.

    `statistic` and `threshold` are NaN rather than zero. A zero here would be a number somebody
    could read off a report, and there is no number: nothing was determined.
    """
    return _reading(condition, None, _NAN, _NAN, detail, material or {"determined": False})


def _trajectories(steps: Sequence[Step]) -> tuple[Trajectory, ...]:
    return tuple(t for s in steps for g in s.groups for t in g.trajectories)


def _groups(steps: Sequence[Step]) -> tuple[Group, ...]:
    return tuple(g for s in steps for g in s.groups)


def _turns(steps: Sequence[Step]) -> tuple[Turn, ...]:
    return tuple(turn for t in _trajectories(steps) for turn in t.turns)


def _window_of(run: Run, step: int | None, window: Window | None, context: int) -> Window:
    """Resolve the two ways of asking into the half-open span the whole module works in."""
    if step is not None and window is not None:
        raise ValueError(
            "ask for a step or a window, not both. `step=k` is `window=(k, k+1)` widened by "
            "`context`, and passing both leaves it ambiguous which one the context applies to."
        )
    if window is not None:
        lo, hi = window
    elif step is not None:
        lo, hi = step, step + 1
    else:
        indices = run.steps.indices
        return (min(indices), max(indices) + 1) if indices else (0, 0)
    if hi <= lo:
        raise ValueError(
            f"the window {(lo, hi)} is empty or reversed. It is half-open [lo, hi), so a one-step "
            f"window at step k is (k, k + 1)."
        )
    return (lo - context, hi + context)


# ---------------------------------------------------------------------------
# 1. QUASI_STATIC
# ---------------------------------------------------------------------------


def _linear_detrend(x: np.ndarray) -> np.ndarray:
    """Residuals about the least-squares line, because AR(1) on a trend measures the trend.

    A training run's mean reward climbs. Fitting an autoregression to a climbing series returns a
    lag-1 coefficient near one whatever the relaxation dynamics are, and a coefficient near one
    turns into an unbounded relaxation time, which turns into an adiabaticity number that says
    "fast driving" on every run ever recorded. Detrending first is what the early-warning
    literature does and it is not optional here.
    """
    t = np.arange(x.size, dtype=float)
    tc = t - t.mean()
    denominator = float(np.dot(tc, tc))
    if denominator <= 0:
        return x - float(x.mean())
    slope = float(np.dot(tc, x - x.mean())) / denominator
    return x - (slope * t + (float(x.mean()) - slope * float(t.mean())))


def _ar1_tau(series: Sequence[float], floors: RegimeFloors) -> tuple[float | None, str]:
    """tau_relax from the early AR(1) coefficient: ``phi^k = exp(-k / tau)``, so ``tau = -1/ln phi``.

    Returns the relaxation time in optimizer steps and a sentence about the fit. Three fits return
    no time at all and each is a different thing:

    A coefficient at or below zero is not a relaxation. The series alternates rather than decays,
    so there is no exponential envelope to read a time off, and reporting a time from it would be
    reporting the sign of a number as if it were a duration.

    A coefficient at or above one is non-stationary. The fit says the series does not return, which
    would make the relaxation time infinite and the adiabaticity number infinite with it. That is a
    tempting `False` and it is not one: at these sample sizes a coefficient above one is more often
    a fit artifact than a physical statement, and the honest answer is that the estimator's own
    assumption failed.

    **The bias runs the dangerous way and is not corrected here.** An OLS lag-1 coefficient is
    biased low by roughly ``(1 + 3 phi) / n``, and detrending first costs about as much again, so
    on a forty-point early window a true 0.75 fits nearer 0.6 and the relaxation time comes out
    short. A short relaxation time makes `Ad` small, which makes `QUASI_STATIC` pass. So the cheap
    rung errs toward licensing rather than toward refusing, which is the wrong direction for a
    safety check, and the reading says so wherever it uses this path. Correcting it means adopting
    a bias-corrected estimator, which is a decision about what the library estimates rather than an
    implementation detail, so it is recorded rather than taken here.
    """
    x = np.asarray([float(v) for v in series], dtype=float)
    if x.size < floors.ar_min_points:
        return None, (
            f"the relaxation fit needs {floors.ar_min_points} points and the early window has "
            f"{x.size}"
        )
    r = _linear_detrend(x)
    denominator = float(np.dot(r[:-1], r[:-1]))
    if denominator <= 0:
        return None, "the detrended series is exactly flat, so no autocorrelation is identified"
    phi = float(np.dot(r[:-1], r[1:])) / denominator
    if phi <= 0:
        return None, (
            f"the early AR(1) coefficient is {phi:.4g}, at or below zero. The series does not "
            f"decay toward its mean, so there is no relaxation time to read off it"
        )
    if phi >= 1:
        return None, (
            f"the early AR(1) coefficient is {phi:.4g}, at or above one. The fit is non-stationary "
            f"and tau_relax is not identified by it; measure it by perturb-and-hold and supply it "
            f"as tau_relax"
        )
    return -1.0 / math.log(phi), (
        f"early AR(1) coefficient {phi:.4g} over {x.size} steps, biased low by roughly "
        f"{(1 + 3 * phi) / x.size:.3g} and with it Ad, so this rung errs toward licensing"
    )


def _relaxation_series(steps: Sequence[Step], which: str) -> list[float]:
    """The series the relaxation time is fitted to, in step order."""
    out: list[float] = []
    for s in steps:
        if which == "entropy":
            value = s.optimizer.entropy
        elif which == "kl_to_previous":
            value = s.optimizer.kl_to_previous
        else:
            means = [g.group_stats.mean for g in s.groups if g.group_stats.mean is not None]
            value = float(np.mean(means)) if means else None
        if value is not None and math.isfinite(value):
            out.append(float(value))
    return out


def _log_lambda_rate(steps: Sequence[Step], key: str | None) -> tuple[float | None, str, str, str]:
    """``max |d log lambda / dt|`` over the window, per optimizer step.

    Returns the rate, the parameter that attained it, a sentence about how, and a failure reason
    when there is no rate. The maximum over parameters rather than a named one, unless the caller
    names one, because the condition has to hold for whatever is actually driving the system and
    the fastest-moving parameter is the one that breaks it first.

    A parameter at or below zero is skipped rather than clamped. A coefficient annealed to exactly
    zero has an unbounded log derivative, which is a real statement about the schedule and not a
    number that belongs in a maximum.
    """
    if len(steps) < 2:
        return None, "", "", "a window of one step has no schedule derivative"
    best_rate = -1.0
    best_key = ""
    best_where = ""
    seen_any_key = False
    seen_positive = False
    for left, right in zip(steps, steps[1:]):
        dt = float(right.index - left.index)
        if dt <= 0:
            continue
        keys = [key] if key is not None else sorted(set(left.schedule) & set(right.schedule))
        for k in keys:
            a, b = left.schedule.get(k), right.schedule.get(k)
            if a is None or b is None:
                continue
            seen_any_key = True
            if a <= 0 or b <= 0:
                continue
            seen_positive = True
            rate = abs(math.log(b) - math.log(a)) / dt
            if rate > best_rate:
                best_rate, best_key = rate, k
                best_where = f"steps {left.index} to {right.index}"
    if best_rate < 0:
        if not seen_any_key:
            named = f" named {key!r}" if key else ""
            return (
                None,
                "",
                "",
                (
                    f"no schedule parameter{named} is recorded on consecutive steps in this window, so "
                    f"there is no d log lambda / dt to compute. An empty schedule is not a flat one"
                ),
            )
        if not seen_positive:
            return (
                None,
                "",
                "",
                (
                    "every recorded schedule parameter is zero or negative somewhere in this window, "
                    "and the log derivative of a parameter that reaches zero is unbounded"
                ),
            )
        return None, "", "", "no consecutive pair of steps in this window carries the schedule"
    return best_rate, best_key, best_where, ""


def _measure_quasi_static(
    steps: Sequence[Step],
    thresholds: RegimeThresholds,
    floors: RegimeFloors,
    inputs: RegimeInputs,
    early: Sequence[Step],
) -> ConditionReading:
    """``Ad = tau_relax * |d log lambda / dt|``.

    Both factors are in optimizer steps, so the product is dimensionless: a relaxation time in
    steps against a driving rate per step. The rung-0 relaxation time comes from the early AR(1)
    coefficient of a recorded series, which is the cheap route to it and is weaker
    than the perturb-and-hold measurement in every way except cost. The worked
    capability report prints this condition as `unknown` with "requires two anneal rates; see G3",
    which is the rung-2 answer; this is the rung that a single record supports.

    The flat-schedule case is special and is not a shortcut. If every recorded schedule parameter
    is constant across the window then the driver is not moving, `Ad` is zero whatever the
    relaxation time is, and the condition holds without the relaxation time having been measured at
    all. Multiplying an unmeasured time by an exact zero and calling the product unknown would
    refuse the one case that is unambiguously quasi-static.
    """
    condition = RegimeCondition.QUASI_STATIC
    rate, key, where, why = _log_lambda_rate(steps, inputs.schedule_parameter)
    if rate is None:
        return _undetermined(condition, why)

    if rate == 0.0:
        return _reading(
            condition,
            True,
            0.0,
            thresholds.ad_max,
            (
                "every recorded schedule parameter is constant across this window, so the driver "
                "is not moving and Ad is zero whatever tau_relax is"
            ),
            {"rate": 0.0, "flat": True},
        )

    if inputs.tau_relax is not None:
        tau, how = float(inputs.tau_relax), "tau_relax supplied by the caller"
    else:
        tau, how = _ar1_tau(_relaxation_series(early, inputs.relaxation_series), floors)
        if tau is None:
            return _undetermined(
                condition,
                (
                    f"the schedule moves at |d log {key} / dt| = {rate:.4g} per step but "
                    f"tau_relax is not available: {how}. Ad is a product and one factor is "
                    f"missing, so no number is reported"
                ),
                {"rate": rate},
            )
        how = f"tau_relax {tau:.4g} steps from the {how}, on the {inputs.relaxation_series} series"

    ad = tau * rate
    return _reading(
        condition,
        ad <= thresholds.ad_max,
        ad,
        thresholds.ad_max,
        f"Ad = {tau:.4g} x {rate:.4g}, driven by {key!r} over {where}; {how}",
        {"tau": tau, "rate": rate, "key": key},
    )


# ---------------------------------------------------------------------------
# 2. LINEAR_RESPONSE
# ---------------------------------------------------------------------------


def _measure_linear_response(
    thresholds: RegimeThresholds, inputs: RegimeInputs
) -> ConditionReading:
    """Measured by Lambda, which is F2's selection-explained fraction and is F2's to compute.

    The condition is "step small enough that the O(eta squared) term is negligible; measured by
    Lambda", and Lambda is the R-squared of the
    observed movement regressed on the first-order selection term. So a high Lambda says the term
    that was expanded explains what moved, which is the operational form of the second-order term
    being negligible.

    Two limitations ride along and belong in any reading that uses this. Lambda is a fraction of
    variance across steps, so it is a window statistic and cannot license a single step inside a
    window where it is low. And a low Lambda has many causes other than a large step: a missing
    feature in the basis lowers it exactly as a second-order term does, so this condition is
    necessary for linear response and not sufficient for it.
    """
    condition = RegimeCondition.LINEAR_RESPONSE
    if inputs.explained_fraction is None:
        return _undetermined(
            condition,
            (
                "Lambda is not available. It is F2's selection-explained fraction "
                "(`selection.explained_fraction`), and nothing in a record computes "
                "it without a featuriser. Supply it as `RegimeInputs.explained_fraction`"
            ),
        )
    value = float(inputs.explained_fraction)
    return _reading(
        condition,
        value >= thresholds.explained_fraction_min,
        value,
        thresholds.explained_fraction_min,
        (
            "Lambda supplied by the caller; it is a window statistic and a fraction of variance, "
            "so it is necessary for linear response and not sufficient for it"
        ),
        {"lambda": value},
    )


# ---------------------------------------------------------------------------
# 3. GROUP_NONDEGENERATE
# ---------------------------------------------------------------------------


def _measure_group_nondegenerate(
    groups: Sequence[Group], thresholds: RegimeThresholds
) -> ConditionReading:
    """K > 1 and std(r) > 0 for a stated fraction of groups.

    The statistic reported is the *degenerate* fraction, not the non-degenerate one, because that
    is what the capability report prints beside the threshold: "degenerate group
    fraction 0.04 (threshold 0.20)". Reporting the complement against the same number would invert
    the verdict silently.

    Degeneracy is read off `GroupStats.degenerate`, which compares the group's standard deviation
    against the *estimator's* epsilon rather than against zero. That is deliberate in the record
    and it is the right boundary here too: the advantage the trainer computed is `0 / (0 + eps)`,
    so what makes a group contribute nothing is the trainer's epsilon and not exact equality.

    Groups whose standard deviation was never recorded are counted separately and can make the
    whole reading indeterminate. If the degenerate fraction is below the threshold when every
    unrecorded group is assumed live and above it when every one is assumed dead, then the answer
    depends entirely on data that is not there, and the honest return is `None`.
    """
    condition = RegimeCondition.GROUP_NONDEGENERATE
    if not groups:
        return _undetermined(
            condition, "the window contains no groups, so there is no group structure to read"
        )
    n_degenerate = 0
    n_unknown = 0
    n_all_fail = 0
    for g in groups:
        stats = g.group_stats
        if stats.all_fail:
            n_all_fail += 1
        if stats.k <= 1:
            n_degenerate += 1
        elif stats.degenerate:
            n_degenerate += 1
        elif stats.std is None:
            n_unknown += 1
        elif stats.std <= 0.0:
            n_degenerate += 1
    total = len(groups)
    low = n_degenerate / total
    high = (n_degenerate + n_unknown) / total
    limit = thresholds.degenerate_group_fraction_max
    tail = (
        f"; {n_all_fail} of {total} groups are all-fail, where a z-scoring estimator rescales "
        f"whatever noise is left to full magnitude"
        if n_all_fail
        else ""
    )
    if n_unknown and low <= limit < high:
        return _undetermined(
            condition,
            (
                f"{n_degenerate} of {total} groups are degenerate and {n_unknown} never recorded a "
                f"standard deviation, so the degenerate fraction is somewhere in "
                f"[{low:.4g}, {high:.4g}] and the threshold {limit:.4g} falls inside it{tail}"
            ),
            {"low": low, "high": high},
        )
    return _reading(
        condition,
        high <= limit,
        low,
        limit,
        (
            f"{n_degenerate} of {total} groups degenerate"
            + (f", {n_unknown} indeterminate and assumed degenerate" if n_unknown else "")
            + tail
        ),
        {"degenerate": n_degenerate, "total": total, "unknown": n_unknown},
    )


# ---------------------------------------------------------------------------
# 4. NEAR_POLICY
# ---------------------------------------------------------------------------


def _measure_near_policy(
    steps: Sequence[Step], trajectories: Sequence[Trajectory], thresholds: RegimeThresholds
) -> ConditionReading:
    """Staleness below a bound, and segment provenance singular. Both halves of the condition.

    Singularity is asserted *per trajectory* and never across the window. A window of forty steps
    necessarily contains forty policy versions, because that is what training is; what the
    condition is about is whether a single trajectory was generated by one policy, which is the
    partial-rollout case that makes a per-trajectory importance ratio undefined rather than stale.
    Reading singularity across the window would fail every multi-step reading by construction.

    Two sources of staleness are combined and the larger wins. `SegmentProvenance.staleness_steps`
    is per segment and is the authoritative one; `OptimizerTelemetry.staleness` is the batch
    histogram and covers rollouts whose provenance a converter could not resolve to segments. A
    record carrying only the histogram still gets an answer, and the detail says which source the
    number came from.
    """
    condition = RegimeCondition.NEAR_POLICY
    segments = [seg for t in trajectories for seg in t.provenance]
    histograms = [s.optimizer.staleness for s in steps if s.optimizer.staleness.total]
    if not segments and not histograms:
        return _undetermined(
            condition,
            (
                "no trajectory in this window carries segment provenance and no step carries a "
                "staleness histogram, so neither half of the condition has an input. Segment "
                "provenance is mandatory on any trajectory with turns, so this window either has "
                "no rollouts or holds only empty ones"
            ),
        )
    mixture = policy_mixture(segments)
    n_mixed = sum(1 for t in trajectories if t.provenance and not t.is_single_policy)
    from_segments = mixture.max_staleness if segments else 0
    from_histogram = max((h.max for h in histograms), default=0)
    worst = max(from_segments, from_histogram)
    source = "segment provenance" if from_segments >= from_histogram else "the staleness histogram"
    singular = n_mixed == 0
    detail = (
        f"max staleness {worst} steps from {source}, mean "
        f"{mixture.mean_staleness:.4g} over {mixture.n_segments} segments; "
        + (
            f"{n_mixed} of {len(trajectories)} trajectories were generated by more than one policy "
            f"version, so a single importance ratio is undefined on them"
            if n_mixed
            else f"all {len(trajectories)} trajectories are single-policy"
        )
    )
    return _reading(
        condition,
        singular and worst <= thresholds.max_staleness_steps,
        float(worst),
        float(thresholds.max_staleness_steps),
        detail,
        {"max_staleness": worst, "mixed": n_mixed},
    )


# ---------------------------------------------------------------------------
# 5. STATIONARY_GRADER
# ---------------------------------------------------------------------------


def _numeric_signature(node: Any, depth: int = 0) -> tuple[Any, ...]:
    """Every named constant in a score tree, without importing the module that defines one.

    `record.scores` owns `ScoreTree` and this module must not depend on it existing, so the walk is
    structural
    rather than typed: a node has a `name`, its numeric parameters are whatever of `weights`,
    `bias`, `constant`, `threshold`, `cap` and `knots` it carries, and its children live under
    `children`, `otherwise`, `primary`/`secondary` or `child`. Anything that is not shaped like that
    contributes nothing and the caller falls through to the next route.

    The failure mode is the safe one and it is worth being explicit about. A walk that misses a
    node finds fewer constants, so it reports *no* signature and `STATIONARY_GRADER` comes back
    undetermined; it cannot report stability it did not establish. A walk that picks up a number it
    should have ignored reports a change that is not one, which refuses. Neither direction produces
    a false "the grader held still".
    """
    if node is None or depth > 32:
        return ()
    name = getattr(node, "name", None)
    if not isinstance(name, str):
        return ()
    parts: list[Any] = [name]
    for attr in ("weights", "bias", "constant", "threshold", "cap", "knots"):
        value = getattr(node, attr, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parts.append((attr, float(value)))
        elif isinstance(value, (tuple, list)) and value:
            parts.append((attr, repr(tuple(value))))
    kids: list[Any] = []
    for attr in ("children", "otherwise", "primary", "secondary", "child"):
        value = getattr(node, attr, None)
        if isinstance(value, (tuple, list)):
            kids.extend(value)
        elif value is not None and hasattr(value, "name"):
            kids.append(value)
    return (tuple(parts),) + tuple(
        sig for kid in kids for sig in _numeric_signature(kid, depth + 1)
    )


def _score_tree_signature(step_: Step) -> tuple[Any, ...] | None:
    """The composition one step scored against, or None when the step carries no score tree.

    The first trajectory of the first group, not all of them. The score tree is the grader's
    composition, so it is a property of the step rather than of the rollout, and walking five
    hundred identical trees per step to answer whether the composition changed *between* steps
    would cost four hundred times what the question is worth.
    """
    for g in step_.groups:
        for t in g.trajectories:
            signature = _numeric_signature(t.scores)
            if signature:
                return signature
    return None


def _measure_stationary_grader(
    steps: Sequence[Step], thresholds: RegimeThresholds, inputs: RegimeInputs
) -> ConditionReading:
    """Grader and rubric weights unchanged across the window.

    Three routes, tried in that order, and they are not equivalent.

    Supplied weights are the direct observation of the thing the condition names, and when they are
    there the reading counts distinct weight vectors and names the step at which they first changed,
    which is the sentence the capability report prints: "rubric weights changed at step 240".

    A score tree on the record is the same observation taken from the record instead of from the
    caller, and it is strictly better because nobody had to remember to pass it. It is read
    structurally rather than through an import, so a record written before score trees existed, or
    by a converter that attaches something else, falls through rather than failing.

    A check standard is the indirect route: it is a frozen probe set whose job
    is to be invariant, so any movement in it is instrument drift by construction. It is a lower
    bound on the grader's movement rather than a measurement of it, because a grader can change in
    ways the standard does not resolve.

    None of the three exists on a bare record, and that is the common case. The grader's
    `ComponentRef` is run-level so it cannot show a within-run change, and the run's own
    `RegimeDeclaration` is a claim rather than a measurement, so it is reported as a disagreement
    and never as a verdict.
    """
    condition = RegimeCondition.STATIONARY_GRADER
    if inputs.grader_weights:
        indices = [s.index for s in steps if s.index in inputs.grader_weights]
        if len(indices) < 2:
            return _undetermined(
                condition,
                (
                    f"grader weights were supplied for {len(indices)} of the {len(steps)} steps in "
                    f"this window, and a window with fewer than two weight vectors cannot exhibit "
                    f"a change"
                ),
            )
        vectors = [
            tuple(sorted((k, float(v)) for k, v in inputs.grader_weights[i].items()))
            for i in indices
        ]
        n_distinct = len(set(vectors))
        changed_at = next(
            (
                nxt
                for nxt, before, after in zip(indices[1:], vectors, vectors[1:])
                if before != after
            ),
            None,
        )
        detail = f"{n_distinct} distinct weight vectors over {len(indices)} recorded steps" + (
            f"; rubric weights changed at step {changed_at}" if changed_at is not None else ""
        )
        return _reading(
            condition,
            n_distinct == 1,
            float(n_distinct),
            1.0,
            detail,
            {"distinct": n_distinct, "changed_at": changed_at},
        )

    trees = [(s.index, _score_tree_signature(s)) for s in steps]
    with_trees = [(i, sig) for i, sig in trees if sig is not None]
    if len(with_trees) >= 2:
        signatures = [sig for _, sig in with_trees]
        n_distinct = len(set(signatures))
        changed_at = next(
            (
                nxt
                for (nxt, _), before, after in zip(with_trees[1:], signatures, signatures[1:])
                if before != after
            ),
            None,
        )
        detail = (
            f"{n_distinct} distinct score-tree compositions over {len(with_trees)} steps carrying "
            f"one, read structurally off the record"
            + (f"; rubric weights changed at step {changed_at}" if changed_at is not None else "")
        )
        return _reading(
            condition,
            n_distinct == 1,
            float(n_distinct),
            1.0,
            detail,
            {"distinct": n_distinct, "changed_at": changed_at, "source": "score_tree"},
        )

    series: dict[str, list[tuple[int, float]]] = {}
    for s in steps:
        for probe in s.probes:
            if probe.channel == "check_standard" and probe.value is not None:
                series.setdefault(probe.name, []).append((s.index, float(probe.value)))
    usable = {name: pts for name, pts in series.items() if len(pts) >= 2}
    if not usable:
        return _undetermined(
            condition,
            (
                "no grader weight schedule was supplied, no two steps in this window carry a score "
                "tree, and no check standard was read twice, so nothing here can see the grader "
                "change. Record a "
                "`ProbeResult(channel='check_standard')` on each step, or supply the weight "
                "schedule as `RegimeInputs.grader_weights`"
            ),
        )
    worst_name, worst_drift, worst_at = "", -1.0, 0
    for name, points in usable.items():
        points.sort()
        values = [v for _, v in points]
        drift = max(values) - min(values)
        if drift > worst_drift:
            worst_name, worst_drift = name, drift
            # The step after the largest single jump: where a reader should look first.
            jumps = [abs(b - a) for a, b in zip(values, values[1:])]
            worst_at = points[jumps.index(max(jumps)) + 1][0]
    n_readings = len(usable[worst_name])
    moved = (
        f"check standard {worst_name!r} did not move across {n_readings} readings"
        if worst_drift == 0.0
        else (
            f"check standard {worst_name!r} moved by {worst_drift:.4g} over {n_readings} readings, "
            f"most of it by step {worst_at}"
        )
    )
    return _reading(
        condition,
        worst_drift <= thresholds.check_standard_tolerance,
        worst_drift,
        thresholds.check_standard_tolerance,
        (
            f"{moved}. A check standard is invariant by construction, so its movement is drift; "
            f"the tolerance is the standard's own reproducibility limit and defaults to zero "
            f"because nothing here has measured one"
        ),
        {"probe": worst_name, "drift": worst_drift},
    )


# ---------------------------------------------------------------------------
# 6. EXOGENOUS_CURRICULUM
# ---------------------------------------------------------------------------


def _measure_exogenous_curriculum(
    steps: Sequence[Step], thresholds: RegimeThresholds, floors: RegimeFloors
) -> ConditionReading:
    """The task distribution is not responding to the score.

    The mechanism this detects is the live one: a sampler that drops the tasks the policy has
    solved, which is DAPO's dynamic sampling and every curriculum built on it. Its signature in a
    record is that a task's score at step t predicts whether it is still there at step t+1. So the
    statistic is a rank-biserial correlation between a task's mean score and its survival to the
    next step, pooled over consecutive step pairs, and the threshold beside it is the critical
    correlation the declared alpha implies at the observed counts rather than a number picked here.

    Three things this cannot do, all of which belong wherever the reading is used. It is one-sided:
    a correlation shows a curriculum responding to the score, and its absence shows only that this
    test at this power did not see one. It only looks one step ahead, so a curriculum that recycles
    tasks on a longer cycle is invisible to it. And pooling across step pairs treats the same task
    at different steps as independent observations, which it is not, so the critical value is
    optimistic and the test over-rejects. Over-rejecting errs toward refusing, which is the right
    direction to be wrong in here.
    """
    condition = RegimeCondition.EXOGENOUS_CURRICULUM
    if len(steps) < 2:
        return _undetermined(
            condition,
            (
                "a window of one step cannot show the task distribution responding to anything. "
                "Widen it with `context=` or ask for a multi-step window"
            ),
        )
    survived: list[float] = []
    dropped: list[float] = []
    for left, right in zip(steps, steps[1:]):
        later = {g.task_ref for g in right.groups}
        scores: dict[Any, list[float]] = {}
        for g in left.groups:
            if g.group_stats.mean is not None:
                scores.setdefault(g.task_ref, []).append(float(g.group_stats.mean))
        for task, values in scores.items():
            mean = float(np.mean(values))
            (survived if task in later else dropped).append(mean)
    n1, n2 = len(survived), len(dropped)
    if n1 == 0 and n2 == 0:
        return _undetermined(
            condition,
            (
                "no group in this window records a mean score, so there is nothing to correlate "
                "against task survival"
            ),
        )
    if n2 == 0:
        return _reading(
            condition,
            True,
            0.0,
            1.0,
            (
                f"every one of the {n1} task appearances is present again at the next step, so "
                f"nothing is being selected out and there is no selection for the score to drive"
            ),
            {"survived": n1, "dropped": 0},
        )
    if n1 == 0:
        return _undetermined(
            condition,
            (
                f"none of the {n2} tasks in this window recurs at the next step, so survival is "
                f"constant and carries no signal about the score. A curriculum operating on a "
                f"longer cycle than one step is invisible to this rung"
            ),
        )
    if n1 < floors.curriculum_min_per_arm or n2 < floors.curriculum_min_per_arm:
        return _undetermined(
            condition,
            (
                f"{n1} tasks survived and {n2} were dropped, against a floor of "
                f"{floors.curriculum_min_per_arm} on each side. The normal approximation behind "
                f"the critical value does not hold at these counts, and a test that cannot reject "
                f"is not a test that passed"
            ),
            {"survived": n1, "dropped": n2},
        )
    a = np.asarray(survived, dtype=float)[:, None]
    b = np.asarray(dropped, dtype=float)[None, :]
    auc = float(((a > b).sum() + 0.5 * (a == b).sum()) / (n1 * n2))
    rho = 2.0 * auc - 1.0
    sd = math.sqrt((n1 + n2 + 1.0) / (3.0 * n1 * n2))
    z = statistics.NormalDist().inv_cdf(1.0 - thresholds.curriculum_alpha / 2.0)
    critical = z * sd
    if critical >= 1.0:
        return _undetermined(
            condition,
            (
                f"at {n1} surviving and {n2} dropped tasks the critical correlation is "
                f"{critical:.4g}, which no observable correlation can exceed. The test cannot "
                f"reject anything here, so it has not passed"
            ),
            {"critical": critical},
        )
    return _reading(
        condition,
        abs(rho) <= critical,
        abs(rho),
        critical,
        (
            f"rank-biserial correlation {rho:+.4g} between a task's mean score and its survival to "
            f"the next step, over {n1} surviving and {n2} dropped appearances; critical value at "
            f"alpha {thresholds.curriculum_alpha:g}. One-sided in meaning: this can show a "
            f"curriculum responding to the score and cannot show that none does"
        ),
        {"rho": rho, "n1": n1, "n2": n2},
    )


# ---------------------------------------------------------------------------
# 7. NO_COMPACTION
# ---------------------------------------------------------------------------


def _measure_no_compaction(
    trajectories: Sequence[Trajectory], inputs: RegimeInputs
) -> ConditionReading:
    """No prefix rewrite inside the measurement window.

    After a rewrite the old prefix is a different string, so the importance ratio is undefined
    rather than stale and the violation behaviour is `refuse` rather than `bound`. The count uses
    `record.compaction.events_in_window` so the boundary convention is the record's own: an event
    at the first turn of the span does not cross it, because everything in the span was then
    generated against one post-rewrite prefix, and a window pinned to a compaction boundary is
    exactly the analysis that stays valid.

    The hard part is not counting, it is knowing what an empty count means. `Trajectory.compaction`
    is an empty tuple both when nothing was compacted and when the converter does not emit
    compaction events at all, and those are opposite claims. Three ways out, in order: a
    single-turn trajectory cannot have a prefix rewritten between turns, so an absence there is a
    proof; a window in which some trajectory carries an event shows the converter emits them, so an
    absence elsewhere is evidence; and otherwise the answer is `None` until somebody says which
    kind of record this is via `RegimeInputs.compaction_recorded`.
    """
    condition = RegimeCondition.NO_COMPACTION
    if not trajectories:
        return _undetermined(condition, "the window contains no trajectories")
    events = [
        e
        for t in trajectories
        for e in events_in_window(t.compaction, first_turn=0, last_turn=max(t.n_turns - 1, 0))
    ]
    summary = summarise(events)
    multi_turn = sum(1 for t in trajectories if t.n_turns > 1)
    any_recorded = any(t.compaction for t in trajectories)
    known = inputs.compaction_recorded
    if known is None:
        known = multi_turn == 0 or any_recorded
    if not known:
        return _undetermined(
            condition,
            (
                f"{multi_turn} of {len(trajectories)} trajectories are multi-turn and not one "
                f"carries a CompactionEvent anywhere in this window. On a converter that emits "
                f"them that is a clean zero; on one that does not it is silence, and nothing here "
                f"can tell the two apart. Set `RegimeInputs.compaction_recorded` to say which"
            ),
            {"multi_turn": multi_turn},
        )
    basis = (
        "no trajectory in this window has more than one turn, so no prefix could be rewritten "
        "between turns"
        if multi_turn == 0
        else (
            f"{summary.n_events} compaction events recorded in this window, of which "
            f"{summary.n_lossy} removed tokens ({summary.tokens_removed} in total)"
            + (f", by {', '.join(summary.methods)}" if summary.methods else "")
        )
    )
    return _reading(
        condition,
        summary.n_lossy == 0,
        float(summary.n_lossy),
        0.0,
        basis,
        {"n_lossy": summary.n_lossy, "n_events": summary.n_events},
    )


# ---------------------------------------------------------------------------
# 8. ABOVE_LOD
# ---------------------------------------------------------------------------


def _measure_above_lod(inputs: RegimeInputs) -> ConditionReading:
    """The effect exceeds the limit of detection.

    This is the one condition of the twelve that is not a property of a run. A limit of detection
    is `3.3 sigma_blank / S` for one measurement substrate, and whether something exceeds it is a
    question about one effect measured on that substrate. A record has no effect in it; it has
    rollouts. So a run-level answer to "is the effect above the limit" is either a statement about
    an effect nobody named, or it is nothing, and it is nothing.

    What that means in practice, and it is worth stating because it looks like a gap: an instrument
    requiring `ABOVE_LOD` should get its check from `Context.lod` at the moment it has a reading to
    compare, which `BaseObservable.preflight` already names in `unchecked` when the limit is
    absent. This condition returns a verdict only when the caller hands both the effect and the
    limit to the regime reading, which is the case where a run-level reading is being used to carry
    a per-reading fact, and the detail says so.
    """
    condition = RegimeCondition.ABOVE_LOD
    if inputs.lod is None or inputs.effect is None:
        missing = []
        if inputs.lod is None:
            missing.append("no LimitOfDetection")
        if inputs.effect is None:
            missing.append("no effect to test against one")
        return _undetermined(
            condition,
            (
                f"this condition is a property of a reading and not of a run, and the run-level "
                f"reading was given {' and '.join(missing)}. Check it where the reading exists, "
                f"against `Context.lod`"
            ),
        )
    lod = inputs.lod
    if not lod.is_determinate:
        return _undetermined(
            condition,
            (
                f"the calibration slope is {lod.sensitivity:.4g}, so the reading does not respond "
                f"to dose and there is no detection limit to be above. Fit a dose sweep first"
            ),
        )
    effect = abs(float(inputs.effect))
    verdict = lod.verdict(effect)
    return _reading(
        condition,
        verdict != "below_lod",
        effect,
        lod.lod,
        (
            f"|effect| {effect:.4g} against LOD {lod.lod:.4g} and LOQ {lod.loq:.4g}: {verdict}. "
            f"Supplied per reading; this is not a property of the run"
        ),
        {"effect": effect, "lod": lod.lod},
    )


# ---------------------------------------------------------------------------
# 9. ESS_ADEQUATE
# ---------------------------------------------------------------------------


def _measure_ess_adequate(
    trajectories: Sequence[Trajectory],
    thresholds: RegimeThresholds,
    floors: RegimeFloors,
    inputs: RegimeInputs,
) -> ConditionReading:
    """The importance weights have not degenerated.

    The weights come from the two logprob streams the record already carries: `logprobs_sampling`
    is what the inference engine assigned when the tokens were drawn and `logprobs_train` is what
    the training engine assigns to the same tokens, so their difference exponentiates to the ratio
    the estimator reweights by. Kish's effective sample size over those weights, as a fraction of
    the token count, is the statistic.

    **That difference is not purely policy drift and this reading must not be read as though it
    were.** `record.turns` is explicit that the two streams are two engines, and instrument E6
    exists precisely to measure how much they disagree on identical tokens. A ratio built from them
    carries both the drift and the engine mismatch, and nothing here can separate the two. A low
    ESS from this estimator says the weights have degenerated; it does not say the policy moved.

    Token level by default, because the clipped objective weights tokens. The sequence-level
    alternative is available and it degenerates by construction on long generations, since a
    sequence log-ratio is a sum over thousands of tokens and the largest one takes all the weight,
    so a sequence-level ESS near one is a statement about generation length as much as about
    off-policyness.
    """
    condition = RegimeCondition.ESS_ADEQUATE
    per_sequence: list[float] = []
    per_token: list[float] = []
    turns_with_streams = 0
    for t in trajectories:
        total = 0.0
        seen = False
        for turn in t.turns:
            gap = turn.logprob_gap()
            if gap is None:
                continue
            turns_with_streams += 1
            seen = True
            mask = turn.loss_mask
            kept = [g for g, m in zip(gap, mask) if m] if mask is not None else list(gap)
            per_token.extend(kept)
            total += float(sum(kept))
        if seen:
            per_sequence.append(total)
    log_w = per_sequence if inputs.ess_level == "sequence" else per_token
    unit = "sequences" if inputs.ess_level == "sequence" else "tokens"
    floor = 2 if inputs.ess_level == "sequence" else floors.ess_min_tokens
    if len(log_w) < floor:
        return _undetermined(
            condition,
            (
                f"{len(log_w)} {unit} in this window carry both logprob streams, against a floor "
                f"of {floor}. Without both streams there is no importance ratio to compute an "
                f"effective sample size over; a converter with only one of them leaves the other "
                f"None rather than filling it in"
            ),
            {"n": len(log_w)},
        )
    x = np.asarray(log_w, dtype=float)
    # Kish's ESS is invariant to a constant rescaling of the weights, so subtracting the maximum
    # in the exponent is exact rather than an approximation and it keeps exp() off the overflow.
    w = np.exp(x - x.max())
    total_w = float(w.sum())
    ess = total_w * total_w / float((w * w).sum())
    fraction = ess / w.size
    return _reading(
        condition,
        fraction >= thresholds.ess_fraction_min,
        fraction,
        thresholds.ess_fraction_min,
        (
            f"Kish ESS {ess:.4g} over {w.size} {unit} ({fraction:.4g} of them), from "
            f"exp(logprobs_train - logprobs_sampling) on {turns_with_streams} turns. That gap is "
            f"policy drift and train-inference engine mismatch together (E6), and this estimator "
            f"cannot separate them"
        ),
        {"ess": ess, "n": int(w.size), "level": inputs.ess_level},
    )


# ---------------------------------------------------------------------------
# 10. LIGHT_TAILED
# ---------------------------------------------------------------------------


def _hill(sample: np.ndarray, k: int) -> float:
    """The Hill estimator on the top ``k`` order statistics of a positive sample."""
    x = np.sort(sample)[::-1]
    threshold = x[k]
    return float(np.mean(np.log(x[:k])) - math.log(threshold))


def _measure_light_tailed(
    trajectories: Sequence[Trajectory],
    thresholds: RegimeThresholds,
    floors: RegimeFloors,
    inputs: RegimeInputs,
) -> ConditionReading:
    """The MGF exists; the Hill estimate is below a stated bound.

    The sample defaults to the per-trajectory advantages, because the advantage is what multiplies
    the gradient and a heavy tail there is what makes a batch mean a statement about its largest
    member. Turn-level step scores are the fallback, and an explicit sample overrides both, since
    the quantity whose tail matters depends on what is being read.

    Two honest limits. The estimate is taken on absolute values at the 95th percentile, which
    measures the heavier of the two tails and not the right tail specifically; a signed quantity
    with one heavy side and one light side reports the heavy one, which is the conservative
    direction. And the exceedance counts a training record supplies are one to two orders of
    magnitude below what a defensible estimate needs, so what comes back at thirty exceedances is a
    flag
    and not a tail claim. The detail carries the count so a reader can see which one they have.
    """
    condition = RegimeCondition.LIGHT_TAILED
    if inputs.tail_sample is not None:
        raw = [float(v) for v in inputs.tail_sample]
        source = "the supplied sample"
    else:
        raw = [t.advantage for t in trajectories if t.advantage is not None]
        source = "per-trajectory advantages"
        if not raw:
            raw = [
                turn.step_score
                for t in trajectories
                for turn in t.turns
                if turn.step_score is not None
            ]
            source = "per-turn step scores"
    values = np.abs(np.asarray([v for v in raw if v is not None and math.isfinite(v)], dtype=float))
    n = values.size
    k = int(math.ceil(0.05 * n))
    if k < floors.hill_min_exceedances or n <= k:
        return _undetermined(
            condition,
            (
                f"{n} values from {source} give {k} exceedances at q = 0.95, against a floor of "
                f"{floors.hill_min_exceedances}. A defensible tail estimate needs "
                f"roughly 1,570 exceedances; below the floor this reports nothing rather than a "
                f"number with no support under it"
            ),
            {"n": n, "k": k},
        )
    ordered = np.sort(values)[::-1]
    if ordered[k] <= 0 or np.unique(ordered[: k + 1]).size < 2:
        return _undetermined(
            condition,
            (
                f"the {k + 1} largest values from {source} are not distinct and positive, so there "
                f"is no tail to fit: the Hill estimator takes log ratios against the threshold "
                f"order statistic and it is {ordered[k]:.4g}"
            ),
            {"n": n, "k": k},
        )
    xi = _hill(values, k)
    return _reading(
        condition,
        xi <= thresholds.hill_max,
        xi,
        thresholds.hill_max,
        (
            f"Hill estimate {xi:.4g} on {k} exceedances of {n} absolute values from {source}. At "
            f"this exceedance count it is a flag rather than a tail claim, and it measures the "
            f"heavier of the two tails because the sample is taken in absolute value"
        ),
        {"xi": xi, "k": k, "n": n},
    )


# ---------------------------------------------------------------------------
# 11. SCALAR_REPRESENTABLE
# ---------------------------------------------------------------------------


def _measure_scalar_representable(
    groups: Sequence[Group], thresholds: RegimeThresholds
) -> ConditionReading:
    """Curl mass below a stated bound.

    `GroupStats.curl_mass` is populated only where the group is a k-wise comparison, which is the
    only place the question is live: a group of independently scored rollouts has no comparison
    structure to be cyclic. So a window with no curl mass anywhere is not a window with zero curl
    mass, it is a window where the question was not asked, and the answer is `None`.

    The maximum over groups rather than the mean. Curl mass is an energy share and a single group
    whose comparisons are cyclic makes the scalar approximation wrong on that group, whatever the
    other groups did; averaging it away is how a local failure becomes invisible. B1 is the
    instrument that estimates this properly, with its four nulls; this reads what the record
    already carries.
    """
    condition = RegimeCondition.SCALAR_REPRESENTABLE
    values = [
        (g.group_stats.curl_mass, g.id) for g in groups if g.group_stats.curl_mass is not None
    ]
    if not values:
        return _undetermined(
            condition,
            (
                f"none of the {len(groups)} groups in this window records a curl mass. It is "
                f"populated only for k-wise comparison groups, so this is the question not having "
                f"been asked rather than the answer being zero. B1 estimates it"
            ),
        )
    worst, where = max(values, key=lambda pair: pair[0])
    return _reading(
        condition,
        float(worst) <= thresholds.curl_mass_max,
        float(worst),
        thresholds.curl_mass_max,
        (
            f"worst curl mass {worst:.4g} of {len(values)} comparison groups, in group {where}. "
            f"Reported as the maximum and not the mean: a scalar potential that fails on one group "
            f"has failed on it"
        ),
        {"curl_mass": float(worst), "n_groups": len(values)},
    )


# ---------------------------------------------------------------------------
# 12. MASK_STABLE
# ---------------------------------------------------------------------------


def _turn_mask_state(turn: Turn) -> str:
    """One turn's mask state, through `record.turns.mask_policy_signature`.

    Called one turn at a time so the record package keeps ownership of the state vocabulary. On a
    single turn the signature is exactly ``role=state``, so the state is the part after the equals
    sign and no real parsing happens. Calling it once over the whole window instead would return a
    role's states joined by pipes, and a per-role set is what the verdict needs.
    """
    return mask_policy_signature([turn]).split("=", 1)[1]


def _measure_mask_stable(
    trajectories: Sequence[Trajectory], groups: Sequence[Group]
) -> ConditionReading:
    """The loss-mask policy is unchanged across the window.

    Measured per role rather than per trajectory, which matters more than it sounds. A trajectory
    whose masks were recorded and one whose masks were not have different signatures for a reason
    that has nothing to do with the mask policy, so comparing whole signatures would report a
    policy change every time a converter recorded masks on some rollouts and not others. Comparing
    the set of *recorded* states per role is immune to that: unrecorded contributes nothing, and
    two different recorded states for one role is a real change.

    The estimator's own `loss_mask_policy` string is the second source and is used when it says
    something. Its default is the literal `"unknown"`, which is a placeholder rather than a policy
    name, so a window where every group says `"unknown"` is a window where nobody wrote the policy
    down and the answer is `None`.
    """
    condition = RegimeCondition.MASK_STABLE
    per_role: dict[str, set[str]] = {}
    for t in trajectories:
        for turn in t.turns:
            state = _turn_mask_state(turn)
            if state != "unrecorded":
                per_role.setdefault(turn.role, set()).add(state)
    declared = {
        g.estimator.loss_mask_policy
        for g in groups
        if g.estimator.loss_mask_policy and g.estimator.loss_mask_policy != "unknown"
    }
    measured_max = max((len(states) for states in per_role.values()), default=0)
    if measured_max == 0 and not declared:
        return _undetermined(
            condition,
            (
                f"no turn in this window records a loss mask and no group names a loss-mask policy "
                f"other than 'unknown', over {len(trajectories)} trajectories and {len(groups)} "
                f"groups. Nothing here can tell whether the policy changed"
            ),
        )
    distinct = max(measured_max, len(declared))
    unstable_roles = sorted(r for r, states in per_role.items() if len(states) > 1)
    detail_parts = []
    if per_role:
        detail_parts.append(
            "measured mask states "
            + ", ".join(f"{r}={'|'.join(sorted(s))}" for r, s in sorted(per_role.items()))
        )
    if declared:
        detail_parts.append(f"declared policies {sorted(declared)}")
    if unstable_roles:
        detail_parts.append(f"roles masked two ways in one window: {', '.join(unstable_roles)}")
    return _reading(
        condition,
        distinct <= 1,
        float(distinct),
        1.0,
        "; ".join(detail_parts),
        {"distinct": distinct, "roles": sorted(per_role)},
    )


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


def measure_regime(
    run: Run,
    *,
    step: int | None = None,
    window: Window | None = None,
    context: int = 0,
    thresholds: RegimeThresholds | None = None,
    floors: RegimeFloors | None = None,
    inputs: RegimeInputs | None = None,
    cross_check: bool = True,
) -> RegimeReading:
    """The twelve regime conditions, measured over one window of one run.

    ``step=k`` measures at one step, ``window=(lo, hi)`` over a half-open span, and neither
    measures over the whole run. ``context`` widens whichever was asked for on both sides, which is
    what makes a per-step reading able to answer the four conditions that are statements about
    change: at ``context=0`` those return `None` saying so rather than passing vacuously.

    All twelve conditions are always present in the returned mapping, including the ones that could
    not be determined. That is a decision with consequences downstream and it is deliberate:
    `EnvelopeSpec.classify` and the capability report both treat a condition *absent* from a
    reading as never measured and one *present* with `holds=None` as measured and indeterminate, so
    including everything means an instrument requiring `LINEAR_RESPONSE` is refused rather than run
    with its envelope unchecked. That is the required behaviour ("unknown is not a
    pass") and it is stricter than what a partial reading would produce.

    ``cross_check`` folds the run's own `RegimeDeclaration` into the detail of any condition where
    the declaration and the measurement disagree. The declaration never becomes a verdict. A run
    that declared `STATIONARY_GRADER` and measures otherwise is a finding, and it is invisible if
    only one of the two is carried.
    """
    thresholds = thresholds or RegimeThresholds()
    floors = floors or RegimeFloors()
    inputs = inputs or RegimeInputs()

    lo, hi = _window_of(run, step, window, context)
    steps = tuple(sorted(run.steps.slice(lo, hi), key=lambda s: s.index))
    indices = run.steps.indices
    if not steps:
        return _with_declaration(run, _empty_window_reading(lo, hi, indices))

    early_hi = min(indices) + floors.ar_early_steps
    early = tuple(sorted(run.steps.slice(min(indices), early_hi), key=lambda s: s.index))
    trajectories = _trajectories(steps)
    groups = _groups(steps)

    conditions = [
        _measure_quasi_static(steps, thresholds, floors, inputs, early),
        _measure_linear_response(thresholds, inputs),
        _measure_group_nondegenerate(groups, thresholds),
        _measure_near_policy(steps, trajectories, thresholds),
        _measure_stationary_grader(steps, thresholds, inputs),
        _measure_exogenous_curriculum(steps, thresholds, floors),
        _measure_no_compaction(trajectories, inputs),
        _measure_above_lod(inputs),
        _measure_ess_adequate(trajectories, thresholds, floors, inputs),
        _measure_light_tailed(trajectories, thresholds, floors, inputs),
        _measure_scalar_representable(groups, thresholds),
        _measure_mask_stable(trajectories, groups),
        _measure_design_crossed(),
    ]
    reading = RegimeReading(conditions={c.condition: c for c in conditions})
    return _with_declaration(run, reading) if cross_check else reading


def _empty_window_reading(lo: int, hi: int, indices: Sequence[int]) -> RegimeReading:
    """Every condition undetermined, with the window that was asked for and the one that exists."""
    have = f"{min(indices)} to {max(indices)}" if indices else "no steps at all"
    detail = (
        f"the window [{lo}, {hi}) contains no recorded steps; the record holds {have}. Nothing "
        f"about the regime is determined here, and none of it is satisfied either"
    )
    return RegimeReading(
        conditions={c: _undetermined(c, detail, {"window": [lo, hi]}) for c in RegimeCondition}
    )


def declaration_disagreements(
    run: Run, reading: RegimeReading
) -> dict[RegimeCondition, tuple[bool, bool | None]]:
    """Where the run's declaration and this measurement differ, as ``(declared, measured)``.

    A thin pass through `RegimeDeclaration.disagreements`, which already counts a measurement of
    `None` as a disagreement with any declaration. That is the right rule: "we said it holds and
    nobody could check" is exactly the state the declaration exists to make visible.
    """
    return run.regime.disagreements(reading)


def _measure_design_crossed() -> ConditionReading:
    """Never determinable from a training record, and that is a measurement rather than a gap.

    `DESIGN_CROSSED` is a property of a **measurement design**: whether every subject was scored by
    every rater at every occasion, which is the design A2's expected-mean-square inversion is
    derived for. A training record carries rollouts and rewards; it does not carry a gauge study, so
    there is nothing here to compute it from and there never will be.

    It returns `None` rather than being left out of the reading, and the difference matters. A
    condition absent from a `RegimeReading` reads as unchecked, which travels onto an instrument as
    "nobody looked". A condition present with `holds=None` says the measurement was attempted here
    and this is the wrong place to attempt it, and it names the place that is. That is what the
    third state exists for, and it is what stops a caller reading silence as a pass.
    """
    return _undetermined(
        RegimeCondition.DESIGN_CROSSED,
        "a crossed design is a property of a gauge study and not of a training run, so no record "
        "determines it. A2 measures it directly from the study it is given",
    )


def _with_declaration(run: Run, reading: RegimeReading) -> RegimeReading:
    """Annotate the details of conditions the run declared and the measurement contradicts."""
    disagreements = declaration_disagreements(run, reading)
    if not disagreements:
        return reading
    conditions = dict(reading.conditions)
    for condition, (declared, measured) in disagreements.items():
        current = conditions.get(condition)
        if current is None:
            continue
        note = run.regime.notes.get(condition, "")
        said = "true" if declared else "false"
        got = "could not be determined" if measured is None else str(bool(measured)).lower()
        extra = (
            f"the run declared this {said}"
            + (f" ({note})" if note else "")
            + f" and it {got}"
            + (
                ""
                if measured is not None
                else ", which is a disagreement rather than a match: an unchecked precondition is "
                "not a satisfied one"
            )
        )
        conditions[condition] = replace(
            current, detail=f"{current.detail}; {extra}" if current.detail else extra
        )
    return RegimeReading(conditions=conditions)


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

#: H5's own envelope. It cannot require regime conditions without requiring itself, and an empty
#: `requires` needs an explicit justification rather than silence, which is the lint rule doing its
#: job on the one instrument most tempted to skip it.
REGIME_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "this instrument measures the envelope, so requiring one would be circular. What would "
        "otherwise be its preconditions are carried inside the reading instead: every condition it "
        "cannot determine comes back as holds=None with the reason, rather than as a verdict "
        "taken under an assumption nobody checked."
    ),
)

_REGIME_ACCESS: AccessMatrix = {Component.RECORD: Access.RECORD}


class RunRegime(BaseObservable):
    """H5. The whole validity envelope, measured per step, consulted by every preflight.

    Reads a record and nothing else, so it runs at `RECORD` access with no grader calls and no GPU.
    That is what lets it sit in front of every other instrument: a preflight that had to spend
    money to find out whether it may run would not be a preflight.

    What it cannot do. It reports twelve conditions and it can determine at most ten of them from a
    record alone today, because `LINEAR_RESPONSE` needs a Lambda that F2 computes and `ABOVE_LOD`
    is a property of a reading rather than of a run. On a record with no check standard, no
    compaction events and one logprob stream, four more come back undetermined. That is the correct
    output and not a shortfall: the alternative is a verdict on a check that did not happen.
    """

    name = "RunRegime"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = None
    deviations = (
        "tau_relax is defined by perturb-and-hold, and the adiabaticity number is settled "
        "with a two-rate collapse test (G3). Rung 0 here fits the early AR(1) coefficient of a "
        "recorded series instead, which is the cheap route and is weaker in every "
        "way but cost.",
        "EXOGENOUS_CURRICULUM has no standard estimator. The one here is a "
        "rank-biserial test of task survival against score across consecutive steps, which detects "
        "score-responsive sampling and cannot prove its absence.",
    )

    # -- the instrument declarations ---------------------------------------
    quantity = "run.regime"
    requires: AccessMatrix = _REGIME_ACCESS
    substrates = frozenset(Substrate)
    #: Not PRE_RUN: there is no record before the run. Not DEPLOYED: only the artifact survives
    #: there, and this reads the process that produced it.
    phases = frozenset({Phase.IN_RUN, Phase.POST_RUN})
    envelope = REGIME_ENVELOPE
    #: `run.regime` is registered with invariance `none`, which resolves to the trivial group: no
    #: registered transformation acts on a set of verdicts. That makes the generated test pass
    #: vacuously, which is honest and is also weak. A non-vacuous check is available: the verdicts
    #: should be invariant under `reward.affine` wherever their thresholds are scale-free, which is
    #: a real property and a real test.
    invariance = "none"
    invariance_relation = INVARIANT
    baselines = ("baseline.run_declaration", "baseline.assume_all_hold")
    rung = 0

    def __init__(
        self,
        run: Run,
        *,
        step: int | None = None,
        window: Window | None = None,
        context: int = 0,
        thresholds: RegimeThresholds | None = None,
        floors: RegimeFloors | None = None,
        inputs: RegimeInputs | None = None,
    ) -> None:
        self.run = run
        self.step = step
        self.window = window
        self.context = context
        self.thresholds = thresholds or RegimeThresholds()
        self.floors = floors or RegimeFloors()
        self.inputs = inputs or RegimeInputs()
        self._computed: RegimeReading | None = None

    # -- the two instrument methods ----------------------------------------

    def compute(self) -> RegimeReading | Refusal:
        """The reading, or a refusal when there is no record in the window to read.

        An empty window is not a regime in which nothing holds, it is an absence of a run. Section
        6.2's `VOID` is the reason for exactly that: not readable, which is a different thing from
        a negative result.
        """
        lo, hi = _window_of(self.run, self.step, self.window, self.context)
        indices = self.run.steps.indices
        if not any(lo <= i < hi for i in indices):
            have = (
                f"steps {min(indices)} to {max(indices)}"
                if indices
                else "no steps at all, which makes this record unreadable rather than empty"
            )
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.VOID,
                detail=(
                    f"the window [{lo}, {hi}) contains no recorded steps of run {self.run.id}, "
                    f"which holds {have}."
                ),
                remedy=(
                    "Ask for a window inside the recorded range, or widen it with `context=`. If "
                    "the record genuinely has no steps, the run is void rather than negative and "
                    "the reason to record is that nothing was written, not that nothing happened."
                ),
                statistics={
                    "window": [lo, hi],
                    "recorded": len(indices),
                    "first": min(indices, default=None),
                    "last": max(indices, default=None),
                },
            )
        return measure_regime(
            self.run,
            step=self.step,
            window=self.window,
            context=self.context,
            thresholds=self.thresholds,
            floors=self.floors,
            inputs=self.inputs,
        )

    def estimate(self, ctx: Context) -> Reading:
        """Preflight, compute, refuse or emit. Never a bare reading, never a silent zero."""
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
        return ctx.emit(self.payload(out))

    # -- what it emits ------------------------------------------------------

    def payload(self, reading: RegimeReading) -> dict[str, Any]:
        """The Evidence value: the twelve, the counts, and what the two baselines got right."""
        holds = [c.holds for c in reading.conditions.values()]
        lo, hi = _window_of(self.run, self.step, self.window, self.context)
        return {
            "conditions": encode_regime_reading(reading),
            "n_hold": sum(1 for h in holds if h is True),
            "n_fail": sum(1 for h in holds if h is False),
            "n_undetermined": sum(1 for h in holds if h is None),
            "window": [lo, hi],
            "baselines": self.baseline_scores(reading),
        }

    def baseline_scores(self, reading: RegimeReading) -> dict[str, Any]:
        """What the two dumb comparators get right, on the conditions that were determined.

        `baseline.run_declaration` is the run's own claim about itself, which costs nothing and is
        the comparator this instrument has to beat to be worth running. `baseline.assume_all_hold`
        is what an uninstrumented pipeline does implicitly, which is to proceed as though every
        precondition were satisfied. Both are scored only over the conditions with a verdict,
        because scoring a comparator on a condition nobody determined would credit it for a coin
        flip.
        """
        determined = {c: r.holds for c, r in reading.conditions.items() if r.holds is not None}
        declared = self.run.regime.declared
        overlap = [c for c in determined if c in declared]
        return {
            "n_determined": len(determined),
            "run_declaration_agreement": (
                sum(1 for c in overlap if declared[c] is determined[c]) / len(overlap)
                if overlap
                else None
            ),
            "run_declaration_n": len(overlap),
            "assume_all_hold_agreement": (
                sum(1 for h in determined.values() if h is True) / len(determined)
                if determined
                else None
            ),
        }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _register() -> None:
    """One rung for `run.regime`, so the capability report knows the estimator exists.

    `spec/QUANTITIES.yaml` gives this quantity one rung, and this is it. A rung 1 would be the
    perturb-and-hold relaxation time and the two-rate collapse test, which need two
    runs rather than a better reading of one, so they are a different instrument (H1) rather than a
    higher rung of this one.
    """
    register_estimator(
        EstimatorEntry(
            quantity="run.regime",
            impl="run.regime.record",
            requires=_REGIME_ACCESS,
            envelope=REGIME_ENVELOPE,
            rung=0,
            bias=BiasStatement(
                direction="unknown",
                why=(
                    "twelve conditions with twelve estimators and no single direction of error. "
                    "Each condition's own bias is in its detail; what is uniform is the direction "
                    "the undetermined cases err in, which is toward refusing rather than toward "
                    "licensing."
                ),
            ),
            cost=CostModel(note="one pass over the window's steps; no grader calls, no GPU"),
            phases=frozenset({Phase.IN_RUN, Phase.POST_RUN}),
            run=None,
        )
    )


_register()


__all__ = [
    "MEASURED_BY",
    "REGIME_ENVELOPE",
    "UNREGISTERED_MEASURED_BY",
    "RegimeFloors",
    "RegimeInputs",
    "RegimeThresholds",
    "RunRegime",
    "Window",
    "declaration_disagreements",
    "measure_regime",
]

"""E4, amplifier safety: is this reward component safe to add.

    AmplifierSafety(c) = Var_group(r_c | all-fail groups) / Var_group(r_c | mixed groups)

Near zero is safe. Order one or above is a live amplifier.

**Mixed means mixed, and it is a three-way partition rather than a complement.** Against a stated
failure floor a group is all-fail, all-pass or mixed, and the denominator is the mixed ones only.
Reading the denominator as "not all-fail" pools all-pass groups into it, and an all-pass group is
not a small mixed group: the task component cannot vary in it either, so whatever the auxiliary is
doing there is measured against a population the ratio is not about. Measured on the 24-step GRPO
record this instrument was built against, where 2 of 48 groups are all-pass: the pooled denominator
moved `length_reward` from 1.864 to 1.932 and `format_reward` from 0.929 to 0.906. The direction is
not signed in general, which is worth saying because the natural guess is that saturation always
suppresses the denominator: it does when the component has run out of scale, and `format_reward`
here goes the other way. What is signed is that the denominator is not the population the formula
names. `groups.partition_by_floor` is the partition and `n_allpass_groups` is on every reading.

**The mechanism, so this file explains itself.** GRPO does not use the group mean, it uses the group
z-score, ``A_i = (r_i - mean_j r_j) / (std_j(r_j) + eps)``. Three things follow.

*Scale invariance destroys dose control.* The normalised advantage is invariant to any positive
rescaling of ``r``, so a shaping coefficient ``lambda`` on an auxiliary term does nothing once the
group is z-scored, and annealing it cannot help either: the dose disappears mathematically.
Potential-based shaping's policy-invariance guarantee does not survive this, because the failure
enters through the advantage estimator one level below returns, which is not where that guarantee
lives. Anyone who has tuned a shaping weight and found it had no effect has met this. (The
invariance is exact only in the limit, and the size of the departure is a ratio rather than an
absolute. Under ``r -> a*r + b`` the numerator scales by ``a`` while the denominator goes to
``a*std + eps``, so the advantage moves by a relative

    eps * (a - 1) / (a * std + eps)

which for ``eps << a*std`` is ``(eps/std) * (a - 1)/a``. It is a function of ``eps/std``, not of
``eps``, so any number quoted for it pins an unstated group standard deviation. At ``a = 2`` and
``eps = 1e-8`` the departure is 1.0e-6 at ``std = 0.005``, 1.0e-7 at ``std = 0.05``, 1.0e-8 at
``std = 0.5`` and 1.0e-9 at ``std = 5``: three orders of magnitude across a range a real reward
group covers. The familiar "about 1e-7 at ``eps = 1e-8``" is the ``std = 0.05`` row of that table.)

*All-fail groups amplify.* When every rollout in a group fails the task, the only variance left is
whatever auxiliary signal is still moving, and the z-score rescales it to full magnitude. The
published measurement is two orders of magnitude, and the demonstration is a dense per-step
prediction reward collapsing GRPO agents on ALFWorld at three model sizes: prediction accuracy goes
to 1.000 while task success goes to zero, and removing the standard-deviation normalisation
recovers most of the baseline.

*Therefore "is this reward signal safe to add" has a formal answer and it is not about magnitude.*
A signal is dangerous if its variance **persists** during the all-fail-dominated phase. A signal
whose variance vanishes at mastery is amplifier-safe. That is a condition on the variance
trajectory, not on the reward's scale or its bounds, and it is directly measurable from a record.

**The mandatory baseline is the component's absolute magnitude**, which is what people currently
look at and which is uninformative for exactly the reason above. Both are rendered on every reading
and the contrast between the two rankings is the instrument's whole argument.

**Two records this instrument declines to answer on, and both are real.** On `verifiers`,
`score_group` mean-centres with no standard-deviation division (`rubrics/rubric.py:406-409`), so the
amplification mechanism is absent and there is no ratio worth reporting; what comes back is that
statement plus the magnitudes, which on a mean-centred estimator are the right diagnostic because
dose control survives there. And on a window with no mixed groups the denominator has no population,
which happens on any early-training record where the policy solves nothing, so the ratio is refused
rather than divided by an empty set.

Kill condition, from the catalogue record: if amplifier safety does not predict collapse on three
held-out runs, it is a description rather than a forecast. That is registered as a frozen prediction
in `tests/acceptance/test_w3_5_estimator.py` rather than resolved after the fact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np

from reward_lens.core.envelope import (
    ConditionReading,
    EnvelopeSpec,
    RegimeCondition,
    RegimeReading,
)
from reward_lens.core.evidence import register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BaselineID, BiasStatement
from reward_lens.core.reading import Refusal, RefusalReason, bounded_refusal
from reward_lens.core.types import Access, Capability, Component, GaugeStatus, Phase
from reward_lens.measure.base import Context, PreflightResult
from reward_lens.measure.estimator._base import EstimatorInstrument
from reward_lens.measure.estimator.groups import (
    FailureFloor,
    component_scores,
    partition_by_floor,
)
from reward_lens.measure.estimator.spec import (
    ALL_SUBSTRATES,
    RECORD_ACCESS,
    collect_specs,
    iter_groups,
    iter_steps,
)
from reward_lens.record.schema import Group, Run, Step
from reward_lens.record.scores import leaves

#: The catalogue's mandatory baseline for E4, spelled as two numbers because "magnitude" is
#: ambiguous and both readings of it are what people quote: the mean absolute value of the
#: component, and its overall standard deviation across the whole window.
AMPLIFIER_BASELINES: tuple[BaselineID, ...] = (
    "baseline.component_magnitude",
    "baseline.component_sd",
)

#: Above this the component is a live amplifier. **Stated**: "near zero is safe; order
#: one or above is a live amplifier". Reported alongside every reading so the verdict is auditable
#: rather than asserted, and it is not an envelope threshold: it labels a reading, it does not gate
#: one.
LIVE_AMPLIFIER_AT = 1.0

#: Below this the component's variance has effectively vanished in the all-fail phase, which is
#: what "amplifier-safe" means. **Chosen: 0.1**, on the argument that a component carrying a tenth
#: of its mixed-group variance into the all-fail phase is contributing under a third of its
#: mixed-group standard deviation, which is the scale the z-score works in.
SAFE_BELOW = 0.1

#: The all-fail fraction at which the phase counts as all-fail-dominated, for rung 2. **Chosen:
#: 0.5**, on the argument that the condition is about the phase in which all-fail groups
#: dominate the update, and a majority of groups is the weakest defensible reading of dominate.
DOMINANCE_AT = 0.5

#: The fraction of its own early-training all-fail variance a component must retain for rung 2 to
#: call the variance persistent. **Chosen: 0.5**, on the argument that a component that has lost
#: half its variance is on a decay path and the safe case is the one where the variance
#: vanishes.
PERSISTENCE_AT = 0.5

#: The degenerate-group fraction above which `GROUP_NONDEGENERATE` fails. **Stated** by the
#: capability report, which prints "GROUP_NONDEGENERATE ok degenerate group fraction 0.04 (threshold
#: 0.20)". This is the same number `measure.rate.regime.RegimeThresholds` carries and it is
#: restated here rather than imported, because importing it would make E4's envelope depend on
#: another package's default and a threshold that moves silently is worse than one that is
#: duplicated visibly.
DEGENERATE_FRACTION_MAX = 0.20

#: `GROUP_NONDEGENERATE`, **measured, not assumed**. The catalogue merge dropped that qualifier,
#: so it is restored here where it can be enforced: `preflight` measures the condition from the
#: groups it holds and uses its own measurement in preference to any verdict the caller supplied.
AMPLIFIER_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.GROUP_NONDEGENERATE}),
    measured_by={RegimeCondition.GROUP_NONDEGENERATE: "estimator.degenerate_fraction"},
    on_violation="refuse",
)

AMPLIFIER_BIAS = BiasStatement(
    direction="unknown",
    why=(
        "the ratio is a ratio of two pooled within-group variances, each unbiased for its own "
        "population, and the ratio of two unbiased estimators is not unbiased. The direction "
        "depends on the relative group counts and is not signed here. What is signed is the "
        "small-sample behaviour: with few all-fail groups the numerator is noisy and the ratio is "
        "heavy-tailed, which is why the reading carries the group counts beside it."
    ),
)


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


def pooled_within_variance(
    per_group: Sequence[Sequence[float | None]],
) -> tuple[float, int, int]:
    """Degrees-of-freedom-weighted pooled within-group variance, and what went into it.

    ``sum_g (n_g - 1) s_g^2 / sum_g (n_g - 1)`` with ``s_g^2`` the unbiased within-group variance.
    Pooling this way rather than averaging the per-group variances is what makes the estimate right
    when the groups have different K, which they do the moment a grader abstains on one rollout.

    Returns the variance, the number of groups that contributed, and the total degrees of freedom.
    A group with fewer than two present values contributes nothing and is counted out, because a
    single observation has no within-group spread and averaging a zero in from it would understate
    the variance rather than leave it alone.
    """
    num = 0.0
    dof = 0
    used = 0
    for values in per_group:
        arr = np.asarray([v for v in values if v is not None], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            continue
        num += (arr.size - 1) * float(arr.var(ddof=1))
        dof += arr.size - 1
        used += 1
    return (num / dof if dof else math.nan), used, dof


def component_names(groups: Sequence[Group]) -> list[str]:
    """Every leaf name that appears on every rollout of every group, in first-seen order.

    Restricted to leaves present everywhere on purpose. A component scored on some rollouts and not
    others gives a within-group variance over a different population per group, and pooling those
    is the unit error this layer exists to make impossible.
    """
    order: list[str] = []
    common: set[str] | None = None
    for group in groups:
        for traj in group.trajectories:
            if traj.scores is None:
                return []
            names = {leaf.name for leaf in leaves(traj.scores)}
            for leaf in leaves(traj.scores):
                if leaf.name not in order:
                    order.append(leaf.name)
            common = names if common is None else (common & names)
    return [n for n in order if common and n in common]


@dataclass(frozen=True)
class ComponentSafety:
    """One component's ratio, the two variances behind it, and the baseline beside it.

    ``var_allpass`` is not part of the ratio. It is carried because the all-pass groups are the ones
    that used to sit in the denominator, and a reader deciding whether the correction mattered on
    their record wants to see the population that left rather than take it on trust.
    """

    component: str
    safety: float
    var_allfail: float
    var_mixed: float
    n_allfail_groups: int
    n_mixed_groups: int
    dof_allfail: int
    dof_mixed: int
    magnitude: float
    sd: float
    var_allpass: float = math.nan
    n_allpass_groups: int = 0
    dof_allpass: int = 0

    @property
    def verdict(self) -> str:
        if not math.isfinite(self.safety):
            return "undefined"
        if self.safety >= LIVE_AMPLIFIER_AT:
            return "live amplifier"
        if self.safety <= SAFE_BELOW:
            return "amplifier-safe"
        return "watch"


@dataclass(frozen=True)
class SafetyTrajectory:
    """Rung 1: the two variances and the ratio, per component per step, with the all-fail fraction.

    ``steps`` and ``allfail_fraction`` are one entry per step. The three variance fields are one
    **series per component**, in ``names`` order, so ``var_allfail[c][s]`` is component ``c`` at
    step ``s``. They were declared `tuple[float, ...]` and built as nested tuples, and three
    `type: ignore[arg-type]` comments carrying the wrong error code suppressed nothing and hid the
    mismatch. The construction and the only consumer both already treated them as nested, so this
    corrects the declaration rather than the behaviour.

    ``names`` is on the object because without it the three parallel series are keyed by an
    ordering the object does not record, and a caller that zips them against a `names` obtained
    somewhere else is one refactor away from attributing a component's variance to its neighbour.
    """

    steps: tuple[int, ...]
    var_allfail: tuple[tuple[float, ...], ...]
    var_mixed: tuple[tuple[float, ...], ...]
    safety: tuple[tuple[float, ...], ...]
    allfail_fraction: tuple[float, ...]
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.names and not all(
            len(series) == len(self.names)
            for series in (self.var_allfail, self.var_mixed, self.safety)
        ):
            raise ValueError(
                f"{len(self.names)} component names against "
                f"{len(self.var_allfail)}/{len(self.var_mixed)}/{len(self.safety)} series. A "
                f"trajectory whose series do not line up with its names attributes one "
                f"component's variance to another."
            )


@register_payload
@dataclass
class AmplifierReading:
    """Amplifier safety for every component, with the magnitude ranking it is meant to displace."""

    #: Component name to its ratio. Empty when the mechanism is absent, which is a statement rather
    #: than a gap: on a mean-centred estimator with no standard-deviation division there is no
    #: amplification to measure.
    safety: dict[str, float] = field(default_factory=dict)
    detail: dict[str, dict[str, float]] = field(default_factory=dict)
    verdicts: dict[str, str] = field(default_factory=dict)
    #: Components ordered most dangerous first, by the ratio.
    ranking: list[str] = field(default_factory=list)
    #: The same components ordered by the baseline, largest magnitude first. Rendered beside the
    #: ranking because the contrast between the two is what the instrument is for.
    magnitude_ranking: list[str] = field(default_factory=list)
    rankings_agree: bool = True
    n_groups: int = 0
    n_allfail_groups: int = 0
    n_mixed_groups: int = 0
    #: Groups where no rollout failed. Not in either side of the ratio, and counted here because
    #: they used to be in the denominator. `n_allfail + n_mixed + n_allpass + n_unscored = n_groups`.
    n_allpass_groups: int = 0
    #: Groups where every rollout abstained on the floor's component, or where the component is not
    #: on every rollout's tree. Neither population, and not a group with no spread.
    n_unscored_groups: int = 0
    mechanism_present: bool = True
    baselines: dict[str, float] = field(default_factory=dict)
    #: Rung 1, when a per-step series is available.
    trajectory: dict[str, Any] = field(default_factory=dict)
    #: Rung 2, when a collapse step could be projected.
    predicted_collapse_step: float | None = None
    predicted_collapse_ci: tuple[float, float] | None = None
    collapse_note: str = ""
    #: Whether `GROUP_NONDEGENERATE` was measured here rather than accepted from the caller, and
    #: whether the caller's supplied verdict disagreed with the measurement.
    envelope_measured: bool = False
    envelope_statistic: float = math.nan
    envelope_disagreed: bool = False
    says: str = ""

    def render(self) -> str:
        return self.says


def _says(reading: AmplifierReading) -> str:
    if not reading.mechanism_present:
        return (
            "This record's estimator mean-centres and does not divide by the group standard "
            "deviation, so the amplification E4 measures is absent and there is no ratio to "
            "report. Dose control survives here, which means the component's own magnitude is the "
            "diagnostic: "
            + ", ".join(
                f"{name} {reading.baselines.get('magnitude/' + name, float('nan')):.4g}"
                for name in reading.magnitude_ranking
            )
            + "."
        )
    lines = []
    for name in reading.ranking:
        s = reading.safety[name]
        mag = reading.detail[name]["magnitude"]
        lines.append(
            f"{name}: AmplifierSafety = {s:.3g} ({reading.verdicts[name]}), mean |r| = {mag:.3g}"
        )
    census = (
        f"{reading.n_allfail_groups} all-fail groups against {reading.n_mixed_groups} mixed, "
        f"of {reading.n_groups}"
    )
    if reading.n_allpass_groups:
        census += (
            f", with {reading.n_allpass_groups} all-pass held out of the denominator because a "
            f"group nobody failed is not a mixed group"
        )
    if reading.n_unscored_groups:
        census += f" and {reading.n_unscored_groups} unscored on the floor's component"
    head = census + ". " + "; ".join(lines) + "."
    if not reading.rankings_agree:
        head += (
            f" The safety ranking is {' > '.join(reading.ranking)} and the magnitude ranking is "
            f"{' > '.join(reading.magnitude_ranking)}. They disagree, which is the point: "
            f"magnitude is what people look at and the z-score removes it."
        )
    if reading.predicted_collapse_step is not None:
        head += f" Projected all-fail dominance at step {reading.predicted_collapse_step:.0f}"
        if reading.predicted_collapse_ci is not None:
            lo, hi = reading.predicted_collapse_ci
            head += f", Fieller interval [{lo:.0f}, {hi:.0f}] at two standard errors"
        head += "."
    # The note travels either way. It used to be printed only when there was no projected step, so
    # the case it exists for, a point projection whose interval is unbounded, was the one case it
    # never reached the reader in.
    if reading.collapse_note:
        head += " " + reading.collapse_note
    return head


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(q / (1.0 - q))


#: Standard errors either side of the fit in the projected-crossing interval. Two rather than a
#: Student t quantile, which is what the first version used and what keeps the boundary of the
#: unbounded case where it has always been. At the four-step floor this function accepts, dof is 2
#: and the two-sided 95% t is 4.30, so on the shortest windows the interval is narrower than a 95%
#: interval and `_project_collapse` says so rather than labelling it 95%.
_CROSSING_Z = 2.0


def _fieller_crossing(
    x: np.ndarray,
    y: np.ndarray,
    slope: float,
    intercept: float,
    target: float,
    z: float = _CROSSING_Z,
) -> tuple[tuple[float, float] | None, str]:
    """Fieller's interval for the x at which the fitted line reaches ``target``.

    The crossing is a ratio of two estimated quantities, ``(target - intercept) / slope``, and the
    interval for a ratio is not the pair of ratios you get from the endpoints of one of them. The
    first version of this perturbed the slope by two standard errors and held the intercept fixed,
    which is not a confidence statement about anything: the intercept is defined at ``x = 0``, so
    the answer depended on where the step index happened to start. Measured on an eight-step window
    of the same fractions, the interval was 0.93 steps wide with the window numbered from 0 and
    58.34 steps wide with the identical window numbered from 400, a 62.4-fold inflation produced by
    renumbering the x-axis and nothing else.

    Fieller's set is every ``x0`` at which the fitted line's distance from ``target`` sits within
    ``z`` standard errors of the fitted value at ``x0``. Written in deviations from the mean step it
    mentions the intercept nowhere, so it slides with the step index and does not stretch.

    Returns ``None`` when the set is unbounded, which happens exactly when the slope is not itself
    ``z`` standard errors clear of zero. That is not a computation that failed. A line whose slope
    could be zero puts no upper bound on where it crosses anything, and the honest interval really
    is the half-line. The note says which of the two it was.

    Reference: Fieller (1954), *JRSS B* 16(2), pp. 175-185, and the inverse-regression form in
    Draper and Smith, *Applied Regression Analysis* 3rd ed., section 3.2.
    """
    n = int(x.size)
    dof = n - 2
    if dof < 1:
        return None, "a crossing interval needs at least three steps."
    xbar = float(x.mean())
    ybar = float(y.mean())
    sxx = float(((x - xbar) ** 2).sum())
    if sxx <= 0.0:
        return None, "every step in this window carries the same index, so no line is identified."
    resid = y - (slope * x + intercept)
    s2 = float((resid**2).sum()) / dof
    if s2 <= 0.0:
        # A perfect straight line through the logits. The interval is a point, and saying so is
        # better than returning None: nothing failed, the residual variance really is zero.
        at = xbar + (target - ybar) / slope
        return (at, at), (
            "the logit fit is exact over this window, so the crossing interval has zero width. "
            "That reflects a residual variance of exactly zero and not a well-determined trend."
        )
    a = slope * slope - z * z * s2 / sxx
    if a <= 0.0:
        t_stat = abs(slope) / math.sqrt(s2 / sxx)
        return None, (
            f"no bounded crossing interval: the logit slope is {t_stat:.2f} standard errors from "
            f"zero, against the {z:.1f} this interval needs, so Fieller's set is unbounded above. "
            f"The point projection stands; treat it as a direction rather than a date, and widen "
            f"the window or wait for more steps before quoting an interval."
        )
    d = target - ybar
    half = z * math.sqrt(s2 * (d * d / sxx + a / n))
    lo = xbar + (slope * d - half) / a
    hi = xbar + (slope * d + half) / a
    return (float(lo), float(hi)), ""


def _project_collapse(
    steps: np.ndarray, fractions: np.ndarray
) -> tuple[float | None, tuple[float, float] | None, str]:
    """The step at which the all-fail fraction is projected to cross `DOMINANCE_AT`.

    Fitted on the logit of the fraction, because the fraction is bounded in [0, 1] and a straight
    line on the raw scale predicts crossings above one. A non-increasing trend gets no crossing and
    says so: an all-fail fraction that is not rising is not on its way to dominance, and inventing a
    step for it would be an extrapolation past the data's own direction.

    The interval is Fieller's and is invariant to where the step index starts. The two standard
    errors it uses are two standard errors and not a 95% level: at the four-step floor the two-sided
    t on two degrees of freedom is 4.30, so a window that short buys a narrower interval than its
    label would suggest, and nothing here labels it 95%.
    """
    if steps.size < 4:
        return None, None, "fewer than four steps; no trend is fitted from three points."
    y = _logit(fractions)
    if not np.all(np.isfinite(y)):
        return None, None, "the all-fail fraction is degenerate at 0 or 1 across the window."
    x = steps.astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    if slope <= 0:
        return (
            None,
            None,
            f"the all-fail fraction is not rising over this window (logit slope {slope:.3g}), "
            f"so no dominance step is projected from it.",
        )
    target = _logit(np.array([DOMINANCE_AT]))[0]
    at = float((target - intercept) / slope)
    ci, note = _fieller_crossing(x, y, float(slope), float(intercept), float(target))
    return at, ci, note


def measure_amplifier_safety(
    subject: Run | Sequence[Step] | Step | Sequence[Group],
    *,
    floor: FailureFloor,
    components: Sequence[str] | None = None,
    trajectory: bool = False,
    instrument: str = "AmplifierSafety",
) -> AmplifierReading | Refusal:
    """The ratio per component, or the refusal that says which population is missing.

    Callable without a `Context` so a test can hand it an all-fail group whose amplification is
    analytic, and so the acceptance test can freeze a ranking before anything is observed.
    """
    groups = list(iter_groups(subject))
    if not groups:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail="this window contains no groups, so there is no group structure to partition",
            remedy="point this at a Run, a Step, or a sequence of Groups carrying at least one group.",
            statistics={"n_groups": 0},
        )

    specs = collect_specs(groups)
    if len(specs) > 1:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=(
                f"this window carries {len(specs)} distinct estimator specifications, so the "
                f"transform being measured changed inside it. A ratio pooled across two operators "
                f"is a number about neither."
            ),
            remedy=(
                "restrict the window to a span where E1 reports `stable`. If the estimator really "
                "did change mid-run, measure each span separately and report both."
            ),
            statistics={"n_specs": len(specs)},
        )
    spec = specs[0]

    names = list(components) if components is not None else component_names(groups)
    if not names:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=(
                "no reward component appears on every rollout of every group, so there is nothing "
                "whose within-group variance can be pooled. Either the trajectories carry no "
                "`ScoreTree`, or the tree's leaves differ between rollouts."
            ),
            remedy=(
                "record the per-leaf scores rather than the composed total. If the per-leaf scores "
                "were never written, they are gone: amplifier safety is a per-component question "
                "and a single total cannot answer it. Pass `components=` to name the leaves "
                "explicitly if they are present under names this cannot see."
            ),
            statistics={"n_groups": len(groups)},
        )

    phases = partition_by_floor(groups, floor)
    all_fail = [p == "all_fail" for p in phases]
    is_mixed = [p == "mixed" for p in phases]
    n_af = sum(all_fail)
    n_mixed = sum(is_mixed)
    n_ap = sum(1 for p in phases if p == "all_pass")
    n_un = sum(1 for p in phases if p == "unscored")

    # Both baselines are computable whatever happens to the ratio, and they are what the caller
    # gets back when the ratio is refused, so they are built first.
    magnitudes: dict[str, float] = {}
    sds: dict[str, float] = {}
    for name in names:
        vals: list[float] = []
        for group in groups:
            scored = component_scores(group, name)
            if scored is None:
                continue
            vals.extend(v for v in scored if v is not None and math.isfinite(v))
        arr = np.asarray(vals, dtype=float)
        magnitudes[name] = float(np.abs(arr).mean()) if arr.size else math.nan
        sds[name] = float(arr.std(ddof=1)) if arr.size > 1 else math.nan

    magnitude_ranking = sorted(
        names, key=lambda n: (-magnitudes[n] if math.isfinite(magnitudes[n]) else 0.0, n)
    )
    baselines = {
        "baseline.component_magnitude": float(
            np.nanmax([magnitudes[n] for n in names]) if names else math.nan
        ),
        "baseline.component_sd": float(np.nanmax([sds[n] for n in names]) if names else math.nan),
        **{f"magnitude/{n}": magnitudes[n] for n in names},
        **{f"sd/{n}": sds[n] for n in names},
    }

    if not spec.z_scored:
        reading = AmplifierReading(
            mechanism_present=False,
            n_groups=len(groups),
            n_allfail_groups=n_af,
            n_mixed_groups=n_mixed,
            n_allpass_groups=n_ap,
            n_unscored_groups=n_un,
            magnitude_ranking=magnitude_ranking,
            baselines=baselines,
        )
        reading.says = _says(reading)
        return bounded_refusal(
            instrument=instrument,
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=(
                f"the recorded estimator (family {spec.family!r}, group_centred="
                f"{spec.group_centred}, std_normalised={spec.std_normalised}) does not divide by "
                f"the group standard deviation, so the amplification amplifier safety measures is "
                f"absent from this record. A variance ratio computed here would be a number nobody "
                f"can act on."
            ),
            remedy=(
                "read the component magnitudes returned as the bound instead. Without the "
                "standard-deviation division a shaping coefficient still does what it says, so "
                "magnitude is the right diagnostic here and dose control is available to you. If "
                "the estimator does z-score and the record says otherwise, fix "
                "`EstimatorSpec.std_normalised` at the tap."
            ),
            bound=_bare_evidence(reading),
            family=spec.family,
            std_normalised=bool(spec.std_normalised),
        )

    if n_af == 0 or n_mixed == 0:
        missing = "all-fail" if n_af == 0 else "mixed"
        reading = AmplifierReading(
            n_groups=len(groups),
            n_allfail_groups=n_af,
            n_mixed_groups=n_mixed,
            n_allpass_groups=n_ap,
            n_unscored_groups=n_un,
            magnitude_ranking=magnitude_ranking,
            baselines=baselines,
        )
        reading.says = (
            f"{n_af} all-fail groups, {n_mixed} mixed, {n_ap} all-pass and {n_un} unscored in this "
            f"window, so one side of the ratio has no population."
        )
        allpass_note = (
            (
                f" {n_ap} of the {len(groups)} groups are all-pass. They are not mixed groups and "
                f"they are not in the denominator: the task component cannot vary in them either, "
                f"so a ratio taken against them is not the contrast this instrument names."
            )
            if n_ap
            else ""
        )
        return bounded_refusal(
            instrument=instrument,
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=(
                f"amplifier safety is a contrast between two populations of groups and this window "
                f"has {n_af} all-fail and {n_mixed} mixed, so the {missing} side is empty. A ratio "
                f"against an empty population is not a small sample, it is no sample."
                + allpass_note
            ),
            remedy=(
                "widen the window to a span that contains both, which for an early-training record "
                "means training until the policy solves something and for a late one means "
                "including the phase before mastery. If the failure floor is wrong, correct it: a "
                "floor set above every achievable score makes every group all-fail, and one set "
                "below every achievable score makes every group all-pass."
            ),
            bound=_bare_evidence(reading),
            n_allfail_groups=n_af,
            n_mixed_groups=n_mixed,
            n_allpass_groups=n_ap,
            n_unscored_groups=n_un,
            failure_at=floor.at,
            failure_component=floor.component,
        )

    safety: dict[str, float] = {}
    detail: dict[str, dict[str, float]] = {}
    verdicts: dict[str, str] = {}
    for name in names:
        af_vals = [
            component_scores(g, name) or [] for g, p in zip(groups, phases) if p == "all_fail"
        ]
        mx_vals = [component_scores(g, name) or [] for g, p in zip(groups, phases) if p == "mixed"]
        ap_vals = [
            component_scores(g, name) or [] for g, p in zip(groups, phases) if p == "all_pass"
        ]
        v_af, n_af_used, dof_af = pooled_within_variance(af_vals)
        v_mx, n_mx_used, dof_mx = pooled_within_variance(mx_vals)
        v_ap, n_ap_used, dof_ap = pooled_within_variance(ap_vals)
        ratio = v_af / v_mx if (math.isfinite(v_mx) and v_mx > 0.0) else math.inf
        entry = ComponentSafety(
            component=name,
            safety=ratio,
            var_allfail=v_af,
            var_mixed=v_mx,
            n_allfail_groups=n_af_used,
            n_mixed_groups=n_mx_used,
            dof_allfail=dof_af,
            dof_mixed=dof_mx,
            magnitude=magnitudes[name],
            sd=sds[name],
            var_allpass=v_ap,
            n_allpass_groups=n_ap_used,
            dof_allpass=dof_ap,
        )
        safety[name] = ratio
        verdicts[name] = entry.verdict
        detail[name] = {
            "var_allfail": v_af,
            "var_mixed": v_mx,
            "var_allpass": v_ap,
            "n_allfail_groups": float(n_af_used),
            "n_mixed_groups": float(n_mx_used),
            "n_allpass_groups": float(n_ap_used),
            "dof_allfail": float(dof_af),
            "dof_mixed": float(dof_mx),
            "dof_allpass": float(dof_ap),
            "magnitude": magnitudes[name],
            "sd": sds[name],
        }

    ranking = sorted(
        names,
        key=lambda n: (-safety[n] if math.isfinite(safety[n]) else -math.inf, n),
    )
    reading = AmplifierReading(
        safety=safety,
        detail=detail,
        verdicts=verdicts,
        ranking=ranking,
        magnitude_ranking=magnitude_ranking,
        rankings_agree=ranking == magnitude_ranking,
        n_groups=len(groups),
        n_allfail_groups=n_af,
        n_mixed_groups=n_mixed,
        n_allpass_groups=n_ap,
        n_unscored_groups=n_un,
        baselines=baselines,
    )

    if trajectory:
        traj = _trajectory(subject, floor=floor, names=names)
        if traj is not None:
            reading.trajectory = {
                "steps": list(traj.steps),
                "allfail_fraction": list(traj.allfail_fraction),
                **{f"var_allfail/{n}": list(v) for n, v in zip(traj.names, traj.var_allfail)},
                **{f"var_mixed/{n}": list(v) for n, v in zip(traj.names, traj.var_mixed)},
                **{f"safety/{n}": list(v) for n, v in zip(traj.names, traj.safety)},
            }
            at, ci, note = _project_collapse(
                np.asarray(traj.steps, dtype=float),
                np.asarray(traj.allfail_fraction, dtype=float),
            )
            reading.predicted_collapse_step = at
            reading.predicted_collapse_ci = ci
            reading.collapse_note = note

    reading.says = _says(reading)
    return reading


def _trajectory(
    subject: Run | Sequence[Step] | Step | Sequence[Group],
    *,
    floor: FailureFloor,
    names: Sequence[str],
) -> SafetyTrajectory | None:
    """Rung 1, or None when the subject carries no per-step series to build one from."""
    steps = list(iter_steps(subject))
    if len(steps) < 2:
        return None
    idx: list[int] = []
    fracs: list[float] = []
    per_name_af: dict[str, list[float]] = {n: [] for n in names}
    per_name_mx: dict[str, list[float]] = {n: [] for n in names}
    per_name_ratio: dict[str, list[float]] = {n: [] for n in names}
    for step in steps:
        groups = list(step.groups)
        if not groups:
            continue
        phases = partition_by_floor(groups, floor)
        idx.append(step.index)
        fracs.append(sum(1 for p in phases if p == "all_fail") / len(groups))
        for name in names:
            af = [
                component_scores(g, name) or [] for g, p in zip(groups, phases) if p == "all_fail"
            ]
            mx = [component_scores(g, name) or [] for g, p in zip(groups, phases) if p == "mixed"]
            v_af, _, _ = pooled_within_variance(af)
            v_mx, _, _ = pooled_within_variance(mx)
            per_name_af[name].append(v_af)
            per_name_mx[name].append(v_mx)
            per_name_ratio[name].append(
                v_af / v_mx if (math.isfinite(v_mx) and v_mx > 0) else math.nan
            )
    if len(idx) < 2:
        return None
    return SafetyTrajectory(
        steps=tuple(idx),
        var_allfail=tuple(tuple(per_name_af[n]) for n in names),
        var_mixed=tuple(tuple(per_name_mx[n]) for n in names),
        safety=tuple(tuple(per_name_ratio[n]) for n in names),
        allfail_fraction=tuple(fracs),
        names=tuple(names),
    )


def _bare_evidence(reading: AmplifierReading) -> Any:
    """An Evidence wrapper for a bound travelling on a refusal, built before any gate applies."""
    from reward_lens.core.evidence import make_evidence
    from reward_lens.core.types import SubjectRef

    return make_evidence(
        observable="AmplifierSafety",
        observable_version="1.0",
        subject=SubjectRef(signals=(), dataset=None, readout="reward"),
        value=reading,
        gauge=GaugeStatus.INVARIANT,
    )


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


class AmplifierSafety(EstimatorInstrument):
    """E4. Whether a reward component's variance persists into the all-fail phase.

    Kill condition, from the catalogue record: if amplifier safety does not predict collapse on
    three held-out runs, it is a description rather than a forecast.
    """

    name = "AmplifierSafety"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "E4"
    deviations = (
        "`Var_group` is the degrees-of-freedom-weighted pooled within-group variance rather than "
        "the mean of the per-group variances. `Var_group` is written without saying which, "
        "and the two differ the moment groups have different K, which happens as soon as a grader "
        "abstains on one rollout",
        "rung 2 projects the step at which the all-fail fraction crosses a stated dominance level "
        "rather than a collapse in task success. The condition is about the phase in "
        "which all-fail groups dominate, and a record contains that fraction directly while it "
        "contains a collapse only after one has happened",
        "an estimator that does not z-score gets a refusal carrying the component magnitudes as a "
        "bound, rather than a ratio. `verifiers` is the framework this fires on",
        "the denominator is the genuinely mixed groups, so a window is partitioned three ways and "
        "not two. The condition is written `Var_group(r_c | mixed groups)` and reading `mixed` as "
        "`not all-fail` pools in the groups where nothing failed, whose task component cannot vary "
        "either. The all-pass count and their pooled variance are on every reading so the size of "
        "the difference is visible on the reader's own record",
    )

    quantity = "estimator.amplifier_safety"
    requires: dict[Component, Access] = RECORD_ACCESS
    substrates = ALL_SUBSTRATES
    phases = frozenset({Phase.IN_RUN, Phase.POST_RUN})
    envelope = AMPLIFIER_ENVELOPE
    invariance = "reward.affine"
    #: Invariant, and exactly so: the numerator and the denominator are variances of the *same*
    #: component, so an affine rescaling `r_c -> a*r_c + b` multiplies both by `a**2` and the shift
    #: cancels in each. The generated test checks this rather than assuming it, because if the
    #: partition into all-fail and mixed moved with the component, the ratio would not be invariant
    #: and that would be a defect in this instrument rather than a property of the estimator.
    invariance_relation = INVARIANT
    baselines = AMPLIFIER_BASELINES
    rung = 0

    def __init__(
        self,
        subject: Run | Sequence[Step] | Step | Sequence[Group] | None = None,
        *,
        floor: FailureFloor | None = None,
        components: Sequence[str] | None = None,
        trajectory: bool = False,
    ) -> None:
        self.subject = subject
        self.floor = floor
        self.components = components
        self.trajectory = bool(trajectory)
        self.rung = 2 if trajectory else 0
        self._measured_regime: ConditionReading | None = None
        self._disagreed = False

    # -- the envelope, measured ------------------------------------------------

    def measure_nondegeneracy(self) -> ConditionReading | None:
        """`GROUP_NONDEGENERATE` from the groups this instrument holds, not from a declaration.

        The catalogue's own entry for E4 says "`GROUP_NONDEGENERATE` measured, not assumed", and
        the qualifier is the whole point: a caller who passes `RegimeReading.of(GROUP_NONDEGENERATE=
        True)` has supplied a verdict, and `ConditionReading.detail` on that constructor says so in
        as many words ("supplied, not measured"). This computes the statistic instead.
        """
        if self.subject is None:
            return None
        groups = list(iter_groups(self.subject))
        if not groups:
            return None
        n_degenerate = sum(1 for g in groups if g.group_stats.degenerate)
        n_singleton = sum(1 for g in groups if g.k < 2)
        fraction = (n_degenerate + n_singleton) / len(groups)
        return ConditionReading(
            condition=RegimeCondition.GROUP_NONDEGENERATE,
            holds=bool(fraction <= DEGENERATE_FRACTION_MAX),
            statistic=fraction,
            threshold=DEGENERATE_FRACTION_MAX,
            detail=(
                f"measured over {len(groups)} groups: {n_degenerate} degenerate, {n_singleton} "
                f"with K < 2"
            ),
        )

    def preflight(self, ctx: Context) -> PreflightResult:
        """Preflight against a regime this instrument measured, in preference to a supplied one."""
        measured = self.measure_nondegeneracy()
        self._measured_regime = measured
        self._disagreed = False
        if measured is None:
            return super().preflight(ctx)
        supplied = ctx.regime_reading
        if supplied is not None:
            prior = supplied.conditions.get(RegimeCondition.GROUP_NONDEGENERATE)
            if prior is not None and prior.holds is not measured.holds:
                self._disagreed = True
            merged = dict(supplied.conditions)
        else:
            merged = {}
        merged[RegimeCondition.GROUP_NONDEGENERATE] = measured
        return super().preflight(replace(ctx, regime_reading=RegimeReading(conditions=merged)))

    def compute(self) -> Any:
        if self.subject is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no record was supplied, so there are no groups to partition",
                remedy=(
                    "pass `subject=` a Run, a Step, or a sequence of Groups carrying per-leaf "
                    "scores, and `floor=FailureFloor(at=..., component=...)` naming what counts as "
                    "a failure on the task component."
                ),
            )
        if self.floor is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    "no failure floor was stated, so the groups cannot be partitioned into "
                    "all-fail and mixed. `GroupStats` records the all-fail verdict without "
                    "recording the threshold it was taken against, so the recorded flag cannot be "
                    "read as authoritative."
                ),
                remedy=(
                    "pass `floor=FailureFloor(at=0.0, component='<task leaf name>')`. The "
                    "composite total will not do: a rollout that fails the task and collects a "
                    "format bonus scores above zero and has passed nothing."
                ),
            )
        out = measure_amplifier_safety(
            self.subject,
            floor=self.floor,
            components=self.components,
            trajectory=self.trajectory,
            instrument=self.name,
        )
        if isinstance(out, AmplifierReading) and self._measured_regime is not None:
            out.envelope_measured = True
            out.envelope_statistic = self._measured_regime.statistic
            out.envelope_disagreed = self._disagreed
        return out


__all__ = [
    "AMPLIFIER_BASELINES",
    "AMPLIFIER_BIAS",
    "AMPLIFIER_ENVELOPE",
    "DEGENERATE_FRACTION_MAX",
    "DOMINANCE_AT",
    "LIVE_AMPLIFIER_AT",
    "PERSISTENCE_AT",
    "SAFE_BELOW",
    "AmplifierReading",
    "AmplifierSafety",
    "ComponentSafety",
    "SafetyTrajectory",
    "component_names",
    "measure_amplifier_safety",
    "pooled_within_variance",
]

"""E2, the degenerate and all-fail group fractions, and where they come from.

A group whose scores have no spread teaches nothing. Under a z-scoring estimator its advantage is
``0 / (0 + eps)``, so it contributes an update of exactly zero if the trainer keeps it and no update
at all if the trainer drops it, and those are different bookkeeping with the same learning content.
A group where every rollout *fails* is the neighbouring case and it is not the same case: the task
component has no spread but an auxiliary component may still be moving, and the z-score rescales
whatever is left to full magnitude. That is E4's subject, and this instrument is where its
population is counted.

**The two fractions are counted differently and it matters.** Degeneracy is read off
`GroupStats.degenerate`, which compares the group's standard deviation against the *estimator's*
epsilon rather than against zero, because what makes a group contribute nothing is the trainer's
epsilon and not exact equality. All-fail cannot be read off `GroupStats.all_fail` without a further
statement, and this instrument refuses rather than reading it anyway. `GroupStats.from_scores`
leaves the flag False when no failure floor was supplied, and it records the verdict without
recording the threshold, so ``all_fail = False`` on every group means either "no group all-failed"
or "nobody said what failure looks like" and the record cannot tell you which. On the real GRPO
record this package was tested against, every flag is False and every group is all-fail on the task
component. A fraction of zero would have been a confident wrong number.

**Rung 1 attributes the degenerate groups rather than counting them.** Four causes, and they call
for different responses. A group that is degenerate *because every rollout failed* is a task the
policy cannot do yet, so the response is curriculum. A group that is degenerate because every
rollout scored at the top of the scale is grader saturation, so the response is a harder rubric or a
wider scale. A group that is degenerate at neither extreme is the grader declining to separate four
different rollouts, which is a resolution problem and is the one an effective group size is about. A
group that is degenerate because every score was an abstention is not a degenerate group at all: it
is a group with no scores, and counting it as tied would flatter the grader.

**The rate is not unreported, and the catalogue record's prior-art line is wrong about that.** TRL
1.9.2 logs ``frac_reward_zero_std`` every step, and it appears in the real record this package was
built against. It is used as a mandatory comparator here rather than ignored: two numbers that are
supposed to be the same number and are not is a finding about one of them.

Kill condition, from the catalogue record: n/a.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BaselineID, BiasStatement
from reward_lens.core.reading import Refusal, RefusalReason, bounded_refusal
from reward_lens.core.types import Access, Capability, Component, GaugeStatus, Phase
from reward_lens.measure.estimator._base import EstimatorInstrument
from reward_lens.measure.estimator.spec import (
    ALL_SUBSTRATES,
    RECORD_ACCESS,
    iter_groups,
    iter_steps,
)
from reward_lens.record.schema import Group, Run, Step
from reward_lens.record.scores import ScoreContext, evaluate, leaves

#: The catalogue's baseline for E2, plus the framework's own logged fraction. The second is the one
#: that can disagree with us, which is what makes it worth carrying.
GROUP_BASELINES: tuple[BaselineID, ...] = (
    "baseline.nominal_group_size",
    "baseline.framework_logged_fraction",
)

#: TRL 1.9.2 logs this every step. It is the mean over *rollouts* of an indicator on the group's
#: standard deviation, which equals the fraction over groups when every group has the same K, so it
#: is compared against ours only when K is constant across the window.
FRAMEWORK_ZERO_STD_KEY = "frac_reward_zero_std"

#: E2 must not require `GROUP_NONDEGENERATE`: its own quantity is the degenerate fraction, so an
#: envelope requiring non-degeneracy would make it refuse whenever it has something to report. That
#: is the same defect A4 and B5 carry, one series over.
GROUP_CENSUS_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "this counts groups and classifies them. Its own subject is the condition every other "
        "instrument in the series declares as an envelope, so requiring that condition here would "
        "make the instrument refuse in exactly the regime it exists to measure."
    ),
)

#: The four causes rung 1 partitions the degenerate groups into. Named as data so a card can render
#: them in a fixed order and a test can assert the partition is exhaustive.
CAUSES: tuple[str, ...] = (
    "task_difficulty",
    "grader_saturation",
    "grader_resolution",
    "all_abstained",
)

DEGENERATE_BIAS = BiasStatement(
    direction="unknown",
    why=(
        "the count is exact for the groups the record carries. Its bias is entirely the record's "
        "sampling policy: a recorder that keeps every group is unbiased, and one that keeps a "
        "sample has whatever bias `Run.sampling_policy` describes. This instrument does not apply "
        "the correction, because the correction is a property of the record rather than of the "
        "count and applying it silently would hide which one you are reading."
    ),
)


# ---------------------------------------------------------------------------
# The failure floor: stated, or refused
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureFloor:
    """What counts as a failure, and what counts as saturation, stated rather than inferred.

    ``component`` names the leaf of the score tree the floor applies to. A composite reward's total
    is not a correctness label: a rollout that fails the task and picks up a format bonus scores
    above zero and has not passed anything. So the floor attaches to a named leaf, and when the tree
    has exactly one leaf that leaf is the default.
    """

    at: float
    component: str | None = None
    #: The score at or above which the grader has run out of scale. None means saturation is not
    #: being tested, which is the honest default: most rubrics have no stated ceiling.
    saturates_at: float | None = None


def component_scores(group: Group, component: str | None) -> list[float | None] | None:
    """One value per rollout for a named leaf, or None when the leaf is not in every tree.

    Returns None rather than a partial list, because a component present on three rollouts of four
    gives a within-group variance over a different population than the group's, and averaging those
    two is the error this whole layer is about.
    """
    out: list[float | None] = []
    for traj in group.trajectories:
        tree = traj.scores
        if tree is None:
            return None
        if component is None:
            found = leaves(tree)
            if len(found) != 1:
                return None
            leaf = found[0]
        else:
            match = [leaf for leaf in leaves(tree) if leaf.name == component]
            if len(match) != 1:
                return None
            leaf = match[0]
        out.append(None if leaf.abstained else leaf.value)
    return out


def total_scores(group: Group) -> list[float]:
    """The composed total per rollout, with an abstention arriving as NaN rather than as zero."""
    return [evaluate(t.scores, ScoreContext()) for t in group.trajectories]


#: The three phases a scored group can be in against a stated failure floor, plus the state where
#: the floor cannot be applied at all. Held as data so E4 renders them in one order and a test can
#: assert the partition is exhaustive.
PHASES: tuple[str, ...] = ("all_fail", "all_pass", "mixed", "unscored")


def group_phase(group: Group, floor: FailureFloor) -> str:
    """Which of the four `PHASES` a group is in, on the floor's named component.

    ``all_pass`` here means *no rollout failed*, which is the complement of ``all_fail`` and not the
    same thing as `classify_groups`'s saturation flag. The saturation flag asks whether the grader
    ran out of scale and needs `FailureFloor.saturates_at` to answer; this asks whether the group
    contains a failure at all, which the floor alone settles. E4 needs the second: its denominator
    is the variance in groups that hold *both* outcomes, and a group where every rollout succeeded
    is as far from that as a group where every rollout failed.

    ``unscored`` is a group where every rollout abstained on the component, or where the component
    is not on every rollout's tree. It belongs in neither population and is counted rather than
    folded into one, because a group with no scores is not a group with no spread.
    """
    values = component_scores(group, floor.component)
    present = [] if values is None else [v for v in values if v is not None]
    if not present:
        return "unscored"
    if all(v <= floor.at for v in present):
        return "all_fail"
    if all(v > floor.at for v in present):
        return "all_pass"
    return "mixed"


def partition_by_floor(groups: Sequence[Group], floor: FailureFloor) -> list[str]:
    """`group_phase` over a window, in record order. One label per group, exhaustive."""
    return [group_phase(g, floor) for g in groups]


def classify_groups(
    groups: Sequence[Group],
    floor: FailureFloor | None,
) -> tuple[list[bool], list[bool], list[str]]:
    """Per group: is it all-fail, is it saturated, and why is it degenerate if it is.

    The all-fail verdict comes from the stated floor applied to the named component. When no floor
    is stated, `GroupStats.all_fail` is used and the caller has taken responsibility for it.

    The second flag is **saturation**, not the complement of the first. It is True only when
    `FailureFloor.saturates_at` was stated and every present value reaches it, because that is the
    question rung 1 asks of a degenerate group: did the grader run out of scale? A group where
    every rollout merely cleared the failure floor is not saturated, and it is not all-fail either.
    Anything that needs the three-way partition wants `partition_by_floor`; folding a groups' phase
    out of these two booleans is how E4's denominator came to pool all-pass groups with mixed ones.
    """
    all_fail: list[bool] = []
    all_pass: list[bool] = []
    cause: list[str] = []
    for group in groups:
        stats = group.group_stats
        if floor is None:
            fails = bool(stats.all_fail)
            passes = False
        else:
            vals = component_scores(group, floor.component)
            present = [] if vals is None else [v for v in vals if v is not None]
            fails = bool(present) and all(v <= floor.at for v in present)
            passes = (
                bool(present)
                and floor.saturates_at is not None
                and all(v >= floor.saturates_at for v in present)
            )
        all_fail.append(fails)
        all_pass.append(passes)
        if not stats.degenerate:
            cause.append("")
        elif stats.mean is None:
            cause.append("all_abstained")
        elif fails:
            cause.append("task_difficulty")
        elif passes:
            cause.append("grader_saturation")
        else:
            cause.append("grader_resolution")
    return all_fail, all_pass, cause


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class GroupCensus:
    """How many groups taught anything, and what the ones that did not have in common."""

    n_groups: int
    n_degenerate: int
    degenerate_fraction: float
    n_all_fail: int
    #: NaN when no failure floor was stated. The `AllFailGroups` instrument refuses in that case
    #: rather than reporting a fraction; this field exists so the census can still be rendered.
    allfail_fraction: float
    floor_stated: bool
    n_all_abstained: int
    k_min: int
    k_max: int
    #: The rung-1 partition of the degenerate groups, as counts. Sums to `n_degenerate`.
    causes: dict[str, int] = field(default_factory=dict)
    #: The same partition as shares of the degenerate groups.
    cause_shares: dict[str, float] = field(default_factory=dict)
    baselines: dict[str, float] = field(default_factory=dict)
    #: What the framework logged for the same quantity, per step, where it logged anything.
    framework_logged: list[float] = field(default_factory=list)
    framework_agrees: bool | None = None
    says: str = ""

    def render(self) -> str:
        return self.says


def _pct(x: float) -> str:
    return f"{100.0 * x:.0f}%"


def _census_says(c: GroupCensus) -> str:
    lines = [
        (
            f"{_pct(c.degenerate_fraction)} of {c.n_groups} groups had no reward spread "
            f"({c.n_degenerate} of {c.n_groups}). They contributed noise, not signal."
            if c.n_degenerate
            else f"Every one of {c.n_groups} groups had some reward spread."
        )
    ]
    if c.floor_stated:
        further = "A further" if c.n_degenerate else "Of them"
        lines.append(
            f"{further} {c.n_all_fail} ({_pct(c.allfail_fraction)}) were all-fail, where whatever "
            f"variance survives is what the z-score rescales."
        )
    if c.causes:
        parts = [f"{k} {v}" for k, v in c.causes.items() if v]
        if parts:
            lines.append("Of the degenerate groups: " + ", ".join(parts) + ".")
    if c.framework_agrees is False:
        lines.append(
            f"The framework logged a mean zero-spread fraction of "
            f"{float(np.mean(c.framework_logged)):.4g} for the same window against ours of "
            f"{c.degenerate_fraction:.4g}. One of the two is wrong."
        )
    return " ".join(lines)


def census_groups(
    subject: Run | Sequence[Step] | Step | Sequence[Group],
    *,
    floor: FailureFloor | None = None,
    instrument: str = "DegenerateGroups",
) -> GroupCensus | Refusal:
    """Count the groups and classify the ones with no spread, or refuse if there are none.

    Callable without a `Context`, so E4 can reuse the classification rather than re-deriving it and
    a test can hand it three hand-built groups.
    """
    groups = list(iter_groups(subject))
    if not groups:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail="this window contains no groups, so there is nothing to count",
            remedy=(
                "point this at a Run, a Step, or a sequence of Groups carrying at least one "
                "group. A window with steps and no groups has lost the group structure, and every "
                "group-relative quantity went with it."
            ),
            statistics={"n_groups": 0},
        )

    all_fail, _all_pass, cause = classify_groups(groups, floor)
    n_degenerate = sum(1 for g in groups if g.group_stats.degenerate)
    counts = {name: sum(1 for c in cause if c == name) for name in CAUSES}
    ks = [g.k for g in groups]

    logged = [
        float(step.optimizer.extra[FRAMEWORK_ZERO_STD_KEY])
        for step in iter_steps(subject)
        if FRAMEWORK_ZERO_STD_KEY in step.optimizer.extra
    ]
    degenerate_fraction = n_degenerate / len(groups)
    agrees: bool | None = None
    if logged and min(ks) == max(ks):
        # The framework's statistic is a mean over rollouts and ours is a fraction over groups.
        # They coincide only when K is constant, which is why the comparison is gated on that.
        agrees = bool(abs(float(np.mean(logged)) - degenerate_fraction) <= 1.0 / len(groups))

    census = GroupCensus(
        n_groups=len(groups),
        n_degenerate=n_degenerate,
        degenerate_fraction=degenerate_fraction,
        n_all_fail=sum(all_fail),
        allfail_fraction=(sum(all_fail) / len(groups)) if floor is not None else math.nan,
        floor_stated=floor is not None,
        n_all_abstained=counts["all_abstained"],
        k_min=min(ks),
        k_max=max(ks),
        causes=counts,
        cause_shares={k: (v / n_degenerate if n_degenerate else 0.0) for k, v in counts.items()},
        baselines={
            "baseline.nominal_group_size": float(max(ks)),
            "baseline.framework_logged_fraction": (
                float(np.mean(logged)) if logged else float("nan")
            ),
        },
        framework_logged=logged,
        framework_agrees=agrees,
    )
    census.says = _census_says(census)
    return census


def _floor_refusal(instrument: str, census: GroupCensus) -> Refusal:
    """The refusal both instruments produce when nobody has said what failure looks like."""
    return bounded_refusal(
        instrument=instrument,
        reason=RefusalReason.ACCESS_INSUFFICIENT,
        detail=(
            "no failure floor was stated, and `GroupStats` records the all-fail verdict without "
            "recording the threshold it was taken against. `all_fail = False` on every group of "
            f"this window therefore means either that none of the {census.n_groups} groups "
            "all-failed or that nobody supplied a floor when the record was written, and the "
            "record cannot distinguish those."
        ),
        remedy=(
            "pass `floor=FailureFloor(at=..., component='<leaf name>')` naming the score that "
            "counts as a failure on the task component of the score tree. The composite total is "
            "not a correctness label: a rollout that fails the task and collects a format bonus "
            "scores above zero. If the record was written with `GroupStats.from_scores(..., "
            "failure_at=...)`, pass `trust_recorded_flag=True` to say so."
        ),
        bound=_census_evidence(census),
        n_groups=census.n_groups,
        n_degenerate=census.n_degenerate,
    )


def _census_evidence(census: GroupCensus) -> Any:
    """A bare Evidence wrapper for the census, so a refusal can still hand back the count.

    Built through `make_evidence` rather than through a `Context`, because the bound travels on a
    refusal and a refusal is produced before any gate has been applied. The trust level that falls
    out is `EXPLORATORY`, which is correct for a number attached to a declined measurement.
    """
    from reward_lens.core.evidence import make_evidence
    from reward_lens.core.types import SubjectRef

    return make_evidence(
        observable="DegenerateGroups",
        observable_version="1.0",
        subject=SubjectRef(signals=(), dataset=None, readout="reward"),
        value=census,
        gauge=GaugeStatus.INVARIANT,
    )


class DegenerateGroups(EstimatorInstrument):
    """E2. How many groups had no reward spread, and why the ones that did not.

    Kill condition, from the catalogue record: n/a.
    """

    name = "DegenerateGroups"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "E2"
    deviations = (
        "the rung-1 attribution has four causes, not two. The catalogue names task difficulty and "
        "grader saturation; a degenerate group at neither extreme is the grader declining to "
        "separate four different rollouts, and a group where every score was an abstention is not "
        "a tied group at all. Folding either into one of the two named causes would attribute a "
        "measurement failure to the task",
        "`baseline.framework_logged_fraction` is compared against ours only when K is constant "
        "across the window, because TRL's `frac_reward_zero_std` is a mean over rollouts and this "
        "is a fraction over groups",
    )

    quantity = "estimator.degenerate_fraction"
    requires: dict[Component, Access] = RECORD_ACCESS
    substrates = ALL_SUBSTRATES
    phases = frozenset({Phase.IN_RUN, Phase.POST_RUN})
    envelope = GROUP_CENSUS_ENVELOPE
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = GROUP_BASELINES
    rung = 0

    def __init__(
        self,
        subject: Run | Sequence[Step] | Step | Sequence[Group] | None = None,
        *,
        floor: FailureFloor | None = None,
        attribute: bool = False,
    ) -> None:
        self.subject = subject
        self.floor = floor
        self.attribute = bool(attribute)
        self.rung = 1 if attribute else 0

    def compute(self) -> Any:
        if self.subject is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no record was supplied, so there are no groups to count",
                remedy=(
                    "pass `subject=` a Run, a Step, or a sequence of Groups. That is RECORD:RECORD "
                    "and nothing else."
                ),
            )
        census = census_groups(self.subject, floor=self.floor, instrument=self.name)
        if isinstance(census, Refusal):
            return census
        if self.attribute and self.floor is None:
            return _floor_refusal(self.name, census)
        return census


class AllFailGroups(EstimatorInstrument):
    """E2's second quantity. The fraction of groups where every rollout failed the task.

    Two instruments rather than one with two readings, because two quantities make two ladders and
    a single `quantity` field cannot name both. The degenerate fraction has a rung 1; this does not,
    which is what the registry says (`rungs: 1` against `rungs: 2`).

    Kill condition, from the catalogue record: n/a.
    """

    name = "AllFailGroups"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "E2"
    deviations = (
        "this refuses when no failure floor was stated rather than reading `GroupStats.all_fail` "
        "as authoritative. The schema records the verdict and not the threshold, so all-False "
        "cannot be distinguished from never-determined, and on the real GRPO record tested here "
        "every flag is False while every group is all-fail on the task component",
    )

    quantity = "estimator.allfail_fraction"
    requires: dict[Component, Access] = RECORD_ACCESS
    substrates = ALL_SUBSTRATES
    phases = frozenset({Phase.IN_RUN, Phase.POST_RUN})
    envelope = GROUP_CENSUS_ENVELOPE
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = GROUP_BASELINES
    rung = 0

    def __init__(
        self,
        subject: Run | Sequence[Step] | Step | Sequence[Group] | None = None,
        *,
        floor: FailureFloor | None = None,
        trust_recorded_flag: bool = False,
    ) -> None:
        self.subject = subject
        self.floor = floor
        self.trust_recorded_flag = bool(trust_recorded_flag)

    def compute(self) -> Any:
        if self.subject is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no record was supplied, so there are no groups to count",
                remedy=(
                    "pass `subject=` a Run, a Step, or a sequence of Groups, and state the failure "
                    "floor with `floor=FailureFloor(at=..., component=...)`."
                ),
            )
        census = census_groups(self.subject, floor=self.floor, instrument=self.name)
        if isinstance(census, Refusal):
            return census
        if self.floor is None:
            if not self.trust_recorded_flag:
                return _floor_refusal(self.name, census)
            groups = list(iter_groups(self.subject))
            n = sum(1 for g in groups if g.group_stats.all_fail)
            census.n_all_fail = n
            census.allfail_fraction = n / len(groups)
            census.floor_stated = True
            census.says = _census_says(census)
        return census


__all__ = [
    "CAUSES",
    "DEGENERATE_BIAS",
    "FRAMEWORK_ZERO_STD_KEY",
    "GROUP_BASELINES",
    "GROUP_CENSUS_ENVELOPE",
    "PHASES",
    "AllFailGroups",
    "DegenerateGroups",
    "FailureFloor",
    "GroupCensus",
    "census_groups",
    "classify_groups",
    "component_scores",
    "group_phase",
    "partition_by_floor",
    "total_scores",
]

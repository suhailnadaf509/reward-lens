"""B4: how often the grader failed, and how often a real number went into the group anyway.

Two rates, and they are not the same rate. An **abstention** is a grader call that produced no
score. A **silent zero** is the subset of those where the framework wrote a number in its place and
carried on, so a grader with a working abstention channel has a silent-zero rate of exactly zero
and an abstention rate that is not. A survey of the abstention column across fifteen grader
interfaces found four of them with no such channel at all: `verifiers` catches any exception from a
reward function and sets `ans = 0.0` (`rubrics/rubric.py:204-217`), veRL's `math_reward` prints the
exception and returns 0.0, its `math_verify` wrapper returns `timeout_score`, and SWE-bench scores
an empty test suite as resolved. TRL is the counter-example and the model here: `None` maps to NaN,
is excluded, and warns.

The consequence is the part that gets left out, and it is why this instrument exists rather than a
log grep. `verifiers`' `score_group` computes `advantage = reward_i - mean(rewards)`
(`rubric.py:406-409`). A substituted zero is a real number inside that mean. So one crashing
component does not damage one rollout: it moves the baseline every other rollout in the group is
measured against, by exactly the amount the substitution changed the mean by. On a group of K, a
rollout that would have scored `r` and was written down as `0` shifts every other rollout's
advantage by `+r/K` in the direction that flatters the group. This instrument computes that shift
from the record instead of describing it, because the record keeps both what the framework used and
the fact that the call failed, and the difference between the two means is exactly the number.

`verifiers` divides by nothing afterwards, and that changes less than it sounds like it does. On an
estimator that z-scores, the substituted zero also moves the denominator, so the contamination is
additive and multiplicative at once and its size depends on the group's spread. With mean-centring
alone the contamination is purely additive, which is simpler and is not smaller in the way that
matters: an additive shift does not shrink when the spread does, so on a group whose rollouts all
scored alike it is most of the advantage.

**What rung 1 cannot see, said here rather than discovered later.** Instrumenting the wrapper counts
the failures that cross it. A grader that swallows its own exception *inside* itself and returns a
plausible number never crosses anything: `is_equiv` from `hendrycks/math`, the answer-equivalence
checker most open RLVR maths pipelines still call, ends in a bare `except: return str1 == str2`, so
every parse failure it has becomes a string comparison and the tap records `returned`. Measured
against it, this instrument reports a silent-zero rate of 0.0, and that number is a lower bound
rather than a measurement. The reading says so. Finding the rest is a source-reading question and
it belongs to the D series, which is the other half of B4's own access line: "GRADER:RECORD with
exception logging, **or source**".

Nothing kills this instrument. It is a defect census, it holds in every regime, and the right
outcome is that the number goes to zero because people fix it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence, TypeVar

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import INVARIANT, TRIVIAL_GROUP
from reward_lens.core.quantity import BaselineID
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Access, Capability, Component, GaugeStatus, Phase
from reward_lens.measure.base import Context
from reward_lens.measure.composition.composition import (
    ALL_SUBSTRATES,
    CompositionInstrument,
    map_tree,
)
from reward_lens.record.scores import (
    AbstentionCensus,
    GroupScores,
    Leaf,
    ScoreContext,
    ScoreTree,
    census,
    evaluate,
    leaves,
)

#: B4's mandatory baseline, and the catalogue prints it as an instruction: "assume zero and show the
#: difference". So the comparator is the framework's own behaviour, the substituted number counted
#: as a score, and the number reported is the difference it made to the group's advantage baseline.
ZEROS_AS_SCORES_BASELINE: BaselineID = "baseline.zeros_counted_as_scores"

B4_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "No regime condition is recorded for B4. It is a defect census over a record: it counts "
        "what is there and asserts nothing about the process that produced it, so no regime of the "
        "run can make the count wrong."
    ),
)

#: `GRADER:RECORD with exception logging, or source`. The access matrix is a conjunction, so the
#: weaker of the two branches is declared: this instrument reads a record. The `or source` branch is
#: a different estimator at a different access level and it belongs to the D series, which is the
#: only thing that can see a grader swallowing its own exception.
B4_ACCESS: dict[Component, Access] = {Component.GRADER: Access.RECORD}

#: All four. `spec/CATALOGUE.yaml` carries `phases: OPEN` for B4; the source it was transcribed
#: from reads `Sub/Ph all`, and that decides it.
ALL_PHASES = frozenset({Phase.PRE_RUN, Phase.IN_RUN, Phase.POST_RUN, Phase.DEPLOYED})

#: So `SilentZeroRate.over(...)` is typed as a `SilentZeroRate` rather than as the shared base.
#: `typing.Self` would say this in one word and the package floor is Python 3.10.
_C = TypeVar("_C", bound="_CensusInstrument")


# ---------------------------------------------------------------------------
# The advantage-baseline consequence, computed rather than described
# ---------------------------------------------------------------------------


def as_the_framework_used_it(tree: ScoreTree) -> ScoreTree:
    """The tree with every silent zero treated as the score the framework treated it as.

    An abstaining leaf that carries a number is two facts: the grader failed, and a real value went
    into the group. `evaluate` uses neither, by design, because a rollout whose grader failed must
    not receive an advantage. But the framework did use it, and the difference between the two
    group means is the whole consequence, so this reconstructs the arm the framework actually ran.

    Only leaves with a finite substituted value change. A leaf that abstained with no value stays
    an abstention, because there is nothing to have been used.
    """

    def clear(node: ScoreTree) -> ScoreTree:
        if (
            isinstance(node, Leaf)
            and node.abstained
            and node.value is not None
            and math.isfinite(float(node.value))
        ):
            return replace(node, abstained=False)
        return node

    return map_tree(tree, clear)


@register_payload
@dataclass(frozen=True)
class BaselineShift:
    """What the silent zeros in one group did to the baseline every advantage is measured against.

    ``shift`` is the framework's group mean minus the mean over the rollouts that were actually
    scored. Every rollout in the group that did *not* fail had its advantage moved by ``-shift``,
    because the advantage is the score minus that mean and only the mean changed.

    ``additive_only`` records whether the estimator divides by the group standard deviation. It
    does not on `verifiers`, so the contamination is exactly this shift and nothing else. On an
    estimator that z-scores, the substituted value also moves the denominator, and then this number
    is the additive part of a contamination that also rescales.
    """

    group: str
    k: int
    n_scored: int
    n_silent_zero: int
    substituted_total: float
    mean_as_used: float
    mean_honest: float
    shift: float
    additive_only: bool

    @property
    def contaminated(self) -> bool:
        return self.n_silent_zero > 0 and math.isfinite(self.shift)


def baseline_shifts(groups: Sequence[GroupScores]) -> tuple[BaselineShift, ...]:
    """One `BaselineShift` per group, over the groups whose composition was recorded.

    Free. It evaluates each recorded tree twice, once as the framework ran it and once excluding the
    calls that failed, and differences the two group means. No grader is called.
    """
    out: list[BaselineShift] = []
    for gi, g in enumerate(groups):
        used: list[float] = []
        honest: list[float] = []
        n_silent = 0
        substituted = 0.0
        for tree, ctx in zip(g.trees, g.contexts):
            if tree is None:
                continue
            for leaf in leaves(tree):
                if leaf.silent_zero and math.isfinite(float(leaf.value or 0.0)):
                    n_silent += 1
                    substituted += float(leaf.value or 0.0)
            used.append(evaluate(as_the_framework_used_it(tree), ctx))
            honest.append(evaluate(tree, ctx))
        live_used = [v for v in used if math.isfinite(v)]
        live_honest = [v for v in honest if math.isfinite(v)]
        mean_used = float(np.mean(live_used)) if live_used else math.nan
        mean_honest = float(np.mean(live_honest)) if live_honest else math.nan
        out.append(
            BaselineShift(
                group=g.id or str(gi),
                k=len(used),
                n_scored=len(live_honest),
                n_silent_zero=n_silent,
                substituted_total=substituted,
                mean_as_used=mean_used,
                mean_honest=mean_honest,
                shift=mean_used - mean_honest,
                additive_only=not g.estimator.std_normalised,
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class AbstentionReading:
    """Both rates, the outcome histogram behind them, and what the substitutions cost.

    ``rate_upper_bound`` is the reason the unattributable leaves are reported rather than folded
    in. A leaf with no call record has an unknown outcome, and a rate that treats unknown outcomes
    as successes is biased downward by exactly the amount that makes it reassuring. So the point
    estimate is over the leaves whose outcome is knowable and the upper bound assumes every unknown
    one failed. When they differ, the honest reading is the interval.

    ``n_boundary_failures`` is the rung-1 evidence: calls that raised or timed out *through* the
    wrapper, which is the only kind of failure instrumenting a wrapper can see. When it is zero and
    ``n_abstained`` is not, the abstentions were recorded some other way. When both are zero the
    reading is a lower bound and ``says`` puts that first, because a grader with no abstention
    channel produces a rate of zero whether it never failed or never told anyone.

    ``n_reconstructed`` separates a call outcome that was *observed* from one that was *assumed*,
    and the distinction was not obvious until this was pointed at a real store. A `GraderCallRef`
    with no ``seq`` was built by a converter rather than by a tap, which `record.scores` says in as
    many words, so its ``returned`` is what the converter had to write in the absence of an outcome
    field rather than something anybody saw. Counting those as successes is the same downward bias
    the unattributable leaves carry, one step less visible, so they are counted and named.
    """

    n_leaves: int
    n_abstained: int
    n_silent_zero: int
    n_unattributable: int
    n_shadowed: int
    n_boundary_failures: int
    n_reconstructed: int
    substituted_total: float
    by_grader: Mapping[str, int]
    outcomes: Mapping[str, int]
    abstention_rate: float
    silent_zero_rate: float
    abstention_rate_upper: float
    silent_zero_rate_upper: float
    shifts: tuple[BaselineShift, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_grader", dict(self.by_grader))
        object.__setattr__(self, "outcomes", dict(self.outcomes))
        object.__setattr__(self, "shifts", tuple(self.shifts))

    @property
    def n_known(self) -> int:
        return self.n_leaves - self.n_unattributable

    @property
    def channel_observed(self) -> bool:
        """Whether this record has ever shown the grader declining to score.

        False does not mean the grader never failed. It means nothing in the record distinguishes a
        grader that never failed from one with no way to say so.
        """
        return self.n_abstained > 0 or self.n_boundary_failures > 0

    @property
    def contaminated_groups(self) -> tuple[BaselineShift, ...]:
        return tuple(s for s in self.shifts if s.contaminated)

    @property
    def max_abs_shift(self) -> float:
        shifts = [abs(s.shift) for s in self.contaminated_groups if math.isfinite(s.shift)]
        return max(shifts) if shifts else 0.0

    @property
    def mean_abs_shift(self) -> float:
        shifts = [abs(s.shift) for s in self.contaminated_groups if math.isfinite(s.shift)]
        return float(np.mean(shifts)) if shifts else 0.0

    def says(self) -> str:
        """B4's sentence, with this record's numbers in it."""
        if not self.channel_observed:
            return (
                f"no grader call in this record is recorded as having failed, over "
                f"{self.n_known} scored leaves. That is a silent-zero rate of 0.0% and it is a "
                f"lower bound rather than a measurement: a grader that catches its own exception "
                f"and returns a plausible number never reaches a wrapper, so nothing here "
                f"distinguishes a grader that never failed from one with no way to say that it did."
            )
        return (
            f"{self.silent_zero_rate:.1%} of {self.n_known} scored leaves were a number the grader "
            f"did not produce, and {self.abstention_rate:.1%} declined to score at all. Those "
            f"substituted numbers entered the group mean, and through it every other rollout's "
            f"advantage."
        )

    def consequence(self) -> str:
        """The advantage-baseline consequence, with numbers when the groups were supplied."""
        mechanism = (
            "The estimator computes `advantage = reward_i - mean(rewards)`, so a substituted "
            "number is not a wrong score for one rollout: it moves the baseline the whole group is "
            "measured against. A component that crashes systematically shifts every advantage in "
            "every group it touches, in whichever direction the substitution differs from the "
            "score that was not produced."
        )
        if not self.shifts:
            return (
                mechanism + " No group structure was supplied, so the size of that shift on this "
                "record was not computed. Pass the groups to get it."
            )
        contaminated = self.contaminated_groups
        if not contaminated:
            return (
                mechanism + f" On this record no group's mean moved: {len(self.shifts)} group(s) "
                f"carry no silent zero, so the baseline is the one the scores imply."
            )
        additive = all(s.additive_only for s in contaminated)
        scale = (
            "The recorded estimator divides by nothing after centring, which is `verifiers`' own "
            "transform (`rubric.py:406-409`), so this contamination is purely additive: it does "
            "not rescale the advantages and it does not shrink when the group's spread does. On a "
            "group whose rollouts scored alike it is most of the advantage."
            if additive
            else "The recorded estimator divides by the group standard deviation, so the "
            "substituted number moves the denominator as well and the contamination rescales "
            "every advantage in the group on top of the shift below."
        )
        return (
            f"{mechanism} On this record {len(contaminated)} of {len(self.shifts)} groups had "
            f"their mean moved by a substituted number, by up to {self.max_abs_shift:.4g} and by "
            f"{self.mean_abs_shift:.4g} on average. Every rollout in those groups whose own grader "
            f"worked had its advantage moved by the negative of that. {scale}"
        )

    def limitation(self) -> str:
        """What this rung cannot see. Always printed, because it is always true."""
        base = (
            "Instrumenting the wrapper counts the failures that cross it. A grader that catches "
            "its own exception and returns a number never crosses anything: `is_equiv` from "
            "`hendrycks/math` ends in a bare `except: return str1 == str2`, so a parse failure "
            "becomes a string comparison and the record says `returned`. This rate is a lower "
            "bound on that class of failure, and reading the source is what finds the rest."
        )
        if self.n_unattributable:
            base += (
                f" {self.n_unattributable} of {self.n_leaves} leaves carry no call record at all, "
                f"so the rate above is over the {self.n_known} whose outcome is knowable; "
                f"assuming every unknown one failed puts the silent-zero rate at "
                f"{self.silent_zero_rate_upper:.1%}."
            )
        if self.n_reconstructed:
            base += (
                f" {self.n_reconstructed} of {self.n_leaves} leaves carry a call record with no "
                f"sequence number, which means a converter reconstructed it from a store that had "
                f"no outcome field rather than a tap observing it. Their outcome is what the "
                f"converter had to write, not what anybody saw."
            )
        if self.n_shadowed:
            base += (
                f" {self.n_shadowed} leaves sit under a binding override, where the pipeline never "
                f"used the score, so their outcome says nothing about the run's live grader "
                f"reliability and is counted separately rather than either way."
            )
        return base

    def render(self) -> str:
        return "\n".join([self.says(), f"    {self.consequence()}", f"    {self.limitation()}"])


def read_census(
    trees: Sequence[ScoreTree | None],
    contexts: Sequence[ScoreContext] | None = None,
    groups: Sequence[GroupScores] = (),
) -> AbstentionReading | Refusal:
    """Count the abstentions and the silent zeros, and price what the substitutions did.

    The counts are `record.scores.census`, reused rather than restated: it already separates the
    abstention from the silent zero, holds the unattributable leaves out of the denominator, and
    reports the ones a binding override shadowed. What is added here is the outcome histogram,
    which says how the failures were observed, and the advantage-baseline shift, which says what
    they cost.
    """
    present = [t for t in trees if t is not None]
    if not present:
        return Refusal(
            instrument=_B4,
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"none of the {len(trees)} trajectories carries a score tree, so there are no "
                f"grader calls to count."
            ),
            remedy=(
                "instrument the grader with reward_lens.tap so each recorded score carries the "
                "call that produced it, and re-run. A total on disk cannot say whether the grader "
                "that produced it succeeded."
            ),
            statistics={"n_trees": len(trees), "n_with_composition": 0},
        )

    counts: AbstentionCensus = census(trees, contexts)
    if counts.n_leaves - counts.n_unattributable <= 0:
        return Refusal(
            instrument=_B4,
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"all {counts.n_leaves} recorded leaves carry no grader call, so no outcome is "
                f"knowable and a rate over them would have an empty denominator. Reporting 0.0 "
                f"here would be the reassuring answer and it would be made up."
            ),
            remedy=(
                "wrap the grader with `reward_lens.tap.instrument_grader` and build each leaf with "
                "`GraderCallRef.from_call`, so the outcome of every call reaches the record. That "
                "is rung 1 of this instrument's ladder and it costs about 800 ns per call."
            ),
            statistics={
                "n_leaves": counts.n_leaves,
                "n_unattributable": counts.n_unattributable,
            },
        )

    outcomes: dict[str, int] = {}
    n_boundary = 0
    n_reconstructed = 0
    for tree in present:
        for leaf in leaves(tree):
            ref = leaf.grader_call
            key = "unrecorded" if ref is None else ref.outcome
            outcomes[key] = outcomes.get(key, 0) + 1
            if ref is None:
                continue
            if ref.raised:
                n_boundary += 1
            if ref.seq is None:
                n_reconstructed += 1

    known = counts.n_leaves - counts.n_unattributable
    return AbstentionReading(
        n_leaves=counts.n_leaves,
        n_abstained=counts.n_abstained,
        n_silent_zero=counts.n_silent_zero,
        n_unattributable=counts.n_unattributable,
        n_shadowed=counts.n_shadowed,
        n_boundary_failures=n_boundary,
        n_reconstructed=n_reconstructed,
        substituted_total=counts.substituted_total,
        by_grader=dict(counts.by_grader),
        outcomes=outcomes,
        abstention_rate=counts.n_abstained / known,
        silent_zero_rate=counts.n_silent_zero / known,
        abstention_rate_upper=(counts.n_abstained + counts.n_unattributable) / counts.n_leaves,
        silent_zero_rate_upper=(counts.n_silent_zero + counts.n_unattributable) / counts.n_leaves,
        shifts=baseline_shifts(groups),
    )


_B4 = "measure.composition.abstention"


# ---------------------------------------------------------------------------
# The two instruments
# ---------------------------------------------------------------------------


class _CensusInstrument(CompositionInstrument):
    """The census, once, behind two quantities.

    `grader.silent_zero_rate` and `grader.abstention_rate` come out of one pass over the record and
    are two registered quantities, so they are two instruments over one computation rather than one
    instrument reporting a number the registry has no name for. Both readings carry the full census
    and both name the same baseline, because the substituted numbers are the thing both of them are
    about.
    """

    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "B4"
    deviations = (
        "rung 1 counts the failures that cross the wrapper. A grader that catches its own "
        "exception and returns a number is invisible to it, and the reading says so rather than "
        "reporting the lower bound as a measurement",
    )

    requires: dict[Component, Access] = B4_ACCESS
    substrates = ALL_SUBSTRATES
    phases = ALL_PHASES
    envelope = B4_ENVELOPE
    #: `spec/QUANTITIES.yaml` declares `none` for both quantities and this is what that resolves to
    #: (`core.quantity.TRIVIAL_GROUP`). It is a decision rather than an omission: no affine
    #: rescaling of the reward changes how many grader calls failed. The generated test
    #: passes with nothing acting on the reading, which is the honest outcome rather than a
    #: manufactured one. The resolved spelling is declared rather than the catalogue's, because
    #: `check_invariance` takes a registered group id and `none` is not one.
    invariance = TRIVIAL_GROUP
    invariance_relation = INVARIANT
    baselines = (ZEROS_AS_SCORES_BASELINE,)
    rung = 1

    #: Which of the two rates this subclass's Evidence value leads with.
    reports: str = ""

    def __init__(
        self,
        trees: Sequence[ScoreTree | None] = (),
        contexts: Sequence[ScoreContext] | None = None,
        *,
        groups: Sequence[GroupScores] = (),
    ) -> None:
        self.groups = tuple(groups)
        if not trees and self.groups:
            trees = [t for g in self.groups for t in g.trees]
            contexts = [c for g in self.groups for c in g.contexts]
        self.trees = tuple(trees)
        self.contexts = None if contexts is None else tuple(contexts)

    @classmethod
    def over(cls: type[_C], groups: Sequence[GroupScores]) -> _C:
        """Build one over a set of groups, so the advantage-baseline shift can be computed."""
        return cls(groups=groups)

    def compute(self) -> Any:
        if not self.trees:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no trajectories were supplied, so there is nothing to count.",
                remedy=(
                    "pass `trees=[...]` from a recorded Group, or `groups=[...]` built with "
                    "`record.scores.group_scores`, which also gets you the advantage-baseline shift."
                ),
            )
        out = read_census(self.trees, self.contexts, self.groups)
        if isinstance(out, Refusal):
            return replace(out, instrument=self.name)
        return out

    def uncertainty(self, computed: AbstentionReading) -> Uncertainty | None:
        """The interval the unattributable leaves open up, and nothing when they open none.

        Not a sampling interval. The point estimate is exact over the leaves whose outcome is
        knowable; the bound is what happens if every unknown one failed. Reporting the pair is the
        only honest treatment of a denominator with a hole in it.
        """
        if not computed.n_unattributable:
            return None
        low, high = (
            (computed.silent_zero_rate, computed.silent_zero_rate_upper)
            if self.reports == "silent_zero"
            else (computed.abstention_rate, computed.abstention_rate_upper)
        )
        return Uncertainty(
            ci_low=float(low),
            ci_high=float(high),
            ci_level=1.0,
            n=computed.n_leaves,
            n_effective=float(computed.n_known),
            method="unattributable-bounds",
        )

    def payload(self, computed: AbstentionReading) -> dict[str, Any]:
        return {
            "reports": self.reports,
            "value": (
                computed.silent_zero_rate
                if self.reports == "silent_zero"
                else computed.abstention_rate
            ),
            "silent_zero_rate": computed.silent_zero_rate,
            "abstention_rate": computed.abstention_rate,
            "silent_zero_rate_upper": computed.silent_zero_rate_upper,
            "abstention_rate_upper": computed.abstention_rate_upper,
            "n_leaves": computed.n_leaves,
            "n_known": computed.n_known,
            "n_abstained": computed.n_abstained,
            "n_silent_zero": computed.n_silent_zero,
            "n_unattributable": computed.n_unattributable,
            "n_shadowed": computed.n_shadowed,
            "n_boundary_failures": computed.n_boundary_failures,
            "n_reconstructed": computed.n_reconstructed,
            "channel_observed": computed.channel_observed,
            "substituted_total": computed.substituted_total,
            "by_grader": dict(computed.by_grader),
            "outcomes": dict(computed.outcomes),
            "n_groups": len(computed.shifts),
            "n_groups_contaminated": len(computed.contaminated_groups),
            "max_abs_baseline_shift": computed.max_abs_shift,
            "mean_abs_baseline_shift": computed.mean_abs_shift,
            "says": computed.says(),
            "interpretation": "\n".join(
                [computed.says(), computed.consequence(), computed.limitation()]
            ),
            "baselines": {ZEROS_AS_SCORES_BASELINE: float(computed.mean_abs_shift)},
        }

    def measure(self, ctx: Context) -> Any:
        out = self._computed if self._computed is not None else self.compute()
        if isinstance(out, Refusal):
            raise ValueError(
                f"{self.name}.measure was called on a measurement that declines to produce "
                f"Evidence: {out.reason.name}. Call `estimate`, which returns the refusal as a "
                f"value carrying its remedy."
            )
        return ctx.emit(self.payload(out), uncertainty=self.uncertainty(out))


class SilentZeroRate(_CensusInstrument):
    """How often a number the grader did not produce went into the group anyway.

    The numerator is a leaf whose call failed and whose recorded value is a real number, which is
    exactly `verifiers`' `except Exception: ans = 0.0`. A framework with a working abstention
    channel reports zero here and a real abstention rate beside it, and the two together are what
    say which kind of run you are looking at.
    """

    name = "SilentZeroRate"
    quantity = "grader.silent_zero_rate"
    reports = "silent_zero"


class AbstentionRate(_CensusInstrument):
    """How often the grader declined to score at all.

    Registered separately because it is not the silent-zero rate: the silent zero is the subset
    where a number was substituted. The abstention rate is a first-class reported quantity, and
    the catalogue registered only its subset.
    """

    name = "AbstentionRate"
    quantity = "grader.abstention_rate"
    reports = "abstention"


__all__ = [
    "ALL_PHASES",
    "B4_ACCESS",
    "B4_ENVELOPE",
    "ZEROS_AS_SCORES_BASELINE",
    "AbstentionRate",
    "AbstentionReading",
    "BaselineShift",
    "SilentZeroRate",
    "as_the_framework_used_it",
    "baseline_shifts",
    "read_census",
]

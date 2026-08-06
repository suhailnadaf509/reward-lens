"""I3: how much of the run sits where the gradient with respect to task quality is exactly zero.

An `Override` sets the score to a constant across a region. Inside that region finishing perfectly
and failing completely score the same, so the derivative of the score with respect to anything
under the override is not small, it is exactly zero. `gate.deadzone_fraction` is the share of
rollouts in that region, and `record.scores.deadzone` already computes it, along with the sharper
number beside it: how often removing the node would actually change the score, which is not the
same as how often the condition fired.

**Two controls run on every reading and neither is optional, because this is the instrument most
likely to overclaim.**

The first is a **graded penalty**, and the contrast has to be against that rather than against
agents in general. Insufficient verification is not a symptom of budget gates: it shows up in
agents with no budget term at all, and one published study of 526 agent-attributable failures found
99.6 percent of them carrying a validation-failure signal. (That figure is somebody else's
measurement, quoted here to say why the control is what it is; nothing in this module reproduces
it.) So the question a dead-zone fraction can answer is narrow and worth stating precisely: not
"does this gate suppress verification", which the number cannot reach, but "how much of the run
does this gate make the score blind to quality on, and how much of that survives replacing the
override with a penalty of the same size delivered gradually". `GradedContrast` is that second arm.

The second is the **decode length, fixed and reported explicitly, separately from the training-time
budget**. A generation-length cap censors the running variable whatever the reward rule does, and a
region drawn by the sampler is not a region drawn by the gate. This is not a hypothetical failure:
a generation-length cap has already turned a reported +15.3 point gain into -8.0 points in this
exact area. Where the whole region sits on the cap, this instrument refuses rather than reporting
the sampler's behaviour as the gate's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from reward_lens.core.evidence import register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BaselineID
from reward_lens.core.reading import Refusal, RefusalReason, refuse_incomplete
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.threshold._base import (
    ALL_SUBSTRATES,
    DEADZONE_ENVELOPE,
    RECORD_ACCESS,
    RECORD_PHASES,
    ThresholdInstrument,
)
from reward_lens.measure.threshold.gates import DecodeLength, Gate, RunningVariable
from reward_lens.record.scores import (
    DeadZone,
    Override,
    ScoreContext,
    ScoreTree,
    deadzone,
    evaluate,
    find,
)

#: The two mandatory controls, named as baselines because that is what they are: the comparator a
#: dead-zone claim is sent back without.
DEADZONE_BASELINES: tuple[BaselineID, ...] = (
    "baseline.graded_penalty",
    "baseline.decode_length_fixed",
)


# ---------------------------------------------------------------------------
# control 1: the graded penalty
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class GradedContrast:
    """The same total penalty, delivered gradually. The arm the override has to beat.

    The graded arm subtracts `slope * (x - cutoff)` from the underlying score above the cutoff,
    with the slope set so the two arms agree at the **largest running-variable value observed in
    the region**. That matching rule is what makes it a control rather than a weaker rule: the
    worst-punished rollout is punished identically under both arms, and everything the two arms
    differ by is the shape of the punishment in between.

    ``deadzone_fraction`` is zero on this arm by construction and it is reported rather than
    asserted, because a reader has to be able to see that the contrast is doing what it says. What
    is not zero by construction, and is the number worth reading, is ``spearman``: how much of the
    ordering by underlying score survives inside the region. Under the override it cannot survive,
    because every rollout in the region has the same score.

    **The sign of ``spearman`` is a finding rather than a detail.** Matching at the far end forces a
    slope equal to the whole override penalty divided by the width of the region, and where the
    score's own gradient in the running variable is shallower than that, the graded arm ends up
    *decreasing* in the underlying score: the better rollout inside the region scores worse. So a
    penalty large enough to reproduce a hard override at its worst point cannot be graded without
    inverting the incentive it was supposed to preserve. The override destroys the ordering and the
    matched graded penalty reverses it, and neither of those is "the same rule, delivered gently".
    `ordering` names which of the three happened.
    """

    slope: float
    matched_at: float
    n_region: int
    deadzone_fraction: float
    spearman: float
    distinct_realised_override: int
    distinct_realised_graded: int
    erased_spread: float | None
    mean_penalty_override: float
    mean_penalty_graded: float

    @property
    def ordering(self) -> str:
        """What the graded arm did to the ordering by underlying score."""
        if not math.isfinite(self.spearman):
            return "destroyed"
        if self.spearman > 0.99:
            return "preserved"
        if self.spearman < -0.99:
            return "inverted"
        return f"partly preserved (rho {self.spearman:+.3f})"

    def render(self) -> str:
        spread = "unknown" if self.erased_spread is None else f"{self.erased_spread:.4g}"
        tail = ""
        if self.ordering == "inverted":
            tail = (
                " The graded slope is steeper than the score's own gradient in the running "
                "variable, so matching the override at its worst point inverts the ordering "
                "rather than preserving it: no graded penalty of this size is gentler."
            )
        return (
            f"graded-penalty control: slope {self.slope:.4g} per unit, matched at "
            f"{self.matched_at:g}. Inside the region the override leaves "
            f"{self.distinct_realised_override} distinct score and the graded penalty leaves "
            f"{self.distinct_realised_graded}; the ordering by underlying score is "
            f"{self.ordering} under the graded arm and destroyed under the override, which cost "
            f"{spread} of score spread. Mean penalty {self.mean_penalty_override:.4g} against "
            f"{self.mean_penalty_graded:.4g}.{tail}"
        )


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, with average ranks for ties. Undefined when either side is constant."""
    if a.size < 3:
        return float("nan")

    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        out = np.empty(v.size, dtype=np.float64)
        out[order] = np.arange(v.size, dtype=np.float64)
        # average the ranks of tied values, which is what makes a constant array rank-constant
        for value in np.unique(v):
            mask = v == value
            out[mask] = out[mask].mean()
        return out

    ra, rb = ranks(a), ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def graded_contrast(
    trees: Sequence[ScoreTree | None],
    contexts: Sequence[ScoreContext],
    running: np.ndarray,
    *,
    node: str,
    gate: Gate,
) -> GradedContrast | None:
    """Re-score the region under a graded penalty matched to the override at the far end.

    Returns None when the region is empty, which is a fact about the gate rather than a failure:
    an override nothing triggers has no dead zone and no contrast to draw.

    The graded arm is computed directly rather than assembled as a `ScoreTree`, because none of the
    eight node types applies a penalty that reads a per-rollout feature and varies continuously in
    it. Building one out of a staircase of overrides would be a second implementation of the thing
    being contrasted against.
    """
    base: list[float] = []
    realised: list[float] = []
    x: list[float] = []
    constant = float("nan")
    for tree, ctx, value in zip(trees, contexts, running):
        if tree is None:
            continue
        found = find(tree, node)
        if not isinstance(found, Override):
            continue
        if not found.condition.holds(ctx):
            continue
        constant = float(found.constant)
        without = evaluate(tree, ctx, {node})
        with_it = evaluate(tree, ctx)
        if not math.isfinite(without) or not math.isfinite(value):
            continue
        base.append(float(without))
        realised.append(float(with_it))
        x.append(float(value))
    if len(base) < 3:
        return None

    b = np.asarray(base, dtype=np.float64)
    r = np.asarray(realised, dtype=np.float64)
    xs = np.asarray(x, dtype=np.float64)
    far = float(xs.max())
    reach = far - gate.cutoff
    if reach <= 0:
        # Every rollout in the region sits on the cutoff, so the region has no width in the running
        # variable and no graded penalty can be matched across it. Returning a slope of zero would
        # be worse than returning nothing: it is the ungated score wearing the control's name.
        return None
    penalty_at_far = float(np.mean(b[xs == far])) - constant
    slope = penalty_at_far / reach
    graded = b - slope * np.clip(xs - gate.cutoff, 0.0, None)
    return GradedContrast(
        slope=float(slope),
        matched_at=far,
        n_region=int(b.size),
        deadzone_fraction=0.0,
        spearman=_spearman(b, graded),
        distinct_realised_override=int(np.unique(np.round(r, 12)).size),
        distinct_realised_graded=int(np.unique(np.round(graded, 12)).size),
        erased_spread=float(np.std(b)) if b.size > 1 else None,
        mean_penalty_override=float(np.mean(b - r)),
        mean_penalty_graded=float(np.mean(b - graded)),
    )


# ---------------------------------------------------------------------------
# the reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class DeadZoneReading:
    """The dead-zone fraction with both mandatory controls attached."""

    gate: Gate
    zone: DeadZone
    contrast: GradedContrast | None
    decode: DecodeLength
    running: str
    unit: str
    n_region_at_decode_cap: int

    @property
    def fraction(self) -> float:
        return self.zone.fraction

    @property
    def says(self) -> str:
        return (
            f"{self.zone.fraction:.1%} of {self.zone.n} rollouts sit in a region where the "
            f"gradient with respect to task quality is exactly zero, and removing the gate would "
            f"change the score for {self.zone.n_binding} of them"
        )

    def render(self) -> str:
        lines = [
            f"I3 dead zone  {self.gate.render()}",
            f"  {self.says}",
            f"  {self.zone.render()}",
        ]
        if self.contrast is not None:
            lines.append(f"  {self.contrast.render()}")
        else:
            lines.append(
                "  graded-penalty control: not drawn. The region holds fewer than three rollouts, "
                "or they all sit exactly on the cutoff, so there is no width to grade a penalty "
                "across and no ordering to preserve or erase."
            )
        lines.append(f"  {self.decode.render()}")
        if self.n_region_at_decode_cap:
            lines.append(
                f"  {self.n_region_at_decode_cap} of {self.zone.n_region} rollouts in the region "
                f"sit on the decode cap, so that much of the region was drawn by the sampler "
                f"rather than by the gate."
            )
        if self.gate.installed:
            lines.append(
                "  the gate is installed, so this is what that rule would have done to these "
                "recorded rollouts. It is not what a policy trained under it would have produced."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# the estimator with its refusals
# ---------------------------------------------------------------------------


def deadzone_fraction(
    trees: Sequence[ScoreTree | None],
    contexts: Sequence[ScoreContext],
    running: RunningVariable,
    gate: Gate,
    decode: DecodeLength,
    *,
    node: str | None = None,
    instrument: str = "DeadZoneFraction",
) -> DeadZoneReading | Refusal:
    """I3's reading, or the refusal that says why there is none."""
    present = [t for t in trees if t is not None]
    if not present:
        return refuse_incomplete(
            instrument,
            field=f"a composition tree on any of {len(trees)} rollouts",
            subject="the record",
            remedy=(
                "instrument the grader with `reward_lens.tap` so the per-leaf scores and the "
                "composition are recorded. A dead zone is a property of the composition, and a "
                "single realised score cannot say whether it was pinned to a constant."
            ),
            n=len(trees),
        )

    zones = deadzone(trees, contexts, node=gate.name if node is None else node)
    if not zones:
        return refuse_incomplete(
            instrument,
            field=f"an Override node called {gate.name!r}",
            subject=f"any of the {len(present)} recorded composition trees",
            remedy=(
                "name a gate the run actually had, with `recorded_gates(trees)`, or install one "
                "counterfactually with `install_gate(tree, cutoff=...)` and read the result as a "
                "validation rather than as a measurement of this run."
            ),
            n_trees=len(present),
            gate=gate.name,
        )

    zone = zones[0]
    values = np.asarray(running.values, dtype=np.float64)
    at_cap = 0
    if running.censored_at is not None and values.size == len(trees):
        in_region = np.zeros(values.size, dtype=bool)
        for i, (tree, ctx) in enumerate(zip(trees, contexts)):
            if tree is None:
                continue
            found = find(tree, zone.node)
            if isinstance(found, Override) and found.condition.holds(ctx):
                in_region[i] = True
        at_cap = int(np.sum(in_region & (values >= running.censored_at)))
        if zone.n_region > 0 and at_cap == zone.n_region:
            return Refusal(
                instrument=instrument,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail=(
                    f"all {zone.n_region} rollouts inside the region sit on the decode cap of "
                    f"{running.censored_at:g} {running.unit}. The region was drawn by the sampler "
                    f"and not by the gate, so the fraction would be a property of "
                    f"`{decode.cap_key}` wearing the gate's name."
                ),
                remedy=(
                    f"raise the sampler's completion cap above the gate at {gate.cutoff:g} "
                    f"{running.unit} and re-run, so a rollout can exceed the training budget "
                    f"without first hitting the generation limit. Report the decode length beside "
                    f"the budget in either case."
                ),
                statistics={
                    "n_region": zone.n_region,
                    "n_at_decode_cap": at_cap,
                    "decode_cap": running.censored_at,
                    "cutoff": float(gate.cutoff),
                },
            )

    contrast = graded_contrast(trees, contexts, values, node=zone.node, gate=gate)
    return DeadZoneReading(
        gate=gate,
        zone=zone,
        contrast=contrast,
        decode=decode,
        running=running.name,
        unit=running.unit,
        n_region_at_decode_cap=at_cap,
    )


class DeadZoneFraction(ThresholdInstrument):
    """I3. The share of the run where the score cannot see task quality at all.

    Kill condition, from the catalogue record: if the graded-penalty control shows the same
    verification suppression. `GradedContrast` is that control and it is on every reading.
    """

    name = "DeadZoneFraction"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "I3"
    deviations = (
        "the graded-penalty arm is computed directly on the recorded leaves rather than assembled "
        "as a `ScoreTree`, because none of the eight node types applies a penalty that reads a "
        "per-rollout feature and varies continuously in it. The arm is a linear penalty above the "
        "cutoff, matched to the override at the largest running-variable value in the region",
        "the fraction counts rollouts inside the region. It does not measure verification "
        "behaviour, which is a property of what the policy wrote rather than of what the "
        "composition did to the score, and this instrument makes no claim about it",
    )

    quantity = "gate.deadzone_fraction"
    requires = RECORD_ACCESS
    substrates = ALL_SUBSTRATES
    phases = RECORD_PHASES
    envelope = DEADZONE_ENVELOPE
    #: `units` in the registry. A fraction of rollouts is not comparable with a fraction of tokens,
    #: and `check_invariance` routes this group to `check_unit_refusal`.
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = DEADZONE_BASELINES
    rung = 0

    def __init__(
        self,
        trees: Sequence[ScoreTree | None] = (),
        contexts: Sequence[ScoreContext] = (),
        running: RunningVariable | None = None,
        gate: Gate | None = None,
        decode: DecodeLength | None = None,
        *,
        node: str | None = None,
    ) -> None:
        self.trees = tuple(trees)
        self.contexts = tuple(contexts)
        self.running = running
        self.gate = gate
        self.decode = decode
        self.node = node

    def compute(self) -> DeadZoneReading | Refusal:
        if self.running is None or self.gate is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no running variable or no gate was supplied",
                remedy=(
                    "pass `running=`, `gate=` and the trees and contexts from "
                    "`measure.threshold.gates.score_contexts(run)`. The dead zone is a property of "
                    "the composition, so the composition has to be supplied."
                ),
            )
        if self.decode is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail=(
                    "no decode length was supplied. A dead-zone fraction reported without the "
                    "sampler's completion cap cannot say whether the region was drawn by the "
                    "reward rule or by the generation limit."
                ),
                remedy=(
                    "pass `decode=measure.threshold.gates.decode_length(run)`. Where the record "
                    "does not carry the cap, that call returns a DecodeLength saying so, which is "
                    "a reading this instrument accepts and prints."
                ),
            )
        return deadzone_fraction(
            self.trees,
            self.contexts,
            self.running,
            self.gate,
            self.decode,
            node=self.node,
            instrument=self.name,
        )


__all__ = [
    "DEADZONE_BASELINES",
    "DeadZoneFraction",
    "DeadZoneReading",
    "GradedContrast",
    "deadzone_fraction",
    "graded_contrast",
]

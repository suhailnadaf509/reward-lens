"""B3: what the reward is actually made of, and what it would be without a piece.

Two instruments, because the catalogue names two quantities and they are answered by different
work. `CompositionTree` reads the shape and reports `grader.composition`: which composition
primitives this run actually used, how many of them a `Mapping[str, float]` of weights could not have
expressed, and what each node is worth one at a time. `CounterfactualComposition` re-evaluates the
recorded tree with a node removed and reports `grader.counterfactual_score`.

Neither of them re-implements anything. `record/scores.py` is the substrate: it owns the eight node
types, `evaluate`, `counterfactual`, `replay_advantages` and the four refusals they produce. This
module is the instrument wrapper, and the work it adds beyond dispatch is three things.

**The reading goes through the estimator.** The question is about advantages, not scores, so the
result depends on the recorded `EstimatorSpec` and a run that did not record one gets a refusal
rather than a plausible number. That refusal is `replay_advantages`', reused rather than restated.
A change that does not survive group centring did not change the update, and this instrument
reports the score-scale comparison beside the advantage-scale one so the difference between the two
is visible rather than assumed. That is the mandatory baseline: the deployed composition, on its own
scale, before the estimator touched it.

**The null leak.** `reward.null` is the group B3 declares, and its assertion is that adding a
per-prompt constant within a group leaves every advantage alone. Measured with the kernel's own
`check_invariance` at 32 draws, that holds on a plain `WeightedSum` to 3.33e-16 and fails at a
maximum deviation of 5.38 the moment the tree contains a hard override, because a constant added to
the task reward does not reach the rollouts the override pinned, so the group is no longer shifted
by a constant and the shift does not cancel. That is not a defect in the estimator or in this
instrument. It is a property of the composition, it is what an override *does*, and it is
measurable, so `null_leak` measures it: how much of the group a constant added at a named node
actually reaches, and how far the advantages move when it reaches only part. A composition whose
reach is 0 or 1 satisfies `reward.null` exactly; one in between does not, and the reach is the
reason. The instrument reports the number rather than declaring an envelope that would make it
refuse on the one composition the catalogue's own example sentence is about.

**No grader calls.** Rung 0 of B3's ladder re-reads the leaves already on disk. Rung 1 re-scores
with a modified tree and needs `GRADER: QUERY`; it is not built here and the ladder says so.

What this cannot do. It reads what a composition *did*, so a counterfactual that changes what a
grader would have said is out of scope. One-at-a-time ablation deltas sum to the total only when
the tree is additive, which on a real composition it usually is not, and `is_additive` says which
case you are in rather than leaving it to be discovered. And the whole instrument is unavailable on
a record that kept only the total: per-leaf scores that were never written down are gone, and the
refusal for that case says so with the remedy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import Evidence, register_payload
from reward_lens.core.gates import require_frame_for_comparison
from reward_lens.core.invariance import INVARIANT, InvariancePayload, default_tol
from reward_lens.core.quantity import BaselineID
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context, run
from reward_lens.record.schema import EstimatorSpec
from reward_lens.record.scores import (
    CounterfactualResult,
    GroupContext,
    GroupGate,
    GroupScores,
    Leaf,
    Lexicographic,
    Mask,
    Override,
    Piecewise,
    PredicateRef,
    Product,
    ScoreContext,
    ScoreTree,
    WeightedSum,
    ablation_deltas,
    counterfactual,
    evaluate,
    is_additive,
    node_names,
    replay_advantages,
    walk,
)

# ---------------------------------------------------------------------------
# The runner both instruments in this package share
# ---------------------------------------------------------------------------


class CompositionInstrument(BaseObservable):
    """Preflight, compute once, return the refusal or emit the Evidence.

    `Observable.measure` returns `Evidence` by contract and `Instrument.estimate` returns
    `Reading`, which is `Evidence | Refusal`. Both contracts are right and they need a seam,
    because every instrument here decides to refuse partway through: the counterfactual cannot know
    whether the recorded estimator is replayable until it has tried to replay it, and the census
    cannot know whether any leaf carries a call outcome until it has counted.

    `measure.frontier._base` and `measure.controls._base` have this shape for this reason and are
    not imported, because reaching into another package's private module is a dependency that is
    invisible at the point where it breaks. This one is shared *within* the package by B3 and B4.
    """

    #: Set by `estimate` for the duration of one call so `measure` does not recompute.
    _computed: Any = None

    def compute(self) -> Any:  # pragma: no cover - abstract
        """The instrument's own work, with no `Context`. Returns a payload or a `Refusal`.

        Written without a Context on purpose: both of these are pure functions of recorded data the
        caller already holds, which is what makes them runnable with no signal, no network and no
        GPU, and testable without standing one up.
        """
        raise NotImplementedError

    def payload(self, computed: Any) -> dict[str, Any]:  # pragma: no cover - abstract
        """The Evidence value: a flat mapping, including the `baselines` key the lint reads."""
        raise NotImplementedError

    def gated_emit(self, ctx: Context, computed: Any) -> Evidence:
        """Hand a computed result to the runner, or apply the runner's gates by hand.

        `run` resolves `ctx.signal.caps` to enforce the capability check and these instruments
        never touch a signal: they read a record. The no-signal branch does what `run` would do
        minus the check that has nothing to check against. Gate 2 is applied in both branches,
        because it depends on the instrument's gauge status and the context's frame rather than on
        the signal, and a gate that is convenient to drop is the one worth keeping.
        """
        self._computed = computed
        try:
            if ctx.signal is not None:
                return run(self, ctx)
            if ctx.is_comparison:
                require_frame_for_comparison(self.gauge_status, ctx.frame)
            return self.measure(ctx)
        finally:
            self._computed = None

    def estimate(self, ctx: Context) -> Reading:
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        out = self.compute()
        if isinstance(out, Refusal):
            return out
        return self.gated_emit(ctx, out)

    def measure(self, ctx: Context) -> Evidence:
        out = self._computed if self._computed is not None else self.compute()
        if isinstance(out, Refusal):
            raise ValueError(
                f"{self.name}.measure was called on a measurement that declines to produce "
                f"Evidence: {out.reason.name}. Call `estimate`, which returns the refusal as a "
                f"value carrying its remedy."
            )
        return ctx.emit(self.payload(out))


# ---------------------------------------------------------------------------
# What a weights dict can and cannot say
# ---------------------------------------------------------------------------

#: The two node types a `Mapping[str, float]` of weights represents without loss. Every other
#: observed primitive is a shape a dict cannot hold, which is that argument stated as a type rather
#: than as a paragraph.
EXPRESSIBLE_BY_WEIGHTS: tuple[type, ...] = (Leaf, WeightedSum)

#: Each inexpressible primitive, and the published composition it was drawn from. The text is the
#: card's, so it is written for someone reading a reward they did not build.
PRIMITIVE_NOTES: Mapping[str, str] = {
    "Product": "multiplicative zeroing: a gate multiplies the total by zero",
    "Override": (
        "a hard override to a constant, which is the one that produces a gradient dead zone "
        "(Kimi K3 pins the task reward to -1 above a scaled token budget)"
    ),
    "Lexicographic": (
        "lexicographic gating: the primary term sets a tier and the secondary can only move the "
        "score inside it (Ling 2.6's length shaping)"
    ),
    "Piecewise": "piecewise-linear anchoring, which creates a dead zone at each clamped end",
    "GroupGate": (
        "a gate conditioned on the group rather than on the rollout (ERNIE's WPSM masks a sample "
        "when group accuracy exceeds one threshold and the sample's own entropy falls below "
        "another)"
    ),
    "Mask": "the rollout is removed from the update entirely (DAPO's overlong loss masking)",
}

#: B3's mandatory baseline. The catalogue prints "the deployed composition", and the honest way to
#: run the deployed composition as a comparator is to ask the same question on the scale it lives
#: on: the score, before the estimator centred anything. A change that shows up there and not in
#: the advantages was absorbed by group centring and did not reach the update.
SCORE_SCALE_BASELINE: BaselineID = "baseline.deployed_score_scale"

#: `CompositionTree`'s baseline is the data model it is arguing against: the number of components a
#: `Mapping[str, float]` of weights would have carried out of this composition.
WEIGHTS_DICT_BASELINE: BaselineID = "baseline.weights_dict_components"

B3_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "the source reads 'leaves recorded, not just the total', which is a "
        "record-completeness requirement rather than a regime condition. Re-evaluating a recorded "
        "tree on recorded leaves is arithmetic over what is on disk: no regime of the run that "
        "produced it can make the arithmetic wrong, and a record that did not keep its per-leaf "
        "scores gets a refusal rather than a violated envelope."
    ),
)

ALL_SUBSTRATES = frozenset(
    {
        Substrate.NEURAL_SCALAR,
        Substrate.NEURAL_GEN,
        Substrate.PROGRAM,
        Substrate.PROCEDURAL,
        Substrate.HUMAN,
        Substrate.COMPOSITE,
    }
)

#: B3 reads a record and it reads the recorded estimator, because the reading is in advantage space.
#: `spec/CATALOGUE.yaml` prints only `GRADER:RECORD with per-leaf scores`, transcribed from a
#: one-line cell; the estimator half is real and is declared here. It is not a stricter gate
#: than the arithmetic needs: without a recorded `EstimatorSpec` there is no advantage to compare.
B3_ACCESS: dict[Component, Access] = {
    Component.GRADER: Access.RECORD,
    Component.ESTIMATOR: Access.RECORD,
}

#: The composition's *shape* needs no estimator: it is a fact about the tree.
COMPOSITION_ACCESS: dict[Component, Access] = {Component.GRADER: Access.RECORD}


# ---------------------------------------------------------------------------
# Walking and rebuilding a tree
# ---------------------------------------------------------------------------


def map_tree(tree: ScoreTree, fn: Callable[[ScoreTree], ScoreTree]) -> ScoreTree:
    """Rebuild a tree bottom up, applying ``fn`` to every node after its children.

    The node types are frozen dataclasses holding tuples of children, so a modified tree is a new
    tree rather than a mutation. Every rebuilt node re-runs its own `__post_init__`, which means a
    rebuild that broke an invariant (a duplicate name, a mismatched weight count) raises at the
    point of the rebuild rather than at the point of evaluation.
    """
    if isinstance(tree, Leaf):
        return fn(tree)
    if isinstance(tree, (WeightedSum, Product)):
        rebuilt: ScoreTree = replace(tree, children=tuple(map_tree(c, fn) for c in tree.children))
    elif isinstance(tree, Override):
        rebuilt = replace(tree, otherwise=map_tree(tree.otherwise, fn))
    elif isinstance(tree, Lexicographic):
        rebuilt = replace(
            tree,
            primary=map_tree(tree.primary, fn),
            secondary=map_tree(tree.secondary, fn),
        )
    elif isinstance(tree, (Piecewise, GroupGate, Mask)):
        rebuilt = replace(tree, child=map_tree(tree.child, fn))
    else:
        raise TypeError(f"not a score tree node: {type(tree).__name__}")
    return fn(rebuilt)


def offset_node(tree: ScoreTree, node: str, constant: float) -> ScoreTree:
    """A copy of ``tree`` with a constant added at one named node.

    A `Leaf` takes the constant on its value and a `WeightedSum` takes it on its bias, which are the
    two places a per-prompt offset can enter a composition without changing its shape. An abstaining
    leaf is left alone: a grader that produced no number does not produce a shifted one either, and
    quietly giving it a value would be the silent zero in another costume.

    Any other node type raises. Offsetting a `Product` or an `Override` is not "add a constant to
    this component", it is a different composition, and there is no defensible place to put the
    number.
    """

    def apply(n: ScoreTree) -> ScoreTree:
        if n.name != node:
            return n
        if isinstance(n, Leaf):
            if n.abstained or n.value is None:
                return n
            return replace(n, value=float(n.value) + constant)
        if isinstance(n, WeightedSum):
            return replace(n, bias=n.bias + constant)
        raise TypeError(
            f"cannot offset node {node!r}: it is a {type(n).__name__}, and a constant has no "
            f"unambiguous place to go in one. Offset a Leaf or a WeightedSum, which are the two "
            f"nodes a per-prompt constant can enter through without changing the composition."
        )

    return map_tree(tree, apply)


def default_offset_node(tree: ScoreTree) -> str | None:
    """Where a per-prompt constant enters this composition, when nobody said.

    The outermost `WeightedSum` is the task reward in every published composition surveyed, which is
    exactly what "a constant added to the task reward" means, and it is the node an override sits
    above. Falling back to the first scoring leaf covers the single-component case,
    which is what both real records shipped here actually contain.
    """
    for node in walk(tree):
        if isinstance(node, WeightedSum):
            return node.name
    for node in walk(tree):
        if isinstance(node, Leaf) and not node.abstained and node.value is not None:
            return node.name
    return None


# ---------------------------------------------------------------------------
# grader.composition
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class CompositionSummary:
    """What the reward is made of, counted.

    ``n_inexpressible`` is the headline and it is the whole argument as an integer: the number
    of distinct composition primitives in this reward that a `Mapping[str, float]` of weights cannot
    represent. Zero means a weights dict would have lost nothing and the whole tree machinery is
    overhead on this run, which is a finding and should be reported as one. Anything above zero
    means the data model every framework ships could not have held this composition, and the
    inventory says which primitives it would have dropped.

    ``ablation`` is what each node is worth one at a time, averaged over the rollouts where it is
    defined. It sums to the total only when ``is_additive`` is True; on a tree with a binding
    override it will not, because the override is worth the whole difference and everything under
    it is worth nothing. That is the finding rather than an error in the arithmetic, which is why
    ``is_additive`` travels beside it.
    """

    n_trees: int
    n_nodes: int
    n_leaves: int
    node_types: Mapping[str, int]
    inexpressible: tuple[str, ...]
    n_inexpressible: int
    is_additive: bool
    weights_dict_components: int
    ablation: Mapping[str, float]
    n_missing_trees: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "inexpressible", tuple(self.inexpressible))
        object.__setattr__(self, "node_types", dict(self.node_types))
        object.__setattr__(self, "ablation", dict(self.ablation))

    def says(self) -> str:
        if not self.n_inexpressible:
            return (
                f"this reward is a weighted sum of {self.n_leaves} recorded components over "
                f"{self.n_trees} rollouts, and a mapping of names to weights would have lost "
                f"nothing."
            )
        which = ", ".join(self.inexpressible)
        return (
            f"this reward composes {self.weights_dict_components} recorded components with "
            f"{self.n_inexpressible} primitive(s) a mapping of names to weights cannot express "
            f"({which}), over {self.n_trees} rollouts."
        )

    def render(self) -> str:
        lines = [self.says()]
        for name in self.inexpressible:
            note = PRIMITIVE_NOTES.get(name, "")
            lines.append(f"    {name}: {note}" if note else f"    {name}")
        if not self.is_additive:
            lines.append(
                "    the tree is not additive, so the one-at-a-time ablation deltas below do not "
                "sum to the total and were never going to."
            )
        for node, delta in sorted(self.ablation.items(), key=lambda kv: -abs(kv[1])):
            lines.append(f"    {node:<24} worth {delta:+.4g} on average when removed")
        if self.n_missing_trees:
            lines.append(
                f"    {self.n_missing_trees} rollout(s) carry no composition at all and are "
                f"excluded from every count above."
            )
        return "\n".join(lines)


def summarise(
    trees: Sequence[ScoreTree | None],
    contexts: Sequence[ScoreContext] | None = None,
) -> CompositionSummary:
    """Read the shape of a set of recorded compositions. Free: it calls nothing and evaluates once.

    ``contexts`` is optional and only fills in ``ablation``, which needs an evaluation per node per
    rollout. Without it the shape is still fully reported, because the shape is a property of the
    tree and the ablation is a property of the tree and the run together.
    """
    present = [t for t in trees if t is not None]
    n_missing = len(trees) - len(present)
    counts: dict[str, int] = {}
    n_nodes = n_leaves = 0
    weights_components = 0
    additive = True
    for tree in present:
        additive = additive and is_additive(tree)
        for node in walk(tree):
            n_nodes += 1
            counts[type(node).__name__] = counts.get(type(node).__name__, 0) + 1
            if isinstance(node, Leaf):
                n_leaves += 1
    # What a weights dict would have carried: the distinct names of the leaves, since that is all a
    # `Mapping[str, float]` can hold. Distinct rather than summed, because the same component scored
    # on 512 rollouts is one entry in the dict.
    weights_components = len(
        {n.name for tree in present for n in walk(tree) if isinstance(n, Leaf)}
    )
    inexpressible = tuple(
        sorted(
            name for name in counts if name not in {cls.__name__ for cls in EXPRESSIBLE_BY_WEIGHTS}
        )
    )

    ablation: dict[str, list[float]] = {}
    if contexts is not None:
        if len(contexts) != len(trees):
            raise ValueError(
                f"{len(trees)} trees against {len(contexts)} contexts. One context per rollout, in "
                f"the same order, or none at all."
            )
        # Distinct names from the shape loop above: `tree` there is non-optional and `node` there is
        # a `ScoreTree`, where here they are the optional entry and a node *name*. Reusing both made
        # the two loops read as one and hid a type change from the checker.
        for maybe_tree, ctx in zip(trees, contexts):
            if maybe_tree is None:
                continue
            for node_name, delta in ablation_deltas(maybe_tree, ctx).items():
                if math.isfinite(delta):
                    ablation.setdefault(node_name, []).append(delta)

    return CompositionSummary(
        n_trees=len(present),
        n_nodes=n_nodes,
        n_leaves=n_leaves,
        node_types=counts,
        inexpressible=inexpressible,
        n_inexpressible=len(inexpressible),
        is_additive=additive,
        weights_dict_components=weights_components,
        ablation={k: float(np.mean(v)) for k, v in ablation.items()},
        n_missing_trees=n_missing,
    )


# ---------------------------------------------------------------------------
# The null leak: how much of the group a per-prompt constant reaches
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class NullLeak:
    """How far this composition is from the invariance B3 declares, measured rather than assumed.

    `reward.null` says that adding a constant per prompt within a group leaves every advantage
    unchanged. It does, when the constant reaches the whole group or none of it: the group mean
    moves by the same amount and the shift cancels. It does not when the constant reaches only part
    of the group, and a hard override is exactly the thing that makes that happen, because the
    constant added below it never reaches the rollouts the override pinned.

    ``reach`` is the fraction of rollouts whose composed score moved at all. ``cancels`` is True
    when every group's reach is 0 or 1, which is the condition under which the declared invariance
    is exact. ``max_advantage_shift`` is what the advantages actually did at the stated constant, so
    the reader has the size on this run's own scale rather than a ratio.

    An abstaining rollout is not reached by anything and is not counted in either direction: it has
    no advantage in either arm, and folding it into the reach would make an all-abstaining group
    look like a perfectly cancelling one.

    **A shift and a state change are two different findings and only one of them is a shift.**
    ``max_advantage_shift`` is a maximum over the rollouts that have an advantage on *both* sides of
    the offset, and ``n_comparable`` counts exactly those. When the constant changes whether the
    group is degenerate at all, some rollouts have an advantage on one side and none on the other,
    there is no difference to take, and they land in ``n_advantage_state_changed`` instead. That is
    the larger event, not a smaller one: under ``degenerate_policy="skip"`` a group whose scores are
    all equal is dropped from the batch, so a constant that breaks the tie takes a group the trainer
    would have skipped and gives it a full contrast, and a constant that creates the tie deletes a
    group's learning signal outright. Both of those used to render as a shift of exactly 0.0, which
    is also what a perfectly cancelling composition renders as. Read the two fields together: a
    reading with ``n_comparable == 0`` has no shift to report and its zero means "nothing was
    comparable", not "nothing happened".
    """

    node: str
    constant: float
    n: int
    n_reached: int
    #: Rollouts carrying a finite advantage on **both** sides of the offset, which are the only ones
    #: `max_advantage_shift` is a maximum over.
    n_comparable: int
    max_advantage_shift: float
    cancels: bool
    per_group_reach: tuple[float, ...] = ()
    #: Whether the constant is large enough that every live score could have registered it. A
    #: constant below the resolution of the scores it is added to reaches some of them and vanishes
    #: into the last bit of others, which reads as a leak and is a rounding artefact, so the reach
    #: is not decidable and the reading says so instead of reporting one.
    resolved: bool = True
    #: Rollouts with an advantage on exactly one side of the offset. Non-zero means the constant
    #: moved the group across its estimator's degeneracy boundary, so the trainer's handling of the
    #: group changed and no per-rollout difference exists to put in `max_advantage_shift`.
    n_advantage_state_changed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "per_group_reach", tuple(float(r) for r in self.per_group_reach))

    @property
    def reach(self) -> float:
        return self.n_reached / self.n if self.n else 0.0

    def says(self) -> str:
        if not self.resolved:
            return (
                f"the constant {self.constant:g} is below the resolution of the scores it was "
                f"added to, so how much of the group it reaches is not decidable and no leak is "
                f"reported. Pass a larger `constant`, or leave it unset to get one scaled to this "
                f"run's own spread."
            )
        if self.cancels:
            return (
                f"a constant added at {self.node!r} reaches {self.reach:.0%} of every group, so it "
                f"cancels in group centring and the advantages do not move: the declared "
                f"`reward.null` invariance is exact here (largest shift "
                f"{self.max_advantage_shift:.3g})."
            )
        head = (
            f"a constant of {self.constant:g} added at {self.node!r} reaches only "
            f"{self.n_reached} of {self.n} rollouts, so it does not cancel in group centring. The "
            f"composition is reading the level of the reward and not only the contrast, which is "
            f"what a hard override does."
        )
        if self.n_advantage_state_changed:
            crossing = (
                f" It also moved {self.n_advantage_state_changed} of {self.n} rollouts across the "
                f"estimator's degeneracy boundary: they have an advantage on one side of the "
                f"offset and none on the other, so the trainer's handling of the group changed "
                f"rather than an advantage moving within it."
            )
            if not self.n_comparable:
                return (
                    head + crossing + " No rollout has an advantage on both sides, so there is no "
                    "shift to report and the 0 beside `max_advantage_shift` means nothing was "
                    "comparable, not that nothing happened."
                )
            return (
                f"{head}{crossing} Among the {self.n_comparable} rollouts that do have an advantage "
                f"on both sides, the largest moved by {self.max_advantage_shift:.4g}."
            )
        return (
            f"{head} It moves an advantage by up to {self.max_advantage_shift:.4g} over the "
            f"{self.n_comparable} rollouts that carry one on both sides."
        )


_LEAK = "measure.composition.null_leak"


def auto_constant(scores: Sequence[float]) -> float:
    """A per-prompt constant on this run's own scale, when the caller does not name one.

    One standard deviation of the live scores, which is the size of the question: "if the level of
    this component had been offset by a typical amount, would the advantages have noticed". Floored
    at a thousandth of the largest score so that a group with no spread still gets a constant its
    own scores can register, since a constant below that floor measures floating-point rounding
    rather than the composition.
    """
    live = [float(s) for s in scores if math.isfinite(s)]
    if not live:
        return 1.0
    floor = max(1e-6, 1e-3 * max(abs(s) for s in live))
    sd = float(np.std(np.asarray(live, dtype=float)))
    return sd if sd > floor else floor


def null_leak(
    groups: Sequence[GroupScores],
    *,
    node: str | None = None,
    constant: float | None = None,
) -> NullLeak | Refusal:
    """Measure how much of each group a per-prompt constant added at one node actually reaches.

    Free, and it calls no grader: it offsets a recorded value, re-evaluates the recorded tree and
    replays the recorded estimator, which is the same three steps the counterfactual takes.

    ``constant`` is a finite difference and it is reported with the result rather than divided out,
    because dividing by it would claim a linearity that only holds for an estimator that does not
    normalise by the group standard deviation. For a mean-centred estimator the shift is exactly
    linear in the constant; for one that z-scores it is not, since a partial shift also changes the
    denominator.

    A rollout counts as reached when its score moves by more than the kernel's own
    `core.invariance.default_tol`, which is `max(1e-9, 1e-7 * |score|)`. Exact inequality was the
    first implementation and it is wrong for a reason worth stating: a constant small enough to
    vanish into the last bit of one score and not another makes a perfectly additive composition
    report a reach of one half, which reads as a leak and is a rounding artefact. The tolerance is
    the kernel's rather than a new one because it is the same question the generated test asks, and
    a second answer to it would be a second convention.

    The reading separates a shift from a degeneracy crossing, and the separation is the reason it
    can report the second at all. `max_advantage_shift` is a maximum over rollouts carrying an
    advantage on both sides of the offset; a rollout carrying one on only one side has no difference
    to contribute and is counted in `n_advantage_state_changed` instead. Both cases arise from the
    same mechanism and neither is small: under `degenerate_policy="skip"` a group whose scores are
    all equal is dropped, so a constant that breaks the tie hands a full contrast to a group the
    trainer would have skipped, and one that creates the tie deletes a live group's signal.
    """
    if not groups:
        return Refusal(
            instrument=_LEAK,
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail="no groups were supplied, so there is nothing to shift.",
            remedy="pass at least one GroupScores. `record.scores.group_scores` builds one from a "
            "recorded Group.",
        )
    trees = [t for g in groups for t in g.trees if t is not None]
    if not trees:
        return Refusal(
            instrument=_LEAK,
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail="no trajectory in these groups carries a score tree.",
            remedy=(
                "instrument the grader with reward_lens.tap so the per-leaf scores are recorded, "
                "and re-run. A composition that was never recorded cannot be recovered from a "
                "total."
            ),
        )
    chosen = node or default_offset_node(trees[0])
    if chosen is None:
        return Refusal(
            instrument=_LEAK,
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                "this composition has no leaf carrying a number and no weighted sum, so there is "
                "nowhere for a per-prompt constant to enter it."
            ),
            remedy=(
                "name the node explicitly with `node=`, or check whether every leaf in this "
                "record abstained, in which case the group has no advantages to be invariant."
            ),
        )
    missing = [
        g.id or str(i)
        for i, g in enumerate(groups)
        for t in g.trees
        if t is not None and chosen not in node_names(t)
    ]
    if missing:
        return Refusal(
            instrument=_LEAK,
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"node {chosen!r} is absent from at least one tree in group(s) "
                f"{sorted(set(missing))[:5]}, so a constant added there would reach some rollouts "
                f"for the wrong reason: because the node is not in their tree at all."
            ),
            remedy=(
                f"name a node every tree carries with `node=`, or restrict the comparison to "
                f"groups whose trees all contain {chosen!r}."
            ),
        )

    if constant is None:
        constant = auto_constant(
            [evaluate(t, c) for g in groups for t, c in zip(g.trees, g.contexts) if t is not None]
        )
    c = float(constant)

    n = n_reached = n_comparable = n_state_changed = 0
    worst = 0.0
    resolved = True
    reaches: list[float] = []
    for gi, g in enumerate(groups):
        before_scores: list[float] = []
        after_scores: list[float] = []
        for tree, ctx in zip(g.trees, g.contexts):
            if tree is None:
                continue
            before_scores.append(evaluate(tree, ctx))
            after_scores.append(evaluate(offset_node(tree, chosen, c), ctx))
        a_before = replay_advantages(before_scores, g.estimator, where=f"group {g.id or gi}")
        if isinstance(a_before, Refusal):
            return a_before
        a_after = replay_advantages(
            after_scores, g.estimator, where=f"group {g.id or gi} (shifted)"
        )
        if isinstance(a_after, Refusal):
            return a_after
        live = [
            i
            for i, (b, a) in enumerate(zip(before_scores, after_scores))
            if math.isfinite(b) and math.isfinite(a)
        ]
        # A rollout is reached when its score moved by more than the kernel's own tolerance for
        # this magnitude, and the constant is resolved only if it clears that tolerance everywhere.
        # Below it, "reached" is a statement about float64 and not about the composition.
        moved = [
            i
            for i in live
            if abs(after_scores[i] - before_scores[i]) > default_tol(before_scores[i])
        ]
        resolved = resolved and all(abs(c) > default_tol(before_scores[i]) for i in live)
        n += len(live)
        n_reached += len(moved)
        reaches.append(len(moved) / len(live) if live else 0.0)
        # Three cases per rollout, not two. An advantage on both sides gives a difference, which is
        # what the shift is a maximum over. An advantage on exactly one side means the constant took
        # the group across its estimator's degeneracy boundary: under `degenerate_policy="skip"` the
        # group is dropped when its scores are all equal, so there is no before-advantage to
        # difference against, and the honest report is that the state changed rather than a
        # fabricated shift against numbers the trainer never computed. An advantage on neither side
        # is an abstention or a group degenerate throughout, and it is not a finding.
        #
        # The one-sided case used to fall through this guard and contribute nothing, so a constant
        # that turned a skipped group into a live one, or deleted a live group's signal outright,
        # reported `max_advantage_shift = 0.0`: the same rendering as a composition that cancels
        # perfectly. Found by the closed-form property test on the draw
        # `values=[0.0, -1.0], over=[True, False], c=1.0`, whose scores are [-1, -1] before and
        # [-1, 0] after.
        #
        # The one-sided case is only a crossing when the advantage that does exist is larger than
        # float64 noise, and that guard is not decoration. `_is_degenerate` compares the group's
        # standard deviation against the estimator's own epsilon, which is zero when the record does
        # not set one, so the verdict turns on whether `np.std` lands on exactly 0.0. It does not
        # always: three copies of 7.665180039911412 have a mean that is exact and a std of 0.0, and
        # the same three plus 14.0 have a mean one ulp high and a std of 3.55e-15, so a group that
        # is degenerate to the bit on both sides is judged degenerate on one and live on the other.
        # The advantages it produces there are -3.55e-15, which is not a contrast anybody trained
        # on. The tolerance is the kernel's `default_tol`, the same one the reach above uses, for
        # the reason the docstring gives: a second answer to this question would be a second
        # convention. Nothing here re-decides degeneracy, which stays the estimator's call.
        for i in range(len(a_before)):
            b_live = math.isfinite(a_before[i])
            a_live = math.isfinite(a_after[i])
            if b_live and a_live:
                n_comparable += 1
                worst = max(worst, abs(a_after[i] - a_before[i]))
            elif b_live or a_live:
                present = a_before[i] if b_live else a_after[i]
                scale = before_scores[i] if b_live else after_scores[i]
                if abs(present) > default_tol(scale):
                    n_state_changed += 1

    return NullLeak(
        node=chosen,
        constant=c,
        n=n,
        n_reached=n_reached,
        n_comparable=n_comparable,
        max_advantage_shift=float(worst),
        # Reach in {0, 1} for every group is the condition the invariance is exact under, and a
        # degeneracy crossing is conjoined rather than assumed away. A constant reaching every live
        # rollout of a group shifts them all by the same amount only when the offset node enters
        # every tree with the same weight; where it does not, the spread changes, and `cancels`
        # must not print "the advantages do not move" beside a group whose trainer handling flipped.
        cancels=all(r in (0.0, 1.0) for r in reaches) and n_state_changed == 0,
        per_group_reach=tuple(reaches),
        resolved=resolved,
        n_advantage_state_changed=n_state_changed,
    )


# ---------------------------------------------------------------------------
# grader.counterfactual_score
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class CounterfactualReading:
    """The counterfactual, its baseline, and how far the composition is from its declared group.

    ``result`` is the substrate's own `CounterfactualResult` held unchanged, so the numbers this
    instrument reports and the numbers the substrate computed are the same objects rather than two
    derivations that could drift.

    ``score_scale_fraction_moved`` is the mandatory baseline. The same comparison, on the deployed
    composition's own score scale, before the estimator centred anything. When it is much larger
    than the advantage-scale fraction, most of what removing the node did was absorbed by group
    centring and did not reach the update, which is a different claim from the headline and one the
    headline alone cannot make.
    """

    result: CounterfactualResult
    leak: NullLeak | None
    score_scale_fraction_moved: float
    score_scale_sign_reversed: int
    score_sd: float
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "notes", tuple(self.notes))

    def interpretation(self) -> str:
        r = self.result
        lines = [r.says()]
        lines.append(
            f"On the deployed composition's own score scale, before the estimator, "
            f"{self.score_scale_fraction_moved:.0%} of scores move by more than one score-scale "
            f"standard deviation ({self.score_sd:.4g}) and {self.score_scale_sign_reversed} change "
            f"sign. The advantage-scale numbers are the ones that reached the update: a shift that "
            f"survives group centring changed the gradient and one that does not, did not."
        )
        if self.leak is not None:
            lines.append(self.leak.says())
        for note in (*r.notes, *self.notes):
            lines.append(f"Note: {note}")
        return " ".join(lines)


_CF = "measure.composition.counterfactual"


def _score_scale(result: CounterfactualResult) -> tuple[float, int, float]:
    """The baseline: the same comparison, run on scores instead of advantages."""
    before = np.asarray(result.scores_before, dtype=float)
    after = np.asarray(result.scores_after, dtype=float)
    live = np.isfinite(before) & np.isfinite(after)
    if not live.any():
        return 0.0, 0, 0.0
    sd = float(np.std(before[np.isfinite(before)]))
    delta = np.abs(after[live] - before[live])
    moved = int((delta > sd).sum())
    reversed_ = int(((np.sign(before[live]) * np.sign(after[live])) < 0).sum())
    return moved / int(live.sum()), reversed_, sd


class CounterfactualComposition(CompositionInstrument):
    """B3's second half: what this run's advantages would have been without a piece of the reward.

    Rung 0. It re-evaluates the recorded tree with a node removed, on the leaves already on disk,
    and runs the group's own recorded estimator over both arms. No grader is called, no GPU is
    touched, and the cost is one pass over the record.

    The reading is in advantage space and not in score space, and that is the whole design. The
    question a lab asks about a composition constant is whether removing it would have changed the
    update, and the update sees advantages. A score change that group centring absorbs did not
    change anything. The score-scale comparison is computed too and reported as the baseline, so
    the two are visible side by side rather than one standing in for the other.
    """

    name = "CounterfactualComposition"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "B3"
    deviations = (
        "rung 1 of B3's ladder re-scores with a modified tree and needs GRADER: QUERY. It is not "
        "built here, so a counterfactual that changes what a grader would have said is out of "
        "scope and this instrument reports only what the recorded leaves support",
        "the reading is in advantage space. `spec/QUANTITIES.yaml` names the quantity "
        "`grader.counterfactual_score` and gives it the unit `reward`, which an advantage shares; "
        "the counterfactual *score* is not invariant under the declared `reward.null` group and "
        "the counterfactual advantage is, so advantage space is the only reading consistent with "
        "the declaration",
    )

    quantity = "grader.counterfactual_score"
    requires: dict[Component, Access] = B3_ACCESS
    substrates = frozenset({Substrate.COMPOSITE})
    phases = frozenset({Phase.POST_RUN})
    envelope = B3_ENVELOPE
    invariance = "reward.null"
    invariance_relation = INVARIANT
    baselines = (SCORE_SCALE_BASELINE,)
    rung = 0

    def __init__(
        self,
        groups: Sequence[GroupScores] = (),
        disabled: Iterable[str] = (),
        *,
        recorded_advantages: Sequence[Sequence[float | None]] | None = None,
        null_node: str | None = None,
        null_constant: float | None = None,
    ) -> None:
        self.groups = tuple(groups)
        self.disabled = tuple(sorted(set(disabled)))
        self.recorded_advantages = recorded_advantages
        self.null_node = null_node
        self.null_constant = None if null_constant is None else float(null_constant)

    def compute(self) -> Any:
        if not self.groups:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    "no groups were supplied, so there is no recorded composition to re-evaluate."
                ),
                remedy=(
                    "pass `groups=[...]` built with `record.scores.group_scores(group)` from a "
                    "recorded Group, one per group you want in the comparison."
                ),
            )
        if not self.disabled:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    "no node was named for removal, and a counterfactual with nothing disabled is "
                    "the baseline arm reported against itself."
                ),
                remedy=(
                    "pass `disabled={'length_override'}` naming at least one node. "
                    "`CompositionTree` reports the node inventory of a record if you do not know "
                    "what it contains."
                ),
            )
        result = counterfactual(
            self.groups,
            self.disabled,
            recorded_advantages=self.recorded_advantages,
        )
        if isinstance(result, Refusal):
            return result

        computed = null_leak(self.groups, node=self.null_node, constant=self.null_constant)
        notes: list[str] = []
        leak: NullLeak | None = None
        if isinstance(computed, Refusal):
            # A diagnostic that could not be computed must not sink the headline it was going to
            # qualify. The reason travels as a note instead.
            notes.append(
                f"the null-leak diagnostic could not be computed: {computed.detail} "
                f"{computed.remedy}"
            )
        else:
            leak = computed
            if not leak.resolved:
                notes.append(
                    f"the null-leak constant {leak.constant:g} is below the resolution of this "
                    f"run's own scores, so the reach it reports is floating-point rounding rather "
                    f"than a property of the composition and no leak is claimed."
                )
            elif not leak.cancels:
                notes.append(
                    "this composition does not satisfy its own declared `reward.null` invariance, "
                    "and the reason is in the reading rather than in the estimator: a constant "
                    "added to the task reward does not reach the rollouts a hard override pinned."
                )

        moved, reversed_, sd = _score_scale(result)
        return CounterfactualReading(
            result=result,
            leak=leak,
            score_scale_fraction_moved=moved,
            score_scale_sign_reversed=reversed_,
            score_sd=sd,
            notes=tuple(notes),
        )

    def payload(self, computed: CounterfactualReading) -> dict[str, Any]:
        r = computed.result
        out: dict[str, Any] = {
            "disabled": list(r.disabled),
            "n": r.n,
            "n_groups": r.n_groups,
            "n_trees_with_node": r.n_trees_with_node,
            "n_comparable": r.n_comparable,
            "fraction_moved": r.fraction_moved,
            "n_moved": r.n_moved,
            "n_sign_reversed": r.n_sign_reversed,
            "fraction_reversed": r.fraction_reversed,
            "sd_reference": r.sd_reference,
            "n_abstained_before": r.n_abstained_before,
            "n_abstained_after": r.n_abstained_after,
            "degenerate_groups_before": r.degenerate_groups_before,
            "degenerate_groups_after": r.degenerate_groups_after,
            "reproduction_error": r.reproduction_error,
            "says": r.says(),
            "interpretation": computed.interpretation(),
            "notes": list(r.notes) + list(computed.notes),
            "baselines": {SCORE_SCALE_BASELINE: float(computed.score_scale_fraction_moved)},
        }
        if computed.leak is not None:
            out["null_reach"] = computed.leak.reach
            out["null_cancels"] = computed.leak.cancels
            out["null_resolved"] = computed.leak.resolved
            out["null_max_advantage_shift"] = computed.leak.max_advantage_shift
            # The shift's denominator and its exception travel with it. A reader who gets the
            # maximum without the count it was taken over cannot tell a cancelling composition from
            # one where no rollout had an advantage on both sides of the offset, and those render
            # as the same 0.0.
            out["null_n_comparable"] = computed.leak.n_comparable
            out["null_advantage_state_changed"] = computed.leak.n_advantage_state_changed
            out["null_node"] = computed.leak.node
            out["null_constant"] = computed.leak.constant
        return out


class CompositionTree(CompositionInstrument):
    """B3's first half: what the reward is actually made of.

    Rung 0 and free. It reads the recorded tree and reports the inventory, the one-at-a-time
    ablation deltas, and the count that carries the argument: how many of the composition
    primitives in this reward a `Mapping[str, float]` of weights could not have expressed. The
    baseline is that mapping, counted, so the comparison is a number rather than a claim.

    Its declared group is `reward.null`, inherited from the quantity, and the honest note about the
    generated test that follows is this: the reading is structural, the group acts on values, so the
    test holds for a reason that has nothing to do with the composition. The real test of that
    declaration is on `grader.counterfactual_score`, where the reading goes through the estimator,
    and `null_leak` is what measures it.
    """

    name = "CompositionTree"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "B3"
    deviations = (
        "the ablation deltas are one-at-a-time and are reported as such. They sum to the total "
        "only when the tree is additive, and `is_additive` travels with them so the reader knows "
        "which case they are in",
    )

    quantity = "grader.composition"
    requires: dict[Component, Access] = COMPOSITION_ACCESS
    substrates = frozenset({Substrate.COMPOSITE})
    phases = frozenset({Phase.POST_RUN})
    envelope = B3_ENVELOPE
    invariance = "reward.null"
    invariance_relation = INVARIANT
    baselines = (WEIGHTS_DICT_BASELINE,)
    rung = 0

    def __init__(
        self,
        trees: Sequence[ScoreTree | None] = (),
        contexts: Sequence[ScoreContext] | None = None,
    ) -> None:
        self.trees = tuple(trees)
        self.contexts = None if contexts is None else tuple(contexts)

    @classmethod
    def over(cls, groups: Sequence[GroupScores]) -> "CompositionTree":
        """Build one over the trees and contexts of a set of groups, in order."""
        return cls(
            trees=[t for g in groups for t in g.trees],
            contexts=[c for g in groups for c in g.contexts],
        )

    def compute(self) -> Any:
        if not self.trees:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no trajectories were supplied, so there is no composition to read.",
                remedy=(
                    "pass `trees=[...]` from a recorded Group, or use "
                    "`CompositionTree.over(groups)`."
                ),
            )
        if all(t is None for t in self.trees):
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail=(
                    f"none of the {len(self.trees)} trajectories carries a score tree, so this "
                    f"record kept a total and not a composition. What the reward was made of is "
                    f"not recoverable from what it added up to."
                ),
                remedy=(
                    "instrument the grader with reward_lens.tap so the per-leaf scores are "
                    "recorded, and re-run. This is B3's kill condition: without per-leaf scores it "
                    "is a request rather than an instrument."
                ),
                statistics={"n_trees": len(self.trees), "n_with_composition": 0},
            )
        return summarise(self.trees, self.contexts)

    def payload(self, computed: CompositionSummary) -> dict[str, Any]:
        return {
            "n_trees": computed.n_trees,
            "n_nodes": computed.n_nodes,
            "n_leaves": computed.n_leaves,
            "node_types": dict(computed.node_types),
            "inexpressible": list(computed.inexpressible),
            "n_inexpressible": computed.n_inexpressible,
            "is_additive": computed.is_additive,
            "weights_dict_components": computed.weights_dict_components,
            "ablation": dict(computed.ablation),
            "n_missing_trees": computed.n_missing_trees,
            "says": computed.says(),
            "interpretation": computed.render(),
            "baselines": {WEIGHTS_DICT_BASELINE: float(computed.weights_dict_components)},
        }


# ---------------------------------------------------------------------------
# The generated invariance test's adaptation
# ---------------------------------------------------------------------------

#: Kimi K3's condition, as the fixture composition uses it: the score is pinned above a scaled
#: per-task token budget. It reads *features* rather than the score, which is what makes the region
#: membership independent of the constant the group action adds.
_OVER_BUDGET = PredicateRef(
    name="over_budget", feature="tokens", op=">", threshold_feature="b0", scale=1.5
)

#: The estimator the generated test replays: group-centred, no standard-deviation division, which
#: is `verifiers`' own transform (`rubric.py:406-409`) and the one case where the arithmetic of the
#: assertion is exact rather than approximate.
NULL_TEST_ESTIMATOR = EstimatorSpec(
    family="grpo", group_centred=True, std_normalised=False, degenerate_policy="skip"
)


def _payload_groups(
    payload: InvariancePayload,
    *,
    with_override: bool,
    estimator: EstimatorSpec,
) -> list[GroupScores]:
    """Turn the kernel's payload into groups whose task reward carries the payload's scores.

    The group acts on ``scores``, and the composition decides where that lands. It lands on the
    task reward, under whatever the composition puts above it, which is the only adaptation that
    makes the declared assertion a claim about the *composition* rather than a claim about the
    estimator. Adding the constant to the composed total instead would make every composition pass,
    override or not, and the test would assert nothing B3 is about.

    The two configurations differ in exactly one thing: whether a hard override sits above the task
    reward. Both carry the same two components and the counterfactual removes the same node in both,
    so the difference in the outcome cannot be attributed to anything else.
    """
    scores = np.asarray(payload.scores, dtype=float)
    gids = (
        np.zeros(scores.shape[0], dtype=int)
        if payload.group_ids is None
        else np.asarray(payload.group_ids)
    )
    out: list[GroupScores] = []
    for g in sorted({int(x) for x in gids.tolist()}):
        idx = [i for i, x in enumerate(gids.tolist()) if int(x) == g]
        trees: list[ScoreTree] = []
        ctxs: list[ScoreContext] = []
        for rank, i in enumerate(idx):
            task = WeightedSum(
                name="task",
                children=(
                    Leaf("component", float(scores[i]), None),
                    Leaf("format", 0.5, None),
                ),
                weights=(1.0, 0.25),
            )
            tree: ScoreTree = (
                Override(
                    name="length_override", condition=_OVER_BUDGET, constant=-1.0, otherwise=task
                )
                if with_override
                else task
            )
            trees.append(tree)
            # Every other rollout is over budget, so the override binds on half the group: the
            # constant reaches the other half and the shift cannot cancel.
            tokens = 400.0 if (with_override and rank % 2 == 1) else 100.0
            ctxs.append(
                ScoreContext(
                    features={"tokens": tokens, "b0": 100.0},
                    group=GroupContext(accuracy=0.5, k=len(idx), id=f"g{g}"),
                )
            )
        out.append(
            GroupScores(trees=tuple(trees), contexts=tuple(ctxs), estimator=estimator, id=f"g{g}")
        )
    return out


def null_probe(
    payload: InvariancePayload,
    *,
    with_override: bool = False,
    estimator: EstimatorSpec = NULL_TEST_ESTIMATOR,
) -> float:
    """The scalar `check_invariance` asserts `reward.null` on: one counterfactual advantage.

    The kernel reduces a reading to a scalar and this instrument's reading is a payload, so the
    projection is supplied here rather than left to each caller to invent. The scalar is the first
    rollout's counterfactual advantage, which *is* `grader.counterfactual_score` for that rollout,
    so the property being asserted is the declared one and not a convenient summary of it.

    ``with_override`` selects the composition, and it is the only thing it selects. The node the
    counterfactual removes is the same in both, so the difference in the outcome is attributable to
    the override and to nothing else. Both configurations are real: the first is the declaration
    holding exactly, the second is it failing, and the failure is a measurement of the composition
    rather than a defect in this instrument.
    """
    groups = _payload_groups(payload, with_override=with_override, estimator=estimator)
    inst = CounterfactualComposition(groups, {"format"})
    out = inst.compute()
    if isinstance(out, Refusal):  # pragma: no cover - the fixture estimator is always replayable
        raise AssertionError(f"the invariance probe was refused: {out.render()}")
    return float(out.result.advantages_after[0])


__all__ = [
    "ALL_SUBSTRATES",
    "B3_ACCESS",
    "B3_ENVELOPE",
    "COMPOSITION_ACCESS",
    "EXPRESSIBLE_BY_WEIGHTS",
    "NULL_TEST_ESTIMATOR",
    "PRIMITIVE_NOTES",
    "SCORE_SCALE_BASELINE",
    "WEIGHTS_DICT_BASELINE",
    "CompositionInstrument",
    "CompositionSummary",
    "CompositionTree",
    "CounterfactualComposition",
    "CounterfactualReading",
    "NullLeak",
    "auto_constant",
    "default_offset_node",
    "map_tree",
    "null_leak",
    "null_probe",
    "offset_node",
    "summarise",
]

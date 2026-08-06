"""The score tree: reward composition is an algebra, and removing one piece is a measurement.

`Mapping[str, float]` of weights is the data model every framework ships and it cannot express five
of the six composition primitives labs actually run: a hard override to a constant, an
automatic loss, lexicographic gating, piecewise-linear anchoring, and a gate conditioned on the
group rather than on the rollout. So the composition is a tree of eight node types, it is recorded
rather than summarised, and three things fall out of having it.

**The counterfactual.** `counterfactual(groups, disabled={"length_override"})` re-evaluates the tree
with one node removed, on the leaves already on disk, with no grader calls and no GPU. It runs the
group's own `EstimatorSpec` over both arms, because the question is what the *advantages* would have
been, not what the scores would have been, and a change that survives group centring is a different
claim from one that does not.

**The gradient dead zone.** An `Override` sets the score to a constant across a region, so inside
that region the derivative of the score with respect to anything under it is exactly zero. Finishing
perfectly and failing completely both score the constant. `deadzone` counts the region and measures
the spread of the underlying score inside it, which is the ordering information the override erased.

**Abstention.** `Leaf.value is None` with `abstained=True` is a distinct outcome from a zero. It
evaluates to `numpy.nan`, it is excluded from the group mean and standard deviation, and a group
where every leaf abstained warns rather than returning zeros. That is TRL's policy generalised. A
leaf can also carry `abstained=True` *and* a value, which is the shape of the defect instrument B4
counts: `verifiers` catches any exception from a reward function and sets `ans = 0.0`
(`rubrics/rubric.py:204-217`), so the framework used a real number and the record keeps both what it
used and the fact that the call failed.

What this module cannot do, said here rather than in a caveats page. It replays **group-relative**
estimators only: PPO-style advantages against a learned value baseline are not recoverable from
scores, and asking for one returns a refusal rather than a guess. It does not re-score anything, so
a counterfactual that changes what a grader would have said (rung 1 of B3's ladder) is out of scope
and needs `GRADER: QUERY`. One-at-a-time ablation deltas do not sum to the total unless the tree is
additive, and `is_additive` says which case you are in. And a predicate that is not a comparison
over recorded features has to have had its outcome recorded, because inventing it would be the
silent zero in another costume.

Two behaviours here are deliberately *not* TRL's.
`grpo_trainer.py` reduces partial-NaN reward rows with `nansum` (lines 2683 and 2716), which turns a
rollout that half-failed into a smaller real number; here a `WeightedSum` with one abstaining term is
NaN, because a total missing a term is not a smaller total. And its `nan_to_num(..., 0.0)` at line
2730 gives a masked rollout an advantage of exactly 0; this module leaves it NaN. For a sum-over-
tokens loss the two agree, since a zero advantage contributes nothing; they differ in the
denominator of any mean-over-batch normaliser, and that difference is a real one.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, Mapping, Sequence, Union

import numpy as np

from reward_lens.core.evidence import register_payload
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.record.schema import EstimatorSpec, Group, Trajectory

if TYPE_CHECKING:
    # Annotation only, so the record layer keeps no runtime dependency on the tap. The arrow points
    # this way by design: `tap/__init__.py` says the tap must not import the record, and the record
    # reading a tap record at the seam is what `GraderCallRef.from_call` is.
    from reward_lens.tap.contract import GraderCall

# ---------------------------------------------------------------------------
# Errors. None of these is a refusal: every one is a defect in the tree or in the call, not an
# anticipated property of the data, and the refusal architecture keeps the two apart on purpose.
# ---------------------------------------------------------------------------


class DuplicateNodeName(ValueError):
    """Two nodes in one tree share a name, so `disabled={name}` would be ambiguous."""


class UnknownNode(KeyError):
    """A counterfactual named a node that appears nowhere in the tree.

    Raised rather than ignored. Ignoring it returns the unchanged tree, which reads as "removing
    that node changes nothing" and is the most expensive possible way to be wrong here.
    """


class MissingFeature(KeyError):
    """A predicate reads a feature the context does not carry.

    Not False, and not zero. A missing feature silently evaluating to False disables an override
    for exactly the rollouts whose features were not recorded, which biases the dead-zone fraction
    downward in the direction that makes the instrument look reassuring.
    """


class MissingGroupStatistic(KeyError):
    """A `GroupGate` needs a group statistic the context does not carry."""


class AllAbstainedWarning(UserWarning):
    """Every rollout in a group abstained, so the group has no advantages at all.

    TRL warns here (`grpo_trainer.py`, the all-NaN row), and warning rather than raising is right:
    one dead group in a run of thousands is a data-quality fact, not a reason to stop.
    """


# ---------------------------------------------------------------------------
# What a grader call left behind
# ---------------------------------------------------------------------------


_RAW_REPR_CAP = 512


def _codec_safe(value: Any) -> tuple[Any, bool]:
    """Coerce a raw grader output into something `ValueCodec` can write, and say if it was coerced.

    The kernel's codec handles primitives, sequences, mappings, arrays and registered dataclasses,
    and raises on anything else. A grader is free to return an object none of those describe, so the
    record either coerces it or refuses to hold it. It coerces, to `repr` truncated at 512
    characters, and the second element of the return says so: a raw output that is a string because
    it always was and one that is a string because we could not store the real thing are different
    facts about the run.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value, False
    if isinstance(value, (list, tuple)):
        items = [_codec_safe(v) for v in value]
        return [p for p, _ in items], any(flag for _, flag in items)
    if isinstance(value, Mapping):
        pairs = {str(k): _codec_safe(v) for k, v in value.items()}
        return (
            {k: p for k, (p, _) in pairs.items()},
            any(flag for _, flag in pairs.values()),
        )
    return repr(value)[:_RAW_REPR_CAP], True


@register_payload
@dataclass(frozen=True)
class GraderCallRef:
    """What one grader call left in the record: facets, latency, cost and raw output.

    The composition schema prints `grader_call: GraderCall` on `Leaf`. The tap already owns a
    `GraderCall` (`tap/contract.py`) and it is a different object with a different job: it is a
    hot-path record holding the host's own arguments and return value **by reference**, built once
    per call at the 185 ns its own docstring measures, and that docstring says serialising those
    references is the drain's decision rather than the record's. This is that decision, taken once,
    in the layer that persists things. Build one with `from_call`.

    ``facets`` is the crossed-design vocabulary: rater, occasion, rubric draw, order, seed. A5 and
    A2 read it to separate variance components, and a grader wrapper that records no facets makes
    the crossed G-study unavailable rather than wrong.
    """

    grader: str
    #: One of the tap's `CallOutcome` values: returned, raised, timed_out, sentinel.
    outcome: str = "returned"
    facets: Mapping[str, Any] = field(default_factory=dict)
    latency_s: float | None = None
    cost_usd: float | None = None
    raw: Any = None
    #: True when ``raw`` is a `repr` because the real output was not codec-safe.
    raw_is_repr: bool = False
    error_type: str | None = None
    error_message: str | None = None
    #: The tap sequence number and step, so a leaf can be traced back to the ring record it came
    #: from. Both are None for a call reconstructed by a converter rather than observed by a tap.
    seq: int | None = None
    step: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "facets", dict(self.facets))

    @property
    def raised(self) -> bool:
        """Whether the call failed, however the framework went on to present it downstream."""
        return self.outcome in ("raised", "timed_out")

    @classmethod
    def from_call(
        cls,
        call: "GraderCall",
        *,
        cost_usd: float | None = None,
        keep_raw: bool = True,
    ) -> "GraderCallRef":
        """Project a live tap record into the record layer.

        ``cost_usd`` is not on the tap's record because the tap cannot know it: a hosted judge's
        price is a property of the account, not of the call. Whoever knows it passes it here.
        """
        raw, is_repr = _codec_safe(call.value) if keep_raw else (None, False)
        return cls(
            grader=call.grader,
            outcome=call.outcome.value,
            facets=dict(call.facets or {}),
            latency_s=call.inner_ns / 1e9,
            cost_usd=cost_usd,
            raw=raw,
            raw_is_repr=is_repr,
            error_type=call.error_type,
            error_message=call.error_message,
            seq=call.seq,
            step=call.step,
        )


# ---------------------------------------------------------------------------
# The context a tree is evaluated against
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class GroupContext:
    """The group's statistics, because a `GroupGate` reads them and a rollout cannot.

    ERNIE's WPSM masks a sample when group accuracy exceeds a threshold **and** that sample's own
    entropy falls below another. The first number belongs to the group and the second to the
    rollout, which is why they arrive on different objects.

    ``accuracy`` is the fraction of the group that solved the task, on whatever definition of solved
    the run used. It is `None` when the record does not carry one, and a gate that needs it then
    raises rather than defaulting: a gate that silently never fires is a gate you cannot audit.
    """

    accuracy: float | None = None
    k: int = 0
    id: str = ""
    extra: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", dict(self.extra))


@register_payload
@dataclass(frozen=True)
class ScoreContext:
    """Everything a tree needs that is not in the tree: this rollout's features and its group.

    ``features`` is the recorded per-rollout numbers a predicate can read: token count, entropy,
    the task's own budget `b0(x)`, whatever the run recorded. It maps directly onto
    `Trajectory.features`.

    ``predicates`` holds the recorded outcome of any condition that is not a comparison over
    features. A judge's verdict that a citation was fabricated is a predicate; there is no
    expression over floats that recomputes it, so either the run recorded what it decided or the
    tree cannot be evaluated. Guessing is not on the menu.
    """

    features: Mapping[str, float] = field(default_factory=dict)
    predicates: Mapping[str, bool] = field(default_factory=dict)
    group: GroupContext = field(default_factory=GroupContext)

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", dict(self.features))
        object.__setattr__(self, "predicates", dict(self.predicates))


_OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}

Op = Literal[">", ">=", "<", "<=", "==", "!="]


@register_payload
@dataclass(frozen=True)
class PredicateRef:
    """A condition, recorded so it can be re-evaluated without re-running anything.

    Two shapes, and the difference matters. A **comparison** over recorded features re-evaluates for
    free at any threshold, which is what makes a sensitivity analysis of the frontier's composition
    constants possible at all: Kimi K3's override fires when the token budget exceeds `tau * b0(x)`,
    and moving `tau` is `replace(pred, scale=0.8)` and one more pass over the same leaves. An
    **opaque** predicate (``feature is None``) carries only a name and reads its recorded outcome out
    of the context, because a judge's verdict is not a function of any float on the record.

    ``expr`` is for humans and for the card. It is derived when not supplied.
    """

    name: str
    feature: str | None = None
    op: Op = ">"
    threshold: float | None = None
    #: The right-hand side read from a feature instead of a constant, for a per-task budget like
    #: `b0(x)`. Multiplied by ``scale``, which is the tunable constant.
    threshold_feature: str | None = None
    scale: float = 1.0
    expr: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "a PredicateRef must be named; the counterfactual addresses it by name"
            )
        if self.feature is not None:
            if self.op not in _OPS:
                raise ValueError(f"unknown comparison {self.op!r}; one of {sorted(_OPS)}")
            if self.threshold is None and self.threshold_feature is None:
                raise ValueError(
                    f"predicate {self.name!r} compares feature {self.feature!r} against nothing. "
                    f"Give it a threshold, or a threshold_feature for a per-task budget."
                )
            if self.threshold is not None and self.threshold_feature is not None:
                raise ValueError(
                    f"predicate {self.name!r} has both a constant threshold and a threshold "
                    f"feature. Which one is the right-hand side is not something to infer."
                )
        if not self.expr:
            object.__setattr__(self, "expr", self._render())

    def _render(self) -> str:
        if self.feature is None:
            return f"recorded({self.name})"
        rhs = self.threshold_feature if self.threshold_feature is not None else self.threshold
        scale = "" if self.scale == 1.0 else f"{self.scale:g} * "
        return f"{self.feature} {self.op} {scale}{rhs}"

    @property
    def is_opaque(self) -> bool:
        return self.feature is None

    def holds(self, ctx: ScoreContext) -> bool:
        """Whether the condition fired for this rollout."""
        if self.feature is None:
            if self.name not in ctx.predicates:
                raise MissingFeature(
                    f"predicate {self.name!r} is opaque, so its outcome has to have been recorded, "
                    f"and this context carries {sorted(ctx.predicates)}. There is no expression "
                    f"over the recorded features that recomputes it."
                )
            return bool(ctx.predicates[self.name])
        if self.feature not in ctx.features:
            raise MissingFeature(
                f"predicate {self.name!r} reads feature {self.feature!r}, which this context does "
                f"not carry; it has {sorted(ctx.features)}. A missing feature is not False: "
                f"treating it as False disables the gate for exactly the rollouts whose features "
                f"went unrecorded."
            )
        if self.threshold_feature is not None:
            if self.threshold_feature not in ctx.features:
                raise MissingFeature(
                    f"predicate {self.name!r} compares against feature "
                    f"{self.threshold_feature!r}, which this context does not carry."
                )
            rhs = self.scale * float(ctx.features[self.threshold_feature])
        else:
            rhs = self.scale * float(self.threshold or 0.0)
        return bool(_OPS[self.op](float(ctx.features[self.feature]), rhs))


# ---------------------------------------------------------------------------
# The eight nodes
# ---------------------------------------------------------------------------


class _Absent:
    """A subtree that was disabled. Not zero, not NaN, and not a value the tree ever produces.

    Zero would be a contribution and NaN would be an abstention. What a disabled node is, is not
    there, so its parent supplies its own identity: a sum drops the term, a product drops the
    factor, and a node with no surviving children is itself absent.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return "ABSENT"


_ABSENT = _Absent()
_Value = Union[float, _Absent]


@register_payload
@dataclass(frozen=True)
class Leaf:
    """One recorded score from one grader call.

    ``value is None`` means the grader did not produce a number, and then ``abstained`` must be
    True. The reverse is not required, and that asymmetry is the whole point: a leaf with
    ``abstained=True`` and ``value=0.0`` is a grader that raised and a framework that substituted a
    zero, which is exactly what `verifiers` does at `rubrics/rubric.py:204-217`. Both facts are kept
    so B4 can report the rate and the baseline "assume zero and show the difference" can be computed
    rather than asserted. Evaluation uses neither: an abstaining leaf is NaN whatever number the
    framework wrote next to it.

    A non-finite ``value`` also requires ``abstained``, because a NaN that arrives without one is
    indistinguishable downstream from a leaf that was never scored.
    """

    name: str
    value: float | None
    grader_call: GraderCallRef | None
    abstained: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("every node needs a name; the counterfactual addresses nodes by name")
        if self.value is None and not self.abstained:
            raise ValueError(
                f"leaf {self.name!r} has no value and does not say it abstained. Those are the same "
                f"event and recording only one of them is how a failed grader becomes a zero: set "
                f"abstained=True."
            )
        if self.value is not None and not math.isfinite(float(self.value)) and not self.abstained:
            raise ValueError(
                f"leaf {self.name!r} carries a non-finite value ({self.value}) and does not say it "
                f"abstained. Downstream those are the same row, so say which one this is."
            )

    @property
    def silent_zero(self) -> bool:
        """The call failed and a number was recorded anyway. B4's numerator."""
        return self.abstained and self.value is not None

    @property
    def substituted(self) -> float | None:
        """What the framework used in place of the score it did not get."""
        return self.value if self.abstained else None


@register_payload
@dataclass(frozen=True)
class WeightedSum:
    """The one primitive a weights dict can express. `bias` is the constant term.

    A term whose child abstains makes the whole sum NaN. That is deliberate and it is a departure
    from TRL, whose `nansum` (`grpo_trainer.py:2683` and `:2716`) drops the missing term and returns
    the rest, which is a smaller number that looks like a real one. A total missing a term is not a
    smaller total.
    """

    name: str
    children: tuple["ScoreTree", ...]
    weights: tuple[float, ...] = ()
    bias: float = 0.0

    def __post_init__(self) -> None:
        _check_name(self.name)
        object.__setattr__(self, "children", tuple(self.children))
        weights = tuple(float(w) for w in self.weights) or (1.0,) * len(self.children)
        if len(weights) != len(self.children):
            raise ValueError(
                f"{self.name!r}: {len(weights)} weights for {len(self.children)} children. Leave "
                f"weights empty for all ones; a mismatch is a term silently dropped or repeated."
            )
        object.__setattr__(self, "weights", weights)
        _check_unique(self)


@register_payload
@dataclass(frozen=True)
class Product:
    """Multiplicative zeroing: a gate multiplies the total by zero.

    An abstaining factor makes the product NaN even when another factor is exactly zero. Arithmetic
    says `0 * x = 0` for any finite `x`, so this is a choice: a rollout whose grader failed must not
    receive an advantage, and a zero here is indistinguishable from the zero the gate produced.
    """

    name: str
    children: tuple["ScoreTree", ...]

    def __post_init__(self) -> None:
        _check_name(self.name)
        object.__setattr__(self, "children", tuple(self.children))
        if not self.children:
            raise ValueError(
                f"{self.name!r}: an empty product is 1.0 by convention and 0.0 by accident. Give "
                f"it factors."
            )
        _check_unique(self)


@register_payload
@dataclass(frozen=True)
class Override:
    """The one that produces a gradient dead zone.

    Kimi K3: override the task reward with -1 for trajectories whose total token budget exceeds
    `tau * b0(x)`. Inside the region the score is the constant, so the derivative of the score with
    respect to anything under ``otherwise`` is exactly zero there, and finishing perfectly and
    failing completely score the same. That is the mechanism `deadzone` measures.

    The constant wins over an abstention, and this is worth knowing rather than discovering: the
    pipeline does not call the grader for an overridden rollout, or calls it and discards the
    answer, so a grader failure inside the region is invisible to the abstention census. The census
    reports the count of leaves it could not see for that reason.
    """

    name: str
    condition: PredicateRef
    constant: float
    otherwise: "ScoreTree"

    def __post_init__(self) -> None:
        _check_name(self.name)
        _check_unique(self)


@register_payload
@dataclass(frozen=True)
class Lexicographic:
    """Ling 2.6: `R_length = p(l)` if correct, `min(p(l), 0)` if incorrect.

    The primary term sets a tier and the secondary can only move the score inside it. With
    ``include_primary=False``, ``threshold=1.0`` and ``cap=0.0`` this is exactly the printed rule:
    the length shaping is free to help a correct answer and can only hurt an incorrect one.

    Disabling the node removes the gating and lets the secondary through unclamped, which is the
    counterfactual "what if length shaping were not conditioned on correctness".
    """

    name: str
    primary: "ScoreTree"
    secondary: "ScoreTree"
    threshold: float = 1.0
    cap: float = 0.0
    include_primary: bool = True

    def __post_init__(self) -> None:
        _check_name(self.name)
        _check_unique(self)


@register_payload
@dataclass(frozen=True)
class Piecewise:
    """Piecewise-linear anchoring: map a raw score onto an anchored scale through knots.

    ``knots`` are `(x, y)` pairs with strictly increasing `x`. Outside the range, ``extrapolate``
    decides: "clamp" holds the end value, which creates a dead zone of its own at each end, and
    "linear" continues the end segment's slope.
    """

    name: str
    child: "ScoreTree"
    knots: tuple[tuple[float, float], ...] = ()
    extrapolate: Literal["clamp", "linear"] = "clamp"

    def __post_init__(self) -> None:
        _check_name(self.name)
        knots = tuple((float(x), float(y)) for x, y in self.knots)
        if len(knots) < 2:
            raise ValueError(
                f"{self.name!r}: a piecewise-linear map needs at least two knots; got {len(knots)}."
            )
        xs = [x for x, _ in knots]
        if any(b <= a for a, b in zip(xs, xs[1:])):
            raise ValueError(
                f"{self.name!r}: knot x values must strictly increase; got {xs}. A repeated x is "
                f"two different outputs for one input."
            )
        object.__setattr__(self, "knots", knots)
        _check_unique(self)

    def apply(self, x: float) -> float:
        xs = [k[0] for k in self.knots]
        ys = [k[1] for k in self.knots]
        if x <= xs[0]:
            if self.extrapolate == "clamp":
                return ys[0]
            slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
            return ys[0] + slope * (x - xs[0])
        if x >= xs[-1]:
            if self.extrapolate == "clamp":
                return ys[-1]
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            return ys[-1] + slope * (x - xs[-1])
        i = int(np.searchsorted(xs, x, side="right")) - 1
        span = xs[i + 1] - xs[i]
        return ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / span


@register_payload
@dataclass(frozen=True)
class GroupGate:
    """ERNIE's WPSM: mask a sample when group accuracy exceeds tau and its entropy falls below eta.

    Both declared conditions must hold. A condition left at None is not part of the gate; a gate
    with neither is a gate that never fires and is refused at construction.

    ``mode`` separates the two things a gate can do, and they are not the same. "mask" removes the
    rollout from the update, which is what WPSM does and what this library treats as the honest
    behaviour. "zero" writes a real 0.0 into the group, which flows into the mean and the standard
    deviation and therefore into every other rollout's advantage. Recording "zero" where the
    pipeline masked is the same defect as a silent zero, one level up the tree.
    """

    name: str
    child: "ScoreTree"
    accuracy_above: float | None = None
    entropy_below: float | None = None
    mode: Literal["mask", "zero"] = "mask"
    entropy_feature: str = "entropy"

    def __post_init__(self) -> None:
        _check_name(self.name)
        if self.accuracy_above is None and self.entropy_below is None:
            raise ValueError(
                f"{self.name!r}: a group gate with no condition never fires. Declare "
                f"accuracy_above, entropy_below, or both."
            )
        _check_unique(self)

    def fires(self, ctx: ScoreContext) -> bool:
        if self.accuracy_above is not None:
            if ctx.group.accuracy is None:
                raise MissingGroupStatistic(
                    f"gate {self.name!r} fires on group accuracy above {self.accuracy_above}, and "
                    f"the group context carries no accuracy. A gate that cannot read its condition "
                    f"is not a gate that never fires."
                )
            if not ctx.group.accuracy > self.accuracy_above:
                return False
        if self.entropy_below is not None:
            if self.entropy_feature not in ctx.features:
                raise MissingFeature(
                    f"gate {self.name!r} fires on {self.entropy_feature!r} below "
                    f"{self.entropy_below}, and this rollout carries {sorted(ctx.features)}."
                )
            if not float(ctx.features[self.entropy_feature]) < self.entropy_below:
                return False
        return True


@register_payload
@dataclass(frozen=True)
class Mask:
    """The rollout is removed from the update. It receives no advantage, not a zero one.

    DAPO's overlong loss masking is this node with no condition: the record says the rollout was
    masked. With a condition it is the general form, and disabling it is the counterfactual "what
    if these rollouts had been trained on".
    """

    name: str
    child: "ScoreTree"
    when: PredicateRef | None = None

    def __post_init__(self) -> None:
        _check_name(self.name)
        _check_unique(self)


ScoreTree = Union[Leaf, WeightedSum, Product, Override, Lexicographic, Piecewise, GroupGate, Mask]

_CONTAINERS = (WeightedSum, Product, Override, Lexicographic, Piecewise, GroupGate, Mask)


def _check_name(name: str) -> None:
    if not name.strip():
        raise ValueError("every node needs a name; the counterfactual addresses nodes by name")


def _children_of(node: "ScoreTree") -> tuple["ScoreTree", ...]:
    if isinstance(node, Leaf):
        return ()
    if isinstance(node, (WeightedSum, Product)):
        return node.children
    if isinstance(node, Override):
        return (node.otherwise,)
    if isinstance(node, Lexicographic):
        return (node.primary, node.secondary)
    return (node.child,)


def _check_unique(node: "ScoreTree") -> None:
    """Names are unique within a tree, checked bottom up as each parent is constructed.

    Wanting a leaf called "format" under each of two rubrics is reasonable, and it makes
    `disabled={"format"}` mean two removals at once. Qualify them instead.
    """
    seen: set[str] = {node.name}
    for child in _children_of(node):
        for name in node_names(child):
            if name in seen:
                raise DuplicateNodeName(
                    f"node name {name!r} appears more than once under {node.name!r}. The "
                    f"counterfactual addresses nodes by name, so a duplicate makes "
                    f"disabled={{{name!r}}} ambiguous. Qualify them, for example "
                    f"'rubric_a.{name}' and 'rubric_b.{name}'."
                )
            seen.add(name)


# ---------------------------------------------------------------------------
# Walking
# ---------------------------------------------------------------------------


def walk(tree: "ScoreTree") -> Iterator["ScoreTree"]:
    """Every node, parents before children, in declaration order."""
    yield tree
    for child in _children_of(tree):
        yield from walk(child)


def node_names(tree: "ScoreTree") -> tuple[str, ...]:
    return tuple(n.name for n in walk(tree))


def leaves(tree: "ScoreTree") -> tuple[Leaf, ...]:
    return tuple(n for n in walk(tree) if isinstance(n, Leaf))


def find(tree: "ScoreTree", name: str) -> "ScoreTree | None":
    for node in walk(tree):
        if node.name == name:
            return node
    return None


def is_additive(tree: "ScoreTree") -> bool:
    """True when the tree is a weighted sum of leaves and nothing else.

    Only then do one-at-a-time ablation deltas sum to the total, and only then does a weights dict
    lose nothing. Measured on real compositions this is usually False, which is the argument for
    the whole tree in one predicate.
    """
    return all(isinstance(n, (Leaf, WeightedSum)) for n in walk(tree))


def overrides(tree: "ScoreTree") -> tuple[Override, ...]:
    return tuple(n for n in walk(tree) if isinstance(n, Override))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    tree: "ScoreTree",
    ctx: ScoreContext,
    disabled: Iterable[str] = (),
) -> float:
    """The score this composition produced for this rollout, or NaN if it produced none.

    NaN means the rollout receives no advantage: a grader abstained, a mask fired, or a gate removed
    it. It is never a zero and never an exception.

    ``disabled`` names nodes to remove. A removed node contributes nothing at all rather than zero,
    so its parent supplies the identity: a sum drops the term and its weight, a product drops the
    factor, an override falls through to ``otherwise``, a mask and a gate pass their child through,
    a piecewise map becomes the identity, and a lexicographic gate stops clamping. Nothing is
    re-scored and no grader is called: this reads the leaves already on the record.
    """
    names = frozenset(disabled)
    if names:
        present = frozenset(node_names(tree))
        missing = sorted(names - present)
        if missing:
            raise UnknownNode(
                f"cannot disable {missing}: no such node in this tree. It has {sorted(present)}. "
                f"Silently ignoring the name would report that removing it changed nothing."
            )
    out = _eval(tree, ctx, names)
    if isinstance(out, _Absent):
        raise ValueError(
            f"disabling {sorted(names)} removes the whole composition, so there is no score left to "
            f"compare against. Disable a node inside the tree rather than the root of it."
        )
    return out


def _eval(node: "ScoreTree", ctx: ScoreContext, disabled: frozenset[str]) -> _Value:
    off = node.name in disabled

    if isinstance(node, Leaf):
        if off:
            return _ABSENT
        if node.abstained or node.value is None:
            return math.nan
        return float(node.value)

    if isinstance(node, WeightedSum):
        if off:
            return _ABSENT
        total = node.bias
        any_live = False
        for child, weight in zip(node.children, node.weights):
            v = _eval(child, ctx, disabled)
            if isinstance(v, _Absent):
                continue
            any_live = True
            total += weight * v
        # Every term removed and no constant term left: nothing here, rather than a zero.
        if not any_live and node.bias == 0.0:
            return _ABSENT
        return total

    if isinstance(node, Product):
        if off:
            return _ABSENT
        total = 1.0
        any_live = False
        for child in node.children:
            v = _eval(child, ctx, disabled)
            if isinstance(v, _Absent):
                continue
            any_live = True
            total *= v
        return total if any_live else _ABSENT

    if isinstance(node, Override):
        base = _eval(node.otherwise, ctx, disabled)
        if off:
            return base
        return node.constant if node.condition.holds(ctx) else base

    if isinstance(node, Lexicographic):
        primary = _eval(node.primary, ctx, disabled)
        secondary = _eval(node.secondary, ctx, disabled)
        if isinstance(secondary, _Absent):
            return primary if node.include_primary else _ABSENT
        if off or isinstance(primary, _Absent):
            gated = secondary
        elif math.isnan(primary):
            # The tier is unknown, so whether the secondary is capped is unknown too.
            gated = math.nan
        else:
            gated = secondary if primary >= node.threshold else min(secondary, node.cap)
        if node.include_primary and not isinstance(primary, _Absent):
            return primary + gated
        return gated

    if isinstance(node, Piecewise):
        v = _eval(node.child, ctx, disabled)
        if isinstance(v, _Absent) or off:
            return v
        return math.nan if math.isnan(v) else node.apply(v)

    if isinstance(node, GroupGate):
        v = _eval(node.child, ctx, disabled)
        if isinstance(v, _Absent) or off:
            return v
        if not node.fires(ctx):
            return v
        return math.nan if node.mode == "mask" else 0.0

    if isinstance(node, Mask):
        v = _eval(node.child, ctx, disabled)
        if isinstance(v, _Absent) or off:
            return v
        if node.when is None or node.when.holds(ctx):
            return math.nan
        return v

    raise TypeError(f"not a score tree node: {type(node).__name__}")


def ablation_deltas(tree: "ScoreTree", ctx: ScoreContext) -> dict[str, float]:
    """What each node is worth, one at a time: `score(tree) - score(tree without node)`.

    This is the cheap answer to "what is the reward actually made of". It is one-at-a-time, so the
    deltas sum to the total only when `is_additive(tree)` is True, and on a tree with a gate or an
    override they will not: an override that binds is worth the whole difference and everything
    under it is worth nothing. That is the finding, not a defect in the arithmetic.

    A node whose removal empties the tree is skipped, since there is no score to difference against.
    """
    base = evaluate(tree, ctx)
    out: dict[str, float] = {}
    for node in walk(tree):
        if node.name == tree.name:
            continue
        try:
            out[node.name] = base - evaluate(tree, ctx, {node.name})
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# B4's substrate: abstention, and the zeros that were not scores
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class AbstentionCensus:
    """How often the grader failed, and how often a real number was recorded anyway.

    ``n_silent_zero`` is B4's numerator: leaves whose call failed and whose recorded value is a
    number. ``substituted_total`` is what those numbers add up to, which is the baseline "assume
    zero and show the difference" computed rather than assumed.

    ``n_unattributable`` counts leaves whose outcome is not knowable: no `GraderCallRef`, and no
    ``abstained`` flag either. They are held out of the rate denominators and never folded into the
    successes, because a rate that treats unknown outcomes as successes is biased downward by
    exactly the amount that makes it reassuring.

    **A leaf that says ``abstained=True`` is a known outcome whether or not a call was recorded
    beside it**, because the leaf is itself the record of the outcome. Counting it as unknown and as
    an abstention at the same time is the arithmetic that returned `abstention_rate` 4.0 with an
    upper bound of 1.6 on a five-leaf record: the numerator held four leaves the denominator had
    removed. Those leaves are counted in ``n_abstained_unattributed`` instead, so the reader still
    sees that nothing says which grader declined; they show up under ``by_grader['unknown']``.
    The producer is `tap.adapters.trl`, whose ``refs.get(name)`` is None for any reward function
    the tap did not see a call from, on the same rows where TRL wrote no score.

    ``n_shadowed`` counts leaves that sit under an override that bound for that rollout. The
    pipeline never used their value, so their outcome tells you nothing about the run's live grader
    reliability, and counting them either way is wrong.
    """

    n_leaves: int
    n_abstained: int
    n_silent_zero: int
    n_unattributable: int
    n_shadowed: int
    substituted_total: float
    by_grader: Mapping[str, int] = field(default_factory=dict)
    n_abstained_unattributed: int = 0

    @property
    def n_known(self) -> int:
        """Leaves whose outcome this record can state. The denominator of both rates."""
        return self.n_leaves - self.n_unattributable

    @property
    def n_no_call_record(self) -> int:
        """Leaves with no `GraderCallRef`, knowable outcome or not. A data-quality count."""
        return self.n_unattributable + self.n_abstained_unattributed

    @property
    def abstention_rate(self) -> float | None:
        """Abstentions over the leaves whose outcome is knowable. None when none are."""
        known = self.n_known
        return None if known <= 0 else self.n_abstained / known

    @property
    def silent_zero_rate(self) -> float | None:
        known = self.n_known
        return None if known <= 0 else self.n_silent_zero / known

    def render(self) -> str:
        rate = self.silent_zero_rate
        head = (
            "no leaf has a recorded call outcome"
            if rate is None
            else (
                f"{rate:.1%} of {self.n_known} scored leaves were a number "
                f"the grader did not produce"
            )
        )
        unattributed = (
            f", {self.n_abstained_unattributed} of them with nothing saying which grader declined"
            if self.n_abstained_unattributed
            else ""
        )
        return (
            f"{head}. {self.n_abstained} abstentions{unattributed}, {self.n_silent_zero} of them "
            f"carrying a substituted value totalling {self.substituted_total:.4g}. "
            f"{self.n_unattributable} leaves carry no call record and no abstention flag, so their "
            f"outcome is unknown; {self.n_shadowed} sit under a binding override and were never "
            f"used."
        )


def census(
    trees: Sequence["ScoreTree | None"],
    contexts: Sequence[ScoreContext] | None = None,
) -> AbstentionCensus:
    """Count abstentions and silent zeros across a set of recorded trees.

    ``contexts`` is optional and only changes ``n_shadowed``: without it, no leaf can be known to
    have been overridden, so none is reported as shadowed.

    A leaf is unattributable only when nothing in it states an outcome: no call record **and** no
    abstention flag. An abstaining leaf with no call record states its outcome and is counted in
    ``n_abstained_unattributed``, which is the count that keeps the two sets disjoint and the rates
    in [0, 1].
    """
    if contexts is not None and len(contexts) != len(trees):
        raise ValueError(
            f"{len(trees)} trees against {len(contexts)} contexts. One context per tree, in the "
            f"same order, or none at all."
        )
    n_leaves = n_abstained = n_silent = n_unattr = n_shadowed = n_abstained_unattr = 0
    substituted = 0.0
    by_grader: dict[str, int] = {}
    for i, tree in enumerate(trees):
        if tree is None:
            continue
        shadowed: set[str] = set()
        if contexts is not None:
            ctx = contexts[i]
            for node in overrides(tree):
                if node.condition.holds(ctx):
                    shadowed.update(n.name for n in walk(node.otherwise))
        for leaf_node in leaves(tree):
            n_leaves += 1
            if leaf_node.name in shadowed:
                n_shadowed += 1
            if leaf_node.grader_call is None and not leaf_node.abstained:
                n_unattr += 1
            if leaf_node.abstained:
                n_abstained += 1
                if leaf_node.grader_call is None:
                    n_abstained_unattr += 1
                grader = (
                    leaf_node.grader_call.grader if leaf_node.grader_call is not None else "unknown"
                )
                by_grader[grader] = by_grader.get(grader, 0) + 1
                if leaf_node.value is not None and math.isfinite(float(leaf_node.value)):
                    n_silent += 1
                    substituted += float(leaf_node.value)
    return AbstentionCensus(
        n_leaves=n_leaves,
        n_abstained=n_abstained,
        n_silent_zero=n_silent,
        n_unattributable=n_unattr,
        n_shadowed=n_shadowed,
        substituted_total=substituted,
        by_grader=by_grader,
        n_abstained_unattributed=n_abstained_unattr,
    )


# ---------------------------------------------------------------------------
# I3's substrate: the gradient dead zone
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class DeadZone:
    """One override's region, and what it cost.

    ``fraction`` is `gate.deadzone_fraction`: the share of rollouts inside
    the region, where the score is the constant and the derivative with respect to task quality is
    exactly zero.

    ``n_binding`` is the sharper number, and it is not the same. A rollout can be inside the region
    and still score the constant by coincidence, or sit under a second override that wins. Binding
    means removing this node changes this rollout's score.

    ``erased_spread`` is the standard deviation of the underlying score inside the region: the
    ordering information the override destroyed. A graded penalty preserves ordering by success
    inside the region and an override does not, so this is the size of that difference on this run.
    """

    node: str
    condition: str
    constant: float
    n: int
    n_region: int
    n_binding: int
    erased_spread: float | None
    distinct_underlying: int

    @property
    def fraction(self) -> float:
        return self.n_region / self.n if self.n else 0.0

    def render(self) -> str:
        spread = "unknown" if self.erased_spread is None else f"{self.erased_spread:.4g}"
        return (
            f"{self.fraction:.1%} of {self.n} rollouts ({self.n_region}) sit where "
            f"{self.condition} holds and the score is pinned to {self.constant:g}, so the gradient "
            f"with respect to task quality is exactly zero there. It binds for {self.n_binding}. "
            f"The underlying score inside the region spans {self.distinct_underlying} distinct "
            f"values with spread {spread}, and that ordering is what the override erased."
        )


def deadzone(
    trees: Sequence["ScoreTree | None"],
    contexts: Sequence[ScoreContext],
    *,
    node: str | None = None,
) -> tuple[DeadZone, ...]:
    """Measure every override's dead zone, or one named override's.

    Free: it reads the recorded leaves and the recorded features, and calls nothing.
    """
    pairs = [(t, c) for t, c in zip(trees, contexts) if t is not None]
    if not pairs:
        return ()
    names: list[str] = []
    for tree, _ in pairs:
        for ov in overrides(tree):
            if node is not None and ov.name != node:
                continue
            if ov.name not in names:
                names.append(ov.name)
    out: list[DeadZone] = []
    for name in names:
        n = n_region = n_binding = 0
        underlying: list[float] = []
        condition = ""
        constant = math.nan
        for tree, ctx in pairs:
            found = find(tree, name)
            if not isinstance(found, Override):
                continue
            n += 1
            condition = found.condition.expr
            constant = found.constant
            if not found.condition.holds(ctx):
                continue
            n_region += 1
            with_it = evaluate(tree, ctx)
            without = evaluate(tree, ctx, {name})
            if not _same(with_it, without):
                n_binding += 1
            base = _eval(found.otherwise, ctx, frozenset())
            if not isinstance(base, _Absent) and math.isfinite(base):
                underlying.append(float(base))
        spread = float(np.std(underlying)) if len(underlying) > 1 else None
        out.append(
            DeadZone(
                node=name,
                condition=condition,
                constant=constant,
                n=n,
                n_region=n_region,
                n_binding=n_binding,
                erased_spread=spread,
                distinct_underlying=len({round(u, 12) for u in underlying}),
            )
        )
    return tuple(out)


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


# ---------------------------------------------------------------------------
# Replaying the estimator
# ---------------------------------------------------------------------------

_REPLAY = "record.scores.replay_advantages"


def replay_advantages(
    scores: Sequence[float],
    spec: EstimatorSpec,
    *,
    where: str = "",
) -> tuple[float, ...] | Refusal:
    """Turn one group's scores into its advantages, exactly as the recorded estimator would have.

    NaN scores are abstentions: excluded from the mean and the standard deviation, and given a NaN
    advantage rather than a zero one. A group where every score is NaN warns and returns all NaN,
    which is TRL's behaviour and the reason it is the model here.

    Group-relative estimators only. An advantage computed against a learned value baseline is not a
    function of the group's scores, so this refuses rather than inventing one, and the refusal says
    what to do instead.
    """
    arr = np.asarray([float(s) for s in scores], dtype=float)
    live = np.isfinite(arr)
    subject = f" for {where}" if where else ""

    if not spec.group_centred:
        return Refusal(
            instrument=_REPLAY,
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"the recorded estimator{subject} is not group-relative (family={spec.family!r}, "
                f"group_centred=False), so an advantage is not a function of this group's scores "
                f"alone and re-deriving one would be an invention."
            ),
            remedy=(
                "Compare the counterfactual on the score scale instead, which this module still "
                "computes, or record the per-rollout value-function baseline so the advantage can "
                "be reconstructed."
            ),
            statistics={"family": spec.family, "group_centred": False, "k": int(arr.size)},
        )
    if spec.std_normalised and spec.std_epsilon is None:
        return Refusal(
            instrument=_REPLAY,
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"the recorded estimator{subject} divides by the group standard deviation and the "
                f"record does not carry std_epsilon. It sits in the denominator of every advantage "
                f"in the group, and 1e-8 and 1e-4 are not the same number."
            ),
            remedy=(
                "Record EstimatorSpec.std_epsilon from the trainer's configuration and re-run this "
                "counterfactual. The framework defaults are in the trainer's own config, not in "
                "this library."
            ),
            statistics={"family": spec.family, "std_normalised": True, "std_epsilon": None},
        )
    if spec.std_normalised and spec.std_ddof is None:
        return Refusal(
            instrument=_REPLAY,
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"the recorded estimator{subject} divides by the group standard deviation and the "
                f"record does not say which one. The population form and Bessel's correction differ "
                f"by sqrt(K/(K-1)), which is 15.5% at K=4 and 41.4% at K=2, against a replay "
                f"tolerance of 1e-4, so the choice decides whether a replay agrees at all."
            ),
            remedy=(
                "Record EstimatorSpec.std_ddof from the trainer: 1 for TRL, whose nanstd applies "
                "Bessel's correction, and 1 for veRL, whose compute_grpo_outcome_advantage calls "
                "torch.std at its correction=1 default. Then re-run this counterfactual."
            ),
            statistics={"family": spec.family, "std_normalised": True, "std_ddof": None},
        )

    if not live.any():
        warnings.warn(
            f"every score in this group{subject} is an abstention, so the group has no advantages. "
            f"It contributes nothing to the update and it is not a group of zeros.",
            AllAbstainedWarning,
            stacklevel=2,
        )
        return tuple(float("nan") for _ in arr)

    present = arr[live]
    mean = float(present.mean())
    # `ddof` is whatever the record says the trainer used; the refusal above is what guarantees it
    # says something. A group of one has no spread under either convention, and `ddof=1` would
    # divide by zero there, so it degrades to the population form and the degenerate check below is
    # what catches the group.
    ddof = int(spec.std_ddof or 0) if present.size > 1 else 0
    std = float(present.std(ddof=ddof))
    eps = float(spec.std_epsilon or 0.0)
    degenerate = _is_degenerate(arr.tolist(), spec)

    if degenerate and spec.degenerate_policy == "unknown":
        return Refusal(
            instrument=_REPLAY,
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"this group{subject} is degenerate (std {std:.6g} against epsilon {eps:.6g}) and "
                f"the record does not say what the trainer does with a degenerate group. Skipping "
                f"it, zeroing it and keeping it give an empty group, a group of zeros, and a group "
                f"of advantages bounded only by epsilon, which are three different answers."
            ),
            remedy=(
                "Record EstimatorSpec.degenerate_policy as 'skip', 'zero' or 'keep' from the "
                "trainer's own handling and re-run. veRL and TRL differ here, so it cannot be "
                "inferred from the family name."
            ),
            statistics={
                "std": std,
                "std_epsilon": eps,
                "k": int(arr.size),
                "n_present": int(live.sum()),
            },
        )

    centred = arr - mean
    if spec.family.lower() in ("rloo", "loo"):
        k = int(live.sum())
        if k < 2:
            centred = np.full_like(centred, math.nan)
        else:
            # r_i - mean_{j != i}(r_j) = k/(k-1) * (r_i - mean). A positive multiple, so it moves no
            # sign and no ratio against the group's own spread; it is applied anyway because the
            # number reported is an advantage and this is what the trainer computed.
            centred = centred * (k / (k - 1))

    if spec.std_normalised:
        out = centred / (std + eps)
    else:
        out = centred

    if degenerate:
        if spec.degenerate_policy == "zero":
            out = np.zeros_like(out)
        elif spec.degenerate_policy == "skip":
            out = np.full_like(out, math.nan)

    # `clip_low` and `clip_high` are the policy-ratio clip and they are deliberately not applied
    # here. The group-relative advantage is `(r_i - mean_j r_j) / (std_j(r_j) + eps)` with no clip
    # term, and ratio clipping is a separate mechanism belonging to the loss: it truncates the
    # update, not the advantage. The TRL tap writes TRL's `epsilon` and `epsilon_high` into these
    # fields (`tap/adapters/trl.py:1066-1067`), which is correct, because that is what they are.
    #
    # Applying them as bounds on the advantage was wrong in a way that was invisible at the default:
    # with `epsilon = 0.2` and no `epsilon_high`, `clip_low == clip_high == 0.2`, so `maximum` then
    # `minimum` pins every live advantage to exactly 0.2. On the 200-step reference record that is
    # **400 of 400 groups collapsing to a single constant**, against recorded advantages spanning
    # -1.13 to +1.30, and `counterfactual` differences two constant vectors and reports that nothing
    # moved. The estimator module documented all of this correctly and shipped `check_replay` to
    # detect it, but nothing obliged a caller to run the detector before reading the number, so the
    # detector shipped and the wrong number stayed.
    #
    # An abstention leaves with no advantage whatever happened in between.
    out = np.where(live, out, math.nan)
    return tuple(float(v) for v in out)


# ---------------------------------------------------------------------------
# The counterfactual
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupScores:
    """One group's recorded trees, their contexts, and the estimator that made them advantages."""

    trees: tuple["ScoreTree | None", ...]
    contexts: tuple[ScoreContext, ...]
    estimator: EstimatorSpec
    id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "trees", tuple(self.trees))
        object.__setattr__(self, "contexts", tuple(self.contexts))
        if len(self.trees) != len(self.contexts):
            raise ValueError(
                f"group {self.id!r}: {len(self.trees)} trees against {len(self.contexts)} contexts. "
                f"A context per rollout, in the same order."
            )


def group_scores(
    group: Group,
    *,
    accuracy: float | None = None,
    accuracy_feature: str = "correct",
    predicates: Mapping[str, Mapping[str, bool]] | None = None,
) -> GroupScores:
    """Adapt a recorded `Group` into the shape the counterfactual reads.

    ``accuracy`` is the group's own accuracy, which `GroupGate` needs and `GroupStats` does not
    carry. When it is not supplied it is computed as the mean of ``accuracy_feature`` over the
    trajectories that record it, and left None when none do. It is not inferred from the scores:
    a composed score is not a correctness label.

    ``predicates`` maps trajectory id to the recorded outcome of any opaque predicate.
    """
    trajs = group.trajectories
    if accuracy is None:
        vals = [
            float(t.features[accuracy_feature])  # type: ignore[index]
            for t in trajs
            if accuracy_feature in t.features
        ]
        accuracy = float(np.mean(vals)) if vals else None
    gctx = GroupContext(accuracy=accuracy, k=len(trajs), id=str(group.id))
    contexts = tuple(
        ScoreContext(
            features={str(k): float(v) for k, v in t.features.items()},
            predicates=dict((predicates or {}).get(str(t.id), {})),
            group=gctx,
        )
        for t in trajs
    )
    return GroupScores(
        trees=tuple(_tree_of(t) for t in trajs),
        contexts=contexts,
        estimator=group.estimator,
        id=str(group.id),
    )


def _tree_of(traj: Trajectory) -> "ScoreTree | None":
    scores = traj.scores
    if scores is None:
        return None
    if isinstance(scores, (Leaf, *_CONTAINERS)):
        return scores
    raise TypeError(
        f"trajectory {traj.id} carries scores of type {type(scores).__name__}, which is not a "
        f"ScoreTree. A bare float or a dict of weights cannot answer a counterfactual: convert it "
        f"to a tree first, and if the per-leaf scores were never recorded, they are gone."
    )


@register_payload
@dataclass(frozen=True)
class CounterfactualResult:
    """What removing a node did to this run's advantages.

    Both arms are computed here rather than taken from the record, so the difference is attributable
    to the node and not to a disagreement between this module and the trainer. ``reproduction_error``
    is the check on that: the largest absolute gap between the baseline advantages computed here and
    the advantages the pipeline recorded, over the rollouts that carry one. A large value means the
    recorded `EstimatorSpec` does not describe what the trainer did, and every number below inherits
    that.

    ``n_comparable`` is the denominator for both headline fractions: a rollout that abstains in
    either arm has no advantage in that arm and cannot be said to have moved or reversed. A sign
    reversal needs both signs to be nonzero and opposite, so an advantage that was exactly zero and
    is now negative is a move rather than a reversal.
    """

    disabled: tuple[str, ...]
    n: int
    n_groups: int
    n_trees_with_node: int
    scores_before: tuple[float, ...]
    scores_after: tuple[float, ...]
    advantages_before: tuple[float, ...]
    advantages_after: tuple[float, ...]
    n_comparable: int
    n_sign_reversed: int
    n_moved: int
    sd_reference: float
    n_abstained_before: int
    n_abstained_after: int
    degenerate_groups_before: int
    degenerate_groups_after: int
    reproduction_error: float | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # The kernel's codec decodes every sequence as a list, so a stored result would otherwise
        # come back unequal to the one that was written.
        for f in ("disabled", "notes"):
            object.__setattr__(self, f, tuple(getattr(self, f)))
        for f in ("scores_before", "scores_after", "advantages_before", "advantages_after"):
            object.__setattr__(self, f, tuple(float(v) for v in getattr(self, f)))

    @property
    def fraction_moved(self) -> float:
        return self.n_moved / self.n_comparable if self.n_comparable else 0.0

    @property
    def fraction_reversed(self) -> float:
        return self.n_sign_reversed / self.n_comparable if self.n_comparable else 0.0

    def says(self) -> str:
        """B3's sentence, with this run's numbers in it."""
        what = ", ".join(self.disabled)
        return (
            f"removing {what} changes {self.fraction_moved:.0%} of advantages by more than one "
            f"standard deviation, and reverses the sign for {self.n_sign_reversed} of {self.n} "
            f"rollouts."
        )

    def render(self) -> str:
        lines = [self.says()]
        lines.append(
            f"    {self.n_comparable} of {self.n} rollouts have an advantage in both arms; "
            f"one standard deviation is {self.sd_reference:.4g} on the baseline arm."
        )
        if self.n_abstained_after != self.n_abstained_before:
            lines.append(
                f"    abstentions move from {self.n_abstained_before} to {self.n_abstained_after}."
            )
        if self.degenerate_groups_after != self.degenerate_groups_before:
            lines.append(
                f"    degenerate groups move from {self.degenerate_groups_before} to "
                f"{self.degenerate_groups_after} of {self.n_groups}."
            )
        if self.reproduction_error is not None:
            lines.append(
                f"    baseline reproduces the recorded advantages to {self.reproduction_error:.3g}."
            )
        lines.extend(f"    note: {n}" for n in self.notes)
        return "\n".join(lines)


_CF = "record.scores.counterfactual"


def counterfactual(
    groups: Sequence[GroupScores],
    disabled: Iterable[str],
    *,
    sd_reference: float | None = None,
    recorded_advantages: Sequence[Sequence[float | None]] | None = None,
) -> CounterfactualResult | Refusal:
    """Re-evaluate the recorded trees with nodes removed, and run the estimator over both arms.

    Rung 0 of B3's ladder: free, no grader calls, no GPU, on the leaves already on disk. The result
    is in advantage space rather than score space because that is where the question lives. A shift
    that survives group centring changed the update; one that does not, did not.

    ``recorded_advantages`` is optional and is the closure check: pass the advantages the pipeline
    wrote and the result carries the largest gap between them and the baseline arm computed here.
    """
    names = tuple(sorted(set(disabled)))
    if not names:
        raise ValueError(
            "a counterfactual with nothing disabled is the baseline. Name at least one node."
        )
    if not groups:
        raise ValueError("no groups to compare.")

    n_missing = sum(1 for g in groups for t in g.trees if t is None)
    n_total = sum(len(g.trees) for g in groups)
    if n_missing:
        return Refusal(
            instrument=_CF,
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"{n_missing} of {n_total} trajectories carry no score tree, so the per-leaf "
                f"composition was not recorded for them. Every advantage in their group depends on "
                f"the group mean and standard deviation, so one missing rollout corrupts all K."
            ),
            remedy=(
                "Restrict the comparison to groups whose trajectories all carry a ScoreTree, or "
                "instrument the grader with reward_lens.tap so the per-leaf scores are recorded, "
                "and re-run. Composition that was never recorded cannot be recovered from a total."
            ),
            statistics={"n_missing": n_missing, "n_total": n_total},
        )

    present: set[str] = set()
    for g in groups:
        for tree in g.trees:
            if tree is not None:
                present.update(node_names(tree))
    unknown = sorted(set(names) - present)
    if unknown:
        raise UnknownNode(
            f"cannot disable {unknown}: no tree in these {len(groups)} groups contains such a node. "
            f"Reporting no change would be a false negative."
        )

    scores_before: list[float] = []
    scores_after: list[float] = []
    adv_before: list[float] = []
    adv_after: list[float] = []
    n_with_node = 0
    degen_before = degen_after = 0
    notes: list[str] = []

    wanted = frozenset(names)
    for gi, g in enumerate(groups):
        before: list[float] = []
        after: list[float] = []
        for tree, ctx in zip(g.trees, g.contexts):
            if tree is None:  # unreachable: the missing-tree refusal above already returned
                continue
            here = wanted & frozenset(node_names(tree))
            n_with_node += 1 if here else 0
            before.append(evaluate(tree, ctx))
            after.append(evaluate(tree, ctx, here))
        a_before = replay_advantages(before, g.estimator, where=f"group {g.id or gi} (baseline)")
        if isinstance(a_before, Refusal):
            return a_before
        a_after = replay_advantages(
            after, g.estimator, where=f"group {g.id or gi} (counterfactual)"
        )
        if isinstance(a_after, Refusal):
            return a_after
        degen_before += int(_is_degenerate(before, g.estimator))
        degen_after += int(_is_degenerate(after, g.estimator))
        scores_before.extend(before)
        scores_after.extend(after)
        adv_before.extend(a_before)
        adv_after.extend(a_after)

    ab = np.asarray(adv_before, dtype=float)
    aa = np.asarray(adv_after, dtype=float)
    comparable = np.isfinite(ab) & np.isfinite(aa)
    n_comparable = int(comparable.sum())

    if sd_reference is None:
        sd = float(np.std(ab[np.isfinite(ab)])) if np.isfinite(ab).any() else 0.0
    else:
        sd = float(sd_reference)
    if sd <= 0.0:
        notes.append(
            "every baseline advantage is identical, so one standard deviation is zero and "
            "'moved by more than one standard deviation' counts any change at all."
        )

    delta = np.zeros_like(ab)
    delta[comparable] = aa[comparable] - ab[comparable]
    n_moved = int((np.abs(delta[comparable]) > sd).sum())
    sign_before = np.sign(ab[comparable])
    sign_after = np.sign(aa[comparable])
    n_reversed = int(((sign_before * sign_after) < 0).sum())

    if degen_after > degen_before:
        policies = sorted({g.estimator.degenerate_policy for g in groups})
        notes.append(
            f"{degen_after - degen_before} group(s) become degenerate once the node is removed, so "
            f"their advantages are decided by the estimator's degenerate policy "
            f"({', '.join(policies)}) rather than by the scores."
        )

    repro: float | None = None
    if recorded_advantages is not None:
        flat = [v for row in recorded_advantages for v in row]
        if len(flat) != len(adv_before):
            raise ValueError(
                f"recorded_advantages has {len(flat)} entries against {len(adv_before)} rollouts."
            )
        gaps = [
            abs(float(r) - b)
            for r, b in zip(flat, adv_before)
            if r is not None and math.isfinite(float(r)) and math.isfinite(b)
        ]
        repro = max(gaps) if gaps else None
        if repro is not None and repro > 1e-6:
            notes.append(
                f"the baseline arm computed here differs from the recorded advantages by up to "
                f"{repro:.3g}, so the recorded EstimatorSpec does not fully describe what the "
                f"trainer did and every number here inherits that gap."
            )

    return CounterfactualResult(
        disabled=names,
        n=len(adv_before),
        n_groups=len(groups),
        n_trees_with_node=n_with_node,
        scores_before=tuple(scores_before),
        scores_after=tuple(scores_after),
        advantages_before=tuple(adv_before),
        advantages_after=tuple(adv_after),
        n_comparable=n_comparable,
        n_sign_reversed=n_reversed,
        n_moved=n_moved,
        sd_reference=sd,
        n_abstained_before=int(sum(1 for s in scores_before if not math.isfinite(s))),
        n_abstained_after=int(sum(1 for s in scores_after if not math.isfinite(s))),
        degenerate_groups_before=degen_before,
        degenerate_groups_after=degen_after,
        reproduction_error=repro,
        notes=tuple(notes),
    )


def _is_degenerate(scores: Sequence[float], spec: EstimatorSpec) -> bool:
    """`std` at or below the estimator's own epsilon, on the scores that are not abstentions.

    The boundary is the estimator's epsilon rather than a constant of this module's, for the reason
    `GroupStats.from_scores` gives: it is the only way the flag means what the trainer meant. An
    estimator that records no epsilon gets zero, which makes exact ties degenerate and nothing else.
    """
    live = [s for s in scores if math.isfinite(s)]
    if not live:
        return True
    std = float(np.std(np.asarray(live, dtype=float)))
    return std <= float(spec.std_epsilon or 0.0)


__all__ = [
    "AbstentionCensus",
    "AllAbstainedWarning",
    "CounterfactualResult",
    "DeadZone",
    "DuplicateNodeName",
    "GraderCallRef",
    "GroupContext",
    "GroupGate",
    "GroupScores",
    "Leaf",
    "Lexicographic",
    "Mask",
    "MissingFeature",
    "MissingGroupStatistic",
    "Override",
    "Piecewise",
    "PredicateRef",
    "Product",
    "ScoreContext",
    "ScoreTree",
    "UnknownNode",
    "WeightedSum",
    "ablation_deltas",
    "census",
    "counterfactual",
    "deadzone",
    "evaluate",
    "find",
    "group_scores",
    "is_additive",
    "leaves",
    "node_names",
    "overrides",
    "replay_advantages",
    "walk",
]

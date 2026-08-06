"""The gate, the running variable it is a threshold on, and how to get both off a record.

Everything in series I needs the same two objects: a per-rollout number that a rule was applied to,
and the rule. This module builds them from a `Run` and nothing else, so no instrument in the
package has to know how a record is laid out.

**Two provenances, and they are never mixed.** A `Gate` carries `installed`. False means the run
really had this gate: an `Override` or a `Piecewise` node is on the recorded composition tree and
the policy was optimised against it. True means the gate was installed counterfactually by
`install_gate`, on a record whose leaves and features are real but whose rule the policy never saw.
The second is a way of validating an estimator against a cutoff whose location you know. It is not
evidence about anybody's run, and every reading that carries an installed gate says so in its own
`render`.

**The decode length is separate from the training budget and is reported separately.** A sampler
that stops at `max_completion_length` censors the running variable from above whatever the reward
rule does, and a region drawn by the sampler is not a region drawn by the gate. This is not a
subtlety: a generation-length cap has already turned a reported +15.3 point gain into -8.0 points
in this exact area. That figure is somebody else's published measurement, quoted to say why this
type exists; nothing here reproduces it. `DecodeLength` exists so a reading cannot be produced
without the cap beside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator, Literal, Sequence

import numpy as np

from reward_lens.core.evidence import register_payload
from reward_lens.record.schema import Run, Step, Trajectory
from reward_lens.record.scores import (
    GroupContext,
    Override,
    PredicateRef,
    ScoreContext,
    ScoreTree,
    overrides,
    walk,
)

GateSide = Literal["above", "below"]
GateKind = Literal["notch", "kink"]

#: The keys a framework writes the sampler's completion cap under. Read in this order; the first
#: one present wins and the key it came from is named on the reading.
DECODE_CAP_KEYS: tuple[str, ...] = (
    "max_completion_length",
    "max_new_tokens",
    "max_response_length",
    "max_gen_len",
    "response_length",
)

#: The per-step telemetry key carrying the fraction of completions that hit the cap.
CLIPPED_RATIO_KEYS: tuple[str, ...] = (
    "completions/clipped_ratio",
    "completions/truncated_ratio",
    "response_length/clip_ratio",
)


# ---------------------------------------------------------------------------
# The running variable
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class RunningVariable:
    """The per-rollout number a threshold rule is applied to.

    ``unit`` is carried rather than assumed because the two obvious choices for a length gate,
    tokens and characters, are different variables with different densities and a gate stated in
    one cannot be tested in the other. Nothing in this package converts between them.

    ``censored_at`` and ``censored_fraction`` are the sampler's cap and how much of the sample sits
    on it. A running variable that is a point mass at its own cap has no density to test, and
    saying so is the difference between a refusal and a confident wrong number.
    """

    name: str
    values: np.ndarray
    unit: str
    source: str
    steps: np.ndarray | None = None
    censored_at: float | None = None
    censored_fraction: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", np.asarray(self.values, dtype=np.float64).ravel())
        if self.steps is not None:
            object.__setattr__(self, "steps", np.asarray(self.steps, dtype=np.int64).ravel())

    @property
    def n(self) -> int:
        return int(np.isfinite(self.values).sum())

    @property
    def n_distinct(self) -> int:
        return int(np.unique(self.values[np.isfinite(self.values)]).size)

    def render(self) -> str:
        v = self.values[np.isfinite(self.values)]
        if v.size == 0:
            return f"{self.name}: no finite values"
        cap = ""
        if self.censored_at is not None:
            cap = f", censored at {self.censored_at:g} on {self.censored_fraction:.1%} of rollouts"
        return (
            f"{self.name} ({self.unit}): n={v.size}, {self.n_distinct} distinct, "
            f"[{v.min():g}, {v.max():g}], median {np.median(v):g}{cap}"
        )


# ---------------------------------------------------------------------------
# The decode length, which is not the training budget
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class DecodeLength:
    """The sampler's completion cap, read off the run, with how hard it binds.

    Reported on every I3 reading and available to I1 and I2, because a hard threshold in the
    running variable that came from the sampler is not evidence about the reward rule. Where the
    cap binds on every rollout the running variable is a point mass and there is no density left to
    test; where it binds on none of them the cap is present and inert; in between, the upper tail
    of the running variable is censored and every statistic that reads the tail is conditional on
    that.
    """

    cap: float | None
    cap_key: str | None
    clipped_fraction: float | None
    clipped_key: str | None
    unit: str = "tokens"
    per_step_clipped: tuple[float, ...] = ()

    @property
    def binds_everywhere(self) -> bool:
        return self.clipped_fraction is not None and self.clipped_fraction >= 1.0

    @property
    def is_known(self) -> bool:
        return self.cap is not None or self.clipped_fraction is not None

    def render(self) -> str:
        if not self.is_known:
            return (
                "decode length: not recorded. The sampler's completion cap is not on this record, "
                "so nothing here can separate a region drawn by the reward rule from a region "
                "drawn by the sampler."
            )
        cap = "not recorded" if self.cap is None else f"{self.cap:g} {self.unit}"
        frac = (
            "not recorded"
            if self.clipped_fraction is None
            else f"{self.clipped_fraction:.1%} of rollouts"
        )
        return f"decode length cap {cap} (key {self.cap_key}), binding on {frac}"


def decode_length(run: Run, *, steps: Sequence[Step] | None = None) -> DecodeLength:
    """The sampler's cap and its binding fraction, from the run's config and its telemetry.

    The cap comes from the recorded framework config and the binding fraction from the per-step
    telemetry, because the two can disagree and the disagreement is informative: a cap of 4096 that
    binds on 40 percent of rollouts is a different run from a cap of 4096 that binds on none.
    """
    config = run.lineage.extra.get("config") or {}
    cap: float | None = None
    cap_key: str | None = None
    for key in DECODE_CAP_KEYS:
        value = config.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            cap, cap_key = float(value), key
            break

    per_step: list[float] = []
    clipped_key: str | None = None
    for step in steps if steps is not None else run.steps:
        for key in CLIPPED_RATIO_KEYS:
            value = step.optimizer.extra.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                per_step.append(float(value))
                clipped_key = key
                break
    clipped = float(np.mean(per_step)) if per_step else None
    return DecodeLength(
        cap=cap,
        cap_key=cap_key,
        clipped_fraction=clipped,
        clipped_key=clipped_key,
        per_step_clipped=tuple(per_step),
    )


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class Gate:
    """A hard threshold on a running variable, and what crossing it costs.

    ``penalised_side`` says which side of the cutoff the rule punishes, which decides which side
    the bunching is expected on. ``kind`` separates the two shapes the bunching literature treats
    differently: a **notch** is a discontinuous drop in the score at the cutoff, which is what an
    `Override` is, and a **kink** is a continuous score with a discontinuous slope, which is what a
    `Piecewise` node with a slope change is. They imply different elasticity formulas and mixing
    them up is a factor of `Delta z / z` in the answer.

    ``penalty_fraction`` is the fractional score drop on crossing, the reward analogue of the
    change in the net-of-tax rate that a public-finance bunching estimate divides by. For an
    override from a score of `s` to a constant `k` it is `(s - k) / |s|` evaluated at the cutoff.
    Without it there is a rung-0 excess mass and no rung-1 elasticity, and the instrument says so
    rather than inventing one.

    ``installed`` is the honesty field. See the module docstring.
    """

    name: str
    cutoff: float
    unit: str
    penalised_side: GateSide = "above"
    kind: GateKind = "notch"
    penalty_fraction: float | None = None
    constant: float | None = None
    installed: bool = False
    provenance: str = ""
    predicate: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.cutoff)):
            raise ValueError(f"gate {self.name!r} has a non-finite cutoff {self.cutoff!r}")
        if self.penalty_fraction is not None and self.penalty_fraction <= 0:
            raise ValueError(
                f"gate {self.name!r} declares a penalty fraction of {self.penalty_fraction}, which "
                f"is not a penalty. A gate that costs nothing to cross generates no bunching, and "
                f"dividing by it produces an infinite elasticity rather than an error."
            )

    @property
    def bunching_side(self) -> GateSide:
        """Where mass piles up: the unpenalised side."""
        return "below" if self.penalised_side == "above" else "above"

    def moved_to(self, cutoff: float) -> "Gate":
        """The same gate at a different cutoff. What `gate_response` sweeps."""
        return replace(self, cutoff=float(cutoff))

    def render(self) -> str:
        origin = (
            "INSTALLED counterfactually; the policy was never optimised against it"
            if self.installed
            else "recorded on the run"
        )
        pen = (
            "penalty fraction not stated"
            if self.penalty_fraction is None
            else f"crossing costs {self.penalty_fraction:.1%} of the score"
        )
        return (
            f"{self.name}: {self.kind} at {self.cutoff:g} {self.unit}, penalising "
            f"{self.penalised_side}, {pen} [{origin}]"
        )


# ---------------------------------------------------------------------------
# Reading a record
# ---------------------------------------------------------------------------


def trajectories(
    run: Run, *, span: tuple[int, int] | None = None
) -> Iterator[tuple[int, Trajectory]]:
    """Every rollout in the run, paired with its step index."""
    stream = run.steps.slice(*span) if span is not None else iter(run.steps)
    for step in stream:
        for group in step.groups:
            for traj in group.trajectories:
                yield step.index, traj


def completion_length(traj: Trajectory, *, unit: str = "characters") -> float:
    """One rollout's completion length, in characters or in tokens.

    Characters are read off the recorded assistant turns and tokens off `Turn.token_ids`. Tokens
    return NaN where the ids were not recorded rather than falling back to characters: the two are
    different variables and a silent substitution is a unit error the record exists to prevent.
    """
    turns = [t for t in traj.turns if t.role == "assistant"]
    if unit == "characters":
        return float(sum(len(t.text or "") for t in turns))
    if unit == "tokens":
        counts = [t.n_tokens for t in turns]
        if not counts or any(c is None for c in counts):
            return float("nan")
        return float(sum(int(c) for c in counts if c is not None))
    raise ValueError(f"unknown length unit {unit!r}; use 'characters' or 'tokens'")


def completion_lengths(
    run: Run, *, unit: str = "characters", span: tuple[int, int] | None = None
) -> RunningVariable:
    """The per-rollout completion length across the run, with the sampler's cap attached.

    This is the running variable a length gate is a threshold on, and it is the one every published
    token-budget rule is stated in. Where the record carries no token ids the token form is all
    NaN, which `RunningVariable.n` reports as zero rather than as a small sample.
    """
    values: list[float] = []
    steps: list[int] = []
    for index, traj in trajectories(run, span=span):
        values.append(completion_length(traj, unit=unit))
        steps.append(index)
    cap = decode_length(run)
    arr = np.asarray(values, dtype=np.float64)
    censored_at: float | None = None
    censored_fraction = 0.0
    if unit == "tokens" and cap.cap is not None:
        censored_at = cap.cap
        finite = arr[np.isfinite(arr)]
        if finite.size:
            censored_fraction = float(np.mean(finite >= cap.cap))
    elif cap.clipped_fraction is not None:
        censored_fraction = float(cap.clipped_fraction)
    return RunningVariable(
        name=f"completion length ({unit})",
        values=arr,
        unit=unit,
        source=f"record: assistant turns of {len(values)} rollouts",
        steps=np.asarray(steps, dtype=np.int64),
        censored_at=censored_at,
        censored_fraction=censored_fraction,
    )


def score_contexts(
    run: Run,
    *,
    running: RunningVariable | None = None,
    feature: str = "completion_length",
    span: tuple[int, int] | None = None,
) -> tuple[tuple[ScoreTree | None, ...], tuple[ScoreContext, ...], np.ndarray]:
    """The recorded trees and their evaluation contexts, with the running variable as a feature.

    `PredicateRef` reads its left-hand side out of `ScoreContext.features`, so a gate stated in
    completion length can only be evaluated if the completion length is a feature. Frameworks
    record the features they happened to record, and TRL records none of them, so this derives the
    length from the recorded turns and adds it under ``feature`` alongside whatever the record
    already carried. Nothing else about the context is invented: the recorded features win on a
    name collision, and the group statistics come off `GroupStats`.
    """
    trees: list[ScoreTree | None] = []
    contexts: list[ScoreContext] = []
    values: list[float] = []
    supplied = None if running is None else np.asarray(running.values, dtype=np.float64)
    i = 0
    stream = run.steps.slice(*span) if span is not None else iter(run.steps)
    for step in stream:
        for group in step.groups:
            gctx = GroupContext(
                accuracy=None,
                k=group.k,
                id=str(group.id),
                extra={"std": float(group.group_stats.std or 0.0)},
            )
            for traj in group.trajectories:
                length = (
                    float(supplied[i])
                    if supplied is not None and i < supplied.size
                    else completion_length(traj)
                )
                features: dict[str, float] = {feature: length}
                features.update({str(k): float(v) for k, v in traj.features.items()})
                trees.append(traj.scores)
                contexts.append(ScoreContext(features=features, group=gctx))
                values.append(length)
                i += 1
    return tuple(trees), tuple(contexts), np.asarray(values, dtype=np.float64)


def recorded_gates(trees: Iterable[ScoreTree | None], *, unit: str = "") -> tuple[Gate, ...]:
    """Every hard gate the run really had, read off its recorded composition trees.

    An `Override` whose condition is a comparison over a recorded feature is a gate whose cutoff is
    readable. An override with an opaque condition is a gate whose cutoff is not a number on the
    record, so it is skipped rather than guessed at, and the caller sees a shorter list than the
    tree has override nodes.
    """
    seen: dict[str, Gate] = {}
    for tree in trees:
        if tree is None:
            continue
        for node in overrides(tree):
            pred = node.condition
            if pred.is_opaque or pred.threshold is None or node.name in seen:
                continue
            side: GateSide = "above" if pred.op in (">", ">=") else "below"
            seen[node.name] = Gate(
                name=node.name,
                cutoff=float(pred.scale * pred.threshold),
                unit=unit or (pred.feature or ""),
                penalised_side=side,
                kind="notch",
                constant=float(node.constant),
                installed=False,
                provenance="recorded Override on the run's own composition tree",
                predicate=pred.expr,
            )
    return tuple(seen.values())


# ---------------------------------------------------------------------------
# Installing a gate a run did not have
# ---------------------------------------------------------------------------


def install_gate(
    tree: ScoreTree,
    *,
    cutoff: float,
    feature: str = "completion_length",
    constant: float = -1.0,
    name: str = "installed_budget_gate",
    op: str = ">",
    unit: str = "characters",
) -> tuple[ScoreTree, Gate]:
    """Wrap a recorded tree in an `Override` firing above ``cutoff``, and describe the gate.

    This is the counterfactual mechanism B3 uses, pointed at a rule the run did not have: the
    leaves are the ones the graders really returned, the features are the ones really recorded, and
    only the composition changes. It costs one pass over the record and calls nothing.

    **The gate it returns is marked installed and it must stay marked.** A record re-scored under a
    rule the policy never saw tells you what that rule would have done to these rollouts. It tells
    you nothing about how a policy would have behaved under it, because the rollouts would have
    been different ones. Every bunching statistic is about the second thing, which is why route 2
    validates an estimator and does not measure a run.
    """
    if name in set(_names(tree)):
        raise ValueError(
            f"the tree already has a node called {name!r}. Installing a second one would make "
            f"`find` ambiguous; pass a different `name`."
        )
    predicate = PredicateRef(
        name=f"{name}_condition",
        feature=feature,
        op=op,  # type: ignore[arg-type]
        threshold=float(cutoff),
    )
    gated = Override(name=name, condition=predicate, constant=float(constant), otherwise=tree)
    side: GateSide = "above" if op in (">", ">=") else "below"
    gate = Gate(
        name=name,
        cutoff=float(cutoff),
        unit=unit,
        penalised_side=side,
        kind="notch",
        constant=float(constant),
        installed=True,
        provenance=(
            "installed by measure.threshold.gates.install_gate on a recorded tree; the policy was "
            "never optimised against it"
        ),
        predicate=predicate.expr,
    )
    return gated, gate


def _names(tree: ScoreTree) -> Iterator[str]:
    for node in walk(tree):
        yield node.name


def penalty_fraction(
    trees: Sequence[ScoreTree | None],
    contexts: Sequence[ScoreContext],
    running: Sequence[float] | np.ndarray,
    *,
    cutoff: float,
    constant: float,
    band: float | None = None,
) -> float:
    """What crossing the gate costs, as a fraction of the score just below it.

    Rung 1 of I2 divides by this, and it is the one input that cannot be read off the gate alone: a
    penalty of one reward point is a rout on a run scoring 0.1 and a rounding error on one scoring
    100. So it is measured from the record, as the mean recorded score in a band just below the
    cutoff against the constant the override writes.

    ``band`` defaults to a tenth of the interquartile range of the running variable, which is wide
    enough to average over sampling noise and narrow enough that the score has not drifted. Returns
    NaN where no rollout falls in the band, which the caller must check rather than divide by.
    """
    from reward_lens.record.scores import evaluate

    x = np.asarray(running, dtype=np.float64).ravel()
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan")
    if band is None:
        q75, q25 = np.percentile(finite, [75, 25])
        band = max(float(q75 - q25) / 10.0, 1e-9)
    scores: list[float] = []
    for tree, ctx, value in zip(trees, contexts, x):
        if tree is None or not math.isfinite(value):
            continue
        if not (cutoff - band <= value < cutoff):
            continue
        s = evaluate(tree, ctx)
        if math.isfinite(s):
            scores.append(float(s))
    if not scores:
        return float("nan")
    mean = float(np.mean(scores))
    if mean == 0.0:
        return float("nan")
    return float((mean - constant) / abs(mean))


@register_payload
@dataclass(frozen=True)
class InstalledGateSet:
    """A set of gates installed on one record, kept together so a reading cannot lose the label."""

    gates: tuple[Gate, ...] = ()
    record: str = ""
    note: str = field(
        default=(
            "installed counterfactually on a recorded run. These gates validate the estimators "
            "against a cutoff whose location is known; they are not evidence about this run's "
            "policy, which never saw them."
        )
    )

    def render(self) -> str:
        return "\n".join([self.note, *(f"  {g.render()}" for g in self.gates)])


__all__ = [
    "CLIPPED_RATIO_KEYS",
    "DECODE_CAP_KEYS",
    "DecodeLength",
    "Gate",
    "GateKind",
    "GateSide",
    "InstalledGateSet",
    "RunningVariable",
    "completion_length",
    "completion_lengths",
    "decode_length",
    "install_gate",
    "penalty_fraction",
    "recorded_gates",
    "score_contexts",
    "trajectories",
]

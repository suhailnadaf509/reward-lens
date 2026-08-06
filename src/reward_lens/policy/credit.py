"""Credit as a conserved measure, not a per-token score (series G).

The failed `VERIF-PRM` card asked a reward model to *estimate* where credit lay and scored AUC
0.2821. Wrong question. In a live run credit assignment is not inferred, it is executed: the
optimizer computes it and throws it away. Define a signed measure on the token lattice by

    mu(k, t) = A_k * grad_theta log pi(y_{k,t} | .),    sum_{k,t} mu(k, t) = grad_theta J

Because the total mass equals the update, **a credit report either accounts for 100% of the step or
it has a bug.** That is what makes this an audit rather than an attribution heuristic, and
`conservation_error` is a reported quantity rather than a test that runs once in CI.

**Nothing here ever materialises a per-token per-parameter gradient.** ``|theta| x T`` is
astronomical and computing it is not the way in. What this module computes are *disintegrations*:
given a partition of the trained positions into ``m`` parts, mask the objective to each part and
take one backward per part with ``retain_graph=True``. That gives ``m`` parameter-space vectors
whose sum is the full gradient, at ``m`` backward passes and three parameter-sized buffers of peak
memory. A 40-turn episode is 40 backwards, which is feasible; the per-token form on a
500,000-token episode is not, and that is precisely why the interesting disintegration is by turn.

**Norms do not add; the projection does.** ``|g_S|`` is not additive over parts, so a report built
out of norms cannot close. The functional that is exactly additive is the projection onto the full
gradient, ``<g_S, g_full> / |g_full|^2``, and those sum to 1 by linearity. This module reports the
projected share as the share, the norm ratio beside it, and their disagreement as `cancellation`,
which is a real quantity: 2.23 on step 0 of the reference record means 55% of the per-rollout
gradient mass cancels between rollouts before the optimizer sees any of it.

**What this cannot do, in three sentences.** The identity `sum mu = dtheta/eta` holds for plain
gradient descent and for nothing else: momentum, Adam's second moment, weight decay and gradient
clipping each break it, and `step_conservation` therefore checks the identity against an actual
`torch.optim.SGD` step and refuses to claim it for any other optimizer. The disintegration is a
statement about *this* gradient at *these* parameters, so it says where the step went and not what
any part of the trajectory contributed to the run. And every number here is a first-order quantity:
removing a turn does not remove its share, because removing it changes every downstream position.

Precision is not a detail on this measurement. `trl-internal-testing/tiny-Qwen3ForCausalLM` is
bfloat16, and on step 0 of the reference record the conservation error is 1.56e-03 in its native
dtype against 1.13e-07 in float32, a factor of fourteen thousand. So it casts to float32 by default,
restores the original dtype afterwards, and reports which dtype produced the number.
"""

from __future__ import annotations

import hashlib
import math
import pickle
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.meta.incremental import Detector, IncrementalValidityReading
from reward_lens.measure.rate.regime import MEASURED_BY
from reward_lens.stats.baselines import ALL_SIX, DetectionTask, is_scored, run_bank

if TYPE_CHECKING:  # pragma: no cover - torch is an extra, imported inside the functions that use it
    import torch


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------


#: Roles whose tokens the loss is taken over, absent an explicit mask on the record. Every framework
#: in scope masks environment tokens, and loss masking is a transform that changes what "per token"
#: means, so this is a stated default rather than an inference: a caller who disagrees passes their
#: own, and the value used is written onto the report.
TRAINED_ROLES: tuple[str, ...] = ("assistant",)


@dataclass(frozen=True)
class Part:
    """One cell of a disintegration: a name, what kind of thing it is, and which positions it holds.

    ``mask`` is over *target* positions, ``(B, T-1)`` boolean, because the objective sums the
    log-probability of token ``t+1`` predicted from position ``t``. Keeping the parts in target
    coordinates rather than in id coordinates is the one place an off-by-one here would be silent:
    a part shifted by one token still sums to the right total and attributes the wrong span.

    ``kind`` is free text and the four this module produces are ``rollout``, ``turn``,
    ``tool_call`` and ``span``. It exists so a report can be grouped after the fact without
    parsing names.
    """

    name: str
    kind: str
    mask: np.ndarray

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError(
                f"part {self.name!r} has a {mask.ndim}-dimensional mask; a part is a set of "
                f"(rollout, target position) pairs and its mask is (B, T-1)."
            )
        object.__setattr__(self, "mask", mask)

    @property
    def n_positions(self) -> int:
        return int(self.mask.sum())


class PartitionError(ValueError):
    """The parts do not tile the trained positions, so no disintegration of them can conserve.

    Raised at construction rather than reported as a bad conservation number, because the two are
    different failures with different remedies. A gap or an overlap in the partition is a caller
    error whose fix is the partition; a conservation error on a partition that does tile is a
    numerical fact about the model and the dtype, and it is the thing this module measures. Letting
    the first show up as the second would put a user error in the units of a measurement.
    """


@dataclass(frozen=True)
class Partition:
    """A set of parts asserted to tile the trained positions exactly, with no gap and no overlap.

    The tiling check is the precondition for conservation and it is enforced here for the same
    reason `Trajectory.__post_init__` enforces the segment tiling: the quantity downstream is
    undefined rather than approximate when it fails. A partition that leaves a trained position
    uncovered produces a conservation error equal to the missing part's share, which reads exactly
    like a numerical problem and is not one.
    """

    parts: tuple[Part, ...]
    #: The trained positions the parts are asserted to cover, ``(B, T-1)`` boolean.
    trained: np.ndarray

    def __post_init__(self) -> None:
        trained = np.asarray(self.trained, dtype=bool)
        object.__setattr__(self, "trained", trained)
        if not self.parts:
            raise PartitionError(
                "a partition with no parts cannot tile anything. Build one with "
                "`by_rollout`, `by_turn` or `by_span`, or pass a single part covering every "
                "trained position if the whole batch is the cell of interest."
            )
        cover = np.zeros_like(trained, dtype=np.int32)
        for part in self.parts:
            if part.mask.shape != trained.shape:
                raise PartitionError(
                    f"part {part.name!r} has mask shape {part.mask.shape} against trained "
                    f"positions of shape {trained.shape}. Every part is a mask over the same "
                    f"lattice; build them from the same batch."
                )
            cover += part.mask.astype(np.int32)
        overlap = int(((cover > 1) & trained).sum())
        gap = int((trained & (cover == 0)).sum())
        stray = int((~trained & (cover > 0)).sum())
        if overlap or gap or stray:
            raise PartitionError(
                f"the {len(self.parts)} parts do not tile the {int(trained.sum()):,} trained "
                f"positions: {gap:,} uncovered, {overlap:,} covered twice, {stray:,} covering a "
                f"position the loss does not see. Conservation is an identity about a partition, "
                f"so fix the partition rather than widening the tolerance: an uncovered position "
                f"produces a conservation error equal to its own share and reads like a numerical "
                f"problem."
            )

    def __len__(self) -> int:
        return len(self.parts)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parts)

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(p.kind for p in self.parts)


def merge_empty(partition: Partition, *, into: str = "unattributed") -> Partition:
    """Fold every part holding no trained position into one named part.

    A turn with no trained tokens is a real and common thing: a user turn, a tool result, a turn
    whose whole span was masked. Its gradient is exactly zero and carrying forty of them makes a
    report unreadable. They are merged rather than dropped, because dropping them would leave a
    partition that no longer tiles and a reader who cannot tell an empty part from an absent one.
    """
    keep = [p for p in partition.parts if p.n_positions > 0]
    empty = [p for p in partition.parts if p.n_positions == 0]
    if not empty:
        return partition
    zeros = np.zeros_like(partition.trained, dtype=bool)
    merged = Part(
        name=f"{into} ({len(empty)} parts with no trained token)", kind="empty", mask=zeros
    )
    return Partition(parts=tuple(keep) + (merged,), trained=partition.trained)


# ---------------------------------------------------------------------------
# The batch: what a disintegration is taken over, in one framework-neutral shape
# ---------------------------------------------------------------------------


#: How the loss is normalised before the gradient is taken. Token-level versus sequence-level
#: aggregation is a transform that changes what a downstream number means, and it is
#: `EstimatorSpec.aggregation` on the record. It is read rather than assumed because it does not
#: cancel: `sequence` reweights rows by their own length and therefore moves the shares, not only
#: the total.
Aggregation = str


@dataclass(frozen=True)
class TrajectoryBatch:
    """Tokenised rollouts and the weights the estimator actually applied to them.

    This is the whole input surface of the numerical path, and it is deliberately not a record type
    and not a framework type: it is built from a `Trajectory` by `batch_from_trajectories`, from a
    SkyRL dump by `batch_from_skyrl`, and by hand in a test. Nothing below it imports `trl`,
    `skyrl` or `reward_lens.record`.

    ``advantages`` is ``(B,)`` or ``(B, T-1)``. Both shapes are real: every framework in scope
    computes a per-token advantage tensor because the loss needs one, and in the outcome-reward case
    every entry of a row is the same number. Keeping the per-token shape rather than collapsing it
    is what lets a record that *does* carry per-step advantages (`verifiers` writes them) be read
    without a second path.

    ``loss_mask`` is in id coordinates, ``(B, T)``, 1 where that token is a target the loss is taken
    over. The target-coordinate mask the parts live in is ``loss_mask[:, 1:]``, and `trained` does
    that shift in exactly one place.
    """

    input_ids: np.ndarray
    attention_mask: np.ndarray
    loss_mask: np.ndarray
    advantages: np.ndarray
    aggregation: Aggregation = "token"
    #: Free-form provenance: which record, which step, which policy version, what was assumed.
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ids = np.asarray(self.input_ids, dtype=np.int64)
        if ids.ndim != 2:
            raise ValueError(f"input_ids must be (B, T); got shape {ids.shape}")
        object.__setattr__(self, "input_ids", ids)
        for name in ("attention_mask", "loss_mask"):
            arr = np.asarray(getattr(self, name)).astype(np.int64)
            if arr.shape != ids.shape:
                raise ValueError(
                    f"{name} has shape {arr.shape} against input_ids {ids.shape}. A mask of the "
                    f"wrong length is a unit mismatch waiting to be summed."
                )
            object.__setattr__(self, name, arr)
        adv = np.asarray(self.advantages, dtype=np.float64)
        if adv.ndim == 1:
            if adv.shape[0] != ids.shape[0]:
                raise ValueError(
                    f"advantages has {adv.shape[0]} entries against {ids.shape[0]} rollouts."
                )
        elif adv.shape != (ids.shape[0], ids.shape[1] - 1):
            raise ValueError(
                f"a per-token advantage must be (B, T-1) in target coordinates; got {adv.shape} "
                f"against input_ids {ids.shape}."
            )
        object.__setattr__(self, "advantages", adv)
        if self.aggregation not in ("token", "sequence", "none"):
            raise ValueError(
                f"aggregation {self.aggregation!r} is not one of 'token', 'sequence', 'none'. "
                f"Read it off `EstimatorSpec.aggregation` rather than choosing one: it changes the "
                f"shares and not only the total."
            )

    @property
    def n_rollouts(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def trained(self) -> np.ndarray:
        """The ``(B, T-1)`` boolean lattice the objective is summed over, in target coordinates."""
        return self.loss_mask[:, 1:].astype(bool)

    @property
    def n_trained(self) -> int:
        return int(self.trained.sum())

    def weights(self) -> np.ndarray:
        """The scalar multiplying ``log pi(y_{k,t})`` in the objective, per target position.

        This is where the advantage, the loss mask and the aggregation meet, and it is the only
        place any of the three is applied. `token` divides by the total number of trained positions
        in the batch, which is DAPO's rule and TRL's `loss_type='dapo'`; `sequence` divides each row
        by its own trained count and then by the number of rows, which is the per-sequence rule;
        `none` is the raw sum, which is what a caller checking the identity against a hand-written
        update wants.
        """
        trained = self.trained.astype(np.float64)
        adv = self.advantages
        per_token = adv if adv.ndim == 2 else adv[:, None] * np.ones_like(trained)
        raw = trained * per_token
        if self.aggregation == "none":
            return raw
        if self.aggregation == "token":
            total = trained.sum()
            return raw / total if total else raw
        counts = trained.sum(axis=1, keepdims=True)
        counts = np.where(counts > 0, counts, 1.0)
        return raw / (counts * max(self.n_rollouts, 1))


# ---------------------------------------------------------------------------
# Partition builders
# ---------------------------------------------------------------------------


def by_rollout(batch: TrajectoryBatch, names: Sequence[str] | None = None) -> Partition:
    """One part per rollout. The coarsest disintegration and the one a group is read through."""
    trained = batch.trained
    parts = []
    for i in range(batch.n_rollouts):
        mask = np.zeros_like(trained, dtype=bool)
        mask[i] = trained[i]
        label = names[i] if names is not None and i < len(names) else f"rollout {i}"
        parts.append(Part(name=label, kind="rollout", mask=mask))
    return Partition(parts=tuple(parts), trained=trained)


@dataclass(frozen=True)
class Segment:
    """A named interval of one rollout, in **id** coordinates, half-open.

    Id coordinates rather than target coordinates because that is what a tokeniser and a record
    hand you: a turn occupies ids ``[lo, hi)``. `by_segment` does the single shift into target
    coordinates, and it is the only place the shift happens.
    """

    row: int
    lo: int
    hi: int
    name: str
    kind: str = "span"


def by_segment(batch: TrajectoryBatch, segments: Sequence[Segment]) -> Partition:
    """A part per named segment, plus one for whatever the segments leave over.

    The leftover part is not optional and it is not a rounding gap. A caller who names the two
    tool-call spans in a turn and nothing else has described 8% of the trained positions, and a
    report of two shares summing to 0.08 with no third row is a report that looks like it lost 92%
    of the step. Naming the remainder is the same discipline `StepCost.residual_share` applies to
    the cost book.
    """
    trained = batch.trained
    by_name: dict[tuple[str, str], np.ndarray] = {}
    for seg in segments:
        if not 0 <= seg.row < batch.n_rollouts:
            raise PartitionError(
                f"segment {seg.name!r} names rollout {seg.row} and the batch has "
                f"{batch.n_rollouts}."
            )
        lo = max(seg.lo - 1, 0)
        hi = max(seg.hi - 1, 0)
        mask = by_name.setdefault((seg.name, seg.kind), np.zeros_like(trained, dtype=bool))
        mask[seg.row, lo:hi] = True
    parts = [
        Part(name=name, kind=kind, mask=mask & trained) for (name, kind), mask in by_name.items()
    ]
    covered = np.zeros_like(trained, dtype=bool)
    for part in parts:
        covered |= part.mask
    leftover = trained & ~covered
    if leftover.any() or not parts:
        parts.append(Part(name="elsewhere", kind="span", mask=leftover))
    return Partition(parts=tuple(parts), trained=trained)


def batch_from_trajectories(
    tokenizer: Any,
    trajectories: Sequence[Any],
    *,
    trained_roles: Sequence[str] = TRAINED_ROLES,
    aggregation: str = "token",
    max_length: int = 2048,
) -> tuple[TrajectoryBatch, list[Segment]] | Refusal:
    """Build a batch and its turn segments from record trajectories.

    The bridge from `record/` to the numerics, and the only function in this module that knows what
    a `Turn` is. It tokenises each turn's own text and concatenates, which is what a trainer does
    (prompt tokenised, completion tokenised, joined) and is not the same as tokenising the
    concatenated string: a tokeniser can merge across a turn boundary, and a turn decomposition
    built on merged tokens attributes a token to whichever turn won the merge.

    Uses `Turn.token_ids` where the record carries them and tokenises the text where it does not,
    saying which on the batch's notes. Uses `Turn.loss_mask` where present; otherwise marks a turn
    trained if its role is in ``trained_roles``, which is a stated default and is recorded.

    Returns the batch and the per-turn `Segment` list, because the second is only derivable while
    the offsets are in scope and recomputing it later is where an off-by-one gets in.
    """
    rows: list[list[int]] = []
    masks: list[list[int]] = []
    advantages: list[float] = []
    segments: list[Segment] = []
    tokenised_from_text = 0
    inferred_masks = 0
    skipped = 0

    for trajectory in trajectories:
        advantage = getattr(trajectory, "advantage", None)
        if advantage is None:
            skipped += 1
            continue
        ids: list[int] = []
        mask: list[int] = []
        bounds: list[tuple[int, int, str, str]] = []
        for turn in getattr(trajectory, "turns", ()):
            role = getattr(turn, "role", "")
            token_ids = getattr(turn, "token_ids", None)
            if token_ids is None:
                token_ids = tuple(
                    tokenizer(getattr(turn, "text", ""), add_special_tokens=False)["input_ids"]
                )
                tokenised_from_text += 1
            loss_mask = getattr(turn, "loss_mask", None)
            if loss_mask is None:
                loss_mask = tuple(role in trained_roles for _ in token_ids)
                inferred_masks += 1
            lo = len(ids)
            ids.extend(int(t) for t in token_ids)
            mask.extend(int(bool(b)) for b in loss_mask)
            bounds.append((lo, len(ids), _turn_label(int(getattr(turn, "index", 0)), role), "turn"))
        if not ids or len(ids) > max_length:
            skipped += 1
            continue
        row = len(rows)
        rows.append(ids)
        masks.append(mask)
        advantages.append(float(advantage))
        segments.extend(Segment(row=row, lo=lo, hi=hi, name=n, kind=k) for lo, hi, n, k in bounds)

    if not rows:
        return Refusal(
            instrument="CreditDisintegration",
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"none of {len(trajectories)} trajectories produced a usable row "
                f"({skipped} carried no advantage or no tokens)"
            ),
            remedy=(
                "a rollout whose every reward function abstained is masked and never receives an "
                "advantage, which is the record being correct. Read a step whose group scored, or "
                "raise `max_length` if the trajectories were dropped for being long."
            ),
            statistics={"n_trajectories": len(trajectories), "n_skipped": skipped},
        )

    width = max(len(r) for r in rows)
    input_ids = np.zeros((len(rows), width), dtype=np.int64)
    attention = np.zeros_like(input_ids)
    loss = np.zeros_like(input_ids)
    for i, (ids, mask) in enumerate(zip(rows, masks)):
        input_ids[i, : len(ids)] = ids
        attention[i, : len(ids)] = 1
        loss[i, : len(mask)] = mask

    notes = [f"{len(rows)} rollouts from the record, right-padded to {width} tokens"]
    if tokenised_from_text:
        notes.append(
            f"{tokenised_from_text} turns carried no `token_ids` and were tokenised from their "
            f"text, per turn rather than over the concatenated string"
        )
    if inferred_masks:
        notes.append(
            f"{inferred_masks} turns carried no `loss_mask`; trained positions were inferred from "
            f"the turn role, treating {list(trained_roles)} as trained"
        )
    if skipped:
        notes.append(f"{skipped} trajectories were skipped for carrying no advantage or no tokens")
    return (
        TrajectoryBatch(
            input_ids=input_ids,
            attention_mask=attention,
            loss_mask=loss,
            advantages=np.asarray(advantages, dtype=np.float64),
            aggregation=aggregation,
            notes=tuple(notes),
        ),
        segments,
    )


def turn_segments(
    turn_bounds: Sequence[Sequence[tuple[int, int, str, str]]],
) -> list[Segment]:
    """Flatten per-rollout turn bounds into segments.

    ``turn_bounds[i]`` is that rollout's list of ``(lo, hi, name, kind)`` in id coordinates. The
    indirection exists because a batch is padded and ragged: rollout 3's turn 2 starts at a
    different id than rollout 0's, and there is no shape that expresses that except a list.
    """
    return [
        Segment(row=i, lo=lo, hi=hi, name=name, kind=kind)
        for i, bounds in enumerate(turn_bounds)
        for (lo, hi, name, kind) in bounds
    ]


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class CreditReport:
    """One step's credit, disintegrated, with the audit that says whether it closed.

    ``projected_share`` is the share. It is ``<g_S, g_full> / |g_full|^2`` and it sums to exactly 1
    by linearity, which is the only reason a share is a meaningful word here. ``norm_share`` is
    ``|g_S| / |g_full|`` beside it and does **not** sum to 1; the gap between the two is
    ``cancellation``, and reporting only the norms is how an attribution heuristic ends up claiming
    170% of a step.

    A projected share can be negative, and that is a finding rather than a defect: a part whose
    gradient opposes the direction the step actually went contributed negative credit to it.

    ``conservation_error`` is ``|sum_S g_S - g_full| / |g_full|`` with ``g_full`` taken by its own
    independent backward pass. Deriving ``g_full`` as the sum of the parts would make the number
    identically zero and the audit vacuous, so it is not done that way anywhere in this module.
    """

    parts: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    projected_share: list[float] = field(default_factory=list)
    norm_share: list[float] = field(default_factory=list)
    n_positions: list[int] = field(default_factory=list)
    conservation_error: float = float("nan")
    cancellation: float = float("nan")
    total_mass: float = float("nan")
    effective_parts: float = float("nan")
    top_share: float = float("nan")
    top_part: str = ""
    n_parts: int = 0
    n_rollouts: int = 0
    n_trained_positions: int = 0
    n_parameters: int = 0
    n_backwards: int = 0
    dtype_used: str = ""
    dtype_native: str = ""
    aggregation: str = "token"
    #: Set only by `step_conservation`, which takes a real optimizer step. None everywhere else,
    #: because `sum mu = dtheta/eta` is a claim about the optimizer and not about the gradient.
    update_conservation_error: float | None = None
    update_optimizer: str = ""
    #: ``eps(dtype) * |theta| / (eta * |g|)``: the smallest relative error the parameter difference
    #: can resolve at this learning rate. Reported because at the reference record's own learning
    #: rate of 1e-06 in float32 it is 7.9, meaning the check cannot resolve the identity at all,
    #: and a reader handed only the error would read a precision limit as a failed audit.
    update_precision_floor: float | None = None
    update_precision_limited: bool = False
    notes: list[str] = field(default_factory=list)
    says: str = ""

    def share_of(self, name: str) -> float:
        return self.projected_share[self.parts.index(name)]

    def render(self) -> str:
        rows = sorted(
            zip(self.parts, self.kinds, self.projected_share, self.norm_share, self.n_positions),
            key=lambda r: -abs(r[2]),
        )
        lines = [self.says or "credit disintegration"]
        for name, kind, proj, norm, n in rows:
            lines.append(
                f"    {proj:+7.2%}  (|g_S|/|g| {norm:5.3f}, {n:5,} tokens)  {name} [{kind}]"
            )
        lines.append(
            f"    conservation error {self.conservation_error:.3e} in {self.dtype_used}; "
            f"cancellation {self.cancellation:.3f}; {self.n_backwards} backward passes"
        )
        return "\n".join(lines)


def _say(report: CreditReport) -> str:
    """The one sentence a reader takes away, built from the numbers that were measured."""
    if not report.parts:
        return "no parts"
    order = sorted(range(len(report.parts)), key=lambda i: -abs(report.projected_share[i]))
    head = order[: min(3, len(order))]
    listed = ", ".join(f"{report.projected_share[i]:.0%} on {report.parts[i]}" for i in head)
    return (
        f"This step's credit disintegrates as: {listed}"
        + (", and the rest elsewhere" if len(order) > len(head) else "")
        + f". Total mass matches the gradient to {report.conservation_error:.1e} "
        f"({report.dtype_used}), over {report.n_backwards} backward passes."
    )


# ---------------------------------------------------------------------------
# The numerics
# ---------------------------------------------------------------------------


def _model_of(policy: Any) -> Any:
    """The torch module behind a policy, or an error naming what to expose.

    `PolicySubject` offers `grad_h` (w.r.t. an activation site) and `token_gradients` (w.r.t. the
    input embeddings) and **no method that differentiates with respect to the parameters**. This is
    the first instrument that needs `grad_theta`, so it reaches the module the way
    `policy.base.site_weights` already does, through `runtime.model`, and the missing protocol
    method is not added here.
    """
    runtime = getattr(policy, "runtime", policy)
    model = getattr(runtime, "model", None)
    if model is None:
        raise TypeError(
            f"{type(policy).__name__} exposes no `runtime.model`, so there are no parameters to "
            f"differentiate. The credit measure needs grad_theta and `PolicySubject` declares no "
            f"method for it: implement `parameters()` on the runtime, or pass an `HFPolicy`."
        )
    return model


def _trainable(model: Any) -> list[Any]:
    return [p for p in model.parameters() if p.requires_grad]


def _flat(grads: Sequence[Any]) -> "torch.Tensor":
    import torch

    return torch.cat([g.reshape(-1).to(torch.float64) for g in grads])


@dataclass(frozen=True)
class _Graph:
    """A built forward graph and the per-position log-probabilities hanging off it."""

    logprobs: Any  # (B, T-1) float32, differentiable
    params: list[Any]
    n_parameters: int


def _build_graph(model: Any, batch: TrajectoryBatch) -> _Graph:
    """One forward pass, retained, giving ``log pi(y_{t+1} | y_{<=t})`` at every target position."""
    import torch

    ids = torch.as_tensor(batch.input_ids, dtype=torch.long)
    attn = torch.as_tensor(batch.attention_mask, dtype=torch.long)
    out = model(input_ids=ids, attention_mask=attn, use_cache=False)
    logits = out.logits.to(torch.float32)
    logsoftmax = torch.log_softmax(logits[:, :-1], dim=-1)
    targets = ids[:, 1:].unsqueeze(-1)
    logprobs = logsoftmax.gather(-1, targets).squeeze(-1)
    params = _trainable(model)
    return _Graph(
        logprobs=logprobs,
        params=params,
        n_parameters=int(sum(p.numel() for p in params)),
    )


def disintegrate(
    policy: Any,
    batch: TrajectoryBatch,
    partition: Partition,
    *,
    dtype: str = "float32",
    conservation_tol: float = 1e-5,
) -> CreditReport | Refusal:
    """Disintegrate one step's credit over a partition, and audit that the parts close.

    ``m + 1`` backward passes over one retained forward graph: one for the full objective and one
    per part. Peak memory is three parameter-sized float64 buffers plus whatever the graph holds,
    which is what makes a forty-part disintegration of a long episode possible at all.

    ``dtype`` casts the model for the measurement and restores it afterwards. The cast is lossless
    in both directions when the native dtype is bfloat16 or float16, because float32 has strictly
    more of both exponent and mantissa, and it is the difference between a conservation error of
    1.56e-03 and one of 1.13e-07 on the reference record. Pass ``dtype=""`` to measure in the model's
    own dtype, which is the right thing to do when the number you want is the numerics floor rather
    than the credit.

    Returns a `Refusal` when the objective has no mass to disintegrate, which is a real case: a
    group whose advantages are all zero (a degenerate group) produces an exactly zero
    gradient, and every share of zero is undefined rather than uniform.
    """
    import torch

    if batch.n_trained == 0:
        return Refusal(
            instrument="CreditDisintegration",
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"the loss mask leaves no trained position across {batch.n_rollouts} rollouts of "
                f"{batch.seq_len} tokens, so the objective is empty"
            ),
            remedy=(
                "supply a loss mask marking the tokens the loss was taken over, or pass "
                "`trained_roles` naming which turn roles are trained. A batch whose mask is all "
                "zero has no gradient and no credit to disintegrate."
            ),
            statistics={"n_rollouts": batch.n_rollouts, "seq_len": batch.seq_len},
        )

    model = _model_of(policy)
    native = str(next(model.parameters()).dtype)
    was_training = model.training
    target = getattr(torch, dtype) if dtype else None
    try:
        model.eval()
        if target is not None:
            model.to(target)
        graph = _build_graph(model, batch)
        weights = torch.as_tensor(batch.weights(), dtype=torch.float32)
        objective = (weights * graph.logprobs).sum()
        full = _flat(torch.autograd.grad(objective, graph.params, retain_graph=True))
        full_norm = float(torch.linalg.vector_norm(full))
        if not math.isfinite(full_norm) or full_norm == 0.0:
            return Refusal(
                instrument="CreditDisintegration",
                reason=RefusalReason.BELOW_LOD,
                detail=(
                    f"the full gradient of the objective has norm {full_norm!r} over "
                    f"{batch.n_trained:,} trained positions, so there is no mass to disintegrate"
                ),
                remedy=(
                    "check the advantages: a group in which every rollout scored the same has an "
                    "advantage of exactly zero on every token, which is a degenerate "
                    "group rather than a measurement problem. Pool a window of steps, or read a "
                    "step whose group separated."
                ),
                statistics={
                    "gradient_norm": full_norm,
                    "n_trained_positions": batch.n_trained,
                    "advantage_absmax": float(np.abs(batch.advantages).max()),
                },
            )

        running = torch.zeros_like(full)
        projected: list[float] = []
        norms: list[float] = []
        for part in partition.parts:
            if part.n_positions == 0:
                projected.append(0.0)
                norms.append(0.0)
                continue
            selector = torch.as_tensor(part.mask, dtype=torch.float32)
            piece = ((weights * selector) * graph.logprobs).sum()
            g_part = _flat(torch.autograd.grad(piece, graph.params, retain_graph=True))
            running += g_part
            projected.append(float(torch.dot(g_part, full)) / (full_norm**2))
            norms.append(float(torch.linalg.vector_norm(g_part)) / full_norm)
        conservation = float(torch.linalg.vector_norm(running - full)) / full_norm
    finally:
        if target is not None:
            model.to(getattr(torch, native.replace("torch.", "")))
        if was_training:
            model.train()

    shares = np.asarray(projected, dtype=np.float64)
    weightsq = float(np.sum(shares**2))
    top = int(np.argmax(np.abs(shares))) if shares.size else 0
    report = CreditReport(
        parts=list(partition.names),
        kinds=list(partition.kinds),
        projected_share=[float(x) for x in shares],
        norm_share=[float(x) for x in norms],
        n_positions=[p.n_positions for p in partition.parts],
        conservation_error=conservation,
        cancellation=float(sum(norms)),
        total_mass=full_norm,
        effective_parts=float(1.0 / weightsq) if weightsq > 0 else float("nan"),
        top_share=float(shares[top]) if shares.size else float("nan"),
        top_part=partition.names[top] if shares.size else "",
        n_parts=len(partition),
        n_rollouts=batch.n_rollouts,
        n_trained_positions=batch.n_trained,
        n_parameters=graph.n_parameters,
        n_backwards=1 + sum(1 for p in partition.parts if p.n_positions > 0),
        dtype_used=dtype or native.replace("torch.", ""),
        dtype_native=native.replace("torch.", ""),
        aggregation=batch.aggregation,
        notes=list(batch.notes),
    )
    if conservation > conservation_tol:
        report.notes.append(
            f"conservation error {conservation:.3e} exceeds the tolerance {conservation_tol:.0e}; "
            f"in {report.dtype_used} on {report.n_parameters:,} parameters that is a numerical "
            f"finding rather than an arithmetic one, and the partition tiles by construction"
        )
    report.says = _say(report)
    return report


#: Optimizers for which `sum_{k,t} mu(k,t) = dtheta / eta` is an identity rather than an
#: approximation. Exactly one, and the shortness of this tuple is the point: momentum carries state
#: across steps, Adam divides by a running second moment, weight decay adds a term the credit
#: measure does not contain, and gradient clipping rescales the whole update by a factor nobody
#: records per parameter. Each of those breaks the identity in a different way and none of them
#: breaks the *disintegration* identity, which is why the two are checked separately.
EXACT_UPDATE_OPTIMIZERS: tuple[str, ...] = ("SGD",)


def step_conservation(
    policy: Any,
    batch: TrajectoryBatch,
    partition: Partition,
    *,
    lr: float,
    dtype: str = "float64",
) -> CreditReport | Refusal:
    """Disintegrate, then take a real optimizer step and check the parts sum to it.

    This is the conservation check in one function. It runs the disintegration, snapshots every
    parameter, drives one `torch.optim.SGD` step on the negated objective at learning rate ``lr``,
    measures ``dtheta / eta``, compares it against the summed parts, and restores the snapshot.

    The optimizer is real rather than hand-written, because the identity being checked is a claim
    about what an optimizer does and asserting it against ``p -= lr * p.grad`` written three lines
    above would be asserting it against itself. It is `SGD` and only `SGD`: on Adam the identity is
    false and the honest thing is to say so rather than to widen a tolerance until it passes, which
    is why `EXACT_UPDATE_OPTIMIZERS` has one member.

    ``dtheta = +eta * grad J`` here, not ``-eta * grad J``, because the objective is the one being
    ascended and the loss handed to the optimizer is its negation. The sign is written out because
    a sign error in this identity produces a conservation error of exactly 2, which looks like a
    catastrophic numerical failure and is not one.

    **The learning rate decides whether this check can resolve anything, and the reference record's
    own learning rate cannot.** ``dtheta / eta`` is recovered by differencing parameters, so the
    error has a floor of ``eps * |theta| / (eta * |g|)``: the update has to be large enough relative
    to a parameter's last representable digit for the subtraction to survive. Measured on the
    reference model at four learning rates and two dtypes, the observed error tracks that floor at a
    constant 0.18 across eight orders of magnitude, which is what makes it a floor rather than a
    guess. In float32 at ``lr = 1e-06`` the floor is 7.9, so the check resolves nothing; in float64
    at the same learning rate it is 1.5e-08 and the measured error is 2.6e-09. Hence the float64
    default, and hence `update_precision_limited`, which is set rather than hidden when the number
    the caller is about to read is the differencing rather than the credit.
    """
    import torch

    report = disintegrate(policy, batch, partition, dtype=dtype)
    if isinstance(report, Refusal):
        return report

    model = _model_of(policy)
    native = str(next(model.parameters()).dtype).replace("torch.", "")
    target = getattr(torch, dtype) if dtype else None
    params = _trainable(model)
    snapshot = [p.detach().clone() for p in params]
    try:
        if target is not None:
            model.to(target)
        params = _trainable(model)
        snapshot32 = [p.detach().clone() for p in params]
        optimizer = torch.optim.SGD(params, lr=lr)
        optimizer.zero_grad(set_to_none=True)
        graph = _build_graph(model, batch)
        weights = torch.as_tensor(batch.weights(), dtype=torch.float32)
        loss = -(weights * graph.logprobs).sum()
        loss.backward()
        summed = _flat([p.grad if p.grad is not None else torch.zeros_like(p) for p in params])
        optimizer.step()
        delta = _flat([(p.detach() - before) for p, before in zip(params, snapshot32)])
        # dtheta = -eta * grad(loss) = +eta * grad(J). `summed` is grad(loss), so the credit
        # measure is its negation and the identity is `delta / lr == -summed`.
        predicted = -summed
        denominator = float(torch.linalg.vector_norm(predicted))
        error = float(torch.linalg.vector_norm(delta / lr - predicted)) / denominator
        theta_norm = float(torch.linalg.vector_norm(_flat(snapshot32)))
        eps = float(torch.finfo(params[0].dtype).eps)
        floor = eps * theta_norm / (lr * denominator) if denominator else float("inf")
    finally:
        with torch.no_grad():
            for p, before in zip(_trainable(model), snapshot):
                p.copy_(before.to(p.dtype))
        if target is not None:
            model.to(getattr(torch, native))
        for p in _trainable(model):
            p.grad = None

    report.update_conservation_error = error
    report.update_optimizer = "SGD"
    report.update_precision_floor = floor
    report.update_precision_limited = bool(floor > 1e-3)
    report.notes.append(
        f"one torch.optim.SGD step at lr={lr:g} moved the parameters by dtheta with "
        f"|dtheta/eta - sum_S g_S| / |sum_S g_S| = {error:.3e}; the identity is checked against "
        f"SGD because it is false for every optimizer that carries state"
    )
    report.notes.append(
        f"differencing {report.n_parameters:,} parameters at lr={lr:g} in "
        f"{report.dtype_used} has a precision floor of {floor:.2e}, and the measured error is "
        f"{error / floor:.2f} times it"
        + (
            ". The check is precision-limited: raise the learning rate or the dtype before "
            "reading this number as an audit of the credit measure."
            if floor > 1e-3
            else "."
        )
    )
    return report


# ---------------------------------------------------------------------------
# Rung 0: consume SkyRL's dump rather than reimplementing it
# ---------------------------------------------------------------------------


#: Where the object on disk says it came from. Verified against SkyRL source on 2026-08-05: the
#: class is `skyrl/backends/skyrl_train/training_batch.py:486` and the rebuild helper is the
#: module-level `_rebuild_tensor_batch` at :112, which exists precisely so the pickle names
#: something importable. Both are matched by name and neither is imported.
SKYRL_MODULE = "skyrl.backends.skyrl_train.training_batch"

#: The only names this reader will resolve out of a dump. Everything else raises. Unpickling is
#: arbitrary code execution by construction, and a reader pointed at somebody else's training
#: artifact is exactly the case where that matters, so the allowlist is the whole security model
#: and it is four entries long.
_SKYRL_ALLOWED: frozenset[tuple[str, str]] = frozenset(
    {
        (SKYRL_MODULE, "_rebuild_tensor_batch"),
        (SKYRL_MODULE, "TrainingInputBatch"),
        (SKYRL_MODULE, "TensorBatch"),
        (SKYRL_MODULE, "TensorList"),
        ("torch", "device"),
    }
)


class _SkyRLStandIn:
    """What `_rebuild_tensor_batch` resolves to here: a marker that keeps the state as data.

    SkyRL's own helper calls `dict.__new__(cls)` and then `obj.__setstate__(state)`, which decodes
    every tensor. Nothing here does that, because the state is already a plain dict of buffers and
    shapes and decoding it with numpy is both cheaper and free of the dependency. The class exists
    only so the unpickler has a callable to name.
    """

    def __init__(self, cls: Any, state: Mapping[str, Any]) -> None:
        self.cls = cls
        self.state = dict(state)


class _SkyRLUnpickler(pickle.Unpickler):
    """An unpickler that resolves four names and refuses every other one.

    This is what "consume the dump rather than reimplement it" costs and it is not much: SkyRL's
    on-disk format is self-describing (a dict of ``{format, data, shape, dtype}`` per tensor), so
    reading it needs the field layout and not the library. The alternative, importing `skyrl` to
    unpickle, would pull Ray, vLLM and Megatron into a reader whose job is to look at an array.
    """

    def find_class(self, module: str, name: str) -> Any:  # noqa: D102
        if (module, name) not in _SKYRL_ALLOWED:
            raise pickle.UnpicklingError(
                f"this reader resolves {sorted(m + '.' + n for m, n in _SKYRL_ALLOWED)} and "
                f"nothing else; the dump asks for {module}.{name}. Unpickling is code execution, "
                f"so an unrecognised name is refused rather than imported. If SkyRL has changed "
                f"its serialisation, re-verify `dump_data` and widen this list deliberately."
            )
        if name == "_rebuild_tensor_batch":
            return _SkyRLStandIn
        if module == "torch":
            import torch

            return torch.device
        return _SkyRLMarker


class _SkyRLMarker:
    """Stands in for `TrainingInputBatch`; only its name is ever used."""


def _decode_skyrl_tensor(blob: Mapping[str, Any] | None) -> np.ndarray | None:
    """One tensor out of SkyRL's serialisation. Numpy for every dtype numpy has.

    The `torch` fallback path exists in SkyRL for dtypes numpy cannot express, bfloat16 above all,
    and it is the one branch here that needs torch. A dump that took that path for a field this
    reader wants is readable only where torch is installed, which is stated rather than discovered.
    """
    if blob is None:
        return None
    fmt = blob.get("format", "numpy")
    if fmt == "torch":
        import io

        import torch

        tensor = torch.load(io.BytesIO(blob["data"]), weights_only=True)
        decoded: np.ndarray = tensor.to(torch.float32).numpy()
        return decoded
    if fmt == "tensor_list":
        raise ValueError(
            "this field is a TensorList (ragged, one tensor per row). The credit measure reads "
            "the rectangular fields; a TensorList field is multimodal input rather than a "
            "per-token quantity."
        )
    array = np.frombuffer(blob["data"], dtype=np.dtype(blob["dtype"]))
    return array.reshape(tuple(blob["shape"])).copy()


@register_payload
@dataclass
class SkyRLDump:
    """One `global_step_N_training_input.pkl`, read without importing SkyRL.

    What is on disk and what is not, verified against source rather than against documentation. The
    intended rung 0 is to consume SkyRL's own dump where it exists, but `dump_data_batch` is a bool
    config field (`config.py:1085`) and the function is `dump_data` (`trainer.py:1238-1244`). The
    sharper problem, which this reader reproduces: **`rewards` is popped at `trainer.py:436` and
    `uids` at `:437`, five lines before the dump at `:441`.** So the dump carries the per-token
    advantage tensor and neither the reward it came from nor the group it belongs to.

    That makes this a partial rung 0 rather than a free one, and the partiality is recoverable in
    one direction only. The group is recoverable: rows sharing a prompt are rows whose
    `response_mask` is zero over the same ids, so `recovered_uids` reconstructs the grouping by
    hashing the prompt. The reward is not recoverable from this file at all, and an instrument that
    needs it has to read the trainer's metrics or re-derive it, which is a different artifact.
    """

    path: str = ""
    keys: list[str] = field(default_factory=list)
    n_rows: int = 0
    seq_len: int = 0
    metadata_keys: list[str] = field(default_factory=list)
    has_advantages: bool = False
    has_rewards: bool = False
    has_uids: bool = False
    recovered_groups: int = 0
    notes: list[str] = field(default_factory=list)

    #: The decoded arrays, kept off the payload's declared fields so a store never inlines a
    #: 500 MB tensor into an Evidence row.
    arrays: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def advantages(self) -> np.ndarray | None:
        return self.arrays.get("advantages")

    @property
    def recovered_uids(self) -> list[str]:
        """A group label per row, recovered from the prompt tokens the response mask excludes.

        SkyRL pops `uids` before the dump, so the grouping is absent and every group-relative
        quantity in this library needs it. Two rows drawn from the same prompt have identical ids
        wherever `response_mask` is zero, so hashing that prefix recovers the partition exactly for
        any sampler that does not perturb the prompt. It does not recover the *original* uid
        strings, and nothing downstream needs them: what is needed is which rows are siblings.
        """
        ids = self.arrays.get("sequences")
        mask = self.arrays.get("response_mask")
        if ids is None or mask is None:
            return []
        out = []
        for row in range(ids.shape[0]):
            prompt = ids[row][mask[row] == 0]
            out.append(hashlib.blake2s(prompt.tobytes(), digest_size=8).hexdigest())
        return out


def read_skyrl_dump(path: Any) -> SkyRLDump | Refusal:
    """Read one SkyRL training-input dump. No `skyrl` import, no `torch` import on the fast path."""
    from pathlib import Path as _Path

    p = _Path(path)
    if not p.exists():
        return Refusal(
            instrument="SkyRLDumpReader",
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=f"no file at {p}",
            remedy=(
                "SkyRL writes these under `<export_path>/dumped_data/` and only when "
                "`trainer.dump_data_batch` is true, which defaults to false "
                "(`config.py:1085`). Set it before the run; the tensor cannot be recovered after."
            ),
            statistics={"path": str(p)},
        )
    with p.open("rb") as handle:
        obj = _SkyRLUnpickler(handle).load()
    if not isinstance(obj, _SkyRLStandIn):
        return Refusal(
            instrument="SkyRLDumpReader",
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=f"{p} unpickled to {type(obj).__name__}, not a TensorBatch rebuild",
            remedy=(
                "point this at a file written by `SkyRLTrainer.dump_data`, whose top-level object "
                "is a `TrainingInputBatch`. A checkpoint or a metrics file is a different artifact."
            ),
            statistics={"path": str(p), "top_level": type(obj).__name__},
        )
    state = obj.state
    batch_dict = state.get("batch_dict", {})
    metadata = dict(state.get("metadata") or {})
    arrays: dict[str, Any] = {}
    notes: list[str] = []
    for key, blob in batch_dict.items():
        try:
            arrays[key] = _decode_skyrl_tensor(blob)
        except ValueError as exc:
            notes.append(f"{key}: {exc}")
    present = {k for k, v in arrays.items() if v is not None}
    ids = arrays.get("sequences")
    dump = SkyRLDump(
        path=str(p),
        keys=sorted(present),
        n_rows=int(ids.shape[0]) if ids is not None else int(state.get("batch_size") or 0),
        seq_len=int(ids.shape[1]) if ids is not None else 0,
        metadata_keys=sorted(metadata),
        has_advantages="advantages" in present,
        has_rewards="rewards" in present,
        has_uids="uids" in metadata,
        notes=notes,
        arrays=arrays,
    )
    dump.recovered_groups = len(set(dump.recovered_uids))
    if not dump.has_rewards or not dump.has_uids:
        dump.notes.append(
            "SkyRL pops `rewards` (trainer.py:436) and `uids` (trainer.py:437) before calling "
            "`dump_data` (trainer.py:441), so neither reaches disk. The grouping is recovered by "
            f"hashing the prompt tokens the response mask excludes, giving "
            f"{dump.recovered_groups} groups over {dump.n_rows} rows; the reward is not "
            "recoverable from this file and has to come from the trainer's own metrics."
        )
    return dump


def batch_from_skyrl(dump: SkyRLDump, *, aggregation: str = "token") -> TrajectoryBatch | Refusal:
    """Turn a dump into the batch the disintegration reads. This is the whole of rung 0.

    The per-token advantage tensor every framework computes and discards is right here, already on
    disk, already aligned to the ids and the mask. Nothing in this function computes an advantage,
    and that is the point of the rung: it consumes the tensor rather than reimplementing the
    estimator that produced it.
    """
    ids = dump.arrays.get("sequences")
    adv = dump.arrays.get("advantages")
    if ids is None or adv is None:
        return Refusal(
            instrument="SkyRLDumpReader",
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"{dump.path} carries {sorted(dump.keys)}; the credit measure needs `sequences` "
                f"and `advantages`"
            ),
            remedy=(
                "dump a batch taken after `compute_advantages_and_returns`, which is where "
                "`advantages` is added (`trainer.py:1126`). A batch dumped before it has the "
                "sequences and not the tensor."
            ),
            statistics={"keys": sorted(dump.keys)},
        )
    mask = dump.arrays.get("loss_mask")
    if mask is None:
        mask = dump.arrays.get("response_mask")
    if mask is None:
        return Refusal(
            instrument="SkyRLDumpReader",
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=f"{dump.path} carries neither `loss_mask` nor `response_mask`",
            remedy=(
                "the mask decides what 'per token' means and it is not inferable from the ids: a "
                "batch with neither cannot say which positions the loss was taken over. Dump one "
                "that carries `loss_mask`."
            ),
            statistics={"keys": sorted(dump.keys)},
        )
    attention = dump.arrays.get("attention_mask")
    if attention is None:
        attention = np.ones_like(ids)
    per_token = np.asarray(adv, dtype=np.float64)
    if per_token.ndim == 2 and per_token.shape[1] == ids.shape[1]:
        per_token = per_token[:, 1:]
    return TrajectoryBatch(
        input_ids=ids,
        attention_mask=attention,
        loss_mask=mask,
        advantages=per_token,
        aggregation=aggregation,
        notes=(
            f"SkyRL dump {dump.path}",
            f"per-token advantages consumed from the dump, not recomputed; "
            f"{dump.recovered_groups} groups recovered by prompt hash because uids were popped",
        ),
    )


# ---------------------------------------------------------------------------
# G2: turn and tool-call disintegration, from the record alone
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class TurnMass:
    """Where a trajectory's advantage-weighted log-probability mass sits, by turn and by tool call.

    **Why turns and not tokens.** Every token in a trajectory receives the identical scalar
    advantage: a 500,000-token SWE-bench episode gives every one of its tokens the same number. So
    the credit measure is uniform over the trajectory by construction and all the structure lives in
    ``grad log pi``, not in ``A``. The disintegration that carries information is the one that
    follows the places log-probability mass concentrates and the loss mask changes, which is the
    turn and the tool call.

    **This is a proxy and the word is load-bearing.** The quantity here is
    ``sum_t |A_k| * (-log pi(y_{k,t}))`` over each turn, normalised. Surprisal is not gradient norm.
    They are related, because a token the policy was already certain of contributes little to either,
    and the relation is not an identity: a token can be surprising and sit in a flat direction of
    parameter space. `CreditDisintegration` measures the real thing at one backward pass per turn,
    and the honest use of this payload is as the cheap estimate whose error against that is
    measurable rather than assumed.

    ``share`` is non-negative and sums to 1, which is what the registry's ``shares`` unit with
    ``scale: normalised`` means. ``signed_share`` carries the same decomposition with the advantage's
    sign kept, so a turn that contributed against the direction of the step can be seen; it also sums
    to 1 and its entries can be negative.
    """

    by_turn: dict[str, float] = field(default_factory=dict)
    signed_by_turn: dict[str, float] = field(default_factory=dict)
    by_tool_call: dict[str, float] = field(default_factory=dict)
    by_span_kind: dict[str, float] = field(default_factory=dict)
    turn_order: list[str] = field(default_factory=list)
    n_trajectories: int = 0
    n_turns: int = 0
    n_turns_scored: int = 0
    n_tool_calls: int = 0
    n_tokens_scored: int = 0
    head_share: float = float("nan")
    tail_share: float = float("nan")
    head_tail_share: float = float("nan")
    n_head: int = 0
    n_tail: int = 0
    effective_turns: float = float("nan")
    top_share: float = float("nan")
    top_turn: str = ""
    tool_call_share: float = 0.0
    proxy: str = "advantage-weighted surprisal"
    notes: list[str] = field(default_factory=list)
    says: str = ""

    def render(self) -> str:
        lines = [self.says]
        for name in self.turn_order:
            lines.append(f"    {self.by_turn[name]:7.2%}  {name}")
        if self.by_tool_call:
            for name, share in sorted(self.by_tool_call.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {share:7.2%}  tool call {name!r}")
        return "\n".join(lines)


def _turn_label(index: int, role: str) -> str:
    return f"turn {index} ({role})"


def turn_mass(
    trajectories: Sequence[Any],
    *,
    trained_roles: Sequence[str] = TRAINED_ROLES,
    n_head: int = 3,
    n_tail: int = 1,
) -> TurnMass | Refusal:
    """G2 at rung 0: the turn and tool-call decomposition a record alone supports.

    Reads `Turn.logprobs_sampling` (falling back to `logprobs_train` and saying which it used) and
    `Turn.loss_mask`, and weights each turn's summed surprisal by its trajectory's advantage.

    Refuses rather than substituting a token count when the per-token log-probabilities are absent.
    A share built from token counts is a different quantity wearing this one's name: it is the
    "uniform attribution over tokens" baseline this instrument is measured against, and returning it
    here under the label `credit.by_turn` would be a silent downgrade of the quantity.
    """
    turns_total = 0
    with_logprobs = 0
    stream_used: set[str] = set()
    scored: dict[str, float] = {}
    signed: dict[str, float] = {}
    order: list[str] = []
    tools: dict[str, float] = {}
    spans: dict[str, float] = {}
    tool_calls = 0
    tokens = 0
    missing_advantage = 0

    for trajectory in trajectories:
        advantage = getattr(trajectory, "advantage", None)
        if advantage is None:
            missing_advantage += 1
            continue
        weight = abs(float(advantage))
        signed_weight = float(advantage)
        for turn in getattr(trajectory, "turns", ()):
            turns_total += 1
            role = getattr(turn, "role", "")
            logprobs = getattr(turn, "logprobs_sampling", None)
            stream = "logprobs_sampling"
            if logprobs is None:
                logprobs = getattr(turn, "logprobs_train", None)
                stream = "logprobs_train"
            if logprobs is None:
                continue
            with_logprobs += 1
            mask = getattr(turn, "loss_mask", None)
            if mask is None:
                mask = tuple(role in trained_roles for _ in logprobs)
            surprisal = [
                -float(lp) for lp, keep in zip(logprobs, mask) if keep and math.isfinite(float(lp))
            ]
            if not surprisal:
                continue
            stream_used.add(stream)
            total = float(sum(surprisal))
            tokens += len(surprisal)
            label = _turn_label(int(getattr(turn, "index", 0)), role)
            if label not in scored:
                order.append(label)
            scored[label] = scored.get(label, 0.0) + weight * total
            signed[label] = signed.get(label, 0.0) + signed_weight * total
            call = getattr(turn, "tool_call", None)
            if call is not None:
                tool_calls += 1
                tools[call.name] = tools.get(call.name, 0.0) + weight * total
            for span in getattr(turn, "spans", ()) or ():
                lo, hi = int(span.start), int(span.end)
                inside = float(sum(surprisal[max(lo, 0) : max(hi, 0)]))
                spans[span.kind] = spans.get(span.kind, 0.0) + weight * inside

    if not trajectories:
        return Refusal(
            instrument="TurnCredit",
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail="no trajectories were supplied",
            remedy="pass the trajectories of a step or a group, read through `open_run`.",
            statistics={"n_trajectories": 0},
        )
    if with_logprobs == 0:
        return Refusal(
            instrument="TurnCredit",
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"none of the {turns_total:,} turns across {len(trajectories):,} trajectories "
                f"carries `logprobs_sampling` or `logprobs_train`"
            ),
            remedy=(
                "the turn decomposition is a statement about log-probability mass, so it needs the "
                "per-token log-probabilities. A `verifiers` record carries them "
                "(`completion_logprobs` lands in `logprobs_sampling`); the TRL tap does not write "
                "them, so on a TRL record use `CreditDisintegration` with a turn partition, which "
                "recomputes them under the policy and measures the real gradient share instead of "
                "this proxy. Substituting token counts here would be the 'uniform attribution over "
                "tokens' baseline reported under this quantity's name."
            ),
            statistics={"n_turns": turns_total, "n_trajectories": len(trajectories)},
        )
    total_mass = float(sum(scored.values()))
    if total_mass <= 0.0:
        return Refusal(
            instrument="TurnCredit",
            reason=RefusalReason.BELOW_LOD,
            detail=(
                f"the advantage-weighted surprisal totals {total_mass:.3g} over {tokens:,} scored "
                f"tokens, so there are no shares to report"
            ),
            remedy=(
                "check the advantages: a step whose every group was degenerate carries an "
                "advantage of exactly zero on every trajectory, which is a degenerate group rather "
                "than a measurement problem. Read a step whose groups separated."
            ),
            statistics={"n_tokens_scored": tokens, "total_mass": total_mass},
        )

    shares = {k: v / total_mass for k, v in scored.items()}
    signed_total = float(sum(signed.values()))
    signed_shares = (
        {k: v / signed_total for k, v in signed.items()}
        if abs(signed_total) > 0
        else {k: float("nan") for k in signed}
    )
    ordered = [shares[name] for name in order]
    head = float(sum(ordered[:n_head]))
    tail = float(sum(ordered[-n_tail:])) if n_tail and len(ordered) > n_head else 0.0
    sumsq = float(sum(s * s for s in shares.values()))
    top = max(shares, key=lambda k: shares[k])
    payload = TurnMass(
        by_turn=shares,
        signed_by_turn=signed_shares,
        by_tool_call={k: v / total_mass for k, v in tools.items()},
        by_span_kind={k: v / total_mass for k, v in spans.items()},
        turn_order=order,
        n_trajectories=len(trajectories),
        n_turns=turns_total,
        n_turns_scored=with_logprobs,
        n_tool_calls=tool_calls,
        n_tokens_scored=tokens,
        head_share=head,
        tail_share=tail,
        head_tail_share=head + tail,
        n_head=min(n_head, len(ordered)),
        n_tail=min(n_tail, max(len(ordered) - n_head, 0)),
        effective_turns=float(1.0 / sumsq) if sumsq > 0 else float("nan"),
        top_share=shares[top],
        top_turn=top,
        tool_call_share=float(sum(tools.values()) / total_mass),
    )
    payload.notes.append(
        f"log-probabilities read from {', '.join(sorted(stream_used))}; the loss mask was "
        + (
            "taken from the record"
            if any(
                getattr(t, "loss_mask", None) is not None for tr in trajectories for t in tr.turns
            )
            else f"inferred from the turn role, treating {list(trained_roles)} as trained"
        )
    )
    if missing_advantage:
        payload.notes.append(
            f"{missing_advantage} of {len(trajectories)} trajectories carry no advantage and "
            f"contributed nothing; a masked rollout never receives one and that is the record "
            f"being correct rather than incomplete"
        )
    payload.says = (
        f"In a {len(payload.turn_order)}-turn decomposition over {payload.n_trajectories} "
        f"trajectories, {payload.head_tail_share:.0%} of the log-probability mass that received "
        f"advantage is in turns {payload.turn_order[0]} to {payload.turn_order[payload.n_head - 1]}"
        + (f" and {payload.turn_order[-1]}" if payload.n_tail else "")
        + f". Effective number of turns {payload.effective_turns:.1f} of "
        f"{len(payload.turn_order)}."
    )
    return payload


# ---------------------------------------------------------------------------
# G3: the implicit process reward model, extracted
# ---------------------------------------------------------------------------


@dataclass
class _Node:
    """One prefix in the trie, with the rollouts that pass through it."""

    depth: int
    members: list[int] = field(default_factory=list)
    children: dict[int, "_Node"] = field(default_factory=dict)
    value: float = float("nan")
    n_samples: int = 0
    from_reroll: bool = False


@register_payload
@dataclass
class ImplicitPRM:
    """GRPO's induced per-step credit function, extracted from artifacts that already exist.

    GRPO with an outcome reward is equivalent to a PRM-aware objective whose process reward model is
    a Monte-Carlo value function. That implicit PRM is not trained and not inferred: it is
    ``q(s) = E[R | prefix s]``, and the induced process reward on the step from a prefix to its
    child is ``q(child) - q(parent)``. Every ingredient is on disk already, and nobody has taken it
    out.

    **The whole thing turns on whether rollouts share prefixes, and that is measured rather than
    assumed.** A group's rollouts all share the prompt, so ``q(root)`` is the group mean, which is
    exactly GRPO's own baseline. Past the root, ``q`` is a Monte-Carlo estimate only where two or
    more rollouts still agree. Where a rollout is alone, ``q`` at every one of its prefixes equals
    its own outcome, the process reward on the first divergent step equals ``r_k - mean(r)``, and
    every step after it is exactly zero. So with no prefix sharing the induced function puts the
    whole outcome advantage on the first token and says nothing about the rest of the trajectory.

    **G3's kill condition, and the two different ways it fires.** `is_constant` is the literal
    condition: the induced process reward has zero variance across every position, so there is
    nothing in it at all. `is_degenerate` is the one that fires on a real GRPO record: no position
    past the root carries a Monte-Carlo value from more than one rollout, so the function is the
    outcome advantage wearing a per-step name. They are separate fields because the remedies differ.
    A constant function is a dead end. A degenerate one is fixed by re-rolling, which is rung 1, and
    on a sampler whose rollouts never branch that is the only way to get a value at a prefix at all.

    `informative_fraction` is the number to read first. It is the fraction of trained positions
    whose value came from two or more rollouts, and it is the fraction of the trajectory on which
    this instrument says anything at all.
    """

    rung: int = 0
    n_rollouts: int = 0
    n_nodes: int = 0
    n_nodes_estimated: int = 0
    max_depth: int = 0
    branching_depth: int = 0
    mean_divergence_depth: float = float("nan")
    max_divergence_depth: int = 0
    root_value: float = float("nan")
    informative_fraction: float = 0.0
    #: The literal kill condition: the induced function has no variation anywhere.
    is_constant: bool = True
    #: The kill condition that fires on a real record: no position past the root has a value
    #: estimated from more than one rollout, so every increment is one rollout's own outcome.
    is_degenerate: bool = True
    #: Fraction of positions whose process reward is exactly zero. On a record with no prefix
    #: sharing this is everything except the first token of each rollout.
    zero_fraction: float = 0.0
    process_reward_sd: float = 0.0
    #: The induced function itself: one entry per (rollout, depth) trained position, in row-major
    #: order, so it can be correlated against a trained PRM or a re-roll on the same positions.
    process_reward: list[float] = field(default_factory=list)
    positions: list[list[int]] = field(default_factory=list)
    n_reroll_calls: int = 0
    reroll_width: int = 0
    agreement: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    says: str = ""

    @property
    def is_killed(self) -> bool:
        """G3's kill condition, evaluated either way it can fire.

        Reporting that it fired is a result rather than a failure: the kill condition is "kill if
        the induced function is constant", and a measured constant is the result.
        """
        return self.is_constant or self.is_degenerate

    def render(self) -> str:
        return "\n".join([self.says, *(f"    {n}" for n in self.notes)])


def implicit_prm(
    completions: Sequence[Sequence[int]],
    rewards: Sequence[float],
    *,
    min_members: int = 2,
    reroll: Callable[[tuple[int, ...], int], Sequence[float]] | None = None,
    reroll_width: int = 4,
    max_reroll_depth: int = 8,
    compare_to: Mapping[str, Sequence[float]] | None = None,
) -> ImplicitPRM | Refusal:
    """Extract the implicit PRM from one group's rollouts.

    Rung 0 needs the record and nothing else: build the prefix trie, set each node's value to the
    mean outcome of the rollouts through it, and difference along each edge. Rung 1 takes a
    ``reroll`` callable, ``(prefix_ids, m) -> m outcome rewards``, and replaces the value at each
    visited prefix with an independent Monte-Carlo estimate. Rung 1 is the only way to get a
    non-degenerate function out of a sampler whose rollouts do not branch, which on a real GRPO
    record is the normal case rather than the exception.

    ``compare_to`` maps a name to a per-position reference (a trained PRM's scores, a re-roll, a
    ground-truth label) aligned to `positions`, and each is reported as a Spearman correlation. That
    is G3's headline sentence, and it is left as an argument because this library ships no PRM and
    inventing one to correlate against would be inventing the answer.
    """
    from reward_lens.stats.effects import spearman_with_ci

    if len(completions) != len(rewards):
        return Refusal(
            instrument="ImplicitPRM",
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=f"{len(completions)} completions against {len(rewards)} rewards",
            remedy="pass one outcome reward per rollout, aligned to the completions.",
            statistics={"n_completions": len(completions), "n_rewards": len(rewards)},
        )
    if len(completions) < 2:
        return Refusal(
            instrument="ImplicitPRM",
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=f"{len(completions)} rollout(s); a Monte-Carlo value needs a group",
            remedy=(
                "pass the whole group. The implicit PRM's root value is the group mean, which is "
                "GRPO's own baseline, and a group of one has no baseline and no value function."
            ),
            statistics={"n_rollouts": len(completions)},
        )
    finite = [float(r) for r in rewards]
    if not all(math.isfinite(r) for r in finite):
        return Refusal(
            instrument="ImplicitPRM",
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail="at least one outcome reward is not finite",
            remedy=(
                "a rollout whose grader abstained has no reward and must be excluded from the "
                "group rather than counted as a zero; the record's `abstained` is that channel."
            ),
            statistics={"rewards": finite},
        )

    root = _Node(depth=0, members=list(range(len(completions))))
    nodes = [root]
    for k, seq in enumerate(completions):
        node = root
        for depth, token in enumerate(seq, start=1):
            child = node.children.get(int(token))
            if child is None:
                child = _Node(depth=depth)
                node.children[int(token)] = child
                nodes.append(child)
            child.members.append(k)
            node = child

    calls = 0
    for node in nodes:
        node.n_samples = len(node.members)
        node.value = float(np.mean([finite[i] for i in node.members]))
    if reroll is not None:
        # Keyed by the prefix itself rather than by its hash: two distinct prefixes that collide
        # would silently share one value, which is a wrong number rather than a slow one.
        prefixes: dict[tuple[int, ...], None] = {}
        for k, seq in enumerate(completions):
            for depth in range(0, min(len(seq), max_reroll_depth) + 1):
                prefixes.setdefault(tuple(int(t) for t in seq[:depth]), None)
        by_prefix: dict[tuple[int, ...], float] = {}
        for prefix in prefixes:
            sampled = [float(x) for x in reroll(prefix, reroll_width)]
            calls += 1
            if sampled:
                by_prefix[prefix] = float(np.mean(sampled))
        for k, seq in enumerate(completions):
            node = root
            for depth in range(0, min(len(seq), max_reroll_depth) + 1):
                key = tuple(int(t) for t in seq[:depth])
                if depth > 0:
                    node = node.children[int(seq[depth - 1])]
                if key in by_prefix:
                    node.value = by_prefix[key]
                    node.n_samples = reroll_width
                    node.from_reroll = True

    process: list[float] = []
    positions: list[list[int]] = []
    informative = 0
    divergence: list[int] = []
    for k, seq in enumerate(completions):
        node = root
        parent_value = root.value
        found = None
        for depth, token in enumerate(seq, start=1):
            node = node.children[int(token)]
            process.append(node.value - parent_value)
            positions.append([k, depth - 1])
            if node.n_samples >= min_members:
                informative += 1
            elif found is None:
                found = depth
            parent_value = node.value
        divergence.append(found if found is not None else len(seq))

    branching = 0
    for node in nodes:
        if len(node.members) >= min_members and len(node.children) >= 2:
            branching = max(branching, node.depth)
    zeros = sum(1 for p in process if p == 0.0)
    # "Constant" is a claim about the function, and testing it with `== 0.0` tests the arithmetic
    # instead: a value function assembled from means of means differences to 1e-17 rather than to
    # zero, so a re-roll that returns a perfectly linear value reads as varying. The tolerance is
    # relative to the scale of the increments, with an absolute floor for the all-zero case.
    scale = float(np.max(np.abs(process))) if process else 0.0
    spread = float(np.std(process)) if process else 0.0
    constant_tol = max(1e-12, 1e-9 * scale)
    payload = ImplicitPRM(
        rung=1 if reroll is not None else 0,
        n_rollouts=len(completions),
        n_nodes=len(nodes),
        n_nodes_estimated=sum(1 for n in nodes if n.n_samples >= min_members),
        max_depth=max((len(s) for s in completions), default=0),
        branching_depth=branching,
        mean_divergence_depth=float(np.mean(divergence)) if divergence else float("nan"),
        max_divergence_depth=int(max(divergence)) if divergence else 0,
        root_value=root.value,
        informative_fraction=float(informative / len(process)) if process else 0.0,
        is_constant=bool(not process or spread <= constant_tol),
        is_degenerate=bool(informative == 0),
        zero_fraction=float(zeros / len(process)) if process else 0.0,
        process_reward_sd=spread,
        process_reward=[float(x) for x in process],
        positions=positions,
        n_reroll_calls=calls,
        reroll_width=reroll_width if reroll is not None else 0,
    )
    for name, reference in (compare_to or {}).items():
        ref = np.asarray(list(reference), dtype=np.float64)
        if ref.size != len(process):
            payload.notes.append(
                f"reference {name!r} has {ref.size} entries against {len(process)} positions and "
                f"was not compared; align it to `positions`"
            )
            continue
        payload.agreement[name] = float(
            spearman_with_ci(np.asarray(process), ref, n_resamples=1000, seed=0).point
        )
    if payload.is_degenerate:
        payload.notes.append(
            f"no position past the root carries a value estimated from {min_members} or more "
            f"rollouts: they diverge after a mean of {payload.mean_divergence_depth:.1f} tokens, so "
            f"past that point every value is one rollout's own outcome, the whole outcome advantage "
            f"lands on the first divergent token and {payload.zero_fraction:.0%} of positions carry "
            f"exactly zero. G3's kill condition fires on this subject. The remedy is a re-roll: "
            f"pass `reroll` to estimate the value at prefixes the sampler never revisited, which is "
            f"rung 1 and is the only estimator that survives here."
        )
    if payload.is_constant:
        payload.notes.append(
            "the induced process reward has zero variance across every position, which is the "
            "literal kill condition of section 5.G: there is nothing in the function to compare "
            "against a trained PRM or a re-roll."
        )
    payload.says = (
        f"GRPO's induced per-step credit function over {payload.n_rollouts} rollouts: "
        f"{payload.informative_fraction:.0%} of trained positions carry a Monte-Carlo value from "
        f"{min_members} or more rollouts, rollouts branch to depth {payload.branching_depth}, and "
        f"the root value (the group mean, which is GRPO's baseline) is {payload.root_value:.4g}. "
        + (
            "The function is constant, so there is nothing in it."
            if payload.is_constant
            else (
                "It is degenerate: every increment is one rollout's own outcome, so this is the "
                "outcome advantage wearing a per-step name."
                if payload.is_degenerate
                else f"The process reward has sd {payload.process_reward_sd:.4g}."
            )
        )
        + (
            f" Estimated at rung 1 from {payload.n_reroll_calls} re-roll calls of width "
            f"{payload.reroll_width}."
            if payload.rung
            else ""
        )
    )
    return payload


# ---------------------------------------------------------------------------
# The envelopes
# ---------------------------------------------------------------------------


#: `NO_COMPACTION` is declared for G1 "for any importance-weighted form". It is required
#: unconditionally here rather than only for that form, and for a stronger reason: a compaction
#: removes tokens from the trajectory *after* they were generated, so the id sequence this
#: instrument tokenises is not the one the gradient was taken over, and every share is attributed
#: to the wrong span rather than being merely biased.
CREDIT_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.NO_COMPACTION}),
    measured_by={RegimeCondition.NO_COMPACTION: MEASURED_BY[RegimeCondition.NO_COMPACTION]},
    on_violation="refuse",
)

#: The catalogue leaves G2's envelope OPEN. `MASK_STABLE` is what it should be: the mask policy
#: decides what "per token" means, so a turn decomposition taken across a window in which the mask
#: changed is comparing two different quantities. That is this package's choice rather than the
#: catalogue's.
TURN_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.MASK_STABLE}),
    measured_by={RegimeCondition.MASK_STABLE: MEASURED_BY[RegimeCondition.MASK_STABLE]},
    on_violation="refuse",
)

#: The catalogue leaves G3's envelope OPEN too. `NEAR_POLICY` is the honest condition and it bites
#: only at rung 1: a re-roll estimates `E[R | prefix]` under the policy doing the rolling, and if
#: that is not the policy that wrote the record then the value function belongs to a different
#: object. Rung 0 reads recorded tokens and recorded rewards and needs nothing, which is why the
#: violation downgrades rather than refusing: the rung-0 reading survives outside the envelope.
PRM_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.NEAR_POLICY}),
    measured_by={RegimeCondition.NEAR_POLICY: MEASURED_BY[RegimeCondition.NEAR_POLICY]},
    on_violation="refuse",
)

#: G1's declared comparator: "uniform attribution over tokens". It is not one of M3's
#: six, so it is named alongside them rather than instead of them. Spelling it as a `baseline.*` id
#: keeps it in the same namespace even though the bank does not hold it, and `credit_increment`
#: constructs it explicitly.
UNIFORM_ATTRIBUTION: str = "baseline.uniform_attribution"

CREDIT_BASELINES: tuple[str, ...] = ALL_SIX + (UNIFORM_ATTRIBUTION,)


# ---------------------------------------------------------------------------
# M9 on the white-box reading
# ---------------------------------------------------------------------------


def credit_increment(
    report: CreditReport,
    *,
    texts: Sequence[str],
    proxy: Sequence[float],
    n_tokens: Sequence[int],
    group_ids: Sequence[int] | None = None,
    seed: int = 0,
    n_resamples: int = 2000,
) -> Any:
    """What the black-box bank already knew about where the credit was.

    **The framing is a deliberate deviation from M9's default shape, because that shape does not fit
    a definition.** M9 asks what an instrument adds to the ones already run, which presumes the
    instrument is a predictor scored against an external criterion. The credit measure is not a
    predictor: it is the exact disintegration of the step, and scoring it against itself would
    return a perfect number that means nothing. So the criterion here **is** the white-box
    measurement, and what is scored against it is every cheap method, the instrument's own
    record-only proxy included.

    That inverts nothing about the question being asked. "What did opening the network buy?" is
    answered directly: if the six dumb baselines recover which rollouts carried the gradient mass as
    well as the advantage-weighted surprisal does, then the backward passes bought the ground truth
    and not the localisation, and the instrument should say so. If they do not, the backward passes
    are the only route to the quantity, which is the strongest case a white-box instrument can make
    for itself.

    ``proxy`` is the own-detector: one record-only score per rollout. ``n_tokens`` builds the
    declared "uniform attribution over tokens" comparator. The label is whether a rollout's measured
    projected credit share is above the median of its own group, which keeps the comparison within
    the group the advantage was computed in rather than across steps with different scales.
    """
    shares = np.asarray(
        [s for s, k in zip(report.projected_share, report.kinds) if k == "rollout"],
        dtype=np.float64,
    )
    if shares.size != len(texts):
        return Refusal(
            instrument="CreditDisintegration",
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"the report holds {shares.size} rollout parts and {len(texts)} texts were "
                f"supplied for the baseline bank"
            ),
            remedy=(
                "run the increment against a rollout-wise disintegration (`by_rollout`) and pass "
                "one completion text per rollout. A turn-wise report has no per-rollout item to "
                "score the bank on."
            ),
            statistics={"n_parts": shares.size, "n_texts": len(texts)},
        )
    groups = (
        np.asarray(group_ids, dtype=np.int64)
        if group_ids is not None
        else np.zeros(shares.size, dtype=np.int64)
    )
    labels = np.zeros(shares.size, dtype=int)
    for g in np.unique(groups):
        sel = groups == g
        labels[sel] = (shares[sel] > float(np.median(shares[sel]))).astype(int)
    if np.unique(labels).size < 2:
        return Refusal(
            instrument="CreditDisintegration",
            reason=RefusalReason.BELOW_LOD,
            detail=(
                f"every one of the {shares.size} rollouts falls on the same side of its group "
                f"median of projected credit share, so there is no contrast to score against"
            ),
            remedy=(
                "widen the window: an increment is defined against a criterion with two classes in "
                "it, and a batch whose credit shares are tied gives none. Pool more steps, or read "
                "a step whose group separated."
            ),
            statistics={"n_rollouts": int(shares.size), "n_positive": int(labels.sum())},
        )

    task = DetectionTask(
        labels=labels,
        texts=tuple(texts),
        seed_labels=tuple(str(g) for g in groups.tolist()),
        name="credit.projected_share",
    )
    bank = run_bank(task)

    def _detector(name: str, scores: Any, note: str) -> Detector:
        values = np.asarray(scores, dtype=np.float64)
        threshold = 0.5 * (float(values[labels == 1].mean()) + float(values[labels == 0].mean()))
        return Detector.from_scores(name, values, labels, threshold=threshold, note=note)

    own = _detector(
        "credit.record_proxy",
        proxy,
        "advantage-weighted surprisal per rollout, computed from the record alone",
    )
    comparators = [
        _detector(
            UNIFORM_ATTRIBUTION,
            np.asarray(n_tokens, dtype=np.float64),
            "section 5.G's declared baseline: attribute credit uniformly over tokens",
        )
    ]
    for bid, reading in sorted(bank.readings.items()):
        if is_scored(reading) and float(np.std(reading.scores)) > 0.0:
            comparators.append(
                _detector(bid, np.asarray(reading.scores, dtype=np.float64), reading.detail)
            )
    increment = IncrementalValidityReading(
        own=own, baselines_run=comparators, n_resamples=n_resamples, seed=seed
    ).compute()
    return increment, bank


# ---------------------------------------------------------------------------
# The instruments
# ---------------------------------------------------------------------------


class CreditDisintegration(BaseObservable):
    """G1. Where this step's credit went, disintegrated, and whether the parts account for it.

    White-box: it takes ``m + 1`` backward passes with respect to the parameters, so an
    `IncrementalValidity` record is mandatory on the reading and this instrument supplies one
    through `Context.emit(incremental=...)`.

    What it cannot do, beyond the module docstring's three. The disintegration is exact and it is
    exact about *the gradient*, not about the update: `sum mu = dtheta/eta` holds under plain SGD
    and under nothing else, so on a run trained with Adam this instrument reports the credit and
    declines to call it the update. And the partition is supplied rather than discovered, so a
    report that puts 41% on "the final answer span" is reporting the span the caller drew: the
    boundary is part of the claim and it belongs in the subject description.
    """

    name = "CreditDisintegration"
    version = "1.0"
    capabilities = Capability.GRADIENTS
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "credit as a conserved measure"
    deviations = (
        "the catalogue gives G1 two quantities and this instrument declares `credit.measure`; "
        "`credit.conservation_error` travels on the payload as a first-class field rather than as "
        "a second instrument, because it is the audit of this reading and not a separate reading",
        "M9's default shape scores an instrument against an external criterion, and the criterion "
        "here is this instrument's own measurement, because the credit measure is a definition "
        "rather than a predictor. What the increment reports is what the black-box bank already "
        "knew about where the mass was; `credit_increment` carries the argument",
        "the identity to `dtheta/eta` is checked against `torch.optim.SGD` only. Under any "
        "optimizer that carries state it is false, and `step_conservation` refuses to claim it "
        "rather than widening a tolerance",
    )

    quantity = "credit.measure"
    requires: AccessMatrix = {
        Component.POLICY: Access.BACKWARD,
        Component.RECORD: Access.RECORD,
    }
    #: A generative policy. The other five substrates have no parameters to differentiate, and a
    #: `NEURAL_SCALAR` grader has parameters but no `log pi` over a trajectory to weight.
    substrates = frozenset({Substrate.NEURAL_GEN})
    phases = frozenset({Phase.IN_RUN, Phase.POST_RUN})
    envelope = CREDIT_ENVELOPE
    #: Two groups, two true relations, and both are declared here. Under
    #: `tokenization` the shares are **invariant**, and the reason is exact rather than approximate:
    #: log-probability is additive under a change of tokenisation that decodes to the same string,
    #: because `log pi(ab) = log pi(a) + log pi(b|a)`. So a share summed over a span defined on the
    #: *text* is unchanged when the tokeniser splits a token inside it. A per-token share is not,
    #: and that is why this instrument reports spans and turns and refuses to report per-token
    #: shares. Under `units` the assertion is a refusal rather than a numeric relation, which is
    #: what `credit.conservation_error` being dimensionless is about.
    invariance = "tokenization, units"
    invariance_relation = {"tokenization": INVARIANT, "units": INVARIANT}
    baselines = CREDIT_BASELINES
    rung = 1

    def __init__(
        self,
        *,
        batch: TrajectoryBatch,
        partition: Partition | None = None,
        dtype: str = "float32",
        conservation_tol: float = 1e-5,
        texts: Sequence[str] = (),
        group_ids: Sequence[int] | None = None,
        proxy: Sequence[float] | None = None,
        lr: float | None = None,
        seed: int = 0,
        n_resamples: int = 2000,
    ) -> None:
        self.batch = batch
        self.partition = partition
        self.dtype = dtype
        self.conservation_tol = float(conservation_tol)
        self.texts = tuple(texts)
        self.group_ids = None if group_ids is None else list(group_ids)
        self.proxy = None if proxy is None else list(proxy)
        self.lr = lr
        self.seed = int(seed)
        self.n_resamples = int(n_resamples)

    def measure(self, ctx: Context) -> Any:
        partition = self.partition or by_rollout(self.batch)
        if self.lr is None:
            report = disintegrate(
                ctx.signal,
                self.batch,
                partition,
                dtype=self.dtype,
                conservation_tol=self.conservation_tol,
            )
        else:
            report = step_conservation(
                ctx.signal, self.batch, partition, lr=self.lr, dtype=self.dtype
            )
        if isinstance(report, Refusal):
            return report

        record = None
        baselines: dict[str, float] = {}
        if self.texts and self.proxy is not None:
            outcome = credit_increment(
                report,
                texts=self.texts,
                proxy=self.proxy,
                n_tokens=[p for p, k in zip(report.n_positions, report.kinds) if k == "rollout"],
                group_ids=self.group_ids,
                seed=self.seed,
                n_resamples=self.n_resamples,
            )
            if isinstance(outcome, Refusal):
                return outcome
            increment, bank = outcome
            if isinstance(increment, Refusal):
                return increment
            record = increment.record
            baselines = dict(bank.as_mapping())
            report.notes.append(increment.says())

        return ctx.emit(
            report,
            uncertainty=Uncertainty(
                ci_low=report.conservation_error,
                ci_high=report.conservation_error,
                ci_level=1.0,
                n=report.n_trained_positions,
                method=(
                    f"the disintegration is exact arithmetic, so its uncertainty is the numerical "
                    f"closure itself: |sum_S g_S - g_full| / |g_full| = "
                    f"{report.conservation_error:.3e} in {report.dtype_used} over "
                    f"{report.n_parameters:,} parameters"
                ),
            ),
            baselines=baselines,
            incremental=record,
        )


class TurnCredit(BaseObservable):
    """G2. Where the log-probability mass that received advantage sits, by turn and by tool call.

    Rung 0, and the whole of what a record supports on its own. Not white-box: it opens nothing and
    differentiates nothing, so no `IncrementalValidity` goes on its reading, and that is the
    correct asymmetry rather than an omission. The reading it produces is the cheap estimate of what
    `CreditDisintegration` measures exactly.

    What it cannot do. Surprisal is not gradient norm. This instrument reports where the
    log-probability mass is, the gradient is what actually moved, and the two agree only to the
    extent that a token the policy was uncertain about also sits in a steep direction of parameter
    space. Nothing here establishes that they do; `CreditDisintegration` measures the gap, and until
    someone runs it on a subject with real turn structure the gap is unmeasured rather than small.
    """

    name = "TurnCredit"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "the agentic form"
    deviations = (
        "the catalogue gives G2 two quantities and this instrument declares `credit.by_turn`; "
        "`credit.by_tool_call` travels on the same payload, because a tool-call share is the same "
        "disintegration read through a different index and computing it twice would let the two "
        "disagree",
        "the catalogue leaves G2's envelope OPEN and this instrument declares `MASK_STABLE`. The "
        "argument is that the mask policy decides what 'per token' means, so a turn "
        "decomposition pooled across a window in which it changed compares two quantities",
    )

    quantity = "credit.by_turn"
    requires: AccessMatrix = {Component.RECORD: Access.RECORD}
    substrates = frozenset(
        {
            Substrate.NEURAL_SCALAR,
            Substrate.NEURAL_GEN,
            Substrate.PROGRAM,
            Substrate.PROCEDURAL,
            Substrate.HUMAN,
            Substrate.COMPOSITE,
        }
    )
    phases = frozenset({Phase.IN_RUN, Phase.POST_RUN})
    envelope = TURN_ENVELOPE
    #: `none`, from the registry, which resolves to the trivial group. A share over turns is a
    #: dimensionless normalised number and no invariance group acts on the turn index. The
    #: retokenisation argument that makes `credit.measure` invariant applies here too and the
    #: registry does not say so, so this instrument follows the registry rather than declaring the
    #: stronger relation on its own.
    invariance = "none"
    invariance_relation = INVARIANT
    baselines = CREDIT_BASELINES
    rung = 0

    def __init__(
        self,
        *,
        trajectories: Sequence[Any],
        trained_roles: Sequence[str] = TRAINED_ROLES,
        n_head: int = 3,
        n_tail: int = 1,
    ) -> None:
        self.trajectories = list(trajectories)
        self.trained_roles = tuple(trained_roles)
        self.n_head = int(n_head)
        self.n_tail = int(n_tail)

    def measure(self, ctx: Context) -> Any:
        payload = turn_mass(
            self.trajectories,
            trained_roles=self.trained_roles,
            n_head=self.n_head,
            n_tail=self.n_tail,
        )
        if isinstance(payload, Refusal):
            return payload
        return ctx.emit(
            payload,
            uncertainty=Uncertainty(
                ci_low=float("nan"),
                ci_high=float("nan"),
                ci_level=0.0,
                n=payload.n_tokens_scored,
                method=(
                    "shares of a recorded total, so there is no sampling uncertainty in the "
                    "arithmetic. The uncertainty that matters is the proxy's disagreement with the "
                    "gradient, which CreditDisintegration measures and this reading does not"
                ),
            ),
        )


class ImplicitPRMReading(BaseObservable):
    """G3. GRPO's induced per-step credit function, extracted rather than inferred.

    Not white-box at rung 0: it reads recorded tokens and recorded outcomes and builds a prefix
    trie. Rung 1 needs a sampler, which is `Access.REPLICATE` on the policy rather than any
    capability on the network.

    What it cannot do, and it is the thing to read first. The extraction is only as informative as
    the prefix sharing in the record, and on a GRPO run with temperature-1 sampling there is none:
    the rollouts of a group diverge at their first token, every value past the root is one
    rollout's own outcome, and the induced function is the outcome advantage placed on token one.
    The instrument reports that as `is_degenerate` with the numbers behind it, which is G3's kill
    condition firing rather than the instrument failing.
    """

    name = "ImplicitPRM"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "the implicit process reward model"
    deviations = (
        "the catalogue leaves G3's envelope OPEN and this instrument declares `NEAR_POLICY`, which "
        "bites at rung 1 only: a re-roll estimates the value under the policy doing the rolling, "
        "so a stale checkpoint estimates a different object's value function",
        "section 5.G's headline compares the induced function against a trained PRM and against "
        "counterfactual re-rolls. This library ships no PRM, so `compare_to` takes references from "
        "the caller and the instrument reports a Spearman correlation against whatever is supplied "
        "rather than inventing a comparator",
    )

    quantity = "credit.implicit_prm"
    requires: AccessMatrix = {Component.RECORD: Access.RECORD}
    substrates = frozenset({Substrate.NEURAL_GEN})
    phases = frozenset({Phase.IN_RUN, Phase.POST_RUN})
    envelope = PRM_ENVELOPE
    invariance = "none"
    invariance_relation = INVARIANT
    baselines = ("baseline.outcome_advantage",) + ALL_SIX
    rung = 0

    def __init__(
        self,
        *,
        completions: Sequence[Sequence[int]],
        rewards: Sequence[float],
        min_members: int = 2,
        reroll: Callable[[tuple[int, ...], int], Sequence[float]] | None = None,
        reroll_width: int = 4,
        max_reroll_depth: int = 8,
        compare_to: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self.completions = [list(c) for c in completions]
        self.rewards = list(rewards)
        self.min_members = int(min_members)
        self.reroll = reroll
        self.reroll_width = int(reroll_width)
        self.max_reroll_depth = int(max_reroll_depth)
        self.compare_to = dict(compare_to or {})

    def measure(self, ctx: Context) -> Any:
        payload = implicit_prm(
            self.completions,
            self.rewards,
            min_members=self.min_members,
            reroll=self.reroll,
            reroll_width=self.reroll_width,
            max_reroll_depth=self.max_reroll_depth,
            compare_to=self.compare_to,
        )
        if isinstance(payload, Refusal):
            return payload
        return ctx.emit(
            payload,
            uncertainty=Uncertainty(
                ci_low=float("nan"),
                ci_high=float("nan"),
                ci_level=0.0,
                n=len(payload.process_reward),
                method=(
                    f"Monte-Carlo values from {payload.n_rollouts} rollouts at rung "
                    f"{payload.rung}; the interval on a node's value is the interval on a mean of "
                    f"as few as one sample, which is why `informative_fraction` is reported "
                    f"instead of a number that would look like precision"
                ),
            ),
        )


# ---------------------------------------------------------------------------
# Estimator registration
# ---------------------------------------------------------------------------


def register_estimators() -> None:
    """Register series G's estimators, idempotently.

    All five quantities already have rows in `spec/QUANTITIES.yaml`, so nothing here proposes one.
    What the rows do not carry is an `EstimatorEntry`, and lint rule two reads the estimator
    registry rather than the catalogue: a quantity with a catalogue row and no estimator reads as an
    open research target to the docs build, and three of these five are built and running.

    `credit.localiser_quality` (G4) and `credit.successor_representation` (G5) get nothing, on
    purpose. G4 needs a re-roll capability and a labelled localiser and is not built here;
    G5 is `OPEN` in the catalogue because the state abstraction for a language trajectory is
    undefined and defining it is the research content, not the implementation.

    Idempotent and re-callable for the reason recorded on `policy/quantities.py`: the estimator
    registry is a process-global that several test modules snapshot and pop, and a module-level
    registration landing inside one of those windows does not come back on its own.
    """
    from reward_lens.core.quantity import (
        ESTIMATORS,
        BiasStatement,
        CostModel,
        EstimatorEntry,
        register_estimator,
    )

    if any(e.quantity.startswith("credit.") for e in ESTIMATORS.values()):
        return

    backward: dict[Component, Access] = {
        Component.POLICY: Access.BACKWARD,
        Component.RECORD: Access.RECORD,
    }
    record_only: dict[Component, Access] = {Component.RECORD: Access.RECORD}

    register_estimator(
        EstimatorEntry(
            quantity="credit.measure",
            impl="credit.disintegration.masked_backward",
            requires=backward,
            envelope=CREDIT_ENVELOPE,
            rung=1,
            bias=BiasStatement(
                direction="approximately_unbiased",
                why=(
                    "the disintegration is exact by linearity of the gradient in the masked "
                    "objective, so the only error is floating point. Measured on a real model, "
                    "1.13e-07 in float32 and 2.70e-16 in float64 against 1.56e-03 in the "
                    "model's native bfloat16, which is why the measurement casts."
                ),
            ),
            cost=CostModel(
                note=(
                    "one retained forward pass, then one backward per non-empty part plus one for "
                    "the full objective. Peak memory is three parameter-sized float64 buffers; no "
                    "per-token per-parameter tensor is ever formed."
                )
            ),
        )
    )
    register_estimator(
        EstimatorEntry(
            quantity="credit.conservation_error",
            impl="credit.disintegration.closure",
            requires=backward,
            envelope=CREDIT_ENVELOPE,
            rung=1,
            bias=BiasStatement(
                direction="upward",
                why=(
                    "the full gradient is taken by its own independent backward pass rather than "
                    "as the sum of the parts, so the residual carries the rounding of both sides "
                    "and is an upper bound on the disintegration's own error. Deriving it from the "
                    "sum would make it identically zero and the audit vacuous."
                ),
            ),
            cost=CostModel(note="free given the disintegration; it is the residual of that sum"),
        )
    )
    # One implementation, two registry rows, and two impl ids because the registry keys on impl and
    # is right to: a name that resolves to two entries depending on import order is not a name. The
    # two shares come off the same pass over the same log-probabilities, which is the point of
    # computing them together rather than twice.
    for quantity, impl in (
        ("credit.by_turn", "credit.turn_mass.by_turn"),
        ("credit.by_tool_call", "credit.turn_mass.by_tool_call"),
    ):
        register_estimator(
            EstimatorEntry(
                quantity=quantity,
                impl=impl,
                requires=record_only,
                envelope=TURN_ENVELOPE,
                rung=0,
                bias=BiasStatement(
                    direction="unknown",
                    why=(
                        "surprisal is a proxy for gradient norm and the two are not identical: a "
                        "token can be surprising and sit in a flat direction of parameter space, "
                        "or confident and sit in a steep one. The direction of the resulting error "
                        "is not signable a priori and is measured by running "
                        "`credit.measure` on the same turns."
                    ),
                ),
                cost=CostModel(note="a pass over the recorded per-token log-probabilities"),
            )
        )
    register_estimator(
        EstimatorEntry(
            quantity="credit.implicit_prm",
            impl="credit.implicit_prm.prefix_trie",
            requires=record_only,
            envelope=PRM_ENVELOPE,
            rung=0,
            bias=BiasStatement(
                direction="unknown",
                why=(
                    "each node's value is a mean over the rollouts through it, so a node with one "
                    "member is that rollout's own outcome and carries the full sampling error of a "
                    "single draw. The bias that produces is toward a function that looks sharper "
                    "than it is, and `informative_fraction` reports how much of the trajectory is "
                    "in that state rather than correcting for it."
                ),
            ),
            cost=CostModel(note="a trie over the recorded completions; no model call at rung 0"),
        )
    )


register_estimators()


__all__ = [
    "CREDIT_BASELINES",
    "CREDIT_ENVELOPE",
    "EXACT_UPDATE_OPTIMIZERS",
    "PRM_ENVELOPE",
    "SKYRL_MODULE",
    "TRAINED_ROLES",
    "TURN_ENVELOPE",
    "UNIFORM_ATTRIBUTION",
    "CreditDisintegration",
    "CreditReport",
    "ImplicitPRM",
    "ImplicitPRMReading",
    "Part",
    "Partition",
    "PartitionError",
    "Segment",
    "SkyRLDump",
    "TrajectoryBatch",
    "TurnCredit",
    "TurnMass",
    "batch_from_skyrl",
    "batch_from_trajectories",
    "by_rollout",
    "by_segment",
    "credit_increment",
    "disintegrate",
    "implicit_prm",
    "merge_empty",
    "read_skyrl_dump",
    "register_estimators",
    "step_conservation",
    "turn_mass",
    "turn_segments",
]

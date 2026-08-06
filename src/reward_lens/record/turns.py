"""The turn: one model action and the environment response to it.

The level below the trajectory and above the token. Agentic RL is hierarchical and reward attaches
at every level, so the record has to be too: a process reward attaches here, a tool result arrives
here, and the loss mask that decides what "per token" means is defined here.

**Two fields that look redundant and are not.** `logprobs_sampling` and `logprobs_train` are the
same tokens scored by two different engines, and they differ. Instrument E6
(`policy.train_infer_logprob_mismatch`) measures that difference and it is the record-level
expression of the numerics floor: an importance ratio built from two engines that
disagree by 0.4 nats is measuring the engines, not the policy. Collapsing the two fields into one
destroys the only measurement that can tell you which one you have. So they stay separate, and a
converter that has only one of them fills one and leaves the other `None`, which is honest and
makes E6 refuse rather than report a zero mismatch.

**What `loss_mask` decides.** Environment tokens are masked everywhere, and the mask policy
changes what a per-token quantity means, which is why `MASK_STABLE` is one of the twelve regime
conditions. `mask_policy_signature` below is the statistic that condition is measured from; the
threshold is not set here, because a threshold in an envelope is not a decision this module gets
to make.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping

from reward_lens.core.types import Span
from reward_lens.record.tensors import TensorRef, ref_from_canonical

#: The canonical schema prints four roles. ``system`` is the fifth and it is added deliberately:
#: every framework's message list can open with a system message, and the two ways to record one
#: without this member are to drop it (which corrupts the prompt reconstruction and every token
#: offset after it) or to relabel it ``user`` (which corrupts the loss-mask attribution, since
#: system tokens and user tokens are masked by different policies in several trainers).
TurnRole = Literal["assistant", "tool", "environment", "user", "system"]


@dataclass(frozen=True)
class ToolCall:
    """A structured tool invocation and what came back.

    ``arguments`` is the raw string the model emitted rather than a parsed object, because a
    malformed tool call is a real and common event and parsing it away at record time destroys the
    evidence that it was malformed. ``ok`` is three-valued: True, False, or None for a call whose
    outcome the recorder could not determine.
    """

    name: str
    arguments: str = ""
    call_id: str | None = None
    result: str | None = None
    ok: bool | None = None
    latency_ms: float | None = None
    error: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "call_id": self.call_id,
            "result": self.result,
            "ok": self.ok,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "ToolCall":
        return cls(
            name=obj["name"],
            arguments=obj.get("arguments", ""),
            call_id=obj.get("call_id"),
            result=obj.get("result"),
            ok=obj.get("ok"),
            latency_ms=obj.get("latency_ms"),
            error=obj.get("error"),
            extra=dict(obj.get("extra", {})),
        )


@dataclass(frozen=True)
class Turn:
    """One model action plus the environment response.

    Five fields beyond the printed schema, each forced by something a framework already records:

    ``tensors`` — a per-turn `TensorRef` map. Routed-expert traces are per-turn and the printed
    schema has nowhere to put them, which matters because routing traces are the one capture the
    cost arithmetic actually permits (about 455 times cheaper than residuals). Typing them as
    `TensorRef` rather than as arrays is what keeps the honest default: a turn whose routing was
    not captured carries an `AbsentRef` and nothing downstream can mistake it for zeros.

    ``truncated`` — whether generation stopped at the length cap rather than at a stop token. The
    last token of a truncated turn has no successor, so its logprob is not a completion logprob,
    and a truncated turn's `step_score` is a score on an unfinished action. Both `verifiers` and
    TRL record this and dropping it loses a fact that changes what the numbers mean.

    ``overlong_prompt`` — `verifiers` records it separately from truncation because a prompt that
    did not fit is a different event from a completion that ran out.

    ``step_advantage`` — the estimator's per-step advantage, which is not `step_score`. Process
    reward and per-step advantage are different numbers and `verifiers` writes both onto the same
    step (`rubric.py:408-412`), so a schema with one field for them silently merges two quantities.

    ``extra`` — the converter's escape hatch, for fields a framework declares as
    ``NotRequired[Any]`` and does not type. Anything landing here is untyped by construction and no
    instrument may read it without saying so.
    """

    index: int
    role: TurnRole
    text: str = ""
    token_ids: tuple[int, ...] | None = None
    logprobs_sampling: tuple[float, ...] | None = None
    logprobs_train: tuple[float, ...] | None = None
    loss_mask: tuple[bool, ...] | None = None
    tool_call: ToolCall | None = None
    spans: tuple[Span, ...] = ()
    step_score: float | None = None
    step_advantage: float | None = None
    truncated: bool | None = None
    overlong_prompt: bool | None = None
    tensors: Mapping[str, TensorRef] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"turn index cannot be negative; got {self.index}")
        n = self.n_tokens
        if n is None:
            return
        for name in ("logprobs_sampling", "logprobs_train", "loss_mask"):
            value = getattr(self, name)
            if value is not None and len(value) != n:
                raise ValueError(
                    f"turn {self.index}: {name} has {len(value)} entries against {n} token ids. "
                    f"A per-token array of the wrong length is a unit mismatch waiting to be "
                    f"averaged; supply None if it was not recorded."
                )

    @property
    def n_tokens(self) -> int | None:
        return None if self.token_ids is None else len(self.token_ids)

    @property
    def n_unmasked(self) -> int | None:
        """Tokens the loss actually sees. None when no mask was recorded, which is not zero."""
        return None if self.loss_mask is None else int(sum(1 for b in self.loss_mask if b))

    @property
    def has_both_logprob_streams(self) -> bool:
        """Whether E6 can be computed on this turn at all."""
        return self.logprobs_sampling is not None and self.logprobs_train is not None

    def logprob_gap(self) -> tuple[float, ...] | None:
        """``logprobs_train - logprobs_sampling`` per token, or None if either stream is missing.

        Returning None rather than zeros is the whole point. A missing stream means the mismatch
        was not measured; zeros would mean it was measured and found to be nothing, and those are
        opposite claims.
        """
        if not self.has_both_logprob_streams:
            return None
        assert self.logprobs_train is not None and self.logprobs_sampling is not None
        return tuple(t - s for t, s in zip(self.logprobs_train, self.logprobs_sampling))

    def __canonical__(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "role": self.role,
            "text": self.text,
            "token_ids": None if self.token_ids is None else list(self.token_ids),
            "logprobs_sampling": (
                None if self.logprobs_sampling is None else list(self.logprobs_sampling)
            ),
            "logprobs_train": None if self.logprobs_train is None else list(self.logprobs_train),
            "loss_mask": None if self.loss_mask is None else [bool(b) for b in self.loss_mask],
            "tool_call": None if self.tool_call is None else self.tool_call.__canonical__(),
            "spans": [s.__canonical__() for s in self.spans],
            "step_score": self.step_score,
            "step_advantage": self.step_advantage,
            "truncated": self.truncated,
            "overlong_prompt": self.overlong_prompt,
            "tensors": {k: v.__canonical__() for k, v in self.tensors.items()},
            "extra": dict(self.extra),
        }

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "Turn":
        ids = obj.get("token_ids")
        lps = obj.get("logprobs_sampling")
        lpt = obj.get("logprobs_train")
        mask = obj.get("loss_mask")
        tc = obj.get("tool_call")
        return cls(
            index=obj["index"],
            role=obj["role"],
            text=obj.get("text", ""),
            token_ids=None if ids is None else tuple(int(i) for i in ids),
            logprobs_sampling=None if lps is None else tuple(float(x) for x in lps),
            logprobs_train=None if lpt is None else tuple(float(x) for x in lpt),
            loss_mask=None if mask is None else tuple(bool(b) for b in mask),
            tool_call=None if tc is None else ToolCall.from_canonical(tc),
            spans=tuple(
                Span(
                    start=s["start"],
                    end=s["end"],
                    kind=s.get("kind", "text"),
                    meta=s.get("meta", {}),
                )
                for s in obj.get("spans", ())
            ),
            step_score=obj.get("step_score"),
            step_advantage=obj.get("step_advantage"),
            truncated=obj.get("truncated"),
            overlong_prompt=obj.get("overlong_prompt"),
            tensors={k: ref_from_canonical(v) for k, v in obj.get("tensors", {}).items()},
            extra=dict(obj.get("extra", {})),
        )


# ---------------------------------------------------------------------------
# Statistics the regime reading is built from
# ---------------------------------------------------------------------------


def mask_policy_signature(turns: Iterable[Turn]) -> str:
    """A signature of which roles are masked and which are not, over a set of turns.

    `MASK_STABLE` asks whether the loss-mask policy is unchanged across a window. The policy is
    not recorded as a string by any framework, so it is inferred from behaviour: for each role,
    whether its tokens are fully masked, fully unmasked, mixed, or unrecorded. Two windows with
    the same signature applied the same policy; two with different signatures did not.

    This returns the statistic. It does not decide whether the condition holds, because the
    threshold belongs to the envelope and the reading is made there.
    """
    per_role: dict[str, set[str]] = {}
    for turn in turns:
        if turn.loss_mask is None:
            state = "unrecorded"
        elif all(turn.loss_mask):
            state = "all"
        elif not any(turn.loss_mask):
            state = "none"
        else:
            state = "mixed"
        per_role.setdefault(turn.role, set()).add(state)
    return ";".join(
        f"{role}={'|'.join(sorted(states))}" for role, states in sorted(per_role.items())
    )


def logprob_mismatch(turns: Iterable[Turn]) -> tuple[float, int]:
    """Mean absolute per-token logprob gap and the token count it was measured over.

    The record-level input to E6. Turns missing either stream contribute nothing and are not
    counted, so a return of ``(0.0, 0)`` means "nothing was comparable" and is distinguishable
    from ``(0.0, 4096)``, which would mean the two engines agreed exactly on four thousand tokens.
    """
    total = 0.0
    n = 0
    for turn in turns:
        gap = turn.logprob_gap()
        if gap is None:
            continue
        total += sum(abs(g) for g in gap)
        n += len(gap)
    return (total / n if n else 0.0, n)


def token_count(turns: Iterable[Turn]) -> int:
    """Total recorded token ids across turns. Turns with no token ids contribute zero."""
    return sum(t.n_tokens or 0 for t in turns)


def renumber(turns: Sequence[Turn]) -> tuple[Turn, ...]:
    """Reindex turns to ``0..n-1`` in the order given.

    Converters assemble turns from framework structures whose own indices restart per step or per
    message list, and the tiling invariant on `SegmentProvenance` is stated over contiguous turn
    positions. This is the one place that renumbering happens.
    """
    from dataclasses import replace as _replace

    return tuple(_replace(t, index=i) for i, t in enumerate(turns))


__all__ = [
    "ToolCall",
    "Turn",
    "TurnRole",
    "logprob_mismatch",
    "mask_policy_signature",
    "renumber",
    "token_count",
]

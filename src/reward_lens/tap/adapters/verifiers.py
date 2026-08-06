"""The ``verifiers`` converter: a ``TrajectoryStep`` stream becomes a `Trajectory`.

Prime Intellect's ``verifiers`` is the only place in the open ecosystem where token ids, per-token
logprobs, MoE routing, reward and advantage already sit in one serialisable object, so the
conversion to this library's record is close to a field mapping rather than a reconstruction from
three sources. That is why this adapter is short and why it is worth having.

Verified against ``verifiers`` at commit ``edafab857aabf830237068193150694cfae50c3b`` (2026-07-30).
Both structures were re-located by content rather than by line number, because line numbers in that
codebase had already drifted once, and ``TrajectoryStep`` is not the v1 dataclass it is sometimes
described as: it is a ``TypedDict`` in ``verifiers/types.py``, and ``verifiers/v1/`` defines no
``TrajectoryStep`` at all.

Nothing here imports ``verifiers``. Every input is read through `_get`, which accepts a mapping or
an object with attributes, so a rollout that arrived as JSON from ``vf-eval`` and a live ``State``
in the same process both convert without the framework being installed. That matters more than it
sounds: the record is the interface, and an analyst holding a JSONL file should not need the
producer's dependency tree.

----

**Two facts about ``verifiers`` this converter carries into the record, because both change what a
downstream number means.**

``score_group`` (``rubrics/rubric.py:406-409``) writes
``state["advantage"] = aggregated_rewards[i] - avg_reward``. That is mean centring with **no
standard-deviation division**, no epsilon and no clip. The amplification mechanism the E series
measures is a statement about the z-score, so on this framework it is not weakened, it is absent,
and `estimator_spec` says so with ``std_normalised=False`` and ``std_epsilon=None`` rather than
implying a denominator that is not there. ``measure/estimator/spec.py`` already carries the same
row under ``verifiers/score_group``, and the ``family`` written here is the string that resolves to
it.

``_call_individual_reward_func`` (``rubrics/rubric.py:204-217``) catches any exception from a reward
function, logs it at ERROR with no counter, and sets ``ans = 0.0``; the group form does the same
with ``[0.0] * len(states)`` at 249-262. So a ``0.0`` in ``state["metrics"]`` is either a genuine
zero or a crashed grader, and **the record cannot tell them apart**. This converter therefore does
two different things depending on what it is given:

- Given tap records (``calls=``, from `reward_lens.tap.instrument_grader`), a leaf whose call
  raised gets ``abstained=True`` with the substituted ``0.0`` still on it, which is exactly
  `record.scores.Leaf.silent_zero` and gives instrument B4 an exact numerator.
- Given no tap records, no leaf is marked abstained, because marking every zero would invent the
  defect rather than measure it. Instead the trajectory carries
  ``features["verifiers_unresolved_zeros"]``: the count of leaves whose value is exactly zero,
  which is the **upper bound** on the silent-zero count and the only honest number available from
  the record alone.

----

**The turn layout, and why it is two turns per step rather than one per message.**

A ``TrajectoryStep``'s ``tokens`` carry one ``prompt_ids`` array for the whole prompt and one
``completion_ids`` array for the whole completion. There is no per-message token boundary in the
required fields, so splitting a multi-message prompt across several turns would mean inventing an
alignment. Each step therefore becomes at most two turns: a **context turn** holding the messages
that are new at this step and the prompt tokens that are new with them, and an **assistant turn**
holding the completion, its logprobs, its mask, its routing trace and its per-step reward and
advantage.

For step ``i > 0`` the new prompt tokens are ``prompt_ids[carry:]`` where ``carry`` is the previous
step's prompt plus completion length, and that is used **only when
``prompt_ids[:carry]`` actually equals the previous prompt ids followed by the previous completion
ids**. ``MultiTurnEnv.get_prompt_messages`` builds each step's prompt as
``prev_prompt + prev_completion + env_response``, so the relation holds on the linear path; it is
documented as overridable and chat templates re-render, so it is checked rather than assumed. When
the check fails the context turn carries ``token_ids=None`` and the step is counted in
`ConversionReport.non_prefix_steps`. A ``None`` there means "we could not say which tokens were
new", which is not the same claim as "there were none".

When the token prefix shrinks rather than merely failing to match, the prefix was rewritten and
that is a `CompactionEvent`, priced in tokens from the arrays themselves. After a rewrite the
importance ratio is undefined rather than stale, which is why it is a record and not a note.

----

**What this converter cannot do, said here rather than on a caveats page.**

It cannot recover staleness. ``TrajectoryStep`` records ``response.model``, which is a model name,
not a checkpoint, and nothing anywhere in the object says how many optimizer steps behind the
current policy the generating weights were. ``staleness_steps`` is therefore whatever the caller
declares, it defaults to zero, and the reason it is zero travels in
``SegmentProvenance.sampling.extra`` so nobody can read it as measured. `NEAR_POLICY` is left
undeclared for the same reason, with the reason in the regime notes.

It cannot separate a per-step process reward from a back-filled trajectory reward. ``score_group``
copies the rollout's reward and advantage down onto every step whose own value is ``None``
(``rubric.py:410-414``), so after scoring a uniform column is the expected state and a genuine
per-step reward is only visible when it is *not* uniform. Both go on the turns as ``step_score``
and ``step_advantage``, and ``features["verifiers_step_score_uniform"]`` says which case you are
looking at.

It cannot tell you the serving engine. ``Engine`` defaults to ``unknown`` and the caller supplies
it. Eager and compiled vLLM have been measured disagreeing with each other about as much as either
disagrees with HuggingFace, so an unrecorded engine is a real gap and recording ``vllm`` without a
revision would not close it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

if TYPE_CHECKING:  # the framework is never imported, and the record is imported lazily
    from reward_lens.record.provenance import SegmentProvenance
    from reward_lens.record.schema import EstimatorSpec, Group, Run, Step, Trajectory
    from reward_lens.record.tensors import Engine, TensorRef, TensorStore
    from reward_lens.record.turns import Turn
    from reward_lens.tap.contract import GraderCall

#: The commit every claim in this module was verified against.
VERIFIERS_COMMIT = "edafab857aabf830237068193150694cfae50c3b"

#: Where the framework substitutes a zero for a crashed reward function. Two bare
#: ``except Exception`` blocks in the individual path, two more in the group path.
SILENT_ZERO_SITE = "verifiers/rubrics/rubric.py:204-217 (group form at 249-262)"

#: Where the advantage is computed. Mean centring, no standard-deviation division.
SCORE_GROUP_SITE = "verifiers/rubrics/rubric.py:406-409"

#: Where the back-fill happens: a step whose own reward or advantage is ``None`` inherits the
#: rollout's, so a uniform column after scoring is expected rather than suspicious.
BACKFILL_SITE = "verifiers/rubrics/rubric.py:410-414"

#: ``TrajectoryStep``'s nine fields, in declaration order (``verifiers/types.py``, ``class
#: TrajectoryStep(TypedDict)``). The acceptance test walks this tuple.
TRAJECTORY_STEP_FIELDS: tuple[str, ...] = (
    "prompt",
    "completion",
    "response",
    "tokens",
    "reward",
    "advantage",
    "is_truncated",
    "trajectory_id",
    "extras",
)

#: ``TrajectoryStepTokens``' ten fields, in declaration order. The last two are ``NotRequired[Any]``
#: and are absent from a text-only rollout, which is why presence is tested rather than assumed.
TOKEN_FIELDS: tuple[str, ...] = (
    "prompt_ids",
    "prompt_mask",
    "completion_ids",
    "completion_mask",
    "completion_logprobs",
    "overlong_prompt",
    "is_truncated",
    "routed_experts",
    "multi_modal_data",
    "prompt_attribution",
)

#: Where each of ``TrajectoryStep``'s nine fields lands. Read as ``field -> the record location``.
#: This is half of the converter's contract and the acceptance test asserts every row of it against
#: a real conversion; a field that moves without this table moving is the failure this adapter
#: exists to prevent.
STEP_FIELD_MAP: Mapping[str, str] = {
    "prompt": "Turn(context).text and Turn(context).extra['messages'], as the delta against the "
    "previous step's prompt + completion",
    "completion": "Turn(assistant).text, plus extra['messages'] and extra['tool_calls']",
    "response": "Turn(assistant).extra['response'] (id, created, model, usage, finish_reason) and "
    "SegmentProvenance.policy_version, which is response.model",
    "tokens": "spread across both turns; see TOKEN_FIELD_MAP",
    "reward": "Turn(assistant).step_score, and Trajectory.features['verifiers_realised_reward'] "
    "when it is uniform across the stream",
    "advantage": "Turn(assistant).step_advantage, and Trajectory.advantage when it is uniform "
    "across the stream",
    "is_truncated": "Turn(assistant).truncated. This is the step-level flag, which verifiers "
    "builds as the union of the engine's finish_reason with the tokens-level clip.",
    "trajectory_id": "Trajectory.id, verbatim, so the record joins back to the framework artifact",
    "extras": "Turn(assistant).extra['step_extras']",
}

#: Where each of ``TrajectoryStepTokens``' ten fields lands. The other half of the contract.
TOKEN_FIELD_MAP: Mapping[str, str] = {
    "prompt_ids": "Turn(context).token_ids",
    "prompt_mask": "Turn(context).loss_mask, as bools; verifiers writes 0 for a prompt token and "
    "1 for a trainable one, which is the same orientation Turn.loss_mask uses",
    "completion_ids": "Turn(assistant).token_ids",
    "completion_mask": "Turn(assistant).loss_mask",
    "completion_logprobs": "Turn(assistant).logprobs_sampling; logprobs_train stays None because "
    "verifiers has only the inference-side stream, which makes E6 refuse rather than report zero",
    "overlong_prompt": "Turn(context).overlong_prompt",
    "is_truncated": "Turn(assistant).extra['tokens_is_truncated']. This is the max_seq_len clip "
    "specifically, kept apart from the step-level flag on Turn.truncated.",
    "routed_experts": "Turn(assistant).tensors['routed_experts'] as a TensorRef, with start, "
    "shape and dtype in Turn(assistant).extra",
    "multi_modal_data": "Turn(assistant).extra['multi_modal_data'], key written only when the "
    "NotRequired field is present",
    "prompt_attribution": "Turn(context).extra['prompt_attribution'], key written only when the "
    "NotRequired field is present",
}

#: Fields of the record that a ``verifiers`` stream cannot fill, and why. Deliberate, not missed.
NOT_FILLED: Mapping[str, str] = {
    "Turn.logprobs_train": "verifiers records the sampling engine's logprobs only. Filling both "
    "columns from one stream would make E6 report a mismatch of exactly zero, which is the "
    "opposite claim from 'the mismatch was not measured'.",
    "Trajectory.advantage_tokens": "verifiers computes one advantage per rollout and copies it "
    "onto every step. There is no per-token advantage tensor anywhere in the object.",
    "Trajectory.labels": "there is no held-out oracle channel in a TrajectoryStep. Blind labels "
    "are attached by whoever holds the answer key, not by a converter.",
    "SegmentProvenance.staleness_steps": "nothing in TrajectoryStep says how far behind the "
    "current policy the generating weights were. The caller declares it and the declaration is "
    "recorded as a declaration.",
    "OptimizerTelemetry": "verifiers is an environment and rubric library. It does not take the "
    "optimizer step and records nothing about one.",
}

#: Places the canonical record and this framework did not line up, stated as facts about the two
#: schemas. Carried on every `ConversionReport`, because a finding that lives only in a build
#: report is a finding nobody reads twice.
CONVERTER_FINDINGS: tuple[str, ...] = (
    "verifiers substitutes 0.0 for any exception raised by a reward function "
    f"({SILENT_ZERO_SITE}) and keeps no counter, so a zero in state['metrics'] is not "
    "distinguishable from a swallowed exception by any reader of the record. Without tap records "
    "the converter reports the count of exact zeros as an upper bound and marks nothing abstained.",
    "TrajectoryStepTokens carries one prompt_ids array per step with no per-message boundary in "
    "the required fields, so a multi-message prompt becomes one context turn. prompt_attribution "
    "carries message_indices per token and would allow an exact split; it is NotRequired, only "
    "renderer clients populate it, and this converter passes it through rather than depending on "
    "it.",
    "A step's prompt tokens are the whole re-rendered prefix, not the delta. The delta is recovered "
    "by checking that the previous prompt and completion ids are a literal prefix of this step's "
    "prompt ids. When the check fails the context turn records token_ids=None rather than a "
    "plausible slice.",
    "verifiers records response.model, which is a model name and not a checkpoint identity. Two "
    "policy versions a hundred optimizer steps apart serve the same string, so "
    "PolicyMixture.singular being True on a converted record means 'one model name', not 'one set "
    "of weights'.",
    "TrajectoryStep has two is_truncated flags with different meanings, one on the step and one on "
    "the tokens. The tokens-level flag is the max_seq_len clip; the step-level flag is that union "
    "with the engine's own finish_reason. Collapsing them loses the reason the sequence ended.",
    "record.tensors.AbsenceReason has seven members and none of them means 'the payload was "
    "present in the source and this converter was given nowhere to put it'. The routing ref for "
    "that case is an AbsentRef with a written remedy, and the presence is also recorded on the "
    "turn so it cannot read as an absence of routing.",
    "The TrajectoryStep stream is opt-in on disk. save_utils.state_to_output copies `trajectory` "
    "onto a written rollout only when --state-columns names it, so an ordinary vf-eval run "
    "serialises the messages, the reward and the per-function metrics and nothing token-level. "
    "The one object in the ecosystem that holds ids, logprobs, routing and advantage together is "
    "therefore usually not the object that reaches disk, and a converter has to handle both.",
    "record.scores.WeightedSum makes the whole sum NaN when any child abstains, including a "
    "weight-0 child. verifiers' Rubric.add_metric registers weight-0 functions whose failure does "
    "not touch state['reward'], so on such a rollout evaluate(scores) is NaN while the framework's "
    "own reward is a number. Both are recorded: the tree and features['verifiers_realised_reward'].",
)

_REPR_CAP = 512

#: verifiers' message roles mapped onto `record.turns.TurnRole`. "text" is the completion-style
#: message and its record role depends on which side of the step it is on, so it is not in here.
_ROLE_MAP: Mapping[str, str] = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
}


# ---------------------------------------------------------------------------
# Reading whatever the caller is holding
# ---------------------------------------------------------------------------


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """One field, from a mapping or from an object with attributes.

    A ``TrajectoryStep`` is a ``TypedDict`` and therefore a dict at runtime, but ``response`` is a
    pydantic model in process and a nested dict once it has been through JSON, and ``State`` is a
    dict subclass whose ``__getitem__`` forwards four keys to ``state["input"]``. One accessor that
    handles all three is the difference between this module working on a live rollout and working
    only on a file.
    """
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _has(obj: Any, key: str) -> bool:
    """Whether a field is present at all, which is not the same as being ``None``.

    ``multi_modal_data`` and ``prompt_attribution`` are ``NotRequired[Any]``, and
    ``parse_response_tokens`` writes them only when they are not ``None``. Absent and present-as-
    ``None`` are different facts about the rollout and the record keeps them apart.
    """
    if obj is None:
        return False
    if isinstance(obj, Mapping):
        return key in obj
    return hasattr(obj, key)


def _json_safe(value: Any, *, cap: int = _REPR_CAP) -> tuple[Any, bool]:
    """Coerce a payload into something the record's JSON writer can hold, and say if it was coerced.

    ``Turn.extra`` is written through ``__canonical__`` and then serialised, so a pixel tensor
    landing there breaks the writer rather than the reader. The same discipline
    `record.scores._codec_safe` uses: primitives, sequences and mappings pass through, anything
    else becomes a ``repr`` capped at 512 characters, and the second element says which happened. A
    field that is a string because it always was and one that is a string because the real thing
    would not serialise are different facts.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value, False
    if isinstance(value, (list, tuple)):
        items = [_json_safe(v, cap=cap) for v in value]
        return [p for p, _ in items], any(flag for _, flag in items)
    if isinstance(value, Mapping):
        pairs = {str(k): _json_safe(v, cap=cap) for k, v in value.items()}
        return ({k: p for k, (p, _) in pairs.items()}, any(f for _, f in pairs.values()))
    dumped = getattr(value, "model_dump", None)
    if callable(dumped):
        try:
            return _json_safe(dumped(), cap=cap)
        except Exception:  # pragma: no cover - a pydantic model that refuses to dump
            pass
    return repr(value)[:cap], True


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_int(value: Any) -> int | None:
    """An int where there is one. ``bool`` is excluded because ``True`` is not a top_k of 1."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _ints(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        return tuple(int(v) for v in value)
    except (TypeError, ValueError):
        return None


def _floats(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None


def _bools(value: Any) -> tuple[bool, ...] | None:
    """A 0/1 integer mask as bools. verifiers writes 1 for a token the loss should see."""
    if value is None:
        return None
    try:
        return tuple(bool(v) for v in value)
    except TypeError:
        return None


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def _messages(value: Any) -> tuple[dict[str, Any], ...]:
    """Normalise ``Messages`` to a tuple of plain dicts.

    ``Messages`` is ``list[Message]`` and a ``Message`` is one of five pydantic models, or the same
    thing as a dict after a round trip through JSON. A bare string is accepted too, because a
    completion-style environment's prompt is one, and it becomes a single ``text`` message rather
    than being dropped.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return ({"role": "text", "content": value},)
    if isinstance(value, Mapping):
        value = [value]
    out: list[dict[str, Any]] = []
    for m in value:
        if isinstance(m, str):
            out.append({"role": "text", "content": m})
            continue
        safe, _ = _json_safe(m)
        if isinstance(safe, Mapping):
            out.append(dict(safe))
        else:
            out.append({"role": "text", "content": safe})
    return tuple(out)


def _content_text(content: Any) -> str:
    """The text of a message's content, which may be a string or a list of content parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, Mapping) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "".join(parts)
    return str(content)


def _render(messages: Sequence[Mapping[str, Any]]) -> str:
    """Messages as one string, one per line, prefixed by role.

    Deliberately not a chat template. The tokens are already on the record and are the thing
    instruments read; this is the human-readable column and it does not pretend to be what the
    model saw.
    """
    lines: list[str] = []
    for m in messages:
        role = str(m.get("role", "?"))
        text = _content_text(m.get("content"))
        if m.get("reasoning_content"):
            text = f"{_content_text(m.get('reasoning_content'))}\n{text}"
        lines.append(f"{role}: {text}" if role != "text" else text)
    return "\n".join(lines)


def _common_prefix(a: Sequence[Any], b: Sequence[Any]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _tool_calls(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        for call in m.get("tool_calls") or ():
            if isinstance(call, Mapping):
                out.append(dict(call))
    return out


def _record_role(framework_role: str, *, position: str) -> str:
    """The record's turn role for a verifiers message.

    Two mappings that are decisions rather than lookups. A ``text`` message has no counterpart in
    the record's five roles, so it takes ``user`` in a prompt and ``assistant`` in a completion,
    which is what it is. And a ``user`` message that arrives *after* the opening prompt is not a
    user: it is what ``env_response`` returned, so it takes ``environment``. The framework's own
    role string is kept on the turn in either case, so neither mapping loses anything.

    ``position`` is one of ``prompt`` (the opening prompt), ``env`` (a message that appeared
    mid-rollout) and ``completion`` (something the model produced).
    """
    if framework_role == "text":
        return "assistant" if position == "completion" else "user"
    role = _ROLE_MAP.get(framework_role, "user")
    if position == "env" and role == "user":
        return "environment"
    return role


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass
class ConversionReport:
    """What the conversion carried and where it could not carry something.

    Every count comes from the conversion that produced it. Mutable and accumulated across calls on
    purpose: a converter used on one rollout and a converter used on a run both want one report at
    the end, and freezing it per call would mean summing dataclasses by hand.
    """

    trajectories: int = 0
    steps_in: int = 0
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    logprobs_carried: int = 0
    #: Steps whose ``tokens`` field was ``None``: a hosted endpoint that returns no token data.
    steps_without_tokens: int = 0
    #: Rollouts that carried no ``trajectory`` at all, which is what an ordinary ``vf-eval`` run
    #: writes: ``state_to_output`` copies the stream only when ``--state-columns trajectory`` names
    #: it. Those rows convert from their messages and carry no token ids anywhere.
    rows_without_trajectory: int = 0
    #: Steps where the previous prompt and completion ids were not a literal prefix of this step's
    #: prompt ids, so the new tokens could not be identified.
    non_prefix_steps: int = 0
    #: Steps whose prompt tokens were *shorter* than the previous prefix, which is a rewrite.
    compaction_events: int = 0
    routing_present: int = 0
    routing_stored: int = 0
    routing_absent: int = 0
    multimodal_steps: int = 0
    attribution_steps: int = 0
    leaves: int = 0
    #: Leaves a tap record proved were a crashed grader. Exact.
    known_abstentions: int = 0
    #: Leaves whose value is exactly zero and which no tap record explains. The upper bound on the
    #: silent-zero count, and the only number the record alone supports.
    unresolved_zeros: int = 0
    policy_versions: tuple[str, ...] = ()
    mask_signatures: tuple[str, ...] = ()
    findings: tuple[str, ...] = CONVERTER_FINDINGS

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def render(self) -> str:
        return (
            f"{self.trajectories} trajectories ({self.rows_without_trajectory} with no "
            f"TrajectoryStep stream), {self.steps_in} steps, {self.turns} turns, "
            f"{self.prompt_tokens}+{self.completion_tokens} tokens, "
            f"{self.logprobs_carried} logprobs, routing {self.routing_stored}/{self.routing_present}"
            f" stored, {self.known_abstentions} known abstentions, "
            f"{self.unresolved_zeros} unresolved zeros"
        )

    def note(self, policy_version: str) -> None:
        if policy_version not in self.policy_versions:
            self.policy_versions = self.policy_versions + (policy_version,)


# ---------------------------------------------------------------------------
# The estimator, which is where E7's first fact lands
# ---------------------------------------------------------------------------


def estimator_spec(
    *,
    weights: Mapping[str, float] | None = None,
    has_group_rewards: bool | None = None,
    provides_advantages: bool | None = None,
) -> "EstimatorSpec":
    """How ``verifiers`` turns scores into advantages, read off ``score_group`` rather than guessed.

    ``rubrics/rubric.py:406-409`` is four lines and they are the whole estimator::

        avg_reward = sum(aggregated_rewards) / num_states
        for i, state in enumerate(states):
            state["reward"] = aggregated_rewards[i]
            state["advantage"] = aggregated_rewards[i] - avg_reward

    Mean centring, and nothing else. No standard-deviation division, so ``std_normalised`` is False
    and ``std_epsilon`` is None rather than zero: there is no epsilon, and writing ``0.0`` would
    say there is one and it is small. `EstimatorSpec.z_scored` is then False, which is what the
    amplifier-safety instrument reads to say the mechanism is absent on this framework instead of
    reporting a ratio with no denominator.

    ``aggregation`` is ``unknown`` and that is not a gap in this converter. ``verifiers`` computes
    rewards and advantages and does not take the optimizer step, so how the per-token loss is
    aggregated is a property of whatever trainer consumed the rollouts. Writing ``sequence`` here
    because the advantage is per rollout would be answering a different question.

    ``family`` is ``verifiers/score_group`` because ``measure/estimator/spec.py`` resolves any
    family beginning ``verifiers`` to its own verified row, and the two must not drift apart.
    """
    from reward_lens.record.schema import EstimatorSpec

    return EstimatorSpec(
        family="verifiers/score_group",
        group_centred=True,
        std_normalised=False,
        std_epsilon=None,
        degenerate_policy=(
            "a group whose rewards are all equal produces an advantage of exactly zero for every "
            "rollout, because the numerator is zero. verifiers does not drop the group, does not "
            "divide by a standard deviation and has no epsilon, so a degenerate group contributes "
            "nothing rather than contributing amplified noise. A rollout whose reward function "
            f"raised is scored 0.0 by {SILENT_ZERO_SITE} and enters the mean as a real score."
        ),
        clip_low=None,
        clip_high=None,
        clip_ratio_c=None,
        aggregation="unknown",
        loss_mask_policy=(
            "the record carries a per-token completion_mask emitted by the client, 0 for a prompt "
            "token and 1 for a trainable one. verifiers does not compute the loss, so it does not "
            "define the masking policy the trainer applies to that mask."
        ),
        off_policy_correction=None,
        kl_penalty=None,
        kl_coefficient=None,
        advantage_whitening=False,
        extra={
            "source": f"{SCORE_GROUP_SITE}, verifiers@{VERIFIERS_COMMIT[:8]}",
            "advantage_formula": "advantage_i = reward_i - mean(rewards over the group)",
            "std_division": "absent",
            "amplification_mechanism": (
                "absent: the E series' amplification ratio is a statement about a z-score and "
                "there is no denominator here to inflate"
            ),
            "silent_zero_site": SILENT_ZERO_SITE,
            "backfill_site": BACKFILL_SITE,
            "reward_weights": dict(weights) if weights else None,
            "has_group_rewards": has_group_rewards,
            "provides_advantages": provides_advantages,
        },
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _routing_ref(
    payload: Any, *, store: "TensorStore | None", where: str, report: ConversionReport
) -> "TensorRef":
    """A ``RoutedExpertsPayload`` as a `TensorRef`, stored where there is somewhere to store it.

    ``RoutedExpertsPayload`` is ``{data, shape, start, dtype}`` with ``data`` deliberately typed
    ``Any`` so pydantic does not try to validate a ``memoryview``. With a `TensorStore` the bytes
    are decoded and content-addressed and the result is a `StoredRef`. Without one they are not
    silently dropped: the returned `AbsentRef` says the payload was on the step and this converter
    had nowhere to put it, which is a different sentence from "the routing was not captured", and
    `AbsenceReason` has no member for it. That gap is in `CONVERTER_FINDINGS`.
    """
    import numpy as np

    from reward_lens.record.tensors import AbsenceReason, AbsentRef

    shape = tuple(int(s) for s in (_get(payload, "shape") or ()))
    dtype = str(_get(payload, "dtype") or "uint8")
    start = int(_get(payload, "start") or 0)
    data = _get(payload, "data")
    report.routing_present += 1

    if store is None:
        report.routing_absent += 1
        return AbsentRef(
            reason=AbsenceReason.NOT_CAPTURED,
            detail=(
                f"{where}: the routed_experts payload was present on this TrajectoryStepTokens "
                f"(shape={shape}, dtype={dtype}, start={start}) and this conversion was given no "
                f"tensor store, so the bytes were not persisted. This is not an absence of routing "
                f"in the run."
            ),
            remedy=(
                "Pass store=TensorStore(path) to the converter and re-run the conversion against "
                "the same rollouts. The bytes are in the source object, not lost."
            ),
            statistics={"start": float(start), "ndim": float(len(shape))},
        )

    try:
        if isinstance(data, (bytes, bytearray, memoryview)):
            array = np.frombuffer(bytes(data), dtype=np.dtype(dtype))
        else:
            array = np.asarray(data, dtype=np.dtype(dtype))
        if shape:
            array = array.reshape(shape)
    except Exception as exc:
        report.routing_absent += 1
        return AbsentRef(
            reason=AbsenceReason.NOT_CAPTURED,
            detail=(
                f"{where}: the routed_experts payload could not be decoded as "
                f"dtype={dtype} shape={shape}: {type(exc).__name__}: {exc}"
            ),
            remedy=(
                "Check that the payload's dtype and shape describe its data buffer. verifiers "
                "keeps the buffer opaque on purpose, so a producer that changed the encoding "
                "without changing the dtype string is the case to look for."
            ),
            statistics={"start": float(start), "ndim": float(len(shape))},
        )
    ref = store.put(array, name="routed_experts")
    report.routing_stored += 1
    return ref


# ---------------------------------------------------------------------------
# Turns
# ---------------------------------------------------------------------------


@dataclass
class _StepTurns:
    """One step's turns, plus what the next step needs to know about this one."""

    turns: list["Turn"]
    prompt_messages: tuple[dict[str, Any], ...]
    completion_messages: tuple[dict[str, Any], ...]
    prompt_ids: tuple[int, ...] | None
    completion_ids: tuple[int, ...] | None
    policy_version: str
    compaction: Any = None


def _step_turns(
    step: Any,
    *,
    index: int,
    first_turn: int,
    previous: "_StepTurns | None",
    store: "TensorStore | None",
    report: ConversionReport,
    trajectory_id: str,
) -> _StepTurns:
    """Turn one ``TrajectoryStep`` into at most two `Turn` objects.

    The context turn exists when the step brought new messages or new prompt tokens; the assistant
    turn always exists, because a step is a model action and recording the action with no turn
    would lose the completion.
    """
    from reward_lens.record.compaction import CompactionEvent
    from reward_lens.record.turns import ToolCall, Turn

    prompt_messages = _messages(_get(step, "prompt"))
    completion_messages = _messages(_get(step, "completion"))
    tokens = _get(step, "tokens")
    response = _get(step, "response")
    policy_version = str(_get(response, "model") or "unknown")
    report.note(policy_version)

    prompt_ids = _ints(_get(tokens, "prompt_ids"))
    prompt_mask = _bools(_get(tokens, "prompt_mask"))
    completion_ids = _ints(_get(tokens, "completion_ids"))
    completion_mask = _bools(_get(tokens, "completion_mask"))
    completion_logprobs = _floats(_get(tokens, "completion_logprobs"))
    if tokens is None:
        report.steps_without_tokens += 1

    # -- the context turn: what is new at this step --------------------------
    carry = 0
    shrank = False
    if previous is None:
        new_messages = prompt_messages
        new_ids: tuple[int, ...] | None = prompt_ids
        new_mask = prompt_mask
        position = "prompt"
    else:
        prior = previous.prompt_messages + previous.completion_messages
        lcp = _common_prefix(prior, prompt_messages)
        new_messages = prompt_messages[lcp:]
        position = "env"
        if previous.prompt_ids is not None and previous.completion_ids is not None:
            carry = len(previous.prompt_ids) + len(previous.completion_ids)
        prior_ids = (previous.prompt_ids or ()) + (previous.completion_ids or ())
        if prompt_ids is None or carry == 0:
            new_ids, new_mask = (None, None)
        elif prompt_ids[:carry] == prior_ids:
            new_ids = prompt_ids[carry:]
            new_mask = prompt_mask[carry:] if prompt_mask is not None else None
        else:
            report.non_prefix_steps += 1
            new_ids, new_mask = (None, None)
        # A prompt that is *shorter* than the prefix it should have extended is a rewrite. The
        # event is built after the turns, because `at_turn` is the index of the first turn
        # generated against the new prefix and that is this step's assistant turn.
        shrank = prompt_ids is not None and carry > 0 and len(prompt_ids) < carry

    turns: list[Turn] = []
    if new_messages or new_ids:
        role = _record_role(
            str(new_messages[-1].get("role", "user")) if new_messages else "user",
            position=position,
        )
        extra: dict[str, Any] = {
            "messages": [dict(m) for m in new_messages],
            "verifiers_roles": [str(m.get("role", "")) for m in new_messages],
            "verifiers_step": index,
        }
        if _has(tokens, "prompt_attribution"):
            payload, coerced = _json_safe(_get(tokens, "prompt_attribution"))
            extra["prompt_attribution"] = payload
            extra["prompt_attribution_is_repr"] = coerced
            report.attribution_steps += 1
        if new_ids is None and prompt_ids is not None and previous is not None:
            extra["prompt_ids_not_prefix_stable"] = True
            extra["step_prompt_ids_len"] = len(prompt_ids)
        turns.append(
            Turn(
                index=first_turn + len(turns),
                role=role,  # type: ignore[arg-type]
                text=_render(new_messages),
                token_ids=new_ids,
                loss_mask=new_mask if new_ids is not None else None,
                overlong_prompt=(
                    bool(_get(tokens, "overlong_prompt")) if tokens is not None else None
                ),
                extra=extra,
            )
        )
        if new_ids is not None:
            report.prompt_tokens += len(new_ids)

    # -- the assistant turn: the action ---------------------------------------
    a_extra: dict[str, Any] = {
        "messages": [dict(m) for m in completion_messages],
        "verifiers_step": index,
        "response": {
            "id": _get(response, "id"),
            "created": _get(response, "created"),
            "model": _get(response, "model"),
            "finish_reason": _get(_get(response, "message"), "finish_reason"),
            "usage": _json_safe(_get(response, "usage"))[0],
        },
    }
    if tokens is not None:
        a_extra["tokens_is_truncated"] = bool(_get(tokens, "is_truncated"))
    extras_payload = _get(step, "extras")
    if extras_payload:
        payload, coerced = _json_safe(extras_payload)
        a_extra["step_extras"] = payload
        a_extra["step_extras_is_repr"] = coerced
    calls = _tool_calls(completion_messages)
    if calls:
        a_extra["tool_calls"] = calls
    if _has(tokens, "multi_modal_data"):
        payload, coerced = _json_safe(_get(tokens, "multi_modal_data"))
        a_extra["multi_modal_data"] = payload
        a_extra["multi_modal_data_is_repr"] = coerced
        report.multimodal_steps += 1

    tensors: dict[str, Any] = {}
    routing = _get(tokens, "routed_experts")
    if routing is not None:
        tensors["routed_experts"] = _routing_ref(
            routing,
            store=store,
            where=f"trajectory {trajectory_id} step {index}",
            report=report,
        )
        a_extra["routed_experts_start"] = int(_get(routing, "start") or 0)
        a_extra["routed_experts_shape"] = [int(s) for s in (_get(routing, "shape") or ())]
        a_extra["routed_experts_dtype"] = str(_get(routing, "dtype") or "")

    tool_call = None
    if len(calls) == 1:
        tool_call = ToolCall(
            name=str(calls[0].get("name", "")),
            arguments=str(calls[0].get("arguments", "")),
            call_id=calls[0].get("id"),
        )

    turns.append(
        Turn(
            index=first_turn + len(turns),
            role="assistant",
            text=_render(completion_messages),
            token_ids=completion_ids,
            logprobs_sampling=completion_logprobs,
            logprobs_train=None,
            loss_mask=completion_mask,
            tool_call=tool_call,
            step_score=_as_float(_get(step, "reward")),
            step_advantage=_as_float(_get(step, "advantage")),
            truncated=bool(_get(step, "is_truncated")) if _has(step, "is_truncated") else None,
            tensors=tensors,
            extra=a_extra,
        )
    )
    if completion_ids is not None:
        report.completion_tokens += len(completion_ids)
    if completion_logprobs is not None:
        report.logprobs_carried += len(completion_logprobs)

    compaction = None
    if shrank:
        assert prompt_ids is not None
        report.compaction_events += 1
        compaction = CompactionEvent(
            at_turn=turns[-1].index,
            tokens_before=carry,
            tokens_after=len(prompt_ids),
            method="prefix_rewrite",
            extra={
                "detected_by": (
                    "this step's prompt_ids are shorter than the previous step's prompt ids "
                    "followed by its completion ids, so the conditioning prefix was rewritten "
                    "between the two steps"
                ),
                "trajectory_id": trajectory_id,
                "verifiers_step": index,
            },
        )

    return _StepTurns(
        turns=turns,
        prompt_messages=prompt_messages,
        completion_messages=completion_messages,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        policy_version=policy_version,
        compaction=compaction,
    )


def _provenance(
    blocks: Sequence[_StepTurns],
    *,
    engine: "Engine | None",
    staleness_steps: int,
    sampling: Mapping[str, Any] | None,
) -> tuple["SegmentProvenance", ...]:
    """Segments keyed on ``response.model``, merged across consecutive steps that agree.

    Mandatory and plural, and here it is genuinely plural: a rollout that resumed under a newer
    served model has two segments and `PolicyMixture.singular` is False for it, which is what
    `NEAR_POLICY` reads. The caveat is in `CONVERTER_FINDINGS`: ``response.model`` is a model name,
    so two checkpoints served under one name look singular.
    """
    from reward_lens.record.provenance import SamplingMeta, SegmentProvenance
    from reward_lens.record.tensors import Engine

    eng = engine or Engine(name="unknown")
    args = sampling or {}
    meta = SamplingMeta(
        temperature=_as_float(args.get("temperature")),
        top_p=_as_float(args.get("top_p")),
        top_k=_opt_int(args.get("top_k")),
        seed=_opt_int(args.get("seed")),
        max_tokens=_opt_int(args.get("max_tokens")),
        extra={
            "policy_version_is": (
                "response.model, which is the served model name. verifiers records no checkpoint "
                "identity and no weight version anywhere in a TrajectoryStep."
            ),
            "staleness_provenance": (
                f"declared by the caller as {staleness_steps}; verifiers records nothing from "
                f"which staleness could be measured, so this is not a measurement"
            ),
        },
    )

    segments: list[SegmentProvenance] = []
    start = 0
    cursor = 0
    current = blocks[0].policy_version if blocks else "unknown"
    for block in blocks:
        if block.policy_version != current:
            segments.append(
                SegmentProvenance(
                    turn_range=(start, cursor),
                    policy_version=current,  # type: ignore[arg-type]
                    staleness_steps=staleness_steps,
                    engine=eng,
                    sampling=meta,
                )
            )
            start = cursor
            current = block.policy_version
        cursor += len(block.turns)
    if cursor > start:
        segments.append(
            SegmentProvenance(
                turn_range=(start, cursor),
                policy_version=current,  # type: ignore[arg-type]
                staleness_steps=staleness_steps,
                engine=eng,
                sampling=meta,
            )
        )
    return tuple(segments)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def _call_index(calls: Iterable["GraderCall"] | None) -> dict[str, "GraderCall"]:
    """Tap records for one rollout, keyed by grader name. Last write wins, which is the last call."""
    out: dict[str, Any] = {}
    for call in calls or ():
        out[str(_get(call, "grader"))] = call
    return out


def score_tree(
    metrics: Mapping[str, float | None],
    *,
    weights: Mapping[str, float] | None = None,
    calls: Iterable["GraderCall"] | None = None,
) -> Any:
    """One rollout's composition as a `ScoreTree`, from ``state["metrics"]`` and the rubric weights.

    ``Rubric`` composes by weighted sum: ``add_reward_func(func, weight=1.0)`` contributes to the
    reward and ``add_metric(func, weight=0.0)`` is recorded and does not. Both land in
    ``state["metrics"]``, so the weights are the only thing that separates them and they are not on
    the rollout: the caller reads them off the rubric and passes them here. Missing weights default
    to 1.0, which is ``add_reward_func``'s default and the wrong answer for a metric, so a caller
    who has the rubric should always pass them.

    **Abstention is only claimed where it can be proved.** With tap records a leaf whose grader
    raised is ``abstained=True`` with the substituted ``0.0`` still on it, which is
    `Leaf.silent_zero` and is B4's exact numerator. Without them nothing is marked, because
    ``verifiers`` erased the distinction before ``state["metrics"]`` was written and inventing it
    here would move a defect census from measurement to assertion. The count of exact zeros goes on
    the trajectory instead, as an upper bound.

    Pure, and deliberately so. The conversion report's leaf and zero counts are taken once per
    trajectory from the tree that was built, whichever branch built it, so the counter and the
    per-trajectory feature can never disagree.
    """
    from reward_lens.record.scores import GraderCallRef, Leaf, WeightedSum

    index = _call_index(calls)
    names = tuple(metrics)
    if not names:
        return None

    leaves: list[Leaf] = []
    ws: list[float] = []
    for name in names:
        value = _as_float(metrics[name])
        call = index.get(name)
        outcome = str(_get(_get(call, "outcome"), "value", None) or _get(call, "outcome") or "")
        raised = outcome in ("raised", "timed_out")
        if call is not None:
            ref = GraderCallRef(
                grader=name,
                outcome=outcome or "returned",
                facets=dict(_get(call, "facets") or {}),
                latency_s=(lambda ns: ns / 1e9 if isinstance(ns, int) else None)(
                    _get(call, "inner_ns")
                ),
                error_type=_get(call, "error_type"),
                error_message=_get(call, "error_message"),
                seq=_get(call, "seq"),
                step=_get(call, "step"),
            )
        else:
            ref = GraderCallRef(
                grader=name,
                outcome="returned",
                facets={
                    "abstention_channel": "none",
                    "framework_substitutes_zero_on_exception": SILENT_ZERO_SITE,
                    "zero_is_ambiguous": value == 0.0,
                    "note": (
                        "verifiers erases the difference between a genuine zero and a crashed "
                        "reward function before state['metrics'] is written. This ref records "
                        "that the outcome was not observed, not that the call returned cleanly."
                    ),
                },
            )
        leaves.append(
            Leaf(name=name, value=value, grader_call=ref, abstained=raised or value is None)
        )
        ws.append(float((weights or {}).get(name, 1.0)))

    if len(leaves) == 1 and ws == [1.0]:
        # An unweighted single reward function is its own composition. Wrapping it in a sum of one
        # would put a node in the tree the rubric does not have, and node names are part of the
        # record's identity.
        return leaves[0]
    return WeightedSum(name="reward", children=tuple(leaves), weights=tuple(ws))


# ---------------------------------------------------------------------------
# The converter
# ---------------------------------------------------------------------------


@dataclass
class VerifiersConverter:
    """Turn ``verifiers`` rollouts into the canonical record.

    Four entry points, in increasing size: `trajectory` from a ``TrajectoryStep`` stream, `rollout`
    from a whole ``State`` or ``RolloutOutput``, `group` from the K rollouts of one prompt, and
    `run` from a set of groups. Each accumulates into `report`, which is the honest account of what
    the conversion carried.

    The options are the things ``verifiers`` does not record and a reader would otherwise have to
    guess. ``engine`` is the serving stack, ``staleness_steps`` is how far behind the current policy
    the rollouts were, ``weights`` is the rubric's own weight vector, and ``store`` is where a
    routed-expert payload goes. Every one of them defaults to the honest empty value and the record
    says that is what it is.
    """

    run_id: str = "verifiers"
    weights: Mapping[str, float] | None = None
    engine: "Engine | None" = None
    staleness_steps: int = 0
    sampling: Mapping[str, Any] | None = None
    store: "TensorStore | None" = None
    failure_at: float | None = None
    framework_version: str = "unknown"
    report: ConversionReport = field(default_factory=ConversionReport)

    def _count(self, tree: Any) -> None:
        """Fold one trajectory's score tree into the report, in the one place it happens.

        Both entry points call this and nothing else counts leaves, so
        ``report.unresolved_zeros`` is by construction the sum of every trajectory's
        ``features["verifiers_unresolved_zeros"]``. A counter that could drift from the field it
        summarises is a counter that will.
        """
        for leaf in _leaves(tree):
            self.report.leaves += 1
            if leaf.abstained:
                self.report.known_abstentions += 1
            elif leaf.value == 0.0:
                self.report.unresolved_zeros += 1

    # -- the trajectory ------------------------------------------------------

    def trajectory(
        self,
        steps: Sequence[Any],
        *,
        trajectory_id: str | None = None,
        task_ref: str = "unknown",
        reward: float | None = None,
        advantage: float | None = None,
        metrics: Mapping[str, float | None] | None = None,
        calls: Iterable["GraderCall"] | None = None,
        labels: Mapping[str, Any] | None = None,
        extra_features: Mapping[str, float] | None = None,
    ) -> "Trajectory":
        """One rollout's ``TrajectoryStep`` stream as a `Trajectory`.

        ``reward`` and ``advantage`` are the rollout-level values from ``state``. When they are not
        supplied they are recovered from the steps, which works because ``score_group`` copies them
        down onto every step whose own value is ``None``: a uniform column is the rollout value. A
        column that is *not* uniform means the environment set genuine per-step values, and then
        there is no single trajectory-level number and this records ``None`` rather than picking
        one.
        """
        from reward_lens.record.schema import FeatureID, TaskID, Trajectory, TrajectoryID
        from reward_lens.record.tensors import CaptureRef, CaptureSpec

        steps = list(steps)
        tid = trajectory_id or str(_get(steps[0], "trajectory_id") if steps else "") or "unknown"

        blocks: list[_StepTurns] = []
        turns: list[Any] = []
        compactions: list[Any] = []
        previous: _StepTurns | None = None
        for i, step in enumerate(steps):
            block = _step_turns(
                step,
                index=i,
                first_turn=len(turns),
                previous=previous,
                store=self.store,
                report=self.report,
                trajectory_id=tid,
            )
            blocks.append(block)
            turns.extend(block.turns)
            if block.compaction is not None:
                compactions.append(block.compaction)
            previous = block
        self.report.steps_in += len(steps)
        self.report.turns += len(turns)
        self.report.trajectories += 1

        step_rewards = [_as_float(_get(s, "reward")) for s in steps]
        step_advantages = [_as_float(_get(s, "advantage")) for s in steps]
        realised = reward if reward is not None else _uniform(step_rewards)
        adv = advantage if advantage is not None else _uniform(step_advantages)

        tree = (
            score_tree(metrics, weights=self.weights, calls=calls)
            if metrics
            else _reward_leaf(realised)
        )
        self._count(tree)

        features: dict[str, float] = {
            "verifiers_n_steps": float(len(steps)),
            "verifiers_prompt_tokens": float(
                sum(t.n_tokens or 0 for t in turns if t.role != "assistant")
            ),
            "verifiers_completion_tokens": float(
                sum(t.n_tokens or 0 for t in turns if t.role == "assistant")
            ),
            "verifiers_step_score_uniform": float(_uniform(step_rewards) is not None),
            "verifiers_step_advantage_uniform": float(_uniform(step_advantages) is not None),
            "verifiers_known_abstentions": float(
                sum(1 for leaf in _leaves(tree) if leaf.abstained)
            ),
            "verifiers_unresolved_zeros": float(
                sum(1 for leaf in _leaves(tree) if not leaf.abstained and leaf.value == 0.0)
            ),
        }
        if realised is not None:
            # What the framework actually used, kept beside the tree for the same reason the TRL
            # adapter keeps it: a record that holds only the metrologically right answer cannot
            # show that the run used a different one.
            features["verifiers_realised_reward"] = realised
        features.update({str(k): float(v) for k, v in (extra_features or {}).items()})

        routing = {
            f"routed_experts/turn{t.index}": t.tensors["routed_experts"]
            for t in turns
            if "routed_experts" in t.tensors
        }
        capture = (
            CaptureRef(spec=CaptureSpec(include_routing=True), tensors=routing) if routing else None
        )

        return Trajectory(
            id=TrajectoryID(tid),
            task_ref=TaskID(task_ref),
            turns=tuple(turns),
            scores=tree,
            advantage=adv,
            advantage_tokens=None,
            provenance=_provenance(
                blocks,
                engine=self.engine,
                staleness_steps=self.staleness_steps,
                sampling=self.sampling,
            ),
            compaction=tuple(compactions),
            labels=dict(labels or {}),
            features={FeatureID(k): v for k, v in features.items()},
            capture=capture,
        )

    # -- the rollout ---------------------------------------------------------

    def rollout(
        self,
        state: Any,
        *,
        calls: Iterable["GraderCall"] | None = None,
        task_ref: str | None = None,
        labels: Mapping[str, Any] | None = None,
    ) -> "Trajectory":
        """A whole ``State`` or ``RolloutOutput``, which carries the metrics the steps do not.

        ``state["metrics"]`` is the per-reward-function breakdown ``score_group`` writes at
        ``rubric.py:415-417`` and it is the only place the composition survives; ``state["reward"]``
        is the weighted total and ``state["advantage"]`` the centred one. A bare ``TrajectoryStep``
        stream has none of that, which is why this entry point exists.

        **A published ``vf-eval`` row usually has no ``trajectory`` at all**, and that is not an
        error in the row. ``save_utils.state_to_output`` copies ``trajectory`` onto the output only
        when it is named in ``--state-columns``, so an ordinary evaluation writes the messages, the
        reward and the per-function metrics and nothing token-level. When there is no stream this
        falls back to the top-level ``prompt`` and ``completion``, which is the whole conversation,
        and every turn carries ``token_ids=None``. That is honest: the ids were never written, and
        a converter that produced empty tuples there would say the rollout had no tokens.
        """
        steps = list(_get(state, "trajectory") or ())
        example = _get(state, "example_id")
        if example is None:
            example = _get(_get(state, "input"), "example_id")
        if example is None:
            example = _get(state, "id")
        task = task_ref or (f"example:{example}" if example is not None else "unknown")
        if not steps:
            return self._messages_only(state, task_ref=task, calls=calls, labels=labels)
        return self.trajectory(
            steps,
            trajectory_id=_get(state, "trajectory_id"),
            task_ref=task,
            reward=_as_float(_get(state, "reward")),
            advantage=_as_float(_get(state, "advantage")),
            metrics=_get(state, "metrics") or None,
            calls=calls,
            labels=labels,
        )

    def _messages_only(
        self,
        state: Any,
        *,
        task_ref: str,
        calls: Iterable["GraderCall"] | None,
        labels: Mapping[str, Any] | None,
    ) -> "Trajectory":
        """A rollout whose ``trajectory`` was not serialised: messages, scores, no tokens.

        One context turn for the prompt and one turn per completion message, keeping each message's
        own role, because a multi-turn rollout's ``completion`` interleaves the model's messages
        with the environment's and collapsing them would attribute environment text to the policy.
        """
        from reward_lens.record.provenance import SamplingMeta, SegmentProvenance
        from reward_lens.record.schema import FeatureID, TaskID, Trajectory, TrajectoryID
        from reward_lens.record.tensors import Engine
        from reward_lens.record.turns import Turn

        prompt = _messages(_get(state, "prompt"))
        completion = _messages(_get(state, "completion"))
        turns: list[Turn] = []
        if prompt:
            turns.append(
                Turn(
                    index=0,
                    role=_record_role(str(prompt[-1].get("role", "user")), position="prompt"),  # type: ignore[arg-type]
                    text=_render(prompt),
                    extra={
                        "messages": [dict(m) for m in prompt],
                        "verifiers_roles": [str(m.get("role", "")) for m in prompt],
                        "token_ids_absent_because": (
                            "this rollout was written without --state-columns trajectory, so no "
                            "TrajectoryStep and no token ids were ever serialised"
                        ),
                    },
                )
            )
        for m in completion:
            role = str(m.get("role", "assistant"))
            turns.append(
                Turn(
                    index=len(turns),
                    role=_record_role(
                        role, position="env" if role != "assistant" else "completion"
                    ),  # type: ignore[arg-type]
                    text=_render([m]),
                    extra={"messages": [dict(m)], "verifiers_roles": [role]},
                )
            )
        self.report.trajectories += 1
        self.report.turns += len(turns)
        self.report.rows_without_trajectory += 1

        reward = _as_float(_get(state, "reward"))
        metrics = _get(state, "metrics") or None
        tree = (
            score_tree(metrics, weights=self.weights, calls=calls)
            if metrics
            else _reward_leaf(reward)
        )
        self._count(tree)
        features: dict[str, float] = {
            "verifiers_n_steps": 0.0,
            "verifiers_known_abstentions": float(
                sum(1 for leaf in _leaves(tree) if leaf.abstained)
            ),
            "verifiers_unresolved_zeros": float(
                sum(1 for leaf in _leaves(tree) if not leaf.abstained and leaf.value == 0.0)
            ),
        }
        if reward is not None:
            features["verifiers_realised_reward"] = reward
        for key in ("input_tokens", "output_tokens"):
            value = _as_float(_get(_get(state, "token_usage"), key))
            if value is not None:
                features[f"verifiers_{key}"] = value

        return Trajectory(
            id=TrajectoryID(str(_get(state, "trajectory_id") or task_ref)),
            task_ref=TaskID(task_ref),
            turns=tuple(turns),
            scores=tree,
            advantage=_as_float(_get(state, "advantage")),
            advantage_tokens=None,
            provenance=(
                (
                    SegmentProvenance(
                        turn_range=(0, len(turns)),
                        policy_version="unknown",  # type: ignore[arg-type]
                        staleness_steps=self.staleness_steps,
                        engine=self.engine or Engine(name="unknown"),
                        sampling=SamplingMeta(
                            extra={
                                "policy_version_is": (
                                    "unknown: this row carries no TrajectoryStep, so not even "
                                    "response.model survived the serialisation"
                                )
                            }
                        ),
                    ),
                )
                if turns
                else ()
            ),
            compaction=(),
            labels=dict(labels or {}),
            features={FeatureID(k): v for k, v in features.items()},
            capture=None,
        )

    # -- the group -----------------------------------------------------------

    def group(
        self,
        rollouts: Sequence[Any],
        *,
        step: int = 0,
        ordinal: int = 0,
        task_ref: str | None = None,
        calls: Mapping[str, Iterable["GraderCall"]] | None = None,
    ) -> "Group":
        """The K rollouts of one prompt, with the estimator that turned their scores into advantages.

        ``std_epsilon`` is ``0.0`` and that is not a threshold this converter chose. ``score_group``
        divides by nothing, so a group teaches nothing exactly when its standard deviation is
        exactly zero, and `GroupStats.degenerate` is ``std <= std_epsilon``. Any positive value
        would be inventing a tolerance the framework does not have.
        """
        from reward_lens.record.schema import Group, GroupID, GroupStats, TaskID, group_id

        trajectories = [
            self.rollout(
                r,
                calls=(calls or {}).get(str(_get(r, "trajectory_id") or ""), ()),
                task_ref=task_ref,
            )
            for r in rollouts
        ]
        task = task_ref or (str(trajectories[0].task_ref) if trajectories else "unknown")
        gid = group_id(run=self.run_id, step=step, task=task, ordinal=ordinal)
        scores: list[float | None] = [_as_float(_get(r, "reward")) for r in rollouts]
        return Group(
            id=GroupID(str(gid)),
            task_ref=TaskID(task),
            trajectories=tuple(trajectories),
            estimator=estimator_spec(weights=self.weights),
            group_stats=GroupStats.from_scores(scores, std_epsilon=0.0, failure_at=self.failure_at),
        )

    # -- the step and the run ------------------------------------------------

    def step(self, groups: Sequence["Group"], *, index: int = 0, **schedule: float) -> "Step":
        """One optimizer update's worth of groups.

        ``verifiers`` does not take the optimizer step, so `OptimizerTelemetry` is empty rather
        than zero-filled and ``index`` is whatever the caller's trainer called it. On a ``vf-eval``
        run there is no optimizer step at all and the index orders batches; the record cannot tell
        those apart, which is the same finding the campaign converter recorded.
        """
        from reward_lens.record.schema import OptimizerTelemetry, Step

        return Step(
            index=index,
            groups=tuple(groups),
            schedule={str(k): float(v) for k, v in schedule.items()},
            optimizer=OptimizerTelemetry(),
        )

    def run(
        self,
        steps: Sequence["Step"],
        *,
        kind: str = "eval",
        model: str | None = None,
        environment: str = "unknown",
        access: Mapping[Any, Any] | None = None,
    ) -> "Run":
        """The whole record.

        ``access`` defaults to `Access.RECORD` on every component, which is what a converted stream
        supports and nothing more: whether the environment can still be run and whether the grader
        can still be called are facts about what the analyst holds now, and a converter that
        asserted them would make every downstream access check wrong.
        """
        from reward_lens.core.types import Access, Component, Substrate
        from reward_lens.record.schema import (
            ComponentRef,
            InMemoryStepStream,
            Run,
            RunID,
            RunLineage,
            run_id,
        )

        policies = self.report.policy_versions
        return Run(
            id=RunID(str(run_id(name=self.run_id, environment=environment))),
            kind=kind,  # type: ignore[arg-type]
            components={
                Component.POLICY: ComponentRef(
                    name=model or (policies[0] if policies else "unknown"),
                    kind="policy",
                    extra={"served_model_names": list(policies)},
                ),
                Component.GRADER: ComponentRef(
                    name=environment,
                    kind="rubric",
                    substrate=Substrate.COMPOSITE,
                    extra={
                        "composition": "weighted sum over Rubric.funcs",
                        "weights": dict(self.weights) if self.weights else None,
                        "abstention_channel": "none",
                        "silent_zero_site": SILENT_ZERO_SITE,
                    },
                ),
                Component.ESTIMATOR: ComponentRef(name="score_group", kind="estimator"),
                Component.TASK: ComponentRef(name=environment, kind="environment"),
            },
            access=(
                dict(access)
                if access is not None
                else {
                    c: Access.RECORD
                    for c in (
                        Component.POLICY,
                        Component.GRADER,
                        Component.ESTIMATOR,
                        Component.TASK,
                        Component.RECORD,
                    )
                }
            ),
            regime=self.regime(),
            steps=InMemoryStepStream(steps),
            lineage=RunLineage(
                framework="verifiers",
                framework_version=self.framework_version,
                extra={
                    "environment": environment,
                    "converter": "reward_lens.tap.adapters.verifiers",
                    "verified_against_commit": VERIFIERS_COMMIT,
                    "report": self.report.render(),
                    "findings": list(self.report.findings),
                },
            ),
        )

    def regime(self) -> Any:
        """Declare only what the conversion actually settled, and note why the rest is absent.

        ``NO_COMPACTION`` is declared from the token arrays: every step's prompt ids were checked
        against the previous step's prompt plus completion ids, and a shrink is a rewrite. It is
        declared False when one was found, True when the check ran on every step and found none,
        and left undeclared when there were no tokens to check with, because "we did not look" is
        not a pass.

        ``NEAR_POLICY`` is never declared, and that is the interesting omission. It needs staleness
        and singular provenance; the provenance is available and the staleness is not, and a
        condition half of which was guessed is worth less than one nobody claimed. The note says so
        and travels with the record.
        """
        from reward_lens.core.envelope import RegimeCondition
        from reward_lens.record.schema import RegimeDeclaration

        declared: dict[Any, bool] = {}
        notes: dict[Any, str] = {
            RegimeCondition.NEAR_POLICY: (
                "not declared: verifiers records response.model but no checkpoint identity and no "
                "weight version, so staleness cannot be measured from the record. The provenance "
                "segments are real; the staleness on them is the caller's declaration."
            ),
            RegimeCondition.STATIONARY_GRADER: (
                "not declared: a Rubric's weights and functions are configuration this converter "
                "never sees, and reward functions receive the whole state, so a grader that "
                "changes with training progress is expressible and invisible here."
            ),
        }
        checked = self.report.steps_in - self.report.steps_without_tokens
        if checked > 0:
            found = self.report.compaction_events
            declared[RegimeCondition.NO_COMPACTION] = found == 0
            notes[RegimeCondition.NO_COMPACTION] = (
                f"checked on {checked} of {self.report.steps_in} steps by comparing each step's "
                f"prompt_ids against the previous step's prompt_ids + completion_ids; "
                f"{found} rewrite(s) found, {self.report.non_prefix_steps} step(s) where the "
                f"prefix relation did not hold at all"
            )
        else:
            notes[RegimeCondition.NO_COMPACTION] = (
                "not declared: no step carried token ids, so there was nothing to compare and no "
                "way to see a prefix rewrite"
            )
        return RegimeDeclaration(
            declared=declared,
            notes=notes,
            declared_by="reward_lens.tap.adapters.verifiers, measured from the token arrays",
        )


# ---------------------------------------------------------------------------
# Small helpers used above
# ---------------------------------------------------------------------------


def _uniform(values: Sequence[float | None]) -> float | None:
    """The single value a column holds, or None if it is empty, has a gap, or disagrees with itself.

    ``score_group`` back-fills the rollout's reward and advantage onto every step whose own value
    is ``None`` (``rubric.py:410-414``), so after scoring a uniform column *is* the rollout value.
    A column that disagrees with itself means the environment set genuine per-step values and there
    is no single trajectory-level number to lift.
    """
    if not values or any(v is None for v in values):
        return None
    first = values[0]
    assert first is not None
    for v in values[1:]:
        assert v is not None
        if not (v == first or (math.isnan(v) and math.isnan(first))):
            return None
    return first


def _reward_leaf(reward: float | None) -> Any:
    """The score tree for a stream with no metrics: one leaf holding what the pipeline used.

    A bare ``TrajectoryStep`` stream carries the total and not the composition, because the
    breakdown lives on ``state["metrics"]``. One leaf is the honest shape of that: it says there
    was one recorded number and does not invent components to hang under it.
    """
    from reward_lens.record.scores import GraderCallRef, Leaf

    if reward is None:
        return None
    return Leaf(
        name="reward",
        value=reward,
        grader_call=GraderCallRef(
            grader="rubric",
            outcome="returned",
            facets={
                "composition": "not recorded on the TrajectoryStep stream; pass metrics= or use "
                "rollout() with the State to recover the per-function breakdown",
                "silent_zero_site": SILENT_ZERO_SITE,
            },
        ),
        abstained=False,
    )


def _leaves(tree: Any) -> tuple[Any, ...]:
    from reward_lens.record.scores import Leaf, leaves

    if tree is None:
        return ()
    if isinstance(tree, Leaf):
        return (tree,)
    return leaves(tree)


def group_by_trajectory(steps: Iterable[Any]) -> dict[str, list[Any]]:
    """Split a flat ``TrajectoryStep`` stream by ``trajectory_id``, preserving order within each.

    The stream a trainer sees is flat and interleaved across concurrent rollouts, and
    ``trajectory_id`` is the only thing that puts it back together. Order within a trajectory is
    arrival order, which is generation order, because ``add_trajectory_step`` appends.
    """
    out: dict[str, list[Any]] = {}
    for step in steps:
        out.setdefault(str(_get(step, "trajectory_id") or "unknown"), []).append(step)
    return out


def convert_trajectory(steps: Sequence[Any], **kwargs: Any) -> "Trajectory":
    """One-shot conversion of a single ``TrajectoryStep`` stream. Keeps no report.

    Convenience for the common case. Anything that needs the conversion report, a tensor store or
    the rubric weights should build a `VerifiersConverter` and keep it.
    """
    converter_keys = {
        "run_id",
        "weights",
        "engine",
        "staleness_steps",
        "sampling",
        "store",
        "failure_at",
        "framework_version",
    }
    options = {k: v for k, v in kwargs.items() if k in converter_keys}
    rest = {k: v for k, v in kwargs.items() if k not in converter_keys}
    return VerifiersConverter(**options).trajectory(steps, **rest)


__all__ = [
    "BACKFILL_SITE",
    "CONVERTER_FINDINGS",
    "ConversionReport",
    "NOT_FILLED",
    "SCORE_GROUP_SITE",
    "SILENT_ZERO_SITE",
    "STEP_FIELD_MAP",
    "TOKEN_FIELDS",
    "TOKEN_FIELD_MAP",
    "TRAJECTORY_STEP_FIELDS",
    "VERIFIERS_COMMIT",
    "VerifiersConverter",
    "convert_trajectory",
    "estimator_spec",
    "group_by_trajectory",
    "score_tree",
]

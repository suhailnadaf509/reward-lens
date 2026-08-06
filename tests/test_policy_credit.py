"""Unit tests for `reward_lens.policy.credit`: the parts that do not need a model.

The acceptance file points the instrument at the reference record and asserts the conservation
clause. This file covers the arithmetic underneath it, the refusal paths, and the one generated
property test the group requires: the credit measure declares `tokenization` INVARIANT and this is where
that claim is checked rather than asserted.
"""

from __future__ import annotations

import math
import pickle
import sys
import types
from random import Random

import numpy as np
import pytest

from reward_lens.core.invariance import (
    GroupAction,
    InvarianceGroup,
    InvariancePayload,
    check_invariance,
)
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Span
from reward_lens.policy.credit import (
    SKYRL_MODULE,
    CreditDisintegration,
    Part,
    Partition,
    PartitionError,
    Segment,
    TrajectoryBatch,
    batch_from_skyrl,
    batch_from_trajectories,
    by_rollout,
    by_segment,
    credit_increment,
    implicit_prm,
    merge_empty,
    read_skyrl_dump,
    turn_mass,
    turn_segments,
)
from reward_lens.record.schema import make_trajectory
from reward_lens.record.turns import ToolCall, Turn


def _batch(n=3, t=6, advantages=(1.0, -0.5, 0.25), prompt=2, aggregation="token"):
    ids = np.arange(n * t, dtype=np.int64).reshape(n, t) + 1
    attention = np.ones_like(ids)
    loss = np.zeros_like(ids)
    loss[:, prompt:] = 1
    return TrajectoryBatch(
        input_ids=ids,
        attention_mask=attention,
        loss_mask=loss,
        advantages=np.asarray(advantages, dtype=np.float64),
        aggregation=aggregation,
    )


# ---------------------------------------------------------------------------
# The batch and its weights
# ---------------------------------------------------------------------------


def test_trained_positions_are_the_mask_shifted_into_target_coordinates():
    """The objective scores token t+1 from position t, so the lattice is `loss_mask[:, 1:]`."""
    batch = _batch()
    assert batch.trained.shape == (3, 5)
    assert batch.n_trained == 3 * 4
    assert batch.trained[0].tolist() == [False, True, True, True, True]


def test_the_three_aggregations_give_three_different_weightings():
    """Token-level versus sequence-level aggregation is a real transform.

    They do not differ by a constant: `sequence` divides each row by its own trained count, so on a
    ragged batch it moves the shares and not only the total. Here the rows are equal length, so the
    check is that the totals differ by the documented factors and the ratios within a row do not.
    """
    raw = _batch(aggregation="none").weights()
    token = _batch(aggregation="token").weights()
    sequence = _batch(aggregation="sequence").weights()

    assert np.allclose(token, raw / 12)
    assert np.allclose(sequence, raw / (4 * 3))
    # Within a row the relative weighting is the advantage and nothing else.
    row = raw[0][raw[0] != 0]
    assert np.allclose(row, 1.0)


def test_a_ragged_batch_makes_sequence_and_token_aggregation_disagree():
    ids = np.arange(12, dtype=np.int64).reshape(2, 6) + 1
    loss = np.zeros_like(ids)
    loss[0, 1:] = 1  # five trained targets
    loss[1, 4:] = 1  # two trained targets
    common = dict(input_ids=ids, attention_mask=np.ones_like(ids), advantages=np.array([1.0, 1.0]))
    token = TrajectoryBatch(loss_mask=loss, aggregation="token", **common).weights()
    sequence = TrajectoryBatch(loss_mask=loss, aggregation="sequence", **common).weights()
    assert token[0].sum() / token[1].sum() == pytest.approx(5 / 2)
    assert sequence[0].sum() / sequence[1].sum() == pytest.approx(1.0)


def test_a_per_token_advantage_survives_as_a_per_token_advantage():
    """Every framework computes the tensor; collapsing it here would be the discard this measures."""
    ids = np.arange(12, dtype=np.int64).reshape(2, 6) + 1
    per_token = np.linspace(0.0, 1.0, 10).reshape(2, 5)
    batch = TrajectoryBatch(
        input_ids=ids,
        attention_mask=np.ones_like(ids),
        loss_mask=np.ones_like(ids),
        advantages=per_token,
        aggregation="none",
    )
    assert batch.advantages.shape == (2, 5)
    assert np.allclose(batch.weights(), per_token)


def test_a_misshapen_advantage_is_rejected_at_construction():
    ids = np.arange(12, dtype=np.int64).reshape(2, 6)
    with pytest.raises(ValueError, match="target coordinates"):
        TrajectoryBatch(
            input_ids=ids,
            attention_mask=np.ones_like(ids),
            loss_mask=np.ones_like(ids),
            advantages=np.zeros((2, 6)),
        )


def test_an_unknown_aggregation_names_where_to_read_the_right_one():
    ids = np.arange(12, dtype=np.int64).reshape(2, 6)
    with pytest.raises(ValueError, match="EstimatorSpec.aggregation"):
        TrajectoryBatch(
            input_ids=ids,
            attention_mask=np.ones_like(ids),
            loss_mask=np.ones_like(ids),
            advantages=np.zeros(2),
            aggregation="mean",
        )


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------


def test_by_rollout_tiles_and_names_every_row():
    batch = _batch()
    partition = by_rollout(batch)
    assert len(partition) == 3
    assert partition.names == ("rollout 0", "rollout 1", "rollout 2")
    assert set(partition.kinds) == {"rollout"}
    assert sum(p.n_positions for p in partition.parts) == batch.n_trained


def test_by_segment_always_names_the_remainder():
    """A caller who names two spans has described part of the step, and the report must say so.

    Without the leftover part a report of two shares summing to 0.08 has no third row and reads as
    though 92% of the step went missing. This is the discipline `StepCost.residual_share` applies to
    the cost book, applied here.
    """
    batch = _batch()
    partition = by_segment(batch, [Segment(row=0, lo=2, hi=4, name="answer", kind="span")])
    assert "elsewhere" in partition.names
    assert len(partition) == 2
    named = next(p for p in partition.parts if p.name == "answer")
    assert named.n_positions == 2


def test_segments_sharing_a_name_across_rollouts_become_one_part():
    """A turn-level view of a batch is one part per turn index, not one per rollout per turn."""
    batch = _batch()
    segments = turn_segments(
        [[(0, 2, "turn 0 (user)", "turn"), (2, 6, "turn 1 (assistant)", "turn")] for _ in range(3)]
    )
    partition = by_segment(batch, segments)
    assert set(partition.names) == {"turn 0 (user)", "turn 1 (assistant)"}
    assistant = next(p for p in partition.parts if p.name == "turn 1 (assistant)")
    assert assistant.n_positions == batch.n_trained


def test_a_partition_with_a_gap_an_overlap_or_a_stray_is_refused():
    batch = _batch()
    trained = batch.trained
    with pytest.raises(PartitionError, match="uncovered"):
        Partition(parts=(Part("half", "span", trained & (np.arange(5) < 3)),), trained=trained)
    with pytest.raises(PartitionError, match="covered twice"):
        Partition(parts=(Part("a", "span", trained), Part("b", "span", trained)), trained=trained)
    everything = np.ones_like(trained, dtype=bool)
    with pytest.raises(PartitionError, match="the loss does not see"):
        Partition(parts=(Part("too much", "span", everything),), trained=trained)


def test_an_empty_partition_names_the_builders():
    with pytest.raises(PartitionError, match="by_rollout"):
        Partition(parts=(), trained=_batch().trained)


def test_merge_empty_folds_untrained_parts_without_breaking_the_tiling():
    """A forty-turn episode has many turns the loss never sees, and dropping them breaks the tile."""
    batch = _batch()
    segments = turn_segments(
        [[(0, 2, "prompt", "turn"), (2, 6, "completion", "turn")] for _ in range(3)]
    )
    partition = by_segment(batch, segments)
    assert any(p.n_positions == 0 for p in partition.parts)
    merged = merge_empty(partition)
    assert all(p.n_positions > 0 or p.kind == "empty" for p in merged.parts)
    assert sum(p.n_positions for p in merged.parts) == batch.n_trained


def test_a_part_mask_of_the_wrong_shape_is_caught():
    with pytest.raises(PartitionError, match="mask shape"):
        Partition(
            parts=(Part("wrong", "span", np.ones((2, 2), dtype=bool)),), trained=_batch().trained
        )


def test_a_one_dimensional_mask_is_not_a_part():
    with pytest.raises(ValueError, match="rollout, target position"):
        Part(name="flat", kind="span", mask=np.ones(5, dtype=bool))


# ---------------------------------------------------------------------------
# The generated invariance test (`tokenization`)
# ---------------------------------------------------------------------------


def _span_share(payload: InvariancePayload) -> float:
    """The reduction under test: the share of the credit measure that falls in span 0.

    This is what `by_segment` computes, isolated from the model so the property test is about the
    aggregation rather than about a network. `extra["contributions"]` is the per-token contribution
    to the objective and `extra["bounds"]` the span boundaries in token coordinates.
    """
    contributions = payload.extra["contributions"]
    bounds = payload.extra["bounds"]
    total = sum(sum(row) for row in contributions)
    inside = 0.0
    for row, (lo, hi) in zip(contributions, bounds):
        inside += sum(row[lo:hi])
    return inside / total


def _retokenisation_group() -> InvarianceGroup:
    """A `tokenization` generator that splits a token and splits its contribution with it.

    The default generator for this group duplicates a token id and leaves everything else
    alone, and its own docstring says an instrument that needs the real thing supplies its own. This
    is the real thing for a log-probability quantity, and the reason it is faithful rather than
    convenient is the chain rule: a finer tokeniser that splits `ab` into `a` then `b` satisfies
    `log pi(ab) = log pi(a) + log pi(b|a)` exactly, so the contribution of the original token is
    exactly the sum of the contributions of the two it becomes. The split point is drawn strictly
    inside span 0 so the boundary itself does not move, which is the case the claim is about.
    """

    def make(seed: int) -> GroupAction:
        def apply(p: InvariancePayload) -> InvariancePayload:
            rng = np.random.default_rng(seed)
            tokens = [list(row) for row in (p.tokens or [])]
            contributions = [list(row) for row in p.extra["contributions"]]
            bounds = [tuple(b) for b in p.extra["bounds"]]
            new_bounds = []
            for i, (lo, hi) in enumerate(bounds):
                if hi - lo < 2:
                    new_bounds.append((lo, hi))
                    continue
                cut = int(rng.integers(lo, hi - 1))
                tokens[i] = (
                    tokens[i][:cut] + [tokens[i][cut], tokens[i][cut]] + tokens[i][cut + 1 :]
                )
                share = contributions[i][cut]
                contributions[i] = (
                    contributions[i][:cut]
                    + [share * 0.25, share * 0.75]
                    + contributions[i][cut + 1 :]
                )
                new_bounds.append((lo, hi + 1))
            return p.replace(
                tokens=tokens,
                extra={"contributions": contributions, "bounds": new_bounds},
            )

        return GroupAction(
            name=f"split one token inside the span, seed={seed}",
            apply=apply,
            params={"seed": float(seed)},
            sample=lambda rng: make(rng.randrange(2**31)),
        )

    return InvarianceGroup(
        id="tokenization",
        generators=(make(0),),
        acts_on="tokens",
        admits=frozenset({"invariant"}),
        assertion="a span share summed over a text-defined span is invariant under retokenisation",
    )


def _invariance_payload() -> InvariancePayload:
    rng = np.random.default_rng(7)
    tokens = [list(range(10, 18)), list(range(30, 38)), list(range(50, 58))]
    contributions = [list(rng.uniform(0.1, 1.0, size=8)) for _ in tokens]
    return InvariancePayload(
        tokens=tokens,
        extra={"contributions": contributions, "bounds": [(2, 6), (2, 6), (2, 6)]},
    )


def test_the_span_share_is_invariant_under_retokenisation():
    """The generated property test for `credit.measure`'s declared group.

    This is the claim that licenses reporting shares by turn and span and refusing to report them
    per token: summing over a span defined on the text survives a change of tokenisation, because
    log-probability is additive under one. Sixty-four drawn group elements, exact to 1e-12.
    """
    report = check_invariance(
        CreditDisintegration(batch=None),
        _retokenisation_group(),
        _invariance_payload(),
        n=64,
        tol=1e-12,
        run=lambda _instrument, payload: _span_share(payload),
    )
    assert report.passed, report.render()
    assert report.n == 64


def test_a_per_token_share_is_not_invariant_which_is_why_it_is_not_reported():
    """The other half of the same claim, and the reason the instrument reports spans.

    Under the same transformation the share attributed to one *token index* moves, because the
    index no longer names the same piece of text. An instrument that reported per-token shares
    would be reporting a coordinate artifact, and `tokenization`'s assertion is exactly that a
    per-token quantity declares a normalisation or refuses.
    """
    payload = _invariance_payload()
    action = _retokenisation_group().generators[0].draw(Random(3))
    moved = action.apply(payload)

    def per_token_share(p, index=3):
        contributions = p.extra["contributions"]
        total = sum(sum(row) for row in contributions)
        return sum(row[index] for row in contributions) / total

    assert per_token_share(payload) != pytest.approx(per_token_share(moved), abs=1e-9)


# ---------------------------------------------------------------------------
# G2 refusals and arithmetic
# ---------------------------------------------------------------------------


def _turn(index, role, n=4, logprobs=True, tool=None, spans=()):
    return Turn(
        index=index,
        role=role,
        text=f"turn {index}",
        token_ids=tuple(range(200, 200 + n)),
        logprobs_sampling=tuple(-0.5 * (j + 1) for j in range(n)) if logprobs else None,
        loss_mask=tuple(role == "assistant" for _ in range(n)),
        tool_call=tool,
        spans=spans,
    )


def test_turn_mass_shares_are_non_negative_and_sum_to_one():
    trajectory = make_trajectory(
        id="t:1",
        task_ref="task:1",
        turns=[_turn(0, "user"), _turn(1, "assistant"), _turn(2, "assistant", n=8)],
        advantage=0.5,
    )
    payload = turn_mass([trajectory])
    assert not isinstance(payload, Refusal)
    assert all(v >= 0 for v in payload.by_turn.values())
    assert sum(payload.by_turn.values()) == pytest.approx(1.0)
    # The longer assistant turn carries more surprisal, so more of the mass.
    assert payload.by_turn["turn 2 (assistant)"] > payload.by_turn["turn 1 (assistant)"]
    assert 1.0 < payload.effective_turns <= 2.0


def test_the_signed_decomposition_keeps_the_advantage_sign():
    """A rollout that pushed against the step is visible, and the magnitude version hides it."""
    positive = make_trajectory(
        id="t:+", task_ref="task:1", turns=[_turn(0, "assistant")], advantage=1.0
    )
    negative = make_trajectory(
        id="t:-", task_ref="task:1", turns=[_turn(0, "assistant")], advantage=-1.0
    )
    payload = turn_mass([positive, negative])
    assert payload.by_turn["turn 0 (assistant)"] == pytest.approx(1.0)
    # The two cancel exactly in the signed view, so its share is undefined rather than zero.
    assert math.isnan(payload.signed_by_turn["turn 0 (assistant)"])


def test_turn_mass_refuses_with_no_trajectories():
    out = turn_mass([])
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE


def test_turn_mass_refuses_when_the_advantage_is_everywhere_zero():
    trajectory = make_trajectory(
        id="t:0", task_ref="task:1", turns=[_turn(0, "assistant")], advantage=0.0
    )
    out = turn_mass([trajectory])
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.BELOW_LOD
    assert "degenerate group rather than a measurement problem" in out.remedy


def test_a_trajectory_with_no_advantage_contributes_nothing_and_is_counted():
    """A rollout whose every reward function abstained is masked, and that is the record working."""
    scored = make_trajectory(
        id="t:1", task_ref="task:1", turns=[_turn(0, "assistant")], advantage=1.0
    )
    abstained = make_trajectory(
        id="t:2", task_ref="task:1", turns=[_turn(0, "assistant")], advantage=None
    )
    payload = turn_mass([scored, abstained])
    assert not isinstance(payload, Refusal)
    assert any("carry no advantage" in note for note in payload.notes)


def test_the_tool_call_share_follows_the_tool_name():
    trajectory = make_trajectory(
        id="t:tool",
        task_ref="task:1",
        turns=[
            _turn(0, "assistant", tool=ToolCall(name="bash")),
            _turn(1, "assistant", n=8, tool=ToolCall(name="bash")),
            _turn(2, "assistant", spans=(Span(start=0, end=2, kind="answer"),)),
        ],
        advantage=1.0,
    )
    payload = turn_mass([trajectory])
    assert set(payload.by_tool_call) == {"bash"}
    assert payload.by_tool_call["bash"] == pytest.approx(payload.tool_call_share)
    assert 0.0 < payload.by_span_kind["answer"] < 1.0


# ---------------------------------------------------------------------------
# G3 edge cases
# ---------------------------------------------------------------------------


def test_the_root_value_is_the_group_mean_which_is_grpo_s_own_baseline():
    payload = implicit_prm([[1], [2], [3], [4]], [1.0, 0.0, 0.5, 0.5])
    assert payload.root_value == pytest.approx(0.5)
    # With one token each, the whole outcome advantage is the first and only increment.
    assert payload.process_reward == pytest.approx([0.5, -0.5, 0.0, 0.0])


def test_mismatched_completions_and_rewards_refuse():
    out = implicit_prm([[1], [2]], [1.0])
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE


def test_a_non_finite_reward_refuses_rather_than_counting_as_zero():
    """The whole argument: grader failure is a distinct outcome and never a real zero."""
    out = implicit_prm([[1], [2]], [1.0, float("nan")])
    assert isinstance(out, Refusal)
    assert "abstained" in out.remedy


def test_agreement_is_reported_against_a_supplied_reference():
    payload = implicit_prm(
        [[1, 2, 3], [1, 2, 9], [1, 5, 6], [1, 5, 7]],
        [1.0, 0.0, 0.5, -0.5],
        compare_to={
            "oracle": [0.0, 0.25, 0.5] + [0.0, 0.25, -0.5] + [0.0, -0.25, 0.5] + [0.0, -0.25, -0.5]
        },
    )
    assert payload.agreement["oracle"] == pytest.approx(1.0)


def test_a_misaligned_reference_is_named_rather_than_compared():
    payload = implicit_prm([[1, 2], [1, 3]], [1.0, 0.0], compare_to={"short": [0.0]})
    assert "short" not in payload.agreement
    assert any("align it to `positions`" in note for note in payload.notes)


# ---------------------------------------------------------------------------
# The record bridge and the SkyRL reader
# ---------------------------------------------------------------------------


def test_batch_from_trajectories_records_what_it_had_to_assume():
    """Tokenising from text and inferring a mask are both defaults, and both are written down."""

    class _Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) % 97 for c in text][:6] or [1]}

    trajectory = make_trajectory(
        id="t:1",
        task_ref="task:1",
        turns=[
            Turn(index=0, role="user", text="prompt here"),
            Turn(index=1, role="assistant", text="answer here"),
        ],
        advantage=0.75,
    )
    built = batch_from_trajectories(_Tokenizer(), [trajectory])
    assert not isinstance(built, Refusal)
    batch, segments = built
    assert batch.n_rollouts == 1
    assert [s.name for s in segments] == ["turn 0 (user)", "turn 1 (assistant)"]
    assert any("tokenised from their text" in n for n in batch.notes)
    assert any("inferred from the turn role" in n for n in batch.notes)


def test_batch_from_trajectories_refuses_when_nothing_carries_an_advantage():
    class _Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2, 3]}

    trajectory = make_trajectory(
        id="t:1",
        task_ref="task:1",
        turns=[Turn(index=0, role="assistant", text="x")],
        advantage=None,
    )
    out = batch_from_trajectories(_Tokenizer(), [trajectory])
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "masked and never receives an advantage" in out.remedy


def _write_dump(path, batch_dict, metadata):
    created = []
    for name in ("skyrl", "skyrl.backends", "skyrl.backends.skyrl_train", SKYRL_MODULE):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
            created.append(name)
    module = sys.modules[SKYRL_MODULE]

    class TrainingInputBatch(dict):
        pass

    def _rebuild_tensor_batch(cls, state):  # pragma: no cover
        raise RuntimeError("not called on the read path")

    TrainingInputBatch.__module__ = SKYRL_MODULE
    TrainingInputBatch.__qualname__ = "TrainingInputBatch"
    _rebuild_tensor_batch.__module__ = SKYRL_MODULE
    _rebuild_tensor_batch.__qualname__ = "_rebuild_tensor_batch"
    module.TrainingInputBatch = TrainingInputBatch
    module._rebuild_tensor_batch = _rebuild_tensor_batch

    class _Shim:
        def __reduce__(self):
            return (
                _rebuild_tensor_batch,
                (
                    TrainingInputBatch,
                    {
                        "batch_dict": {
                            k: (
                                None
                                if v is None
                                else {
                                    "format": "numpy",
                                    "data": v.tobytes(),
                                    "shape": v.shape,
                                    "dtype": str(v.dtype),
                                }
                            )
                            for k, v in batch_dict.items()
                        },
                        "batch_size": 2,
                        "device": None,
                        "metadata": metadata,
                    },
                ),
            )

    with open(path, "wb") as handle:
        pickle.dump(_Shim(), handle)
    for name in created:
        sys.modules.pop(name, None)


def test_a_dump_without_advantages_refuses_and_names_where_they_are_added(tmp_path):
    path = tmp_path / "early.pkl"
    ids = np.arange(12, dtype=np.int64).reshape(2, 6)
    _write_dump(path, {"sequences": ids, "loss_mask": np.ones_like(ids)}, {})
    dump = read_skyrl_dump(path)
    assert dump.has_advantages is False
    out = batch_from_skyrl(dump)
    assert isinstance(out, Refusal)
    assert "compute_advantages_and_returns" in out.remedy


def test_a_dump_without_a_mask_refuses_because_a_mask_is_not_inferable(tmp_path):
    path = tmp_path / "nomask.pkl"
    ids = np.arange(12, dtype=np.int64).reshape(2, 6)
    _write_dump(path, {"sequences": ids, "advantages": np.ones((2, 6))}, {})
    out = batch_from_skyrl(read_skyrl_dump(path))
    assert isinstance(out, Refusal)
    assert "decides what 'per token' means" in out.detail or "per token" in out.remedy


def test_group_recovery_hashes_the_prompt_the_response_mask_excludes(tmp_path):
    path = tmp_path / "groups.pkl"
    ids = np.array([[1, 1, 9, 9], [1, 1, 8, 8], [2, 2, 7, 7], [2, 2, 6, 6]], dtype=np.int64)
    response = np.array([[0, 0, 1, 1]] * 4, dtype=np.int64)
    _write_dump(
        path,
        {"sequences": ids, "response_mask": response, "advantages": np.ones((4, 4))},
        {},
    )
    dump = read_skyrl_dump(path)
    uids = dump.recovered_uids
    assert len(set(uids)) == 2
    assert uids[0] == uids[1] and uids[2] == uids[3] and uids[0] != uids[2]


def test_credit_increment_refuses_when_the_report_is_not_rollout_wise():
    from reward_lens.policy.credit import CreditReport

    report = CreditReport(parts=["turn 0"], kinds=["turn"], projected_share=[1.0])
    out = credit_increment(report, texts=["a", "b"], proxy=[1.0, 2.0], n_tokens=[3, 4])
    assert isinstance(out, Refusal)
    assert "by_rollout" in out.remedy

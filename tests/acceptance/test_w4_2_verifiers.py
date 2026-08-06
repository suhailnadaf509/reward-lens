"""Acceptance: *a ``TrajectoryStep`` stream converts to ``Trajectory`` with logprobs, routing
and advantage preserved, field by field, asserted.*

Field by field is meant literally. ``TrajectoryStep`` has nine fields and ``TrajectoryStepTokens``
has ten, and every one of the nineteen is asserted here to land somewhere in the record or to be
dropped on purpose with the reason written down. A field silently lost is the failure this package
exists to prevent, so the tests below are mostly not about behaviour: they are about the presence
of a value at a named location.

The fixture is the shape ``verifiers`` itself produces. ``MultiTurnEnv.add_model_response`` builds
every ``TrajectoryStep`` in the library at one place, and ``parse_response_tokens`` builds every
``TrajectoryStepTokens``; the two-step stream in `stream` below is what those two functions emit
for a two-turn rollout, with the routing, multimodal and attribution fields filled because no test
fixture in the framework fills them and a converter that has never seen them is untested on them.

Two facts from E7 are asserted directly, because both change what a downstream number
means. ``score_group`` mean-centres with no standard-deviation division, so the amplifier
mechanism is absent rather than weak. And ``rubrics/rubric.py:204-217`` substitutes ``0.0`` for any
exception, so a zero is ambiguous unless something upstream of the rubric observed the call.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from reward_lens.record.compaction import CompactionEvent
from reward_lens.record.provenance import check_tiling, policy_mixture
from reward_lens.record.reader import open_run
from reward_lens.record.scores import Leaf, ScoreContext, WeightedSum, evaluate
from reward_lens.record.tensors import AbsentRef, StoredRef, TensorStore, resolve
from reward_lens.record.writer import RecordWriter
from reward_lens.tap.adapters.verifiers import (
    CONVERTER_FINDINGS,
    NOT_FILLED,
    SILENT_ZERO_SITE,
    STEP_FIELD_MAP,
    TOKEN_FIELD_MAP,
    TOKEN_FIELDS,
    TRAJECTORY_STEP_FIELDS,
    VerifiersConverter,
    convert_trajectory,
    estimator_spec,
    group_by_trajectory,
)
from reward_lens.tap.contract import CallOutcome, GraderCall

# ---------------------------------------------------------------------------
# The subject: what verifiers' own add_model_response emits, for two turns
# ---------------------------------------------------------------------------

ROUTING = np.arange(12, dtype=np.uint8).reshape(3, 4)

PROMPT_0 = [
    {"role": "system", "content": "You solve puzzles."},
    {"role": "user", "content": "What is 2+2?"},
]
COMPLETION_0 = [{"role": "assistant", "content": "Let me check."}]
ENV_RESPONSE = [{"role": "user", "content": "Use the tool."}]
COMPLETION_1 = [{"role": "assistant", "content": "4"}]


def _step_0() -> dict:
    return {
        "prompt": list(PROMPT_0),
        "completion": list(COMPLETION_0),
        "response": {
            "id": "resp_0",
            "created": 1772058897,
            "model": "Qwen/Qwen3-30B-A3B",
            "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            "message": {"finish_reason": "stop", "is_truncated": False},
        },
        "tokens": {
            "prompt_ids": [11, 12, 13, 14],
            "prompt_mask": [0, 0, 0, 0],
            "completion_ids": [21, 22, 23],
            "completion_mask": [1, 1, 1],
            "completion_logprobs": [-0.11, -0.22, -0.33],
            "overlong_prompt": False,
            "is_truncated": False,
            "routed_experts": {
                "data": ROUTING.tobytes(),
                "shape": [3, 4],
                "start": 4,
                "dtype": "uint8",
            },
            "multi_modal_data": {"pixel_values": "<tensor>", "n": 1},
            "prompt_attribution": {
                "message_indices": [0, 0, 1, 1],
                "message_roles": ["system", "user"],
            },
        },
        "reward": 0.75,
        "advantage": 0.25,
        "is_truncated": False,
        "trajectory_id": "traj-abc",
        "extras": {"turn": 0, "note": "first action"},
    }


def _step_1() -> dict:
    return {
        "prompt": PROMPT_0 + COMPLETION_0 + ENV_RESPONSE,
        "completion": list(COMPLETION_1),
        "response": {
            "id": "resp_1",
            "created": 1772058912,
            "model": "Qwen/Qwen3-30B-A3B",
            "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
            "message": {"finish_reason": "length", "is_truncated": True},
        },
        "tokens": {
            # the previous prompt (11,12,13,14) then the previous completion (21,22,23) then
            # the environment's two new tokens: the prefix relation MultiTurnEnv maintains
            "prompt_ids": [11, 12, 13, 14, 21, 22, 23, 31, 32],
            "prompt_mask": [0, 0, 0, 0, 1, 1, 1, 0, 0],
            "completion_ids": [41],
            "completion_mask": [1],
            "completion_logprobs": [-0.44],
            "overlong_prompt": False,
            "is_truncated": True,
            "routed_experts": None,
        },
        "reward": 0.75,
        "advantage": 0.25,
        "is_truncated": True,
        "trajectory_id": "traj-abc",
        "extras": {},
    }


@pytest.fixture
def stream() -> list[dict]:
    return [_step_0(), _step_1()]


@pytest.fixture
def store(tmp_path: Path) -> TensorStore:
    return TensorStore(tmp_path / "tensors")


@pytest.fixture
def converted(stream: list[dict], store: TensorStore):
    converter = VerifiersConverter(run_id="w42", store=store, weights={"correct": 1.0, "fmt": 0.5})
    trajectory = converter.trajectory(
        stream,
        metrics={"correct": 1.0, "fmt": 0.5},
        task_ref="task:puzzle",
    )
    return converter, trajectory


def _context(trajectory) -> tuple:
    """The context turn and the assistant turn of each step, in order."""
    return tuple(trajectory.turns)


# ---------------------------------------------------------------------------
# The clause, part 1: every TrajectoryStep field lands
# ---------------------------------------------------------------------------


def test_the_field_map_covers_every_declared_field_and_nothing_else():
    """The map is the contract, so it is checked against the field lists rather than trusted."""
    assert set(STEP_FIELD_MAP) == set(TRAJECTORY_STEP_FIELDS)
    assert set(TOKEN_FIELD_MAP) == set(TOKEN_FIELDS)
    assert len(TRAJECTORY_STEP_FIELDS) == 9
    assert len(TOKEN_FIELDS) == 10
    for name, where in {**STEP_FIELD_MAP, **TOKEN_FIELD_MAP}.items():
        assert where.strip(), f"{name} has no recorded destination"


def test_prompt_lands_as_the_context_turns(converted):
    _, trajectory = converted
    turns = _context(trajectory)
    # step 0's whole prompt is the opening context turn
    assert [m["content"] for m in turns[0].extra["messages"]] == [
        "You solve puzzles.",
        "What is 2+2?",
    ]
    assert turns[0].extra["verifiers_roles"] == ["system", "user"]
    assert "You solve puzzles." in turns[0].text
    # step 1's prompt is the previous prompt, the previous completion and the environment's reply,
    # so only the environment's reply is new
    assert [m["content"] for m in turns[2].extra["messages"]] == ["Use the tool."]
    assert turns[2].role == "environment", (
        "a user-role message that arrived after the opening prompt is what env_response returned"
    )
    assert turns[2].extra["verifiers_roles"] == ["user"], "the framework's own role is kept"


def test_completion_lands_as_the_assistant_turns(converted):
    _, trajectory = converted
    turns = _context(trajectory)
    assert turns[1].role == "assistant"
    assert turns[3].role == "assistant"
    assert [m["content"] for m in turns[1].extra["messages"]] == ["Let me check."]
    assert [m["content"] for m in turns[3].extra["messages"]] == ["4"]


def test_response_lands_and_becomes_the_policy_version(converted):
    _, trajectory = converted
    turns = _context(trajectory)
    assert turns[1].extra["response"] == {
        "id": "resp_0",
        "created": 1772058897,
        "model": "Qwen/Qwen3-30B-A3B",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
    }
    assert turns[3].extra["response"]["finish_reason"] == "length"
    assert [str(p.policy_version) for p in trajectory.provenance] == ["Qwen/Qwen3-30B-A3B"]


def test_reward_lands_on_every_turn_and_on_the_trajectory(converted):
    _, trajectory = converted
    turns = _context(trajectory)
    assert turns[1].step_score == 0.75
    assert turns[3].step_score == 0.75
    assert trajectory.features["verifiers_realised_reward"] == 0.75
    assert trajectory.features["verifiers_step_score_uniform"] == 1.0, (
        "a uniform column is what score_group's back-fill at rubric.py:410-414 produces"
    )


def test_advantage_is_preserved(converted):
    """The clause names advantage explicitly. It is on every turn and on the trajectory."""
    _, trajectory = converted
    turns = _context(trajectory)
    assert turns[1].step_advantage == 0.25
    assert turns[3].step_advantage == 0.25
    assert trajectory.advantage == 0.25


def test_a_disagreeing_advantage_column_does_not_become_a_trajectory_advantage(stream):
    """Per-step advantages that differ are not a trajectory-level number, so there is not one."""
    stream[1]["advantage"] = -0.5
    trajectory = convert_trajectory(stream)
    assert [t.step_advantage for t in trajectory.turns if t.role == "assistant"] == [0.25, -0.5]
    assert trajectory.advantage is None
    assert trajectory.features["verifiers_step_advantage_uniform"] == 0.0


def test_step_is_truncated_lands_on_the_assistant_turn(converted):
    _, trajectory = converted
    turns = _context(trajectory)
    assert turns[1].truncated is False
    assert turns[3].truncated is True


def test_trajectory_id_is_kept_verbatim(converted):
    """A content hash here would break the join back to the framework's own artifact."""
    _, trajectory = converted
    assert str(trajectory.id) == "traj-abc"


def test_extras_lands(converted):
    _, trajectory = converted
    turns = _context(trajectory)
    assert turns[1].extra["step_extras"] == {"turn": 0, "note": "first action"}
    assert turns[1].extra["step_extras_is_repr"] is False
    assert "step_extras" not in turns[3].extra, "an empty extras dict writes no key"


# ---------------------------------------------------------------------------
# The clause, part 2: every TrajectoryStepTokens field lands
# ---------------------------------------------------------------------------


def test_prompt_ids_land_as_the_delta_not_the_re_render(converted):
    """Step 1's prompt_ids are the whole re-rendered prefix; only the new tail is a new turn."""
    _, trajectory = converted
    turns = _context(trajectory)
    assert turns[0].token_ids == (11, 12, 13, 14)
    assert turns[2].token_ids == (31, 32)
    assert trajectory.n_tokens == 4 + 3 + 2 + 1, "no token is counted twice"


def test_prompt_mask_lands_as_the_loss_mask(converted):
    _, trajectory = converted
    turns = _context(trajectory)
    assert turns[0].loss_mask == (False, False, False, False)
    assert turns[2].loss_mask == (False, False)
    assert turns[0].n_unmasked == 0


def test_completion_ids_and_mask_land(converted):
    _, trajectory = converted
    turns = _context(trajectory)
    assert turns[1].token_ids == (21, 22, 23)
    assert turns[1].loss_mask == (True, True, True)
    assert turns[3].token_ids == (41,)
    assert turns[3].loss_mask == (True,)


def test_completion_logprobs_are_preserved_exactly(converted):
    """The clause names logprobs explicitly. Exact equality, not approximate."""
    _, trajectory = converted
    turns = _context(trajectory)
    assert turns[1].logprobs_sampling == (-0.11, -0.22, -0.33)
    assert turns[3].logprobs_sampling == (-0.44,)
    assert len(turns[1].logprobs_sampling) == len(turns[1].token_ids)


def test_logprobs_train_is_left_none_on_purpose(converted):
    """verifiers has one logprob stream. Filling both would make E6 report a mismatch of zero."""
    _, trajectory = converted
    for turn in trajectory.turns:
        assert turn.logprobs_train is None
        assert turn.logprob_gap() is None
    assert "Turn.logprobs_train" in NOT_FILLED


def test_overlong_prompt_lands_on_the_context_turn(converted):
    _, trajectory = converted
    turns = _context(trajectory)
    assert turns[0].overlong_prompt is False
    assert turns[2].overlong_prompt is False


def test_an_overlong_prompt_is_recorded_as_a_prompt_event_not_a_truncation(stream):
    """verifiers keeps the two apart, and so does the record: the completion is emptied."""
    stream[0]["tokens"]["overlong_prompt"] = True
    stream[0]["tokens"]["completion_ids"] = []
    stream[0]["tokens"]["completion_mask"] = []
    stream[0]["tokens"]["completion_logprobs"] = []
    trajectory = convert_trajectory([stream[0]])
    assert trajectory.turns[0].overlong_prompt is True
    assert trajectory.turns[1].token_ids == (), "no completion tokens, which is not None"
    assert trajectory.turns[1].n_tokens == 0


def test_the_two_is_truncated_flags_are_kept_apart(converted):
    """A max_seq_len clip and an engine finish_reason are different events with one name each."""
    _, trajectory = converted
    turns = _context(trajectory)
    assert turns[1].extra["tokens_is_truncated"] is False
    assert turns[3].extra["tokens_is_truncated"] is True
    assert turns[3].truncated is True


def test_routed_experts_is_preserved_byte_for_byte(converted, store):
    """The clause names routing explicitly. It round-trips through the tensor store unchanged."""
    _, trajectory = converted
    turns = _context(trajectory)
    ref = turns[1].tensors["routed_experts"]
    assert isinstance(ref, StoredRef)
    assert ref.shape == (3, 4)
    assert ref.dtype == "uint8"
    store.flush()
    back = resolve(ref, store=store)
    assert isinstance(back, np.ndarray)
    assert np.array_equal(back, ROUTING)
    # start is not on a StoredRef and it is the offset the routing begins at, so it is kept
    assert turns[1].extra["routed_experts_start"] == 4
    assert turns[1].extra["routed_experts_shape"] == [3, 4]
    assert trajectory.capture is not None
    assert trajectory.capture.spec.include_routing is True
    assert "routed_experts/turn1" in trajectory.capture.tensors


def test_routing_without_a_store_is_an_absence_that_says_the_payload_was_there(stream):
    """Never a silent drop. The AbsentRef distinguishes 'not persisted' from 'not captured'."""
    trajectory = convert_trajectory(stream)
    ref = trajectory.turns[1].tensors["routed_experts"]
    assert isinstance(ref, AbsentRef)
    assert "was present" in ref.detail
    assert "shape=(3, 4)" in ref.detail
    assert "TensorStore" in ref.remedy, "the remedy is an instruction, not a description"
    assert trajectory.turns[1].extra["routed_experts_shape"] == [3, 4]


def test_multi_modal_data_lands_and_says_when_it_was_coerced(converted):
    _, trajectory = converted
    turns = _context(trajectory)
    assert turns[1].extra["multi_modal_data"] == {"pixel_values": "<tensor>", "n": 1}
    assert turns[1].extra["multi_modal_data_is_repr"] is False
    assert "multi_modal_data" not in turns[3].extra, (
        "the field is NotRequired and absent on step 1, so no key is written"
    )


def test_prompt_attribution_lands_on_the_context_turn(converted):
    _, trajectory = converted
    turns = _context(trajectory)
    assert turns[0].extra["prompt_attribution"] == {
        "message_indices": [0, 0, 1, 1],
        "message_roles": ["system", "user"],
    }
    assert "prompt_attribution" not in turns[2].extra


def test_an_unserialisable_payload_is_capped_and_flagged_rather_than_crashing_the_writer(stream):
    stream[0]["tokens"]["multi_modal_data"] = {"pixel_values": object()}
    trajectory = convert_trajectory(stream)
    extra = trajectory.turns[1].extra
    assert isinstance(extra["multi_modal_data"]["pixel_values"], str)
    assert extra["multi_modal_data_is_repr"] is True
    json.dumps(trajectory.turns[1].__canonical__())


# ---------------------------------------------------------------------------
# The clause, part 3: nothing is dropped without a written reason
# ---------------------------------------------------------------------------


def test_every_deliberate_omission_carries_its_reason():
    """Five record fields a verifiers stream cannot fill. Each says why, in a sentence."""
    assert set(NOT_FILLED) == {
        "Turn.logprobs_train",
        "Trajectory.advantage_tokens",
        "Trajectory.labels",
        "SegmentProvenance.staleness_steps",
        "OptimizerTelemetry",
    }
    for field, reason in NOT_FILLED.items():
        assert len(reason) > 60, f"{field}'s reason is too short to be a reason"


def test_the_omissions_are_actually_omitted(converted):
    _, trajectory = converted
    assert trajectory.advantage_tokens is None
    assert dict(trajectory.labels) == {}
    for turn in trajectory.turns:
        assert turn.logprobs_train is None


def test_staleness_is_declared_and_says_it_is_a_declaration(converted):
    _, trajectory = converted
    segment = trajectory.provenance[0]
    assert segment.staleness_steps == 0
    assert "not a measurement" in segment.sampling.extra["staleness_provenance"]
    assert "no checkpoint identity" in segment.sampling.extra["policy_version_is"]


# ---------------------------------------------------------------------------
# E7, fact one: the amplifier mechanism is absent, not weak
# ---------------------------------------------------------------------------


def test_the_estimator_spec_records_mean_centring_with_no_std_division():
    spec = estimator_spec()
    assert spec.family == "verifiers/score_group"
    assert spec.group_centred is True
    assert spec.std_normalised is False
    assert spec.std_epsilon is None, "there is no epsilon; 0.0 would say there is one and it is 0"
    assert spec.z_scored is False, "E2 and E4 read this to say the mechanism is absent"
    assert spec.clip_low is None and spec.clip_high is None
    assert spec.advantage_whitening is False
    assert "absent" in spec.extra["amplification_mechanism"]
    assert spec.extra["advantage_formula"] == (
        "advantage_i = reward_i - mean(rewards over the group)"
    )


def test_the_recorded_family_resolves_to_the_verified_defaults_row():
    """The converter and ``measure/estimator/spec.py`` must not drift apart."""
    from reward_lens.measure.estimator.spec import _FAMILY_PREFIXES, FRAMEWORK_DEFAULTS

    spec = estimator_spec()
    row = next(k for prefix, k in _FAMILY_PREFIXES if spec.family.startswith(prefix))
    defaults = FRAMEWORK_DEFAULTS[row]
    assert defaults["group_centred"] == spec.group_centred
    assert defaults["std_normalised"] == spec.std_normalised
    assert defaults["std_epsilon"] == spec.std_epsilon
    assert defaults["clip_low"] == spec.clip_low
    assert defaults["advantage_whitening"] == spec.advantage_whitening


def test_the_group_reproduces_score_groups_own_arithmetic():
    """advantage_i = reward_i - mean(rewards), and the converted advantages equal it exactly."""
    rewards = [1.0, 0.25, 0.0, 0.75]
    mean = sum(rewards) / len(rewards)
    rollouts = [
        {
            "trajectory_id": f"t{i}",
            "example_id": 7,
            "reward": r,
            "advantage": r - mean,
            "metrics": {"correct": r},
            "trajectory": [
                {
                    "prompt": [{"role": "user", "content": "q"}],
                    "completion": [{"role": "assistant", "content": "a"}],
                    "response": {"model": "m", "message": {"finish_reason": "stop"}},
                    "tokens": {
                        "prompt_ids": [1],
                        "prompt_mask": [0],
                        "completion_ids": [2],
                        "completion_mask": [1],
                        "completion_logprobs": [-0.5],
                        "overlong_prompt": False,
                        "is_truncated": False,
                        "routed_experts": None,
                    },
                    "reward": r,
                    "advantage": r - mean,
                    "is_truncated": False,
                    "trajectory_id": f"t{i}",
                    "extras": {},
                }
            ],
        }
        for i, r in enumerate(rewards)
    ]
    converter = VerifiersConverter(run_id="grp", weights={"correct": 1.0})
    group = converter.group(rollouts, task_ref="task:7")
    assert [t.advantage for t in group.trajectories] == [r - mean for r in rewards]
    assert sum(t.advantage for t in group.trajectories) == pytest.approx(0.0)
    assert group.group_stats.k == 4
    assert group.group_stats.mean == pytest.approx(mean)


def test_a_degenerate_group_is_flagged_at_std_zero_because_there_is_no_epsilon():
    """verifiers divides by nothing, so the degenerate boundary is exactly std == 0."""
    rollouts = [
        {"trajectory_id": f"t{i}", "reward": 0.5, "advantage": 0.0, "metrics": {"c": 0.5}}
        for i in range(4)
    ]
    converter = VerifiersConverter(weights={"c": 1.0})
    group = converter.group(rollouts)
    assert group.group_stats.std == 0.0
    assert group.group_stats.degenerate is True

    rollouts[0]["reward"] = 0.5000001
    group2 = VerifiersConverter(weights={"c": 1.0}).group(rollouts)
    assert group2.group_stats.std > 0.0
    assert group2.group_stats.degenerate is False, (
        "any positive spread is a live group here; a tolerance would be invented"
    )


# ---------------------------------------------------------------------------
# E7, fact two: the silent zero
# ---------------------------------------------------------------------------


def test_without_tap_records_a_zero_is_bounded_rather_than_claimed(stream):
    """The record cannot tell a genuine zero from a swallowed exception, so it does not pretend to."""
    trajectory = convert_trajectory(stream, metrics={"correct": 1.0, "fmt": 0.0, "len": 0.0})
    leaves = trajectory.scores.children
    assert [leaf.abstained for leaf in leaves] == [False, False, False], (
        "marking every zero abstained would invent the defect rather than measure it"
    )
    assert trajectory.features["verifiers_unresolved_zeros"] == 2.0
    assert trajectory.features["verifiers_known_abstentions"] == 0.0
    ambiguous = [leaf.grader_call.facets["zero_is_ambiguous"] for leaf in leaves]
    assert ambiguous == [False, True, True]
    assert (
        SILENT_ZERO_SITE in leaves[0].grader_call.facets["framework_substitutes_zero_on_exception"]
    )


def test_with_tap_records_the_silent_zero_is_exact(stream):
    """Upstream of the rubric the outcome is observable, and then the record says so exactly."""
    raised = GraderCall(
        seq=3,
        run_id="r",
        grader="fmt",
        outcome=CallOutcome.RAISED,
        inner_ns=4_000_000,
        added_ns=900,
        error_type="ZeroDivisionError",
        error_message="division by zero",
    )
    returned = GraderCall(
        seq=4,
        run_id="r",
        grader="correct",
        outcome=CallOutcome.RETURNED,
        inner_ns=1000,
        added_ns=90,
    )
    trajectory = convert_trajectory(
        stream, metrics={"correct": 1.0, "fmt": 0.0}, calls=[raised, returned]
    )
    fmt = next(leaf for leaf in trajectory.scores.children if leaf.name == "fmt")
    assert fmt.abstained is True
    assert fmt.value == 0.0, "the framework used a real number and the record keeps it"
    assert fmt.silent_zero is True, "B4's numerator, exactly rather than as a bound"
    assert fmt.substituted == 0.0
    assert fmt.grader_call.error_type == "ZeroDivisionError"
    assert fmt.grader_call.outcome == "raised"
    assert trajectory.features["verifiers_known_abstentions"] == 1.0
    assert trajectory.features["verifiers_unresolved_zeros"] == 0.0


def test_the_bound_is_never_below_the_truth(stream):
    """The unresolved count plus the known count is an upper bound on the real silent-zero count."""
    trajectory = convert_trajectory(stream, metrics={"a": 0.0, "b": 0.0, "c": 1.0})
    bound = (
        trajectory.features["verifiers_unresolved_zeros"]
        + trajectory.features["verifiers_known_abstentions"]
    )
    truth_upper = sum(1 for leaf in trajectory.scores.children if leaf.value == 0.0)
    assert bound >= truth_upper


def test_an_abstaining_term_makes_the_sum_nan_rather_than_smaller(stream):
    """A total missing a term is not a smaller total. This is the record's rule, not TRL's."""
    raised = GraderCall(
        seq=1, run_id="r", grader="fmt", outcome=CallOutcome.RAISED, inner_ns=1, added_ns=1
    )
    trajectory = convert_trajectory(
        stream,
        metrics={"correct": 1.0, "fmt": 0.0},
        calls=[raised],
        weights={"correct": 1.0, "fmt": 0.5},
    )
    value = evaluate(trajectory.scores, ScoreContext())
    assert np.isnan(value)
    assert trajectory.features["verifiers_realised_reward"] == 0.75, (
        "what the framework actually used is kept beside the tree, because the record has to be "
        "able to show that the run used a different number"
    )


def test_the_weights_are_on_the_tree_and_not_folded_into_the_leaves(stream):
    trajectory = convert_trajectory(
        stream, metrics={"correct": 1.0, "fmt": 1.0}, weights={"correct": 1.0, "fmt": 0.5}
    )
    assert isinstance(trajectory.scores, WeightedSum)
    assert trajectory.scores.weights == (1.0, 0.5)
    assert [leaf.value for leaf in trajectory.scores.children] == [1.0, 1.0]
    assert evaluate(trajectory.scores, ScoreContext()) == pytest.approx(1.5)


def test_a_single_unweighted_reward_function_is_its_own_composition(stream):
    trajectory = convert_trajectory(stream, metrics={"correct": 1.0})
    assert isinstance(trajectory.scores, Leaf)
    assert trajectory.scores.name == "correct"


# ---------------------------------------------------------------------------
# Structure: provenance, compaction, and the prefix check
# ---------------------------------------------------------------------------


def test_the_provenance_tiles_the_turns(converted):
    _, trajectory = converted
    check_tiling(trajectory.provenance, len(trajectory.turns), where="acceptance")
    assert trajectory.provenance[0].turn_range == (0, 4)
    assert policy_mixture(trajectory.provenance).singular is True


def test_two_served_models_produce_two_segments(stream):
    """A rollout that resumed under a different served model is not singular, and says so."""
    stream[1]["response"]["model"] = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    trajectory = convert_trajectory(stream)
    assert [p.turn_range for p in trajectory.provenance] == [(0, 2), (2, 4)]
    assert policy_mixture(trajectory.provenance).singular is False
    check_tiling(trajectory.provenance, len(trajectory.turns), where="acceptance")


def test_a_shrinking_prefix_is_recorded_as_a_compaction(stream):
    """After a prefix rewrite the importance ratio is undefined, so it is a record, not a note."""
    stream[1]["tokens"]["prompt_ids"] = [11, 12, 31, 32]
    stream[1]["tokens"]["prompt_mask"] = [0, 0, 0, 0]
    converter = VerifiersConverter()
    trajectory = converter.trajectory(stream)
    assert len(trajectory.compaction) == 1
    event = trajectory.compaction[0]
    assert isinstance(event, CompactionEvent)
    assert event.tokens_before == 7 and event.tokens_after == 4
    assert event.method == "prefix_rewrite"
    assert event.is_lossless is False
    assert trajectory.was_compacted is True
    assert converter.report.compaction_events == 1


def test_a_prefix_that_does_not_match_records_none_rather_than_a_plausible_slice(stream):
    """A chat template that re-rendered the prefix makes the delta unknowable. None says that."""
    stream[1]["tokens"]["prompt_ids"] = [99, 98, 97, 96, 95, 94, 93, 31, 32]
    stream[1]["tokens"]["prompt_mask"] = [0] * 9
    converter = VerifiersConverter()
    trajectory = converter.trajectory(stream)
    context = trajectory.turns[2]
    assert context.token_ids is None
    assert context.extra["prompt_ids_not_prefix_stable"] is True
    assert context.extra["step_prompt_ids_len"] == 9
    assert converter.report.non_prefix_steps == 1
    assert len(trajectory.compaction) == 0, "the same length is not a rewrite, only a re-render"


def test_no_compaction_is_declared_from_the_tokens_and_not_assumed(stream):
    from reward_lens.core.envelope import RegimeCondition

    converter = VerifiersConverter()
    converter.trajectory(stream)
    regime = converter.regime()
    assert regime.declared[RegimeCondition.NO_COMPACTION] is True
    assert "checked on 2 of 2 steps" in regime.notes[RegimeCondition.NO_COMPACTION]
    assert RegimeCondition.NEAR_POLICY not in regime.declared, (
        "staleness cannot be measured from this framework, so the condition is not claimed"
    )
    assert "staleness cannot be measured" in regime.notes[RegimeCondition.NEAR_POLICY]


def test_no_compaction_is_not_declared_when_there_were_no_tokens_to_check(stream):
    from reward_lens.core.envelope import RegimeCondition

    for step in stream:
        step["tokens"] = None
    converter = VerifiersConverter()
    converter.trajectory(stream)
    regime = converter.regime()
    assert RegimeCondition.NO_COMPACTION not in regime.declared, "not looking is not a pass"
    assert "nothing to compare" in regime.notes[RegimeCondition.NO_COMPACTION]


def test_a_stream_with_no_tokens_carries_none_rather_than_empty_arrays(stream):
    for step in stream:
        step["tokens"] = None
    converter = VerifiersConverter()
    trajectory = converter.trajectory(stream)
    for turn in trajectory.turns:
        assert turn.token_ids is None
        assert turn.logprobs_sampling is None
    assert converter.report.steps_without_tokens == 2
    assert trajectory.n_tokens == 0


def test_a_flat_stream_is_split_by_trajectory_id(stream):
    other = _step_0()
    other["trajectory_id"] = "traj-xyz"
    grouped = group_by_trajectory([stream[0], other, stream[1]])
    assert sorted(grouped) == ["traj-abc", "traj-xyz"]
    assert len(grouped["traj-abc"]) == 2
    assert len(grouped["traj-xyz"]) == 1


# ---------------------------------------------------------------------------
# The record survives a write and a read
# ---------------------------------------------------------------------------


def test_the_converted_run_round_trips_through_the_writer(stream, tmp_path):
    converter = VerifiersConverter(
        run_id="w42-roundtrip",
        weights={"correct": 1.0, "fmt": 0.5},
        framework_version="edafab85",
    )
    rollout = {
        "trajectory_id": "traj-abc",
        "example_id": 12,
        "reward": 0.75,
        "advantage": 0.25,
        "metrics": {"correct": 1.0, "fmt": 0.5},
        "trajectory": stream,
    }
    group = converter.group([rollout], task_ref="task:12")
    run = converter.run(
        [converter.step([group])], kind="eval", environment="gsm8k", model="Qwen/Qwen3-30B-A3B"
    )
    RecordWriter(tmp_path).write(run)
    back = open_run(tmp_path, str(run.id))
    trajectory = list(back.steps)[0].groups[0].trajectories[0]

    assert str(trajectory.id) == "traj-abc"
    assert trajectory.advantage == 0.25
    assistant = [t for t in trajectory.turns if t.role == "assistant"]
    assert assistant[0].logprobs_sampling == (-0.11, -0.22, -0.33)
    assert assistant[0].token_ids == (21, 22, 23)
    assert assistant[1].truncated is True
    assert isinstance(assistant[0].tensors["routed_experts"], AbsentRef)
    assert back.lineage.framework == "verifiers"
    assert back.lineage.framework_version == "edafab85"


# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------


def test_importing_the_adapter_pulls_no_framework():
    """The record is the interface, so the converter runs without verifiers installed.

    A subprocess rather than a check on this one, because the tests above have already imported
    the record and ``sys.modules`` here is poisoned. ``pydantic`` is deliberately not in the list:
    it arrives through ``core/config.py`` and is a kernel dependency rather than a framework.
    """
    code = (
        "import sys; import reward_lens.tap.adapters.verifiers as m; "
        "print(sorted(k for k in ('verifiers','torch','trl','transformers','datasets') "
        "if k in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", f"the adapter imported {out.stdout.strip()} at module scope"


def test_the_findings_travel_with_the_record(stream, tmp_path):
    converter = VerifiersConverter()
    converter.trajectory(stream)
    assert converter.report.findings == CONVERTER_FINDINGS
    assert len(CONVERTER_FINDINGS) >= 7
    for finding in CONVERTER_FINDINGS:
        assert len(finding) > 80


# ---------------------------------------------------------------------------
# Re-verification against the framework's own source, when it is reachable
# ---------------------------------------------------------------------------


def _verifiers_types() -> Path | None:
    root = os.environ.get("REWARD_LENS_VERIFIERS_SOURCE")
    if not root:
        return None
    path = Path(root) / "verifiers" / "types.py"
    return path if path.exists() else None


@pytest.mark.skipif(_verifiers_types() is None, reason="set REWARD_LENS_VERIFIERS_SOURCE")
def test_the_field_lists_match_the_framework_source():
    """E7's field lists, re-checked against source rather than carried forward."""
    import ast

    tree = ast.parse(_verifiers_types().read_text(encoding="utf-8"))
    found = {
        node.name: [stmt.target.id for stmt in node.body if isinstance(stmt, ast.AnnAssign)]
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    }
    assert tuple(found["TrajectoryStep"]) == TRAJECTORY_STEP_FIELDS
    assert tuple(found["TrajectoryStepTokens"]) == TOKEN_FIELDS


# ---------------------------------------------------------------------------
# The real subject: published vf-eval runs from the community environments hub
# ---------------------------------------------------------------------------


def _corpus() -> Path | None:
    root = os.environ.get("REWARD_LENS_VERIFIERS_CORPUS")
    if not root or not Path(root).is_dir():
        return None
    return Path(root)


@pytest.mark.skipif(_corpus() is None, reason="set REWARD_LENS_VERIFIERS_CORPUS")
def test_real_published_rollouts_convert():
    """Every row of every published ``results.jsonl`` in the corpus becomes a `Trajectory`.

    These are real ``vf-eval`` outputs from public environments, and none of them carries a
    ``trajectory``: ``save_utils.state_to_output`` copies the stream only when ``--state-columns``
    names it. So this asserts the messages-only path and the fact that it is the path the published
    ecosystem actually needs, and it asserts that no row is silently skipped.
    """
    converter = VerifiersConverter(run_id="hub")
    rows = 0
    for path in sorted(_corpus().glob("*.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            trajectory = converter.rollout(row)
            assert trajectory.turns, f"{path.name}: a rollout converted to no turns"
            check_tiling(trajectory.provenance, len(trajectory.turns), where=path.name)
            rows += 1
    assert rows > 0
    assert converter.report.trajectories == rows
    assert converter.report.rows_without_trajectory == rows, (
        "no published run in this corpus was written with --state-columns trajectory"
    )

"""Unit and property tests for the ``verifiers`` converter.

The acceptance file asserts the field map. This one asserts the arithmetic underneath it: that the
token reconstruction is lossless whenever the framework's own prefix relation holds, that the
message-delta logic does not drop or duplicate a message, and that the pieces behave on inputs
nobody would write by hand.
"""

from __future__ import annotations

import json
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reward_lens.record.provenance import check_tiling
from reward_lens.record.scores import Leaf, ScoreContext, WeightedSum, evaluate
from reward_lens.tap.adapters.verifiers import (
    VerifiersConverter,
    _common_prefix,
    _content_text,
    _get,
    _has,
    _json_safe,
    _messages,
    _record_role,
    _uniform,
    convert_trajectory,
    estimator_spec,
    group_by_trajectory,
    score_tree,
)

# ---------------------------------------------------------------------------
# The accessors, which are what let this run without verifiers installed
# ---------------------------------------------------------------------------


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_get_reads_mappings_and_objects_alike():
    assert _get({"a": 1}, "a") == 1
    assert _get(_Obj(a=1), "a") == 1
    assert _get({"a": 1}, "b", "fallback") == "fallback"
    assert _get(None, "a", 7) == 7


def test_has_separates_absent_from_present_and_none():
    """``multi_modal_data`` is NotRequired, so the two cases are different facts."""
    assert _has({"a": None}, "a") is True
    assert _has({}, "a") is False
    assert _has(None, "a") is False


def test_json_safe_passes_primitives_and_caps_the_rest():
    assert _json_safe(1.5) == (1.5, False)
    assert _json_safe(None) == (None, False)
    assert _json_safe([1, "x", {"k": 2}]) == ([1, "x", {"k": 2}], False)
    value, coerced = _json_safe(object())
    assert coerced is True and isinstance(value, str)
    nested, coerced = _json_safe({"outer": [object()]})
    assert coerced is True
    assert isinstance(nested["outer"][0], str)


def test_json_safe_truncates_at_the_cap():
    """A pixel tensor's repr is unbounded and the record has to stay writable."""

    class Big:
        def __repr__(self):
            return "x" * 5000

    value, coerced = _json_safe(Big())
    assert coerced is True
    assert len(value) == 512


def test_json_safe_output_is_always_json_serialisable():
    payload = {"a": object(), "b": [object(), 1], "c": {"d": object()}}
    value, coerced = _json_safe(payload)
    assert coerced is True
    json.dumps(value)


def test_content_text_handles_both_content_shapes():
    assert _content_text("plain") == "plain"
    assert _content_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    assert _content_text([{"type": "image_url", "image_url": {"url": "u"}}]) == ""
    assert _content_text(None) == ""


def test_messages_normalises_the_three_shapes_it_can_receive():
    assert _messages(None) == ()
    assert _messages("raw prompt") == ({"role": "text", "content": "raw prompt"},)
    assert _messages([{"role": "user", "content": "q"}]) == ({"role": "user", "content": "q"},)
    assert _messages({"role": "user", "content": "q"}) == ({"role": "user", "content": "q"},)


def test_record_role_is_the_two_documented_decisions():
    assert _record_role("user", position="prompt") == "user"
    assert _record_role("user", position="env") == "environment"
    assert _record_role("tool", position="env") == "tool"
    assert _record_role("system", position="prompt") == "system"
    assert _record_role("text", position="prompt") == "user"
    assert _record_role("text", position="completion") == "assistant"


def test_common_prefix_is_the_ordinary_one():
    assert _common_prefix([], []) == 0
    assert _common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert _common_prefix([1, 2], [1, 2, 3]) == 2
    assert _common_prefix([{"a": 1}], [{"a": 1}]) == 1


def test_uniform_is_the_backfill_detector():
    assert _uniform([]) is None
    assert _uniform([0.5, 0.5, 0.5]) == 0.5
    assert _uniform([0.5, 0.4]) is None
    assert _uniform([0.5, None]) is None, "a gap is not a uniform column"
    assert _uniform([math.nan, math.nan]) is not None, "NaN equals NaN for this purpose"


# ---------------------------------------------------------------------------
# The score tree, on hand-computed values
# ---------------------------------------------------------------------------


def test_score_tree_weights_and_total_are_hand_checkable():
    tree = score_tree({"a": 2.0, "b": 3.0}, weights={"a": 0.5, "b": 2.0})
    assert isinstance(tree, WeightedSum)
    assert evaluate(tree, ScoreContext()) == pytest.approx(0.5 * 2.0 + 2.0 * 3.0)


def test_a_weight_zero_metric_is_kept_in_the_tree():
    """``Rubric.add_metric`` registers weight 0. Dropping it would lose half the composition."""
    tree = score_tree(
        {"reward_fn": 1.0, "diagnostic": 9.0}, weights={"reward_fn": 1.0, "diagnostic": 0.0}
    )
    assert [leaf.name for leaf in tree.children] == ["reward_fn", "diagnostic"]
    assert tree.weights == (1.0, 0.0)
    assert evaluate(tree, ScoreContext()) == pytest.approx(1.0)


def test_a_none_metric_is_an_abstention_not_a_zero():
    tree = score_tree({"a": None, "b": 1.0})
    a = tree.children[0]
    assert a.value is None and a.abstained is True
    assert math.isnan(evaluate(tree, ScoreContext()))


def test_score_tree_is_pure():
    """It takes no report, so the counters cannot drift from the features they summarise."""
    import inspect

    assert "report" not in inspect.signature(score_tree).parameters


def test_empty_metrics_produce_no_tree():
    assert score_tree({}) is None


# ---------------------------------------------------------------------------
# The estimator arithmetic
# ---------------------------------------------------------------------------


@given(
    rewards=st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=12,
    )
)
def test_score_groups_advantages_always_sum_to_zero(rewards):
    """``advantage_i = reward_i - mean(r)`` is the whole estimator, so the column is centred."""
    mean = sum(rewards) / len(rewards)
    advantages = [r - mean for r in rewards]
    assert sum(advantages) == pytest.approx(0.0, abs=1e-6)
    rollouts = [
        {"trajectory_id": f"t{i}", "reward": r, "advantage": a, "metrics": {"c": r}}
        for i, (r, a) in enumerate(zip(rewards, advantages))
    ]
    group = VerifiersConverter(weights={"c": 1.0}).group(rollouts)
    assert [t.advantage for t in group.trajectories] == advantages
    assert group.group_stats.k == len(rewards)


def test_the_estimator_spec_is_stable_across_calls():
    assert estimator_spec() == estimator_spec()


def test_the_estimator_spec_names_where_every_claim_came_from():
    spec = estimator_spec()
    assert "rubric.py:406-409" in spec.extra["source"]
    assert "rubric.py:204-217" in spec.extra["silent_zero_site"]
    assert "rubric.py:410-414" in spec.extra["backfill_site"]


# ---------------------------------------------------------------------------
# The turn reconstruction, as a property
# ---------------------------------------------------------------------------


def _prefix_stable_stream(n_steps: int, prompt_len: int, completion_lens, env_lens):
    """A stream obeying MultiTurnEnv's own relation: prompt_i = prompt_{i-1} + completion_{i-1} + env."""
    steps = []
    ids = list(range(1000, 1000 + prompt_len))
    prompt_messages = [{"role": "user", "content": "start"}]
    for i in range(n_steps):
        c_len = completion_lens[i]
        completion_ids = list(range(2000 + 100 * i, 2000 + 100 * i + c_len))
        completion_messages = [{"role": "assistant", "content": f"turn {i}"}]
        steps.append(
            {
                "prompt": list(prompt_messages),
                "completion": list(completion_messages),
                "response": {"model": "m", "message": {"finish_reason": "stop"}},
                "tokens": {
                    "prompt_ids": list(ids),
                    "prompt_mask": [0] * len(ids),
                    "completion_ids": completion_ids,
                    "completion_mask": [1] * c_len,
                    "completion_logprobs": [-0.1 * (j + 1) for j in range(c_len)],
                    "overlong_prompt": False,
                    "is_truncated": False,
                    "routed_experts": None,
                },
                "reward": 1.0,
                "advantage": 0.0,
                "is_truncated": False,
                "trajectory_id": "t",
                "extras": {},
            }
        )
        e_len = env_lens[i]
        env_ids = list(range(5000 + 100 * i, 5000 + 100 * i + e_len))
        ids = ids + completion_ids + env_ids
        prompt_messages = (
            prompt_messages + completion_messages + [{"role": "user", "content": f"env {i}"}]
        )
    return steps


@settings(max_examples=60, deadline=None)
@given(
    n_steps=st.integers(min_value=1, max_value=5),
    prompt_len=st.integers(min_value=1, max_value=8),
    completion_lens=st.lists(st.integers(min_value=1, max_value=6), min_size=5, max_size=5),
    env_lens=st.lists(st.integers(min_value=1, max_value=5), min_size=5, max_size=5),
)
def test_a_prefix_stable_stream_reconstructs_every_token_exactly_once(
    n_steps, prompt_len, completion_lens, env_lens
):
    """The turn token ids, concatenated, are the final full sequence with nothing lost or doubled."""
    steps = _prefix_stable_stream(n_steps, prompt_len, completion_lens, env_lens)
    converter = VerifiersConverter()
    trajectory = converter.trajectory(steps)
    rebuilt = [i for turn in trajectory.turns for i in (turn.token_ids or ())]
    expected = list(steps[-1]["tokens"]["prompt_ids"]) + list(steps[-1]["tokens"]["completion_ids"])
    assert rebuilt == expected
    assert trajectory.n_tokens == len(expected)
    assert converter.report.non_prefix_steps == 0
    assert converter.report.compaction_events == 0
    check_tiling(trajectory.provenance, len(trajectory.turns), where="property")


@settings(max_examples=60, deadline=None)
@given(
    n_steps=st.integers(min_value=1, max_value=5),
    prompt_len=st.integers(min_value=1, max_value=8),
    completion_lens=st.lists(st.integers(min_value=1, max_value=6), min_size=5, max_size=5),
    env_lens=st.lists(st.integers(min_value=1, max_value=5), min_size=5, max_size=5),
)
def test_every_logprob_survives_and_stays_aligned_with_its_token(
    n_steps, prompt_len, completion_lens, env_lens
):
    steps = _prefix_stable_stream(n_steps, prompt_len, completion_lens, env_lens)
    trajectory = convert_trajectory(steps)
    assistant = [t for t in trajectory.turns if t.role == "assistant"]
    assert len(assistant) == n_steps
    for turn, step in zip(assistant, steps):
        assert turn.logprobs_sampling == tuple(step["tokens"]["completion_logprobs"])
        assert len(turn.logprobs_sampling) == len(turn.token_ids)


@settings(max_examples=60, deadline=None)
@given(
    n_steps=st.integers(min_value=1, max_value=5),
    prompt_len=st.integers(min_value=1, max_value=8),
    completion_lens=st.lists(st.integers(min_value=1, max_value=6), min_size=5, max_size=5),
    env_lens=st.lists(st.integers(min_value=1, max_value=5), min_size=5, max_size=5),
)
def test_no_message_is_dropped_or_duplicated(n_steps, prompt_len, completion_lens, env_lens):
    """The delta over messages partitions the final conversation exactly."""
    steps = _prefix_stable_stream(n_steps, prompt_len, completion_lens, env_lens)
    trajectory = convert_trajectory(steps)
    rebuilt = [m for turn in trajectory.turns for m in turn.extra["messages"]]
    final = list(steps[-1]["prompt"]) + list(steps[-1]["completion"])
    assert rebuilt == final


@given(
    ids=st.lists(
        st.tuples(st.sampled_from(["a", "b", "c"]), st.integers(min_value=0, max_value=3)),
        max_size=20,
    )
)
def test_group_by_trajectory_partitions_the_stream(ids):
    stream = [{"trajectory_id": t, "seq": s} for t, s in ids]
    grouped = group_by_trajectory(stream)
    assert sum(len(v) for v in grouped.values()) == len(stream)
    for tid, steps in grouped.items():
        assert [s["seq"] for s in steps] == [s for t, s in ids if t == tid], "order is preserved"


def test_group_by_trajectory_names_the_missing_id_rather_than_dropping_the_step():
    grouped = group_by_trajectory([{"seq": 1}, {"trajectory_id": None, "seq": 2}])
    assert grouped == {"unknown": [{"seq": 1}, {"trajectory_id": None, "seq": 2}]}


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_an_empty_stream_converts_to_an_empty_trajectory():
    """An immediate refusal is a recordable event and it has no generating policy for any turn."""
    trajectory = convert_trajectory([], trajectory_id="empty", task_ref="t")
    assert trajectory.turns == ()
    assert trajectory.provenance == ()
    assert trajectory.advantage is None
    check_tiling(trajectory.provenance, 0, where="empty")


def test_a_rollout_with_no_trajectory_uses_its_messages():
    """What every published vf-eval row looks like: messages, a reward, and no tokens."""
    converter = VerifiersConverter()
    trajectory = converter.rollout(
        {
            "id": 4,
            "prompt": [{"role": "user", "content": "q"}],
            "completion": [
                {"role": "assistant", "content": "thinking"},
                {"role": "tool", "content": "result"},
                {"role": "assistant", "content": "done"},
            ],
            "reward": 1.0,
        }
    )
    assert [t.role for t in trajectory.turns] == ["user", "assistant", "tool", "assistant"]
    assert all(t.token_ids is None for t in trajectory.turns)
    assert isinstance(trajectory.scores, Leaf)
    assert trajectory.scores.value == 1.0
    assert converter.report.rows_without_trajectory == 1
    assert "trajectory" in trajectory.turns[0].extra["token_ids_absent_because"]
    check_tiling(trajectory.provenance, len(trajectory.turns), where="messages only")


def test_a_rollout_with_neither_messages_nor_a_stream_is_still_a_trajectory():
    trajectory = VerifiersConverter().rollout({"id": 1, "reward": 0.0})
    assert trajectory.turns == ()
    assert trajectory.provenance == ()


def test_a_mask_of_the_wrong_length_is_refused_at_construction():
    """A per-token array of the wrong length is a unit mismatch waiting to be averaged."""
    step = {
        "prompt": [{"role": "user", "content": "q"}],
        "completion": [{"role": "assistant", "content": "a"}],
        "response": {"model": "m", "message": {"finish_reason": "stop"}},
        "tokens": {
            "prompt_ids": [1, 2],
            "prompt_mask": [0],
            "completion_ids": [3],
            "completion_mask": [1],
            "completion_logprobs": [-0.1],
            "overlong_prompt": False,
            "is_truncated": False,
            "routed_experts": None,
        },
        "reward": 1.0,
        "advantage": 0.0,
        "is_truncated": False,
        "trajectory_id": "t",
        "extras": {},
    }
    with pytest.raises(ValueError, match="prompt_mask|loss_mask"):
        convert_trajectory([step])


def test_a_tool_call_lands_and_several_are_all_kept():
    step = {
        "prompt": [{"role": "user", "content": "q"}],
        "completion": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "name": "search", "arguments": '{"q": "x"}'},
                    {"id": "c2", "name": "read", "arguments": "{}"},
                ],
            }
        ],
        "response": {"model": "m", "message": {"finish_reason": "tool_calls"}},
        "tokens": None,
        "reward": None,
        "advantage": None,
        "is_truncated": False,
        "trajectory_id": "t",
        "extras": {},
    }
    trajectory = convert_trajectory([step])
    assistant = trajectory.turns[-1]
    assert assistant.tool_call is None, "two calls in one message do not become one tool_call"
    assert [c["name"] for c in assistant.extra["tool_calls"]] == ["search", "read"]

    step["completion"][0]["tool_calls"] = [{"id": "c1", "name": "search", "arguments": "{}"}]
    single = convert_trajectory([step])
    assert single.turns[-1].tool_call.name == "search"
    assert single.turns[-1].tool_call.call_id == "c1"


def test_a_routing_payload_that_cannot_be_decoded_becomes_an_absence_with_the_error(tmp_path):
    from reward_lens.record.tensors import AbsentRef, TensorStore

    step = {
        "prompt": [{"role": "user", "content": "q"}],
        "completion": [{"role": "assistant", "content": "a"}],
        "response": {"model": "m", "message": {"finish_reason": "stop"}},
        "tokens": {
            "prompt_ids": [1],
            "prompt_mask": [0],
            "completion_ids": [2],
            "completion_mask": [1],
            "completion_logprobs": [-0.1],
            "overlong_prompt": False,
            "is_truncated": False,
            "routed_experts": {
                "data": b"\x01\x02\x03",
                "shape": [4, 4],
                "start": 0,
                "dtype": "uint8",
            },
        },
        "reward": 1.0,
        "advantage": 0.0,
        "is_truncated": False,
        "trajectory_id": "t",
        "extras": {},
    }
    converter = VerifiersConverter(store=TensorStore(tmp_path))
    trajectory = converter.trajectory([step])
    ref = trajectory.turns[1].tensors["routed_experts"]
    assert isinstance(ref, AbsentRef)
    assert "could not be decoded" in ref.detail
    assert converter.report.routing_absent == 1
    assert converter.report.routing_stored == 0


def test_the_report_counters_agree_with_the_per_trajectory_features():
    converter = VerifiersConverter()
    total_unresolved = 0.0
    for metrics in ({"a": 0.0, "b": 1.0}, {"c": 0.0}, {"d": 0.0, "e": 0.0}):
        trajectory = converter.trajectory([], metrics=metrics, trajectory_id="x", task_ref="t")
        total_unresolved += trajectory.features["verifiers_unresolved_zeros"]
    assert converter.report.unresolved_zeros == total_unresolved == 4

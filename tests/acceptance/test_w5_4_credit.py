"""Acceptance: G1, G2, G3, the credit measure.

The clause, in two halves, and this file asserts both:

    the disintegrations sum to the update norm within tolerance (the conservation test), and one
    framework's existing tensor dump is consumed rather than reimplemented.

**The first half is asserted at the tolerance achieved, not at one hoped for.** Measured on step 0
of the 200-step reference record with the model that wrote it, over eight real rollouts and 107
trained positions: the eight per-rollout gradients sum to the full gradient with a relative residual
of 1.126e-07 in float32 and 2.704e-16 in float64, and one real `torch.optim.SGD` step at the
record's own learning rate of 1e-06 moves the parameters by `dtheta` with
`|dtheta/eta - sum_S g_S| / |sum_S g_S| = 4.938e-09`. Every threshold below is set from those
numbers with one order of magnitude of headroom, and the float64 case is asserted at machine
epsilon because that is what it reaches.

**The second half is a real consumer of a real format.** E7 is right that the specified rung 0
points at the wrong thing: `dump_data_batch` is a bool config field at `config.py:1085` and the
function is `dump_data` at `trainer.py:1238-1244`. E7 is also right about the sharper problem, and
this file reproduces it as a test rather than repeating it as a claim: `rewards` is popped at
`trainer.py:436` and `uids` at `:437`, five lines before the dump, so the one mechanism the
specification says already writes the interpretable tensor writes neither the reward it came from
nor the group it belongs to. The reader here consumes what does reach disk, recovers the grouping
by hashing the prompt, and never imports `skyrl`.

**What this subject cannot support, asserted rather than worked around.** The reference record's
1,600 trajectories are two turns each, one user and one assistant, and contain zero tool calls. So
the turn decomposition is exercised on what is there (and correctly puts 100% of the credit on the
assistant turn and 0% on the masked user turn), the tool-call structure is exercised on a
trajectory constructed through the record schema, and the subject a real turn-concentration claim
needs is named in `test_the_subject_a_turn_concentration_claim_needs`.
"""

from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the credit measure differentiates; needs [white-box]")

from reward_lens.core.reading import Refusal, RefusalReason  # noqa: E402
from reward_lens.core.types import Span  # noqa: E402
from reward_lens.measure.base import lint_instrument, lint_reading  # noqa: E402
from reward_lens.policy.credit import (  # noqa: E402
    SKYRL_MODULE,
    CreditDisintegration,
    ImplicitPRMReading,
    Part,
    Partition,
    PartitionError,
    TrajectoryBatch,
    TurnCredit,
    batch_from_skyrl,
    batch_from_trajectories,
    by_rollout,
    by_segment,
    disintegrate,
    implicit_prm,
    read_skyrl_dump,
    step_conservation,
    turn_mass,
)
from reward_lens.policy.hf import from_pretrained  # noqa: E402
from reward_lens.record.reader import open_run  # noqa: E402
from reward_lens.record.schema import make_trajectory  # noqa: E402
from reward_lens.record.turns import ToolCall, Turn  # noqa: E402

TINY_POLICY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
LONG = Path(__file__).resolve().parents[1] / "fixtures" / "grpo_run" / "long"
LONG_RUN = "run:f77bf75940ab982bbc35407af99cc094"

#: The record's own learning rate, read off `Step.schedule` rather than chosen here.
RECORD_LR = 1e-06


@pytest.fixture(scope="module")
def policy():
    return from_pretrained(TINY_POLICY)


@pytest.fixture(scope="module")
def first_step():
    run = open_run(LONG, LONG_RUN)
    return next(iter(run.steps))


@pytest.fixture(scope="module")
def real_batch(policy, first_step):
    """Step 0 of the reference record, tokenised through the model that wrote it."""
    built = batch_from_trajectories(policy.tokenizer, list(first_step.trajectories))
    assert not isinstance(built, Refusal), built
    return built


# ---------------------------------------------------------------------------
# The clause, first half: the disintegrations sum to the update
# ---------------------------------------------------------------------------


def test_the_disintegration_closes_on_a_real_step(real_batch, policy, first_step):
    """Eight real rollouts, eight gradients, and they add up to the one the optimizer would take.

    The tolerance asserted is 1e-06 and the value measured is 1.126e-07. The full gradient is taken
    by its own backward pass rather than as the sum of the parts, so this is an audit rather than an
    identity restated: summing the parts to define the total would make the residual zero by
    construction.
    """
    batch, _ = real_batch
    assert batch.n_rollouts == 8, "the reference record runs two groups of K=4"
    assert batch.n_trained == 107

    report = disintegrate(policy, batch, by_rollout(batch))
    assert not isinstance(report, Refusal), report

    assert report.conservation_error < 1e-6, (
        f"the eight per-rollout gradients do not sum to the full one: "
        f"{report.conservation_error:.3e}"
    )
    assert report.n_backwards == 9, "one backward per part plus one for the full objective"
    assert report.n_parameters == 2_453_368
    # The projected shares are the additive functional, so they close exactly.
    assert sum(report.projected_share) == pytest.approx(1.0, abs=1e-6)
    # The norms are not additive, and the gap is a measured quantity rather than an error.
    assert report.cancellation > 1.0
    assert report.cancellation == pytest.approx(2.2285, abs=0.01)


def test_the_update_identity_holds_against_a_real_optimizer_step(real_batch, policy):
    """`sum_{k,t} mu(k,t) = dtheta / eta`, checked by taking the step and differencing.

    The optimizer is a real `torch.optim.SGD` rather than a hand-written `p -= lr * p.grad`, because
    the claim is about what an optimizer does and checking it against three lines written above the
    assertion would be checking it against itself.

    Asserted at 1e-07; measured 4.938e-09 at the record's own learning rate in float64.
    """
    batch, _ = real_batch
    report = step_conservation(policy, batch, by_rollout(batch), lr=RECORD_LR, dtype="float64")
    assert not isinstance(report, Refusal), report

    assert report.update_optimizer == "SGD"
    assert report.update_conservation_error is not None
    assert report.update_conservation_error < 1e-7, (
        f"the summed credit measure does not reproduce the update: "
        f"{report.update_conservation_error:.3e}"
    )
    # In float64 the disintegration itself reaches machine epsilon.
    assert report.conservation_error < 1e-14
    # And the check is not precision-limited here, which is the fact that licenses reading it.
    assert report.update_precision_limited is False
    assert report.update_conservation_error < report.update_precision_floor


def test_the_update_check_is_precision_limited_in_float32_and_says_so(real_batch, policy):
    """The same identity at the same learning rate in float32 resolves nothing, and reports that.

    This is the failure mode the `update_precision_floor` field exists to prevent. `dtheta/eta` is
    recovered by differencing parameters, so its error has a floor of `eps * |theta| / (eta * |g|)`.
    At the reference record's learning rate of 1e-06 in float32 that floor is 7.9: the update is
    smaller than the last representable digit of the parameters it moves. A reader handed only the
    error would see 0.15 and read a precision limit as a failed audit.
    """
    batch, _ = real_batch
    report = step_conservation(policy, batch, by_rollout(batch), lr=RECORD_LR, dtype="float32")
    assert not isinstance(report, Refusal), report

    assert report.update_precision_limited is True
    assert report.update_precision_floor > 1.0
    # The disintegration is unaffected: it never differences a parameter.
    assert report.conservation_error < 1e-6
    assert any("precision-limited" in n for n in report.notes)


def test_precision_is_reported_because_the_native_dtype_cannot_carry_the_audit(real_batch, policy):
    """bfloat16 gives a conservation error four orders of magnitude worse, and that is why we cast.

    Measured on this step: 1.560e-03 native against 1.126e-07 in float32. The number is not a defect
    in the disintegration, it is the numerics floor showing up in the one place that makes it
    visible, and an instrument that measured in the model's own dtype would report it as credit.
    """
    batch, _ = real_batch
    native = disintegrate(policy, batch, by_rollout(batch), dtype="")
    assert not isinstance(native, Refusal), native
    assert native.dtype_used == "bfloat16"
    assert native.conservation_error > 1e-4
    assert native.conservation_error < 1e-2

    cast = disintegrate(policy, batch, by_rollout(batch), dtype="float32")
    assert cast.conservation_error < native.conservation_error / 1000


def test_a_partition_that_does_not_tile_is_refused_at_construction(real_batch):
    """A gap in the partition produces a conservation error that looks numerical and is not."""
    batch, _ = real_batch
    trained = batch.trained
    half = trained.copy()
    half[0] = False
    with pytest.raises(PartitionError, match="uncovered"):
        Partition(parts=(Part(name="most of it", kind="rollout", mask=half),), trained=trained)

    overlap = Part(name="everything", kind="rollout", mask=trained)
    with pytest.raises(PartitionError, match="covered twice"):
        Partition(parts=(overlap, overlap), trained=trained)


def test_a_degenerate_group_refuses_rather_than_reporting_uniform_shares(policy, real_batch):
    """Every advantage zero is E3, not a measurement problem, and every share of zero is
    undefined rather than uniform."""
    batch, _ = real_batch
    flat = TrajectoryBatch(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        loss_mask=batch.loss_mask,
        advantages=np.zeros(batch.n_rollouts),
        aggregation=batch.aggregation,
    )
    out = disintegrate(policy, flat, by_rollout(flat))
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.BELOW_LOD
    assert "degenerate group" in out.remedy


# ---------------------------------------------------------------------------
# The clause, second half: a framework's tensor dump is consumed, not reimplemented
# ---------------------------------------------------------------------------


def _write_skyrl_dump(path: Path, batch_dict: dict, metadata: dict) -> None:
    """Write a pickle structurally identical to what `SkyRLTrainer.dump_data` leaves on disk.

    Faithful to `TensorBatch.__reduce__`, which returns
    ``(_rebuild_tensor_batch, (type(self), self.__getstate__()))`` with the state a plain dict of
    ``{"batch_dict", "batch_size", "device", "metadata"}`` and each tensor serialised as
    ``{"format": "numpy", "data": bytes, "shape": ..., "dtype": ...}``.

    The fixture builds the file rather than shipping one because a checked-in pickle is an opaque
    binary nobody can review, and because `skyrl` is not installed here, which is the whole point:
    the reader under test must not need it either. `__module__` and `__qualname__` are set so pickle
    writes SkyRL's own dotted paths, and the stand-in modules are removed afterwards so nothing
    later in the process can resolve them.
    """
    created = []
    for name in ("skyrl", "skyrl.backends", "skyrl.backends.skyrl_train", SKYRL_MODULE):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
            created.append(name)
    module = sys.modules[SKYRL_MODULE]

    class TrainingInputBatch(dict):
        pass

    def _rebuild_tensor_batch(cls, state):  # pragma: no cover - never called on the read path
        raise RuntimeError("the reader under test must not call SkyRL's rebuild helper")

    TrainingInputBatch.__module__ = SKYRL_MODULE
    TrainingInputBatch.__qualname__ = "TrainingInputBatch"
    _rebuild_tensor_batch.__module__ = SKYRL_MODULE
    _rebuild_tensor_batch.__qualname__ = "_rebuild_tensor_batch"
    module.TrainingInputBatch = TrainingInputBatch
    module._rebuild_tensor_batch = _rebuild_tensor_batch

    def _serialise(arr):
        return {
            "format": "numpy",
            "data": arr.tobytes(),
            "shape": arr.shape,
            "dtype": str(arr.dtype),
        }

    class _Shim:
        def __reduce__(self):
            return (
                _rebuild_tensor_batch,
                (
                    TrainingInputBatch,
                    {
                        "batch_dict": {k: _serialise(v) for k, v in batch_dict.items()},
                        "batch_size": next(iter(batch_dict.values())).shape[0],
                        "device": None,
                        "metadata": metadata,
                    },
                ),
            )

    with path.open("wb") as handle:
        pickle.dump(_Shim(), handle)
    for name in created:
        sys.modules.pop(name, None)


@pytest.fixture
def skyrl_dump(tmp_path):
    """A dump shaped exactly as SkyRL leaves one: two prompts, four rollouts, no rewards, no uids."""
    ids = np.arange(4 * 10, dtype=np.int64).reshape(4, 10)
    ids[:2, :4] = np.array([11, 12, 13, 14])
    ids[2:, :4] = np.array([21, 22, 23, 24])
    response = np.zeros((4, 10), dtype=np.int64)
    response[:, 4:] = 1
    advantages = np.zeros((4, 10), dtype=np.float64)
    advantages[:, 4:] = np.array([[1.0], [-1.0], [0.5], [-0.5]])
    path = tmp_path / "global_step_7_training_input.pkl"
    _write_skyrl_dump(
        path,
        {
            "sequences": ids,
            "attention_mask": np.ones_like(ids),
            "loss_mask": response,
            "response_mask": response,
            "advantages": advantages,
            "action_log_probs": np.full((4, 10), -0.5),
        },
        # `uids` is absent because SkyRL pops it at trainer.py:437, before the dump.
        {"global_step": 7},
    )
    return path


def test_the_skyrl_dump_is_consumed_without_importing_skyrl(skyrl_dump):
    """Rung 0: read the per-token advantage tensor a framework already writes.

    The reader resolves four names through a restricted unpickler and decodes the buffers with
    numpy. `skyrl` is not installed in this environment and is not in `sys.modules` afterwards,
    which is the assertion that separates consuming the format from depending on the library.
    """
    assert "skyrl" not in sys.modules

    dump = read_skyrl_dump(skyrl_dump)
    assert not isinstance(dump, Refusal), dump
    assert dump.n_rows == 4
    assert dump.seq_len == 10
    assert dump.has_advantages is True
    assert dump.advantages.shape == (4, 10)
    # The per-token advantage tensor came off disk. Nothing here recomputed an advantage.
    assert dump.advantages[0, 4:].tolist() == [1.0] * 6

    assert "skyrl" not in sys.modules, "the reader must not import the library to read its format"


def test_the_dump_carries_neither_rewards_nor_uids(skyrl_dump):
    """E7, reproduced as a test rather than repeated as a claim.

    The specification calls this rung "consume SkyRL's `dump_data_batch` where it exists" and
    treats it as free. It is partial: `rewards` is popped at `trainer.py:436` and `uids` at `:437`, and `dump_data` is
    called at `:441`. So the tensor reaches disk and the two fields that make it interpretable do
    not. The group is recoverable because rows from one prompt share the ids the response mask
    excludes; the reward is not recoverable from this file at all.
    """
    dump = read_skyrl_dump(skyrl_dump)
    assert dump.has_rewards is False
    assert dump.has_uids is False
    assert dump.recovered_groups == 2, "two prompts, recovered by hashing the masked prefix"
    assert any("trainer.py:436" in note and "trainer.py:437" in note for note in dump.notes)


def test_the_consumed_dump_disintegrates_and_closes(skyrl_dump, policy):
    """The end of the rung: what came off disk goes straight into the conservation audit."""
    dump = read_skyrl_dump(skyrl_dump)
    batch = batch_from_skyrl(dump)
    assert not isinstance(batch, Refusal), batch
    assert batch.advantages.ndim == 2, "the per-token tensor is kept per-token, not collapsed"

    report = disintegrate(policy, batch, by_rollout(batch))
    assert not isinstance(report, Refusal), report
    assert report.conservation_error < 1e-6
    assert sum(report.projected_share) == pytest.approx(1.0, abs=1e-6)
    assert any("consumed from the dump, not recomputed" in n for n in report.notes)


def test_the_unpickler_refuses_a_name_outside_the_allowlist(tmp_path):
    """Unpickling is code execution, so an unrecognised name is refused rather than imported."""
    path = tmp_path / "hostile.pkl"
    with path.open("wb") as handle:
        pickle.dump({"posix": __import__("os").system}, handle)
    with pytest.raises(pickle.UnpicklingError, match="resolves"):
        read_skyrl_dump(path)


def test_a_missing_dump_refuses_with_the_config_flag_that_produces_one(tmp_path):
    out = read_skyrl_dump(tmp_path / "nothing.pkl")
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "dump_data_batch" in out.remedy


# ---------------------------------------------------------------------------
# G2: turn and tool-call disintegration, and what this subject cannot support
# ---------------------------------------------------------------------------


def test_the_turn_decomposition_puts_all_credit_on_the_trained_turn(real_batch, policy):
    """The mechanics, asserted on what the reference record actually has.

    Its trajectories are two turns: a user turn carrying the prompt and an assistant turn carrying
    the completion. The loss is taken over the completion only, so the correct answer is exactly
    100% on the assistant turn and exactly 0% on the user turn, and the conservation residual is
    exactly zero because one part carries the whole objective. That is a real check of the
    coordinate shift from turn bounds to target positions, and it is not a check of turn
    concentration, because a two-turn trajectory has no concentration to measure.
    """
    batch, segments = real_batch
    report = disintegrate(policy, batch, by_segment(batch, segments))
    assert not isinstance(report, Refusal), report

    assert report.parts == ["turn 0 (user)", "turn 1 (assistant)"]
    shares = dict(zip(report.parts, report.projected_share))
    assert shares["turn 0 (user)"] == pytest.approx(0.0, abs=1e-12)
    assert shares["turn 1 (assistant)"] == pytest.approx(1.0, abs=1e-9)
    assert report.conservation_error == pytest.approx(0.0, abs=1e-12)


def test_turn_mass_refuses_on_this_record_rather_than_substituting_token_counts(first_step):
    """The TRL tap writes no per-token log-probabilities, so G2's rung 0 cannot run here.

    The refusal is the correct output and the remedy names both routes out: a `verifiers` record
    carries `completion_logprobs`, and on a TRL record the way to the quantity is the white-box
    disintegration. Substituting token counts would return the declared "uniform attribution over
    tokens" baseline under the name `credit.by_turn`, which is the silent downgrade the contract
    forbids.
    """
    out = turn_mass(list(first_step.trajectories))
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "logprobs_sampling" in out.detail
    assert "uniform attribution over tokens" in out.remedy


def _tool_calling_trajectory():
    """A six-turn tool-calling trajectory built through the record schema.

    Constructed, and labelled as constructed. The reference record contains zero tool calls across
    1,600 trajectories, so there is nothing real here to point at; what this establishes is that the
    tool-call and span indices are wired correctly, which is a different claim from turn
    concentration and is the only one this fixture supports.
    """
    turns = []
    for i in range(6):
        role = "assistant" if i % 2 == 0 else "tool"
        n = 5 + i
        turns.append(
            Turn(
                index=i,
                role=role,
                text=f"turn {i}",
                token_ids=tuple(range(100, 100 + n)),
                logprobs_sampling=tuple(-0.5 * (j + 1) for j in range(n)),
                loss_mask=tuple(role == "assistant" for _ in range(n)),
                tool_call=(
                    ToolCall(name="bash" if i == 2 else "grep")
                    if role == "assistant" and i
                    else None
                ),
                spans=(Span(start=1, end=3, kind="tool_call_arguments"),) if i == 2 else (),
            )
        )
    return make_trajectory(id="traj:tool", task_ref="task:tool", turns=turns, advantage=0.8)


def test_tool_call_and_span_shares_are_indexed_correctly():
    """Structural: the shares find the tool calls and the argument span, and the turns sum to one."""
    payload = turn_mass([_tool_calling_trajectory()])
    assert not isinstance(payload, Refusal), payload

    assert payload.n_tool_calls == 2
    assert sum(payload.by_turn.values()) == pytest.approx(1.0)
    # Only the three assistant turns are trained; the tool turns are masked and score nothing.
    assert set(payload.by_turn) == {
        "turn 0 (assistant)",
        "turn 2 (assistant)",
        "turn 4 (assistant)",
    }
    # Turn 0 carries no tool call, so the two named calls carry everything except its share.
    assert payload.tool_call_share == pytest.approx(
        1.0 - payload.by_turn["turn 0 (assistant)"], abs=1e-9
    )
    assert set(payload.by_tool_call) == {"bash", "grep"}
    assert 0.0 < payload.by_span_kind["tool_call_arguments"] < payload.by_turn["turn 2 (assistant)"]


def test_the_subject_a_turn_concentration_claim_needs():
    """Names the limit rather than working around it. This test asserts the gap, not a result.

    The headline for G2 is "in a 40-turn episode, 71% of the log-probability mass that received
    advantage is in turns 1 to 3 and turn 40", and it exists to confirm or refute the report that
    RLVR reallocates probability at a small set of critical positions near the start and end of a
    trajectory. Nothing in this build can make that claim:

    - the reference record is 3,200 turns over 1,600 trajectories, which is exactly two turns each,
      one of them masked, so its turn decomposition has one non-empty cell by construction;
    - it contains no tool call at all;
    - and the AISI series, which is the subject for anything about a transition, is a rollout-level
      artifact with per-step labels rather than per-token log-probabilities.

    The subject this claim needs is an agentic record with tens of turns per trajectory, tool calls,
    and per-token log-probabilities on disk. `verifiers` produces exactly that shape
    (`TrajectoryStepTokens.completion_logprobs` lands in `Turn.logprobs_sampling`), so the gap is a
    subject to record rather than a capability to build.
    """
    run = open_run(LONG, LONG_RUN)
    turns_per_trajectory = set()
    tool_calls = 0
    n_trajectories = 0
    for step in run.steps.slice(0, 20):
        for trajectory in step.trajectories:
            n_trajectories += 1
            turns_per_trajectory.add(trajectory.n_turns)
            tool_calls += sum(1 for t in trajectory.turns if t.tool_call is not None)
    assert turns_per_trajectory == {2}, "every trajectory is one user turn and one assistant turn"
    assert tool_calls == 0, "the reference record contains no tool call"
    assert n_trajectories == 160


# ---------------------------------------------------------------------------
# G3: the implicit process reward model, and its kill condition firing
# ---------------------------------------------------------------------------


def test_the_implicit_prm_is_degenerate_on_every_group_of_the_reference_record(policy):
    """G3's kill condition, measured across all 400 groups rather than argued.

    GRPO with an outcome reward is equivalent to a PRM-aware objective whose process reward model is
    the Monte-Carlo value `q(s) = E[R | prefix s]`. Extracting it needs rollouts that share prefixes,
    because that is the only place a prefix has more than one sample. On this record they never do:
    all 400 groups have a longest common completion prefix of zero tokens, so every rollout is alone
    from its first token, every value past the root is that rollout's own outcome, the whole outcome
    advantage lands on token one and 92% of trained positions carry exactly zero.

    Reporting that is closing the package. The package's own kill condition is "kill if the
    induced function is constant", and this is the measured form of it.
    """
    run = open_run(LONG, LONG_RUN)
    tokenizer = policy.tokenizer
    groups = killed = degenerate = 0
    informative = []
    divergence = []
    for step in run.steps:
        for group in step.groups:
            completions, rewards = [], []
            for trajectory in group.trajectories:
                text = "".join(t.text for t in trajectory.turns if t.role == "assistant")
                completions.append(tokenizer(text, add_special_tokens=False)["input_ids"])
                rewards.append(float(trajectory.advantage))
            payload = implicit_prm(completions, rewards)
            assert not isinstance(payload, Refusal), payload
            groups += 1
            killed += payload.is_killed
            degenerate += payload.is_degenerate
            informative.append(payload.informative_fraction)
            divergence.append(payload.mean_divergence_depth)

    assert groups == 400
    assert killed == 400
    assert degenerate == 400
    assert max(informative) == 0.0, "no position anywhere carries a value from two rollouts"
    assert set(divergence) == {1.0}, "rollouts diverge at their first token, in every group"


def test_the_implicit_prm_is_informative_when_rollouts_share_a_prefix():
    """The other side of the same measurement: given branching, the function is non-degenerate.

    Four rollouts on a shared first token that split into two pairs. The root value is the group
    mean, which is GRPO's own baseline; the depth-1 increment is zero because every rollout is still
    in the same node; the depth-2 increment separates the pairs; and the depth-3 increment carries
    each rollout's own outcome.
    """
    payload = implicit_prm([[1, 2, 3], [1, 2, 9], [1, 5, 6], [1, 5, 7]], [1.0, 0.0, 0.5, -0.5])
    assert not isinstance(payload, Refusal), payload
    assert payload.is_degenerate is False
    assert payload.is_constant is False
    assert payload.is_killed is False
    assert payload.branching_depth == 2
    assert payload.root_value == pytest.approx(0.25)
    assert payload.informative_fraction == pytest.approx(2 / 3)
    # rollout 0: no movement at depth 1, then +0.25 into its pair, then +0.5 to its own outcome.
    assert payload.process_reward[:3] == pytest.approx([0.0, 0.25, 0.5])


def test_the_reroll_rung_estimates_a_value_where_the_sampler_never_branched():
    """Rung 1 is the remedy the degenerate case points at, and it is a real second estimator.

    With no prefix sharing there is no second sample at any prefix, so the only way to a value is to
    draw one. `reroll` takes ``(prefix, m) -> m outcome rewards``; here it is a deterministic stub,
    because what this asserts is the plumbing and the arithmetic, not a model's behaviour.
    """
    seen = []

    def reroll(prefix, m):
        seen.append(prefix)
        return [float(sum(prefix)) + i for i in range(m)]

    payload = implicit_prm([[1, 2, 3], [4, 5, 6]], [1.0, 0.0], reroll=reroll, reroll_width=3)
    assert not isinstance(payload, Refusal), payload
    assert payload.rung == 1
    assert payload.n_reroll_calls == len(seen) == 7  # the empty prefix plus three per rollout
    assert payload.reroll_width == 3
    # Every node now has three samples, so nothing is degenerate any more.
    assert payload.is_degenerate is False
    assert payload.process_reward_sd > 0.0


def test_a_constant_induced_function_is_reported_as_the_kill_condition():
    """A re-roll whose value is linear in prefix length gives a constant increment: nothing there."""
    payload = implicit_prm(
        [[1, 2, 3], [4, 5, 6]],
        [1.0, 0.0],
        reroll=lambda prefix, m: [0.1 * len(prefix) + 0.01 * i for i in range(m)],
        reroll_width=3,
    )
    assert payload.is_constant is True
    assert payload.is_killed is True
    assert any("zero variance" in n for n in payload.notes)


def test_the_implicit_prm_refuses_a_group_of_one():
    out = implicit_prm([[1, 2, 3]], [1.0])
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "GRPO's own baseline" in out.remedy


# ---------------------------------------------------------------------------
# The white-box reading carries an IncrementalValidity, and lint agrees
# ---------------------------------------------------------------------------


def _record_proxy(policy, batch):
    """The record-only estimate: |A| times the summed surprisal of the trained tokens.

    Recomputed under the policy here because the TRL tap writes no per-token log-probabilities. That
    makes it a rung-0 *estimator* evaluated with a forward pass rather than a rung-0 *reading* taken
    from a record, and the distinction is worth keeping: on a record that carried the
    log-probabilities this would cost nothing at all.
    """
    import torch.nn.functional as F

    model = policy.runtime.model
    native = next(model.parameters()).dtype
    model.to(torch.float32)
    try:
        ids = torch.as_tensor(batch.input_ids)
        attention = torch.as_tensor(batch.attention_mask)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits.float()
        logits = logits[:, :-1]
        targets = ids[:, 1:]
        logprobs = -F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        ).view(targets.shape)
    finally:
        model.to(native)
    trained = torch.as_tensor(batch.trained)
    surprisal = -(logprobs * trained).sum(dim=1).numpy()
    return (np.abs(batch.advantages) * surprisal).tolist()


def test_the_white_box_reading_carries_an_incremental_validity(policy):
    """An `IncrementalValidity` is mandatory on a white-box reading, and lint rule four checks it.

    The framing is the one `credit_increment` argues for and it is a declared deviation. M9 presumes
    the instrument is a predictor scored against an external criterion; the credit measure is a
    definition, so scoring it against itself would return a perfect number meaning nothing. The
    criterion here is the white-box measurement and what competes against it is every cheap method,
    the instrument's own record-only proxy included. That answers the rule's actual question directly:
    if the dumb baselines already say which rollouts carried the mass, the backward passes bought
    the ground truth rather than the localisation.

    Measured over 32 real rollouts from four steps: the proxy scores 0.938, the best of the
    comparators that could run scores 0.500, the increment is +0.281 [+0.125, +0.438] and the error
    correlation with the best baseline is +0.26. Read it with the confound named: the proxy and the
    criterion share |A| as a factor, so part of that increment is structural rather than
    informational, and the number that is not confounded is the error correlation.
    """
    run = open_run(LONG, LONG_RUN)
    trajectories, group_ids = [], []
    for step in run.steps.slice(0, 4):
        for ordinal, group in enumerate(step.groups):
            for trajectory in group.trajectories:
                trajectories.append(trajectory)
                group_ids.append(step.index * 10 + ordinal)

    built = batch_from_trajectories(policy.tokenizer, trajectories)
    assert not isinstance(built, Refusal), built
    batch, _ = built
    assert batch.n_rollouts == 32

    texts = [" ".join(t.text for t in tr.turns if t.role == "assistant") for tr in trajectories]
    instrument = CreditDisintegration(
        batch=batch,
        texts=texts,
        proxy=_record_proxy(policy, batch),
        group_ids=group_ids,
        n_resamples=2000,
    )
    from reward_lens.measure.base import Context
    from reward_lens.measure.base import run as run_instrument

    reading = run_instrument(instrument, Context(signal=policy, view=texts))
    assert not isinstance(reading, Refusal), reading

    # The obligation itself.
    assert reading.incremental is not None
    assert lint_reading(reading, instrument) == []
    assert reading.quantity == "credit.measure", "assert the emitted quantity, not the declared one"

    record = reading.incremental
    assert record.baseline_id
    assert -1.0 <= record.error_correlation <= 1.0
    assert reading.value.conservation_error < 1e-6
    assert reading.baselines, "the bank ran and its results travel with the reading"


def test_all_three_instruments_pass_lint():
    """Lint rule one to three on every instrument this package ships."""
    instruments = [
        CreditDisintegration(batch=None),
        TurnCredit(trajectories=[]),
        ImplicitPRMReading(completions=[], rewards=[]),
    ]
    for instrument in instruments:
        assert lint_instrument(instrument) == [], instrument.name


def test_only_the_gradient_instrument_is_white_box():
    """G2 and G3 read a record and open nothing, so the rule puts no obligation on their readings."""
    from reward_lens.measure.base import is_white_box

    assert is_white_box(CreditDisintegration(batch=None)) is True
    assert is_white_box(TurnCredit(trajectories=[])) is False
    assert is_white_box(ImplicitPRMReading(completions=[], rewards=[])) is False

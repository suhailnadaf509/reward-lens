"""The TRL tap.

The clause: *a real small GRPO run produces a complete record with a measured and
published instrument effect, and the run is unaffected by a deliberately failing tap.*

Both halves are here and they are asserted against the same run rather than against two
convenient ones. The run is real: a real ``GRPOTrainer``, a real model with real weights that
really change, real sampling, real advantages, on CPU because the machine's driver is too old for
its torch build. It is small on purpose. Scale is not what the clause asks for and a tiny model is
what makes it possible to run all three arms in one test and compare their weights bit for bit.

"Unaffected" is read as strictly as it can be read. Not "trained fine", not "no exception":
**byte-identical parameters**, hashed over the whole state dict, against an arm with no tap in the
process at all. ``test_two_untapped_runs_are_byte_identical`` establishes first that the
comparison means anything, because a byte-identical assertion against a non-deterministic baseline
passes for the wrong reason.

The failing tap is failed on every surface it has. Its ring raises on every ``offer``, so the
recorder segment inside ``instrument_grader`` throws once per grader call; and every callback body
and both adapter hooks raise as well. Then the run has to finish with the same weights, and the
tap has to have noticed and counted its own failures rather than quietly doing nothing.

The instrument effect is measured two ways because they answer different questions. In the run,
the guard's own accounting says what the tap added per grader call while a real trainer was
driving it. Out of the run, a microbenchmark times the whole wrapped stack against the bare
callable, which is the only way to catch a cost the tap does not charge itself for. The second
number is the one to publish, because it is measured from outside and includes everything.
"""

from __future__ import annotations

import hashlib
import statistics
import time
from dataclasses import dataclass
from typing import Any

import pytest

torch = pytest.importorskip("torch")
trl = pytest.importorskip("trl")
datasets = pytest.importorskip("datasets")

from datasets import Dataset  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from reward_lens.core.envelope import RegimeCondition  # noqa: E402
from reward_lens.core.types import Access, Component  # noqa: E402
from reward_lens.record.schema import SamplingScheme  # noqa: E402
from reward_lens.tap.adapters.trl import METRIC_NAMES, TRLTap  # noqa: E402
from reward_lens.tap.contract import CallOutcome, TapBudget  # noqa: E402
from reward_lens.tap.ring import TapRing  # noqa: E402

#: 0.6M parameters, four layers, a real Qwen3 architecture with random weights. TRL's own test
#: model, so it is the one the framework is exercised against upstream too.
MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"

SEED = 1234
STEPS = 12
BATCH = 8
K = 4

#: Wide enough that nothing in these tests breaches by accident. The breach path is the grader
#: wrapper's own acceptance test; what this file is measuring is what the tap costs when it is working.
GENEROUS = TapBudget(
    max_added_latency_ms_p99=1000.0,
    max_resident_bytes=64 * 1024 * 1024,
    max_added_alloc_bytes_per_step=64 * 1024 * 1024,
)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def length_reward(prompts, completions, completion_ids, **kwargs):
    """A trivial grader with a real abstention channel.

    Length in characters, scaled. The fourth completion of every batch returns ``None``, which is
    TRL's abstention: it maps to ``torch.nan`` at ``grpo_trainer.py:1661`` and is excluded from the
    group baseline. It is here so the record has abstentions in it and the ``None``-versus-zero
    distinction is exercised on a real run rather than only in a unit test.

    The period is 7, which lands one abstention inside the first group of an eight-row generation
    batch and none in the second. That is deliberate: a group where *every* function abstains on
    every row takes a different path in TRL (``unscorable_mask`` at 2679) and is a case for a
    separate test rather than a thing to trip over here.
    """
    return [None if (i % 7 == 3) else len(c) / 50.0 for i, c in enumerate(completions)]


def digest(model: Any) -> str:
    """A hash over every parameter, so "unaffected" is a bit-level claim and not a vibe.

    Bytes rather than values, via a ``uint8`` view. Numpy has no bfloat16, and a checkpoint loaded
    at its stored dtype is bfloat16 here while the trainer's live copy is float32, so anything
    that goes through numpy's dtypes either raises or silently compares two different roundings.
    """
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def build_trainer(out_dir, reward_funcs, *, steps: int = STEPS, log_completions: bool = False):
    prompts = [f"question {i}: count upward from {i}" for i in range(BATCH * 4)]
    args = GRPOConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=BATCH,
        num_generations=K,
        max_completion_length=12,
        max_steps=steps,
        logging_steps=1,
        report_to=[],
        seed=SEED,
        use_cpu=True,
        save_strategy="no",
        disable_tqdm=True,
        log_completions=log_completions,
    )
    return GRPOTrainer(
        model=MODEL,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=Dataset.from_dict({"prompt": prompts}),
    )


@dataclass
class Arm:
    digest: str
    seconds: float
    tap: TRLTap | None = None
    trainer: Any = None


def run_untapped(out_dir) -> Arm:
    trainer = build_trainer(out_dir, length_reward)
    t0 = time.perf_counter()
    trainer.train()
    return Arm(digest=digest(trainer.model), seconds=time.perf_counter() - t0, trainer=trainer)


def run_tapped(out_dir, *, tap: TRLTap, log_completions: bool = False) -> Arm:
    trainer = build_trainer(out_dir, tap.wrap(length_reward), log_completions=log_completions)
    tap.attach(trainer)
    t0 = time.perf_counter()
    trainer.train()
    return Arm(
        digest=digest(trainer.model),
        seconds=time.perf_counter() - t0,
        tap=tap,
        trainer=trainer,
    )


class ExplodingRing(TapRing):
    """A ring that raises on every write, which makes the recorder segment fail on every call.

    This is the sharpest available shape of "a deliberately failing tap". The failure is inside
    ``instrument_grader``'s own recorder, on the host's thread, once per grader call, for the whole
    run, and it is a single distinct exception so the guard never reaches the third-distinct
    disable and never gets to protect itself by switching off. The tap has to survive its own bug
    for every one of the run's calls.
    """

    def offer(self, record):
        raise RuntimeError("deliberate tap failure")


class ExplodingTap(TRLTap):
    """Every surface the adapter has, failed. Nothing may reach TRL."""

    def _before(self, kwargs):
        try:
            raise RuntimeError("deliberate _before failure")
        except Exception as exc:
            self._note(exc)

    def _after(self, kwargs, guard):
        try:
            raise RuntimeError("deliberate _after failure")
        except Exception as exc:
            self._note(exc)

    def _on_step_end(self, state):
        raise RuntimeError("deliberate on_step_end failure")

    def _on_pre_optimizer_step(self, state, kwargs):
        raise RuntimeError("deliberate on_pre_optimizer_step failure")

    def _on_log(self, state, logs):
        raise RuntimeError("deliberate on_log failure")

    def _on_train_end(self, state):
        raise RuntimeError("deliberate on_train_end failure")


@pytest.fixture(scope="module")
def arms(tmp_path_factory):
    """All three arms, run once. Ordered so the untapped baseline is first in the process."""
    root = tmp_path_factory.mktemp("w4_1")
    baseline = run_untapped(root / "baseline")
    working = run_tapped(
        root / "tapped",
        tap=TRLTap(run_id="w4-1-acceptance", budget=GENEROUS, emit_extra=True),
        log_completions=True,
    )
    broken = run_tapped(
        root / "broken",
        tap=ExplodingTap(
            run_id="w4-1-broken",
            budget=GENEROUS,
            ring=ExplodingRing(capacity=4096),
        ),
    )
    repeat = run_untapped(root / "baseline-repeat")
    return {
        "baseline": baseline,
        "working": working,
        "broken": broken,
        "repeat": repeat,
        "root": root,
    }


# ---------------------------------------------------------------------------
# the comparison has to mean something first
# ---------------------------------------------------------------------------


def test_two_untapped_runs_are_byte_identical(arms):
    """Without this the byte-identical claim below would pass against a moving target."""
    assert arms["baseline"].digest == arms["repeat"].digest, (
        "two untapped runs at the same seed produced different weights, so a byte-identical "
        "comparison against the baseline proves nothing about the tap"
    )


# ---------------------------------------------------------------------------
# clause, second half: the run is unaffected by a deliberately failing tap
# ---------------------------------------------------------------------------


def test_deliberately_failing_tap_leaves_the_run_byte_identical(arms):
    assert arms["broken"].digest == arms["baseline"].digest, (
        "a tap that raises on every grader call and in every callback changed the trained weights"
    )


def test_the_failing_tap_really_did_fail_on_every_call(arms):
    """A tap that silently did nothing would also pass the test above. This is the control."""
    tap = arms["broken"].tap
    guard = tap.wrapped[0].guard
    assert guard.recorder_exceptions >= STEPS, (
        f"the recorder raised {guard.recorder_exceptions} times over {STEPS} steps; the exploding "
        f"ring should have raised at least once per grader call"
    )
    assert tap.adapter_exceptions >= STEPS
    assert any("deliberate" in key for key in tap.adapter_exception_keys)
    assert any("deliberate tap failure" in key for key in guard.distinct_recorder_exceptions)


def test_a_failing_tap_does_not_disable_itself_on_one_repeated_bug(arms):
    """One distinct exception on every call is one bug, and the guard survives it deliberately.

    ``DISTINCT_EXCEPTION_LIMIT`` is three *distinct* exceptions, not three occurrences. Disabling
    on the first would make a tap useless against a single edge case it can already survive, and
    the failing arm here is the demonstration: the same exception fires on every call of the run
    and the tap keeps going, keeps counting, and keeps the host untouched.
    """
    guard = arms["broken"].tap.wrapped[0].guard
    assert len(guard.distinct_recorder_exceptions) == 1
    assert not guard.disabled


def test_the_failing_tap_still_returns_the_graders_own_values(arms):
    """Fail-open means the host's data path is untouched, not merely that nothing raised."""
    logs = arms["broken"].trainer._logs
    rewards = list(logs["rewards"]["length_reward"])
    assert rewards, "no rewards reached TRL's own log under the failing tap"
    assert any(r == r for r in rewards), "every reward was NaN under the failing tap"


# ---------------------------------------------------------------------------
# a working tap is also invisible, which is a stronger statement than the clause
# ---------------------------------------------------------------------------


def test_a_working_tap_also_leaves_the_run_byte_identical(arms):
    """The clause only asks this of a broken tap. A working one has no excuse either.

    Nothing the tap does consumes randomness or touches a tensor: it reads counters, appends to a
    ring, and pushes three scalars into a metric dict. If this ever fails, something in the tap
    has started participating in the computation.
    """
    assert arms["working"].digest == arms["baseline"].digest


# ---------------------------------------------------------------------------
# clause, first half: a real GRPO run produces a complete record
# ---------------------------------------------------------------------------


def test_the_run_is_real(arms):
    """Weights moved, completions were sampled, advantages were computed. Not a mock."""
    from transformers import AutoModelForCausalLM

    trainer = arms["working"].trainer
    assert trainer.state.global_step == STEPS
    pristine = digest(AutoModelForCausalLM.from_pretrained(MODEL))
    assert arms["working"].digest != pristine, (
        "the trained weights match a freshly loaded checkpoint, so no optimizer step did anything"
    )
    logs = trainer._logs
    assert len(logs["completion"]) == BATCH
    assert len(logs["advantages"]) == BATCH
    assert any(abs(a) > 0 for a in logs["advantages"]), "every advantage was zero"


def test_the_record_has_all_five_levels(arms):
    run = arms["working"].tap.finish()
    steps = tuple(run.steps)
    assert len(steps) == STEPS, f"expected {STEPS} steps in the record, got {len(steps)}"
    groups = [g for s in steps for g in s.groups]
    assert len(groups) == STEPS * (BATCH // K)
    trajectories = [t for g in groups for t in g.trajectories]
    assert len(trajectories) == STEPS * BATCH
    assert all(len(t.turns) == 2 for t in trajectories), "prompt and completion are both turns"
    assert {t.turns[0].role for t in trajectories} == {"user"}
    assert {t.turns[1].role for t in trajectories} == {"assistant"}
    assert any(t.turns[1].text for t in trajectories), "no completion text reached the record"


def test_segment_provenance_is_present_on_every_trajectory(arms):
    """Mandatory and plural by the record format, and ``check_tiling`` enforces the cover.

    One segment here is a claim rather than a default: ``num_iterations`` and
    ``steps_per_generation`` are both 1, so every rollout was consumed by the optimizer step that
    generated it.
    """
    run = arms["working"].tap.finish()
    for step in run.steps:
        for group in step.groups:
            for traj in group.trajectories:
                assert traj.provenance, "a trajectory reached the record with no provenance"
                assert traj.provenance[0].turn_range == (0, len(traj.turns))
                assert traj.provenance[0].staleness_steps == 0
                assert traj.provenance[0].sampling.seed == SEED
                assert traj.provenance[0].engine.name == "transformers"


def test_the_record_carries_advantages_and_scores_and_abstentions(arms):
    """The score is a ``ScoreTree`` rather than a float, and an abstention stays an abstention."""
    import math

    from reward_lens.record.scores import Leaf, ScoreContext, evaluate

    run = arms["working"].tap.finish()
    trajectories = [t for s in run.steps for g in s.groups for t in g.trajectories]
    assert any(t.advantage is not None for t in trajectories)
    assert all(isinstance(t.scores, Leaf) for t in trajectories), (
        "one reward function at weight 1.0 is its own composition; a sum of one would put a node "
        "in the tree that the run does not have"
    )
    assert all(t.scores.name == "length_reward" for t in trajectories)
    assert any(t.scores.value is not None for t in trajectories)

    abstained = [t for t in trajectories if t.scores.abstained]
    assert abstained, "the grader abstained once per batch and none of it reached the record"
    assert all(t.scores.value is None for t in abstained)
    assert all(math.isnan(evaluate(t.scores, ScoreContext())) for t in abstained), (
        "an abstention became a zero on the way into the record, which is the failure mode the "
        "whole score channel exists to prevent"
    )
    assert all(t.features == {} for t in abstained), "and it must not become a realised reward"


def test_the_leaf_points_back_at_the_grader_call_that_produced_it(arms):
    """The record and the tap's own log are joinable, which is what makes B4 possible later."""
    run = arms["working"].tap.finish()
    steps = tuple(run.steps)
    refs = [t.scores.grader_call for s in steps for g in s.groups for t in g.trajectories]
    assert all(r is not None for r in refs)
    assert {r.grader for r in refs} == {"length_reward"}
    assert {r.outcome for r in refs} == {"returned"}
    assert sorted({r.step for r in refs}) == list(range(STEPS))
    assert sorted({r.seq for r in refs}) == list(range(1, STEPS + 1))


def test_the_record_keeps_the_reward_trl_actually_used_beside_the_trees_refusal(arms):
    run = arms["working"].tap.finish()
    scored = [
        t for s in run.steps for g in s.groups for t in g.trajectories if not t.scores.abstained
    ]
    assert scored
    assert all(t.features["trl_realised_reward"] == pytest.approx(t.scores.value) for t in scored)


def test_group_stats_count_the_abstentions_rather_than_scoring_them(arms):
    run = arms["working"].tap.finish()
    stats = [g.group_stats for s in run.steps for g in s.groups]
    assert all(s.k == K for s in stats)
    assert sum(s.n_abstained for s in stats) > 0


def test_the_estimator_spec_says_exactly_how_scores_became_advantages(arms):
    """The format asks for EXACTLY, and GRPO is one of the estimators where that is achievable."""
    run = arms["working"].tap.finish()
    spec = next(iter(run.steps)).groups[0].estimator
    assert spec.family.startswith("grpo/")
    assert spec.group_centred is True
    assert spec.std_normalised is True
    assert spec.std_epsilon == 1e-4
    assert spec.clip_low is not None and spec.clip_high is not None
    assert spec.extra["scale_rewards"] == "group"
    assert spec.extra["multi_objective_aggregation"] == "sum_then_normalize"


def test_the_sampling_policy_says_what_fraction_was_recorded(arms):
    run = arms["working"].tap.finish()
    policy = run.sampling_policy
    assert policy.scheme is SamplingScheme.FULL
    assert policy.recorded == STEPS
    assert policy.unit == "step"
    assert "0 dropped" in policy.notes, "the ring's drop count has to travel with the record"


def test_the_run_declares_only_the_regime_conditions_the_config_settles(arms):
    run = arms["working"].tap.finish()
    declared = run.regime.declared
    assert declared[RegimeCondition.NEAR_POLICY] is True
    assert declared[RegimeCondition.NO_COMPACTION] is True
    assert declared[RegimeCondition.MASK_STABLE] is True
    assert RegimeCondition.STATIONARY_GRADER not in declared, (
        "TRL hands the grader trainer_state so it can shape reward by training progress; nothing "
        "on the config says whether this grader does, so declaring it either way is a guess"
    )
    assert run.regime.declared_by


def test_access_claims_only_the_rung_a_record_supports(arms):
    run = arms["working"].tap.finish()
    assert run.access[Component.GRADER] is Access.RECORD
    assert Access.QUERY not in run.access[Component.GRADER]
    assert run.lineage.framework == "trl"
    assert run.lineage.framework_version == trl.__version__


def test_the_record_survives_a_round_trip_through_the_store(arms):
    """ "Complete" has to mean serialisable, or it means a pile of live objects in one process.

    ``RecordWriter`` and ``open_run`` are the boundary between Plane A's output and Plane B's
    input, and everything downstream of the tap reads the record rather than the tap. So the run is
    written, reopened from disk, and the levels are counted again on the way back.
    """
    from reward_lens.record.reader import open_run
    from reward_lens.record.writer import RecordWriter

    root = arms["root"] / "store"
    run = arms["working"].tap.finish()
    report = RecordWriter(root).write(run)
    assert report.steps == STEPS
    assert report.trajectories == STEPS * BATCH
    assert report.turns == STEPS * BATCH * 2

    reopened = open_run(root, str(run.id))
    steps = tuple(reopened.steps)
    assert [s.index for s in steps] == list(range(STEPS))
    trajectories = [t for s in steps for g in s.groups for t in g.trajectories]
    assert len(trajectories) == STEPS * BATCH
    assert all(t.provenance for t in trajectories), "provenance did not survive the round trip"
    assert reopened.sampling_policy.scheme is SamplingScheme.FULL
    assert reopened.lineage.framework == "trl"
    assert steps[0].groups[0].estimator.family.startswith("grpo/")
    assert any(t.turns[1].text for t in trajectories)


def test_every_step_carries_its_own_instrument_effect(arms):
    """The format puts the effect on the step, and it is a term in the budget rather than prose."""
    run = arms["working"].tap.finish()
    effects = [s.instrument for s in run.steps]
    assert all(e.invocations > 0 for e in effects)
    assert all(e.disabled is False for e in effects)
    assert all(e.breach is None for e in effects)
    assert max(e.added_latency_ms_p99 for e in effects) < 1.0


# ---------------------------------------------------------------------------
# the instrumentation channel: log_metric and log_extra rather than a parallel one
# ---------------------------------------------------------------------------


def test_the_taps_metrics_arrive_in_trls_own_metric_stream(arms):
    """The clause is specific about using ``log_metric`` rather than building a side channel.

    ``_log_metric`` (``grpo_trainer.py:1598``) appends into ``_pending_metrics``; the flush at
    2769-2774 averages within the step, gathers across processes, and appends into ``_metrics``,
    from where ``log()`` puts it in ``state.log_history`` alongside ``kl`` and ``entropy``.
    """
    history = arms["working"].trainer.state.log_history
    logged = [entry for entry in history if METRIC_NAMES[0] in entry]
    assert logged, f"{METRIC_NAMES[0]} never reached TRL's log history"
    assert all(entry[METRIC_NAMES[0]] > 0 for entry in logged)
    assert any(METRIC_NAMES[1] in entry for entry in history)
    assert all(entry.get(METRIC_NAMES[2], 1.0) == 1.0 for entry in logged)


def test_the_taps_extra_column_arrives_in_trls_completions_table(arms):
    """``log_extra`` writes a column into the parquet TRL already produces per logging step."""
    pd = pytest.importorskip("pandas")
    files = sorted((arms["root"] / "tapped" / "completions").glob("*.parquet"))
    assert files, "log_completions=True produced no completions table"
    frame = pd.read_parquet(files[-1])
    assert "reward_lens_seq" in frame.columns
    assert len(frame) == BATCH
    assert frame["reward_lens_seq"].nunique() == 1


def test_metric_key_set_is_invariant_to_tap_state(arms):
    """The key set must not depend on whether the tap is working, because the flush is collective.

    ``for name in sorted(self._pending_metrics)`` at 2769 calls ``accelerator.gather`` once per
    name. Sorting fixes the order across ranks; it does not fix the set. A tap that emitted only
    while enabled would give a breached rank a different key set from its peers, and a rank that
    enters a collective loop a different number of times hangs. So the names are constant.
    """
    working = arms["working"].tap
    broken = arms["broken"].tap
    assert working.metric_names() == METRIC_NAMES
    assert broken.metric_names() == METRIC_NAMES
    assert working.metric_names() == broken.metric_names()


# ---------------------------------------------------------------------------
# the second call site, and the branch that decides which one is used
# ---------------------------------------------------------------------------


def test_an_async_grader_stays_async_through_the_wrapper():
    """TRL branches on ``iscoroutinefunction`` at 1654, and it does not follow ``__wrapped__``.

    A synchronous wrapper around an ``async def`` grader would move it from the ``await`` call
    site at 1671-1673 onto the synchronous one at 1659-1661, where TRL iterates the return value
    and would get a coroutine. That is a host crash caused by the tap, and it is the reason the
    wrapper mirrors coroutine-ness instead of using one shape for everything.
    """
    import inspect

    async def async_grader(prompts, completions, completion_ids, **kwargs):
        return [1.0] * len(completions)

    tap = TRLTap(run_id="async-check", budget=GENEROUS)
    wrapped = tap.wrap(async_grader)
    assert inspect.iscoroutinefunction(wrapped)
    assert wrapped.__name__ == "async_grader"


def test_wrapping_does_not_rename_the_graders_metrics():
    """``get_callable_name`` fills ``reward_func_names``, which becomes ``rewards/{name}/mean``."""
    from trl.trainer.utils import get_callable_name

    tap = TRLTap(run_id="name-check", budget=GENEROUS)
    assert get_callable_name(tap.wrap(length_reward)) == get_callable_name(length_reward)


def test_a_reward_model_is_passed_through_rather_than_broken():
    """TRL checks ``isinstance(reward_func, nn.Module)`` at 1638 before it considers calling it."""
    model = torch.nn.Linear(2, 1)
    tap = TRLTap(run_id="module-check", budget=GENEROUS)
    out = tap.wrap([length_reward, model])
    assert out[1] is model, "wrapping a reward model would move it onto the callable branch"
    assert len(tap.skipped) == 1
    assert "nn.Module" in tap.skipped[0].reason


# ---------------------------------------------------------------------------
# the step index, and the off-by-one an analyst would otherwise hit
# ---------------------------------------------------------------------------


def test_step_index_seen_by_grader_is_pre_increment(arms):
    """The grader sees ``global_step`` before the optimizer increments it. The table sees after.

    So the first batch is step 0 to the tap and lands in ``completions_00001.parquet``. Recording
    the grader's own view keeps the tap's records aligned with the call that produced them, and
    this is where the difference is written down rather than rediscovered.
    """
    tap = arms["working"].tap
    indices = sorted(tap._buckets)
    assert indices == list(range(STEPS)), f"expected steps 0..{STEPS - 1}, got {indices}"
    files = sorted(p.name for p in (arms["root"] / "tapped" / "completions").glob("*.parquet"))
    assert files[0] == "completions_00001.parquet"
    assert len(files) == STEPS


def test_on_pre_optimizer_step_fires_and_carries_the_model(arms):
    """Recorded because it is the only moment in any framework with live per-parameter gradients.

    ``transformers/trainer.py`` calls it at 1772, between ``_clip_grad_norm`` at 1769 and
    ``optimizer.step()`` at 1773. The specification says the hook passes no model; it does, via
    ``callback_handler.call_event`` at ``trainer_callback.py:543-556``, which injects ``model``,
    ``optimizer``, ``lr_scheduler`` and both dataloaders into every hook. G1 is what will use the
    gradients; this only establishes that the seam is real and fires once per step.
    """
    tap = arms["working"].tap
    seen = [b.grads for b in tap._buckets.values() if b.grads]
    assert len(seen) == STEPS, f"the hook fired on {len(seen)} of {STEPS} steps"
    assert all(g["model_in_kwargs"] for g in seen)
    assert all(g["optimizer_in_kwargs"] for g in seen)
    assert all(g["param_tensors_with_grad"] == g["param_tensors"] > 0 for g in seen), (
        "every parameter should carry a live, already-clipped gradient at this hook"
    )


def test_the_clipped_grad_norm_reaches_the_record(arms):
    """``_clip_grad_norm`` returns the norm at 1769 and the hook is not given it, so it comes from
    the log payload instead, where it is already clipped."""
    run = arms["working"].tap.finish()
    norms = [s.optimizer.grad_norm_clipped for s in run.steps]
    assert sum(n is not None for n in norms) >= STEPS - 1
    assert all(n >= 0 for n in norms if n is not None)


def test_kl_and_entropy_reach_the_record(arms):
    run = arms["working"].tap.finish()
    entropies = [s.optimizer.entropy for s in run.steps]
    assert sum(e is not None for e in entropies) >= STEPS - 1


# ---------------------------------------------------------------------------
# the measured instrument effect, which is the number to publish
# ---------------------------------------------------------------------------


def test_the_instrument_effect_is_measured_and_composes(arms):
    effect = arms["working"].tap.instrument_effect
    assert effect.calls == STEPS
    assert effect.added_ns_p50 > 0, "an overhead of exactly zero means it was not measured"
    assert effect.enabled and not effect.breaches
    term = effect.as_term("trl_tap_overhead")
    assert term.kind == "B" and term.distribution == "rectangular"
    assert term.value >= 0.0 and term.contribution >= 0.0


def test_the_measured_overhead_from_outside(arms):
    """The published number: the whole wrapped stack against the bare callable, from outside.

    The guard times its own recorder segments, which is what it can do from inside and is
    necessarily an underestimate: the ring offer, the adapter's step read and the metric push all
    sit outside the timed region. So the number that gets published is this one, measured the way
    a user would measure it, with the arguments shaped the way TRL shapes them.

    Three levels rather than two, because a single "wrapped versus bare" number cannot say which
    layer costs what, and the grader wrapper already published its own from-outside figure of 1.89
    to 2.03 microseconds. Separating them is what makes this measurement comparable to that one
    instead of replacing it.

    **The bare number is not a stand-in for a real grader.** Scoring 32 completions by their length
    takes about 1.6 microseconds, and a real grader is a parse (milliseconds) or a judge over HTTP
    (seconds). So the fraction printed here is the worst case by three to six orders of magnitude
    and it is reported as a ceiling, not as a typical value.

    Not asserted tightly. A box under load is not a measurement instrument and a threshold that
    fails on a noisy runner gets deleted within a month. The assertion is the order of magnitude
    and the direction; the number is printed and goes in the report.
    """
    n_rows = BATCH * K
    completions = ["a completion of a realistic length, roughly forty chars"] * n_rows
    prompts = ["a prompt"] * n_rows
    ids = [[1, 2, 3]] * n_rows

    class FakeState:
        global_step = 7

    pushed: list[tuple[str, float]] = []
    kwargs = {
        "trainer_state": FakeState(),
        "log_metric": lambda name, value: pushed.append((name, value)),
        "log_extra": lambda column, values: None,
    }

    tap = TRLTap(run_id="bench", budget=GENEROUS, emit_extra=True)
    full = tap.wrap(length_reward)
    wrapper_only = full.instrumented

    def timed(fn, reps: int) -> list[int]:
        out = []
        for _ in range(reps):
            t0 = time.perf_counter_ns()
            fn(prompts=prompts, completions=completions, completion_ids=ids, **kwargs)
            out.append(time.perf_counter_ns() - t0)
        return out

    reps = 4000
    for fn in (length_reward, wrapper_only, full):
        timed(fn, 200)
    bare = timed(length_reward, reps)
    w3_1 = timed(wrapper_only, reps)
    w4_1 = timed(full, reps)

    def med(xs):
        return statistics.median(xs)

    def p99(xs):
        return statistics.quantiles(xs, n=100)[98]

    print(
        f"\nW4.1 instrument effect, measured from outside, {reps} calls per level,\n"
        f"one call = {n_rows} completions, {len(METRIC_NAMES)} metrics pushed, "
        f"one extra column of {n_rows} values:\n"
        f"  bare grader                    {med(bare) / 1000:8.3f} us median  "
        f"{p99(bare) / 1000:8.3f} us p99\n"
        f"  + instrument_grader           {med(w3_1) / 1000:8.3f} us median  "
        f"{p99(w3_1) / 1000:8.3f} us p99\n"
        f"  + the TRL adapter             {med(w4_1) / 1000:8.3f} us median  "
        f"{p99(w4_1) / 1000:8.3f} us p99\n"
        f"  ---\n"
        f"  the grader wrapper adds        {(med(w3_1) - med(bare)) / 1000:8.3f} us median  "
        f"{(p99(w3_1) - p99(bare)) / 1000:8.3f} us p99\n"
        f"  the adapter adds on top        {(med(w4_1) - med(w3_1)) / 1000:8.3f} us median  "
        f"{(p99(w4_1) - p99(w3_1)) / 1000:8.3f} us p99\n"
        f"  whole stack adds               {(med(w4_1) - med(bare)) / 1000:8.3f} us median  "
        f"{(p99(w4_1) - p99(bare)) / 1000:8.3f} us p99\n"
        f"  as a fraction of this (trivial) grader's own time: "
        f"{(med(w4_1) - med(bare)) / med(bare):.2%}\n"
    )
    added = med(w4_1) - med(bare)
    assert 0 < added < 50_000, (
        f"the whole stack added {added} ns per call. Below zero is a broken measurement; above 50 "
        f"us means something on the hot path has started doing work that scales, which is the "
        f"defect a RingStats read on this path caused once already."
    )
    assert med(w4_1) >= med(w3_1) >= med(bare), "the levels are not ordered; the measurement is bad"
    assert len(pushed) == (reps + 200) * len(METRIC_NAMES)


def test_the_tap_is_a_small_fraction_of_the_run(arms):
    """The run-level number, which is the one a lab actually cares about.

    Two things it is not. It is not a comparison of the two arms' wall clocks: those differ by
    several percent in either direction between repeats of the same arm, so the difference is
    machine noise and not a measurement of anything. And the per-call median here rests on twelve
    samples, which is a weak quantile; the microbenchmark above is the one with a distribution
    behind it.
    """
    tap = arms["working"].tap
    effect = tap.instrument_effect
    tapped_seconds = arms["working"].seconds
    fraction_of_run = effect.added_ns_total / 1e9 / tapped_seconds
    print(
        f"\nW4.1 in-run instrument effect over {STEPS} steps "
        f"({effect.window_n} calls behind the quantiles, so the p99 here is the largest of "
        f"{effect.window_n} rather than a distribution):\n"
        f"  tap added        {effect.added_ns_total / 1e3:9.1f} us total, "
        f"{effect.added_ns_p50 / 1e3:.3f} us median per call\n"
        f"  grader itself    {effect.inner_ns_total / 1e3:9.1f} us total\n"
        f"  training wall    {tapped_seconds:9.3f} s  (untapped {arms['baseline'].seconds:.3f} s)\n"
        f"  tap as a fraction of training wall clock: {fraction_of_run:.3e}\n"
    )
    assert fraction_of_run < 1e-3, (
        "the tap took more than a thousandth of the run's wall clock, which for a wrapper that "
        "appends to a list means something on the hot path is doing work"
    )


def test_the_ring_dropped_nothing(arms):
    stats = arms["working"].tap.handle.ring.stats()
    assert stats.dropped == 0, f"the ring dropped records: {stats.render()}"
    assert stats.accepted == STEPS


def test_every_recorded_call_is_a_returned_call(arms):
    """The grader never raised, so every outcome should be RETURNED. A SENTINEL here would mean
    the whole call abstained, which is different from the per-row abstentions inside it."""
    tap = arms["working"].tap
    outcomes = {c.outcome for b in tap._buckets.values() for c in b.calls}
    assert outcomes == {CallOutcome.RETURNED}

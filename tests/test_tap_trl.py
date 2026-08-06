"""The TRL adapter, tested without TRL.

Everything here runs on a base install. That is not a convenience, it is the design being checked:
``reward_lens.tap.adapters.trl`` imports its framework lazily, inside functions, so importing it
pulls no torch, no transformers and no trl, and the parts that build a record out of what a run
produced are ordinary Python over ordinary objects. A fake trainer with the four attributes the
adapter reads is enough to drive the whole step boundary.

The real run lives in ``tests/acceptance/test_w4_1_trl_tap.py`` and needs all three. What is here
is the half that should still be exercised on a machine that has none of them, plus the properties
that are easier to falsify with a fake than with a trainer: a disabled tap that still emits the
same metric keys, a ragged extra column that never happens, and a step buffer that overflows into
an honest sampling policy rather than a silent truncation.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import math
import subprocess
import sys

import pytest

from reward_lens.core.envelope import RegimeCondition
from reward_lens.core.types import Access, Component
from reward_lens.record.schema import SamplingScheme
from reward_lens.record.scores import ScoreContext, WeightedSum, evaluate
from reward_lens.tap.adapters.trl import (
    EXTRA_COLUMN,
    METRIC_NAMES,
    TRLTap,
    _as_float,
    _is_torch_module,
    _total,
)
from reward_lens.tap.contract import BreachKind, CallOutcome, TapBreach, TapBudget
from reward_lens.tap.ring import TapRing

GENEROUS = TapBudget(
    max_added_latency_ms_p99=1000.0,
    max_resident_bytes=16 * 1024 * 1024,
    max_added_alloc_bytes_per_step=16 * 1024 * 1024,
)


# ---------------------------------------------------------------------------
# a trainer-shaped object, with only what the adapter actually reads off one
# ---------------------------------------------------------------------------


class FakeArgs:
    """``GRPOConfig``, cut down to the fields ``_read_config`` asks for.

    Deliberately missing several of them, because that is the case that matters: TRL drops config
    fields between releases (``max_prompt_length`` was on ``GRPOConfig`` when the specification was
    written and is not on 1.9.2) and a config read that raises on a missing field would take down a
    training loop over a rename.
    """

    beta = 0.04
    temperature = 0.9
    top_p = 1.0
    epsilon = 0.2
    epsilon_high = None
    delta = None
    seed = 11
    loss_type = "grpo"
    scale_rewards = "group"
    multi_objective_aggregation = "sum_then_normalize"
    importance_sampling_level = "token"
    mask_truncated_completions = False
    num_iterations = 1
    steps_per_generation = 1
    generation_batch_size = 4
    max_completion_length = 16
    num_generations = 2
    reward_weights = None
    log_completions = True
    use_vllm = False
    optim = "adamw_torch"
    per_device_train_batch_size = 4


class FakeState:
    def __init__(self, global_step: int = 0) -> None:
        self.global_step = global_step


class FakeTrainer:
    """The four things the callback reads: ``args``, ``num_generations``, ``_logs``, ``_metrics``."""

    def __init__(self) -> None:
        self.args = FakeArgs()
        self.num_generations = 2
        self.model = None
        self.callbacks: list = []
        self._logs = {
            "prompt": [],
            "completion": [],
            "advantages": [],
            "rewards": {},
        }
        self._metrics = {"train": {}}

    def add_callback(self, callback) -> None:
        self.callbacks.append(callback)


def make_kwargs(n: int = 4, step: int = 0):
    """The call shape TRL builds at ``grpo_trainer.py:1618-1631``, with the two live callbacks."""
    pushed_metrics: list[tuple[str, float]] = []
    pushed_extra: list[tuple[str, list]] = []
    kwargs = {
        "trainer_state": FakeState(step),
        "log_metric": lambda name, value: pushed_metrics.append((name, value)),
        "log_extra": lambda column, values: pushed_extra.append((column, list(values))),
        # Line 1618 splats every remaining dataset column in, so the key set is open-ended.
        "difficulty": ["easy"] * n,
        "answer": ["42"] * n,
    }
    return kwargs, pushed_metrics, pushed_extra


def grader(prompts, completions, completion_ids, **kwargs):
    return [len(c) / 10.0 for c in completions]


def drive(tap: TRLTap, trainer: FakeTrainer, *, steps: int, rows: int = 4):
    """Run the whole loop by hand: wrap, attach, call the grader, fire the callbacks."""
    wrapped = tap.wrap(grader)
    tap.attach(trainer)
    callback = trainer.callbacks[0]
    for step in range(steps):
        kwargs, metrics, extra = make_kwargs(rows, step)
        prompts = [f"p{step}"] * rows
        completions = [f"completion {step} row {i}" for i in range(rows)]
        wrapped(
            prompts=prompts,
            completions=completions,
            completion_ids=[[1, 2]] * rows,
            **kwargs,
        )
        state = FakeState(step + 1)
        trainer._logs["prompt"] = prompts
        trainer._logs["completion"] = completions
        trainer._logs["advantages"] = [0.5, -0.5] * (rows // 2)
        trainer._metrics["train"] = {"entropy": [1.25], "kl": [0.01]}
        callback.on_step_end(trainer.args, state, None)
        callback.on_log(trainer.args, state, None, logs={"grad_norm": 0.3, "learning_rate": 1e-6})
    callback.on_train_end(trainer.args, FakeState(steps), None)
    return wrapped, metrics, extra


# ---------------------------------------------------------------------------
# the import graph
# ---------------------------------------------------------------------------


def test_importing_the_adapter_pulls_no_framework():
    """``tap/`` is torch-free and an adapter must not be the thing that changes that.

    A subprocess rather than a check on this one, because the acceptance test in the same session
    imports all three and ``sys.modules`` would already be poisoned.
    """
    code = (
        "import sys; import reward_lens.tap.adapters.trl as m; "
        "print(sorted(k for k in ('torch','trl','transformers','datasets') if k in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", f"the adapter imported {out.stdout.strip()} at module scope"


def test_the_adapters_package_star_import_pulls_nothing():
    module = importlib.import_module("reward_lens.tap.adapters")
    assert module.__all__ == []


# ---------------------------------------------------------------------------
# wrap: the three properties TRL reads off the callable
# ---------------------------------------------------------------------------


def test_wrap_preserves_the_name_that_becomes_the_metric_key():
    """``get_callable_name`` fills ``reward_func_names``, which becomes ``rewards/{name}/mean``."""
    tap = TRLTap(budget=GENEROUS)
    assert tap.wrap(grader).__name__ == "grader"


def test_wrap_preserves_the_signature_the_dispatch_reads():
    tap = TRLTap(budget=GENEROUS)
    assert inspect.signature(tap.wrap(grader)) == inspect.signature(grader)


def test_wrap_mirrors_coroutine_ness_so_the_call_site_does_not_move():
    """TRL branches on ``iscoroutinefunction`` at 1654 and does not follow ``__wrapped__``.

    A sync wrapper round an async grader would move it from the ``await`` site at 1671-1673 onto
    the synchronous one at 1659-1661, where TRL iterates the return value and would be handed a
    coroutine. That is a host crash caused by the tap.
    """

    async def async_grader(prompts, completions, completion_ids, **kwargs):
        return [1.0] * len(completions)

    tap = TRLTap(budget=GENEROUS)
    wrapped = tap.wrap(async_grader)
    assert inspect.iscoroutinefunction(wrapped)
    assert not inspect.iscoroutinefunction(tap.wrap(grader))


def test_an_async_grader_is_recorded_through_the_await_call_site():
    async def async_grader(prompts, completions, completion_ids, **kwargs):
        return [2.0] * len(completions)

    tap = TRLTap(budget=GENEROUS)
    wrapped = tap.wrap(async_grader)
    kwargs, metrics, _ = make_kwargs(2, step=3)
    out = asyncio.run(
        wrapped(prompts=["p"] * 2, completions=["c"] * 2, completion_ids=[[1]] * 2, **kwargs)
    )
    assert out == [2.0, 2.0]
    records = tap.handle.ring.drain()
    assert len(records) == 1
    assert records[0].step == 3
    assert {name for name, _ in metrics} == set(METRIC_NAMES)


def test_wrap_returns_the_shape_it_was_given():
    tap = TRLTap(budget=GENEROUS)
    assert callable(tap.wrap(grader))
    assert isinstance(TRLTap(budget=GENEROUS).wrap([grader]), list)


def test_a_reward_model_is_passed_through_and_recorded_as_skipped():
    """TRL checks ``isinstance(reward_func, nn.Module)`` at 1638 before it considers calling it."""

    class FakeModule:
        __module__ = "torch.nn.modules.module"

    module = FakeModule()
    tap = TRLTap(budget=GENEROUS)
    out = tap.wrap([grader, module])
    assert out[1] is module
    assert len(tap.skipped) == 1
    assert tap.skipped[0].kind == "reward model"
    assert "1638" in tap.skipped[0].reason


def test_is_torch_module_does_not_import_torch_to_answer():
    class NotAModule:
        pass

    assert _is_torch_module(NotAModule()) is False


# ---------------------------------------------------------------------------
# fidelity: what the host gets back
# ---------------------------------------------------------------------------


def test_the_host_gets_back_the_graders_own_object():
    """Identity, not equality. Equality would pass for a copy and a copy is the bug."""
    payload = [1.0, 2.0, 3.0]

    def returns_the_same_list(prompts, completions, completion_ids, **kwargs):
        return payload

    tap = TRLTap(budget=GENEROUS)
    wrapped = tap.wrap(returns_the_same_list)
    kwargs, _, _ = make_kwargs(3)
    out = wrapped(prompts=["p"] * 3, completions=["c"] * 3, completion_ids=[[1]] * 3, **kwargs)
    assert out is payload


def test_a_grader_exception_reaches_the_host_unchanged():
    """The opposite requirement from the recorder's, and a wrapper that swallows would alter the
    host's behaviour, which is the one thing Plane A may never do."""

    def raises(prompts, completions, completion_ids, **kwargs):
        raise ValueError("the grader itself failed")

    tap = TRLTap(budget=GENEROUS)
    wrapped = tap.wrap(raises)
    kwargs, _, _ = make_kwargs(1)
    with pytest.raises(ValueError, match="the grader itself failed"):
        wrapped(prompts=["p"], completions=["c"], completion_ids=[[1]], **kwargs)
    record = tap.handle.ring.drain()[0]
    assert record.outcome is CallOutcome.RAISED
    assert record.error_type == "ValueError"


def test_a_failing_recorder_never_reaches_the_host():
    class ExplodingRing(TapRing):
        def offer(self, record):
            raise RuntimeError("deliberate")

    tap = TRLTap(budget=GENEROUS, ring=ExplodingRing(capacity=8))
    wrapped = tap.wrap(grader)
    kwargs, _, _ = make_kwargs(2)
    out = wrapped(prompts=["p"] * 2, completions=["cc"] * 2, completion_ids=[[1]] * 2, **kwargs)
    assert out == [0.2, 0.2]
    assert wrapped.guard.recorder_exceptions == 1


# ---------------------------------------------------------------------------
# the two channel hazards
# ---------------------------------------------------------------------------


def test_the_metric_key_set_does_not_depend_on_whether_the_tap_works():
    """The deadlock guard, asserted rather than promised.

    ``grpo_trainer.py:2769-2774`` iterates ``sorted(self._pending_metrics)`` and calls
    ``accelerator.gather`` once per name. Sorting fixes the order across ranks; it does not fix the
    set. Plane A disables per process, so a tap that emitted only while enabled would give a
    breached rank a different key set from its peers, and a rank that enters a collective loop a
    different number of times hangs rather than raising. Fixed names on every call is the fix.
    """
    tap = TRLTap(budget=GENEROUS)
    wrapped = tap.wrap(grader)
    kwargs, metrics, _ = make_kwargs(2)
    args = {"prompts": ["p"] * 2, "completions": ["cc"] * 2, "completion_ids": [[1]] * 2}

    wrapped(**args, **kwargs)
    while_enabled = [name for name, _ in metrics]

    wrapped.guard.disable(
        TapBreach(kind=BreachKind.LATENCY_P99, observed=9.0, limit=1.0, at_call=1)
    )
    assert wrapped.guard.disabled
    metrics.clear()
    wrapped(**args, **kwargs)
    while_disabled = [name for name, _ in metrics]

    assert while_enabled == while_disabled == list(METRIC_NAMES)


def test_a_disabled_tap_reports_itself_as_disabled_in_the_metric_it_still_emits():
    tap = TRLTap(budget=GENEROUS)
    wrapped = tap.wrap(grader)
    kwargs, metrics, _ = make_kwargs(2)
    args = {"prompts": ["p"] * 2, "completions": ["cc"] * 2, "completion_ids": [[1]] * 2}
    wrapped(**args, **kwargs)
    assert dict(metrics)[METRIC_NAMES[2]] == 1.0
    wrapped.guard.disable(
        TapBreach(kind=BreachKind.LATENCY_P99, observed=9.0, limit=1.0, at_call=1)
    )
    metrics.clear()
    wrapped(**args, **kwargs)
    assert dict(metrics)[METRIC_NAMES[2]] == 0.0


def test_the_extra_column_is_always_one_value_per_completion():
    """``log()`` at 3314-3321 hands the extra columns to ``pd.DataFrame``, which raises on ragged
    columns. A partial push, or a push on some batches and not others, crashes the host inside its
    own logging, so the length is checked here rather than discovered there."""
    tap = TRLTap(budget=GENEROUS, emit_extra=True)
    wrapped = tap.wrap(grader)
    for rows in (1, 4, 7):
        kwargs, _, extra = make_kwargs(rows)
        wrapped(
            prompts=["p"] * rows,
            completions=["cc"] * rows,
            completion_ids=[[1]] * rows,
            **kwargs,
        )
        assert len(extra) == 1
        column, values = extra[0]
        assert column == EXTRA_COLUMN
        assert len(values) == rows


def test_the_extra_column_is_still_emitted_when_the_tap_is_disabled():
    """Same reason as the metric names, and a sharper consequence: skipping the column on a
    disabled step leaves TRL's own parquet holding a column that no longer lines up with the
    prompts beside it."""
    tap = TRLTap(budget=GENEROUS, emit_extra=True)
    wrapped = tap.wrap(grader)
    wrapped.guard.disable(
        TapBreach(kind=BreachKind.RESIDENT_BYTES, observed=9.0, limit=1.0, at_call=1)
    )
    kwargs, _, extra = make_kwargs(3)
    wrapped(prompts=["p"] * 3, completions=["cc"] * 3, completion_ids=[[1]] * 3, **kwargs)
    assert len(extra) == 1 and len(extra[0][1]) == 3


def test_the_extra_column_is_off_by_default():
    tap = TRLTap(budget=GENEROUS)
    wrapped = tap.wrap(grader)
    kwargs, _, extra = make_kwargs(3)
    wrapped(prompts=["p"] * 3, completions=["cc"] * 3, completion_ids=[[1]] * 3, **kwargs)
    assert extra == []


def test_the_open_ended_kwargs_do_not_break_anything():
    """Line 1618 splats every remaining dataset column into ``reward_kwargs``, so no tap may assume
    a closed key set. ``make_kwargs`` puts two dataset columns in and nothing here looks at them."""
    tap = TRLTap(budget=GENEROUS)
    wrapped = tap.wrap(grader)
    kwargs, _, _ = make_kwargs(2)
    assert "difficulty" in kwargs and "answer" in kwargs
    wrapped(prompts=["p"] * 2, completions=["cc"] * 2, completion_ids=[[1]] * 2, **kwargs)
    record = tap.handle.ring.drain()[0]
    assert "difficulty" in record.kwargs


# ---------------------------------------------------------------------------
# the step boundary and the record
# ---------------------------------------------------------------------------


def test_the_step_index_comes_from_the_trainer_state():
    tap = TRLTap(budget=GENEROUS)
    wrapped = tap.wrap(grader)
    for step in (0, 5, 41):
        kwargs, _, _ = make_kwargs(2, step)
        wrapped(prompts=["p"] * 2, completions=["cc"] * 2, completion_ids=[[1]] * 2, **kwargs)
    assert [r.step for r in tap.handle.ring.drain()] == [0, 5, 41]


def test_the_record_has_all_five_levels():
    tap = TRLTap(run_id="fake", budget=GENEROUS)
    drive(tap, FakeTrainer(), steps=3, rows=4)
    run = tap.finish()
    steps = tuple(run.steps)
    assert [s.index for s in steps] == [0, 1, 2]
    groups = [g for s in steps for g in s.groups]
    assert len(groups) == 3 * 2  # four rows, num_generations 2
    trajectories = [t for g in groups for t in g.trajectories]
    assert len(trajectories) == 3 * 4
    assert all(len(t.turns) == 2 for t in trajectories)
    assert all(t.turns[0].role == "user" and t.turns[1].role == "assistant" for t in trajectories)
    assert all(t.provenance[0].turn_range == (0, 2) for t in trajectories)


def test_the_record_carries_the_scores_the_grader_returned():
    tap = TRLTap(run_id="fake", budget=GENEROUS)
    drive(tap, FakeTrainer(), steps=2, rows=4)
    run = tap.finish()
    first = next(iter(run.steps)).groups[0].trajectories[0]
    assert first.scores.name == "grader"
    assert first.scores.value == pytest.approx(len("completion 0 row 0") / 10.0)
    assert first.scores.abstained is False
    assert evaluate(first.scores, ScoreContext()) == pytest.approx(first.scores.value)
    assert first.advantage == 0.5


def test_one_unweighted_grader_is_its_own_tree_rather_than_a_sum_of_one():
    """``node_names`` is part of the record's identity, so a node the run does not have does not
    belong in it. A single reward function at weight 1.0 is a leaf."""
    from reward_lens.record.scores import Leaf

    tap = TRLTap(run_id="fake", budget=GENEROUS)
    drive(tap, FakeTrainer(), steps=1, rows=2)
    tree = next(iter(tap.finish().steps)).groups[0].trajectories[0].scores
    assert isinstance(tree, Leaf)


def test_the_leaf_points_back_at_the_grader_call_that_produced_it():
    tap = TRLTap(run_id="fake", budget=GENEROUS)
    drive(tap, FakeTrainer(), steps=2, rows=4)
    steps = tuple(tap.finish().steps)
    ref = steps[1].groups[0].trajectories[0].scores.grader_call
    assert ref is not None
    assert ref.grader == "grader"
    assert ref.outcome == "returned"
    assert ref.step == 1
    assert ref.facets["rows_in_call"] == 4
    assert ref.facets["call_latency_s"] >= 0.0
    assert ref.latency_s is None, (
        "TRL scores a whole batch per call, so a per-row latency would be the batch's and anybody "
        "summing the column would get a number B x G times too large"
    )


def test_the_estimator_spec_is_read_off_the_config():
    tap = TRLTap(run_id="fake", budget=GENEROUS)
    drive(tap, FakeTrainer(), steps=1)
    spec = next(iter(tap.finish().steps)).groups[0].estimator
    assert spec.family == "grpo/grpo"
    assert spec.group_centred and spec.std_normalised
    assert spec.std_epsilon == 1e-4
    assert spec.aggregation == "sequence"
    assert spec.kl_coefficient == pytest.approx(0.04)
    assert spec.advantage_whitening is False
    assert spec.off_policy_correction == "token"


def test_the_optimizer_telemetry_reaches_the_step():
    tap = TRLTap(run_id="fake", budget=GENEROUS)
    drive(tap, FakeTrainer(), steps=2)
    steps = tuple(tap.finish().steps)
    assert all(s.optimizer.grad_norm_clipped == pytest.approx(0.3) for s in steps)
    assert all(s.optimizer.entropy == pytest.approx(1.25) for s in steps)
    assert all(s.optimizer.kl_to_ref == pytest.approx(0.01) for s in steps)
    assert all(s.schedule["learning_rate"] == pytest.approx(1e-6) for s in steps)
    assert all(s.schedule["num_generations"] == 2.0 for s in steps)


def test_every_step_carries_the_instrument_effect():
    tap = TRLTap(run_id="fake", budget=GENEROUS)
    drive(tap, FakeTrainer(), steps=3)
    effects = [s.instrument for s in tap.finish().steps]
    assert all(e.invocations > 0 and not e.disabled for e in effects)
    assert [e.invocations for e in effects] == sorted(e.invocations for e in effects)


def test_the_regime_declares_only_what_the_config_settles():
    tap = TRLTap(run_id="fake", budget=GENEROUS)
    drive(tap, FakeTrainer(), steps=1)
    declared = tap.finish().regime.declared
    assert declared[RegimeCondition.NEAR_POLICY] is True
    assert declared[RegimeCondition.NO_COMPACTION] is True
    assert declared[RegimeCondition.MASK_STABLE] is True
    assert RegimeCondition.STATIONARY_GRADER not in declared
    assert RegimeCondition.GROUP_NONDEGENERATE not in declared


def test_reusing_a_generation_across_steps_is_declared_not_on_policy():
    """``num_iterations`` or ``steps_per_generation`` above 1 means a rollout is consumed by an
    optimizer step later than the one that produced it, so the segment is stale by construction."""
    trainer = FakeTrainer()
    trainer.args.num_iterations = 2
    tap = TRLTap(run_id="fake", budget=GENEROUS)
    drive(tap, trainer, steps=1)
    run = tap.finish()
    assert run.regime.declared[RegimeCondition.NEAR_POLICY] is False
    traj = next(iter(run.steps)).groups[0].trajectories[0]
    assert traj.provenance[0].staleness_steps == 1


def test_access_claims_only_the_rung_a_record_supports():
    tap = TRLTap(run_id="fake", budget=GENEROUS)
    drive(tap, FakeTrainer(), steps=1)
    run = tap.finish()
    assert run.access[Component.GRADER] is Access.RECORD
    assert Access.QUERY not in run.access[Component.GRADER]
    assert run.access[Component.POLICY] is Access.RECORD


def test_a_caller_can_declare_a_higher_rung_when_it_is_true():
    tap = TRLTap(run_id="fake", budget=GENEROUS)
    drive(tap, FakeTrainer(), steps=1)
    run = tap.finish(access={Component.GRADER: Access.RECORD | Access.QUERY | Access.SOURCE})
    assert Access.QUERY in run.access[Component.GRADER]


# ---------------------------------------------------------------------------
# what the record does not hold, said out loud
# ---------------------------------------------------------------------------


def test_a_full_step_buffer_becomes_a_selective_sampling_policy_not_a_silent_truncation():
    tap = TRLTap(run_id="fake", budget=GENEROUS, max_steps=2)
    drive(tap, FakeTrainer(), steps=5)
    policy = tap.finish().sampling_policy
    assert policy.scheme is SamplingScheme.SELECTIVE
    assert policy.recorded == 2
    assert policy.population == 5
    assert policy.rate == pytest.approx(0.4)
    assert policy.unit == "step"
    assert "no arithmetic correction" in policy.selected_by


def test_a_complete_run_is_a_full_sampling_policy_carrying_the_rings_drop_count():
    tap = TRLTap(run_id="fake", budget=GENEROUS)
    drive(tap, FakeTrainer(), steps=3)
    policy = tap.finish().sampling_policy
    assert policy.scheme is SamplingScheme.FULL
    assert policy.rate == 1.0
    assert "0 dropped" in policy.notes


def test_a_skipped_reward_model_is_named_in_the_sampling_notes_and_on_the_component():
    class FakeModule:
        __module__ = "torch.nn.modules.module"

    tap = TRLTap(run_id="fake", budget=GENEROUS)
    tap.wrap([grader, FakeModule()])
    tap.attach(FakeTrainer())
    run = tap.finish()
    assert "not instrumented" in run.sampling_policy.notes
    assert run.components[Component.GRADER].extra["n_skipped"] == 1


def test_max_steps_below_one_is_refused_at_construction():
    with pytest.raises(ValueError, match="at least 1"):
        TRLTap(max_steps=0)


# ---------------------------------------------------------------------------
# abstention is not a zero, at every level it passes through
# ---------------------------------------------------------------------------


def test_an_abstention_stays_none_through_the_record():
    def abstains(prompts, completions, completion_ids, **kwargs):
        return [None if i == 1 else 1.0 for i in range(len(completions))]

    tap = TRLTap(run_id="fake", budget=GENEROUS)
    trainer = FakeTrainer()
    wrapped = tap.wrap(abstains)
    tap.attach(trainer)
    kwargs, _, _ = make_kwargs(4, 0)
    wrapped(
        prompts=["p"] * 4,
        completions=["cc"] * 4,
        completion_ids=[[1]] * 4,
        **kwargs,
    )
    trainer._logs["completion"] = ["cc"] * 4
    trainer._logs["prompt"] = ["p"] * 4
    trainer.callbacks[0].on_step_end(trainer.args, FakeState(1), None)
    trajectories = [t for s in tap.finish().steps for g in s.groups for t in g.trajectories]
    assert trajectories[1].scores.value is None
    assert trajectories[1].scores.abstained is True
    assert math.isnan(evaluate(trajectories[1].scores, ScoreContext()))
    assert trajectories[1].features == {}, "an abstained row has no realised reward, not a zero"
    assert trajectories[0].scores.value == 1.0
    assert trajectories[0].features["trl_realised_reward"] == 1.0


def test_group_stats_are_computed_from_the_totals_trl_actually_used():
    """The group baseline TRL computed used ``nansum``, so the record's statistics have to as well
    or they describe a run that did not happen."""

    def abstains(prompts, completions, completion_ids, **kwargs):
        return [None if i == 1 else 1.0 for i in range(len(completions))]

    tap = TRLTap(run_id="fake", budget=GENEROUS)
    trainer = FakeTrainer()
    wrapped = tap.wrap(abstains)
    tap.attach(trainer)
    kwargs, _, _ = make_kwargs(4, 0)
    wrapped(prompts=["p"] * 4, completions=["cc"] * 4, completion_ids=[[1]] * 4, **kwargs)
    trainer._logs["completion"] = ["cc"] * 4
    trainer.callbacks[0].on_step_end(trainer.args, FakeState(1), None)
    stats = [g.group_stats for s in tap.finish().steps for g in s.groups]
    assert stats[0].k == 2
    assert stats[0].n_abstained == 1
    assert stats[0].mean == pytest.approx(1.0), "the abstention was excluded, not counted as zero"


def test_total_is_none_only_when_everything_abstained():
    """Matching TRL: an all-abstain row is NaN at 2679, and a partly-abstaining row is summed over
    what is left, because ``nansum`` treats a NaN as a zero. The record reproduces the run rather
    than quietly improving on it."""
    assert _total({"a": None, "b": None}) is None
    assert _total({"a": None, "b": 2.0}) == 2.0
    assert _total({"a": 1.0, "b": 2.0}) == 3.0
    assert _total({"a": 1.0, "b": 2.0}, [0.5, 2.0]) == 4.5
    assert _total({}) is None


def test_the_tree_refuses_a_partial_abstention_where_trl_substitutes_a_zero():
    """The divergence, pinned. It is a finding rather than a bug on either side.

    ``nansum`` at ``grpo_trainer.py:2683`` treats a NaN as a zero, so a row where one of two
    graders abstained is summed over the other and the missing component is silently worth
    nothing; TRL only refuses when *every* function abstained (``unscorable_mask`` at 2679).
    ``evaluate`` on the recorded tree returns NaN instead, because a sum with a missing term is not
    a number. Both are on the record: the tree carries the refusal and
    ``features["trl_realised_reward"]`` carries the number the optimizer used, so the rows where
    they disagree are exactly the rows where the substitution happened and they are findable.
    """

    def partly(prompts, completions, completion_ids, **kwargs):
        return [1.0] * len(completions)

    def always_abstains(prompts, completions, completion_ids, **kwargs):
        return [None] * len(completions)

    tap = TRLTap(run_id="fake", budget=GENEROUS)
    trainer = FakeTrainer()
    wrapped = tap.wrap([partly, always_abstains])
    tap.attach(trainer)
    kwargs, _, _ = make_kwargs(2, 0)
    for func in wrapped:
        func(prompts=["p"] * 2, completions=["cc"] * 2, completion_ids=[[1]] * 2, **kwargs)
    trainer._logs["completion"] = ["cc"] * 2
    trainer.callbacks[0].on_step_end(trainer.args, FakeState(1), None)

    traj = next(iter(tap.finish().steps)).groups[0].trajectories[0]
    assert isinstance(traj.scores, WeightedSum)
    assert {leaf.name for leaf in traj.scores.children} == {"partly", "always_abstains"}
    assert math.isnan(evaluate(traj.scores, ScoreContext())), (
        "the tree should refuse a sum with a missing term"
    )
    assert traj.features["trl_realised_reward"] == 1.0, (
        "and the number TRL actually used should still be on the record beside the refusal"
    )


def test_a_nan_score_is_read_as_an_abstention_rather_than_a_number():
    assert _as_float(float("nan")) is None
    assert _as_float(None) is None
    assert _as_float("not a number") is None
    assert _as_float(0.0) == 0.0
    assert _as_float(3) == 3.0


# ---------------------------------------------------------------------------
# the adapter cannot raise into the host, whatever it is handed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"trainer_state": None},
        {"trainer_state": object()},
        {"log_metric": None},
        {"log_metric": "not callable"},
        {"trainer_state": FakeState(2), "log_metric": lambda *a: 1 / 0},
    ],
)
def test_a_malformed_call_context_never_stops_the_grader(kwargs):
    """A framework that renames a key, passes a ``None``, or hands over a callback that raises must
    cost the record and nothing else. Every one of these is a real shape TRL could produce across a
    release, and the recorded consequence is a counted adapter exception."""
    tap = TRLTap(budget=GENEROUS, emit_extra=True)
    wrapped = tap.wrap(grader)
    out = wrapped(prompts=["p"], completions=["cc"], completion_ids=[[1]], **kwargs)
    assert out == [0.2]


def test_a_trainer_that_cannot_be_attached_to_is_counted_not_raised():
    class Hostile:
        args = FakeArgs()

        def add_callback(self, callback):
            raise RuntimeError("no callbacks here")

    tap = TRLTap(budget=GENEROUS)
    tap.attach(Hostile())
    assert tap.adapter_exceptions == 1
    assert not tap._attached


def test_finish_works_on_a_run_that_never_started():
    """Plane B fails loud, but an empty record is a legitimate answer rather than a failure: it
    says the tap was attached and nothing was scored."""
    tap = TRLTap(run_id="empty", budget=GENEROUS)
    run = tap.finish()
    assert tuple(run.steps) == ()
    assert run.sampling_policy.recorded == 0
    assert run.lineage.framework == "trl"

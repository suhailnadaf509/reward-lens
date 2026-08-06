"""The regime reading, condition by condition.

Two things are being tested and the second one is the harder one. The first is that each of the
twelve regime conditions reads `True` on a record built to satisfy it and `False` on one
built to violate it. The second is that each reads `None` when its input is genuinely absent,
because a `False` that means "I could not tell" is the defect this whole module exists to prevent
and it is invisible in any test that only checks the two easy states.

So every condition gets at least three tests: one that holds, one that fails, and one that comes
back undetermined with a sentence naming what would settle it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reward_lens.core.budget import LimitOfDetection
from reward_lens.core.envelope import RegimeCondition
from reward_lens.core.invariance import GROUPS, InvariancePayload, check_invariance
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Access, Component
from reward_lens.measure.base import lint_instrument
from reward_lens.measure.rate.regime import (
    MEASURED_BY,
    UNREGISTERED_MEASURED_BY,
    RegimeFloors,
    RegimeInputs,
    RegimeThresholds,
    RunRegime,
    declaration_disagreements,
    measure_regime,
)
from reward_lens.record.compaction import CompactionEvent
from reward_lens.record.provenance import SamplingMeta, SegmentProvenance
from reward_lens.record.schema import (
    EstimatorSpec,
    Group,
    GroupID,
    GroupStats,
    InMemoryStepStream,
    OptimizerTelemetry,
    ProbeResult,
    RegimeDeclaration,
    Run,
    RunID,
    StalenessDistribution,
    Step,
    TaskID,
    make_trajectory,
)
from reward_lens.record.tensors import Engine
from reward_lens.record.turns import Turn

ENGINE = Engine(name="vllm", revision="0.11.2", dtype="bfloat16")


# ---------------------------------------------------------------------------
# Builders. Deliberately explicit: a fixture that fills in a default staleness
# or a default mask would hide exactly the field under test.
# ---------------------------------------------------------------------------


def turn(
    index: int,
    *,
    role: str = "assistant",
    n_tokens: int = 4,
    masked: bool | None = True,
    lp_sampling: list[float] | None = None,
    lp_train: list[float] | None = None,
    step_score: float | None = None,
) -> Turn:
    ids = tuple(range(100, 100 + n_tokens))
    return Turn(
        index=index,
        role=role,  # type: ignore[arg-type]
        token_ids=ids,
        loss_mask=None if masked is None else tuple([masked] * n_tokens),
        logprobs_sampling=None if lp_sampling is None else tuple(lp_sampling),
        logprobs_train=None if lp_train is None else tuple(lp_train),
        step_score=step_score,
    )


def traj(
    ident: str,
    *,
    task: str = "t0",
    turns: list[Turn] | None = None,
    advantage: float | None = None,
    provenance: list[SegmentProvenance] | None = None,
    compaction: tuple[CompactionEvent, ...] = (),
    staleness: int = 0,
):
    ts = turns if turns is not None else [turn(0)]
    if provenance is None and ts:
        provenance = [
            SegmentProvenance(
                turn_range=(0, len(ts)),
                policy_version="p0",  # type: ignore[arg-type]
                staleness_steps=staleness,
                engine=ENGINE,
                sampling=SamplingMeta(temperature=1.0),
            )
        ]
    return make_trajectory(
        id=ident,
        task_ref=task,
        turns=ts,
        advantage=advantage,
        provenance=provenance,
        compaction=compaction,
    )


def group(
    ident: str,
    *,
    task: str = "t0",
    trajectories: list | None = None,
    stats: GroupStats | None = None,
    mask_policy: str = "unknown",
) -> Group:
    ts = trajectories if trajectories is not None else [traj(f"{ident}-0"), traj(f"{ident}-1")]
    return Group(
        id=GroupID(ident),
        task_ref=TaskID(task),
        trajectories=tuple(ts),
        estimator=EstimatorSpec(family="grpo", loss_mask_policy=mask_policy),
        group_stats=stats or GroupStats(k=len(ts), mean=0.5, std=0.3),
    )


def step(
    index: int,
    *,
    groups: list[Group] | None = None,
    schedule: dict[str, float] | None = None,
    entropy: float | None = None,
    staleness_hist: dict[int, int] | None = None,
    probes: tuple[ProbeResult, ...] = (),
) -> Step:
    return Step(
        index=index,
        groups=tuple(groups if groups is not None else [group(f"g{index}")]),
        schedule=dict(schedule or {}),
        optimizer=OptimizerTelemetry(
            entropy=entropy,
            staleness=StalenessDistribution(counts=dict(staleness_hist or {})),
        ),
        probes=probes,
    )


def make_run(steps: list[Step], *, declared: dict | None = None) -> Run:
    return Run(
        id=RunID("run-under-test"),
        kind="train",
        components={},
        access={Component.RECORD: Access.RECORD},
        regime=RegimeDeclaration(declared=declared or {}),
        steps=InMemoryStepStream(steps),
    )


def read(run: Run, condition: RegimeCondition, **kwargs):
    return measure_regime(run, **kwargs).conditions[condition]


# ---------------------------------------------------------------------------
# The shape of a reading
# ---------------------------------------------------------------------------


def test_all_twelve_conditions_are_present_in_every_reading():
    """Absent means never measured and present-with-None means measured and indeterminate.

    The regime reading maps every condition, so this producer always emits twelve. The
    consequence is deliberate and it is strict: `EnvelopeSpec.classify` counts a present-with-None
    as a failure rather than as an unchecked box, so an instrument requiring LINEAR_RESPONSE is
    refused on a record where nobody computed Lambda.
    """
    reading = measure_regime(make_run([step(0), step(1)]))
    assert set(reading.conditions) == set(RegimeCondition)
    assert len(reading.conditions) == 13


def test_an_undetermined_condition_carries_no_number():
    """NaN rather than zero, because a zero is a number somebody can read off a report."""
    reading = measure_regime(make_run([step(0), step(1)]))
    for condition, cr in reading.conditions.items():
        if cr.holds is None:
            assert math.isnan(cr.statistic), condition
            assert math.isnan(cr.threshold), condition
            assert cr.detail, f"{condition} came back unknown with no sentence saying why"


def test_every_reading_carries_a_provenance_id():
    reading = measure_regime(make_run([step(0), step(1)]))
    for cr in reading.conditions.values():
        assert cr.provenance is not None and cr.provenance.startswith("regime:")


def test_provenance_moves_when_the_material_moves():
    """Two readings that disagree have visibly different material behind them."""
    quiet = read(make_run([step(0), step(1)]), RegimeCondition.GROUP_NONDEGENERATE)
    dead = read(
        make_run(
            [
                step(
                    i,
                    groups=[
                        group(f"g{i}", stats=GroupStats(k=4, mean=0.0, std=0.0, degenerate=True))
                    ],
                )
                for i in (0, 1)
            ]
        ),
        RegimeCondition.GROUP_NONDEGENERATE,
    )
    assert quiet.provenance != dead.provenance


# ---------------------------------------------------------------------------
# 1. QUASI_STATIC
# ---------------------------------------------------------------------------


def _annealed(n: int, rate: float) -> list[Step]:
    """A schedule with an exactly constant log derivative, so Ad is exact rather than fitted."""
    return [step(i, schedule={"kl_coefficient": math.exp(rate * i)}) for i in range(n)]


def test_quasi_static_is_the_product_of_the_relaxation_time_and_the_driving_rate():
    run = make_run(_annealed(6, rate=0.25))
    cr = read(run, RegimeCondition.QUASI_STATIC, inputs=RegimeInputs(tau_relax=2.0))
    assert cr.statistic == pytest.approx(0.5)
    assert cr.threshold == 1.0
    assert cr.holds is True


def test_quasi_static_fails_when_the_driver_outruns_the_relaxation():
    run = make_run(_annealed(6, rate=1.0))
    cr = read(run, RegimeCondition.QUASI_STATIC, inputs=RegimeInputs(tau_relax=3.0))
    assert cr.holds is False
    assert cr.statistic == pytest.approx(3.0)


def test_a_flat_schedule_is_quasi_static_without_a_relaxation_time():
    """Ad is zero whatever tau is, and refusing that case would refuse the one clean pass."""
    run = make_run([step(i, schedule={"lr": 1e-5}) for i in range(4)])
    cr = read(run, RegimeCondition.QUASI_STATIC)
    assert cr.holds is True
    assert cr.statistic == 0.0
    assert "not moving" in cr.detail


def test_an_unrecorded_schedule_is_not_a_flat_schedule():
    cr = read(make_run([step(0), step(1)]), RegimeCondition.QUASI_STATIC)
    assert cr.holds is None
    assert "An empty schedule is not a flat one" in cr.detail


def test_quasi_static_is_undetermined_on_a_one_step_window():
    run = make_run(_annealed(6, rate=0.5))
    cr = read(run, RegimeCondition.QUASI_STATIC, step=3)
    assert cr.holds is None
    assert "one step has no schedule derivative" in cr.detail


def test_the_relaxation_time_comes_off_the_early_ar1_coefficient():
    """Rung 0's tau, fitted rather than supplied, on a series with a planted phi."""
    rng = np.random.default_rng(11)
    phi = 0.7
    x, series = 0.0, []
    for _ in range(60):
        x = phi * x + rng.normal(0, 0.1)
        series.append(x)
    steps = [
        step(
            i,
            schedule={"kl_coefficient": math.exp(0.05 * i)},
            groups=[group(f"g{i}", stats=GroupStats(k=4, mean=series[i], std=0.3))],
        )
        for i in range(60)
    ]
    cr = read(make_run(steps), RegimeCondition.QUASI_STATIC)
    planted_tau = -1.0 / math.log(phi)
    assert cr.holds is not None
    assert cr.statistic / 0.05 == pytest.approx(planted_tau, rel=0.6)
    assert "early AR(1) coefficient" in cr.detail


def test_a_trending_series_does_not_masquerade_as_a_slow_relaxation():
    """Reward climbs during training; without detrending the fit returns phi near one every time."""
    steps = [
        step(
            i,
            schedule={"kl_coefficient": math.exp(0.05 * i)},
            groups=[group(f"g{i}", stats=GroupStats(k=4, mean=0.01 * i, std=0.3))],
        )
        for i in range(40)
    ]
    cr = read(make_run(steps), RegimeCondition.QUASI_STATIC)
    # A pure line detrends to exactly zero residual, so no autocorrelation is identified and the
    # honest answer is that tau is unavailable rather than enormous.
    assert cr.holds is None
    assert "flat" in cr.detail or "not identified" in cr.detail


# ---------------------------------------------------------------------------
# 2. LINEAR_RESPONSE
# ---------------------------------------------------------------------------


def test_linear_response_is_undetermined_without_lambda_and_says_whose_job_it_is():
    cr = read(make_run([step(0), step(1)]), RegimeCondition.LINEAR_RESPONSE)
    assert cr.holds is None
    assert "F2" in cr.detail and "selection.explained_fraction" in cr.detail


def test_linear_response_reads_the_supplied_lambda():
    run = make_run([step(0), step(1)])
    assert (
        read(
            run, RegimeCondition.LINEAR_RESPONSE, inputs=RegimeInputs(explained_fraction=0.62)
        ).holds
        is True
    )
    assert (
        read(
            run, RegimeCondition.LINEAR_RESPONSE, inputs=RegimeInputs(explained_fraction=0.11)
        ).holds
        is False
    )


# ---------------------------------------------------------------------------
# 3. GROUP_NONDEGENERATE
# ---------------------------------------------------------------------------


def _mixed_groups(n_degenerate: int, n_live: int, *, unknown: int = 0) -> list[Group]:
    out = []
    for i in range(n_degenerate):
        out.append(group(f"d{i}", stats=GroupStats(k=4, mean=0.0, std=0.0, degenerate=True)))
    for i in range(n_live):
        out.append(group(f"l{i}", stats=GroupStats(k=4, mean=0.5, std=0.4)))
    for i in range(unknown):
        out.append(group(f"u{i}", stats=GroupStats(k=4, mean=0.5, std=None)))
    return out


def test_the_statistic_is_the_degenerate_fraction_against_the_stated_threshold():
    """The worked report prints "degenerate group fraction 0.04 (threshold 0.20)"."""
    cr = read(make_run([step(0, groups=_mixed_groups(1, 24))]), RegimeCondition.GROUP_NONDEGENERATE)
    assert cr.threshold == 0.20
    assert cr.statistic == pytest.approx(0.04)
    assert cr.holds is True


def test_a_batch_of_all_fail_groups_fails_the_condition():
    cr = read(make_run([step(0, groups=_mixed_groups(9, 1))]), RegimeCondition.GROUP_NONDEGENERATE)
    assert cr.holds is False
    assert cr.statistic == pytest.approx(0.9)


def test_a_group_of_one_is_degenerate_because_k_must_exceed_one():
    solo = [group(f"s{i}", stats=GroupStats(k=1, mean=0.5, std=0.0)) for i in range(4)]
    cr = read(make_run([step(0, groups=solo)]), RegimeCondition.GROUP_NONDEGENERATE)
    assert cr.holds is False


def test_unrecorded_spread_makes_the_fraction_indeterminate_when_it_straddles_the_threshold():
    cr = read(
        make_run([step(0, groups=_mixed_groups(1, 4, unknown=5))]),
        RegimeCondition.GROUP_NONDEGENERATE,
    )
    assert cr.holds is None
    assert "somewhere in" in cr.detail


def test_unrecorded_spread_that_cannot_change_the_verdict_still_answers():
    cr = read(
        make_run([step(0, groups=_mixed_groups(8, 1, unknown=1))]),
        RegimeCondition.GROUP_NONDEGENERATE,
    )
    assert cr.holds is False


# ---------------------------------------------------------------------------
# 4. NEAR_POLICY
# ---------------------------------------------------------------------------


def test_near_policy_holds_at_the_stated_staleness_bound():
    """The worked report prints "max staleness 2 steps (threshold 8)"."""
    g = group("g0", trajectories=[traj("a", staleness=2), traj("b", staleness=1)])
    cr = read(make_run([step(0, groups=[g])]), RegimeCondition.NEAR_POLICY)
    assert cr.threshold == 8.0
    assert cr.statistic == 2.0
    assert cr.holds is True


def test_near_policy_fails_past_the_bound():
    g = group("g0", trajectories=[traj("a", staleness=40)])
    assert read(make_run([step(0, groups=[g])]), RegimeCondition.NEAR_POLICY).holds is False


def test_a_partial_rollout_fails_near_policy_at_zero_staleness():
    """Two policies on one trajectory make a single importance ratio undefined, not stale."""
    turns = [turn(0), turn(1)]
    split = traj(
        "resumed",
        turns=turns,
        provenance=[
            SegmentProvenance(
                turn_range=(0, 1), policy_version="p0", staleness_steps=0, engine=ENGINE
            ),
            SegmentProvenance(
                turn_range=(1, 2), policy_version="p1", staleness_steps=0, engine=ENGINE
            ),
        ],
    )
    cr = read(
        make_run([step(0, groups=[group("g0", trajectories=[split])])]), RegimeCondition.NEAR_POLICY
    )
    assert cr.holds is False
    assert cr.statistic == 0.0
    assert "more than one policy version" in cr.detail


def test_singularity_is_per_trajectory_and_not_across_the_window():
    """A forty-step window holds forty policy versions; that is training, not a violation."""
    steps = []
    for i in range(4):
        t = make_trajectory(
            id=f"t{i}",
            task_ref="t0",
            turns=[turn(0)],
            provenance=[
                SegmentProvenance(
                    turn_range=(0, 1),
                    policy_version=f"p{i}",  # type: ignore[arg-type]
                    staleness_steps=0,
                    engine=ENGINE,
                )
            ],
        )
        steps.append(step(i, groups=[group(f"g{i}", trajectories=[t])]))
    assert read(make_run(steps), RegimeCondition.NEAR_POLICY).holds is True


def test_the_staleness_histogram_answers_when_segments_do_not():
    g = group("g0", trajectories=[traj("a", staleness=0)])
    cr = read(
        make_run([step(0, groups=[g], staleness_hist={0: 500, 12: 12})]),
        RegimeCondition.NEAR_POLICY,
    )
    assert cr.statistic == 12.0
    assert cr.holds is False
    assert "staleness histogram" in cr.detail


def test_near_policy_is_undetermined_with_no_rollouts_to_read():
    empty = group("g0", trajectories=[traj("e", turns=[])])
    cr = read(make_run([step(0, groups=[empty])]), RegimeCondition.NEAR_POLICY)
    assert cr.holds is None
    assert "no trajectory in this window carries segment provenance" in cr.detail


# ---------------------------------------------------------------------------
# 5. STATIONARY_GRADER
# ---------------------------------------------------------------------------


def test_stationary_grader_is_undetermined_with_neither_a_standard_nor_a_schedule():
    cr = read(make_run([step(0), step(1)]), RegimeCondition.STATIONARY_GRADER)
    assert cr.holds is None
    assert "check_standard" in cr.detail and "grader_weights" in cr.detail


class _FakeSum:
    """A score-tree node in shape only, to check the structural walk without importing the real one.

    The walk is deliberately not typed against `reward_lens.record.scores`, so the test is not
    either. What it pins is the contract the walk relies on: a `name`, numeric parameters under
    known attribute names, and children under `children`.
    """

    def __init__(self, name, children=(), weights=(), bias=0.0):
        self.name = name
        self.children = tuple(children)
        self.weights = tuple(weights)
        self.bias = bias


def _tree(weight: float):
    return _FakeSum(
        "total",
        children=[_FakeSum("correctness"), _FakeSum("length")],
        weights=(1.0, weight),
    )


def _step_with_tree(index: int, weight: float) -> Step:
    t = make_trajectory(
        id=f"t{index}",
        task_ref="t0",
        turns=[turn(0)],
        scores=_tree(weight),
        provenance=[
            SegmentProvenance(
                turn_range=(0, 1),
                policy_version="p0",  # type: ignore[arg-type]
                staleness_steps=0,
                engine=ENGINE,
            )
        ],
    )
    return step(index, groups=[group(f"g{index}", trajectories=[t])])


def test_a_score_tree_on_the_record_answers_without_anybody_passing_weights():
    run = make_run([_step_with_tree(i, weight=0.3) for i in range(4)])
    cr = read(run, RegimeCondition.STATIONARY_GRADER)
    assert cr.holds is True
    assert "read structurally off the record" in cr.detail


def test_a_reweighted_score_tree_names_the_step_the_rubric_changed_at():
    steps = [_step_with_tree(i, weight=0.3 if i < 2 else 0.9) for i in range(4)]
    cr = read(make_run(steps), RegimeCondition.STATIONARY_GRADER)
    assert cr.holds is False
    assert cr.statistic == 2.0
    assert "changed at step 2" in cr.detail


def test_a_score_object_the_walk_does_not_recognise_falls_through_to_undetermined():
    """The failure mode of the structural read is silence, never a claim that nothing changed."""
    t = make_trajectory(
        id="t0",
        task_ref="t0",
        turns=[turn(0)],
        scores={"total": 0.7},
        provenance=[
            SegmentProvenance(
                turn_range=(0, 1),
                policy_version="p0",  # type: ignore[arg-type]
                staleness_steps=0,
                engine=ENGINE,
            )
        ],
    )
    steps = [step(i, groups=[group(f"g{i}", trajectories=[t])]) for i in range(3)]
    assert read(make_run(steps), RegimeCondition.STATIONARY_GRADER).holds is None


def test_a_rubric_weight_change_names_the_step_it_happened_at():
    run = make_run([step(i) for i in range(4)])
    weights = {
        0: {"helpfulness": 1.0},
        1: {"helpfulness": 1.0},
        2: {"helpfulness": 0.4},
        3: {"helpfulness": 0.4},
    }
    cr = read(run, RegimeCondition.STATIONARY_GRADER, inputs=RegimeInputs(grader_weights=weights))
    assert cr.holds is False
    assert "changed at step 2" in cr.detail


def test_unchanged_weights_hold():
    run = make_run([step(i) for i in range(3)])
    weights = {i: {"helpfulness": 1.0, "harmlessness": 0.5} for i in range(3)}
    cr = read(run, RegimeCondition.STATIONARY_GRADER, inputs=RegimeInputs(grader_weights=weights))
    assert cr.holds is True
    assert cr.statistic == 1.0


def test_a_check_standard_that_moves_is_grader_drift_by_construction():
    probes = [
        (ProbeResult(name="frozen-50", channel="check_standard", value=v),)
        for v in (0.71, 0.71, 0.66)
    ]
    run = make_run([step(i, probes=probes[i]) for i in range(3)])
    cr = read(run, RegimeCondition.STATIONARY_GRADER)
    assert cr.holds is False
    assert cr.statistic == pytest.approx(0.05)
    assert "frozen-50" in cr.detail


def test_a_check_standard_read_once_cannot_show_a_change():
    run = make_run(
        [
            step(0, probes=(ProbeResult(name="frozen-50", channel="check_standard", value=0.7),)),
            step(1),
        ]
    )
    assert read(run, RegimeCondition.STATIONARY_GRADER).holds is None


def test_a_held_out_probe_is_not_a_check_standard():
    """A held-out eval measures generalisation, so its movement is the run working."""
    run = make_run(
        [
            step(i, probes=(ProbeResult(name="gsm8k", channel="held_out", value=0.4 + 0.1 * i),))
            for i in range(3)
        ]
    )
    assert read(run, RegimeCondition.STATIONARY_GRADER).holds is None


# ---------------------------------------------------------------------------
# 6. EXOGENOUS_CURRICULUM
# ---------------------------------------------------------------------------


def _curriculum_steps(n_tasks: int, *, drop_high: bool) -> list[Step]:
    """Two steps. In the responsive case the tasks that scored well are the ones dropped."""
    scores = {f"task{i}": i / n_tasks for i in range(n_tasks)}
    first = [
        group(f"a{i}", task=t, stats=GroupStats(k=4, mean=s, std=0.3))
        for i, (t, s) in enumerate(scores.items())
    ]
    if drop_high:
        keep = [t for t, s in scores.items() if s < 0.5]
    else:
        keep = [t for i, t in enumerate(scores) if i % 2 == 0]
    second = [
        group(f"b{i}", task=t, stats=GroupStats(k=4, mean=scores[t], std=0.3))
        for i, t in enumerate(keep)
    ]
    return [step(0, groups=first), step(1, groups=second)]


def test_a_curriculum_that_drops_the_solved_tasks_is_detected():
    cr = read(make_run(_curriculum_steps(24, drop_high=True)), RegimeCondition.EXOGENOUS_CURRICULUM)
    assert cr.holds is False
    assert cr.statistic == pytest.approx(1.0)
    assert "rank-biserial" in cr.detail


def test_dropping_tasks_at_random_does_not_read_as_a_curriculum():
    cr = read(
        make_run(_curriculum_steps(40, drop_high=False)), RegimeCondition.EXOGENOUS_CURRICULUM
    )
    assert cr.holds is True


def test_a_fixed_task_set_holds_without_a_test():
    steps = [
        step(i, groups=[group(f"{i}-{j}", task=f"task{j}") for j in range(6)]) for i in range(3)
    ]
    cr = read(make_run(steps), RegimeCondition.EXOGENOUS_CURRICULUM)
    assert cr.holds is True
    assert "nothing is being selected out" in cr.detail


def test_too_few_tasks_is_undetermined_rather_than_passed():
    cr = read(make_run(_curriculum_steps(6, drop_high=True)), RegimeCondition.EXOGENOUS_CURRICULUM)
    assert cr.holds is None
    assert "floor" in cr.detail


def test_a_one_step_window_cannot_see_a_curriculum():
    cr = read(
        make_run(_curriculum_steps(24, drop_high=True)),
        RegimeCondition.EXOGENOUS_CURRICULUM,
        step=0,
    )
    assert cr.holds is None
    assert "one step" in cr.detail


# ---------------------------------------------------------------------------
# 7. NO_COMPACTION
# ---------------------------------------------------------------------------


def _multi_turn(ident: str, *, compaction: tuple[CompactionEvent, ...] = ()):
    return traj(ident, turns=[turn(0), turn(1), turn(2)], compaction=compaction)


def test_a_lossy_prefix_rewrite_fails_the_condition():
    event = CompactionEvent(at_turn=2, tokens_before=8000, tokens_after=900, method="summarise")
    g = group("g0", trajectories=[_multi_turn("a", compaction=(event,)), _multi_turn("b")])
    cr = read(make_run([step(0, groups=[g])]), RegimeCondition.NO_COMPACTION)
    assert cr.holds is False
    assert cr.statistic == 1.0
    assert "summarise" in cr.detail


def test_a_single_turn_window_proves_no_prefix_was_rewritten():
    cr = read(make_run([step(0)]), RegimeCondition.NO_COMPACTION)
    assert cr.holds is True
    assert "no prefix could be rewritten" in cr.detail


def test_silence_from_a_converter_is_not_a_clean_zero():
    g = group("g0", trajectories=[_multi_turn("a"), _multi_turn("b")])
    cr = read(make_run([step(0, groups=[g])]), RegimeCondition.NO_COMPACTION)
    assert cr.holds is None
    assert "nothing here can tell the two apart" in cr.detail


def test_saying_the_converter_emits_compaction_settles_it():
    g = group("g0", trajectories=[_multi_turn("a"), _multi_turn("b")])
    cr = read(
        make_run([step(0, groups=[g])]),
        RegimeCondition.NO_COMPACTION,
        inputs=RegimeInputs(compaction_recorded=True),
    )
    assert cr.holds is True


def test_a_compaction_that_removed_nothing_does_not_fail_the_condition():
    lossless = CompactionEvent(at_turn=1, tokens_before=900, tokens_after=900, method="rerender")
    g = group("g0", trajectories=[_multi_turn("a", compaction=(lossless,))])
    cr = read(make_run([step(0, groups=[g])]), RegimeCondition.NO_COMPACTION)
    assert cr.holds is True
    assert cr.statistic == 0.0


# ---------------------------------------------------------------------------
# 8. ABOVE_LOD
# ---------------------------------------------------------------------------


def test_above_lod_is_a_property_of_a_reading_and_not_of_a_run():
    cr = read(make_run([step(0), step(1)]), RegimeCondition.ABOVE_LOD)
    assert cr.holds is None
    assert "property of a reading and not of a run" in cr.detail


def test_above_lod_answers_when_the_effect_and_the_limit_arrive_together():
    lod = LimitOfDetection(sigma_blank=0.01, sensitivity=1.0, blank_n=30)
    run = make_run([step(0), step(1)])
    assert (
        read(run, RegimeCondition.ABOVE_LOD, inputs=RegimeInputs(lod=lod, effect=0.5)).holds is True
    )
    below = read(run, RegimeCondition.ABOVE_LOD, inputs=RegimeInputs(lod=lod, effect=0.01))
    assert below.holds is False
    assert below.threshold == pytest.approx(0.033)


def test_a_limit_with_no_calibration_slope_is_undetermined():
    lod = LimitOfDetection(sigma_blank=0.01, sensitivity=0.0)
    cr = read(
        make_run([step(0), step(1)]),
        RegimeCondition.ABOVE_LOD,
        inputs=RegimeInputs(lod=lod, effect=0.5),
    )
    assert cr.holds is None
    assert "does not respond to dose" in cr.detail


# ---------------------------------------------------------------------------
# 9. ESS_ADEQUATE
# ---------------------------------------------------------------------------


def _logprob_traj(ident: str, gaps: list[float]):
    n = len(gaps)
    sampling = [-1.0] * n
    training = [-1.0 + g for g in gaps]
    return traj(
        ident,
        turns=[turn(0, n_tokens=n, lp_sampling=sampling, lp_train=training)],
    )


def test_matched_engines_and_no_drift_give_a_full_effective_sample():
    g = group("g0", trajectories=[_logprob_traj("a", [0.0] * 64)])
    cr = read(make_run([step(0, groups=[g])]), RegimeCondition.ESS_ADEQUATE)
    assert cr.statistic == pytest.approx(1.0)
    assert cr.holds is True
    assert "E6" in cr.detail


def test_one_token_taking_all_the_weight_degenerates_the_sample():
    gaps = [0.0] * 63 + [25.0]
    g = group("g0", trajectories=[_logprob_traj("a", gaps)])
    cr = read(make_run([step(0, groups=[g])]), RegimeCondition.ESS_ADEQUATE)
    assert cr.holds is False
    assert cr.statistic < 0.05


def test_one_logprob_stream_is_undetermined_rather_than_a_ratio_of_one():
    """A converter with only the sampling stream leaves the other None, and this reads that."""
    g = group("g0", trajectories=[traj("a", turns=[turn(0, n_tokens=64, lp_sampling=[-1.0] * 64)])])
    cr = read(make_run([step(0, groups=[g])]), RegimeCondition.ESS_ADEQUATE)
    assert cr.holds is None
    assert "both logprob streams" in cr.detail


def test_the_loss_mask_decides_which_tokens_are_weighted():
    """The tokens the loss does not see do not get a weight, and here they carry all of it."""

    def with_mask(mask):
        t = Turn(
            index=0,
            role="assistant",
            token_ids=tuple(range(64)),
            loss_mask=mask,
            logprobs_sampling=tuple([-1.0] * 64),
            logprobs_train=tuple([-1.0] * 32 + [20.0] * 32),
        )
        g = group("g0", trajectories=[traj("a", turns=[t])])
        return read(make_run([step(0, groups=[g])]), RegimeCondition.ESS_ADEQUATE)

    masked = with_mask(tuple([True] * 32 + [False] * 32))
    assert masked.holds is True
    assert masked.statistic == pytest.approx(1.0)

    # The same turn with no mask recorded weights all 64 tokens, and the 32 that were masked out
    # carry a ratio of e^21 between them, so half the sample takes all of the weight.
    unmasked = with_mask(None)
    assert unmasked.statistic == pytest.approx(0.5, abs=0.01)
    assert unmasked.statistic < masked.statistic


# ---------------------------------------------------------------------------
# 10. LIGHT_TAILED
# ---------------------------------------------------------------------------


def _advantage_run(values) -> Run:
    g = group(
        "g0",
        trajectories=[traj(f"t{i}", advantage=float(v)) for i, v in enumerate(values)],
        stats=GroupStats(k=len(values), mean=0.0, std=1.0),
    )
    return make_run([step(0, groups=[g])])


def test_a_gaussian_advantage_distribution_is_light_tailed():
    rng = np.random.default_rng(3)
    cr = read(_advantage_run(rng.normal(size=1200)), RegimeCondition.LIGHT_TAILED)
    assert cr.holds is True
    assert cr.statistic < 0.25


def test_a_pareto_advantage_distribution_is_not():
    rng = np.random.default_rng(3)
    cr = read(_advantage_run(rng.pareto(1.5, size=1200) + 1.0), RegimeCondition.LIGHT_TAILED)
    assert cr.holds is False
    assert cr.statistic > 0.4


def test_too_few_exceedances_reports_nothing_rather_than_a_number():
    rng = np.random.default_rng(3)
    cr = read(_advantage_run(rng.normal(size=200)), RegimeCondition.LIGHT_TAILED)
    assert cr.holds is None
    assert "1,570" in cr.detail


def test_a_record_with_no_advantages_is_undetermined():
    cr = read(make_run([step(0), step(1)]), RegimeCondition.LIGHT_TAILED)
    assert cr.holds is None


# ---------------------------------------------------------------------------
# 11. SCALAR_REPRESENTABLE
# ---------------------------------------------------------------------------


def test_curl_mass_is_read_off_the_comparison_groups():
    groups = [
        group("a", stats=GroupStats(k=4, mean=0.5, std=0.3, curl_mass=0.01)),
        group("b", stats=GroupStats(k=4, mean=0.5, std=0.3, curl_mass=0.02)),
    ]
    cr = read(make_run([step(0, groups=groups)]), RegimeCondition.SCALAR_REPRESENTABLE)
    assert cr.holds is True
    assert cr.statistic == pytest.approx(0.02)


def test_one_cyclic_group_fails_the_window():
    groups = [
        group("a", stats=GroupStats(k=4, mean=0.5, std=0.3, curl_mass=0.01)),
        group("b", stats=GroupStats(k=4, mean=0.5, std=0.3, curl_mass=0.214)),
    ]
    cr = read(make_run([step(0, groups=groups)]), RegimeCondition.SCALAR_REPRESENTABLE)
    assert cr.holds is False
    assert cr.statistic == pytest.approx(0.214)


def test_a_window_with_no_curl_mass_is_a_question_nobody_asked():
    cr = read(make_run([step(0), step(1)]), RegimeCondition.SCALAR_REPRESENTABLE)
    assert cr.holds is None
    assert "k-wise comparison groups" in cr.detail


# ---------------------------------------------------------------------------
# 12. MASK_STABLE
# ---------------------------------------------------------------------------


def test_one_mask_policy_across_the_window_holds():
    g = group("g0", trajectories=[traj("a"), traj("b")])
    cr = read(make_run([step(0, groups=[g]), step(1, groups=[g])]), RegimeCondition.MASK_STABLE)
    assert cr.holds is True
    assert cr.statistic == 1.0


def test_two_mask_policies_for_one_role_fail():
    on = traj("a", turns=[turn(0, masked=True)])
    off = traj("b", turns=[turn(0, masked=False)])
    g = group("g0", trajectories=[on, off])
    cr = read(make_run([step(0, groups=[g])]), RegimeCondition.MASK_STABLE)
    assert cr.holds is False
    assert "assistant" in cr.detail


def test_partial_recording_is_not_a_policy_change():
    """One rollout with masks and one without differ for a reason that is not the mask policy."""
    recorded = traj("a", turns=[turn(0, masked=True)])
    unrecorded = traj("b", turns=[turn(0, masked=None)])
    g = group("g0", trajectories=[recorded, unrecorded])
    assert read(make_run([step(0, groups=[g])]), RegimeCondition.MASK_STABLE).holds is True


def test_no_masks_and_no_declared_policy_is_undetermined():
    g = group("g0", trajectories=[traj("a", turns=[turn(0, masked=None)])])
    cr = read(make_run([step(0, groups=[g])]), RegimeCondition.MASK_STABLE)
    assert cr.holds is None
    assert "no group names a loss-mask policy" in cr.detail


def test_the_estimator_spec_answers_when_the_masks_are_not_recorded():
    a = group("a", trajectories=[traj("x", turns=[turn(0, masked=None)])], mask_policy="token_mean")
    b = group("b", trajectories=[traj("y", turns=[turn(0, masked=None)])], mask_policy="seq_mean")
    cr = read(make_run([step(0, groups=[a, b])]), RegimeCondition.MASK_STABLE)
    assert cr.holds is False
    assert "declared policies" in cr.detail


# ---------------------------------------------------------------------------
# The declaration cross-check
# ---------------------------------------------------------------------------


def test_a_run_that_declared_a_condition_it_violates_says_so_in_the_reading():
    g = group("g0", trajectories=[traj("a", staleness=40)])
    run = make_run([step(0, groups=[g])], declared={RegimeCondition.NEAR_POLICY: True})
    cr = measure_regime(run).conditions[RegimeCondition.NEAR_POLICY]
    assert cr.holds is False
    assert "the run declared this true" in cr.detail


def test_a_declaration_nobody_could_check_is_a_disagreement():
    run = make_run([step(0), step(1)], declared={RegimeCondition.STATIONARY_GRADER: True})
    reading = measure_regime(run)
    disagreements = declaration_disagreements(run, reading)
    assert disagreements[RegimeCondition.STATIONARY_GRADER] == (True, None)
    assert "unchecked precondition is not a satisfied one" in (
        reading.conditions[RegimeCondition.STATIONARY_GRADER].detail
    )


def test_the_cross_check_can_be_turned_off_without_changing_a_verdict():
    g = group("g0", trajectories=[traj("a", staleness=40)])
    run = make_run([step(0, groups=[g])], declared={RegimeCondition.NEAR_POLICY: True})
    plain = measure_regime(run, cross_check=False).conditions[RegimeCondition.NEAR_POLICY]
    annotated = measure_regime(run).conditions[RegimeCondition.NEAR_POLICY]
    assert plain.holds is annotated.holds
    assert len(annotated.detail) > len(plain.detail)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def test_context_widens_a_per_step_reading_into_one_that_can_see_change():
    run = make_run(_annealed(9, rate=0.5))
    assert read(run, RegimeCondition.QUASI_STATIC, step=4).holds is None
    widened = read(
        run, RegimeCondition.QUASI_STATIC, step=4, context=2, inputs=RegimeInputs(tau_relax=4.0)
    )
    assert widened.holds is False
    assert widened.statistic == pytest.approx(2.0)


def test_asking_for_a_step_and_a_window_at_once_is_an_error():
    with pytest.raises(ValueError, match="not both"):
        measure_regime(make_run([step(0)]), step=0, window=(0, 1))


def test_a_reversed_window_is_an_error():
    with pytest.raises(ValueError, match="half-open"):
        measure_regime(make_run([step(0)]), window=(4, 2))


def test_a_window_outside_the_record_determines_nothing():
    reading = measure_regime(make_run([step(0), step(1)]), window=(50, 60))
    assert all(cr.holds is None for cr in reading.conditions.values())
    assert "contains no recorded steps" in reading.conditions[RegimeCondition.NEAR_POLICY].detail


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


def test_the_instrument_passes_the_section_4_2_lint():
    assert lint_instrument(RunRegime(make_run([step(0)]))) == []


def test_the_instrument_carries_all_six_declarations():
    inst = RunRegime(make_run([step(0)]))
    assert inst.quantity == "run.regime"
    assert inst.requires == {Component.RECORD: Access.RECORD}
    assert inst.substrates and inst.phases
    assert inst.envelope is not None and inst.envelope.unconditional
    assert inst.invariance == "none"
    assert inst.baselines


def test_an_empty_window_is_a_refusal_with_a_reason_and_a_remedy():
    out = RunRegime(make_run([step(0), step(1)]), window=(90, 100)).compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.VOID
    assert "contains no recorded steps" in out.detail
    assert "Ask for a window inside the recorded range" in out.remedy
    assert out.statistics["recorded"] == 2


def test_the_payload_scores_both_baselines():
    g = group("g0", trajectories=[traj("a", staleness=40)])
    run = make_run([step(0, groups=[g])], declared={RegimeCondition.NEAR_POLICY: True})
    inst = RunRegime(run)
    payload = inst.payload(measure_regime(run))
    assert payload["n_hold"] + payload["n_fail"] + payload["n_undetermined"] == 13
    assert payload["baselines"]["run_declaration_agreement"] == 0.0
    assert payload["baselines"]["assume_all_hold_agreement"] < 1.0
    assert set(payload["conditions"]) == {c.name for c in RegimeCondition}


def test_measure_raises_rather_than_emitting_evidence_for_a_void_window():
    inst = RunRegime(make_run([step(0)]), window=(90, 100))
    with pytest.raises(ValueError, match="declines to produce Evidence"):
        inst.measure(object())  # type: ignore[arg-type]


def test_every_condition_names_a_quantity_for_measured_by():
    assert set(MEASURED_BY) == set(RegimeCondition)
    assert all(qid for qid in MEASURED_BY.values())


def test_every_measured_by_target_resolves_in_the_registry():
    """The gap is closed, and this is the test that was written to notice.

    Five of the twelve used to name a quantity `spec/QUANTITIES.yaml` did not carry, and
    `EnvelopeSpec` checked only that a required condition appeared in `measured_by`, not that the
    id resolved. So an envelope naming an unregistered quantity constructed cleanly and read as
    rigour. The earlier version of this test pinned the gap to the module's declared list precisely
    so that the day the rows landed it would fail and the declaration would be deleted rather than
    drift; they landed, it failed, and the declaration is gone.

    What replaces it is stronger than the pin: `EnvelopeSpec.__post_init__` now rejects a
    `measured_by` id that does not resolve, so this cannot silently reopen.
    """
    from reward_lens.core.quantity import QUANTITIES

    missing = {qid for qid in MEASURED_BY.values() if qid not in QUANTITIES}
    assert missing == set(), (
        f"{sorted(missing)} name no row in spec/QUANTITIES.yaml. An envelope condition whose "
        f"measuring quantity does not exist cannot be measured, and reads as rigour anyway."
    )
    assert UNREGISTERED_MEASURED_BY == frozenset()


# ---------------------------------------------------------------------------
# The generated invariance test
# ---------------------------------------------------------------------------


def test_the_generated_invariance_test_passes_under_the_declared_group():
    """`run.regime` declares `none`, which resolves to the trivial group.

    The report is a pass and it is vacuous, which the report itself says: no registered transformation
    acts on a set of verdicts. It is recorded here so the gate is visibly met and
    visibly weak, and the non-vacuous check that is available follows immediately below.
    """
    inst = RunRegime(make_run([step(0)]))
    report = check_invariance(inst, GROUPS["trivial"], InvariancePayload(), run=lambda i, p: 0.0)
    assert report.passed
    assert "trivial group has no generators" in report.skipped


def test_the_verdicts_do_not_move_under_an_affine_rescaling_of_the_reward():
    """The non-vacuous version, on the two conditions whose inputs an affine rescaling touches.

    `GROUP_NONDEGENERATE` compares a group's spread against the estimator's epsilon and
    `LIGHT_TAILED` is a Hill estimate, which is scale-free by construction. Rescaling the reward
    by a positive factor and re-deriving the group statistics must leave both verdicts alone, and a
    reading that moved here would be reading a level rather than a contrast.

    The epsilon is rescaled with the reward, which is not a convenience. E13 records
    that the GRPO advantage is not affine-invariant for a fixed positive epsilon, and the same
    caveat lands here: the degeneracy verdict is invariant under the group only if the trainer's
    epsilon is in reward units. A record whose epsilon is a fixed 1e-8 against a reward rescaled by
    1e-6 has a different degeneracy boundary, and that is a property of the estimator rather than
    of this reading.
    """
    rng = np.random.default_rng(5)
    base = rng.normal(size=(200, 4))

    def verdicts(scale: float, shift: float) -> tuple:
        groups = []
        for i, row in enumerate(base):
            scores = list(scale * row + shift)
            groups.append(
                group(
                    f"g{i}",
                    trajectories=[
                        traj(f"t{i}-{j}", advantage=float(scale * v)) for j, v in enumerate(row)
                    ],
                    stats=GroupStats.from_scores(scores, std_epsilon=1e-8 * abs(scale)),
                )
            )
        reading = measure_regime(make_run([step(0, groups=groups)]))
        return (
            reading.holds(RegimeCondition.GROUP_NONDEGENERATE),
            reading.holds(RegimeCondition.LIGHT_TAILED),
        )

    reference = verdicts(1.0, 0.0)
    assert None not in reference, "a vacuous pass: neither verdict was determined at all"
    for a, b in ((3.7, -2.0), (0.1, 40.0), (10.0, 0.0)):
        assert verdicts(a, b) == reference, f"the verdicts moved under r -> {a}r + {b}"


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@settings(max_examples=40, deadline=None)
@given(
    n_steps=st.integers(min_value=1, max_value=4),
    n_groups=st.integers(min_value=1, max_value=3),
    k=st.integers(min_value=1, max_value=4),
    stale=st.integers(min_value=0, max_value=30),
    std=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
)
def test_a_reading_is_always_twelve_conditions_and_never_raises(n_steps, n_groups, k, stale, std):
    steps = [
        step(
            i,
            groups=[
                group(
                    f"g{i}-{j}",
                    task=f"task{j}",
                    trajectories=[traj(f"t{i}-{j}-{m}", staleness=stale) for m in range(k)],
                    stats=GroupStats(k=k, mean=0.5, std=std, degenerate=std <= 1e-8),
                )
                for j in range(n_groups)
            ],
        )
        for i in range(n_steps)
    ]
    reading = measure_regime(make_run(steps))
    assert set(reading.conditions) == set(RegimeCondition)
    for cr in reading.conditions.values():
        assert cr.holds in (True, False, None)
        if cr.holds is None:
            assert math.isnan(cr.statistic) and cr.detail


@settings(max_examples=30, deadline=None)
@given(
    stale=st.integers(min_value=0, max_value=60),
    bound=st.integers(min_value=0, max_value=60),
)
def test_near_policy_is_exactly_the_staleness_comparison_on_single_policy_rollouts(stale, bound):
    g = group("g0", trajectories=[traj("a", staleness=stale)])
    cr = read(
        make_run([step(0, groups=[g])]),
        RegimeCondition.NEAR_POLICY,
        thresholds=RegimeThresholds(max_staleness_steps=bound),
    )
    assert cr.holds is (stale <= bound)


@settings(max_examples=25, deadline=None)
@given(rate=st.floats(min_value=0.01, max_value=2.0), tau=st.floats(min_value=0.1, max_value=20.0))
def test_the_adiabaticity_number_is_the_product_it_is_defined_as(rate, tau):
    run = make_run(_annealed(4, rate=rate))
    cr = read(run, RegimeCondition.QUASI_STATIC, inputs=RegimeInputs(tau_relax=tau))
    assert cr.statistic == pytest.approx(tau * rate, rel=1e-9)
    assert cr.holds is (cr.statistic <= 1.0)


def test_the_floors_are_the_only_thing_between_measured_and_undetermined():
    """Lowering a floor turns an undetermined reading into a measured one and nothing else."""
    rng = np.random.default_rng(7)
    run = _advantage_run(rng.normal(size=200))
    assert read(run, RegimeCondition.LIGHT_TAILED).holds is None
    relaxed = read(run, RegimeCondition.LIGHT_TAILED, floors=RegimeFloors(hill_min_exceedances=5))
    assert relaxed.holds is True

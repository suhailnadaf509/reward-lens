"""Acceptance: the canonical process record, its store, and the four things that gate it.

The specification's clause is three sentences: *a synthetic 401-step run with multi-turn
trajectories round-trips; a trajectory spanning two policy versions preserves both; a statistic
computed from a deliberately skewed sample is unbiased in a test.* Four more are asserted here
because each of them is a place where a schema can look correct and be wrong:

- the segment-provenance tiling invariant is enforced at construction, not checked later;
- a `RecomputeRef` that cannot be honoured becomes an `AbsentRef` and never a zero;
- `StepStream` reads steps 200 to 210 of a 401-step run without materialising the rest, asserted
  on what the reader touched rather than on the slice being right;
- a `verifiers`-shaped `TrajectoryStep` maps into a `Trajectory` with every one of the ten token
  fields preserved.

The last one uses a mapping written inline rather than a converter. `record/convert/` belongs
elsewhere; what is being asserted here is that the schema is *adequate* to hold the framework's
structure, which is the fact those packages depend on and the fact that is expensive to discover
late.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.envelope import RegimeCondition, RegimeReading
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Access, Component, Site, Substrate
from reward_lens.record import (
    AbsenceReason,
    AbsentRef,
    CaptureRef,
    CaptureSpec,
    CompactionEvent,
    ComponentRef,
    Engine,
    EstimatorSpec,
    Group,
    GroupID,
    GroupStats,
    InMemoryStepStream,
    InstrumentEffect,
    OptimizerTelemetry,
    ProbeResult,
    RecomputeRef,
    RecordReader,
    RecordSamplingPolicy,
    RecordWriter,
    RegimeDeclaration,
    Run,
    RunID,
    RunLineage,
    SamplingMeta,
    SamplingScheme,
    SegmentProvenance,
    StalenessDistribution,
    Step,
    TaskID,
    TensorStore,
    TilingError,
    ToolCall,
    Trajectory,
    Turn,
    make_trajectory,
    policy_versions,
    resolve,
)
from reward_lens.record.tensors import honour

N_STEPS = 401
CHUNK = 100
GROUPS_PER_STEP = 2
K = 2
TURNS_PER_TRAJECTORY = 3

VLLM = Engine(
    name="vllm",
    revision="0.11.2",
    attention_impl="flash_attn_2",
    dtype="bfloat16",
    compiled=False,
)
HF = Engine(name="transformers", revision="5.14.1", attention_impl="sdpa", dtype="bfloat16")


# ---------------------------------------------------------------------------
# The synthetic run
# ---------------------------------------------------------------------------


def _turns(step: int, k: int) -> tuple[Turn, ...]:
    """Three turns: a prompt, an assistant action with a tool call, and the tool response."""
    return (
        Turn(
            index=0,
            role="user",
            text=f"task at step {step}",
            token_ids=(101, 102, 103),
            loss_mask=(False, False, False),
        ),
        Turn(
            index=1,
            role="assistant",
            text="calling the tool",
            token_ids=(201 + k, 202, 203, 204),
            logprobs_sampling=(-0.11, -0.42, -1.30, -0.05),
            logprobs_train=(-0.13, -0.40, -1.33, -0.05),
            loss_mask=(True, True, True, True),
            tool_call=ToolCall(name="search", arguments='{"q": "x"}', ok=True, latency_ms=12.5),
            step_score=0.25 * k,
            step_advantage=0.1 * k,
        ),
        Turn(
            index=2,
            role="tool",
            text="tool said something",
            token_ids=(301, 302),
            loss_mask=(False, False),
        ),
    )


def _step(index: int) -> Step:
    groups = []
    for g in range(GROUPS_PER_STEP):
        trajectories = []
        for k in range(K):
            trajectories.append(
                make_trajectory(
                    id=f"traj:{index}.{g}.{k}",
                    task_ref=f"task:{g}",
                    turns=_turns(index, k),
                    scores={"total": float(k), "components": {"correct": float(k)}},
                    advantage=float(k) - 0.5,
                    policy_version=f"pv:{index}",
                    engine=VLLM,
                    features={"length": float(9 + k)},
                )
            )
        stats = GroupStats.from_scores(
            [t.advantage for t in trajectories], std_epsilon=1e-8, failure_at=-1.0
        )
        groups.append(
            Group(
                id=GroupID(f"grp:{index}.{g}"),
                task_ref=TaskID(f"task:{g}"),
                trajectories=tuple(trajectories),
                estimator=EstimatorSpec(
                    family="grpo",
                    group_centred=True,
                    std_normalised=True,
                    std_epsilon=1e-8,
                    degenerate_policy="zero",
                    clip_low=0.8,
                    clip_high=1.2,
                    aggregation="token",
                    loss_mask_policy="mask_environment",
                ),
                group_stats=stats,
            )
        )
    return Step(
        index=index,
        groups=tuple(groups),
        schedule={"lr": 1e-6, "beta": 0.04, "temperature": 1.0, "K": float(K)},
        optimizer=OptimizerTelemetry(
            grad_norm_clipped=1.0 + index * 1e-4,
            grad_norm_unclipped=1.2 + index * 1e-4,
            clip_fraction=0.03,
            kl_to_ref=0.011,
            kl_to_previous=0.004,
            entropy=1.75,
            update_norm=0.02,
            param_group_norms={"attn": 0.7, "mlp": 0.9},
            staleness=StalenessDistribution({0: GROUPS_PER_STEP * K}),
        ),
        probes=(ProbeResult(name="heldout_acc", channel="held_out", value=0.5, n=200),),
        instrument=InstrumentEffect(
            added_latency_ms_p50=0.4, added_latency_ms_p99=2.1, invocations=4
        ),
        regime_measured=RegimeReading.of(GROUP_NONDEGENERATE=True, NEAR_POLICY=True),
    )


def _run() -> Run:
    return Run(
        id=RunID("run:w21-acceptance"),
        kind="train",
        components={
            Component.POLICY: ComponentRef(
                name="policy", kind="actor", substrate=Substrate.NEURAL_GEN
            ),
            Component.GRADER: ComponentRef(
                name="composite", kind="rubric", substrate=Substrate.COMPOSITE
            ),
        },
        access={Component.POLICY: Access.RECORD, Component.GRADER: Access.RECORD | Access.QUERY},
        regime=RegimeDeclaration(
            declared={RegimeCondition.STATIONARY_GRADER: True},
            notes={RegimeCondition.STATIONARY_GRADER: "rubric weights frozen at step 0"},
            declared_by="the run's operator",
        ),
        steps=InMemoryStepStream([_step(i) for i in range(N_STEPS)]),
        lineage=RunLineage(
            git_sha="0" * 40, framework="trl", framework_version="1.9.2", library_version="3.0.0a1"
        ),
        sampling_policy=RecordSamplingPolicy(
            scheme=SamplingScheme.UNIFORM, rate=1.0, population=N_STEPS, recorded=N_STEPS
        ),
    )


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    root = tmp_path_factory.mktemp("record")
    report = RecordWriter(root, chunk_steps=CHUNK).write(_run())
    return root, report


# ---------------------------------------------------------------------------
# Clause 1 — a synthetic 401-step run with multi-turn trajectories round-trips
# ---------------------------------------------------------------------------


def test_a_401_step_run_with_multi_turn_trajectories_round_trips(written):
    root, report = written
    assert report.steps == N_STEPS
    assert report.groups == N_STEPS * GROUPS_PER_STEP
    assert report.trajectories == N_STEPS * GROUPS_PER_STEP * K
    assert report.turns == N_STEPS * GROUPS_PER_STEP * K * TURNS_PER_TRAJECTORY
    # 401 steps in partitions of 100 is five partitions, the last holding one step.
    assert report.partitions == 5

    reader = RecordReader(root, "run:w21-acceptance")
    run = reader.run()
    assert run.kind == "train"
    assert run.n_steps == N_STEPS
    assert run.components[Component.GRADER].substrate is Substrate.COMPOSITE
    assert run.access[Component.GRADER] == Access.RECORD | Access.QUERY
    assert run.regime.declares(RegimeCondition.STATIONARY_GRADER) is True
    assert run.lineage.framework_version == "1.9.2"
    assert run.sampling_policy.scheme is SamplingScheme.UNIFORM

    original = _step(137)
    got = reader.step(137)
    assert got.index == original.index
    assert got.schedule == original.schedule
    assert got.optimizer.grad_norm_unclipped == pytest.approx(
        original.optimizer.grad_norm_unclipped
    )
    assert got.optimizer.param_group_norms == {"attn": 0.7, "mlp": 0.9}
    assert got.optimizer.staleness.counts == {0: GROUPS_PER_STEP * K}
    assert got.probes[0].channel == "held_out"
    assert got.instrument.added_latency_ms_p99 == pytest.approx(2.1)
    assert got.regime_measured.holds(RegimeCondition.NEAR_POLICY) is True

    assert len(got.groups) == GROUPS_PER_STEP
    group = got.groups[0]
    assert group.k == K
    assert group.estimator.z_scored is True
    assert group.estimator.std_epsilon == 1e-8
    assert group.estimator.loss_mask_policy == "mask_environment"
    assert group.group_stats.k == K

    traj = group.trajectories[1]
    assert traj.n_turns == TURNS_PER_TRAJECTORY
    assert traj.scores == {"total": 1.0, "components": {"correct": 1.0}}
    assert traj.advantage == pytest.approx(0.5)
    assert traj.features == {"length": 10.0}

    action = traj.turns[1]
    assert action.role == "assistant"
    assert action.token_ids == (202, 202, 203, 204)
    assert action.logprobs_sampling == (-0.11, -0.42, -1.30, -0.05)
    assert action.logprobs_train == (-0.13, -0.40, -1.33, -0.05)
    assert action.loss_mask == (True, True, True, True)
    assert action.tool_call is not None and action.tool_call.name == "search"
    assert action.tool_call.latency_ms == pytest.approx(12.5)
    assert action.step_score == pytest.approx(0.25)
    assert action.step_advantage == pytest.approx(0.1)
    assert traj.turns[2].role == "tool"
    assert traj.turns[0].loss_mask == (False, False, False)


def test_the_two_logprob_streams_survive_as_two_streams(written):
    """E6 is only computable if the record kept both. This is the field-level check."""
    root, _ = written
    action = RecordReader(root, "run:w21-acceptance").step(200).groups[0].trajectories[0].turns[1]
    assert action.has_both_logprob_streams
    gap = action.logprob_gap()
    assert gap is not None
    assert gap[0] == pytest.approx(-0.02, abs=1e-9)
    assert gap[2] == pytest.approx(-0.03, abs=1e-9)


# ---------------------------------------------------------------------------
# Clause 2 — a trajectory spanning two policy versions preserves both
# ---------------------------------------------------------------------------


def _partial_rollout_trajectory() -> Trajectory:
    """The Kimi-K3 shape: a rollout started under one policy and finished under a newer one.

    Five turns, split 2/3 across two policy versions, with the resumption served by a different
    engine at three optimizer steps of staleness, and a prefix compaction at the resumption point.
    """
    turns = tuple(
        Turn(
            index=i,
            role="assistant" if i % 2 == 0 else "tool",
            text=f"turn {i}",
            token_ids=tuple(range(1000 + 10 * i, 1000 + 10 * i + 4)),
            logprobs_sampling=(-0.1, -0.2, -0.3, -0.4),
            loss_mask=(i % 2 == 0,) * 4,
        )
        for i in range(5)
    )
    return Trajectory(
        id="traj:partial",
        task_ref=TaskID("task:long"),
        turns=turns,
        scores={"total": 0.75},
        advantage=0.2,
        advantage_tokens=None,
        provenance=(
            SegmentProvenance(
                turn_range=(0, 2),
                policy_version="pv:1200",
                staleness_steps=0,
                engine=VLLM,
                sampling=SamplingMeta(temperature=1.0, top_p=0.95, seed=7),
            ),
            SegmentProvenance(
                turn_range=(2, 5),
                policy_version="pv:1203",
                staleness_steps=3,
                engine=HF,
                sampling=SamplingMeta(temperature=0.7, top_p=1.0, seed=11),
            ),
        ),
        compaction=(
            CompactionEvent(at_turn=2, tokens_before=8, tokens_after=3, method="summarise_prefix"),
        ),
        labels={},
        features={},
        capture=None,
    )


def test_a_trajectory_spanning_two_policy_versions_preserves_both(tmp_path):
    traj = _partial_rollout_trajectory()
    step = Step(
        index=1203,
        groups=(
            Group(
                id=GroupID("grp:partial"),
                task_ref=TaskID("task:long"),
                trajectories=(traj,),
                estimator=EstimatorSpec(family="grpo", group_centred=True),
                group_stats=GroupStats(k=1),
            ),
        ),
        schedule={"lr": 5e-7},
        optimizer=OptimizerTelemetry(staleness=StalenessDistribution({0: 1, 3: 1})),
    )
    run = Run(
        id=RunID("run:partial"),
        kind="train",
        components={},
        access={},
        regime=RegimeDeclaration(),
        steps=InMemoryStepStream([step]),
    )
    RecordWriter(tmp_path, chunk_steps=CHUNK).write(run)

    got = RecordReader(tmp_path, "run:partial").step(1203).groups[0].trajectories[0]

    assert len(got.provenance) == 2
    assert policy_versions(got.provenance) == ("pv:1200", "pv:1203")
    first, second = sorted(got.provenance, key=lambda s: s.turn_range)
    assert first.turn_range == (0, 2)
    assert second.turn_range == (2, 5)
    assert first.staleness_steps == 0
    assert second.staleness_steps == 3
    # The engine identity survives in full, not as an opaque string: the limit of detection is
    # cached per (model, engine, revision, dtype, attention_impl, layer).
    assert first.engine.name == "vllm"
    assert first.engine.attention_impl == "flash_attn_2"
    assert first.engine.compiled is False
    assert second.engine.name == "transformers"
    assert second.engine.revision == "5.14.1"
    assert first.engine.id != second.engine.id
    assert first.sampling.seed == 7 and second.sampling.seed == 11
    assert second.sampling.temperature == pytest.approx(0.7)

    # The two facts NEAR_POLICY reads, and the one NO_COMPACTION reads.
    assert got.is_single_policy is False
    assert got.max_staleness == 3
    assert got.was_compacted is True
    assert got.compaction[0].method == "summarise_prefix"
    assert got.compaction[0].tokens_removed == 5

    # And the turns are attributable to their generating policy after the round trip.
    assert [t.index for t in got.turns_of("pv:1200")] == [0, 1]
    assert [t.index for t in got.turns_of("pv:1203")] == [2, 3, 4]


def test_a_kimi_k3_shaped_rollout_holds_three_policies_and_two_compactions(tmp_path):
    """The full partial-rollout shape, not the two-segment simplification of it.

    Twelve turns resumed twice, so three policy versions at 0, 40 and 41 steps of staleness, served
    by two different engines, with two prefix rewrites by different methods. Every one of those is a
    fact some instrument's envelope turns on, and the question this answers is whether the schema
    can hold the shape at all rather than whether it happens to hold the easy case.
    """
    turns = tuple(
        Turn(index=i, role="assistant" if i % 2 == 0 else "tool", token_ids=(i, i + 1))
        for i in range(12)
    )
    traj = Trajectory(
        id="traj:k3",
        task_ref=TaskID("task:cyber"),
        turns=turns,
        scores={"total": -1.0, "override_fired": True},
        advantage=-0.9,
        advantage_tokens=AbsentRef.of(
            AbsenceReason.COMPACTED, detail="prefix rewritten twice inside this rollout"
        ),
        provenance=(
            SegmentProvenance(
                turn_range=(0, 4), policy_version="pv:900", staleness_steps=0, engine=VLLM
            ),
            SegmentProvenance(
                turn_range=(4, 9), policy_version="pv:940", staleness_steps=40, engine=VLLM
            ),
            SegmentProvenance(
                turn_range=(9, 12), policy_version="pv:941", staleness_steps=41, engine=HF
            ),
        ),
        compaction=(
            CompactionEvent(
                at_turn=4, tokens_before=131072, tokens_after=8192, method="summarise_prefix"
            ),
            CompactionEvent(
                at_turn=9, tokens_before=131072, tokens_after=16384, method="drop_tool_output"
            ),
        ),
        labels={},
        features={"output_tokens": 131072.0},
        capture=None,
    )
    step = Step(
        index=941,
        groups=(
            Group(
                id=GroupID("grp:k3"),
                task_ref=TaskID("task:cyber"),
                trajectories=(traj,),
                estimator=EstimatorSpec(
                    family="grpo", group_centred=True, off_policy_correction="none"
                ),
                group_stats=GroupStats(k=1),
            ),
        ),
        schedule={"tau": 0.9},
        optimizer=OptimizerTelemetry(staleness=StalenessDistribution({0: 1, 40: 1, 41: 1})),
    )
    run = Run(
        id=RunID("run:k3"),
        kind="train",
        components={},
        access={},
        regime=RegimeDeclaration(),
        steps=InMemoryStepStream([step]),
    )
    RecordWriter(tmp_path).write(run)
    got = RecordReader(tmp_path, "run:k3").step(941).groups[0].trajectories[0]

    assert policy_versions(got.provenance) == ("pv:900", "pv:940", "pv:941")
    assert [s.staleness_steps for s in sorted(got.provenance, key=lambda s: s.turn_range)] == [
        0,
        40,
        41,
    ]
    assert {s.engine.name for s in got.provenance} == {"vllm", "transformers"}
    assert len(got.compaction) == 2
    assert [c.at_turn for c in got.compaction] == [4, 9]
    assert sum(c.tokens_removed for c in got.compaction) == 237_568
    # The per-token advantage is honestly absent because the prefix it lived on no longer exists,
    # and the record says COMPACTED rather than handing back zeros.
    assert isinstance(got.advantage_tokens, AbsentRef)
    assert got.advantage_tokens.reason is AbsenceReason.COMPACTED
    # And the window arithmetic NO_COMPACTION is measured from survives the round trip.
    from reward_lens.record import crosses

    assert crosses(got.compaction, first_turn=0, last_turn=11) is True
    assert crosses(got.compaction, first_turn=9, last_turn=11) is False


def test_a_recompute_ref_survives_the_round_trip_as_a_recipe(tmp_path):
    """The union's third member in the record: a recipe, not bytes and not an absence."""
    recipe = RecomputeRef(
        model_fp="mfp:70b",
        engine=VLLM,
        token_ids=(9, 8, 7),
        sampling_seed=1234,
        capture_spec=CaptureSpec(
            sites=(Site(layer=40, point="resid_post"),), dtype="bfloat16", stride=8
        ),
        expected_numerics_floor=0.0083,
    )
    traj = make_trajectory(
        id="traj:recipe",
        task_ref="task",
        turns=(Turn(index=0, role="assistant", token_ids=(9, 8, 7)),),
        capture=CaptureRef(spec=recipe.capture_spec, tensors={"resid_40": recipe}),
        engine=VLLM,
    )
    run = Run(
        id=RunID("run:recipe"),
        kind="audit",
        components={},
        access={},
        regime=RegimeDeclaration(),
        steps=InMemoryStepStream(
            [
                Step(
                    index=0,
                    groups=(
                        Group(
                            id=GroupID("g"),
                            task_ref=TaskID("task"),
                            trajectories=(traj,),
                            estimator=EstimatorSpec(),
                            group_stats=GroupStats(k=1),
                        ),
                    ),
                    schedule={},
                    optimizer=OptimizerTelemetry(),
                )
            ]
        ),
    )
    RecordWriter(tmp_path).write(run)
    got = RecordReader(tmp_path, "run:recipe").step(0).groups[0].trajectories[0]
    ref = got.capture.tensors["resid_40"]
    assert isinstance(ref, RecomputeRef)
    assert ref.model_fp == "mfp:70b"
    assert ref.token_ids == (9, 8, 7)
    assert ref.sampling_seed == 1234
    assert ref.expected_numerics_floor == pytest.approx(0.0083)
    assert ref.engine.attention_impl == "flash_attn_2"
    assert ref.capture_spec.sites == (Site(layer=40, point="resid_post"),)
    assert ref.capture_spec.stride == 8
    assert ref.id == recipe.id
    # And with nobody to run it, resolving is an absence rather than an array of zeros.
    assert isinstance(resolve(ref), AbsentRef)


def test_the_seam_for_scores_and_labels_round_trips_a_registered_payload(tmp_path):
    """What the score tree and the blind labels have to do to plug in: decorate their dataclasses.

    The score tree and the blind label go through the kernel's `ValueCodec`, which reconstructs any
    dataclass registered with `register_payload`. This stands in a stub for each and asserts the
    contract, so the two later packages find out at merge time rather than at read time.
    """
    from dataclasses import dataclass as _dataclass

    from reward_lens.core.evidence import register_payload

    @register_payload
    @_dataclass(frozen=True)
    class _StubLeaf:
        name: str
        value: float | None
        abstained: bool = False

    @register_payload
    @_dataclass(frozen=True)
    class _StubBlind:
        payload: float

    traj = make_trajectory(
        id="traj:seam",
        task_ref="task",
        turns=(Turn(index=0, role="assistant"),),
        scores=_StubLeaf(name="correctness", value=None, abstained=True),
        labels={"hacked": _StubBlind(payload=1.0)},
        engine=VLLM,
    )
    run = Run(
        id=RunID("run:seam"),
        kind="eval",
        components={},
        access={},
        regime=RegimeDeclaration(),
        steps=InMemoryStepStream(
            [
                Step(
                    index=0,
                    groups=(
                        Group(
                            id=GroupID("g"),
                            task_ref=TaskID("task"),
                            trajectories=(traj,),
                            estimator=EstimatorSpec(),
                            group_stats=GroupStats(k=1),
                        ),
                    ),
                    schedule={},
                    optimizer=OptimizerTelemetry(),
                )
            ]
        ),
    )
    RecordWriter(tmp_path).write(run)
    got = RecordReader(tmp_path, "run:seam").step(0).groups[0].trajectories[0]
    assert isinstance(got.scores, _StubLeaf)
    assert got.scores.value is None and got.scores.abstained is True
    # The blind label comes back as a blind, not as its payload. The codec never unwraps.
    assert isinstance(got.labels["hacked"], _StubBlind)


# ---------------------------------------------------------------------------
# Clause 3 — a statistic from a deliberately skewed sample is unbiased
# ---------------------------------------------------------------------------


def test_a_statistic_from_a_deliberately_skewed_sample_is_unbiased():
    """Nine of every ten high-reward trajectories recorded, one of every ten low-reward ones.

    The naive mean of what survived is wrong by a wide margin and stays wrong at any sample size,
    because the bias is in the design rather than in the noise. The design-corrected mean converges
    on the population value, and "unbiased" is asserted the only way it can be: over replicates,
    against the Monte Carlo standard error of the estimator itself.
    """
    rng = np.random.default_rng(20260801)
    n_population = 5000
    high = rng.normal(0.8, 0.1, n_population // 2)
    low = rng.normal(0.2, 0.1, n_population // 2)
    population = np.concatenate([high, low])
    strata = np.array(["high"] * (n_population // 2) + ["low"] * (n_population // 2))
    truth = float(population.mean())

    policy = RecordSamplingPolicy(
        scheme=SamplingScheme.STRATIFIED,
        strata_rates={"high": 0.9, "low": 0.1},
        stratum_key="reward_band",
        unit="trajectory",
    )

    replicates = 200
    corrected = np.empty(replicates)
    naive = np.empty(replicates)
    for r in range(replicates):
        keep = rng.random(n_population) < np.where(strata == "high", 0.9, 0.1)
        values = population[keep]
        got = policy.estimate_mean(values, strata=list(strata[keep]))
        assert not isinstance(got, Refusal), got
        corrected[r] = got.value
        naive[r] = float(values.mean())

    se = float(corrected.std(ddof=1) / np.sqrt(replicates))
    bias = float(corrected.mean() - truth)
    assert abs(bias) < 4 * se, f"corrected mean is biased by {bias:.5f} against se {se:.5f}"

    naive_bias = float(naive.mean() - truth)
    assert naive_bias > 20 * se, "the skew was not strong enough for this test to mean anything"
    assert naive_bias == pytest.approx(0.24, abs=0.02)


def test_the_correction_reports_its_own_design_effect():
    policy = RecordSamplingPolicy(
        scheme=SamplingScheme.STRATIFIED, strata_rates={"high": 0.9, "low": 0.1}
    )
    got = policy.estimate_mean([1.0] * 90 + [0.0] * 10, strata=["high"] * 90 + ["low"] * 10)
    assert not isinstance(got, Refusal)
    assert got.n == 100
    assert got.method == "hajek"
    # Weights are 1/0.9 on ninety units and 1/0.1 on ten, so the ten rare units carry more than
    # half the weight and the effective sample size is well under a hundred.
    assert got.ess < 70
    assert got.design_effect > 1.4
    assert got.max_weight == pytest.approx(10.0)


def test_a_selective_sampling_policy_refuses_rather_than_returning_a_biased_number():
    """Top-k selection gives some units inclusion probability zero. No reweighting recovers them."""
    policy = RecordSamplingPolicy(
        scheme=SamplingScheme.SELECTIVE, selected_by="kept the 100 highest-reward rollouts per step"
    )
    got = policy.estimate_mean([1.0, 2.0, 3.0])
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "inclusion probability of zero" in got.detail
    assert got.remedy.strip()


def test_a_stratum_with_no_recorded_rate_refuses_and_names_it():
    policy = RecordSamplingPolicy(scheme=SamplingScheme.STRATIFIED, strata_rates={"high": 0.9})
    got = policy.estimate_mean([1.0, 2.0], strata=["high", "low"])
    assert isinstance(got, Refusal)
    assert "'low'" in got.detail


def test_an_effective_sample_size_below_a_declared_floor_refuses_with_both_numbers():
    policy = RecordSamplingPolicy(
        scheme=SamplingScheme.STRATIFIED, strata_rates={"common": 1.0, "rare": 1e-4}
    )
    got = policy.estimate_mean([1.0] * 20 + [5.0], strata=["common"] * 20 + ["rare"], min_ess=10.0)
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.ESS_BELOW_FLOOR
    assert got.statistics["floor"] == 10.0
    assert got.statistics["ess"] < 10.0


# ---------------------------------------------------------------------------
# Clause 4 — the tiling invariant is enforced at construction
# ---------------------------------------------------------------------------


def _traj_with(provenance, n_turns=4):
    turns = tuple(Turn(index=i, role="assistant", text="t") for i in range(n_turns))
    return Trajectory(
        id="traj:tiling",
        task_ref=TaskID("task"),
        turns=turns,
        scores=None,
        advantage=None,
        advantage_tokens=None,
        provenance=tuple(provenance),
        compaction=(),
        labels={},
        features={},
        capture=None,
    )


def _seg(lo, hi, version="pv:0", staleness=0):
    return SegmentProvenance(
        turn_range=(lo, hi), policy_version=version, staleness_steps=staleness, engine=VLLM
    )


def test_a_tiling_with_a_gap_is_refused_at_construction():
    with pytest.raises(TilingError) as exc:
        _traj_with([_seg(0, 2), _seg(3, 4)])
    assert "gap" in str(exc.value)
    assert "turns 2 to 2" in str(exc.value)


def test_a_tiling_with_an_overlap_is_refused_at_construction():
    with pytest.raises(TilingError) as exc:
        _traj_with([_seg(0, 3, "pv:a"), _seg(2, 4, "pv:b")])
    assert "overlap" in str(exc.value)
    assert "pv:a" in str(exc.value) and "pv:b" in str(exc.value)


def test_provenance_that_does_not_reach_the_last_turn_is_refused():
    with pytest.raises(TilingError, match="ends at turn 3"):
        _traj_with([_seg(0, 3)])


def test_provenance_that_does_not_start_at_the_first_turn_is_refused():
    with pytest.raises(TilingError, match="starts at turn 1"):
        _traj_with([_seg(1, 4)])


def test_a_trajectory_with_turns_and_no_provenance_is_refused():
    with pytest.raises(TilingError, match="mandatory"):
        _traj_with([])


def test_an_exact_tiling_across_two_policies_constructs():
    traj = _traj_with([_seg(0, 1, "pv:a"), _seg(1, 4, "pv:b", staleness=2)])
    assert traj.is_single_policy is False
    assert traj.max_staleness == 2


def test_a_compaction_beyond_the_last_turn_is_refused():
    turns = tuple(Turn(index=i, role="assistant") for i in range(3))
    with pytest.raises(ValueError, match="compaction at turn 9"):
        Trajectory(
            id="traj:c",
            task_ref=TaskID("t"),
            turns=turns,
            scores=None,
            advantage=None,
            advantage_tokens=None,
            provenance=(_seg(0, 3),),
            compaction=(CompactionEvent(at_turn=9, tokens_before=4, tokens_after=1, method="x"),),
            labels={},
            features={},
            capture=None,
        )


# ---------------------------------------------------------------------------
# Clause 5 — an unhonourable RecomputeRef becomes an AbsentRef, never a zero
# ---------------------------------------------------------------------------


def _recipe(floor: float = 0.0035) -> RecomputeRef:
    return RecomputeRef(
        model_fp="mfp:deadbeef",
        engine=VLLM,
        token_ids=(1, 2, 3, 4),
        sampling_seed=17,
        capture_spec=CaptureSpec(dtype="bfloat16", stride=1),
        expected_numerics_floor=floor,
    )


class _Declines:
    def can_honour(self, ref):
        return False

    def recompute(self, ref):  # pragma: no cover - never reached
        raise AssertionError("recompute must not be called after can_honour said no")


class _ReturnsZerosWithBadAgreement:
    """The exact failure mode this design exists to prevent, made concrete.

    A recomputer that hands back a zero array and reports 3.28% disagreement, which is the hybrid
    MoE number, against an expected floor of 0.35%.
    """

    def can_honour(self, ref):
        return True

    def recompute(self, ref):
        return np.zeros(8), 0.0328


class _CannotMeasureItsOwnAgreement:
    def can_honour(self, ref):
        return True

    def recompute(self, ref):
        return np.ones(8), float("nan")


class _Faithful:
    def can_honour(self, ref):
        return True

    def recompute(self, ref):
        return np.arange(8, dtype=float), 0.0018


def test_a_recompute_ref_with_no_recomputer_is_absent_not_zero():
    got = honour(_recipe(), None)
    assert isinstance(got, AbsentRef)
    assert got.reason is AbsenceReason.RECOMPUTE_UNSUPPORTED
    assert not isinstance(got, np.ndarray)
    assert got.remedy.strip()


def test_a_declining_recomputer_gives_an_absent_ref_naming_the_engine():
    got = honour(_recipe(), _Declines())
    assert isinstance(got, AbsentRef)
    assert got.reason is AbsenceReason.RECOMPUTE_UNAVAILABLE
    assert "vllm" in got.detail


def test_a_recompute_that_misses_the_numerics_floor_is_absent_and_carries_both_numbers():
    got = honour(_recipe(floor=0.0035), _ReturnsZerosWithBadAgreement())
    assert isinstance(got, AbsentRef)
    assert got.reason is AbsenceReason.NUMERICS_FLOOR_EXCEEDED
    assert got.statistics["achieved"] == pytest.approx(0.0328)
    assert got.statistics["expected"] == pytest.approx(0.0035)
    # The zeros the recomputer produced never escape.
    assert not isinstance(got, np.ndarray)


def test_an_unmeasured_agreement_is_not_a_met_floor():
    got = honour(_recipe(), _CannotMeasureItsOwnAgreement())
    assert isinstance(got, AbsentRef)
    assert got.reason is AbsenceReason.NUMERICS_FLOOR_EXCEEDED


def test_a_recompute_inside_the_floor_returns_the_array():
    got = honour(_recipe(floor=0.0035), _Faithful())
    assert isinstance(got, np.ndarray)
    assert got.tolist() == list(range(8))


def test_an_absent_ref_converts_to_a_refusal_with_a_remedy():
    got = honour(_recipe(), None)
    refusal = got.as_refusal("policy.selection_term")
    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert refusal.statistics["absence"] == "RECOMPUTE_UNSUPPORTED"
    assert "Recomputer" in refusal.remedy


def test_an_absent_tensor_ref_survives_the_round_trip_as_an_absence(tmp_path):
    """The bit that matters: absence is a value in the record, not a missing key or a zero."""
    absent = AbsentRef.of(AbsenceReason.NOT_CAPTURED, detail="egress budget was 0")
    traj = make_trajectory(
        id="traj:absent",
        task_ref="task",
        turns=(Turn(index=0, role="assistant", token_ids=(1, 2)),),
        advantage_tokens=absent,
        capture=CaptureRef(spec=CaptureSpec(include_routing=True), tensors={"resid": absent}),
        engine=VLLM,
    )
    step = Step(
        index=0,
        groups=(
            Group(
                id=GroupID("g"),
                task_ref=TaskID("task"),
                trajectories=(traj,),
                estimator=EstimatorSpec(),
                group_stats=GroupStats(k=1),
            ),
        ),
        schedule={},
        optimizer=OptimizerTelemetry(),
    )
    run = Run(
        id=RunID("run:absent"),
        kind="audit",
        components={},
        access={},
        regime=RegimeDeclaration(),
        steps=InMemoryStepStream([step]),
    )
    RecordWriter(tmp_path).write(run)
    got = RecordReader(tmp_path, "run:absent").step(0).groups[0].trajectories[0]
    assert isinstance(got.advantage_tokens, AbsentRef)
    assert got.advantage_tokens.reason is AbsenceReason.NOT_CAPTURED
    assert got.advantage_tokens.detail == "egress budget was 0"
    assert isinstance(got.capture.tensors["resid"], AbsentRef)
    resolved = resolve(got.capture.tensors["resid"])
    assert isinstance(resolved, AbsentRef)


# ---------------------------------------------------------------------------
# Clause 6 — steps 200 to 210 without materialising the rest
# ---------------------------------------------------------------------------


def test_reading_steps_200_to_210_does_not_materialise_the_rest(written):
    """Asserted on what the reader touched, not on the slice being right.

    A reader that decoded all 401 steps and returned eleven would pass a correctness assertion and
    fail this one, which is the whole reason the counters exist.
    """
    root, _ = written
    reader = RecordReader(root, "run:w21-acceptance")
    stream = reader.run().steps

    assert len(stream) == N_STEPS
    assert reader.stats.files_opened == [], "opening the run must read no step data"

    reader.stats.reset()
    got = list(stream.slice(200, 211))

    assert [s.index for s in got] == list(range(200, 211))
    assert reader.stats.steps_materialised == 11
    assert reader.stats.groups_materialised == 11 * GROUPS_PER_STEP
    assert reader.stats.trajectories_materialised == 11 * GROUPS_PER_STEP * K
    assert reader.stats.turns_materialised == 11 * GROUPS_PER_STEP * K * TURNS_PER_TRAJECTORY

    # One partition out of five, and no file from any other.
    assert reader.stats.partitions_opened == ("step_range=000200-000299",)
    assert len(reader.stats.files_opened) == 4

    # And the bytes: one partition of five, so well under a third of the run.
    fraction = reader.stats.bytes_read / reader.total_bytes()
    assert fraction < 0.30, f"read {fraction:.1%} of the record to answer for 11 of 401 steps"

    # Rows were scanned to find the wanted ones, but only the wanted ones were decoded.
    assert reader.stats.rows_decoded < reader.stats.rows_scanned


def test_the_stream_is_resumable_from_an_arbitrary_step(written):
    root, _ = written
    reader = RecordReader(root, "run:w21-acceptance")
    reader.stats.reset()
    got = [s.index for s in reader.run().steps.resume_from(398)]
    assert got == [398, 399, 400]
    assert reader.stats.partitions_opened == (
        "step_range=000300-000399",
        "step_range=000400-000499",
    )


def test_the_stream_chunks_at_the_partition_size(written):
    root, _ = written
    stream = RecordReader(root, "run:w21-acceptance").run().steps
    sizes = [len(c) for c in stream.chunks()]
    assert sizes == [CHUNK, CHUNK, CHUNK, CHUNK, 1]


def test_a_full_iteration_still_yields_every_step_in_order(written):
    root, _ = written
    indices = [s.index for s in RecordReader(root, "run:w21-acceptance").run().steps]
    assert indices == list(range(N_STEPS))


# ---------------------------------------------------------------------------
# Clause 7 — a verifiers TrajectoryStep maps in with all ten token fields
# ---------------------------------------------------------------------------

#: `verifiers/types.py:256-272` and `:300-309` at commit edafab85, as re-verified in
#: RECON-4-FRAMEWORKS. Both are TypedDicts; `multi_modal_data` and `prompt_attribution` are
#: `NotRequired[Any]`, so they are optional and untyped at the source.
VERIFIERS_TOKEN_FIELDS = (
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


def _verifiers_trajectory_step() -> dict:
    return {
        "prompt": [{"role": "system", "content": "be terse"}, {"role": "user", "content": "2+2?"}],
        "completion": [{"role": "assistant", "content": "4"}],
        "response": {"id": "resp-1", "model": "qwen3-8b", "finish_reason": "length"},
        "tokens": {
            "prompt_ids": [1, 2, 3, 4, 5],
            "prompt_mask": [0, 0, 0, 0, 0],
            "completion_ids": [61, 62, 63],
            "completion_mask": [1, 1, 0],
            "completion_logprobs": [-0.25, -0.5, -1.5],
            "overlong_prompt": False,
            "is_truncated": True,
            "routed_experts": [[3, 11], [4, 11], [7, 2]],
            "multi_modal_data": {"pixel_values_shape": [1, 3, 224, 224]},
            "prompt_attribution": {"segments": [[0, 2], [2, 5]]},
        },
        "reward": 0.5,
        "advantage": -0.125,
        "is_truncated": True,
        "trajectory_id": "vf-traj-0",
        "extras": {"task": "gsm8k", "attempt": 2},
    }


def _map_verifiers_step(vstep: dict, store: TensorStore) -> Trajectory:
    """The mapping the converter will own, written here to show the schema is adequate to hold it.

    Three turns come out of one `TrajectoryStep`: the system message, the user message, and the
    assistant completion. Splitting the prompt by message is what makes the system role survive,
    and the system role is why `TurnRole` carries a fifth member the printed schema does not.
    """
    tok = vstep["tokens"]
    routed = store.put(np.asarray(tok["routed_experts"], dtype=np.int32), name="routed_experts")

    prompt_turns = []
    # The prompt's token ids are one flat array over the whole message list, so they attach to the
    # last prompt turn and the earlier ones carry text only. `prompt_attribution` is the framework's
    # own map from segments back to messages and it rides along untyped, as it does at the source.
    for i, message in enumerate(vstep["prompt"]):
        last = i == len(vstep["prompt"]) - 1
        prompt_turns.append(
            Turn(
                index=i,
                role=message["role"],
                text=message["content"],
                token_ids=tuple(tok["prompt_ids"]) if last else None,
                loss_mask=tuple(bool(m) for m in tok["prompt_mask"]) if last else None,
                overlong_prompt=tok["overlong_prompt"] if last else None,
                extra=({"prompt_attribution": tok.get("prompt_attribution")} if last else {}),
            )
        )
    completion = Turn(
        index=len(prompt_turns),
        role="assistant",
        text=vstep["completion"][0]["content"],
        token_ids=tuple(tok["completion_ids"]),
        loss_mask=tuple(bool(m) for m in tok["completion_mask"]),
        logprobs_sampling=tuple(tok["completion_logprobs"]),
        logprobs_train=None,
        truncated=tok["is_truncated"],
        step_score=vstep["reward"],
        step_advantage=vstep["advantage"],
        tensors={"routed_experts": routed},
        extra={
            "multi_modal_data": tok.get("multi_modal_data"),
            "response": vstep["response"],
            "extras": vstep["extras"],
        },
    )
    return make_trajectory(
        id=vstep["trajectory_id"],
        task_ref="task:gsm8k",
        turns=(*prompt_turns, completion),
        scores={"reward": vstep["reward"]},
        advantage=vstep["advantage"],
        policy_version="pv:vf",
        # verifiers score_group is mean-centred with no standard-deviation division
        # (rubric.py:406-409), so the estimator recorded beside it must say so.
        engine=Engine(name="vllm", revision="unknown", dtype="bfloat16"),
    )


def test_a_verifiers_trajectory_step_maps_in_with_all_ten_token_fields(tmp_path):
    vstep = _verifiers_trajectory_step()
    assert set(vstep["tokens"]) == set(VERIFIERS_TOKEN_FIELDS), "the source field list drifted"

    store = TensorStore(tmp_path / "tensors")
    traj = _map_verifiers_step(vstep, store)
    store.flush()

    step = Step(
        index=0,
        groups=(
            Group(
                id=GroupID("grp:vf"),
                task_ref=TaskID("task:gsm8k"),
                trajectories=(traj,),
                estimator=EstimatorSpec(
                    family="verifiers",
                    group_centred=True,
                    std_normalised=False,
                    aggregation="sequence",
                ),
                group_stats=GroupStats(k=1),
            ),
        ),
        schedule={},
        optimizer=OptimizerTelemetry(),
    )
    run = Run(
        id=RunID("run:vf"),
        kind="train",
        components={
            Component.GRADER: ComponentRef(
                name="Rubric", kind="rubric", substrate=Substrate.COMPOSITE
            )
        },
        access={},
        regime=RegimeDeclaration(),
        steps=InMemoryStepStream([step]),
        lineage=RunLineage(framework="verifiers", framework_version="0.2.1+edafab85"),
    )
    writer = RecordWriter(tmp_path / "record")
    writer.open_run(run)
    writer.append_step(step)
    writer.close()

    # The tensor store the mapping wrote lives beside the record; point the reader at it.
    got = RecordReader(tmp_path / "record", "run:vf").step(0).groups[0].trajectories[0]
    prompt = got.turns[1]
    completion = got.turns[2]
    tok = vstep["tokens"]

    # 1, 2 — prompt ids and mask
    assert list(prompt.token_ids) == tok["prompt_ids"]
    assert [int(b) for b in prompt.loss_mask] == tok["prompt_mask"]
    # 3, 4 — completion ids and mask
    assert list(completion.token_ids) == tok["completion_ids"]
    assert [int(b) for b in completion.loss_mask] == tok["completion_mask"]
    # 5 — completion logprobs, in the sampling stream because that is where they came from
    assert list(completion.logprobs_sampling) == tok["completion_logprobs"]
    assert completion.logprobs_train is None
    # 6 — overlong_prompt
    assert prompt.overlong_prompt is tok["overlong_prompt"]
    # 7 — is_truncated
    assert completion.truncated is tok["is_truncated"]
    # 8 — routed_experts, through the tensor store rather than inline
    ref = completion.tensors["routed_experts"]
    array = resolve(ref, store=TensorStore(tmp_path / "tensors"))
    assert isinstance(array, np.ndarray)
    assert array.tolist() == tok["routed_experts"]
    # 9, 10 — the two NotRequired[Any] fields, carried untyped as they are at the source
    assert completion.extra["multi_modal_data"] == tok["multi_modal_data"]
    assert prompt.extra["prompt_attribution"] == tok["prompt_attribution"]

    # And the TrajectoryStep's own nine fields.
    assert got.turns[0].role == "system" and got.turns[0].text == "be terse"
    assert got.turns[1].role == "user"
    assert completion.text == "4"
    assert completion.extra["response"]["finish_reason"] == "length"
    assert completion.step_score == pytest.approx(0.5)
    assert completion.step_advantage == pytest.approx(-0.125)
    assert str(got.id) == "vf-traj-0"
    assert completion.extra["extras"] == {"task": "gsm8k", "attempt": 2}
    # verifiers mean-centres and does not divide by a standard deviation, so E4's amplification
    # mechanism is absent on this framework and the record says so.
    assert RecordReader(tmp_path / "record", "run:vf").step(0).groups[0].estimator.z_scored is False

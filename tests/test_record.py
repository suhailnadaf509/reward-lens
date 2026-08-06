"""Unit tests for `reward_lens.record`: the schema's invariants, the store, and the arithmetic.

The acceptance file asserts the specification's clauses. This one asserts the parts underneath
them, and in particular the things that are cheap to get subtly wrong: the half-open convention on
turn ranges, whether an abstention lands in a group mean, whether a NaN survives JSON, whether the
tensor store deduplicates, and whether importing this package pulls a compiled dependency.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest

from reward_lens.core.envelope import RegimeCondition, RegimeReading
from reward_lens.core.reading import Refusal
from reward_lens.core.types import Access, Component, Site
from reward_lens.record import (
    AbsenceReason,
    AbsentRef,
    CaptureSpec,
    CompactionEvent,
    ComponentRef,
    Engine,
    EstimatorSpec,
    Group,
    GroupID,
    GroupStats,
    InMemoryStepStream,
    OptimizerTelemetry,
    RecordNotFound,
    RecordReader,
    RecordSamplingPolicy,
    RecordWriter,
    RegimeDeclaration,
    Run,
    RunID,
    SamplingScheme,
    SegmentProvenance,
    StalenessDistribution,
    Step,
    StoredRef,
    TaskID,
    TensorStore,
    TilingError,
    Turn,
    check_tiling,
    crosses,
    events_in_window,
    group_id,
    logprob_mismatch,
    make_trajectory,
    mask_policy_signature,
    policy_mixture,
    renumber,
    residual_bytes,
    resolve,
    run_id,
    single_segment,
    task_id,
    trajectory_id,
)
from reward_lens.record.compaction import summarise
from reward_lens.record.writer import ParquetUnavailable, partition_lo, partition_name

ENGINE = Engine(name="vllm", revision="0.11.2", attention_impl="flash_attn_2", dtype="bfloat16")


# ---------------------------------------------------------------------------
# The arithmetic that forces the honest default
# ---------------------------------------------------------------------------


def test_the_storage_arithmetic_in_section_2_2_1_reproduces():
    """16.00 KiB per token per layer at 70B, 16.4 TB for one layer, 1.31 PB for eighty."""
    per_token_per_layer = residual_bytes(d_model=8192)
    assert per_token_per_layer == 16384
    assert per_token_per_layer / 1024 == 16.00

    one_layer = residual_bytes(d_model=8192, tokens=10**9)
    assert one_layer == 16_384_000_000_000
    assert one_layer / 1e12 == pytest.approx(16.4, abs=0.05)

    eighty = residual_bytes(d_model=8192, tokens=10**9, layers=80)
    assert eighty / 1e15 == pytest.approx(1.31, abs=0.005)


def test_residual_bytes_rejects_a_nonsense_model():
    with pytest.raises(ValueError):
        residual_bytes(d_model=0)


# ---------------------------------------------------------------------------
# Import discipline
# ---------------------------------------------------------------------------


def test_importing_the_record_does_not_import_torch():
    """The record is core, and core installs with no compiled dependency."""
    r = subprocess.run(
        [sys.executable, "-c", "import reward_lens.record, sys; print('torch' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert r.stdout.strip() == "False", r.stdout


def test_importing_the_record_does_not_import_pyarrow_or_safetensors():
    """Both are optional backends and neither may be pulled by the import."""
    prog = (
        "import reward_lens.record, sys\n"
        "print('pyarrow' in sys.modules, 'safetensors' in sys.modules)\n"
    )
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "False False", r.stdout


# ---------------------------------------------------------------------------
# Turns
# ---------------------------------------------------------------------------


def test_a_per_token_array_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="against 3 token ids"):
        Turn(index=0, role="assistant", token_ids=(1, 2, 3), loss_mask=(True, False))


def test_a_missing_logprob_stream_gives_none_rather_than_zeros():
    turn = Turn(index=0, role="assistant", token_ids=(1, 2), logprobs_sampling=(-0.1, -0.2))
    assert turn.has_both_logprob_streams is False
    assert turn.logprob_gap() is None


def test_the_logprob_mismatch_statistic_distinguishes_agreement_from_absence():
    both = Turn(
        index=0,
        role="assistant",
        token_ids=(1, 2),
        logprobs_sampling=(-1.0, -2.0),
        logprobs_train=(-1.0, -2.0),
    )
    one = Turn(index=1, role="assistant", token_ids=(3,), logprobs_sampling=(-1.0,))
    assert logprob_mismatch([both]) == (0.0, 2)
    assert logprob_mismatch([one]) == (0.0, 0)
    assert logprob_mismatch([]) == (0.0, 0)


def test_the_mask_policy_signature_separates_masked_roles_from_unmasked_ones():
    turns = [
        Turn(index=0, role="user", token_ids=(1,), loss_mask=(False,)),
        Turn(index=1, role="assistant", token_ids=(2,), loss_mask=(True,)),
        Turn(index=2, role="tool", token_ids=(3,), loss_mask=(False,)),
    ]
    before = mask_policy_signature(turns)
    assert before == "assistant=all;tool=none;user=none"
    changed = turns[:2] + [Turn(index=2, role="tool", token_ids=(3,), loss_mask=(True,))]
    assert mask_policy_signature(changed) != before


def test_an_unrecorded_mask_is_its_own_state_not_an_unmasked_one():
    assert mask_policy_signature([Turn(index=0, role="tool")]) == "tool=unrecorded"


def test_renumber_reindexes_in_order():
    turns = renumber([Turn(index=9, role="user"), Turn(index=4, role="assistant")])
    assert [t.index for t in turns] == [0, 1]
    assert [t.role for t in turns] == ["user", "assistant"]


def test_n_unmasked_is_none_when_no_mask_was_recorded():
    assert Turn(index=0, role="assistant", token_ids=(1, 2)).n_unmasked is None
    assert (
        Turn(index=0, role="assistant", token_ids=(1, 2), loss_mask=(True, False)).n_unmasked == 1
    )


# ---------------------------------------------------------------------------
# Segment provenance
# ---------------------------------------------------------------------------


def test_a_turn_range_is_half_open():
    seg = SegmentProvenance(
        turn_range=(2, 5), policy_version="pv", staleness_steps=0, engine=ENGINE
    )
    assert seg.n_turns == 3
    assert seg.covers(2) and seg.covers(4)
    assert not seg.covers(5)


def test_an_empty_or_reversed_turn_range_is_refused():
    with pytest.raises(TilingError, match="empty or reversed"):
        SegmentProvenance(turn_range=(3, 3), policy_version="p", staleness_steps=0, engine=ENGINE)
    with pytest.raises(TilingError, match="empty or reversed"):
        SegmentProvenance(turn_range=(4, 2), policy_version="p", staleness_steps=0, engine=ENGINE)


def test_negative_staleness_is_refused():
    with pytest.raises(TilingError, match="cannot be negative"):
        SegmentProvenance(turn_range=(0, 1), policy_version="p", staleness_steps=-1, engine=ENGINE)


def test_a_trajectory_with_no_turns_tiles_with_no_segments():
    check_tiling((), 0)
    with pytest.raises(TilingError, match="nothing for them to cover"):
        check_tiling(single_segment(1, "p", engine=ENGINE), 0)


def test_single_segment_of_an_empty_trajectory_is_empty():
    assert single_segment(0, "p", engine=ENGINE) == ()


def test_the_policy_mixture_weights_by_segment_not_by_turn():
    segs = (
        SegmentProvenance(
            turn_range=(0, 100), policy_version="a", staleness_steps=0, engine=ENGINE
        ),
        SegmentProvenance(
            turn_range=(100, 101), policy_version="b", staleness_steps=40, engine=ENGINE
        ),
    )
    mix = policy_mixture(segs)
    assert mix.n_segments == 2
    assert mix.n_policy_versions == 2
    assert mix.singular is False
    assert mix.max_staleness == 40
    assert mix.mean_staleness == pytest.approx(20.0)


def test_two_engines_on_one_trajectory_are_counted():
    segs = (
        SegmentProvenance(turn_range=(0, 1), policy_version="a", staleness_steps=0, engine=ENGINE),
        SegmentProvenance(
            turn_range=(1, 2), policy_version="a", staleness_steps=0, engine=Engine(name="sglang")
        ),
    )
    mix = policy_mixture(segs)
    assert mix.singular is True
    assert mix.n_engines == 2


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


def test_a_compaction_event_must_name_its_method():
    with pytest.raises(ValueError, match="must name its method"):
        CompactionEvent(at_turn=1, tokens_before=10, tokens_after=2, method="  ")


def test_a_compaction_that_removed_nothing_is_visible_as_such():
    lossless = CompactionEvent(at_turn=1, tokens_before=8, tokens_after=8, method="rerender")
    assert lossless.is_lossless
    assert lossless.tokens_removed == 0
    assert lossless.compression_ratio == pytest.approx(1.0)
    assert crosses([lossless], first_turn=0, last_turn=5) is False


def test_a_window_pinned_to_a_compaction_boundary_does_not_cross_it():
    """Everything from the boundary onward shares one prefix, so that analysis stays valid."""
    event = CompactionEvent(at_turn=4, tokens_before=100, tokens_after=20, method="summarise")
    assert crosses([event], first_turn=4, last_turn=9) is False
    assert crosses([event], first_turn=3, last_turn=9) is True
    assert crosses([event], first_turn=0, last_turn=3) is False


def test_events_in_window_returns_them_in_turn_order():
    a = CompactionEvent(at_turn=7, tokens_before=9, tokens_after=1, method="x")
    b = CompactionEvent(at_turn=3, tokens_before=9, tokens_after=1, method="y")
    assert events_in_window([a, b], first_turn=0, last_turn=10) == (b, a)


def test_the_compaction_summary_separates_lossy_events_from_recorded_ones():
    events = [
        CompactionEvent(at_turn=2, tokens_before=10, tokens_after=3, method="summarise"),
        CompactionEvent(at_turn=5, tokens_before=4, tokens_after=4, method="rerender"),
    ]
    s = summarise(events)
    assert s.n_events == 2
    assert s.n_lossy == 1
    assert s.tokens_removed == 7
    assert s.methods == ("rerender", "summarise")


def test_two_compactions_at_the_same_turn_are_refused():
    turns = (Turn(index=0, role="assistant"), Turn(index=1, role="assistant"))
    with pytest.raises(ValueError, match="two compaction events at turn 1"):
        make_trajectory(
            id="t",
            task_ref="k",
            turns=turns,
            compaction=(
                CompactionEvent(at_turn=1, tokens_before=4, tokens_after=1, method="a"),
                CompactionEvent(at_turn=1, tokens_before=1, tokens_after=1, method="b"),
            ),
            engine=ENGINE,
        )


# ---------------------------------------------------------------------------
# Group statistics
# ---------------------------------------------------------------------------


def test_an_abstention_is_excluded_from_the_mean_rather_than_scored_zero():
    """TRL's policy generalised. verifiers substitutes 0.0 from a bare except and this does not."""
    stats = GroupStats.from_scores([1.0, 1.0, None], std_epsilon=1e-8)
    assert stats.k == 3
    assert stats.n_abstained == 1
    assert stats.mean == pytest.approx(1.0)
    assert stats.std == pytest.approx(0.0)


def test_a_group_where_every_grader_abstained_has_no_mean_at_all():
    stats = GroupStats.from_scores([None, None], std_epsilon=1e-8)
    assert stats.mean is None and stats.std is None
    assert stats.degenerate is True
    assert stats.n_abstained == 2


def test_the_degenerate_flag_uses_the_estimators_own_epsilon():
    scores = [1.0, 1.0 + 1e-9]
    assert GroupStats.from_scores(scores, std_epsilon=1e-8).degenerate is True
    assert GroupStats.from_scores(scores, std_epsilon=1e-12).degenerate is False


def test_all_fail_stays_false_when_the_failure_score_is_unknown():
    assert GroupStats.from_scores([0.0, 0.0], std_epsilon=1e-8).all_fail is False
    assert GroupStats.from_scores([0.0, 0.0], std_epsilon=1e-8, failure_at=0.0).all_fail is True


def test_ranks_are_recorded_highest_first():
    stats = GroupStats.from_scores([0.1, 0.9, 0.5], std_epsilon=1e-8)
    assert stats.ranks == (2, 0, 1)


def test_a_nan_group_mean_survives_the_json_round_trip():
    stats = GroupStats(k=2, mean=float("nan"), std=float("inf"))
    back = GroupStats.from_canonical(json.loads(json.dumps(stats.__canonical__())))
    assert np.isnan(back.mean)
    assert np.isposinf(back.std)


def test_z_scored_needs_both_centring_and_normalisation():
    assert EstimatorSpec(group_centred=True, std_normalised=True).z_scored is True
    assert EstimatorSpec(group_centred=True, std_normalised=False).z_scored is False


# ---------------------------------------------------------------------------
# The regime declaration
# ---------------------------------------------------------------------------


def test_a_declaration_contradicted_by_the_measurement_is_a_disagreement():
    decl = RegimeDeclaration(declared={RegimeCondition.STATIONARY_GRADER: True})
    reading = RegimeReading.of(STATIONARY_GRADER=False)
    assert decl.disagreements(reading) == {RegimeCondition.STATIONARY_GRADER: (True, False)}


def test_an_undeterminable_measurement_disagrees_with_any_declaration():
    decl = RegimeDeclaration(declared={RegimeCondition.NO_COMPACTION: True})
    assert decl.disagreements(RegimeReading.of(NO_COMPACTION=None)) == {
        RegimeCondition.NO_COMPACTION: (True, None)
    }
    assert decl.disagreements(None) == {RegimeCondition.NO_COMPACTION: (True, None)}


def test_a_declaration_the_measurement_confirms_is_not_a_disagreement():
    decl = RegimeDeclaration(declared={RegimeCondition.MASK_STABLE: True})
    assert decl.disagreements(RegimeReading.of(MASK_STABLE=True)) == {}


# ---------------------------------------------------------------------------
# The record sampling policy
# ---------------------------------------------------------------------------


def test_a_zero_recording_rate_is_not_a_policy():
    with pytest.raises(ValueError, match="absent record"):
        RecordSamplingPolicy(scheme=SamplingScheme.UNIFORM, rate=0.0)


def test_a_stratified_policy_without_rates_is_refused_at_construction():
    with pytest.raises(ValueError, match="needs strata_rates"):
        RecordSamplingPolicy(scheme=SamplingScheme.STRATIFIED)


def test_a_selective_policy_must_say_what_selected_the_units():
    with pytest.raises(ValueError, match="must say what selected"):
        RecordSamplingPolicy(scheme=SamplingScheme.SELECTIVE)


def test_correctability_is_a_property_of_the_design():
    assert RecordSamplingPolicy().is_correctable is True
    assert RecordSamplingPolicy(scheme=SamplingScheme.UNIFORM, rate=0.1).is_correctable is True
    assert (
        RecordSamplingPolicy(scheme=SamplingScheme.SELECTIVE, selected_by="top-k").is_correctable
        is False
    )


def test_the_horvitz_thompson_total_recovers_a_known_population_total():
    policy = RecordSamplingPolicy(scheme=SamplingScheme.UNIFORM, rate=0.25)
    got = policy.estimate_total([2.0] * 25)
    assert not isinstance(got, Refusal)
    assert got.value == pytest.approx(200.0)
    assert got.method == "horvitz_thompson"


def test_a_full_capture_correction_is_the_identity():
    got = RecordSamplingPolicy().estimate_mean([1.0, 2.0, 3.0])
    assert not isinstance(got, Refusal)
    assert got.value == pytest.approx(2.0)
    assert got.method == "identity"
    assert got.ess == pytest.approx(3.0)
    assert got.design_effect == pytest.approx(1.0)


def test_correcting_an_empty_sample_refuses_rather_than_dividing_by_zero():
    got = RecordSamplingPolicy().estimate_mean([])
    assert isinstance(got, Refusal)
    assert "nothing to correct" in got.detail


def test_the_observed_fraction_is_none_when_the_population_was_not_counted():
    assert RecordSamplingPolicy(recorded=10).observed_fraction is None
    assert RecordSamplingPolicy(population=40, recorded=10).observed_fraction == pytest.approx(0.25)


def test_explicit_inclusion_probabilities_are_used_per_unit():
    policy = RecordSamplingPolicy(scheme=SamplingScheme.EXPLICIT)
    got = policy.estimate_mean([1.0, 0.0], inclusion=[0.5, 0.5])
    assert not isinstance(got, Refusal)
    assert got.value == pytest.approx(0.5)
    missing = policy.estimate_mean([1.0], inclusion=None)
    assert isinstance(missing, Refusal)


# ---------------------------------------------------------------------------
# The tensor store
# ---------------------------------------------------------------------------


def test_the_store_is_content_addressed_and_deduplicates(tmp_path):
    store = TensorStore(tmp_path)
    a = store.put(np.arange(10, dtype=np.float32), name="resid")
    b = store.put(np.arange(10, dtype=np.float32), name="resid")
    c = store.put(np.arange(10, dtype=np.float64), name="resid")
    assert a.digest == b.digest
    assert a.digest != c.digest, "dtype is part of the content"
    assert len(list((tmp_path / "shards").glob("*.npy"))) == 2


def test_a_stored_array_reads_back_exactly(tmp_path):
    store = TensorStore(tmp_path)
    arr = np.random.default_rng(0).normal(size=(4, 3))
    ref = store.put(arr, name="x")
    store.flush()
    assert np.array_equal(TensorStore(tmp_path).get(ref), arr)


def test_the_manifest_survives_a_reopen(tmp_path):
    store = TensorStore(tmp_path)
    ref = store.put(np.ones(4), name="ones")
    store.flush()
    reopened = TensorStore(tmp_path)
    assert ref.digest in reopened.entries
    assert reopened.total_bytes() == ref.nbytes


def test_a_missing_shard_is_an_absence_and_not_an_exception(tmp_path):
    store = TensorStore(tmp_path)
    ref = store.put(np.ones(4), name="ones")
    (tmp_path / "shards" / ref.shard).unlink()
    got = store.get(ref)
    assert isinstance(got, AbsentRef)
    assert got.reason is AbsenceReason.SHARD_MISSING


def test_resolving_a_stored_ref_with_no_store_is_an_absence():
    ref = StoredRef(digest="tns:x", shard="x.npy", key="k", dtype="float32", shape=(1,))
    got = resolve(ref)
    assert isinstance(got, AbsentRef)
    assert got.reason is AbsenceReason.SHARD_MISSING


@pytest.mark.skipif(
    importlib.util.find_spec("safetensors") is None, reason="the safetensors container is optional"
)
def test_the_safetensors_container_round_trips(tmp_path):
    store = TensorStore(tmp_path, container="safetensors")
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    ref = store.put(arr, name="resid")
    store.flush()
    assert ref.container == "safetensors"
    assert ref.shard.endswith(".safetensors")
    assert np.array_equal(TensorStore(tmp_path).get(ref), arr)


def test_an_absent_ref_without_a_remedy_cannot_be_constructed():
    with pytest.raises(ValueError, match="carries no remedy"):
        AbsentRef(reason=AbsenceReason.NOT_CAPTURED)


def test_a_capture_spec_rejects_a_stride_below_one():
    with pytest.raises(ValueError, match="stride"):
        CaptureSpec(stride=0)


def test_a_capture_spec_round_trips_its_sites():
    spec = CaptureSpec(sites=(Site(layer=12, point="resid_post"), Site(layer=3, point="mlp_out")))
    back = CaptureSpec.from_canonical(json.loads(json.dumps(spec.__canonical__())))
    assert back.sites == spec.sites


def test_engine_identity_is_content_derived_and_stable():
    a = Engine(name="vllm", revision="0.11.2", attention_impl="flash_attn_2", dtype="bfloat16")
    b = Engine(name="vllm", revision="0.11.2", attention_impl="flash_attn_2", dtype="bfloat16")
    c = Engine(name="vllm", revision="0.11.2", attention_impl="flash_attn_2", dtype="float16")
    assert a.id == b.id
    assert a.id != c.id, "dtype is part of the instrument"
    assert a.id.startswith("eng:")


def test_compiled_and_eager_are_different_engines():
    eager = Engine(name="vllm", compiled=False)
    compiled = Engine(name="vllm", compiled=True)
    assert eager.id != compiled.id


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------


def test_content_derived_ids_carry_a_readable_prefix():
    assert run_id(name="x", seed=1).startswith("run:")
    assert group_id(run="r", step=0, task="t").startswith("grp:")
    assert trajectory_id(group="g", ordinal=0).startswith("traj:")
    assert task_id(dataset="d", index=3).startswith("task:")


def test_the_same_material_gives_the_same_run_id():
    assert run_id(name="x", seed=1) == run_id(name="x", seed=1)
    assert run_id(name="x", seed=1) != run_id(name="x", seed=2)


# ---------------------------------------------------------------------------
# The writer and the reader
# ---------------------------------------------------------------------------


def _tiny_run(run: str = "run:tiny", n: int = 12) -> Run:
    steps = []
    for i in range(n):
        traj = make_trajectory(
            id=f"traj:{i}",
            task_ref="task:0",
            turns=(Turn(index=0, role="assistant", token_ids=(1, 2)),),
            scores={"total": float(i)},
            advantage=float(i),
            engine=ENGINE,
        )
        steps.append(
            Step(
                index=i,
                groups=(
                    Group(
                        id=GroupID(f"grp:{i}"),
                        task_ref=TaskID("task:0"),
                        trajectories=(traj,),
                        estimator=EstimatorSpec(family="rloo"),
                        group_stats=GroupStats(k=1),
                    ),
                ),
                schedule={"lr": 1e-6},
                optimizer=OptimizerTelemetry(staleness=StalenessDistribution({0: 1})),
            )
        )
    return Run(
        id=RunID(run),
        kind="bon",
        components={Component.GRADER: ComponentRef(name="g")},
        access={Component.GRADER: Access.RECORD},
        regime=RegimeDeclaration(),
        steps=InMemoryStepStream(steps),
    )


def test_the_partition_layout_is_hive_style_and_step_aligned():
    assert partition_name(200, 100) == "step_range=000200-000299"
    assert partition_lo(275, 100) == 200
    assert partition_lo(99, 100) == 0


def test_a_writer_reports_what_it_wrote(tmp_path):
    report = RecordWriter(tmp_path, chunk_steps=5).write(_tiny_run())
    assert report.steps == 12
    assert report.turns == 12
    assert report.partitions == 3
    assert report.fmt == "jsonl"


def test_the_last_written_step_is_readable_from_the_manifest(tmp_path):
    assert RecordWriter.last_step_written(tmp_path, "run:tiny") is None
    RecordWriter(tmp_path, chunk_steps=5).write(_tiny_run())
    assert RecordWriter.last_step_written(tmp_path, "run:tiny") == 11


def test_opening_a_directory_with_no_manifest_names_the_problem(tmp_path):
    (tmp_path / "runs" / "run_missing").mkdir(parents=True)
    with pytest.raises(RecordNotFound, match="never called close"):
        RecordReader(tmp_path, "run:missing")


def test_a_record_can_be_opened_by_its_own_directory(tmp_path):
    report = RecordWriter(tmp_path, chunk_steps=5).write(_tiny_run())
    reader = RecordReader.at(report.root)
    assert reader.run().n_steps == 12


def test_asking_for_a_step_that_is_not_there_is_a_keyerror(tmp_path):
    RecordWriter(tmp_path, chunk_steps=5).write(_tiny_run())
    with pytest.raises(KeyError, match="step 99"):
        RecordReader(tmp_path, "run:tiny").step(99)


def test_a_slice_outside_the_run_yields_nothing_and_opens_nothing(tmp_path):
    RecordWriter(tmp_path, chunk_steps=5).write(_tiny_run())
    reader = RecordReader(tmp_path, "run:tiny")
    assert list(reader.run().steps.slice(50, 60)) == []
    assert reader.stats.files_opened == []


def test_a_run_id_with_a_colon_becomes_a_usable_directory_name(tmp_path):
    report = RecordWriter(tmp_path).write(_tiny_run("run:ab:cd", n=1))
    assert report.root.name == "run_ab_cd"
    assert RecordReader(tmp_path, "run:ab:cd").run().id == "run:ab:cd"


def test_the_writer_holds_one_partition_at_a_time(tmp_path):
    """Peak memory is bounded by chunk_steps, so a 40,000-step run does not need 40,000 in hand."""
    writer = RecordWriter(tmp_path, chunk_steps=5)
    run = _tiny_run(n=12)
    writer.open_run(run)
    for step in run.steps:
        writer.append_step(step)
        assert len(writer._buffer) == 1
    writer.close()


def test_chunk_steps_below_one_is_refused(tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        RecordWriter(tmp_path, chunk_steps=0)


@pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is not None, reason="pyarrow is installed here"
)
def test_asking_for_parquet_without_pyarrow_says_so(tmp_path):
    with pytest.raises(ParquetUnavailable, match="no compiled dependency"):
        RecordWriter(tmp_path, fmt="parquet")


@pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="the parquet backend")
def test_parquet_and_jsonl_read_back_the_same_record(tmp_path):
    """Two containers, one logical table. A record must not mean different things in each."""
    got = {}
    for fmt in ("jsonl", "parquet"):
        root = tmp_path / fmt
        RecordWriter(root, fmt=fmt, chunk_steps=5).write(_tiny_run())
        reader = RecordReader(root, "run:tiny")
        assert reader.fmt == fmt
        steps = list(reader.run().steps.slice(5, 8))
        got[fmt] = [
            (
                s.index,
                s.schedule,
                s.groups[0].estimator.family,
                s.groups[0].trajectories[0].scores,
                s.groups[0].trajectories[0].advantage,
                s.groups[0].trajectories[0].turns[0].token_ids,
                s.optimizer.staleness.counts,
            )
            for s in steps
        ]
    assert got["jsonl"] == got["parquet"]


@pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="the parquet backend")
def test_the_parquet_reader_also_opens_only_the_partitions_it_needs(tmp_path):
    RecordWriter(tmp_path, fmt="parquet", chunk_steps=5).write(_tiny_run())
    reader = RecordReader(tmp_path, "run:tiny")
    list(reader.run().steps.slice(5, 7))
    assert reader.stats.partitions_opened == ("step_range=000005-000009",)
    assert reader.stats.steps_materialised == 2


def test_a_torn_terminator_on_a_table_is_healed_before_the_next_append(tmp_path):
    """The evidence store's newline defect, in the record's tables.

    A writer killed after a row's bytes and before its newline leaves a line that parses. Appending
    onto it would weld two rows into one unparseable line, which poisons every later read of that
    partition.
    """
    writer = RecordWriter(tmp_path, chunk_steps=2)
    run = _tiny_run(n=4)
    writer.open_run(run)
    steps = list(run.steps)
    writer.append_step(steps[0])
    writer.append_step(steps[1])
    writer.append_step(steps[2])  # flushes partition 0
    table = writer._dir / "steps" / "step_range=000000-000001" / "steps.jsonl"
    with table.open("rb+") as fh:
        fh.seek(-1, 2)
        fh.truncate()
    writer._buffer.setdefault(0, {t: [] for t in ("steps", "groups", "trajectories", "turns")})
    writer._buffer[0]["steps"].append({"run_id": "run:tiny", "step_index": 1, "payload": "{}"})
    writer._flush_partition(0)
    lines = table.read_bytes().splitlines()
    for line in lines:
        json.loads(line)
    assert len(lines) == 3

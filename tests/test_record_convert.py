"""Unit and property tests for `reward_lens.record.convert`.

Everything here runs against synthetic evidence stores built in a temp directory. The real
campaign store is a $17.73 archive and the acceptance tests are the only things that touch it, so
these tests are the ones that run in a second and catch the arithmetic.

The synthetic store is written as raw envelopes rather than through `EvidenceStore.append`, because
the payloads have to carry a ``__type__`` no process here can import. That is the situation the
converter exists for, and building the fixture any other way would test a store the campaign never
wrote.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from reward_lens.core.envelope import ConditionReading, RegimeCondition, RegimeReading
from reward_lens.core.evidence import Evidence
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import Access, Capability, Component, Phase, Substrate
from reward_lens.record.convert import (
    CampaignStore,
    capabilities_in_record,
    convert_campaign,
    metric_arcs_from_reason,
    reader_access,
    regime_over,
    run_instrument,
    sweep,
)
from reward_lens.record.convert.campaign import (
    NO_ESTIMATOR,
    UNRECORDED_POLICY,
    campaign_arms,
    count_run,
)
from reward_lens.record.convert.instruments import context_for
from reward_lens.record.reader import RecordReader
from reward_lens.record.tensors import AbsentRef
from reward_lens.record.writer import RecordWriter

SCORE_TYPE = "campaign.payloads.ScoreBank"
PRM_TYPE = "campaign.payloads.StepScores"
CAPTURE_TYPE = "campaign.payloads.CaptureManifest"

FINITE = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6, width=32)


# ---------------------------------------------------------------------------
# The synthetic store
# ---------------------------------------------------------------------------


def _array(value: np.ndarray, payloads: Path, name: str) -> dict[str, Any]:
    """Write an array as a sidecar and return the codec's reference to it."""
    payloads.mkdir(parents=True, exist_ok=True)
    with open(payloads / f"{name}.npy", "wb") as fh:
        np.save(fh, value)
    return {
        "__ndarray__": {
            "sidecar": f"{name}.npy",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    }


def _envelope(
    ident: str,
    observable: str,
    value: Any,
    *,
    roster: str = "rm-a",
    slice_name: str = "bank-a",
    created_at: str = "2026-07-19T00:00:00+00:00",
    arc: str = "model:rm-a",
) -> dict[str, Any]:
    return {
        "id": f"ev:{ident}",
        "observable": observable,
        "observable_version": "1.0",
        "subject": {
            "signals": [roster],
            "dataset": None,
            "readout": None,
            "frame": None,
            "interventions": [],
            "extra": {"roster_key": roster, "slice": slice_name},
        },
        "value": value,
        "uncertainty": {
            "ci_low": None,
            "ci_high": None,
            "ci_level": None,
            "n": None,
            "n_effective": None,
            "seed_spread": None,
            "method": "none",
        },
        "gauge": "invariant",
        "calibration": None,
        "trust": 0,
        "provenance": {
            "git_sha": "abc123",
            "config_hash": None,
            "seeds": [],
            "cost": {"gpu_seconds": 0.0, "tokens": 0, "wall_seconds": 0.0},
            "oracle_calls": [],
            "parents": [],
            "study": None,
            "extra": {"gpu": "cpu", "arc": arc},
        },
        "created_at": created_at,
    }


def score_bank(
    ident: str,
    scores: np.ndarray,
    payloads: Path,
    *,
    layout: str = "flat",
    roster: str = "rm-a",
    slice_name: str = "bank-a",
    created_at: str = "2026-07-19T00:00:00+00:00",
) -> dict[str, Any]:
    n = scores.shape[0]
    items = [f"{slice_name.split('::')[0]}::row{i}" for i in range(n)]
    return _envelope(
        ident,
        "campaign.scores",
        {
            "__type__": SCORE_TYPE,
            "fields": {
                "item_ids": {"__seq__": items},
                "scores": _array(scores, payloads, ident),
                "layout": layout,
                "meta": {"__map__": {}},
            },
        },
        roster=roster,
        slice_name=slice_name,
        created_at=created_at,
    )


def _leaf_field(leaf: Any, name: str) -> Any:
    """Read a field off a `ScoreTree` leaf or off the mapping used when the score tree is absent."""
    return leaf[name] if isinstance(leaf, dict) else getattr(leaf, name)


def write_store(root: Path, envelopes: Sequence[dict[str, Any]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "payloads").mkdir(exist_ok=True)
    with (root / "evidence.jsonl").open("w", encoding="utf-8") as fh:
        for env in envelopes:
            fh.write(json.dumps(env, ensure_ascii=False) + "\n")
    return root


@pytest.fixture()
def simple_store(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    payloads = root / "payloads"
    flat = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    quads = np.asarray([[0.0, 1.0, 2.0, 3.0], [4.0, 4.0, 4.0, 4.0]], dtype=np.float32)
    return write_store(
        root,
        [
            score_bank("a", flat, payloads, layout="flat", slice_name="bank-a"),
            score_bank(
                "b",
                quads,
                payloads,
                layout="best-of-4",
                slice_name="bank-b",
                created_at="2026-07-19T01:00:00+00:00",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Reading the store
# ---------------------------------------------------------------------------


def test_the_store_reads_payloads_the_kernel_codec_refuses(simple_store: Path) -> None:
    """The situation this package exists for: `EvidenceStore.find` cannot read this store.

    The campaign's payload classes are in a package reward-lens does not depend on, so the strict
    codec raises. That is the right default and it is why `CampaignStore` has its own decoder.

    The refusal is asserted only when `campaign.payloads` is genuinely absent, and that condition is
    checked rather than assumed. `_PAYLOAD_REGISTRY` is a process-global that nothing cleans and the
    codec imports a missing module before giving up, so one earlier test that puts the campaign
    worktree on `sys.path`, which the closure regression does, makes the class resolvable for the
    rest of the session and the strict codec then reads the store correctly. Asserting the raise
    unconditionally pins the environment rather than the code, and it failed in the full suite while
    passing alone. What matters either way is the second half: `CampaignStore` returns the same
    field mapping whichever happened.
    """
    import importlib.util

    from reward_lens.core.evidence import PayloadTypeUnregistered

    def _campaign_importable() -> bool:
        # find_spec raises rather than returning None when a parent package is itself absent,
        # which is the ordinary case here, so the exception is the answer and not an error.
        try:
            return importlib.util.find_spec("campaign.payloads") is not None
        except (ImportError, ValueError):
            return False

    kernel = EvidenceStore(simple_store, readonly=True)
    if not _campaign_importable():
        with pytest.raises(PayloadTypeUnregistered):
            kernel.find(observable="campaign.scores")

    campaign = CampaignStore(simple_store)
    assert len(campaign) == 2
    assert campaign.payload_types() == {SCORE_TYPE: 2}
    value = campaign.value(campaign.by_observable("campaign.scores")[0])
    assert value["layout"] == "flat"
    assert np.asarray(value["scores"]).tolist() == [1.0, 2.0, 3.0]


def test_a_blinded_payload_stops_permissive_decoding(tmp_path: Path) -> None:
    """Permissive decoding is licensed by a census, and the census has teeth.

    A `Blind` that decodes to a field dict hands over the label the type exists to withhold, so a
    store carrying one must not be read this way at all.
    """
    root = tmp_path / "leaky"
    payloads = root / "payloads"
    env = score_bank("a", np.asarray([1.0], dtype=np.float32), payloads)
    env["value"]["fields"]["meta"] = {
        "__type__": "reward_lens.record.labels.Blind",
        "fields": {"_value": 1, "key": "is_hack", "quality": None},
    }
    write_store(root, [env])

    store = CampaignStore(root)
    assert "reward_lens.record.labels.Blind" in store.payload_types()
    with pytest.raises(ValueError, match="strict=False"):
        store.assert_no_blind_payloads()


def test_sidecars_are_counted_rather_than_assumed(simple_store: Path) -> None:
    report = CampaignStore(simple_store).sidecar_report()
    assert report == {
        "referenced": 2,
        "in_primary": 2,
        "via_search_path": 0,
        "unresolved": 0,
        "orphaned_in_primary": 0,
    }


def test_a_sidecar_only_the_second_store_holds_is_still_found(tmp_path: Path) -> None:
    """E3: two stores, and the arrays the first lacks live in the second."""
    root = tmp_path / "primary"
    other = tmp_path / "other" / "payloads"
    payloads = root / "payloads"
    env = score_bank("moved", np.asarray([7.0, 8.0], dtype=np.float32), payloads)
    write_store(root, [env])
    other.mkdir(parents=True)
    (payloads / "moved.npy").rename(other / "moved.npy")

    store = CampaignStore(root, sidecar_dirs=[other])
    assert store.sidecar_report()["via_search_path"] == 1
    value = store.value(store.rows[0])
    assert np.asarray(value["scores"]).tolist() == [7.0, 8.0]


def test_the_converter_leaves_the_store_byte_identical(simple_store: Path) -> None:
    """The archive is read-only, and read-only is checked rather than intended."""
    before = {
        p: (p.stat().st_size, p.read_bytes())
        for p in sorted(simple_store.rglob("*"))
        if p.is_file()
    }
    run, report = convert_campaign(simple_store)
    count_run(run, report)
    after = {
        p: (p.stat().st_size, p.read_bytes())
        for p in sorted(simple_store.rglob("*"))
        if p.is_file()
    }
    assert before == after

    store = CampaignStore(simple_store)
    assert store.store.readonly is True
    with pytest.raises(RuntimeError, match="readonly"):
        store.store.append(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The shape of the converted run
# ---------------------------------------------------------------------------


def test_the_run_is_an_eval_with_no_optimizer_telemetry(simple_store: Path) -> None:
    run, report = convert_campaign(simple_store)
    assert run.kind == "eval"
    assert run.n_steps == 2
    assert report.store_rows == 2

    step = next(iter(run.steps))
    # An evaluation ran no optimizer, so every telemetry field is None rather than zero. Zero
    # would say the gradient norm was measured and found to be nothing.
    assert step.optimizer.grad_norm_clipped is None
    assert step.optimizer.grad_norm_unclipped is None
    assert step.optimizer.kl_to_ref is None
    assert step.schedule == {}
    assert step.groups[0].estimator == NO_ESTIMATOR


def test_ids_join_back_to_the_store_they_came_from(simple_store: Path) -> None:
    """A content hash here would make a group impossible to look up in the source store."""
    run, _ = convert_campaign(simple_store)
    step = next(iter(run.steps))
    group = step.groups[0]
    assert str(group.id) == "rm-a/bank-a/bank-a::row0"
    assert str(group.task_ref) == "bank-a::row0"
    assert str(group.trajectories[0].id) == "rm-a/bank-a/bank-a::row0#0"


def test_the_stream_decodes_only_the_steps_the_slice_covers(simple_store: Path) -> None:
    """Laziness is the reason `Run.steps` is not a tuple, so it is asserted rather than described."""
    run, _ = convert_campaign(simple_store)
    picked = list(run.steps.slice(1, 2))
    assert [s.index for s in picked] == [1]
    assert len(picked[0].groups) == 2  # bank-b has two items
    assert run.steps.indices == (0, 1)
    assert len(run.steps) == 2


def test_group_size_follows_the_layout(simple_store: Path) -> None:
    run, _ = convert_campaign(simple_store)
    flat, quads = list(run.steps)
    assert all(g.k == 1 for g in flat.groups)
    assert all(g.k == 4 for g in quads.groups)
    assert quads.groups[0].group_stats.k == 4
    # The second row is four identical scores, so it is degenerate at any epsilon.
    assert quads.groups[1].group_stats.degenerate is True
    assert quads.groups[0].group_stats.degenerate is False


def test_a_bank_response_is_one_turn_with_its_text_unrecorded(simple_store: Path) -> None:
    """Zero turns would say the rollout was empty, which is false."""
    run, _ = convert_campaign(simple_store)
    traj = next(iter(run.steps)).groups[0].trajectories[0]
    assert traj.n_turns == 1
    assert traj.turns[0].role == "assistant"
    assert traj.turns[0].text == ""
    assert traj.turns[0].token_ids is None
    assert traj.n_tokens == 0
    assert [str(s.policy_version) for s in traj.provenance] == [UNRECORDED_POLICY]


def test_arms_are_one_run_per_grader_cross_linked(tmp_path: Path) -> None:
    """The record format's own answer to thirteen graders on shared banks."""
    root = tmp_path / "fleet"
    payloads = root / "payloads"
    write_store(
        root,
        [
            score_bank("a", np.asarray([1.0, 2.0], dtype=np.float32), payloads, roster="rm-a"),
            score_bank(
                "b",
                np.asarray([3.0, 4.0], dtype=np.float32),
                payloads,
                roster="rm-b",
                created_at="2026-07-19T02:00:00+00:00",
            ),
        ],
    )
    runs = campaign_arms(root)
    assert sorted(runs) == ["rm-a", "rm-b"]
    assert runs["rm-a"].arms == (runs["rm-b"].id,)
    assert runs["rm-b"].arms == (runs["rm-a"].id,)
    # An arm is one grader, so its substrate is a model rather than a fleet.
    assert runs["rm-a"].component(Component.GRADER).substrate is Substrate.NEURAL_SCALAR
    fleet, _ = convert_campaign(root)
    assert fleet.component(Component.GRADER).substrate is Substrate.COMPOSITE


def test_capture_manifests_become_absent_refs_not_missing_keys(tmp_path: Path) -> None:
    """The record travelled and the tensors did not, which is what SHARD_MISSING means."""
    root = tmp_path / "captured"
    payloads = root / "payloads"
    manifest = _envelope(
        "cap",
        "campaign.capture",
        {
            "__type__": CAPTURE_TYPE,
            "fields": {
                "store_key": "k",
                "roster_key": "rm-a",
                "slice_name": "bank-a",
                "sites": {"__seq__": ["L0.resid_post", "L1.resid_post"]},
                "position": "final",
                "dtype": "float16",
                "n_items": 2,
                "bytes_total": 4096,
                "content_hash": "cap:deadbeef",
                "meta": {"__map__": {"activation_root": "/store/captures"}},
            },
        },
    )
    write_store(
        root,
        [manifest, score_bank("a", np.asarray([1.0, 2.0], dtype=np.float32), payloads)],
    )
    run, report = convert_campaign(root)
    assert report.capture_manifests == 1
    assert report.absent_tensors == 2

    traj = next(iter(run.steps)).groups[0].trajectories[0]
    assert traj.capture is not None
    assert [s.layer for s in traj.capture.spec.sites] == [0, 1]
    for ref in traj.capture.tensors.values():
        assert isinstance(ref, AbsentRef)
        assert ref.reason.name == "SHARD_MISSING"
        assert "/store/captures" in ref.detail
        assert ref.remedy


def test_the_report_names_every_observable_the_record_has_no_level_for(tmp_path: Path) -> None:
    root = tmp_path / "mixed"
    payloads = root / "payloads"
    write_store(
        root,
        [
            score_bank("a", np.asarray([1.0], dtype=np.float32), payloads),
            _envelope("t", "campaign.index.table", {"__map__": {"rows": 3}}),
        ],
    )
    _, report = convert_campaign(root)
    assert report.unrepresented == {"campaign.index.table": 1}
    assert report.unrepresented_rows == 1
    assert any("no level for a measurement made about a bank" in f for f in report.findings)


# ---------------------------------------------------------------------------
# Round trip through the writer and the reader
# ---------------------------------------------------------------------------


def test_a_converted_run_round_trips_through_the_record_writer(
    simple_store: Path, tmp_path: Path
) -> None:
    """Convert, write, read back, and compare. The writer is the record's own format."""
    run, _ = convert_campaign(simple_store)
    original = list(run.steps)

    # bank-a is three items at K=1 and bank-b is two at K=4, so 5 groups and 11 trajectories.
    written = RecordWriter(tmp_path / "records", chunk_steps=1).write(run, original)
    assert (written.steps, written.groups, written.trajectories, written.turns) == (2, 5, 11, 11)

    back = RecordReader.at(written.root).run()
    assert back.kind == run.kind
    assert str(back.id) == str(run.id)
    assert back.access == run.access
    assert back.sampling_policy.scheme == run.sampling_policy.scheme
    assert back.component(Component.GRADER).name == run.component(Component.GRADER).name

    reread = list(back.steps)
    assert [s.index for s in reread] == [s.index for s in original]
    for before, after in zip(original, reread):
        assert [str(g.id) for g in before.groups] == [str(g.id) for g in after.groups]
        assert [g.group_stats.__canonical__() for g in before.groups] == [
            g.group_stats.__canonical__() for g in after.groups
        ]
        assert before.regime_measured.conditions.keys() == after.regime_measured.conditions.keys()
        for condition, reading in before.regime_measured.conditions.items():
            assert after.regime_measured.conditions[condition].holds == reading.holds
        for gb, ga in zip(before.groups, after.groups):
            assert [str(t.id) for t in gb.trajectories] == [str(t.id) for t in ga.trajectories]
            assert [t.n_turns for t in gb.trajectories] == [t.n_turns for t in ga.trajectories]


# ---------------------------------------------------------------------------
# The regime reading
# ---------------------------------------------------------------------------


def test_the_regime_reports_what_is_known_and_stays_silent_on_the_rest(simple_store: Path) -> None:
    run, _ = convert_campaign(simple_store)
    flat, quads = list(run.steps)

    # A K=1 bank has no group structure, which is a failure at any threshold.
    assert flat.regime_measured.holds(RegimeCondition.GROUP_NONDEGENERATE) is False
    # bank-b's first row has spread and its second does not, so the fraction is 0.5 and the
    # verdict is nobody's to give here.
    assert quads.regime_measured.holds(RegimeCondition.GROUP_NONDEGENERATE) is None
    assert quads.regime_measured.conditions[RegimeCondition.GROUP_NONDEGENERATE].statistic == 0.5

    # Nobody recorded which policy wrote the bank responses, so this is unknown and not a pass.
    assert flat.regime_measured.holds(RegimeCondition.NEAR_POLICY) is None
    assert flat.regime_measured.holds(RegimeCondition.NO_COMPACTION) is True
    assert flat.regime_measured.holds(RegimeCondition.STATIONARY_GRADER) is True

    # The six conditions nobody measured are absent rather than present-and-None, because
    # `EnvelopeSpec.classify` tells those apart and the difference is the point.
    for condition in (
        RegimeCondition.QUASI_STATIC,
        RegimeCondition.LINEAR_RESPONSE,
        RegimeCondition.ABOVE_LOD,
        RegimeCondition.ESS_ADEQUATE,
        RegimeCondition.LIGHT_TAILED,
        RegimeCondition.SCALAR_REPRESENTABLE,
    ):
        assert condition not in flat.regime_measured.conditions


def test_a_bounded_fold_says_how_bounded_it_is(simple_store: Path) -> None:
    run, _ = convert_campaign(simple_store)
    reading, folded, total = regime_over(run, limit=1)
    assert (folded, total) == (1, 2)
    detail = reading.conditions[RegimeCondition.NO_COMPACTION].detail
    assert "folded over 1 of 2 steps" in detail

    whole, folded, total = regime_over(run)
    assert (folded, total) == (2, 2)
    assert "folded over" not in whole.conditions[RegimeCondition.NO_COMPACTION].detail


# ---------------------------------------------------------------------------
# The instrument harness
# ---------------------------------------------------------------------------


def test_the_reader_matrix_is_not_the_recorder_matrix(tmp_path: Path) -> None:
    """`Run.access` is what the campaign could do; the reader can only read."""
    root = tmp_path / "acc"
    payloads = root / "payloads"
    write_store(
        root,
        [
            score_bank("a", np.asarray([1.0], dtype=np.float32), payloads),
            _envelope(
                "cap",
                "campaign.capture",
                {
                    "__type__": CAPTURE_TYPE,
                    "fields": {
                        "store_key": "k",
                        "roster_key": "rm-a",
                        "slice_name": "bank-a",
                        "sites": {"__seq__": ["L0.resid_post"]},
                        "position": "final",
                        "dtype": "float16",
                        "n_items": 1,
                        "bytes_total": 8,
                        "content_hash": "cap:1",
                        "meta": {"__map__": {}},
                    },
                },
            ),
        ],
    )
    run, _ = convert_campaign(root)
    assert run.access[Component.GRADER] & Access.FORWARD
    assert run.access[Component.GRADER] & Access.QUERY
    assert reader_access(run)[Component.GRADER] == Access.RECORD


def test_capabilities_are_measured_from_the_record(simple_store: Path) -> None:
    run, _ = convert_campaign(simple_store)
    caps, scanned, total = capabilities_in_record(run)
    assert caps & Capability.SCORES
    # Nothing captured, nothing spanned, no per-step score in a plain score bank.
    assert not (caps & Capability.ACTIVATIONS)
    assert not (caps & Capability.SPAN_TYPES)
    assert not (caps & Capability.STEP_SCORES)
    assert (scanned, total) == (2, 2)


def test_a_capability_a_record_cannot_hold_is_a_refusal_and_never_a_raise(
    simple_store: Path,
) -> None:
    """`measure.base.run` raises `CapabilityError` here; the harness refuses instead."""
    from reward_lens.measure.indices.contested import Contested

    run, _ = convert_campaign(simple_store)
    ctx, _, _, _ = context_for(run, caps=Capability.SCORES)
    outcome = run_instrument(Contested(), ctx, caps=Capability.SCORES, window=(2, 2))

    assert outcome.kind == "refusal"
    assert outcome.reading.reason is RefusalReason.RECORD_INCOMPLETE
    assert "ACTIVATIONS" in outcome.reading.detail
    assert outcome.reading.remedy
    assert outcome.reading.statistics["steps_scanned"] == 2


def test_a_bounded_capability_scan_says_so_in_the_refusal(simple_store: Path) -> None:
    from reward_lens.measure.indices.contested import Contested

    run, _ = convert_campaign(simple_store)
    ctx, _, _, _ = context_for(run, caps=Capability.SCORES)
    outcome = run_instrument(Contested(), ctx, caps=Capability.SCORES, window=(1, 2))
    assert "lower bound" in outcome.reading.detail


def test_the_harness_can_also_say_yes(simple_store: Path) -> None:
    """A gate that can only refuse is not a gate.

    `CoverageDisparity` is record-only, needs `SCORES`, and its envelope needs a stationary grader.
    Handed its input and a regime where that holds, it returns Evidence through the same path
    everything else refuses on.
    """
    from reward_lens.measure.base import Context
    from reward_lens.measure.indices.coverage_disparity import CoverageDisparity

    run, _ = convert_campaign(simple_store)
    caps, _, _ = capabilities_in_record(run)
    ctx, _, _, _ = context_for(run, caps=caps)
    ctx = Context(
        signal=ctx.signal,
        access=ctx.access,
        substrate=Substrate.NEURAL_SCALAR,
        phase=Phase.POST_RUN,
        regime_reading=RegimeReading(
            conditions={
                RegimeCondition.STATIONARY_GRADER: ConditionReading(
                    condition=RegimeCondition.STATIONARY_GRADER,
                    holds=True,
                    statistic=0.0,
                    threshold=0.0,
                )
            }
        ),
    )
    outcome = run_instrument(
        CoverageDisparity(coverage=[0.1, 0.4, 0.9]), ctx, caps=caps, window=(2, 2)
    )
    assert outcome.kind == "evidence"
    assert isinstance(outcome.reading, Evidence)
    assert outcome.reading.value["n_properties"] == 3


def test_an_index_with_no_injected_input_refuses_rather_than_noting(simple_store: Path) -> None:
    """The case the `note` class was invented for now returns what the contract asks for.

    Twelve shipped indices used to answer "none injected" with an `Evidence` carrying a note. That
    satisfied the letter of "Evidence or a Refusal" and told the reader nothing they could act on,
    which is why the sweep classified it as a third kind: counting a "none injected" string as a
    reading would have made the acceptance number a fiction.

    All twelve now refuse, with `ACCESS_INSUFFICIENT` and a remedy naming the constructor argument
    to supply, because the input is one the reader can supply where they are standing. The `note`
    class stays in the sweep rather than being deleted: it is the check that nobody reintroduces
    the pattern, and it is worth more as an empty bucket than as a removed one.
    """
    from reward_lens.core.reading import Refusal, RefusalReason
    from reward_lens.measure.base import Context
    from reward_lens.measure.indices.coverage_disparity import CoverageDisparity

    run, _ = convert_campaign(simple_store)
    caps, _, _ = capabilities_in_record(run)
    base, _, _, _ = context_for(run, caps=caps)
    ctx = Context(
        signal=base.signal,
        access=base.access,
        substrate=Substrate.NEURAL_SCALAR,
        phase=Phase.POST_RUN,
        regime_reading=RegimeReading(
            conditions={
                RegimeCondition.STATIONARY_GRADER: ConditionReading(
                    condition=RegimeCondition.STATIONARY_GRADER,
                    holds=True,
                    statistic=0.0,
                    threshold=0.0,
                )
            }
        ),
    )
    outcome = run_instrument(CoverageDisparity(), ctx, caps=caps, window=(2, 2))
    assert outcome.kind == "refusal"
    assert isinstance(outcome.reading, Refusal)
    assert outcome.reading.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "coverage" in outcome.reading.detail
    assert "CoverageDisparity(coverage=" in outcome.reading.remedy


def test_the_sweep_never_converts_an_exception_into_a_refusal(simple_store: Path) -> None:
    """An escaped exception is recorded as one, so the acceptance test can see it."""

    class Exploding:
        name = "Exploding"
        version = "1.0"
        capabilities = Capability.NONE
        requires = {Component.RECORD: Access.RECORD}

        def estimate(self, ctx: Any) -> Any:
            raise RuntimeError("a bug, not an anticipated condition")

    run, _ = convert_campaign(simple_store)
    report = sweep(run, [Exploding()], caps=Capability.SCORES, caps_steps=2)
    assert len(report.exceptions) == 1
    assert isinstance(report.exceptions[0].error, RuntimeError)
    assert report.exceptions[0].kind == "exception"
    assert not [o for o in report.outcomes if isinstance(o.reading, Refusal)]


# ---------------------------------------------------------------------------
# The arc parser
# ---------------------------------------------------------------------------


@given(
    observable=st.sampled_from(
        ["campaign.bias.battery", "campaign.scores", "campaign.subspace.flat"]
    ),
    roster=st.sampled_from(["armorm", "skywork-v2-qwen3-8b", "tulu-rm"]),
    bank=st.sampled_from(["rb2-full", "diagnostic-v3", "ppe-best-of-k::gpqa"]),
)
def test_a_missing_intermediate_becomes_an_arc_naming_all_three(
    observable: str, roster: str, bank: str
) -> None:
    reason = (
        f"missing intermediate: no intermediate '{observable}' for roster_key='{roster}' "
        f"slice='{bank}'; the arc that produces it has not run or its shard was not merged"
    )
    arcs = metric_arcs_from_reason(reason, spec_id="campaign-x", metrics=["m1", "m2"])
    assert set(arcs) == {"m1", "m2"}
    for arc in arcs.values():
        assert observable in arc and roster in arc and bank in arc


def test_a_subject_resolution_failure_names_the_step_that_failed() -> None:
    arcs = metric_arcs_from_reason(
        "resolve_subjects failed: PermissionError(13, 'Permission denied')",
        spec_id="campaign-gauge-e19",
        metrics=["raw_cos_v01_v02"],
    )
    assert arcs == {"raw_cos_v01_v02": "arc:campaign-gauge-e19/resolve_subjects"}


def test_a_reason_naming_nothing_produces_no_arc_rather_than_a_guess() -> None:
    assert metric_arcs_from_reason("something else went wrong", spec_id="s", metrics=["m"]) == {}


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@st.composite
def _score_grid(draw: st.DrawFn) -> tuple[int, int, np.ndarray]:
    """A rows-by-k float32 score bank, drawn as one value so a counterexample can be pinned.

    The three used to be separate arguments with the values behind `st.data()`, which meant the
    only record of a failing draw was the local `.hypothesis/` directory, and that is gitignored.
    """
    rows = draw(st.integers(min_value=1, max_value=8))
    k = draw(st.integers(min_value=1, max_value=4))
    xs = draw(st.lists(FINITE, min_size=rows * k, max_size=rows * k))
    return rows, k, np.asarray(xs, dtype=np.float32).reshape(rows, k)


#: The draw that broke this property on a fresh seed after it had passed for a long time. Row 5 is
#: [1.0, 524287.96875, -492901.0], whose terms sum to 31387.96875 with a condition number of 32.4.
#: In float32 the partial sum 524288.96875 rounds up to 524289.0, an absolute error of 0.03125
#: which the cancellation then multiplies out to 1.03e-06 relative on the mean. The converter
#: computes the group mean in float64 and lands on 10462.65625, the exact answer; the assertion
#: below used to compare it against `values[i].mean()`, a float32 accumulation that lands on
#: 10462.6669921875. The property was fine and the reference was the less accurate number.
_CANCELLATION_DRAW = (
    6,
    3,
    np.asarray(
        [0.0] * 15 + [1.0, 524287.96875, -492901.0],
        dtype=np.float32,
    ).reshape(6, 3),
)


@settings(
    deadline=None, max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@example(grid=_CANCELLATION_DRAW)
@given(grid=_score_grid())
def test_every_item_is_a_group_and_every_column_a_trajectory(
    tmp_path_factory: pytest.TempPathFactory, grid: tuple[int, int, np.ndarray]
) -> None:
    rows, k, values = grid
    root = tmp_path_factory.mktemp("prop")
    payloads = root / "payloads"
    write_store(root, [score_bank("p", values, payloads, layout="flat" if k == 1 else "bank")])

    run, _ = convert_campaign(root)
    step = next(iter(run.steps))
    assert len(step.groups) == rows
    assert all(g.k == k for g in step.groups)
    assert sum(g.k for g in step.groups) == rows * k
    for i, group in enumerate(step.groups):
        assert str(group.task_ref).endswith(f"::row{i}")
        # The reference accumulates in float64 because the implementation does: the converter
        # widens each recorded score with `float(row[k])` and `GroupStats.from_scores` averages a
        # float64 array. A float32 reference here measures numpy's float32 summation order rather
        # than the grouping this test is named after, and on a row that cancels it is the reference
        # that is wrong by more than the tolerance.
        expected = float(values[i].astype(np.float64).mean())
        assert group.group_stats.mean == pytest.approx(expected, rel=1e-6, abs=1e-6)


@settings(
    deadline=None, max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    rows=st.integers(min_value=1, max_value=6),
    mask=st.data(),
)
def test_a_non_finite_score_is_an_abstention_and_never_a_number(
    tmp_path_factory: pytest.TempPathFactory, rows: int, mask: st.DataObject
) -> None:
    """A grader that produced nothing must not flow into a mean as a zero."""
    flags = mask.draw(st.lists(st.booleans(), min_size=rows * 2, max_size=rows * 2))
    values = np.asarray([math.nan if f else 1.5 for f in flags], dtype=np.float32).reshape(rows, 2)
    root = tmp_path_factory.mktemp("nan")
    payloads = root / "payloads"
    write_store(root, [score_bank("n", values, payloads, layout="pairs")])

    run, _ = convert_campaign(root)
    step = next(iter(run.steps))
    for i, group in enumerate(step.groups):
        expected = int(np.isnan(values[i]).sum())
        assert group.group_stats.n_abstained == expected
        for k, traj in enumerate(group.trajectories):
            leaf = traj.scores
            if math.isnan(float(values[i][k])):
                assert _leaf_field(leaf, "abstained") is True
                assert _leaf_field(leaf, "value") is None
            else:
                assert _leaf_field(leaf, "abstained") is False
        if expected == 2:
            assert group.group_stats.mean is None
        else:
            assert group.group_stats.mean == pytest.approx(1.5, abs=1e-6)


@settings(
    deadline=None, max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(verdicts=st.lists(st.sampled_from([True, False, None]), min_size=1, max_size=6))
def test_folding_a_regime_never_upgrades_a_failure(
    verdicts: list[bool | None],
) -> None:
    """The fold is conservative: one failing step makes the window fail, and unknown outranks pass."""
    from reward_lens.record.convert.instruments import regime_over as fold
    from reward_lens.record.schema import (
        InMemoryStepStream,
        OptimizerTelemetry,
        RegimeDeclaration,
        Run,
        RunID,
        Step,
    )

    steps = [
        Step(
            index=i,
            groups=(),
            schedule={},
            optimizer=OptimizerTelemetry(),
            regime_measured=RegimeReading(
                conditions={
                    RegimeCondition.STATIONARY_GRADER: ConditionReading(
                        condition=RegimeCondition.STATIONARY_GRADER,
                        holds=v,
                        statistic=0.0,
                        threshold=0.0,
                    )
                }
            ),
        )
        for i, v in enumerate(verdicts)
    ]
    run = Run(
        id=RunID("run:fold"),
        kind="eval",
        components={},
        access={},
        regime=RegimeDeclaration(),
        steps=InMemoryStepStream(steps),
    )
    reading, folded, total = fold(run)
    assert (folded, total) == (len(verdicts), len(verdicts))
    expected = False if False in verdicts else (None if None in verdicts else True)
    assert reading.holds(RegimeCondition.STATIONARY_GRADER) is expected

"""`record/labels.py`: the runtime half of the blind-label barrier.

The type checker is the primary barrier and it is tested in `test_record_labels_typing.py`, which
runs mypy. Everything here is the part a type checker cannot see:

- the two frames are field-disjoint, so there is no attribute path from the visible half to the
  held-out half whatever the annotations say;
- the name blocklist covers `features`, which is `Mapping[FeatureID, float]` and therefore accepts
  a label cast to a float with no complaint from anything;
- `check_detector` reads a detector's annotations at runtime, for the callers who do not run mypy;
- and the adjudication row, which is the thing that makes reading a label auditable rather than
  impossible. Three of those tests are about what the row must *not* contain.

The property tests are next door in `test_record_labels_properties.py`, behind an `importorskip`,
because `hypothesis` is not in the base install and the base-install job in CI runs this suite
against the bare wheel.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import SubjectRef, canonical_bytes
from reward_lens.record.labels import (
    BLOCKED_NAMES,
    BLOCKED_PREFIXES,
    UNMEASURED,
    Blind,
    LabelLeak,
    LabelQuality,
    OracleFrame,
    ReadPurpose,
    RolloutFrame,
    adjudicate,
    adjudicate_frame,
    blind,
    blocked_names,
    check_detector,
    detector_findings,
    is_blocked,
    label_reads,
    normalise_name,
    split_trajectory,
)
from reward_lens.record.schema import FeatureID, decode_foreign, encode_foreign, make_trajectory
from reward_lens.record.turns import Turn

MEASURED = LabelQuality(
    error_rate=0.03,
    n_audited=200,
    method="two raters on a stratified sample, third adjudicating disagreements",
    measured_by="pool-a",
)


def a_turn(index: int = 0, text: str = "perhaps") -> Turn:
    return Turn(
        index=index,
        role="assistant",
        text=text,
        token_ids=(1, 2, 3),
        logprobs_sampling=None,
        logprobs_train=None,
        loss_mask=None,
        tool_call=None,
    )


# ---------------------------------------------------------------------------
# There is no .unwrap()
# ---------------------------------------------------------------------------


def test_a_blind_has_no_unwrap_no_value_and_no_get() -> None:
    """The three names a reader reaches for first. None of them exists.

    Asserted at runtime as well as under mypy because the static check only sees the call sites it
    was pointed at, and `getattr(label, "value")` is invisible to it.
    """
    label = blind(True, key="hacked")
    for name in ("unwrap", "value", "get", "open", "reveal"):
        assert not hasattr(label, name), (
            f"Blind grew a {name!r}; the whole design is that it has none"
        )


def test_the_payload_is_reachable_only_through_a_private_name() -> None:
    """Stated as a test because it is a limitation, not a guarantee.

    Python has no private state. `_value` reads the label and this test says so out loud, so that
    nobody reads the module docstring's honesty as a figure of speech. What the underscore buys is
    that the access is greppable and does not look like ordinary code.
    """
    label = blind("hacked", key="verdict")
    assert label._value == "hacked"
    public = [n for n in dir(label) if not n.startswith("_")]
    assert public == sorted(["key", "quality", "fingerprint", "is_measured"])


def test_equality_is_identity_because_equality_on_a_boolean_label_is_the_whole_label() -> None:
    """`blind(True) == True` would be a one-bit oracle needing no adjudication row, and a boolean
    label is exactly one bit. Two Blinds therefore compare by identity."""
    a = blind(True, key="hacked")
    b = blind(True, key="hacked")
    assert a != b
    assert a == a
    assert a.fingerprint == b.fingerprint  # what a round-trip test compares instead


def test_bool_raises_rather_than_answering_true_for_every_label() -> None:
    """`if label:` on a `Blind[bool]` is true whatever the label is, so it is a silent wrong
    answer rather than a leak. The module's one unforgivable output is a confident wrong number."""
    label = blind(False, key="hacked")
    with pytest.raises(LabelLeak, match="no truth value"):
        bool(label)


def test_repr_and_str_never_render_the_payload() -> None:
    """A record printed into a log, a traceback or a prompt is a leakage path that does not look
    like an access at all."""
    label = blind("SENTINEL-PAYLOAD-9f3a", key="verdict", quality=MEASURED)
    for rendered in (repr(label), str(label), f"{label}", format(label)):
        assert "SENTINEL-PAYLOAD-9f3a" not in rendered
    assert "key='verdict'" in repr(label)
    assert "measured" in repr(label)


def test_canonicalisation_hides_the_payload_but_the_record_codec_keeps_it() -> None:
    """Two paths, on purpose.

    `content_hash` names evidence ids and sidecar files, and an id derived from an oracle value is
    a copy of the oracle in a filename. The record codec has to keep the payload or a record cannot
    round-trip. Both behaviours are asserted here so that neither one gets "fixed" into the other.
    """
    label = blind("SENTINEL-PAYLOAD-9f3a", key="verdict")
    assert b"SENTINEL-PAYLOAD-9f3a" not in canonical_bytes(label)
    assert set(label.__canonical__()) == {"key", "fingerprint", "quality"}
    assert decode_foreign(encode_foreign(label))._value == "SENTINEL-PAYLOAD-9f3a"


def test_the_fingerprint_is_the_documented_digest() -> None:
    """Recomputed independently rather than pinned to a copied string.

    BLAKE2b-128 over the canonical JSON of `{"key": ..., "value": ...}`, tagged `lbl:`, which is
    what `content_hash` does and what the docstring claims it does.
    """
    label = blind(True, key="hacked")
    material = json.dumps(
        {"key": "hacked", "value": True}, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    expected = "lbl:" + hashlib.blake2b(material, digest_size=16).hexdigest()
    assert label.fingerprint == expected


def test_a_label_needs_a_key() -> None:
    with pytest.raises(ValueError, match="needs a key"):
        blind(True, key="  ")


# ---------------------------------------------------------------------------
# Runtime test 1 of 3: disjoint field sets
# ---------------------------------------------------------------------------


def test_the_two_frames_have_disjoint_field_sets() -> None:
    """The first of the three tests the clause names.

    Field-disjoint means there is no attribute path from the visible half to the held-out half, so
    a detector holding a `RolloutFrame` cannot reach the answer key even through `getattr`. The
    join key is `trajectory_id` on one and `trajectory_ref` on the other, which is what keeps the
    two joinable without putting a shared name in both field sets.
    """
    rollout = {f.name for f in RolloutFrame.__dataclass_fields__.values()}
    oracle = {f.name for f in OracleFrame.__dataclass_fields__.values()}

    assert rollout & oracle == set()
    assert rollout == {"trajectory_id", "task_id", "turns", "n_tokens", "advantage", "features"}
    assert oracle == {"trajectory_ref", "labels", "quality", "onset_step", "labeller"}


def test_no_public_attribute_of_the_visible_frame_is_named_like_the_answer_key() -> None:
    """Fields are not the whole surface. A property named `label` would pass the disjointness test
    above and leak anyway, so the blocklist is applied to the class's public names too."""
    public = [n for n in dir(RolloutFrame) if not n.startswith("_")]
    assert blocked_names(public) == ()


def test_a_detector_cannot_reach_the_oracle_through_the_frame_it_is_given() -> None:
    """The property the disjointness is for, asserted on instances rather than on field names."""
    trajectory = make_trajectory(
        id="t1",
        task_ref="task-1",
        turns=[a_turn()],
        labels={"hacked": blind(True, key="hacked")},
        features={"len_tokens": 3.0},
    )
    rollout, oracle = split_trajectory(trajectory)

    assert not hasattr(rollout, "labels")
    assert not hasattr(rollout, "oracle")
    assert list(oracle.labels) == ["hacked"]
    assert rollout.trajectory_id == oracle.trajectory_ref == "t1"
    assert rollout.features == {FeatureID("len_tokens"): 3.0}
    assert rollout.n_turns == 1

    # The label metrology is a property of the labelling campaign rather than of one rollout, so
    # it is passed in rather than read off the trajectory, and it defaults to unmeasured.
    assert oracle.quality is UNMEASURED
    _, described = split_trajectory(
        trajectory, quality=MEASURED, onset_step=140, labeller="aisi-2026-03"
    )
    assert described.quality == MEASURED
    assert described.onset_step == 140
    assert described.labeller == "aisi-2026-03"


# ---------------------------------------------------------------------------
# Runtime test 2 of 3: the name blocklist
# ---------------------------------------------------------------------------


def test_a_label_smuggled_into_features_as_a_float_is_caught_by_name() -> None:
    """The second of the three. This is the leak no type checker can see.

    `features` is `Mapping[FeatureID, float]`, so `features["is_hack"] = 1.0` type checks perfectly
    and reaches every detector that reads features. The blocklist is the only thing between that
    line and a scored experiment.
    """
    with pytest.raises(LabelLeak) as exc:
        RolloutFrame(
            trajectory_id="t1",
            task_id="task-1",
            turns=(),
            n_tokens=0,
            advantage=None,
            features={
                FeatureID("is_hack"): 1.0,
                FeatureID("Gold-Label"): 0.0,
                FeatureID("len_tokens"): 3.0,
            },
        )
    message = str(exc.value)
    assert "'is_hack'" in message and "'Gold-Label'" in message
    assert "len_tokens" not in message
    assert "2 feature name(s)" in message


def test_the_blocklist_matches_whole_names_after_normalisation() -> None:
    """Exact matches on the normalised name, plus a short prefix list. Hand-checked both ways."""
    assert normalise_name("Gold-Label") == "gold_label"
    assert normalise_name(" ORACLE.score ") == "oracle_score"

    for name in ("label", "gold", "y_true", "is_hacked", "onset_step", "GROUND_TRUTH"):
        assert is_blocked(name), name
    for name in ("oracle_free_baseline", "gold_standard_ref", "label_confidence"):
        assert is_blocked(name), f"{name} starts with a blocked prefix"

    for name in ("len_tokens", "target_length", "n_turns", "advantage", "reward_pass"):
        assert not is_blocked(name), name


def test_the_blocklist_cannot_catch_an_unconventional_name_and_the_test_says_so() -> None:
    """Recorded as a test rather than as a caveat in a docstring.

    A label written in as `f47` passes, and no list of English words will ever catch it. The guard
    is against the hurried converter and this is the boundary of what it buys.
    """
    frame = RolloutFrame(
        trajectory_id="t1",
        task_id="task-1",
        turns=(),
        n_tokens=0,
        advantage=None,
        features={FeatureID("f47"): 1.0},
    )
    assert frame.features[FeatureID("f47")] == 1.0
    # 32 English words and nine prefixes, which is the whole of it. Pinned so that growing the
    # list is a decision somebody made rather than a drift nobody read.
    assert len(BLOCKED_NAMES) == 32
    assert len(BLOCKED_PREFIXES) == 9


def test_the_oracle_frame_refuses_a_raw_label() -> None:
    """A converter reading someone else's schema is where a raw `bool` gets in."""
    with pytest.raises(LabelLeak, match="must be a Blind"):
        OracleFrame(trajectory_ref="t1", labels={"hacked": True})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# Runtime test 3 of 3: introspection of the detector protocol's annotation
# ---------------------------------------------------------------------------


def good_detector(frame: RolloutFrame) -> float:
    return float(frame.n_tokens)


def detector_taking_the_oracle(frame: OracleFrame) -> float:
    return float(len(frame.labels))


def detector_taking_a_bare_label(label: Blind[bool]) -> float:
    return 0.0


def detector_taking_a_mapping_of_labels(
    frame: RolloutFrame, labels: dict[str, Blind[bool]]
) -> float:
    return float(len(labels))


def detector_with_an_unannotated_parameter(frame) -> float:
    """Unannotated on purpose. This is the signature `check_detector` has to report."""
    return 0.0


def detector_with_no_return_annotation(frame: RolloutFrame):
    return 0.0


def detector_with_open_kwargs(frame: RolloutFrame, **extra) -> float:
    return 0.0


def detector_returning_the_oracle(frame: RolloutFrame) -> Blind[bool]:
    return blind(True, key="hacked")


class StatefulDetector:
    def __call__(self, frame: RolloutFrame) -> float:
        return float(frame.n_tokens)


class StatefulLeakyDetector:
    def __call__(self, frame: OracleFrame) -> float:
        return float(len(frame.labels))


def test_a_well_formed_detector_passes_introspection() -> None:
    assert detector_findings(good_detector) == ()
    assert detector_findings(StatefulDetector()) == ()
    check_detector(good_detector)
    check_detector(StatefulDetector())


@pytest.mark.parametrize(
    "detector,expected",
    [
        (detector_taking_the_oracle, "reaches the held-out labels"),
        (detector_taking_a_bare_label, "reaches the held-out labels"),
        (detector_taking_a_mapping_of_labels, "reaches the held-out labels"),
        (detector_with_an_unannotated_parameter, "has no annotation"),
        (detector_with_no_return_annotation, "the return is not annotated"),
        (detector_with_open_kwargs, "accepts anything"),
        (detector_returning_the_oracle, "hands the oracle back"),
        (StatefulLeakyDetector(), "reaches the held-out labels"),
    ],
)
def test_introspection_reads_the_detector_annotations(detector: object, expected: str) -> None:
    """The third of the three, one case per way a signature reaches the oracle.

    The container case matters most: `dict[str, Blind[bool]]` mentions no oracle type at the top
    level, and a check that only looked at `isinstance(annotation, type)` would pass it. The
    unannotated case is the other one worth arguing about, and treating it as a finding is
    deliberate: an unannotated parameter is precisely the hole a type checker cannot close, so
    reporting it as clean would be reporting success on the one signature nothing can check.
    """
    findings = detector_findings(detector)
    assert len(findings) == 1
    assert expected in findings[0]
    with pytest.raises(LabelLeak, match="can reach the held-out labels"):
        check_detector(detector)


def test_an_unresolvable_annotation_is_a_finding_rather_than_a_crash() -> None:
    """A forward reference to a name that does not exist is exactly as unverifiable as no
    annotation at all, and the check has to say so rather than raise NameError at the caller."""

    def detector(frame: "NoSuchFrame") -> float:  # noqa: F821
        return 0.0

    findings = detector_findings(detector)
    assert len(findings) == 1
    assert "could not be resolved" in findings[0]


# ---------------------------------------------------------------------------
# Adjudication: the audited way out
# ---------------------------------------------------------------------------


def test_adjudicate_returns_the_label_and_writes_exactly_one_row(tmp_path) -> None:
    store = EvidenceStore(tmp_path)
    label = blind("hacked", key="verdict", quality=MEASURED)

    got = adjudicate(
        label,
        instrument="D1",
        purpose=ReadPurpose.SCORING,
        why="scoring the hedging detector against the held-out verdicts",
        subject="t1",
        store=store,
    )

    assert got == "hacked"
    rows = label_reads(store)
    assert len(rows) == 1
    row = rows[0].value
    assert row.key == "verdict"
    assert row.instrument == "D1"
    assert row.purpose == "scoring"
    assert row.why.startswith("scoring the hedging detector")
    assert row.n_labels == 1
    assert row.error_rate == 0.03
    assert row.n_audited == 200


def test_the_row_carries_a_fingerprint_and_never_the_label(tmp_path) -> None:
    """The row is the audit trail and it must not become a second copy of the answer key.

    The evidence store is read at `RECORD` access by every access profile, including the
    person whose detector is being scored. A row holding the value would be the oracle sitting
    outside the type system, reachable by the one access level everybody has. Asserted against the
    raw bytes on disk rather than against the payload object, because what matters is what a
    reader of the file can see.
    """
    store = EvidenceStore(tmp_path)
    label = blind("SENTINEL-PAYLOAD-9f3a", key="verdict", quality=MEASURED)

    adjudicate(
        label,
        instrument="D1",
        purpose=ReadPurpose.SCORING,
        why="scoring",
        subject="t1",
        store=store,
    )

    on_disk = (tmp_path / "evidence.jsonl").read_text(encoding="utf-8")
    assert "SENTINEL-PAYLOAD-9f3a" not in on_disk
    assert label.fingerprint in on_disk


def test_two_reads_of_the_same_label_leave_two_rows(tmp_path) -> None:
    """The dedup trap, asserted because it would be silent.

    Evidence ids are content-derived and `created_at` is excluded from the hash, so two identical
    reads would collapse to one row and a hundred reads would leave one. `LabelRead` carries
    `read_at` and a per-process `read_index` for exactly this reason. An audit trail that counts
    wrong is worse than no audit trail.
    """
    store = EvidenceStore(tmp_path)
    label = blind(True, key="hacked", quality=MEASURED)
    for _ in range(3):
        adjudicate(
            label,
            instrument="D1",
            purpose=ReadPurpose.SCORING,
            why="scoring",
            subject="t1",
            store=store,
        )

    rows = label_reads(store)
    assert len(rows) == 3
    assert len({r.id for r in rows}) == 3
    assert [r.value.read_index for r in rows] == sorted(r.value.read_index for r in rows)


def test_scoring_against_unmeasured_labels_is_refused_with_a_reason_and_a_remedy(tmp_path) -> None:
    """Scoring against labels with no measured error rate measures the labels.

    The refusal is a value. It carries the numbers that produced it (`error_rate=None`,
    `n_audited=0`) and a remedy that names the call the reader should make instead, which is the
    difference between a tool that looks careful and a tool that looks broken.
    """
    store = EvidenceStore(tmp_path)
    label = blind(True, key="hacked")

    reading = adjudicate(
        label,
        instrument="D1",
        purpose=ReadPurpose.SCORING,
        why="scoring the hedging detector",
        subject="t1",
        store=store,
    )

    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.LABEL_QUALITY_UNKNOWN
    assert reading.instrument == "D1"
    assert "error_rate=None" in reading.detail
    assert "n_audited=0" in reading.detail
    assert "LabelQuality(error_rate=..., n_audited=..., method=..., measured_by=...)" in (
        reading.remedy
    )
    assert "purpose=ReadPurpose.AUDIT" in reading.remedy
    assert reading.statistics == {"keys": ["hacked"], "error_rate": None, "n_audited": 0}
    assert label_reads(store) == (), "a refusal opened nothing, so it writes no row"


def test_an_audit_read_opens_the_labels_that_scoring_cannot(tmp_path) -> None:
    """The escape from the circularity: measuring the error rate means reading the labels."""
    store = EvidenceStore(tmp_path)
    label = blind(True, key="hacked", quality=UNMEASURED)

    got = adjudicate(
        label,
        instrument="L2",
        purpose=ReadPurpose.AUDIT,
        why="hand-auditing 200 labels to measure the answer key's error rate",
        subject=SubjectRef(extra={"trajectory": "t1"}),
        store=store,
    )

    assert got is True
    assert label_reads(store)[0].value.purpose == "audit"


def test_adjudicate_demands_a_reason(tmp_path) -> None:
    store = EvidenceStore(tmp_path)
    with pytest.raises(ValueError, match="needs a reason"):
        adjudicate(
            blind(True, key="hacked", quality=MEASURED),
            instrument="D1",
            purpose=ReadPurpose.SCORING,
            why="   ",
            subject="t1",
            store=store,
        )


def test_a_frame_opens_in_one_row_carrying_every_key(tmp_path) -> None:
    """The bulk path. A thousand trajectories adjudicated one label at a time writes a thousand
    rows and leaves the reviewer to reconstruct the pass; this writes one row per trajectory."""
    store = EvidenceStore(tmp_path)
    frame = OracleFrame(
        trajectory_ref="t1",
        labels={
            "hacked": blind(True, key="hacked"),
            "verdict": blind("exploited", key="verdict"),
            "onset": blind(140, key="onset"),
        },
        quality=MEASURED,
        labeller="aisi-2026-03",
    )

    opened = adjudicate_frame(
        frame,
        instrument="D1",
        purpose=ReadPurpose.SCORING,
        why="scoring the hedging detector against all three held-out labels",
        store=store,
    )

    assert opened == {"hacked": True, "verdict": "exploited", "onset": 140}
    rows = label_reads(store)
    assert len(rows) == 1
    assert rows[0].value.n_labels == 3
    assert rows[0].value.key == "hacked,onset,verdict"
    assert rows[0].value.error_rate == 0.03


def test_a_frame_refuses_whole_rather_than_returning_the_audited_subset(tmp_path) -> None:
    """A partial answer key silently scoped to the labels that happened to be audited is a
    different experiment from the one the caller wrote."""
    store = EvidenceStore(tmp_path)
    frame = OracleFrame(
        trajectory_ref="t1",
        labels={
            "hacked": blind(True, key="hacked", quality=MEASURED),
            "onset": blind(140, key="onset"),
        },
    )

    reading = adjudicate_frame(
        frame,
        instrument="D1",
        purpose=ReadPurpose.SCORING,
        why="scoring",
        store=store,
    )

    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.LABEL_QUALITY_UNKNOWN
    assert reading.statistics["keys"] == ["onset"]
    assert label_reads(store) == ()


def test_an_audit_row_over_a_mixed_frame_reports_the_unmeasured_label(tmp_path) -> None:
    """The row reports the worst quality in the frame, and unmeasured is worse than any rate.

    Only reachable on the audit path, since scoring refuses a mixed frame outright. It matters
    because the alternative is a row reading `error_rate=0.03` while describing a frame where one
    label has never been looked at, which is the row a reviewer would most confidently misread.
    """
    store = EvidenceStore(tmp_path)
    frame = OracleFrame(
        trajectory_ref="t1",
        labels={
            "hacked": blind(True, key="hacked", quality=MEASURED),
            "onset": blind(140, key="onset"),
        },
    )

    opened = adjudicate_frame(
        frame,
        instrument="L2",
        purpose=ReadPurpose.AUDIT,
        why="checking both labels by hand before either is used for scoring",
        store=store,
    )

    assert opened == {"hacked": True, "onset": 140}
    row = label_reads(store)[0].value
    assert row.error_rate is None
    assert row.n_audited == 0


def test_a_labels_own_quality_overrides_the_frames() -> None:
    frame = OracleFrame(
        trajectory_ref="t1",
        labels={
            "hacked": blind(True, key="hacked", quality=MEASURED),
            "onset": blind(140, key="onset"),
        },
        quality=UNMEASURED,
    )
    assert frame.quality_of("hacked").is_measured
    assert not frame.quality_of("onset").is_measured


# ---------------------------------------------------------------------------
# Label quality
# ---------------------------------------------------------------------------


def test_a_rate_measured_on_nothing_is_rejected() -> None:
    """A rate with no sample size behind it is a guess with a number attached to it."""
    with pytest.raises(ValueError, match="n_audited is 0"):
        LabelQuality(error_rate=0.03)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        LabelQuality(error_rate=1.4, n_audited=10)
    assert not UNMEASURED.is_measured
    assert LabelQuality(error_rate=0.0, n_audited=1).is_measured

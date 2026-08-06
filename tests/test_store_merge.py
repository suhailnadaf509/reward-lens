"""Cross-process merge of per-shard evidence stores (`EvidenceStore.merge`).

The store is single-writer, so a fan-out run leaves one shard store per worker and the report
step folds them into one DAG. The merge must preserve envelope content byte for byte (ids are
content-derived), land parents before children, refuse a parent that resolves nowhere, tolerate
a shard that never got written, and union the content-addressed payload sidecars so bulk arrays
stay readable from the merged store. Each of those promises gets a test here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from reward_lens.core import (
    EvidenceStore,
    ModelFP,
    Provenance,
    ProvenanceError,
    SubjectRef,
    make_evidence,
)

_T0 = "2026-07-01T00:00:00+00:00"
_T1 = "2026-07-01T00:00:05+00:00"


def _subject() -> SubjectRef:
    return SubjectRef(signals=(ModelFP("mfp:shard-demo"),), dataset="ds:shard", readout="reward")


def _evidence(value, *, observable="A", parents=(), created_at=_T0):
    return make_evidence(
        observable=observable,
        observable_version="1",
        subject=_subject(),
        value=value,
        provenance=Provenance(parents=tuple(parents)),
        created_at=created_at,
    )


def _write_shard_line(store: EvidenceStore, evidence) -> None:
    # A worker whose store was seeded elsewhere can hold a derived envelope whose parent is not
    # local, and `append` would rightly refuse it there. That is the situation merge exists for,
    # so the shard is written the way its file actually looks: one envelope per line.
    with store.jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(evidence.envelope(sidecar_dir=store.payloads)) + "\n")


def test_cross_shard_parent_lands_in_dependency_order(tmp_path):
    leaf = _evidence(1.0, created_at=_T0)
    child = _evidence(2.0, observable="B", parents=[leaf.id], created_at=_T1)

    shard_a = EvidenceStore(tmp_path / "shard_a")
    shard_a.append(leaf)
    shard_b = EvidenceStore(tmp_path / "shard_b")
    _write_shard_line(shard_b, child)

    # The shards are given child-first; ascending created_at across shards fixes the order.
    merged = EvidenceStore.merge([shard_b.path, shard_a.path], tmp_path / "merged")
    assert merged.merge_report == {"merged": 2, "skipped_existing": 0, "missing_shards": []}
    assert merged.get(child.id).provenance.parents == (leaf.id,)

    # The parent's line precedes the child's on disk, so replaying the JSONL through `append`
    # would succeed too, and ids survived byte for byte: a fresh index resolves the DAG.
    ids = [json.loads(line)["id"] for line in merged.jsonl.read_text().strip().splitlines()]
    assert ids == [leaf.id, child.id]
    reopened = EvidenceStore(tmp_path / "merged")
    assert reopened.parents(reopened.get(child.id))[0].id == leaf.id


def test_equal_timestamps_still_land_parent_first(tmp_path):
    # Two workers stamping the same second can present the child before its parent even under
    # the created_at sort; the merge defers the child a pass instead of appending out of order.
    leaf = _evidence(1.0, created_at=_T0)
    child = _evidence(2.0, observable="B", parents=[leaf.id], created_at=_T0)

    shard_a = EvidenceStore(tmp_path / "shard_a")
    _write_shard_line(shard_a, child)
    shard_b = EvidenceStore(tmp_path / "shard_b")
    shard_b.append(leaf)

    merged = EvidenceStore.merge([shard_a.path, shard_b.path], tmp_path / "merged")
    ids = [json.loads(line)["id"] for line in merged.jsonl.read_text().strip().splitlines()]
    assert ids == [leaf.id, child.id]


def test_dangling_parent_is_refused_before_anything_is_written(tmp_path):
    orphan = _evidence(3.0, parents=["ev:nowhere"], created_at=_T0)
    shard = EvidenceStore(tmp_path / "shard")
    _write_shard_line(shard, orphan)

    dest = tmp_path / "merged"
    with pytest.raises(ProvenanceError) as excinfo:
        EvidenceStore.merge([shard.path], dest)
    message = str(excinfo.value)
    assert orphan.id in message
    assert "ev:nowhere" in message
    assert str(shard.path) in message
    # The refusal happened before any write: the destination holds no envelopes.
    assert len(EvidenceStore(dest)) == 0


def test_remerge_is_idempotent(tmp_path):
    shard = EvidenceStore(tmp_path / "shard")
    shard.append(_evidence(1.0, created_at=_T0))
    shard.append(_evidence(2.0, observable="B", created_at=_T1))

    dest = tmp_path / "merged"
    first = EvidenceStore.merge([shard.path], dest)
    assert first.merge_report["merged"] == 2

    again = EvidenceStore.merge([shard.path], dest)
    assert again.merge_report["merged"] == 0
    assert again.merge_report["skipped_existing"] == 2
    assert len(again) == 2
    assert len(again.jsonl.read_text().strip().splitlines()) == 2


def test_missing_shard_is_reported_not_fatal(tmp_path):
    shard = EvidenceStore(tmp_path / "real")
    ev = _evidence(1.0, created_at=_T0)
    shard.append(ev)
    ghost = tmp_path / "never-written"

    merged = EvidenceStore.merge([shard.path, ghost], tmp_path / "merged")
    assert merged.merge_report["merged"] == 1
    assert merged.merge_report["missing_shards"] == [str(ghost)]
    assert ev.id in merged


def _tear_last_line(store: EvidenceStore) -> None:
    """Simulate a writer killed mid-append: keep only half of the file's final line."""
    raw = store.jsonl.read_bytes()
    body = raw.rstrip(b"\n")
    cut = body.rfind(b"\n") + 1
    store.jsonl.write_bytes(raw[:cut] + body[cut : cut + (len(body) - cut) // 2])


def test_merge_skips_a_shards_torn_tail_without_mutating_it(tmp_path):
    # A worker killed (or still writing) mid-append leaves its shard with a truncated final
    # line. The merge must not choke on it, and it must not touch the shard file either: the
    # writer may be alive, and a read-side truncate would corrupt its next append. The
    # complete envelopes merge; the tail stays in place for the shard's own writer.
    shard = EvidenceStore(tmp_path / "shard")
    kept = _evidence(1.0, created_at=_T0)
    shard.append(kept)
    shard.append(_evidence(2.0, observable="B", created_at=_T1))
    _tear_last_line(shard)
    torn_bytes = shard.jsonl.read_bytes()

    with pytest.warns(RuntimeWarning, match="torn final line"):
        merged = EvidenceStore.merge([shard.path], tmp_path / "merged")
    assert merged.merge_report["merged"] == 1
    assert kept.id in merged
    # The shard was not mutated: no quarantine sidecar, bytes identical.
    assert not (shard.path / "evidence.jsonl.partial").exists()
    assert shard.jsonl.read_bytes() == torn_bytes

    # The shard's own writer (a re-run of the killed arc) quarantines the tail on load and
    # proceeds; the healed shard then re-merges idempotently.
    with pytest.warns(RuntimeWarning, match="torn final line"):
        EvidenceStore(shard.path)
    assert (shard.path / "evidence.jsonl.partial").exists()
    again = EvidenceStore.merge([shard.path], tmp_path / "merged")
    assert again.merge_report == {"merged": 0, "skipped_existing": 1, "missing_shards": []}


def test_readonly_store_neither_repairs_nor_appends(tmp_path):
    # A readonly open of a torn store reads the good prefix, leaves the file byte-identical,
    # and refuses appends; a writer open afterwards still repairs normally.
    shard = EvidenceStore(tmp_path / "shard")
    kept = _evidence(1.0, created_at=_T0)
    shard.append(kept)
    shard.append(_evidence(2.0, observable="B", created_at=_T1))
    _tear_last_line(shard)
    torn_bytes = shard.jsonl.read_bytes()

    with pytest.warns(RuntimeWarning, match="torn final line"):
        ro = EvidenceStore(shard.path, readonly=True)
    assert kept.id in ro
    assert shard.jsonl.read_bytes() == torn_bytes
    with pytest.raises(RuntimeError, match="readonly"):
        ro.append(_evidence(3.0, observable="C", created_at=_T1))


def test_remerge_recovers_a_destination_killed_mid_merge(tmp_path):
    # A merge killed mid-write leaves the DESTINATION with a torn tail. Re-merging must
    # quarantine it and land the interrupted envelope again from its shard.
    shard = EvidenceStore(tmp_path / "shard")
    first = _evidence(1.0, created_at=_T0)
    second = _evidence(2.0, observable="B", created_at=_T1)
    shard.append(first)
    shard.append(second)

    dest = tmp_path / "merged"
    EvidenceStore.merge([shard.path], dest)
    _tear_last_line(EvidenceStore(dest))

    with pytest.warns(RuntimeWarning, match="torn final line"):
        merged = EvidenceStore.merge([shard.path], dest)
    assert merged.merge_report == {"merged": 1, "skipped_existing": 1, "missing_shards": []}
    assert first.id in merged and second.id in merged
    ids = [json.loads(line)["id"] for line in merged.jsonl.read_text().strip().splitlines()]
    assert ids == [first.id, second.id]


def test_payload_sidecars_are_unioned(tmp_path):
    # An array above the inline threshold goes to a content-addressed .npy sidecar; the merge
    # must carry that file across or the merged envelope is unreadable.
    arr = np.arange(300, dtype=np.float32)
    ev = _evidence({"depths": arr, "peak": 0.7}, created_at=_T0)
    shard = EvidenceStore(tmp_path / "shard")
    shard.append(ev)
    shard_sidecars = {p.name for p in shard.payloads.iterdir()}
    assert shard_sidecars

    EvidenceStore.merge([shard.path], tmp_path / "merged")
    merged = EvidenceStore(tmp_path / "merged")  # fresh index, reads only dest's payloads
    got = merged.get(ev.id)
    assert np.allclose(got.value["depths"], arr)
    assert {p.name for p in merged.payloads.iterdir()} == shard_sidecars

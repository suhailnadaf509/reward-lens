"""Acceptance: a store killed mid-append reopens without loss.

The clause this file discharges: *a store killed mid-append reopens without loss and the partial
bytes land in `.partial`; two shards with a cross-shard parent merge correctly and a shard with a
dangling parent is rejected before anything is written.*

The merge half of the clause is discharged by `tests/test_store_merge.py`, ported from the
campaign branch along with the code it tests. This file covers the half that test does not: the
writer's own store, opened after its writer was killed.

Three distinct residues a killed writer can leave, and they need three different repairs:

1. A truncated final line. It fails to parse, and only the tail may fail to parse, so it is
   quarantined into `<name>.partial` and the file is truncated back to the last good newline.
2. A complete final envelope with no trailing newline. This one is the nasty case, because it
   parses fine, so the torn-tail quarantine never sees it. Appending straight onto it welds two
   envelopes into one unparseable line and poisons every later load of the store.
3. A payload sidecar truncated mid-`np.save`. Existence at the content-addressed name is what
   every later run trusts, so the write goes to a temp name and is renamed into place.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from reward_lens.core import (
    EvidenceStore,
    ModelFP,
    Provenance,
    SubjectRef,
    make_evidence,
)


def _evidence(value, *, observable="A", created_at="2026-07-01T00:00:00+00:00"):
    return make_evidence(
        observable=observable,
        observable_version="1",
        subject=SubjectRef(
            signals=(ModelFP("mfp:hardening"),), dataset="ds:hardening", readout="reward"
        ),
        value=value,
        provenance=Provenance(),
        created_at=created_at,
    )


def _kill_mid_append(store: EvidenceStore, evidence, *, keep_bytes: int) -> bytes:
    """Simulate a process killed partway through writing one envelope's bytes.

    An append is not atomic at the filesystem level: a large envelope spans several write
    syscalls, so this is the one corruption an append-only writer can produce by itself.
    """
    line = (json.dumps(evidence.envelope(sidecar_dir=store.payloads)) + "\n").encode("utf-8")
    partial = line[:keep_bytes]
    with store.jsonl.open("ab") as fh:
        fh.write(partial)
    return partial


def test_a_store_killed_mid_append_reopens_without_losing_the_good_rows(tmp_path):
    """The clause. Two good envelopes survive, the torn third is quarantined."""
    store = EvidenceStore(tmp_path / "store")
    first = _evidence(1.0)
    second = _evidence(2.0, observable="B")
    store.append(first)
    store.append(second)

    torn_bytes = _kill_mid_append(store, _evidence(3.0, observable="C"), keep_bytes=40)

    # Reopening is where the repair happens, and it warns rather than failing silently.
    with pytest.warns(RuntimeWarning, match="quarantined a torn final line"):
        reopened = EvidenceStore(tmp_path / "store")

    # Nothing complete was lost.
    assert len(reopened) == 2
    assert first.id in reopened
    assert second.id in reopened

    # The partial bytes were preserved, not discarded.
    partial = (tmp_path / "store" / "evidence.jsonl.partial").read_bytes()
    assert partial.startswith(torn_bytes)

    # And the JSONL now ends at a complete envelope, so the next append is safe.
    raw = (tmp_path / "store" / "evidence.jsonl").read_bytes()
    assert raw.endswith(b"\n")
    assert len(raw.strip().splitlines()) == 2

    third = _evidence(4.0, observable="D")
    reopened.append(third)
    assert len(EvidenceStore(tmp_path / "store")) == 3


def test_a_missing_trailing_newline_is_healed_rather_than_welded(tmp_path):
    """The case the quarantine cannot see, because the tail parses.

    Without the heal, the next append concatenates onto the unterminated line and produces
    `{...}{...}`, which is not JSON, and from then on every load of the store raises.
    """
    store = EvidenceStore(tmp_path / "store")
    store.append(_evidence(1.0))

    # Drop the terminator the way a kill between the write and the flush would.
    raw = store.jsonl.read_bytes()
    assert raw.endswith(b"\n")
    store.jsonl.write_bytes(raw[:-1])

    reopened = EvidenceStore(tmp_path / "store")
    assert len(reopened) == 1
    reopened.append(_evidence(2.0, observable="B"))

    lines = store.jsonl.read_bytes().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # both parse; nothing was welded

    assert len(EvidenceStore(tmp_path / "store")) == 2


def test_real_corruption_anywhere_but_the_tail_still_raises(tmp_path):
    """The tolerance is exactly one torn final line. Anything else must stay loud.

    A store that quietly skips bad lines wherever it finds them is a store that silently loses
    evidence, which is the failure this whole subsystem exists to make impossible.
    """
    store = EvidenceStore(tmp_path / "store")
    store.append(_evidence(1.0))
    store.append(_evidence(2.0, observable="B"))

    lines = store.jsonl.read_bytes().splitlines(keepends=True)
    store.jsonl.write_bytes(b"{not json at all\n" + b"".join(lines))

    with pytest.raises(json.JSONDecodeError):
        EvidenceStore(tmp_path / "store")


def test_a_readonly_reader_never_mutates_a_live_writers_store(tmp_path):
    """Truncating a torn tail from the read side corrupts the writer's next append."""
    store = EvidenceStore(tmp_path / "store")
    store.append(_evidence(1.0))
    _kill_mid_append(store, _evidence(2.0, observable="B"), keep_bytes=30)
    before = store.jsonl.read_bytes()

    with pytest.warns(RuntimeWarning, match="without touching the file"):
        reader = EvidenceStore(tmp_path / "store", readonly=True)

    assert len(reader) == 1
    assert store.jsonl.read_bytes() == before
    assert not (tmp_path / "store" / "evidence.jsonl.partial").exists()

    with pytest.raises(RuntimeError, match="readonly"):
        reader.append(_evidence(3.0, observable="C"))


def test_a_payload_sidecar_never_appears_truncated_at_its_trusted_name(tmp_path):
    """Content-addressed existence is trusted forever, so the write must be atomic.

    The assertion is on the mechanism rather than on a simulated kill: `np.save` writes to a
    temp name and `os.replace` moves it into place, so no reader can observe a partial file at
    the address a later run will skip recomputation for.
    """
    store = EvidenceStore(tmp_path / "store")
    arr = np.arange(4096, dtype=np.float64)  # well over the inline threshold
    ev = _evidence(arr)

    seen: list[str] = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append(f"{os.path.basename(str(src))} -> {os.path.basename(str(dst))}")
        return real_replace(src, dst)

    os.replace = spy
    try:
        store.append(ev)
    finally:
        os.replace = real_replace

    assert any("-> " in s and ".tmp" in s for s in seen), seen
    sidecars = list((tmp_path / "store" / "payloads").glob("*.npy"))
    assert len(sidecars) == 1
    assert not list((tmp_path / "store" / "payloads").glob("*.tmp*"))
    np.testing.assert_array_equal(EvidenceStore(tmp_path / "store").get(ev.id).value, arr)

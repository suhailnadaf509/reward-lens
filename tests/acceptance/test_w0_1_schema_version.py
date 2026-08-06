"""Acceptance: an envelope says what version it is, and old ones migrate forward.

The clause this file discharges: *a store written at v1 reads at v2 with the migration applied and
a test asserts the transformed value*, plus one addition, *the campaign store loads and reports
version 0.*

2.0.1 shipped without a schema version, so every envelope ever written by this project is
unversioned. That is not a hypothetical: the campaign evidence store is 1,363 rows and not one of
them carries the key. The sniff is therefore load-bearing and is written exactly once.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest

from reward_lens.core import EvidenceStore, ModelFP, Provenance, SubjectRef, make_evidence
from reward_lens.core.migrations import (
    MIGRATIONS,
    SCHEMA_VERSION,
    MigrationError,
    migrate,
    payload_of,
    register,
    sniff_version,
    with_payload,
)

#: The campaign evidence store's jsonl, which is not in this repository. There is no default:
#: point ``REWARD_LENS_CAMPAIGN_STORE`` at the store directory or the test below skips.
_CAMPAIGN_ENV = os.environ.get("REWARD_LENS_CAMPAIGN_STORE")
_CAMPAIGN = pathlib.Path(_CAMPAIGN_ENV) / "evidence.jsonl" if _CAMPAIGN_ENV else None


def _evidence(value):
    return make_evidence(
        observable="A",
        observable_version="1",
        subject=SubjectRef(signals=(ModelFP("mfp:migrate"),), dataset="ds:m", readout="reward"),
        value=value,
        provenance=Provenance(),
    )


@pytest.fixture()
def synthetic_v2():
    """Register a 1 -> 2 migration for the duration of one test, then take it away again.

    A migration that only stamps a version number cannot demonstrate that the machinery
    transforms anything, so the acceptance test supplies one that does.
    """
    calls: list[str] = []

    @register(1, 2)
    def _double_the_scalar(env):
        calls.append(env["id"])
        out = dict(env)
        if isinstance(out["value"], (int, float)):
            out["value"] = out["value"] * 2
        return out

    yield calls
    MIGRATIONS.pop((1, 2), None)


def test_a_store_written_at_v1_reads_at_v2_with_the_migration_applied(tmp_path, synthetic_v2):
    """The clause."""
    store = EvidenceStore(tmp_path / "store")
    ev = _evidence(0.62)
    store.append(ev)

    on_disk = json.loads((tmp_path / "store" / "evidence.jsonl").read_text().strip())
    assert on_disk["schema_version"] == 1
    assert on_disk["value"] == 0.62

    migrated = migrate(on_disk, to=2)
    assert migrated["schema_version"] == 2
    assert migrated["value"] == pytest.approx(1.24)
    assert synthetic_v2 == [ev.id]

    # The envelope on disk is untouched: migration is a read-time transform, not a rewrite.
    assert json.loads((tmp_path / "store" / "evidence.jsonl").read_text().strip())["value"] == 0.62


def test_an_unversioned_envelope_is_version_zero_and_migrates_to_current():
    ev = _evidence(1.0)
    env = ev.envelope()
    unversioned = {k: v for k, v in env.items() if k != "schema_version"}

    assert sniff_version(unversioned) == 0
    assert sniff_version(migrate(unversioned)) == SCHEMA_VERSION


def test_the_campaign_store_loads_and_reports_version_zero():
    """The addition, against the real store rather than a fixture."""
    if _CAMPAIGN is None or not _CAMPAIGN.exists():
        pytest.skip("no campaign evidence store; set REWARD_LENS_CAMPAIGN_STORE")
    versions = {
        sniff_version(json.loads(line))
        for line in _CAMPAIGN.read_text().splitlines()
        if line.strip()
    }
    assert versions == {0}


def test_the_version_is_not_part_of_the_content_id():
    """Two envelopes of the same measurement at different versions are the same measurement.

    If the version were hashed, the first migration would change every id in the store and break
    every parent reference in the DAG at once.
    """
    a = _evidence(1.0)
    b = make_evidence(
        observable="A",
        observable_version="1",
        subject=SubjectRef(signals=(ModelFP("mfp:migrate"),), dataset="ds:m", readout="reward"),
        value=1.0,
        provenance=Provenance(),
        created_at=a.created_at,
    )
    assert a.id == b.id
    env = a.envelope()
    env["schema_version"] = 99
    assert env["id"] == a.id


def test_a_broken_chain_raises_rather_than_half_migrating(tmp_path):
    """A half-migrated record read as a whole one is the confident wrong answer, in miniature."""
    env = _evidence(1.0).envelope()
    with pytest.raises(MigrationError, match="no migration from schema version"):
        migrate(env, to=SCHEMA_VERSION + 2)


def test_reading_a_newer_store_refuses_instead_of_guessing():
    env = _evidence(1.0).envelope()
    env["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(MigrationError, match="Downgrading is not supported"):
        migrate(env)


def test_migrations_must_step_one_version_at_a_time():
    with pytest.raises(ValueError, match="one version at a time"):
        register(1, 3)(lambda env: env)


def test_a_payload_migration_goes_through_the_codec_not_the_json(tmp_path):
    """The trap the module docstring names, made a test.

    A large array is not in the envelope; the envelope holds a reference to a content-addressed
    sidecar. A migration that edits `env["value"]` as if it were the numbers rewrites the
    reference and leaves the payload exactly where it was.
    """
    store = EvidenceStore(tmp_path / "store")
    arr = np.arange(4096, dtype=np.float64)
    ev = _evidence(arr)
    store.append(ev)

    env = json.loads((tmp_path / "store" / "evidence.jsonl").read_text().strip())
    # Not the numbers: a reference.
    assert "__ndarray__" in env["value"]
    assert "sidecar" in env["value"]["__ndarray__"]

    payload = payload_of(env, store.payloads)
    np.testing.assert_array_equal(payload, arr)

    rewritten = with_payload(env, payload + 1.0, store.payloads)
    np.testing.assert_array_equal(payload_of(rewritten, store.payloads), arr + 1.0)
    # A new content address, because the content changed. The old sidecar is untouched.
    assert rewritten["value"]["__ndarray__"]["sidecar"] != env["value"]["__ndarray__"]["sidecar"]
    np.testing.assert_array_equal(payload_of(env, store.payloads), arr)

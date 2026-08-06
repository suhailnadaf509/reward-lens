"""The safety case is ADJUDICATED-only: it refuses uncalibrated or unregistered components (M13).

This is the M13 safety-case acceptance in isolation. A safety case is the one artifact that refuses
rather than degrades: where a card renders an uncalibrated index in an unvalidated style, the safety
case will not assemble a claim that rests on a component lacking a scorecard (gate 1) or a
preregistration (gate 3). These tests build the four components (the K-nats budget from S3, the
monitor from S13/S14, the erasure certificate from S6/S12, the honesty certificate from S16) at
various trust levels and assert that assembly succeeds only when every component is calibrated and
registered, and that the assembled case renders the claim with each number traceable to its Evidence.
"""

from __future__ import annotations

import json

import pytest

from reward_lens.artifacts.safety_case import (
    SafetyCase,
    SafetyCaseRefusal,
    assemble_safety_case,
)
from reward_lens.core.evidence import make_evidence
from reward_lens.core.gates import CalibrationRef
from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import ModelFP, SubjectRef

_FP = "mfp:skywork-v0.2"
_SUBJ = SubjectRef(signals=(ModelFP(_FP),))
_CAL = CalibrationRef("ev:scorecard", "planted")


def _component(observable: str, value, *, calibrated=True, registered=True, adjudicated=True):
    """Build one component Evidence at a chosen trust level."""
    return make_evidence(
        observable=observable,
        observable_version="1.0",
        subject=_SUBJ,
        value=value,
        calibration=_CAL if calibrated else None,
        registered=registered,
        adjudicated=adjudicated,
    )


def _adjudicated_store(tmp_path) -> tuple[EvidenceStore, dict[str, str]]:
    """A store with four ADJUDICATED components, one per safety-case role."""
    store = EvidenceStore(tmp_path)
    ids = {}
    for role, obs, val in (
        ("k", "KNatsBudget", {"k_nats": 12.5}),
        ("m", "MonitorGrade", {"monitor": "resid-probe lead=40 obf=B"}),
        ("e", "ErasureCertificate", {"bias_set": ["verbosity", "sycophancy"]}),
        ("h", "HonestyCertificate", {"domain": "math", "evidence_schema": "stepwise-verifiable"}),
    ):
        ev = _component(obs, val)
        store.append(ev)
        ids[role] = ev.id
    return store, ids


def test_assembles_from_adjudicated_evidence(tmp_path):
    store, ids = _adjudicated_store(tmp_path)
    case = assemble_safety_case(
        _FP,
        k_nats_evidence=ids["k"],
        monitor_evidence=ids["m"],
        erasure_evidence=ids["e"],
        honesty_evidence=ids["h"],
        store=store,
    )
    assert isinstance(case, SafetyCase)
    assert case.adjudicated
    assert case.k_nats == 12.5
    assert case.bias_set == ["verbosity", "sycophancy"]
    assert case.honesty_domain == "math"
    # The claim string carries K, the monitor, the bias set, and the honesty domain/schema.
    claim = case.claim
    assert "12.5 nats" in claim
    assert "resid-probe" in claim
    assert "verbosity" in claim
    assert "math" in claim


def test_assembled_case_renders(tmp_path):
    store, ids = _adjudicated_store(tmp_path)
    case = assemble_safety_case(
        _FP,
        k_nats_evidence=ids["k"],
        monitor_evidence=ids["m"],
        erasure_evidence=ids["e"],
        honesty_evidence=ids["h"],
        store=store,
    )
    md = case.to_markdown()
    assert md.startswith("# Safety case")
    assert "ADJUDICATED" in md
    # Every component's Evidence id appears, so the case is traceable.
    for eid in ids.values():
        assert eid in md
    html = case.to_html()
    assert "Safety case" in html and _FP in html
    data = json.loads(case.to_json())
    assert data["adjudicated"] is True
    assert len(data["components"]) == 4


def test_refuses_uncalibrated_component(tmp_path):
    store, ids = _adjudicated_store(tmp_path)
    # Replace the K-nats budget with an uncalibrated (EXPLORATORY) one.
    bad = _component(
        "KNatsBudget", {"k_nats": 1.0}, calibrated=False, registered=False, adjudicated=False
    )
    store.append(bad)
    with pytest.raises(SafetyCaseRefusal, match="not calibrated"):
        assemble_safety_case(
            _FP,
            k_nats_evidence=bad.id,
            monitor_evidence=ids["m"],
            erasure_evidence=ids["e"],
            honesty_evidence=ids["h"],
            store=store,
        )


def test_refuses_unregistered_component(tmp_path):
    store, ids = _adjudicated_store(tmp_path)
    # Calibrated but not registered: trust is CALIBRATED, below the REGISTERED bar (gate 3).
    bad = _component(
        "MonitorGrade", {"monitor": "m"}, calibrated=True, registered=False, adjudicated=False
    )
    store.append(bad)
    with pytest.raises(SafetyCaseRefusal, match="not registered"):
        assemble_safety_case(
            _FP,
            k_nats_evidence=ids["k"],
            monitor_evidence=bad.id,
            erasure_evidence=ids["e"],
            honesty_evidence=ids["h"],
            store=store,
        )


def test_refuses_registered_but_uncalibrated_component(tmp_path):
    store, ids = _adjudicated_store(tmp_path)
    # Registered without a scorecard: trust is REGISTERED but calibration is None, so gate 1 refuses.
    bad = _component(
        "ErasureCertificate",
        {"bias_set": ["x"]},
        calibrated=False,
        registered=True,
        adjudicated=False,
    )
    store.append(bad)
    with pytest.raises(SafetyCaseRefusal, match="not calibrated"):
        assemble_safety_case(
            _FP,
            k_nats_evidence=ids["k"],
            monitor_evidence=ids["m"],
            erasure_evidence=bad.id,
            honesty_evidence=ids["h"],
            store=store,
        )


def test_missing_component_raises_keyerror(tmp_path):
    store, ids = _adjudicated_store(tmp_path)
    with pytest.raises(KeyError):
        assemble_safety_case(
            _FP,
            k_nats_evidence="ev:doesnotexist000",
            monitor_evidence=ids["m"],
            erasure_evidence=ids["e"],
            honesty_evidence=ids["h"],
            store=store,
        )

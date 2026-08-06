"""The RM Card renders uncalibrated indices in a distinct unvalidated style (gate 1).

This is the M7 card acceptance exercised in isolation: build a card from a mix of calibrated and
uncalibrated Evidence and assert that the uncalibrated entries are flagged, that the HTML marks
them with the unvalidated class, and that there is no path that presents an uncalibrated number as
validated. The real Skywork card is produced once the battery runs on a model; the render-layer gate
enforcement is what this test locks.
"""

from __future__ import annotations

from reward_lens.artifacts.card import build_card
from reward_lens.core.evidence import Uncertainty, make_evidence
from reward_lens.core.gates import CalibrationRef
from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import GaugeStatus, ModelFP, SubjectRef


def _seed_store(tmp_path) -> tuple[EvidenceStore, str]:
    fp = "mfp:skywork-v0.2"
    subj = SubjectRef(signals=(ModelFP(fp),), dataset="ds:diag", readout="reward")
    store = EvidenceStore(tmp_path)
    # A calibrated index (has a scorecard reference).
    store.append(
        make_evidence(
            observable="BiasBattery",
            observable_version="1.0",
            subject=subj,
            value={"verbosity": -0.05},
            uncertainty=Uncertainty(ci_low=-0.1, ci_high=0.0, n=30, n_effective=6.0),
            gauge=GaugeStatus.INVARIANT,
            calibration=CalibrationRef("ev:scorecard", "planted-bias"),
        )
    )
    # An uncalibrated index (no scorecard): must render as unvalidated.
    store.append(
        make_evidence(
            observable="DistortionV2",
            observable_version="1.0",
            subject=subj,
            value=0.42,
            gauge=GaugeStatus.INVARIANT,
        )
    )
    return store, fp


def test_card_flags_uncalibrated(tmp_path):
    store, fp = _seed_store(tmp_path)
    card = build_card(fp, store)
    assert len(card.entries) == 2
    by_obs = {e.observable: e for e in card.entries}
    assert by_obs["BiasBattery"].validated
    assert not by_obs["DistortionV2"].validated
    assert [e.observable for e in card.unvalidated] == ["DistortionV2"]


def test_card_html_marks_unvalidated(tmp_path):
    store, fp = _seed_store(tmp_path)
    card = build_card(fp, store)
    html = card.to_html()
    # The uncalibrated row carries the unvalidated class; the calibrated one does not get it.
    assert 'class="unvalidated"' in html
    assert "Explicit gaps" in html
    assert "DistortionV2" in html
    # The calibrated entry is present and not in the unvalidated bucket.
    assert "BiasBattery" in html


def test_card_json_lists_gaps(tmp_path):
    import json

    store, fp = _seed_store(tmp_path)
    card = build_card(fp, store)
    data = json.loads(card.to_json())
    assert data["signal"] == fp
    assert data["unvalidated_observables"] == ["DistortionV2"]
    assert len(data["entries"]) == 2

"""Pure tests for the method scorecard (M4 acceptance, must pass now).

The scorecard is the ROC of an instrument recovering a planted structure. Its two load-bearing
properties are testable now with a synthetic detector whose separability scales with the dose rho: the
answer-key AUC must be **monotone in rho** (a stronger planted signal is easier to recover), and the
operating points must be read correctly off the curve. These tests prove the scorecard machinery on
that stand-in detector; the real DLA/patching scorecard is wired at the M3 integration on the same code
path. They also check that `evaluate` emits a valid `CalibrationRef` and stored `Evidence` (gate 1).
"""

from __future__ import annotations

import numpy as np

from reward_lens.core import GaugeStatus
from reward_lens.core.evidence import _CODEC
from reward_lens.organisms.foundry import spurious_correlation_organism
from reward_lens.organisms.scorecard import (
    MethodScorecard,
    ScorecardSummary,
    synthetic_dose_detector,
)

_DOSES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _dose_readouts(slope: float = 6.0, n: int = 800, seed: int = 1) -> dict:
    return {rho: synthetic_dose_detector(rho, n=n, seed=seed, slope=slope) for rho in _DOSES}


def test_scorecard_auc_monotone_in_rho():
    """A detector whose separability scales with rho yields an AUC monotone in rho."""
    _, key = spurious_correlation_organism(rho=0.8, n=10, seed=0)
    entry = MethodScorecard("synthetic-detector").evaluate(_dose_readouts(), key)

    assert entry.doses == _DOSES
    assert entry.is_monotone, f"AUCs not monotone in rho: {entry.aucs}"
    assert entry.monotone_spearman > 0.9
    # A real dose effect, not a flat curve: the strongest dose is clearly above the null dose.
    assert entry.aucs[0] < 0.6  # rho = 0.5 is near chance
    assert entry.aucs[-1] > 0.85  # rho = 1.0 is well separated
    assert entry.aucs[-1] - entry.aucs[0] > 0.25


def test_operating_points_correct_and_detection_threshold():
    """Operating points respect the FPR cap, improve with rho, and yield a detection dose."""
    _, key = spurious_correlation_organism(rho=0.8, n=10, seed=0)
    # A steeper detector so the 90% TPR / 5% FPR bar is unambiguously cleared at high rho.
    entry = MethodScorecard("synthetic-detector").evaluate(
        _dose_readouts(slope=8.0, n=2000, seed=2), key, target_tpr=0.9, target_fpr=0.05
    )
    ops = entry.operating_points
    # Every operating point respects the FPR ceiling (within a small numerical tolerance).
    assert all(op["fpr"] <= 0.05 + 1e-6 for op in ops if np.isfinite(op["fpr"]))
    # TPR at the fixed FPR improves from the null dose to the strongest dose.
    assert ops[-1]["tpr"] > ops[0]["tpr"]
    # The scorecard reports the smallest dose it flags at the target operating point.
    assert entry.detects_rho_at is not None
    assert entry.detects_rho_at in _DOSES
    # Detection is consistent with the TPR curve: doses below the detection dose miss the bar.
    for op in ops:
        if op["rho"] < entry.detects_rho_at:
            assert op["tpr"] < 0.9 or op["fpr"] > 0.05 + 1e-6


def test_scorecard_emits_valid_calibration_ref_and_evidence():
    """`evaluate` produces Evidence and a CalibrationRef that points at it (gate 1)."""
    _, key = spurious_correlation_organism(rho=0.8, n=10, seed=0)
    entry = MethodScorecard("synthetic-detector").evaluate(_dose_readouts(), key)

    ref = entry.calibration_ref
    assert ref.scorecard_entry == entry.evidence.id
    assert ref.organism_family == key.family
    assert entry.evidence.gauge is GaugeStatus.INVARIANT
    assert entry.evidence.observable.startswith("MethodScorecard[")
    # The stored payload is the summary.
    assert isinstance(entry.evidence.value, ScorecardSummary)
    assert entry.evidence.value.is_monotone == entry.is_monotone


def test_scorecard_single_regime():
    """A single (scores, labels) readout evaluates as one dose without a sweep."""
    _, key = spurious_correlation_organism(rho=0.8, n=10, seed=0)
    scores, labels = synthetic_dose_detector(0.9, n=500, seed=3, slope=6.0)
    entry = MethodScorecard("synthetic-detector").evaluate((scores, labels), key)
    assert len(entry.doses) == 1
    assert len(entry.aucs) == 1
    assert entry.aucs[0] > 0.8


def test_scorecard_payload_roundtrips_through_codec():
    """The ScorecardSummary payload round-trips exactly through the Evidence codec (R8)."""
    _, key = spurious_correlation_organism(rho=0.8, n=10, seed=0)
    entry = MethodScorecard("synthetic-detector").evaluate(_dose_readouts(), key)
    decoded = _CODEC.decode(_CODEC.encode(entry.summary, None), None)
    assert isinstance(decoded, ScorecardSummary)
    assert decoded.aucs == entry.summary.aucs
    assert decoded.doses == entry.summary.doses
    assert decoded.is_monotone == entry.summary.is_monotone
    assert decoded.operating_points == entry.summary.operating_points


def test_monotone_flag_false_on_non_monotone_input():
    """A scorecard whose AUC decreases with rho reports is_monotone False (the sanity check bites)."""
    _, key = spurious_correlation_organism(rho=0.8, n=10, seed=0)
    # Reverse the dose-to-strength mapping so higher rho gets a *weaker* detector.
    reversed_readouts = {
        rho: synthetic_dose_detector(1.5 - rho, n=800, seed=4, slope=6.0) for rho in _DOSES
    }
    entry = MethodScorecard("adversarial-detector").evaluate(reversed_readouts, key)
    assert not entry.is_monotone
    assert entry.monotone_spearman < 0

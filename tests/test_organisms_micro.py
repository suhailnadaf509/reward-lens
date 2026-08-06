"""The M4 acceptance test: the CI micro-organism (R6, torch-gated).

This is the go/no-go for the whole design. It trains a planted rule into the tiny CPU
trunk, verifies the rule governs behaviour out of distribution, and runs the built-in linear detector
to recover the rule on the held-out OOD split. If the detector cannot recover the planted rule on a
system whose ground truth is known by construction, then no scorecard on a production signal can be
trusted, and CI must fail. That failure mode is exactly what the recovery assertion here enforces.

The test skips with a clear message when torch is not importable (the pure foundry and scorecard tests
carry M4's non-torch acceptance); it must pass once torch is present.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip(
    "torch",
    reason="torch not importable yet; the micro-organism CI hook (R6) needs it. The pure foundry and "
    "scorecard tests carry M4's non-torch acceptance in the meantime.",
)


def test_micro_organism_calibration_recovers_planted_rule():
    """Train, verify OOD rule-governance, and recover the planted rule above threshold (M4 accept).

    CI fails if the built-in detector cannot recover the micro-organism rule: ``recovered`` is the
    single go/no-go bit, and it is ``True`` only when the mean-difference direction fit on the training
    split separates chosen from rejected on the unseen OOD split above the recovery threshold.
    """
    from reward_lens.organisms.micro import micro_organism_calibration

    result = micro_organism_calibration(
        seed=0, rho=0.85, n_train=140, n_ood=140, detector_threshold=0.75, verify_threshold=0.9
    )

    # (b) the tiny trunk actually learned the planted rule in-sample.
    assert result.trained.train_accuracy >= 0.9, result.summary()

    # (c) the rule provably governs behaviour out of distribution (verify accepted it).
    assert result.verification.accepted, result.verification.reason
    assert result.answer_key.governs_behavior_oob is True

    # (d) the built-in detector recovers the planted rule on the OOD split above threshold.
    assert result.detector.recovered, (
        f"CI GATE: the detector failed to recover the micro-organism rule "
        f"(OOD AUC {result.detector.ood_auc:.3f} < {result.detector.threshold}). This "
        f"is the go/no-go signal; a red detector here means the calibration floor is unbuildable."
    )
    assert result.detector.ood_auc >= 0.75

    # The recovered direction aligns with the trained reward head, a mechanistic cross-check.
    assert result.detector.cosine_with_reward > 0.3, result.summary()


def test_micro_organism_scorecard_from_real_detector():
    """The trained detector's OOD readout scores through the real MethodScorecard (gate 1 wiring)."""
    import numpy as np

    from reward_lens.organisms.micro import micro_organism_calibration
    from reward_lens.organisms.scorecard import MethodScorecard

    result = micro_organism_calibration(seed=1, rho=0.9, n_train=120, n_ood=120)

    # Build a (scores, labels) readout from the detector direction on the OOD split and score it.
    from reward_lens.organisms.foundry import spurious_correlation_organism
    from reward_lens.organisms.micro import _capture_final_token

    ood_data, _ = spurious_correlation_organism(rho=0.9, n=120, seed=1, split="ood")
    signal = result.trained.signal
    layer = result.detector.layer
    chosen = [f"{p.prompt_text} {p.chosen.text}".strip() for p in ood_data]
    rejected = [f"{p.prompt_text} {p.rejected.text}".strip() for p in ood_data]
    hc = _capture_final_token(signal, chosen, layer, 64) @ result.detector.direction
    hr = _capture_final_token(signal, rejected, layer, 64) @ result.detector.direction
    scores = np.concatenate([hc, hr])
    labels = np.concatenate([np.ones(len(hc)), np.zeros(len(hr))])

    entry = MethodScorecard("mean-difference-detector").evaluate(
        (scores, labels), result.answer_key
    )
    assert entry.aucs[0] >= 0.75
    assert entry.calibration_ref.scorecard_entry == entry.evidence.id

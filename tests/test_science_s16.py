"""S16 runs end to end as a frozen study, recovering a planted eval-awareness direction.

The calibration arm builds synthetic activations with a planted benchmark-vs-organic direction and a
planted reward loading on it, then shows the probe recovers the separability above chance and that
steering along the recovered direction inflates the reward. Certified radii and the production
eval-awareness read ride in-flight subsystems and a real reward population, so they are recorded as
honestly gated (inconclusive) follow-ons.
"""

from __future__ import annotations

from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import TrustLevel
from reward_lens.studies import render_report, run_study
from studies.s16_robustness.analysis import build_spec


def test_s16_recovers_eval_awareness_and_inflation(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    # The eval-awareness direction is decodable above chance and steering it inflates the reward.
    assert result.outcomes["H1-probe-above-chance"] == "confirmed", result.metrics
    assert result.outcomes["H2-recognition-inflates"] == "confirmed", result.metrics
    assert not result.killed
    assert result.metrics["probe_balanced_acc"] > 0.6
    assert result.metrics["delta_r_per_steer"] > 0.05
    # The probe recovers the planted separability, not merely something above chance.
    assert result.metrics["separability_recovery_error"] < 0.1

    # The headline Evidence is REGISTERED and traces back to the probe (a real DAG).
    inflation = store.find(observable="S16.RecognitionInflation")
    assert inflation and inflation[0].trust is TrustLevel.REGISTERED
    parents = store.parents(inflation[0])
    assert parents and parents[0].observable == "S16.EvalAwarenessProbe"


def test_s16_gates_certified_radius_and_production_read(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    # Neither the certified radius nor the production read is fabricated; both are void.
    assert result.outcomes["H3-certified-radius"] == "void"
    assert result.outcomes["H4-production-eval-awareness"] == "void"
    assert "certified_radius" not in result.metrics
    assert "real_probe_balanced_acc" not in result.metrics

    certify = store.find(observable="S16.CertifiedRadiusGate")
    assert certify and certify[0].trust is TrustLevel.REGISTERED
    assert "reward_lens.interventions" in certify[0].value["need"]

    real = store.find(observable="S16.ProductionEvalAwarenessGate")
    assert real and real[0].value["status"] == "gated"
    assert "reward_lens.measure.indices" in real[0].value["need"]

    report = render_report(frozen, result, store)
    assert "CONFIRMED" in report
    assert "VOID" in report

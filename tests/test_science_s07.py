"""S7 runs end to end as a frozen study, recovering a planted feature-entry order.

The calibration arm builds a synthetic checkpoint sequence whose feature-entry order is planted by
construction and shows the entry-order instrument recovers it, so the bias-before-quality reading is
calibrated before it would be turned on a real training run. The real Pythia-RM checkpoint sweep
rides an in-flight subsystem and a GPU run, so it is recorded as an honestly gated (inconclusive)
follow-on rather than fabricated.
"""

from __future__ import annotations

from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import TrustLevel
from reward_lens.studies import render_report, run_study
from studies.s07_embryology.analysis import build_spec


def test_s07_recovers_planted_entry_order(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    # The entry-order instrument recovers the plant, and surface biases lead quality features.
    assert result.outcomes["H1-order-recovery"] == "confirmed", result.metrics
    assert result.outcomes["H2-bias-before-quality"] == "confirmed", result.metrics
    assert not result.killed
    assert result.metrics["order_recovery"] > 0.9
    assert result.metrics["bias_before_quality"] > 1.0
    assert result.metrics["quality_mean_entry"] > result.metrics["bias_mean_entry"]

    # The headline Evidence is REGISTERED and traces back to the measured separations (a real DAG).
    order = store.find(observable="S07.EntryOrder")
    assert order and order[0].trust is TrustLevel.REGISTERED
    parents = store.parents(order[0])
    assert parents and parents[0].observable == "S07.CheckpointSeparations"


def test_s07_gates_the_real_sweep_honestly(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    # The real Pythia sweep is not fabricated: its metric is absent, so the runner voids it
    # rather than adjudicating it, and a REGISTERED gate Evidence states what the arm needs.
    assert result.outcomes["H3-real-pythia-sweep"] == "void"
    assert "real_bias_before_quality" not in result.metrics

    gate = store.find(observable="S07.RealPythiaSweepGate")
    assert gate and gate[0].trust is TrustLevel.REGISTERED
    assert gate[0].value["status"] == "gated"
    # The dynamics bias-entry curve is calibrated and used here; what is gated is the GPU-scale real
    # RM-Pythia training run its calibrated curve would then sweep.
    need = gate[0].value["need"]
    assert "reward_lens.dynamics" in need
    assert "train_rm_pythia" in need and "GPU" in need

    report = render_report(frozen, result, store)
    assert "CONFIRMED" in report
    assert "VOID" in report

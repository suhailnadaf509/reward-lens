"""The two Atlas meta-studies run end to end, calibrating VCE (T13) and the audit half-life (T11).

Universality plants a shared value subspace and shows the sign of value convergence excess is
recovered (positive when values converge beyond world-modeling), updating scoreboard row T13.
Performativity plants a bias dial on an organism where truth is known forever and shows a causally
grounded metric outlasts an observational one under developer optimization, updating row T11. Both
studies' real-population arms are recorded as honestly gated (inconclusive) follow-ons.
"""

from __future__ import annotations

from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import TrustLevel
from reward_lens.studies import Scoreboard, render_report, run_study
from studies.atlas_meta.performative import build_spec as build_performative_spec
from studies.atlas_meta.universality import build_spec as build_universality_spec


def test_universality_calibrates_vce_sign(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_universality_spec(), store=store)

    # Positive VCE on the convergent pair, a clear margin over the null pair, beating the RUM null.
    assert result.outcomes["H1-vce-positive"] == "confirmed", result.metrics
    assert result.outcomes["H2-vce-sign-separates"] == "confirmed", result.metrics
    assert result.outcomes["H3-beats-rum-null"] == "confirmed", result.metrics
    assert result.outcomes["H4-real-rm-pair"] == "void"
    assert not result.killed
    assert result.metrics["vce_convergent"] > 0.05
    assert result.metrics["vce_convergent"] > result.metrics["vce_null"]

    # The VCE Evidence is REGISTERED and traces back to the subspace alignments (a real DAG).
    vce = store.find(observable="AT.ValueConvergenceExcess")
    assert vce and vce[0].trust is TrustLevel.REGISTERED
    parents = store.parents(vce[0])
    assert parents and parents[0].observable == "AT.SubspaceAlignments"


def test_universality_updates_t13(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_universality_spec(), store=store)

    board = Scoreboard(tmp_path / "scoreboard.json")
    board.update_from_result(frozen.study_id, frozen.spec.hypotheses, result)
    assert board.rows["T13"].status == "confirmed"
    assert board.rows["T13"].adjudicating_evidence  # cites the evidence that settled it


def test_performative_calibrates_half_life_ordering(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_performative_spec(), store=store)

    # The causal metric outlasts the observational one, whose correlation with truth halves in time.
    assert result.outcomes["H1-halflife-ordering"] == "confirmed", result.metrics
    assert result.outcomes["H2-observational-decays"] == "confirmed", result.metrics
    assert result.outcomes["H3-real-audit-loop"] == "void"
    assert not result.killed
    assert result.metrics["half_life_gap"] > 5.0
    assert result.metrics["half_life_causal"] > result.metrics["half_life_obs"]

    half = store.find(observable="AT.AuditHalfLife")
    assert half and half[0].trust is TrustLevel.REGISTERED
    parents = store.parents(half[0])
    assert parents and parents[0].observable == "AT.AuditTrajectories"


def test_performative_updates_t11_and_gates(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_performative_spec(), store=store)

    board = Scoreboard(tmp_path / "scoreboard.json")
    board.update_from_result(frozen.study_id, frozen.spec.hypotheses, result)
    assert board.rows["T11"].status == "confirmed"

    gate = store.find(observable="AT.RealAuditLoopGate")
    assert gate and gate[0].value["status"] == "gated"
    assert "real base population" in gate[0].value["need"]

    report = render_report(frozen, result, store)
    assert "CONFIRMED" in report
    assert "VOID" in report

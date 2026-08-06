"""S2 runs end to end as a frozen study, emitting REGISTERED Evidence and updating T6 and T8.

This is one of milestone M6's three cheap sciences, exercised on controlled inputs where the answer
is known by construction. It proves the worked trace: a science is a frozen
spec plus a thin analysis function over the kernel (here geometry), producing REGISTERED Evidence
whose headline metrics name the geometry Evidence they were derived from, and folding its outcomes
into the theorem scoreboard.
"""

from __future__ import annotations

from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import TrustLevel
from reward_lens.studies import Scoreboard, render_report, run_study
from studies.s02_gauge.analysis import build_spec


def test_s02_runs_and_registers(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    # Both experiments confirmed on the controlled construction.
    assert result.outcomes["H1-canonical-stability"] == "confirmed", result.metrics
    assert result.outcomes["H2-cyclic-recovery"] == "confirmed", result.metrics
    assert not result.killed
    # The canonical cosine saw through the gauge; the raw cosine did not.
    assert result.metrics["canonical_cos"] > 0.9
    assert result.metrics["canonical_minus_raw"] > 0.4
    assert result.metrics["cyclic_recovery"] > 0.1

    # The study's headline Evidence is REGISTERED and traces to its geometry parent (a real DAG).
    stab = store.find(observable="S02.GaugeStability")
    assert stab and stab[0].trust is TrustLevel.REGISTERED
    parents = store.parents(stab[0])
    assert parents and parents[0].observable == "geometry.effective_angle"

    rank = store.find(observable="S02.PreferenceRank")
    assert rank and rank[0].trust is TrustLevel.REGISTERED


def test_s02_updates_scoreboard(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    board = Scoreboard(tmp_path / "scoreboard.json")
    board.update_from_result(frozen.study_id, frozen.spec.hypotheses, result)
    assert board.rows["T6"].status == "confirmed"
    assert board.rows["T8"].status == "confirmed"
    assert board.rows["T6"].adjudicating_evidence  # cites the evidence that settled it

    report = render_report(frozen, result, store)
    assert "CONFIRMED" in report
    assert frozen.study_id in report

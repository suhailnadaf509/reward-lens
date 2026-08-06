"""S11 runs end to end as a frozen study, emitting REGISTERED Evidence and updating T7.

This is one of milestone M6's cheap sciences. The contested-direction probe is calibrated on a real
tiny ClassifierRM's reward direction with a planted orthogonal contested direction, so the answer is
known by construction (gate 1): the probe must decode disagreement above chance and report
the direction as orthogonal to w_r. The verdict-before-critique probe needs a real reasoning judge, so
it is recorded as inconclusive-because-gated rather than failed. The study folds its confirmatory
outcomes into the theorem scoreboard at T7.
"""

from __future__ import annotations

from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import TrustLevel
from reward_lens.studies import Scoreboard, render_report, run_study
from studies.s11_values.analysis import build_spec


def test_s11_runs_and_registers(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    # The contested-direction calibration confirms both arms; the study is not killed.
    assert result.outcomes["H1-contested-decodes"] == "confirmed", result.metrics
    assert result.outcomes["H2-contested-orthogonal"] == "confirmed", result.metrics
    assert not result.killed, result.killed_by

    # The probe decoded the planted contested label well above chance, and the recovered direction is
    # largely orthogonal to the reward direction w_r.
    assert result.metrics["contested_probe_bal_acc"] > 0.6
    assert result.metrics["contested_reward_cos_abs"] < 0.3

    # The contested-direction Evidence is REGISTERED.
    contested = store.find(observable="S11.ContestedDirection")
    assert contested and contested[0].trust is TrustLevel.REGISTERED

    # The verdict-before-critique arm is gated: it emits no adjudicated metric, so the runner
    # voids it, and its Evidence records the gate explicitly.
    assert result.outcomes["H3-verdict-before-critique"] == "void"
    assert "verdict_prefix_match_rate" not in result.metrics
    verdict = store.find(observable="S11.VerdictBeforeCritique")
    assert verdict and verdict[0].trust is TrustLevel.REGISTERED
    assert verdict[0].value["gated"] is True


def test_s11_updates_scoreboard(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    board = Scoreboard(tmp_path / "scoreboard.json")
    board.update_from_result(frozen.study_id, frozen.spec.hypotheses, result)
    # H1/H2 confirm T7; the gated H3 is inconclusive and does not move the row.
    assert board.rows["T7"].status == "confirmed"
    assert board.rows["T7"].adjudicating_evidence

    report = render_report(frozen, result, store)
    assert "CONFIRMED" in report
    assert frozen.study_id in report

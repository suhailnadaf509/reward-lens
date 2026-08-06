"""S3 runs end to end as a frozen study, emitting REGISTERED Evidence and updating T9.

This is one of milestone M6's cheap sciences, exercised on a synthetic-but-honest base-policy draw
where the susceptibility is analytically known by construction (gate 1). It proves the
worked trace: the strongest program in the corpus is a frozen spec plus a thin analysis
function over the kernel (here the loops susceptibility and best-of-n arms), producing REGISTERED
Evidence and folding its outcomes into the theorem scoreboard at T9.
"""

from __future__ import annotations

from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import TrustLevel
from reward_lens.studies import Scoreboard, render_report, run_study
from studies.s03_thermo.analysis import build_spec


def test_s03_runs_and_registers(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    # H1 (chi calibration) confirmed; the study is not killed by its own kill criterion.
    assert result.outcomes["H1-chi-recovery"] == "confirmed", result.metrics
    assert not result.killed, result.killed_by

    # The susceptibility spectrum recovered the planted structure, and the f = r diagonal recovered
    # the teacher variance w_r^T Sigma_pi w_r.
    assert result.metrics["chi_recovery_corr"] > 0.98
    assert result.metrics["teacher_variance_rel_error"] < 0.05
    assert result.metrics["hack_mode_match"] == 1.0

    # The inline Hill estimator recovered a planted Pareto tail exponent within tolerance.
    assert result.metrics["pareto_hill_rel_error"] < 0.2

    # The best-of-n transfer arm consumes loops.bon, which is importable in this repo, so it runs and
    # its rank agreement clears the registered threshold (and H2 confirms).
    assert "chi_bon_spearman" in result.metrics, "loops.bon arm should have run"
    assert result.metrics["chi_bon_spearman"] > 0.3
    assert result.outcomes["H2-bon-transfer"] == "confirmed", result.metrics

    # The study's headline Evidence is REGISTERED and descends from the base-policy draw (a real DAG).
    sus = store.find(observable="S03.Susceptibility")
    assert sus and sus[0].trust is TrustLevel.REGISTERED
    assert store.parents(sus[0]), "susceptibility Evidence should cite a parent"

    draw = store.find(observable="S03.BasePolicyDraw")
    assert draw and draw[0].trust is TrustLevel.REGISTERED

    tail = store.find(observable="S03.TailMetrology")
    assert tail and tail[0].trust is TrustLevel.REGISTERED

    transfer = store.find(observable="S03.BoNTransfer")
    assert transfer and transfer[0].trust is TrustLevel.REGISTERED


def test_s03_updates_scoreboard(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    board = Scoreboard(tmp_path / "scoreboard.json")
    board.update_from_result(frozen.study_id, frozen.spec.hypotheses, result)
    assert board.rows["T9"].status == "confirmed"
    assert board.rows["T9"].adjudicating_evidence  # cites the evidence that settled it

    report = render_report(frozen, result, store)
    assert "CONFIRMED" in report
    assert frozen.study_id in report

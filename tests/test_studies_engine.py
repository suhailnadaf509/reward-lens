"""M6 engine tests: freeze, run, adjudicate, report, scoreboard.

These exercise the studies engine end to end on a minimal analysis, with no model: freezing a spec
produces a stable content-derived StudyID and records the git sha; running it emits REGISTERED
Evidence into the store; the runner adjudicates the hypotheses against the FROZEN predictions (not
whatever the analysis claims); a fired kill criterion is recorded; the report renders; and the
scoreboard folds the outcome into the addressed theorem row. The three real cheap sciences (S2, S3,
S11) plug their analysis functions into exactly this engine once the battery and geometry land.
"""

from __future__ import annotations

from reward_lens.core.types import Capability, GaugeStatus, TrustLevel
from reward_lens.measure.base import BaseObservable
from reward_lens.studies import (
    Hypothesis,
    KillCriterion,
    Prediction,
    Scoreboard,
    StudyResult,
    StudySpec,
    SubjectQuery,
    freeze,
    render_report,
    run_study,
)


class _Meta:
    fingerprint = "mfp:study-test"


class _FakeSignal:
    caps = Capability.SCORES | Capability.ACTIVATIONS
    meta = _Meta()


class _MeanReward(BaseObservable):
    """A trivial Observable that emits a fixed effect, standing in for a real battery observable."""

    name = "MeanReward"
    gauge_status = GaugeStatus.INVARIANT

    def measure(self, ctx):
        return ctx.emit(0.62, subject_extra={"note": "smoke"})


def analysis_confirms(run) -> StudyResult:
    """Analysis that measures a mean and reports it as the tested metric.

    It runs the observable under the study (so the Evidence is REGISTERED), then returns the metric
    the frozen prediction will be checked against. It deliberately does NOT set outcomes; the runner
    adjudicates against the frozen predictions.
    """
    signal = run.signal("primary")
    ev = run.measure(_MeanReward(), signal)
    return StudyResult(
        outcomes={}, metrics={"mean_reward": float(ev.value)}, summary="smoke analysis"
    )


def _spec(threshold: float = 0.3, kill_threshold: float = 0.9) -> StudySpec:
    return StudySpec(
        id="smoke-thermo",
        title="Smoke study: mean reward exceeds threshold",
        science="S03-thermo",
        hypotheses=(
            Hypothesis(
                id="H1",
                statement="mean reward exceeds the registered threshold",
                prediction=Prediction(metric="mean_reward", comparator=">", threshold=threshold),
                scoreboard_row="T9",
            ),
        ),
        analysis="tests.test_studies_engine.analysis_confirms",
        subjects=SubjectQuery(signals=("mfp:study-test",)),
        kill_criteria=(
            KillCriterion(
                id="K1",
                metric="mean_reward",
                comparator=">",
                threshold=kill_threshold,
                description="reward implausibly high, likely a scoring bug",
            ),
        ),
    )


def test_freeze_is_stable_and_versioned():
    a = freeze(_spec(0.3))
    b = freeze(_spec(0.3))
    assert a.study_id == b.study_id
    assert a.study_id.startswith("study:smoke-thermo@v1#")
    # A different registered prediction threshold yields a different frozen id (I4).
    c = freeze(_spec(0.5))
    assert c.study_id != a.study_id


def test_run_emits_registered_evidence_and_confirms(tmp_path):
    from reward_lens.core.store import EvidenceStore

    store = EvidenceStore(tmp_path)
    frozen, result = run_study(
        _spec(threshold=0.3),
        subjects={"primary": _FakeSignal()},
        store=store,
    )
    # The measured evidence is REGISTERED (gate 3) and lives in the store.
    assert len(result.evidence) == 1
    ev = store.get(result.evidence[0])
    assert ev.trust is TrustLevel.REGISTERED
    assert ev.provenance.study == frozen.study_id
    # 0.62 > 0.3 so H1 confirms; 0.62 < 0.9 so the kill criterion does not fire.
    assert result.outcomes["H1"] == "confirmed"
    assert not result.killed


def test_prediction_can_refute(tmp_path):
    from reward_lens.core.store import EvidenceStore

    store = EvidenceStore(tmp_path)
    # Register a prediction the data will not meet: mean_reward > 0.8 (actual is 0.62).
    _, result = run_study(
        _spec(threshold=0.8),
        subjects={"primary": _FakeSignal()},
        store=store,
    )
    assert result.outcomes["H1"] == "refuted"


def test_kill_criterion_fires(tmp_path):
    from reward_lens.core.store import EvidenceStore

    store = EvidenceStore(tmp_path)
    # Kill if mean_reward > 0.5; actual 0.62 fires it.
    _, result = run_study(
        _spec(threshold=0.3, kill_threshold=0.5),
        subjects={"primary": _FakeSignal()},
        store=store,
    )
    assert result.killed
    assert "K1" in result.killed_by


def test_report_and_scoreboard(tmp_path):
    from reward_lens.core.store import EvidenceStore

    store = EvidenceStore(tmp_path)
    frozen, result = run_study(
        _spec(threshold=0.3),
        subjects={"primary": _FakeSignal()},
        store=store,
    )
    report = render_report(frozen, result, store)
    assert "CONFIRMED" in report
    assert frozen.study_id in report
    assert "MeanReward" in report

    board = Scoreboard(tmp_path / "scoreboard.json")
    board.update_from_result(frozen.study_id, frozen.spec.hypotheses, result)
    assert board.rows["T9"].status == "confirmed"
    assert result.evidence[0] in board.rows["T9"].adjudicating_evidence
    md = board.render_markdown()
    # Confirmed and refuted statuses render uppercased so a refutation is as prominent as a
    # confirmation (I4); open and mixed stay lowercase.
    assert "T9" in md and "CONFIRMED" in md
    # Persisted and reloadable (composes across studies).
    board2 = Scoreboard(tmp_path / "scoreboard.json")
    assert board2.rows["T9"].status == "confirmed"

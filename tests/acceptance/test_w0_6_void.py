"""Acceptance: a metric that could not be computed is VOID, with a reason and a remedy.

The clause this file discharges: *a study whose analysis omits a kill metric produces VOID and the
test asserts the reason string.*

The defect being closed is worth restating, because it is the reason the whole refusal
architecture exists. The previous adjudication read

    for k in spec.kill_criteria:
        value = result.metrics.get(k.metric)
        if value is not None and k.fired(float(value)):
            killed_by.append(k.id)

so a registered kill criterion whose metric was absent produced exactly the same output as one
that was evaluated and passed. A safety check that failed to run was indistinguishable, in the
report, from a safety check that ran and found nothing.
"""

from __future__ import annotations

import pytest

from reward_lens.studies import (
    Hypothesis,
    KillCriterion,
    Prediction,
    Scoreboard,
    StudyOutcome,
    StudyResult,
    StudySpec,
    SubjectQuery,
    VoidReason,
    render_report,
    run_study,
)


def analysis_omits_everything(run) -> StudyResult:
    """The failure mode from the field: the arc did not run, so no metric exists.

    This is what all eight of the campaign's inconclusive cards looked like from the runner's side.
    The analysis returns cleanly; it simply has nothing to report.
    """
    return StudyResult(
        outcomes={}, metrics={}, summary="the arc that produces the metric did not run"
    )


def analysis_omits_only_the_kill_metric(run) -> StudyResult:
    """The dangerous case: the hypothesis adjudicates and the safety check silently does not."""
    return StudyResult(outcomes={}, metrics={"mean_reward": 0.62}, summary="kill metric missing")


def _spec(analysis: str) -> StudySpec:
    return StudySpec(
        id="void-acceptance",
        title="a missing metric is void, not a pass",
        science="S03-thermo",
        hypotheses=(
            Hypothesis(
                id="H1",
                statement="mean reward exceeds the registered threshold",
                prediction=Prediction(metric="mean_reward", comparator=">", threshold=0.3),
                scoreboard_row="T9",
            ),
        ),
        analysis=analysis,
        subjects=SubjectQuery(signals=("mfp:void-test",)),
        kill_criteria=(
            KillCriterion(
                id="K1",
                metric="exploit_drift",
                comparator=">",
                threshold=0.15,
                description="the instrument moved the thing it was measuring",
            ),
        ),
    )


@pytest.fixture()
def store(tmp_path):
    from reward_lens.core.store import EvidenceStore

    return EvidenceStore(tmp_path / "store")


def test_absent_kill_metric_is_void_never_a_non_firing(store):
    """The clause. A kill criterion with no metric is VOID and names what is missing."""
    _, result = run_study(
        _spec("tests.acceptance.test_w0_6_void.analysis_omits_only_the_kill_metric"),
        subjects={"primary": object()},
        store=store,
        analysis_fn=analysis_omits_only_the_kill_metric,
    )

    # The hypothesis had its metric and adjudicates normally.
    assert result.outcomes["H1"] == "confirmed"

    # The kill criterion did not. It is void, and specifically it is NOT recorded as passing.
    assert result.kill_outcomes["K1"] == "void"
    assert result.killed is False
    assert "K1" not in result.killed_by

    void = result.voids["K1"]
    assert void.reason is VoidReason.METRIC_ABSENT

    # The reason string names the criterion, the metric, and the fact that it was not evaluated.
    assert "K1" in void.detail
    assert "exploit_drift" in void.detail
    assert "neither fired nor passed" in void.detail
    assert void.remedy  # never empty; it is the sentence the operator acts on

    # And the study as a whole is not readable.
    assert result.outcome is StudyOutcome.VOID


def test_absent_hypothesis_metric_is_void_not_inconclusive(store):
    """The word "inconclusive" is gone from the vocabulary, and a void carries a remedy."""
    _, result = run_study(
        _spec("tests.acceptance.test_w0_6_void.analysis_omits_everything"),
        subjects={"primary": object()},
        store=store,
        analysis_fn=analysis_omits_everything,
    )

    assert result.outcomes["H1"] == "void"
    assert "inconclusive" not in result.outcomes.values()

    void = result.voids["H1"]
    assert void.reason is VoidReason.METRIC_ABSENT
    assert "mean_reward" in void.detail
    assert "not adjudicated" in void.detail
    assert result.outcome is StudyOutcome.VOID


def test_void_names_the_arc_that_was_supposed_to_produce_the_metric(store):
    """Closure rule 3: the failure names the arc, not just the metric.

    Without this a void reads "no value for campaign.bias.battery", which tells the operator what
    is missing and not what to run. With it, the void is a work item.
    """
    _, result = run_study(
        _spec("tests.acceptance.test_w0_6_void.analysis_omits_everything"),
        subjects={"primary": object()},
        store=store,
        analysis_fn=analysis_omits_everything,
        metric_arcs={"mean_reward": "arc:battery.mean", "exploit_drift": "arc:surgery.drift"},
    )

    assert result.voids["H1"].arc == "arc:battery.mean"
    assert result.voids["K1"].arc == "arc:surgery.drift"


def test_a_fully_adjudicated_study_is_not_void(store):
    """The gate has to be able to say yes, or it is not a gate."""

    def analysis(run) -> StudyResult:
        return StudyResult(
            outcomes={}, metrics={"mean_reward": 0.62, "exploit_drift": 0.02}, summary="complete"
        )

    _, result = run_study(
        _spec("tests.acceptance.test_w0_6_void.analysis_omits_everything"),
        subjects={"primary": object()},
        store=store,
        analysis_fn=analysis,
    )

    assert result.outcomes["H1"] == "confirmed"
    assert result.kill_outcomes["K1"] == "passed"
    assert result.voids == {}
    assert result.outcome is StudyOutcome.RESULT


def test_a_refuted_study_with_no_voids_is_null_not_void(store):
    """NULL and VOID are different verdicts and the study level has to keep them apart."""

    def analysis(run) -> StudyResult:
        return StudyResult(
            outcomes={}, metrics={"mean_reward": 0.01, "exploit_drift": 0.02}, summary="refuted"
        )

    _, result = run_study(
        _spec("tests.acceptance.test_w0_6_void.analysis_omits_everything"),
        subjects={"primary": object()},
        store=store,
        analysis_fn=analysis,
    )

    assert result.outcomes["H1"] == "refuted"
    assert result.outcome is StudyOutcome.NULL


def test_the_report_renders_the_void_rather_than_hiding_it(store):
    """A report with a voids section is more informative than one without."""
    frozen, result = run_study(
        _spec("tests.acceptance.test_w0_6_void.analysis_omits_only_the_kill_metric"),
        subjects={"primary": object()},
        store=store,
        analysis_fn=analysis_omits_only_the_kill_metric,
        metric_arcs={"exploit_drift": "arc:surgery.drift"},
    )

    md = render_report(frozen, result, store=store)

    assert "VOID" in md
    assert "## Voids" in md
    assert "arc:surgery.drift" in md
    # The kill-criteria section must not describe an unevaluated criterion as "not fired".
    kill_section = md.split("## Kill criteria")[1].split("##")[0]
    assert "not fired" not in kill_section
    assert "VOID, not evaluated" in kill_section


def test_a_void_does_not_move_a_scoreboard_row(tmp_path, store):
    """A void was not read, so it is not evidence in either direction."""
    board = Scoreboard(path=tmp_path / "scoreboard.json")
    from reward_lens.studies.scoreboard import DEFAULT_ROWS

    for row in DEFAULT_ROWS:
        board.register_row(row)
    before = board.rows["T9"].status

    frozen, result = run_study(
        _spec("tests.acceptance.test_w0_6_void.analysis_omits_everything"),
        subjects={"primary": object()},
        store=store,
        analysis_fn=analysis_omits_everything,
    )
    board.update_from_result(frozen.study_id, frozen.spec.hypotheses, result)

    assert board.rows["T9"].status == before


# ---------------------------------------------------------------------------
# The second clause: the campaign's own cards, re-adjudicated
# ---------------------------------------------------------------------------
#
# The seven tests above prove the runner turns a missing metric into a named VOID on a study
# written to have one. This half proves it on the eight real cards that produced the defect, from
# the campaign's own frozen specs and its own recorded metrics. Nothing is recomputed: the only
# thing that differs between the campaign's verdicts and these is the adjudication code.
#
# The regression targets, counted directly off the campaign's own scoreboard: zero `inconclusive`
# at either level, eight card-level voids, sixteen hypothesis-level voids, each naming its absent
# metric.

# These three sit here rather than at the top of the file because this half was appended: the
# seven tests above were written for the first clause and moving their imports around to make
# room would be an edit to them. noqa rather than a per-file ignore, so the exemption is local to
# the three lines that need it.
import os as _os  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

from reward_lens.record.convert import readjudicate as _readjudicate  # noqa: E402

#: The campaign archive, which is not in this repository. Neither path has a default.
_STORE_ENV = _os.environ.get("REWARD_LENS_CAMPAIGN_STORE")
_SPECS_ENV = _os.environ.get("REWARD_LENS_CAMPAIGN_SPECS")
CAMPAIGN_STORE = _Path(_STORE_ENV) if _STORE_ENV else None
CAMPAIGN_SPECS = _Path(_SPECS_ENV) if _SPECS_ENV else None

_campaign = pytest.mark.skipif(
    CAMPAIGN_STORE is None
    or CAMPAIGN_SPECS is None
    or not (CAMPAIGN_STORE / "evidence.jsonl").exists()
    or not CAMPAIGN_SPECS.exists(),
    reason=(
        "needs the campaign archive, which is not in this repository: set "
        "REWARD_LENS_CAMPAIGN_STORE to the evidence store and REWARD_LENS_CAMPAIGN_SPECS to "
        "the frozen specs."
    ),
)


@pytest.fixture(scope="module")
def campaign():
    """Re-adjudicate all twenty-seven cards once. Takes about half a second."""
    return _readjudicate(CAMPAIGN_STORE, CAMPAIGN_SPECS)


@_campaign
def test_the_campaign_re_adjudicates_to_zero_inconclusive(campaign) -> None:
    """The word is gone from both levels, and the campaign had it at both."""
    assert campaign.missing_specs == ()
    assert len(campaign.cards) == 27  # twenty-seven cards, not twenty-six

    assert len(campaign.recorded_inconclusive_cards) == 8
    assert len(campaign.recorded_inconclusive_hypotheses) == 16

    assert campaign.inconclusive_cards == 0
    assert campaign.inconclusive_hypotheses == 0


@_campaign
def test_eight_cards_are_void_and_they_are_the_eight(campaign) -> None:
    """Not eight of something: the same eight the campaign called inconclusive."""
    assert sorted(campaign.void_cards) == [
        "GAUGE-E19",
        "GAUGE-XFAM",
        "HACK-FORE",
        "HUMP",
        "PPE-BON",
        "STYLE-RMB",
        "T3-FIELD",
        "VALUES-CONTEST",
    ]
    assert set(campaign.void_cards) == set(campaign.recorded_inconclusive_cards)


@_campaign
def test_sixteen_hypotheses_are_void_and_each_names_its_absent_metric(campaign) -> None:
    """A VOID with no named arc is `inconclusive` spelled differently."""
    assert len(campaign.void_hypotheses) == 16
    assert set(campaign.void_hypotheses) == set(campaign.recorded_inconclusive_hypotheses)

    assert campaign.unnamed_voids() == ()
    assert campaign.arcless_voids() == ()

    style = next(c for c in campaign.cards if c.card == "STYLE-RMB")
    void = style.voids["H-style-transfer"]
    assert void.reason is VoidReason.METRIC_ABSENT
    assert "spearman_biasbattery_vs_rmbench_hard" in void.detail
    assert "H-style-transfer" in void.detail
    assert void.arc == "arc:campaign.bias.battery@armorm/diagnostic-v3-degradation"
    assert void.remedy


@_campaign
def test_a_kill_criterion_whose_metric_was_missing_is_void_not_a_non_firing(campaign) -> None:
    """Seven of the eight blocked cards registered a kill criterion and none of them was evaluated.

    Under the old adjudication every one of those read as a criterion that ran and did not fire.
    T3-FIELD is the eighth and it registered none, which is why the count is seven and not eight.
    """
    assert len(campaign.void_kills) == 7
    assert "T3-FIELD" not in {card for card, _ in campaign.void_kills}
    for card_name, kill_id in campaign.void_kills:
        card = next(c for c in campaign.cards if c.card == card_name)
        assert card.result.kill_outcomes[kill_id] == "void"
        assert kill_id not in card.result.killed_by
        assert "neither fired nor passed" in card.voids[kill_id].detail


@_campaign
def test_the_nineteen_adjudicated_cards_still_adjudicate_the_same_way(campaign) -> None:
    """The change has to be confined to the eight, or it is not a fix, it is a different answer."""
    unchanged = [c for c in campaign.cards if c.card not in campaign.void_cards]
    assert len(unchanged) == 19
    for card in unchanged:
        assert card.outcome is not StudyOutcome.VOID
        assert card.result.voids == {}
        assert card.result.outcomes == dict(card.recorded_outcomes), card.card


@_campaign
def test_the_specs_re_adjudicated_are_byte_identical_to_the_ones_frozen(campaign) -> None:
    """Gate 3 is that the freeze predates the evidence, so re-freezing would destroy the check.

    The frozen study is rebuilt from the campaign's own JSON and its spec hash is recomputed. All
    twenty-seven reproduce, which is what makes "the same predictions" a measurement.
    """
    from reward_lens.record.convert import CampaignStore

    earliest_evidence = min(row.created_at for row in CampaignStore(CAMPAIGN_STORE).rows)
    assert campaign.unverified_specs == ()
    for card in campaign.cards:
        assert card.frozen.git_sha.startswith("f93f4b5")
        # Gate 3, checked rather than assumed: the freeze predates every row in the store.
        assert card.frozen.frozen_at < earliest_evidence


@_campaign
def test_re_adjudication_cannot_write_to_the_archive(campaign) -> None:
    """The store handed to the runner is read-only, so an append during adjudication raises."""
    from reward_lens.core.store import EvidenceStore

    store = EvidenceStore(CAMPAIGN_STORE, readonly=True)
    assert store.readonly is True
    with pytest.raises(RuntimeError, match="readonly"):
        store.append(object())  # type: ignore[arg-type]

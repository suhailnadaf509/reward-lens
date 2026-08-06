"""Acceptance: a campaign-level count is registered only if the rule that produced it was frozen.

The clause this file discharges: *a campaign-level count with no `MetaAnalysisPlan` renders as
`EXPLORATORY` with the word printed beside it; a count covered by a frozen plan does not.*

Both halves are here, and so is a third test that is the reason the other two are worth having.
A check that has only ever been pointed at fixtures it was written alongside proves that the code
runs. This one is pointed at the previous campaign's own published summary, re-derived from its own
evidence store, and it shows the check firing on the sentence it exists to catch.

That sentence is `campaign-results/RESULTS.md` line 5: "19 of 27 frozen cards adjudicated against
the merged evidence store; 16 of 53 frozen hypotheses confirmed, 21 refuted, 16 inconclusive; 8
kill criteria fired, counting the meta-ledger throughout." Every one of those numbers is correct.
Every one of the 27 adjudication rows behind them carries `trust: 2`, which is REGISTERED. The
sentence adding them up is registered against nothing, because `specs/frozen/manifest.json` holds
eight keys and none of them is an aggregation rule.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from reward_lens.core.types import TrustLevel
from reward_lens.studies.meta_analysis import (
    CountClaim,
    ExploratoryReason,
    MetaAnalysisPlan,
    Stability,
    boundary_sensitivity,
    campaign_counts,
    correct_family,
    cover,
    freeze_meta_plan,
    near_threshold,
    p_value_predictions,
    predictions_from_cards,
    render_count,
)

#: The campaign archive, which is not in this repository. Neither path has a default: point
#: ``REWARD_LENS_CAMPAIGN_STORE`` at the evidence store, and ``REWARD_LENS_CAMPAIGN_SPECS`` at
#: the frozen specs if they are not at the store's own ``../../specs/frozen``.
_CAMPAIGN_ENV = os.environ.get("REWARD_LENS_CAMPAIGN_STORE")
_SPECS_ENV = os.environ.get("REWARD_LENS_CAMPAIGN_SPECS")
CAMPAIGN_STORE = Path(_CAMPAIGN_ENV) if _CAMPAIGN_ENV else None
if _SPECS_ENV:
    CAMPAIGN_SPECS = Path(_SPECS_ENV)
elif CAMPAIGN_STORE is not None:
    CAMPAIGN_SPECS = CAMPAIGN_STORE.parent.parent / "specs" / "frozen"
else:
    CAMPAIGN_SPECS = None

requires_campaign = pytest.mark.skipif(
    CAMPAIGN_STORE is None
    or CAMPAIGN_SPECS is None
    or not (CAMPAIGN_STORE / "evidence.jsonl").exists()
    or not CAMPAIGN_SPECS.exists(),
    reason=(
        "no campaign evidence store with frozen specs. It is the archive the 2.0 campaign "
        "produced and it is not in the repository; set REWARD_LENS_CAMPAIGN_STORE and "
        "REWARD_LENS_CAMPAIGN_SPECS to point at it."
    ),
)

#: The three studies a small worked example counts over. Enumerating them is what makes the
#: inclusion rule checkable; the prose beside it is what makes it reviewable.
EXAMPLE_STUDIES = ("card-a", "card-b", "card-c")


def _claim(value: int = 2, label: str = "confirmed", **kw) -> CountClaim:
    fields = {
        "unit": "hypothesis",
        "over": EXAMPLE_STUDIES,
        "statement": f"{value} of 5 frozen hypotheses {label}",
        "evidence_at": "2026-08-01T12:00:00+00:00",
    }
    fields.update(kw)
    return CountClaim(label=label, value=value, **fields)


def _plan(**kw) -> MetaAnalysisPlan:
    fields = {
        "id": "example-campaign",
        "studies": EXAMPLE_STUDIES,
        "unit": "hypothesis",
        "labels": ("confirmed", "refuted"),
        "frozen_at": "2026-07-01T00:00:00+00:00",
    }
    fields.update(kw)
    return freeze_meta_plan(
        "One tally mark per registered hypothesis, summed over every included study.",
        "The three cards named in `studies`, fixed before any of them was adjudicated.",
        "A hypothesis whose frozen prediction held is confirmed; one that did not is refuted; "
        "one whose metric was never computed is void and enters no tally.",
        **fields,
    )


# ---------------------------------------------------------------------------
# The clause, first half: no plan means the word is printed
# ---------------------------------------------------------------------------


def test_a_count_with_no_plan_renders_exploratory_with_the_word_printed() -> None:
    """The first half of the clause, taken literally: the word appears next to the number."""
    claim = _claim()
    rendered = render_count(claim, None)

    assert "EXPLORATORY" in rendered
    # Next to the number, not in a footnote. The first line carries both.
    first_line = rendered.splitlines()[0]
    assert "2 of 5 frozen hypotheses confirmed" in first_line
    assert "EXPLORATORY" in first_line

    coverage = cover(claim, None)
    assert not coverage.covered
    assert coverage.trust is TrustLevel.EXPLORATORY
    assert coverage.reasons[0][0] is ExploratoryReason.NO_PLAN


def test_the_count_is_printed_rather_than_omitted_or_blocked() -> None:
    """The rule is a label, not a suppression and not an exception.

    Getting this wrong in either direction defeats the purpose. Omitting the count hides the
    finding from the reader; raising hides it from the document. The number has to survive.
    """
    claim = _claim(value=16, statement="16 of 53 frozen hypotheses confirmed")
    rendered = render_count(claim, None)  # must not raise
    assert "16" in rendered
    assert "16 of 53 frozen hypotheses confirmed" in rendered


def test_the_label_says_what_it_means_and_what_would_remove_it() -> None:
    """A badge nobody can act on is decoration. The render carries the remedy."""
    rendered = render_count(_claim(), None)
    assert "not preregistered" in rendered
    assert "Freeze a MetaAnalysisPlan" in rendered
    assert "aggregation" in rendered and "inclusion" in rendered.lower()


# ---------------------------------------------------------------------------
# The clause, second half: a frozen plan removes the word
# ---------------------------------------------------------------------------


def test_a_count_covered_by_a_frozen_plan_does_not_render_exploratory() -> None:
    """The second half of the clause. The badge names the plan instead."""
    plan = _plan()
    claim = _claim()
    rendered = render_count(claim, plan)

    assert "EXPLORATORY" not in rendered
    assert "REGISTERED" in rendered
    assert plan.short_id in rendered
    assert "2 of 5 frozen hypotheses confirmed" in rendered

    coverage = cover(claim, plan)
    assert coverage.covered
    assert coverage.trust is TrustLevel.REGISTERED
    assert coverage.reasons == ()


def test_a_frozen_plan_verifies_its_own_hash() -> None:
    plan = _plan()
    assert plan.spec_hash.startswith("meta:")
    assert plan.hash_verified
    assert plan.frozen_at == "2026-07-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# The ways a plan fails to cover a count. Without these the object launders anything.
# ---------------------------------------------------------------------------


def test_a_plan_edited_after_freezing_covers_nothing() -> None:
    """Changing the aggregation rule and keeping the old hash is the whole attack."""
    from dataclasses import replace

    tampered = replace(_plan(), aggregation="One tally mark per study, not per hypothesis.")
    coverage = cover(_claim(), tampered)
    assert not coverage.covered
    assert ExploratoryReason.HASH_MISMATCH in {r for r, _ in coverage.reasons}
    assert "EXPLORATORY" in coverage.render()


def test_a_plan_frozen_after_the_evidence_covers_nothing() -> None:
    """A plan written once the numbers are in describes the count; it does not commit to it."""
    late = _plan(frozen_at="2026-08-02T00:00:00+00:00")
    coverage = cover(_claim(evidence_at="2026-08-01T12:00:00+00:00"), late)
    assert not coverage.covered
    assert ExploratoryReason.PLAN_POSTDATES_EVIDENCE in {r for r, _ in coverage.reasons}


def test_a_count_over_a_different_set_of_studies_is_not_covered() -> None:
    """Dropping a card from the tally is the selection the retraction was about."""
    coverage = cover(_claim(over=("card-a", "card-b")), _plan())
    assert not coverage.covered
    reasons = {r for r, _ in coverage.reasons}
    assert ExploratoryReason.OUTSIDE_INCLUSION in reasons
    assert "card-c" in coverage.render()


def test_a_count_over_extra_studies_is_not_covered() -> None:
    coverage = cover(_claim(over=EXAMPLE_STUDIES + ("card-d",)), _plan())
    assert not coverage.covered
    assert "card-d" in coverage.render()


def test_a_plan_with_prose_inclusion_only_cannot_cover_a_count() -> None:
    """Three of the five specified fields are free prose, and prose is not checkable.

    A plan built exactly to the five-field shape carries an inclusion rule a reviewer can read and
    nothing a program can compare a count against, so it leaves the count exploratory. That is the
    honest outcome and the remedy names the fix.
    """
    prose_only = _plan(studies=())
    coverage = cover(_claim(), prose_only)
    assert not coverage.covered
    assert ExploratoryReason.INCLUSION_UNCHECKABLE in {r for r, _ in coverage.reasons}
    assert "List the study ids" in coverage.render()


def test_a_plan_that_aggregates_studies_does_not_cover_a_count_of_hypotheses() -> None:
    coverage = cover(_claim(unit="hypothesis"), _plan(unit="study"))
    assert not coverage.covered
    assert ExploratoryReason.UNIT_UNDECLARED in {r for r, _ in coverage.reasons}


def test_a_label_the_scoring_rule_never_declared_is_not_covered() -> None:
    coverage = cover(_claim(label="mixed"), _plan())
    assert not coverage.covered
    assert ExploratoryReason.LABEL_UNDECLARED in {r for r, _ in coverage.reasons}


# ---------------------------------------------------------------------------
# The third assertion: point it at the claim it exists to catch
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def campaign_predictions():
    """The campaign's 78 registered predictions, re-derived from its own evidence store.

    Nothing is recomputed. `readjudicate` reads the metrics back out of the `campaign.adjudication.*`
    rows and re-runs the frozen predictions against them, and all 27 spec hashes verify.
    """
    from reward_lens.record.convert.readjudicate import readjudicate

    report = readjudicate(CAMPAIGN_STORE, CAMPAIGN_SPECS)
    assert len(report.cards) == 27
    assert report.unverified_specs == ()
    return predictions_from_cards(report.cards)


@requires_campaign
def test_the_campaigns_own_meta_claim_reproduces_from_its_evidence(campaign_predictions) -> None:
    """The published sentence, re-derived. If these drift, the test below is about nothing."""
    hypotheses = [p for p in campaign_predictions if p.kind == "hypothesis"]
    kills = [p for p in campaign_predictions if p.kind == "kill"]

    assert len(hypotheses) == 53
    assert sum(1 for p in hypotheses if p.outcome == "confirmed") == 16
    assert sum(1 for p in hypotheses if p.outcome == "refuted") == 21
    assert len({p.study for p in kills if p.outcome == "fired"}) == 8


@requires_campaign
def test_the_check_would_have_caught_the_campaigns_headline(campaign_predictions) -> None:
    """The claim this object exists for, run against the real thing.

    `campaign-results/specs/frozen/manifest.json` records `spec_hashes` and `study_ids` for all 27
    cards and no aggregation rule, so the plan covering the headline is `None`. All three counts
    render `EXPLORATORY`, and every number inside them is registered.
    """
    claims = campaign_counts(campaign_predictions, evidence_at="2026-07-19T18:18:15.029012+00:00")
    values = {c.label: c.value for c in claims}
    assert values == {"confirmed": 16, "refuted": 21, "fired": 8}

    for claim in claims:
        coverage = cover(claim, None)
        assert not coverage.covered, claim.statement
        assert coverage.trust is TrustLevel.EXPLORATORY
        rendered = coverage.render()
        assert "EXPLORATORY" in rendered.splitlines()[0]
        assert str(claim.value) in rendered.splitlines()[0]

    rendered = render_count(claims[0], None)
    assert "16 of 53 frozen hypotheses confirmed" in rendered
    assert "EXPLORATORY" in rendered


@requires_campaign
def test_a_plan_frozen_over_the_campaign_would_have_covered_it(campaign_predictions) -> None:
    """The counterfactual, so the first test is not just asserting that nothing ever passes.

    A plan naming the 27 cards, the unit and the labels, frozen on the campaign's own freeze date
    of 2026-07-18 (before the 2026-07-19 adjudication), covers the same three counts. Nothing about
    the campaign's numbers had to change; only the rule for adding them up had to exist first.
    """
    claims = campaign_counts(campaign_predictions, evidence_at="2026-07-19T18:18:15.029012+00:00")
    hypothesis_claims = [c for c in claims if c.unit == "hypothesis"]
    plan = freeze_meta_plan(
        "One tally mark per registered hypothesis, summed over all 27 frozen cards.",
        "Every card in the freeze manifest, fixed at the freeze and not revisited after "
        "adjudication.",
        "A hypothesis whose frozen prediction held is confirmed; one that did not is refuted; one "
        "whose metric was never computed is void and enters no tally.",
        id="campaign-2026-07",
        studies=hypothesis_claims[0].over,
        unit="hypothesis",
        labels=("confirmed", "refuted", "void"),
        frozen_at="2026-07-18T23:46:57.951556+00:00",
    )
    for claim in hypothesis_claims:
        coverage = cover(claim, plan)
        assert coverage.covered, coverage.render()
        assert coverage.trust is TrustLevel.REGISTERED
        assert "EXPLORATORY" not in coverage.render()


# ---------------------------------------------------------------------------
# Multiplicity and null-boundary sensitivity, on the same family
# ---------------------------------------------------------------------------


@requires_campaign
def test_one_campaign_confirmation_does_not_survive_the_family(campaign_predictions) -> None:
    """Across-card multiplicity, which the campaign disclosed it did not do.

    `RESULTS.md` line 240 says multiplicity control was per registered p-value within a card, and
    that no card carried two, so the correction was inert. Across cards the family is four
    evaluated p-values, and one of the campaign's 16 confirmations does not survive it.
    """
    hypotheses = [p for p in campaign_predictions if p.kind == "hypothesis"]
    family = p_value_predictions(hypotheses)
    assert len(family) == 4

    correction = correct_family(family, alpha=0.05)
    assert sum(correction.uncorrected) == 2
    assert sum(correction.bh_rejected) == 1
    assert sum(correction.by_rejected) == 1
    # e-BH needs an e-value of n/alpha = 80 for a single rejection at n = 4, and no admissible
    # calibrator takes p = 0.001 that far. Reported, not hidden.
    assert correction.ebh_k == 0
    assert correction.lost_to_correction == ("CONF-PARTIAL/H-partial-perm",)


@requires_campaign
def test_the_boundary_sensitive_verdicts_are_named(campaign_predictions) -> None:
    """Which verdicts would flip under a small move in the estimator, and which cannot be told.

    No evidence row in the campaign store carries a confidence interval, so no verdict's stability
    is established and every continuous row comes back `UNKNOWN` rather than being scored against
    an invented standard error. What is computable exactly is the flip distance, and four verdicts
    sit within ten percent of their frozen threshold.
    """
    hypotheses = [p for p in campaign_predictions if p.kind == "hypothesis"]
    rows = boundary_sensitivity(hypotheses)
    assert len(rows) == 37

    assert all(r.stability is Stability.UNKNOWN for r in rows if r.comparator not in ("==", "!="))
    assert all(r.uncertainty is None for r in rows)
    assert any("Record ci_low and ci_high" in r.note for r in rows)

    close = near_threshold(rows, relative_cut=0.10)
    assert {f"{r.study}/{r.owner}" for r in close} == {
        "META-LEDGER/H-brier",
        "CONF-PARTIAL/H-partial-perm",
        "SURGERY/H-erase",
        "JUDGE-VBC/H-vbc",
    }

    # The closest of them is the refutation that fired the campaign's meta kill criterion: a Brier
    # score of 0.26 against a frozen 0.25.
    brier = next(r for r in close if r.owner == "H-brier")
    assert brier.outcome == "refuted"
    assert brier.flip_distance == pytest.approx(0.01, abs=1e-9)

    # The one the family correction drops is also one of the four. Two independent reasons to
    # doubt the same confirmation.
    assert "CONF-PARTIAL/H-partial-perm" in {f"{r.study}/{r.owner}" for r in close}

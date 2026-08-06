"""X8 acceptance: the leaderboard meta-analysis, its freeze, and its refusals.

The clauses this file discharges:

  - every effect size in the analysis is a quotation that is still in the dossier, on the line it is
    attributed to, checked before anything is computed;
  - the spec is frozen, and its hash is a content hash of the predictions and thresholds, so editing
    a threshold after the fact is visible as a different study id;
  - the freeze reports honestly whether its git stamp is worth anything;
  - the analysis refuses rather than reports when the arithmetic runs out, at the two places it can:
    fewer than three poolable studies, and a funnel-asymmetry test below ten;
  - the prediction interval is reported, is distinguished from the confidence interval in the prose,
    and is wider;
  - the write-up is generated from the numbers and reads correctly if the result falls the other
    way, which is checked by making it fall the other way;
  - every metric a registered prediction or kill criterion names is produced by the analysis, which
    is plan closure for this study.

The regression pins on the pooled value, tau2 and both intervals are deliberate. They are the
numbers a write-up quotes, so a change to the extraction, the scale, the correction or the estimator
should break this file and be defended, not pass quietly.
"""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.experiments import x8_leaderboard_meta as x8
from reward_lens.stats import meta

pytestmark = pytest.mark.skipif(
    x8.DOSSIER is None or not x8.DOSSIER.exists(),
    reason=(
        "no evidence base. X8 is an analysis of quotations and does not run without the file they "
        "are quoted from; set REWARD_LENS_EVAL_DOSSIER to it."
    ),
)


@pytest.fixture(scope="module")
def result() -> x8.X8Result:
    return x8.analyse()


# ---------------------------------------------------------------------------
# Traceability
# ---------------------------------------------------------------------------


def test_every_quoted_number_is_still_in_the_dossier_where_it_says_it_is():
    """The anti-fabrication check, run over the effect sizes and the framing claims alike.

    Twenty-odd substrings across six studies plus the section's own summary sentences. A line number
    that has drifted, a quote that was transcribed rather than read, or an edit to the source all
    fail here. This already caught one real error during the build: the Dlugosz numerator sits in a
    sentence that wraps across two lines and had been recorded against the wrong half of it.
    """
    checks = x8.verify_quotes()
    bad = [c for c in checks if not c.found]
    assert not bad, "\n".join(
        f"{c.key} line {c.line}: {c.quote!r} not in {c.actual!r}" for c in bad
    )
    assert len(checks) >= 18
    assert {c.key for c in checks} == {e.key for e in x8.SIX} | {"framing"}


def test_the_dossier_is_the_one_the_line_numbers_were_read_against():
    """A sha mismatch is not a failure, it is a prompt to recheck. Recorded, and asserted here.

    If this ever fails, the source has been edited: rerun `verify_quotes`, and if that still passes
    the line numbers survived the edit and the constant is what needs updating.
    """
    assert x8.dossier_sha256() == x8.DOSSIER_SHA256


def test_every_extracted_effect_size_carries_both_its_numbers_and_their_lines():
    for e in x8.SIX:
        assert e.sources, f"{e.key} has no source line"
        if e.extracted:
            assert e.unresolved is not None and e.total is not None
            assert 0 <= e.unresolved <= e.total
            # The numerator or the denominator has to be visible in a quoted string.
            joined = " ".join(q for _, q in e.sources)
            assert str(e.total) in joined or str(e.unresolved) in joined
        else:
            assert e.unresolved is None and e.total is None
            assert e.proportion is None


def test_the_two_studies_without_denominators_are_named_rather_than_dropped(result):
    assert {e.key for e in result.refused} == {"mandujano_reyes", "zhuang"}
    assert {e.key for e in result.extracted} == {
        "chandrahas",
        "kotawala",
        "chacon_sartori",
        "dlugosz",
    }
    section = x8.findings_section(result)
    for e in result.refused:
        assert e.arxiv in section
    assert "could not be pooled" in section


# ---------------------------------------------------------------------------
# The freeze
# ---------------------------------------------------------------------------


def test_the_spec_hash_is_a_content_hash_of_the_predictions():
    """Two freezes of the same spec agree; a changed threshold does not. That is the whole mechanism.

    Frozen at a fixed timestamp so the comparison is of the spec and not of the clock.
    """
    a = x8.freeze_x8(frozen_at="2026-08-01T00:00:00+00:00")
    b = x8.freeze_x8(frozen_at="2026-08-01T00:00:00+00:00")
    assert a.frozen.spec_hash == b.frozen.spec_hash
    assert a.frozen.study_id == b.frozen.study_id
    assert str(a.frozen.study_id).startswith("study:x8-leaderboard-meta@v1#")

    from reward_lens.core.types import content_hash

    spec = x8.study_spec()
    moved = dataclasses.replace(
        spec,
        hypotheses=(
            dataclasses.replace(
                spec.hypotheses[0],
                prediction=dataclasses.replace(spec.hypotheses[0].prediction, threshold=0.05),
            ),
            *spec.hypotheses[1:],
        ),
    )
    assert content_hash(moved.__canonical__(), "spec") != a.frozen.spec_hash


def test_the_freeze_says_whether_its_git_stamp_means_anything():
    """A dirty tree does not stop the freeze here, but it does have to be visible in the output.

    The campaign refuses outright and that is right for a spend commitment against a commit somebody
    will check out later. This analysis is written alongside other work in one checkout, so refusing
    would mean the predictions never get hashed. The compromise has to be stated rather than hidden:
    the spec hash is exact either way and the git sha is what carries the +dirty marker.
    """
    outcome = x8.freeze_x8(frozen_at="2026-08-01T00:00:00+00:00")
    assert outcome.provisional == (not outcome.clean)
    if outcome.provisional:
        assert outcome.dirty_paths
        assert "PROVISIONAL" in outcome.detail
        assert "+dirty" in outcome.frozen.git_sha or outcome.frozen.git_sha == "unknown"
        assert outcome.frozen.spec_hash in outcome.detail
    else:
        assert "Clean tree" in outcome.detail


def test_the_freeze_runs_before_the_analysis():
    """`run` fixes the order and does not take it as an argument."""
    outcome, res = x8.run()
    assert outcome.frozen.frozen_at <= res.extras.get("__unused__", "9999")
    assert res.metrics["k_extracted"] == 4.0
    import inspect

    src = inspect.getsource(x8.run)
    assert src.index("freeze_x8") < src.index("analyse(")


def test_the_freeze_honesty_note_is_hashed_into_the_spec():
    """The admission about what the preregistration protects is part of the frozen content.

    If it were only in the write-up it could be softened later without the study id changing, which
    is exactly the property preregistration exists to remove.
    """
    spec = x8.study_spec()
    assert spec.notes == x8.FREEZE_HONESTY
    assert "inclusion is not preregistered" in spec.notes
    assert "hand-computed" in spec.notes
    assert "H3 names a quantity that had not been computed" in spec.notes


# ---------------------------------------------------------------------------
# Plan closure
# ---------------------------------------------------------------------------


def test_every_registered_metric_is_produced_by_the_analysis(result):
    """A prediction naming a metric nothing computes is a study that cannot be adjudicated.

    The library's word for that is PLAN_NOT_CLOSED and it is meant to be caught before the run. This
    is the same check for this study, run after, which is the weaker version and still worth having.
    """
    spec = x8.study_spec()
    named = {h.prediction.metric for h in spec.hypotheses} | {k.metric for k in spec.kill_criteria}
    missing = named - set(result.metrics)
    assert not missing, f"registered metrics with no value: {sorted(missing)}"
    assert set(result.outcomes) == {h.id for h in spec.hypotheses}
    assert set(result.kill_outcomes) == {k.id for k in spec.kill_criteria}
    assert all(v in {"confirmed", "refuted", "void"} for v in result.outcomes.values())
    assert all(v in {"fired", "passed", "void"} for v in result.kill_outcomes.values())


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


def test_the_primary_fit_is_the_one_the_write_up_quotes(result):
    """Regression pins on the four numbers a reader takes away.

    Recomputed from 3/8, 4/9, 3/4 and 10/20 on the logit scale with a 0.5 correction applied to
    every study, Paule-Mandel tau2, t(k-2) prediction interval. If any of those choices changes,
    this fails and the change gets defended.
    """
    p = result.primary
    assert p.fit.k == 4
    assert p.observed == pytest.approx((0.375, 4 / 9, 0.75, 0.5))
    assert p.pooled_p == pytest.approx(0.4869, abs=5e-4)
    assert p.ci_p == pytest.approx((0.3439, 0.6320), abs=5e-4)
    assert p.prediction_p == pytest.approx((0.2051, 0.7773), abs=5e-4)
    assert p.fit.het.tau2 == pytest.approx(0.0, abs=1e-9)
    assert p.fit.het.q == pytest.approx(1.2603, abs=5e-4)
    assert p.fit.het.q_df == 3
    assert p.fit.het.i2 == pytest.approx(0.0, abs=1e-9)


def test_the_prediction_interval_is_wider_than_the_confidence_interval_and_says_so(result):
    """The clause the module exists for, checked on the numbers and then on the prose."""
    p = result.primary
    assert p.prediction_p[0] < p.ci_p[0]
    assert p.prediction_p[1] > p.ci_p[1]
    # tau2 is zero here, so the entire gap is the critical value, and it is exactly t(2)/z.
    assert p.fit.width_ratio == pytest.approx(x8.student_t_over_z(4), rel=1e-9)
    assert p.fit.width_ratio == pytest.approx(2.1953, abs=5e-4)
    section = x8.findings_section(result)
    assert "about the *mean*" in section
    assert "about the *next* study" in section
    assert "prediction interval" in section


def test_tau2_is_zero_and_the_write_up_refuses_to_call_that_agreement(result):
    """The whole small-k argument, checked where it would be tempting to leave it out.

    tau2_hat is 0.000, which a careless write-up would report as "the six studies agree". The
    Q-profile upper limit is above 3 on the logit scale, which is a between-study standard deviation
    of nearly 2, and Cochran's Q at this k has under a third of the power it would need to see
    heterogeneity the size of the typical within-study variance.
    """
    het = result.primary.fit.het
    assert het.tau2 == pytest.approx(0.0, abs=1e-9)
    assert het.tau2_ci[0] == pytest.approx(0.0, abs=1e-9)
    assert het.tau2_ci[1] > 1.0
    assert not het.is_reliable
    assert result.power["q_test_at_typical_variance"] < 0.5
    section = x8.findings_section(result)
    assert "Q-profile interval" in section
    assert "statement about the test" in section
    assert "The wide interval is the finding." in section


def test_the_confidence_interval_excludes_the_null_and_the_prediction_interval_does_not(result):
    """The case the module was built to make legible, and it is the case that actually occurred.

    Worth pinning because it is the sentence the piece turns on: the average is established and the
    next leaderboard is not.
    """
    null_logit = math.log(x8.NULL_FRACTION / (1 - x8.NULL_FRACTION))
    ci_excludes, pi_excludes = result.primary.fit.excludes(null_logit)
    assert ci_excludes and not pi_excludes
    section = x8.findings_section(result)
    assert "The average is established and the next leaderboard is not." in section


def test_the_default_package_settings_would_have_collapsed_the_prediction_interval(result):
    """With tau2 at zero, a normal critical value makes the prediction interval the CI exactly.

    That is the concrete cost of the convention this module does not follow, and it is checked here
    rather than asserted in prose.
    """
    nr = result.extras["normal_rule_prediction_p"]
    assert nr == pytest.approx(result.primary.ci_p, abs=1e-9)
    assert "the same pair of numbers twice" in x8.findings_section(result)


# ---------------------------------------------------------------------------
# Baselines, power, sensitivity
# ---------------------------------------------------------------------------


def test_the_three_baselines_are_all_reported(result):
    """A pooled estimate with nothing to be compared against is a number, not a result."""
    b = result.baselines
    assert isinstance(b["vote_count"], meta.VoteCount)
    assert b["vote_count"].k == 4 and b["vote_count"].positive == 4
    assert b["unweighted_mean"] == pytest.approx(0.5174, abs=5e-4)
    assert b["fixed_effect_p"] == pytest.approx(result.primary.pooled_p, abs=1e-9)
    section = x8.findings_section(result)
    for name in ("Vote count", "Unweighted mean", "Fixed-effect model"):
        assert name in section


def test_publication_bias_is_refused_rather_than_estimated(result):
    """Four studies. Egger returns a refusal with a reason and a remedy, and the write-up says so."""
    egger = result.baselines["egger"]
    assert isinstance(egger, Refusal)
    assert egger.reason is RefusalReason.ESS_BELOW_FLOOR
    assert egger.statistics["k"] == 4
    assert "could not be assessed" in egger.remedy
    assert "not assessed" in x8.findings_section(result)


def test_power_is_reported_at_the_realised_k_and_across_the_range_tau2_could_take(result):
    """One power number at a single assumed tau2 would be the flattering one. Three are reported."""
    p = result.power
    assert p["at_tau2_zero"] > p["at_tau2_upper"]
    assert 0.0 <= p["at_tau2_upper"] <= 1.0
    assert p["delta_logit"] == pytest.approx(math.log(0.75 / 0.25), rel=1e-12)
    assert "Power at the realised sample size" in x8.findings_section(result)


def test_no_pre_registered_sensitivity_run_moves_the_headline_across_the_threshold(result):
    """K2's condition, evaluated. It passed here; the criterion exists for when it does not."""
    assert len(result.sensitivity) >= 10
    assert result.metrics["sensitivity_sign_flips"] == 0.0
    assert result.kill_outcomes["K2"] == "passed"
    names = {r.name for r in result.sensitivity}
    for expected in (
        "kotawala-all-pairs",
        "dlugosz-after-confound",
        "with-mandujano-bound",
        "double-arcsine",
        "tau2-DL",
        "tau2-REML",
        "hartung-knapp",
        "normal-prediction-rule",
    ):
        assert expected in names


def test_the_two_studies_that_could_not_be_pooled_are_offered_a_bounded_run(result):
    """The excluded Bayesian IRT result enters exactly once, at a bound, clearly labelled.

    Adding it takes k from 4 to 5, and the prediction interval narrows sharply on the degrees of
    freedom alone, which is a useful thing for a reader to see: at this size the interval is driven
    as much by how many studies there are as by what they found.
    """
    run = next(r for r in result.sensitivity if r.name == "with-mandujano-bound")
    assert run.k == 5
    primary_width = result.primary.prediction_p[1] - result.primary.prediction_p[0]
    assert (run.prediction_p[1] - run.prediction_p[0]) < primary_width
    assert "Both integers are inferred and neither is quoted" in run.description


# ---------------------------------------------------------------------------
# The write-up reads correctly whichever way the result falls
# ---------------------------------------------------------------------------


def test_the_write_up_takes_the_other_branch_when_the_pooled_fraction_is_small(monkeypatch):
    """Rerun the whole analysis on counts that put the pooled fraction below the threshold.

    Nothing about the section is written around the result that happened. This substitutes counts
    that refute H1, reruns everything, and checks the prose switched to the correct branch and the
    verdict table says refuted. A section that needed rewriting when the numbers moved would fail
    here.
    """
    small = tuple(
        dataclasses.replace(e, unresolved=1) if e.extracted and e.key != "chacon_sartori" else e
        for e in x8.SIX
    )
    small = tuple(
        dataclasses.replace(e, unresolved=0) if e.key == "chacon_sartori" else e for e in small
    )
    monkeypatch.setattr(x8, "SIX", small)
    res = x8.analyse()
    assert res.metrics["pooled_proportion"] < x8.NULL_FRACTION
    assert res.outcomes["H1"] == "refuted"
    section = x8.findings_section(res)
    assert "not distinguishable from it" in section
    assert "The average is established and the next leaderboard is not." not in section
    # The parts that are about the design rather than the result must survive either way.
    assert "The wide interval is the finding." in section
    assert "### Baselines" in section
    assert "**refuted**" in section
    # A boundary count makes the uncorrected-logit run undefined. It has to appear as unavailable
    # rather than crash the analysis or vanish from the table.
    dead = [r for r in res.sensitivity if not r.available]
    assert [r.name for r in dead] == ["no-continuity-correction"]
    assert "sits on a boundary" in dead[0].unavailable_reason
    assert "not available" in section


def test_fewer_than_three_poolable_studies_fires_the_kill_criterion(monkeypatch):
    """K1. The prediction interval does not exist below three studies, so the analysis stops.

    The refusal is the deliverable in that world: six papers that cannot be pooled is a shorter
    piece and a true one.
    """
    gutted = tuple(
        dataclasses.replace(e, unresolved=None, total=None, status="no-denominator")
        if e.key in {"chandrahas", "kotawala"}
        else e
        for e in x8.SIX
    )
    monkeypatch.setattr(x8, "SIX", gutted)
    with pytest.raises(RuntimeError, match="kill criterion K1 fired"):
        x8.analyse()


def test_a_drifted_quote_stops_the_run_before_anything_is_computed(monkeypatch, tmp_path):
    """The check has to be able to fail, so make it fail."""
    fake = tmp_path / "dossier.md"
    fake.write_text("\n".join(f"line {i}" for i in range(1, 2000)), encoding="utf-8")
    with pytest.raises(ValueError, match="not on the lines they are attributed to"):
        x8.assert_quotes_verify(fake)
    with pytest.raises(FileNotFoundError, match="every effect size in it is a quotation"):
        x8.verify_quotes(tmp_path / "nope.md")


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def test_the_command_line_writes_a_json_artifact_that_carries_every_quoted_number(tmp_path):
    """The write-up is reproducible from the artifact, so a claim in it can be traced to a command."""
    out = tmp_path / "x8.json"
    md = tmp_path / "x8.md"
    assert x8.main(["--json", str(out), "--markdown", str(md), "--quiet"]) == 0
    payload = json.loads(out.read_text())
    assert payload["frozen"]["spec_hash"].startswith("spec:")
    assert payload["dossier_sha256"] == x8.DOSSIER_SHA256
    assert len(payload["extraction"]) == 6
    assert sum(1 for e in payload["extraction"] if e["status"] == "extracted") == 4
    assert payload["primary"]["prediction_rule"] == "t(k-2)"
    assert payload["primary"]["prediction_df"] == 2
    assert payload["metrics"]["k_extracted"] == 4.0
    assert payload["outcomes"] == {"H1": "confirmed", "H2": "confirmed", "H3": "confirmed"}
    assert md.read_text().startswith("## X8: ")

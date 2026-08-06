"""The arithmetic, on subjects whose answer was planted.

Neither study is run. What is asserted here is that the two decompositions recover splits that were
put there on purpose, that both instruments refuse when the thing that makes the split possible is
missing, that both kill conditions fire when they should and not when they should not, that both
plans close, and that every instrument this package ships passes `lint_instrument`.

That last one is E56 and it is why it is the first test in the file. Four instruments shipped
failing lint rule 1 while their package read `done`, because the acceptance test rendered readings
and never linted the declarations. A package can discharge its clause completely and still not
exist by the architecture's own definition.

The power figures in both study specs' `notes` are recomputed here rather than quoted, in the three
slow tests at the end. They are the design's measured resolution and every one of them is a number
this file produced.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from reward_lens.core.invariance import InvariancePayload, check_invariance, check_unit_refusal
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.threshold import ThresholdQuantity, rank
from reward_lens.studies.plan import check_closure
from reward_lens.studies.void import StudyOutcome
from reward_lens.studies.w6_monitor import (
    KILL_TAU,
    W6_MONITOR,
    AuditedFamilyMass,
    MonitorDegradation,
    MonitorTrace,
    base_depth_for,
    decompose_mass,
    degradation_curve,
    fit_half_life,
    planted_family_counts,
    planted_monitor_bank,
    planted_monitor_trace,
    rank_monitors,
    sample_from_counts,
    zipf_base,
)
from reward_lens.studies.w6_monitor.study import (
    analyse_w6_4,
    analyse_w6_5,
    freeze_w6_4,
    freeze_w6_5,
    w6_4_plan,
    w6_4_spec,
    w6_5_plan,
    w6_5_spec,
)
from reward_lens.verifier import DisclosureDecision, DisclosureRequired, is_sensitive, redact

#: The denominator every fixture here uses: rewarded rollouts in one window.
N_REWARDED = 40_000


@pytest.fixture(scope="module")
def evasion_curve():
    """A monitor losing 1.585 halvings, all of it the policy and none of it the monitor."""
    trace = planted_monitor_trace(
        "evasion-only", evasion_halvings=1.585, degradation_halvings=0.0, seed=7
    )
    return degradation_curve(trace, n_surrogates=400, seed=3)


@pytest.fixture(scope="module")
def degradation_only_curve():
    trace = planted_monitor_trace(
        "degradation-only", evasion_halvings=0.0, degradation_halvings=1.585, seed=7
    )
    return degradation_curve(trace, n_surrogates=400, seed=3)


@pytest.fixture(scope="module")
def mixed_curve():
    trace = planted_monitor_trace("mixed", evasion_halvings=0.79, degradation_halvings=0.79, seed=7)
    return degradation_curve(trace, n_surrogates=400, seed=3)


# ---------------------------------------------------------------------------
# The declarations, which fail independently of the measurements (E56)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", W6_MONITOR, ids=lambda c: c.__name__)
def test_every_instrument_this_package_ships_lints_clean(cls):
    """Lint rule 1 through 3, on every instrument, because they fail independently of the reading."""
    assert lint_instrument(cls()) == []


@pytest.mark.parametrize("cls", W6_MONITOR, ids=lambda c: c.__name__)
def test_every_instrument_declares_a_registered_quantity_with_a_unit(cls):
    from reward_lens.core.quantity import QUANTITIES

    inst = cls()
    assert inst.quantity in QUANTITIES
    assert QUANTITIES.get(inst.quantity).invariance == inst.invariance


@pytest.mark.parametrize("cls", W6_MONITOR, ids=lambda c: c.__name__)
def test_the_generated_invariance_test_passes(cls):
    inst = cls()
    report = check_invariance(inst, inst.invariance, InvariancePayload(), run=lambda i, p: 1.0)
    assert report.passed, report


def test_a_half_life_does_not_rank_against_a_detection_curve():
    """The `units` group's real assertion: two quantities in different units do not compare.

    96 steps and an AUROC of 0.94 are not a large number and a small one, and there is no factor
    between them. This is the pair a reader of this package is most likely to try to rank, because
    both answer a question they will phrase as "which monitor is better".
    """
    assert check_unit_refusal(
        rank,
        ThresholdQuantity("monitor.half_life", 96.0),
        ThresholdQuantity("monitor.degradation_curve", 0.94),
    )
    out = rank(
        ThresholdQuantity("monitor.half_life", 96.0),
        ThresholdQuantity("verifier.fp_catalogue", 0.02),
    )
    assert isinstance(out, Refusal) and out.reason is RefusalReason.UNIT_MISMATCH


@pytest.mark.parametrize("cls", W6_MONITOR, ids=lambda c: c.__name__)
def test_an_instrument_with_nothing_to_read_refuses_with_a_remedy(cls):
    """Rule 1: a refusal is a value carrying a reason and an instruction, never an exception."""
    out = cls().estimate(Context())
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert out.remedy.strip() and out.remedy != out.detail
    assert "pass `" in out.remedy


# ---------------------------------------------------------------------------
# The split, by decay half-life
# ---------------------------------------------------------------------------


def test_pure_evasion_is_attributed_entirely_to_the_policy(evasion_curve):
    """The frozen pool did not move, so none of the loss is the monitor's."""
    assert evasion_curve.has_reference
    assert evasion_curve.evasion_share == pytest.approx(1.0, abs=0.05)
    assert evasion_curve.points[-1].degradation == pytest.approx(0.0, abs=0.05)
    assert evasion_curve.total_loss == pytest.approx(1.585, abs=0.15)


def test_pure_degradation_is_attributed_entirely_to_the_monitor(degradation_only_curve):
    """The frozen pool lost exactly what the live pool lost, so the policy did nothing."""
    assert degradation_only_curve.evasion_share == pytest.approx(0.0, abs=0.10)
    assert degradation_only_curve.points[-1].degradation == pytest.approx(
        degradation_only_curve.total_loss, abs=0.12
    )


def test_a_planted_mixture_recovers_its_share(mixed_curve):
    assert mixed_curve.evasion_share == pytest.approx(0.5, abs=0.08)


def test_the_two_decompositions_are_exact_identities(mixed_curve):
    """`total = evasion + degradation` and `total = separation_loss + spread_loss`, per point."""
    for point in mixed_curve.points:
        if not point.usable or not np.isfinite(point.degradation):
            continue
        assert point.total == pytest.approx(point.evasion + point.degradation, abs=1e-12)
        assert point.total == pytest.approx(point.separation_loss + point.spread_loss, abs=1e-9)


def test_the_half_life_recovers_the_planted_decay(evasion_curve):
    """1.585 halvings over a 190-step horizon is a half-life of 119.9 steps."""
    assert evasion_curve.half_life is not None, evasion_curve.half_life_detail
    assert evasion_curve.half_life_steps == pytest.approx(190.0 / 1.585, rel=0.06)
    assert evasion_curve.half_life.informative
    lo, hi = evasion_curve.half_life.ci
    assert lo < evasion_curve.half_life_steps < hi


def test_a_monitor_that_held_refuses_rather_than_returning_a_large_number():
    trace = planted_monitor_trace("held", evasion_halvings=0.0, degradation_halvings=0.0, seed=7)
    curve = degradation_curve(trace, n_surrogates=200, seed=3)
    assert curve.half_life is None and curve.held
    assert curve.half_life_reason == "BELOW_LOD"
    assert curve.half_life_steps == float("inf")
    assert "did not lose discriminability" in curve.half_life_detail
    assert "shortest half-life this horizon could resolve" in curve.half_life_remedy


def test_a_trend_the_surrogates_reproduce_is_refused():
    """The same discipline, on a new statistic: autocorrelated noise manufactures trends.

    An AR(1) series with no decay in it at all, fed to the fitter directly. Whatever slope least
    squares finds, a block bootstrap of the same series finds one as steep often enough that the
    fitted half-life is arithmetic rather than a measurement.
    """
    rng = np.random.default_rng(11)
    n = 24
    x, series = np.arange(n) * 10.0, np.zeros(n)
    for i in range(1, n):
        series[i] = 0.85 * series[i - 1] + rng.normal(0.0, 1.0)
    d_primes = 3.0 * np.exp(0.25 * series)  # positive, autocorrelated, no trend by construction
    out = fit_half_life(x, d_primes, n_surrogates=400, seed=2)
    if isinstance(out, Refusal):
        assert out.reason is RefusalReason.BELOW_LOD
        assert "surrogate" in out.detail or "not a decay" in out.detail
    else:
        # A run where the noise happened not to trend downward is a pass, not a silent one:
        # the surrogate rate still has to be inside budget for the fit to have been returned.
        assert out.informative and out.surrogate_rate <= 0.05


def test_a_curve_with_no_check_standard_refuses_and_does_not_fall_back():
    """The rung-2 reading has no rung-1 fallback, on purpose."""
    trace = planted_monitor_trace("no-ref", evasion_halvings=1.0, seed=1)
    stripped = MonitorTrace(
        name=trace.name,
        steps=trace.steps,
        hack_scores=trace.hack_scores,
        clean_scores=trace.clean_scores,
        in_loop_far=0.05,
    )
    out = degradation_curve(stripped)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.NO_MATCHED_CONTROL
    assert "frozen reference pool" in out.detail
    assert "re-score it with the same monitor at every evaluation step" in out.remedy
    assert "check_standard_drift" in out.remedy


# ---------------------------------------------------------------------------
# The ranking, and the kill condition
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def anti_aligned_ranking():
    curves = [
        degradation_curve(t, n_surrogates=200, seed=5)
        for t in planted_monitor_bank(10, correlated=False, seed=11)
    ]
    return rank_monitors(curves)


@pytest.fixture(scope="module")
def correlated_ranking():
    curves = [
        degradation_curve(t, n_surrogates=200, seed=5)
        for t in planted_monitor_bank(10, correlated=True, seed=11)
    ]
    return rank_monitors(curves)


def test_the_kill_fires_when_the_two_rankings_agree(correlated_ranking):
    """A bank where the best-starting monitor also decays slowest makes the figure of merit useless."""
    assert correlated_ranking.kendall_tau >= KILL_TAU
    assert correlated_ranking.kill_fired
    assert correlated_ranking.top1_agrees
    assert correlated_ranking.discordant_pairs <= 4
    assert "KILL FIRED" in correlated_ranking.render()


def test_the_kill_does_not_fire_when_they_disagree(anti_aligned_ranking):
    assert anti_aligned_ranking.kendall_tau < KILL_TAU
    assert not anti_aligned_ranking.kill_fired
    assert not anti_aligned_ranking.top1_agrees
    assert anti_aligned_ranking.total_pairs == 45


def test_a_ranking_over_unmatched_false_alarm_rates_refuses(anti_aligned_ranking):
    """E54, sharper: here the operating point sets the pressure, not only the scale."""
    from dataclasses import replace

    curves = list(anti_aligned_ranking.half_lives)
    traces = planted_monitor_bank(10, correlated=False, seed=11)
    mixed = []
    for i, t in enumerate(traces):
        far = 0.05 if i < 5 else 0.01  # a factor of five
        mixed.append(degradation_curve(replace(t, in_loop_far=far), n_surrogates=100, seed=5))
    out = rank_monitors(mixed)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.GAUGE_MISMATCH
    assert "operating point is the frame" in out.detail
    assert out.statistics["ratio"] == pytest.approx(5.0, rel=1e-6)
    assert "choose_threshold" in out.remedy
    del curves


def test_a_ranking_over_too_few_monitors_refuses():
    curves = [
        degradation_curve(t, n_surrogates=100, seed=5) for t in planted_monitor_bank(3, seed=2)
    ]
    out = rank_monitors(curves)
    assert isinstance(out, Refusal) and out.reason is RefusalReason.ESS_BELOW_FLOOR


def test_a_bank_where_nothing_decayed_refuses_rather_than_ranking_ties():
    curves = [
        degradation_curve(
            planted_monitor_trace(f"flat_{i}", evasion_halvings=0.0, seed=100 + i),
            n_surrogates=100,
            seed=5,
        )
        for i in range(5)
    ]
    out = rank_monitors(curves)
    assert isinstance(out, Refusal) and out.reason is RefusalReason.BELOW_LOD
    assert "horizon" in out.detail


def test_a_monitor_that_held_ranks_above_every_monitor_that_decayed():
    """ "Held" is an answer and it sorts. A missing value would drop the winner from its own table."""
    traces = list(planted_monitor_bank(5, seed=4))
    traces.append(planted_monitor_trace("holds", evasion_halvings=0.0, seed=99))
    curves = [degradation_curve(t, n_surrogates=100, seed=5) for t in traces]
    out = rank_monitors(curves)
    assert not isinstance(out, Refusal), getattr(out, "detail", out)
    assert out.by_half_life[0] == "holds"
    assert out.n_censored == 1


# ---------------------------------------------------------------------------
# The split, by rewarded rollouts
# ---------------------------------------------------------------------------


def _decompose(**kw):
    base_counts, trained_counts, audit = planted_family_counts(seed=5, **kw)
    return decompose_mass(
        sample_from_counts("base", base_counts, N_REWARDED),
        sample_from_counts("trained", trained_counts, N_REWARDED),
        audit,
        grader="planted-verifier",
        arm="leaky",
        n_boot=800,
        seed=2,
    )


@pytest.fixture(scope="module")
def selection_only():
    return _decompose(selected=[f"family_{i:02d}" for i in range(3)], selection_factor=3.0)


@pytest.fixture(scope="module")
def exploitation_only():
    return _decompose(novel={"exploit_new_A": 300, "exploit_new_B": 180})


def test_pure_selection_puts_everything_in_the_selection_term(selection_only):
    assert selection_only.exploitation == pytest.approx(0.0, abs=1e-12)
    assert selection_only.selection > 0
    assert selection_only.selection_share == pytest.approx(1.0, abs=1e-9)
    assert selection_only.n_families_novel == 0


def test_pure_exploitation_puts_everything_in_the_exploitation_term(exploitation_only):
    assert exploitation_only.selection == pytest.approx(0.0, abs=1e-12)
    assert exploitation_only.exploitation > 0
    assert exploitation_only.n_families_novel == 2


def test_the_split_is_an_exact_identity(selection_only, exploitation_only):
    """`selection + exploitation == mass_trained - mass_base`, to floating point."""
    for d in (
        selection_only,
        exploitation_only,
        _decompose(
            selected=[f"family_{i:02d}" for i in range(3)],
            selection_factor=2.0,
            novel={"exploit_new_A": 250},
        ),
    ):
        assert d.selection + d.exploitation == pytest.approx(
            d.mass_trained - d.mass_base, abs=1e-15
        )


def test_selection_reads_as_a_forecast_and_exploitation_reads_as_a_floor(
    selection_only, exploitation_only
):
    """The reading the row exists to produce: is the pre-training audit a forecast or a floor?"""
    assert selection_only.audit_is_a_forecast
    assert selection_only.exploitation_excess <= 0
    assert not exploitation_only.audit_is_a_forecast
    assert exploitation_only.exploitation_excess > 0


def test_the_good_turing_bound_is_not_zero_when_the_base_has_a_singleton_tail(selection_only):
    """A spectrum with no singletons bounds novel mass at exactly zero and calls everything novel."""
    assert selection_only.singletons_base > 0
    assert selection_only.good_turing_bound > 0


def test_the_horse_race_ties_when_the_audit_only_mirrors_the_base_policy():
    """An audit that says what the base policy's own errors say adds nothing, and the number says so."""
    base_counts, trained_counts, audit = planted_family_counts(
        audit_tracks="base", audit_noise=0.0, novel={"exploit_new_A": 300}, seed=5
    )
    d = decompose_mass(
        sample_from_counts("base", base_counts, N_REWARDED),
        sample_from_counts("trained", trained_counts, N_REWARDED),
        audit,
        n_boot=800,
        seed=2,
    )
    assert d.rho_audit == pytest.approx(d.rho_base, abs=1e-12)
    assert d.audit_advantage == pytest.approx(0.0, abs=1e-12)
    assert not d.audit_beats_base_policy


def test_the_horse_race_favours_an_audit_that_saw_the_novel_families():
    """And the interval is powered by the families the two orderings disagree on, not by all of them.

    Twelve novel families, because two is not enough. With two novel families out of fifty-six the
    point estimate is positive and the family-level bootstrap interval contains zero, since most
    resamples of fifty-six families draw at most one of the two the audit is right about. That is
    the correct answer and it is a design constraint rather than a bug: **the horse race's power
    comes from the number of families where the audit and the base policy disagree**, and a run
    whose training discovers only a handful of new failure modes cannot settle it however many
    rollouts it collects. Recorded in the study's notes.
    """
    novel = {f"exploit_new_{chr(65 + i)}": 240 - 10 * i for i in range(12)}
    base_counts, trained_counts, audit = planted_family_counts(
        audit_tracks="realised", audit_noise=0.0, novel=novel, seed=5
    )
    d = decompose_mass(
        sample_from_counts("base", base_counts, N_REWARDED),
        sample_from_counts("trained", trained_counts, N_REWARDED),
        audit,
        n_boot=1200,
        seed=2,
    )
    assert d.n_families_novel == 12
    assert d.audit_advantage > 0
    assert d.audit_beats_base_policy, d.audit_advantage_ci


def test_the_horse_race_interval_covers_zero_when_only_two_families_disagree():
    """The same design with two novel families instead of twelve, so the limit is visible."""
    base_counts, trained_counts, audit = planted_family_counts(
        audit_tracks="realised",
        audit_noise=0.0,
        novel={"exploit_new_A": 400, "exploit_new_B": 260},
        seed=5,
    )
    d = decompose_mass(
        sample_from_counts("base", base_counts, N_REWARDED),
        sample_from_counts("trained", trained_counts, N_REWARDED),
        audit,
        n_boot=1200,
        seed=2,
    )
    assert d.audit_advantage > 0
    assert not d.audit_beats_base_policy
    assert d.audit_advantage_ci[0] <= 0.0


def test_the_split_refuses_without_a_base_policy_sample():
    """The structural guard: without it the reading is a property of the verifier, already answered."""
    _b, trained_counts, audit = planted_family_counts(novel={"exploit_new_A": 300}, seed=5)
    out = decompose_mass(None, sample_from_counts("t", trained_counts, N_REWARDED), audit)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.NO_MATCHED_CONTROL
    assert "property of the verifier and not of the policy" in out.remedy
    assert "base_depth_for" in out.remedy


def test_the_split_refuses_on_too_few_false_positives():
    out = decompose_mass(
        sample_from_counts("base", {"a": 5, "b": 3}, 1000),
        sample_from_counts("trained", {"a": 6, "b": 4}, 1000),
        {},
    )
    assert isinstance(out, Refusal) and out.reason is RefusalReason.ESS_BELOW_FLOOR


def test_base_depth_for_is_the_cheap_lever_and_says_so():
    """Quadrupling the base sample shrinks the bound; the function inverts that for a target."""
    shallow = zipf_base(n_tail=30)
    f1 = sum(1 for v in shallow.values() if v == 1)
    n = sum(shallow.values())
    assert base_depth_for(0.0005, f1, n) >= n
    assert base_depth_for(0.0005, f1, n) == math.ceil(f1 / 0.0005)
    assert base_depth_for(0.001, f1, n) < base_depth_for(0.0001, f1, n)


# ---------------------------------------------------------------------------
# The detector is dual-use
# ---------------------------------------------------------------------------


def test_the_family_list_is_sensitive_on_the_payload_and_on_the_evidence_row():
    """Both halves, because they fail independently: `redact` reads one and the store reads the other."""
    base_counts, trained_counts, audit = planted_family_counts(novel={"exploit_new_A": 300}, seed=5)
    inst = AuditedFamilyMass(
        sample_from_counts("base", base_counts, N_REWARDED),
        sample_from_counts("trained", trained_counts, N_REWARDED),
        audit,
        grader="planted-verifier",
        arm="leaky",
        n_boot=200,
    )
    out = inst.estimate(Context())
    assert not isinstance(out, Refusal), getattr(out, "detail", "")
    assert out.value.sensitive is True
    assert is_sensitive(out.value) and is_sensitive(out)
    assert out.subject.extra.get("sensitive") == "true"
    # E35: the quantity reaches the row rather than being dropped on the no-signal path.
    assert out.quantity == "verifier.fp_catalogue"
    assert out.observable == "AuditedFamilyMass"


def test_every_emitted_payload_round_trips_through_the_value_codec():
    """A payload the store cannot decode is a row that reads back as a dict (E28's shape)."""
    from reward_lens.core.evidence import ValueCodec

    codec = ValueCodec()
    trace = planted_monitor_trace("codec", evasion_halvings=1.0, degradation_halvings=0.5, seed=3)
    curve = MonitorDegradation(trace, n_surrogates=100).estimate(Context())
    assert not isinstance(curve, Refusal), getattr(curve, "detail", "")
    back = codec.decode(codec.encode(curve.value))
    assert type(back) is type(curve.value)
    assert back.monitor == curve.value.monitor
    assert back.half_life.half_life_steps == pytest.approx(curve.value.half_life_steps)
    assert len(back.points) == len(curve.value.points)


def test_the_rendered_form_withholds_the_family_list(exploitation_only):
    text = exploitation_only.render()
    assert "exploit_new_A" not in text
    assert "families withheld" in text
    assert "exploit_new_A" in exploitation_only.render(include_families=True)


def test_redaction_drops_the_families_and_keeps_every_number(exploitation_only):
    reduced = redact(exploitation_only)
    assert reduced.by_family == {}
    assert reduced.novel_families == ()
    assert reduced.sensitive is False
    assert reduced.withheld == len(exploitation_only.by_family)
    for field in ("selection", "exploitation", "good_turing_bound", "rho_audit", "rho_base"):
        assert getattr(reduced, field) == getattr(exploitation_only, field)


def test_publishing_the_family_list_needs_a_recorded_decision(exploitation_only):
    with pytest.raises(DisclosureRequired):
        exploitation_only.for_publication()
    decided = exploitation_only.with_disclosure(
        DisclosureDecision(decided_by="maintainer", reason="paper appendix", scope="published")
    )
    assert decided.for_publication() is decided


# ---------------------------------------------------------------------------
# The studies: frozen, closed, and honestly void
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("plan_fn", [w6_4_plan, w6_5_plan], ids=["w6-4", "w6-5"])
def test_the_plan_closes(plan_fn):
    """Every registered prediction and every kill criterion is produced by some arc."""
    report = check_closure(plan_fn())
    assert report.closed and not report.gaps
    assert report.order


@pytest.mark.parametrize("freeze_fn", [freeze_w6_4, freeze_w6_5], ids=["w6-4", "w6-5"])
def test_the_spec_freezes_and_the_id_is_stable(freeze_fn):
    a, b = freeze_fn(), freeze_fn()
    assert a.study_id == b.study_id
    assert a.spec_hash == b.spec_hash
    assert a.study_id.startswith("study:")


@pytest.mark.parametrize("spec_fn", [w6_4_spec, w6_5_spec], ids=["w6-4", "w6-5"])
def test_every_hypothesis_and_kill_criterion_is_mechanically_evaluable(spec_fn):
    spec = spec_fn()
    assert spec.hypotheses and spec.kill_criteria
    for h in spec.hypotheses:
        assert h.prediction.rationale.strip(), h.id
        assert h.prediction.check(h.prediction.threshold + 1.0) in (True, False)
    for k in spec.kill_criteria:
        assert k.description.strip(), k.id
        assert k.fired(k.threshold) in (True, False)


@pytest.mark.parametrize("analyse", [analyse_w6_4, analyse_w6_5], ids=["w6-4", "w6-5"])
def test_the_analysis_voids_with_a_named_reason_rather_than_returning_zeros(analyse):
    """A study that has not been bought is void per metric, not confirmed and not silently empty."""
    result = analyse()
    assert result.outcome is StudyOutcome.VOID
    assert set(result.outcomes.values()) == {"void"}
    assert set(result.kill_outcomes.values()) == {"void"}
    assert result.voids
    for void in result.voids.values():
        assert "no run exists" in void.detail
        assert void.remedy.strip()


def test_the_price_is_arithmetic_from_one_stated_constant():
    from reward_lens.studies.w6_monitor.price import (
        RATE_HIGH,
        RATE_LOW,
        W6_4_PRICE,
        W6_5_PRICE,
        ranked,
    )
    from reward_lens.studies.w6_monitor.study import GPU_HOURS_PER_ARM

    assert GPU_HOURS_PER_ARM == pytest.approx(11_520.0 / 30.0)
    for price in (W6_4_PRICE, W6_5_PRICE):
        assert price.gpu_hours == pytest.approx(price.arms * GPU_HOURS_PER_ARM)
        assert price.dollars_low == pytest.approx(price.gpu_hours * RATE_LOW)
        assert price.dollars_high == pytest.approx(price.gpu_hours * RATE_HIGH)
        assert price.assumptions
    # The stated preemptible floor is above the top of the stated dollar range.
    assert 11_520.0 * RATE_HIGH > 23_000.0
    assert ranked()[0] is W6_4_PRICE


# ---------------------------------------------------------------------------
# The power figures the two specs quote, recomputed
# ---------------------------------------------------------------------------


def _tau_power(m: int, true_tau: float, kill: float = KILL_TAU, reps: int = 1500, seed: int = 0):
    """P(the kill does not fire) when the true rank association between the two rankings is `tau`."""
    from scipy.stats import kendalltau

    rho = math.sin(true_tau * math.pi / 2)
    chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(reps):
        z = rng.normal(size=(m, 2)) @ chol.T
        if kendalltau(z[:, 0], z[:, 1]).statistic < kill:
            hits += 1
    return hits / reps


# Slower than the rest of this file and deliberately kept in the default run: these are the
# numbers the two study specs quote, and a spec quoting a figure no test reproduces is the thing
# rule 2 exists to stop.
def test_the_w6_5_kill_condition_is_powered_at_ten_monitors():
    """The catalogue states the kill over ten monitors; ten is checked rather than assumed.

    Reproduces the figures in `W6_5_NOTES`. The tolerances are loose because this is 1,500 draws
    against the notes' 4,000, which is the trade for a test that runs in the suite.
    """
    assert _tau_power(10, 0.2, seed=1) > 0.97
    assert _tau_power(10, 0.4, seed=2) > 0.95
    assert _tau_power(10, 0.6, seed=3) > 0.85
    # Twenty monitors buy back the 0.6 cell, which is what the notes say to do if it matters.
    assert _tau_power(20, 0.6, seed=4) > _tau_power(10, 0.6, seed=3)
    # And when the two rankings are truly identical the kill fires essentially always.
    assert _tau_power(10, 0.999, reps=400, seed=5) < 0.02


# Slower than the rest of this file and deliberately kept in the default run: these are the
# numbers the two study specs quote, and a spec quoting a figure no test reproduces is the thing
# rule 2 exists to stop.
def test_the_w6_5_split_resolves_to_the_precision_the_spec_claims():
    """400 rollouts per class per step resolves the evasion share to about +/- 0.03."""
    shares, half_lives = [], []
    for s in range(25):
        trace = planted_monitor_trace(
            "m", evasion_halvings=1.1, degradation_halvings=1.1, n_rollouts=400, seed=1000 + s
        )
        curve = degradation_curve(trace, n_surrogates=80, seed=s)
        if np.isfinite(curve.evasion_share):
            shares.append(curve.evasion_share)
        if curve.half_life is not None:
            half_lives.append(curve.half_life_steps)

    shares, half_lives = np.array(shares), np.array(half_lives)
    assert shares.mean() == pytest.approx(0.5, abs=0.03)
    assert shares.std() < 0.05
    truth = 190.0 / 2.2
    assert half_lives.mean() == pytest.approx(truth, rel=0.05)
    assert half_lives.std() / half_lives.mean() < 0.05


# Slower than the rest of this file and deliberately kept in the default run: these are the
# numbers the two study specs quote, and a spec quoting a figure no test reproduces is the thing
# rule 2 exists to stop.
def test_the_w6_4_exploitation_bound_is_calibrated_and_the_base_depth_is_the_lever():
    """Two claims in `W6_4_NOTES`: the bound does not bias, and depth is what buys resolution.

    Both samples are drawn multinomially from one true family distribution, so under the null the
    novel mass is entirely a sampling artefact and the excess should sit at zero. Under a planted 5%
    novel fraction it should recover that fraction's mass.
    """
    true = zipf_base()
    families = list(true)
    p = np.array([true[f] for f in families], dtype=np.float64)
    p /= p.sum()

    def simulate(n_base: int, novel_frac: float, reps: int = 80, seed: int = 0):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(reps):
            base = Counter(dict(zip(families, rng.multinomial(n_base, p))))
            n_novel = int(round(1000 * novel_frac))
            trained = Counter(dict(zip(families, rng.multinomial(1000 - n_novel, p))))
            if n_novel:
                trained["exploit_new_A"] = n_novel
            d = decompose_mass(
                sample_from_counts("b", base, N_REWARDED),
                sample_from_counts("t", trained, N_REWARDED),
                dict(true),
                n_boot=1,
            )
            out.append(d.exploitation_excess)
        return np.array(out)

    null_shallow = simulate(500, 0.0, seed=1)
    null_deep = simulate(2000, 0.0, seed=2)
    planted = simulate(2000, 0.05, seed=3)

    # Unbiased under the null, at both depths.
    assert abs(null_shallow.mean()) < 0.0005
    assert abs(null_deep.mean()) < 0.0005
    # Depth is the lever: quadrupling the base sample tightens the null by a large factor.
    assert null_deep.std() < null_shallow.std() / 2.0
    # And a planted novel mass comes back at its planted size.
    assert planted.mean() == pytest.approx(0.05 * 1000 / N_REWARDED, rel=0.15)

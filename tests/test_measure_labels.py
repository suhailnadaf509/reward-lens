"""Unit, property and invariance tests for series L.

The hand-computed cases are the ones that matter here, and there are three of them: the Wilson
interval against a published table, the chain composition on numbers that add to a whole number, and
the stratified rank statistic on a table small enough to count by hand. Everything else in this
package is arithmetic on top of those three, so if they are right the rest is checkable by property.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from reward_lens.core.gates import CalibrationRef, compute_trust
from reward_lens.core.invariance import (
    InvariancePayload,
    check_invariance,
    check_unit_refusal,
)
from reward_lens.core.quantity import QUANTITIES, ladder, load_quantities
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.reference import MatrixDescription, ReferenceMaterial, Transfer
from reward_lens.core.types import TrustLevel
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.labels import (
    INSTRUMENTS,
    AuditSample,
    LabelErrorAudit,
    LocalisationSeries,
    Plant,
    PositionStratifiedNull,
    ReferenceCertificate,
    SignalLocalityText,
    StabilityPoint,
    TwoSidedVerifierError,
    Verdict,
    audit_error_rate,
    between_seed_homogeneity,
    certify,
    compose_chain,
    dose_response_characterisation,
    independent_rater_rate,
    irt_surface,
    register_all,
    rescore_against_position,
    signal_locality,
    stability_uncertainty,
    stratified_auc,
    two_rater_bounds,
    two_sided_error,
    wilson_interval,
)
from reward_lens.measure.labels.reference import compare_u_char

load_quantities()
register_all()

SLOW = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])


# ---------------------------------------------------------------------------
# The Wilson interval, against a published table
# ---------------------------------------------------------------------------

#: `(k, n, low, high)` at 95%. These four are the ones that appear as worked examples wherever the
#: Wilson interval is introduced, and each was checked against the arithmetic by hand before it was
#: written here. The stronger check is `test_wilson_endpoints_solve_the_defining_equation` below: a
#: table can be mistranscribed and the defining equation cannot.
PUBLISHED_WILSON = [
    (0, 20, 0.0000, 0.1611),
    (2, 10, 0.0567, 0.5098),
    (3, 10, 0.1078, 0.6032),
    (5, 10, 0.2366, 0.7634),
]


@pytest.mark.parametrize("k,n,low,high", PUBLISHED_WILSON)
def test_wilson_matches_published_table(k, n, low, high):
    p = wilson_interval(k, n)
    assert p.low == pytest.approx(low, abs=5e-5)
    assert p.high == pytest.approx(high, abs=5e-5)
    assert p.point == pytest.approx(k / n)


@given(
    n=st.integers(min_value=2, max_value=5000),
    frac=st.floats(min_value=0.001, max_value=0.999),
)
@SLOW
def test_wilson_endpoints_solve_the_defining_equation(n, frac):
    """The Wilson interval is the set of `p` with `|p_hat - p| <= z*sqrt(p(1-p)/n)`.

    So each endpoint that is not on a boundary satisfies that with equality, and checking it is a
    check on the derivation rather than on a remembered table.
    """
    k = int(round(frac * n))
    assume(0 < k < n)
    p = wilson_interval(k, n)
    z = 1.959963984540054
    p_hat = k / n
    for end in (p.low, p.high):
        if 0.0 < end < 1.0:
            assert abs(p_hat - end) == pytest.approx(
                z * math.sqrt(end * (1.0 - end) / n), rel=1e-9, abs=1e-12
            )


def test_wilson_zero_errors_is_not_zero_uncertainty():
    """The case the whole choice of interval is about: 0 wrong out of 20 is not a perfect label set."""
    p = wilson_interval(0, 20)
    assert p.point == 0.0
    assert p.high > 0.16
    assert "0.1611" in f"{p.high:.4f}"


def test_wilson_unaudited_is_the_whole_line():
    p = wilson_interval(0, 0)
    assert math.isnan(p.point)
    assert (p.low, p.high) == (0.0, 1.0)
    assert not p.is_measured


def test_wilson_rejects_an_untabulated_level():
    with pytest.raises(ValueError, match="refuses to approximate"):
        wilson_interval(1, 10, level=0.975)


@given(
    n=st.integers(min_value=1, max_value=5000),
    frac=st.floats(min_value=0.0, max_value=1.0),
)
@SLOW
def test_wilson_contains_the_point_and_stays_in_the_unit_interval(n, frac):
    k = int(round(frac * n))
    p = wilson_interval(k, n)
    assert 0.0 <= p.low <= p.point <= p.high <= 1.0


@given(frac=st.floats(min_value=0.05, max_value=0.95))
@SLOW
def test_wilson_narrows_with_the_sample(frac):
    small = wilson_interval(int(round(frac * 50)), 50)
    large = wilson_interval(int(round(frac * 5000)), 5000)
    assert large.width < small.width


# ---------------------------------------------------------------------------
# The chain composition, on numbers that add up
# ---------------------------------------------------------------------------


def test_u_crm_composes_in_quadrature_exactly():
    """3, 4, 12 is a Pythagorean quadruple: 9 + 16 + 144 = 169, so u_CRM is exactly 0.13."""
    m = ReferenceMaterial(
        id="triple",
        kind="planted_organism",
        assigned_value=0.75,
        u_characterisation=0.03,
        u_homogeneity=0.04,
        u_stability=0.12,
        matrix=MatrixDescription(system="test"),
    )
    assert m.is_certified
    assert m.u_crm == pytest.approx(0.13, abs=1e-12)
    assert m.u_crm_lower_bound == pytest.approx(0.13, abs=1e-12)


def test_u_crm_is_none_when_a_term_was_never_measured():
    m = ReferenceMaterial(
        id="partial",
        kind="planted_organism",
        assigned_value=0.75,
        u_characterisation=0.03,
        u_homogeneity=0.04,
        matrix=MatrixDescription(system="test"),
    )
    assert m.u_crm is None
    assert m.u_crm_lower_bound == pytest.approx(0.05, abs=1e-12)
    assert m.uncharacterised == ("u_stability",)


def test_chain_total_composes_with_the_transfers():
    """u_total^2 = u_working^2 + t32^2 + u_char^2 + u_bb^2 + u_stab^2 + u_instrument^2."""
    cert = certify(
        [Plant(d, d, seed=s) for d in (0.6, 0.8, 1.0) for s in (0, 1, 2)],
        reference_id="exact",
        assigned_dose=0.8,
        matrix=MatrixDescription(system="test"),
        stability=[StabilityPoint(steps=t, response=0.8) for t in (0, 100, 200)],
    )
    chain = compose_chain(
        cert,
        transfers=(Transfer(from_level="primary", to_level="reference_method", value=0.419),),
        u_working=0.1,
        u_instrument=0.05,
    )
    assert chain.is_certified
    material = cert.material()
    expected = math.sqrt(0.1**2 + 0.419**2 + material.u_crm**2 + 0.05**2)
    assert chain.u_total == pytest.approx(expected, rel=1e-12)


def test_chain_refuses_a_total_when_the_reference_is_uncertified():
    cert = certify(
        [Plant(d, d, seed=0) for d in (0.6, 0.8, 1.0)],
        reference_id="single-seed",
        assigned_dose=0.8,
        matrix=MatrixDescription(system="test"),
    )
    chain = compose_chain(cert, u_working=0.1)
    assert chain.u_total is None
    assert chain.u_total_lower_bound > 0.0
    assert "not characterised" in chain.render()


# ---------------------------------------------------------------------------
# L1, the certificate
# ---------------------------------------------------------------------------


def test_dose_response_fit_on_an_exact_line():
    """A line with no scatter has zero residual, so u_char is zero and the fit says so."""
    fit = dose_response_characterisation([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0])
    assert fit.slope == pytest.approx(2.0)
    assert fit.intercept == pytest.approx(1.0)
    assert fit.s_resid == pytest.approx(0.0, abs=1e-12)
    assert fit.u_char == pytest.approx(0.0, abs=1e-12)
    assert fit.dof == 2


def test_dose_response_fit_with_a_hand_computed_residual():
    """Three points, one residual dof. Residuals are (-1, 2, -1)/3 scaled by the fit.

    `u_char` used to be the bare ratio `s_resid/|slope| = 0.408248`, which is a scatter about the
    line rather than the uncertainty on a dose read back off it. It is now the inverse-prediction
    uncertainty at the sweep centre in its individual form, `sqrt(1 + 1/3)` times that, or
    0.471405. The bare ratio survives as `sensitivity_scaled_residual`, and `u_char_at` gives the
    value at an arbitrary dose in either form.
    """
    fit = dose_response_characterisation([0.0, 1.0, 2.0], [0.0, 3.0, 4.0])
    assert fit.slope == pytest.approx(2.0)
    assert fit.intercept == pytest.approx(1.0 / 3.0)
    # residuals: -1/3, 2/3, -1/3 -> SSE = 1/9 + 4/9 + 1/9 = 6/9
    assert fit.s_resid == pytest.approx(math.sqrt(2.0 / 3.0))
    assert fit.sensitivity_scaled_residual == pytest.approx(math.sqrt(2.0 / 3.0) / 2.0)
    assert fit.sensitivity_scaled_residual == pytest.approx(0.408248290463863)
    assert fit.u_char == pytest.approx(0.4714045207910318)
    assert fit.u_char == pytest.approx(fit.sensitivity_scaled_residual * math.sqrt(4.0 / 3.0))
    assert "one residual degree of freedom" in fit.note


def test_u_char_at_takes_the_form_that_matches_what_else_the_certificate_reports():
    """The two inverse-prediction forms, hand-checked on the same three points.

    The sweep is 0, 1, 2 so `xbar = 1` and `Sxx = 2`. At the centre the leverage term vanishes and
    the two forms are `sqrt(1 + 1/3)` and `sqrt(1/3)` times the ratio. At dose 2 the leverage term
    is `1/2`, which is larger than either, and it is what makes reading a dose off the end of a
    sweep worse than reading one off the middle.
    """
    fit = dose_response_characterisation([0.0, 1.0, 2.0], [0.0, 3.0, 4.0])
    base = fit.sensitivity_scaled_residual
    assert fit.u_char_at(1.0, individual=True) == pytest.approx(base * math.sqrt(1.0 + 1.0 / 3.0))
    assert fit.u_char_at(1.0, individual=False) == pytest.approx(base * math.sqrt(1.0 / 3.0))
    assert fit.u_char_at(2.0, individual=True) == pytest.approx(
        base * math.sqrt(1.0 + 1.0 / 3.0 + 0.5)
    )
    assert fit.u_char_at(2.0, individual=False) == pytest.approx(base * math.sqrt(1.0 / 3.0 + 0.5))
    # The centre is where the characterisation is best, which is the fact the old deviation note
    # got right and then attached to the wrong term.
    assert fit.u_char_at(1.0, individual=True) < fit.u_char_at(2.0, individual=True)


def test_dose_response_refuses_a_flat_readout():
    fit = dose_response_characterisation([0.5, 0.7, 0.9], [1.0, 1.0, 1.0])
    assert not math.isfinite(fit.u_char)
    assert "does not move" in fit.note


def test_two_dose_levels_do_not_characterise():
    fit = dose_response_characterisation([0.5, 1.0], [1.0, 2.0])
    assert not fit.is_usable
    assert "leaves no residual" in fit.note


def test_homogeneity_is_none_below_three_seeds():
    plants = [Plant(0.8, 0.8 + 0.01 * s, seed=s) for s in (0, 1)]
    value, detail = between_seed_homogeneity(plants)
    assert value is None
    assert detail["min_seeds_required"] == 3
    assert "siblings" in detail["why_none"]


def test_homogeneity_pools_within_dose_by_degrees_of_freedom():
    """Two dose levels, three seeds each, spreads chosen so the pooled sd is exactly 1."""
    plants = [Plant(0.6, v, seed=i) for i, v in enumerate((-1.0, 0.0, 1.0))]
    plants += [Plant(1.0, v, seed=i) for i, v in enumerate((9.0, 10.0, 11.0))]
    value, detail = between_seed_homogeneity(plants)
    # each group has sum of squared deviations 2, pooled over 4 dof -> sqrt(4/4) = 1
    assert value == pytest.approx(1.0)
    assert detail["pooled_dof"] == 4
    assert detail["n_dose_levels_used"] == 2


def test_homogeneity_does_not_pool_across_pipelines():
    plants = [Plant(0.8, v, seed=i, pipeline="a") for i, v in enumerate((0.0, 0.1, 0.2))]
    plants += [Plant(0.8, v, seed=i, pipeline="b") for i, v in enumerate((5.0, 5.1, 5.2))]
    value, detail = between_seed_homogeneity(plants)
    assert detail["n_pipelines"] == 2
    assert detail["n_dose_levels_used"] == 2
    # Pooling across the pipelines would give a spread near 2.5 rather than near 0.1.
    assert value == pytest.approx(0.1, abs=1e-9)


def test_stability_is_none_below_three_checkpoints():
    value, detail = stability_uncertainty([StabilityPoint(0, 0.8), StabilityPoint(100, 0.8)])
    assert value is None
    assert "re-measure" in detail["why_none"]


def test_stability_is_not_zero_when_no_drift_was_found():
    """The point of the isochronous form: 'we looked and saw nothing' is small, not zero."""
    points = [StabilityPoint(t, 0.8 + 0.001 * (i % 2)) for i, t in enumerate((0, 100, 200, 300))]
    value, detail = stability_uncertainty(points, horizon_steps=200)
    assert value is not None
    assert value > 0.0
    assert detail["horizon_steps"] == 200


def test_certify_refuses_with_no_plants():
    refusal = certify(
        [], reference_id="empty", assigned_dose=0.8, matrix=MatrixDescription(system="x")
    )
    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.REFERENCE_UNCERTIFIED
    assert "three or more dose levels" in refusal.remedy


def test_certify_refuses_mixed_response_kinds():
    plants = [Plant(0.6, 0.6, seed=0), Plant(0.8, 0.9, seed=0, response_kind="auc")]
    refusal = certify(
        plants, reference_id="mixed", assigned_dose=0.7, matrix=MatrixDescription(system="x")
    )
    assert isinstance(refusal, Refusal)
    assert "one response kind at a time" in refusal.remedy


# ---------------------------------------------------------------------------
# The trust cap, which is the rule the series exists to enforce
# ---------------------------------------------------------------------------


def _uncertified() -> ReferenceMaterial:
    return ReferenceMaterial(
        id="uncertified",
        kind="planted_organism",
        assigned_value=0.75,
        u_characterisation=0.06,
        matrix=MatrixDescription(system="test"),
    )


def _certified() -> ReferenceMaterial:
    return ReferenceMaterial(
        id="certified",
        kind="planted_organism",
        assigned_value=0.75,
        u_characterisation=0.06,
        u_homogeneity=0.14,
        u_stability=0.02,
        matrix=MatrixDescription(system="test"),
    )


def test_compute_trust_caps_at_calibrated_when_homogeneity_is_uncharacterised():
    cal = CalibrationRef(scorecard_entry="ev:x", organism_family="fam")
    assert compute_trust(calibration=cal, registered=True) is TrustLevel.REGISTERED
    assert (
        compute_trust(calibration=cal, registered=True, reference=_uncertified())
        is TrustLevel.CALIBRATED
    )
    assert (
        compute_trust(calibration=cal, registered=True, adjudicated=True, reference=_uncertified())
        is TrustLevel.CALIBRATED
    )


def test_compute_trust_lifts_when_the_reference_is_certified():
    cal = CalibrationRef(scorecard_entry="ev:x", organism_family="fam")
    assert (
        compute_trust(calibration=cal, registered=True, reference=_certified())
        is TrustLevel.REGISTERED
    )
    assert (
        compute_trust(calibration=cal, registered=True, adjudicated=True, reference=_certified())
        is TrustLevel.ADJUDICATED
    )


def test_the_cap_reaches_the_emitted_evidence():
    inst = ReferenceCertificate(
        [Plant(d, d, seed=0) for d in (0.6, 0.8, 1.0)],
        reference_id="single-seed",
        assigned_dose=0.8,
        matrix=MatrixDescription(system="test"),
    )
    reading = inst.estimate(Context(study="study:frozen@v1#abc"))
    assert reading.trust is TrustLevel.CALIBRATED
    assert reading.reference == "single-seed"
    assert reading.value.u_homogeneity is None


def test_the_cap_lifts_on_the_emitted_evidence_when_all_three_exist():
    inst = ReferenceCertificate(
        [Plant(d, d + 0.001 * s, seed=s) for d in (0.6, 0.8, 1.0) for s in (0, 1, 2)],
        reference_id="three-seed",
        assigned_dose=0.8,
        matrix=MatrixDescription(system="test"),
        stability=[StabilityPoint(t, 0.8 + 1e-4 * i) for i, t in enumerate((0, 100, 200))],
    )
    reading = inst.estimate(Context(study="study:frozen@v1#abc"))
    assert reading.trust is TrustLevel.REGISTERED
    assert reading.value.u_homogeneity is not None
    assert reading.value.u_stability is not None
    assert reading.value.rung == 2


# ---------------------------------------------------------------------------
# L2
# ---------------------------------------------------------------------------


def test_ceiling_is_one_minus_the_error_rate_with_the_interval_reflected():
    result = audit_error_rate(AuditSample(n_audited=200, n_wrong=8), reported_score=0.97)
    assert result.error_rate.point == pytest.approx(0.04)
    assert result.ceiling == pytest.approx(0.96)
    assert result.ceiling_low == pytest.approx(1.0 - result.error_rate.high)
    assert result.ceiling_high == pytest.approx(1.0 - result.error_rate.low)
    assert result.headroom == pytest.approx(0.97 - 0.96)
    assert "part of what it is matching is the annotation error" in result.interpretation


def test_measured_rate_converts_to_a_label_quality_the_record_layer_accepts():
    result = audit_error_rate(
        AuditSample(n_audited=200, n_wrong=8, method="two raters", measured_by="pool-a")
    )
    quality = result.as_label_quality()
    assert quality.is_measured
    assert quality.error_rate == pytest.approx(0.04)
    assert quality.n_audited == 200


def test_two_rater_bounds_report_the_independence_model_and_not_half_the_disagreement():
    """This test used to assert the defect.

    It read `assert result.error_rate.point == pytest.approx(0.05)` on a disagreement rate of 0.10,
    which pinned the point estimate to `d/2`, the infimum of the identified set and the most
    flattering value in it. `d/2` is a lower bound on the *average* of the two raters' rates and is
    not an estimate of anything. The point is now the independence model, `(1 - sqrt(1-2d))/2`,
    which at `d = 0.10` is 0.052786 rather than 0.050000, and `d/2` survives as `low`.
    """
    a = [0] * 100
    b = [0] * 90 + [1] * 10
    result = two_rater_bounds(a, b)
    assert result.error_rate.k == 10
    assert result.error_rate.point == pytest.approx((1.0 - math.sqrt(0.8)) / 2.0)
    assert result.error_rate.point == pytest.approx(0.052786404500042)
    assert result.error_rate.point > 0.05, "the point must not sit at d/2, the optimistic end"
    assert result.error_rate.low == pytest.approx(wilson_interval(10, 100).low / 2.0)
    assert result.rung == 2
    assert "shared bias" in result.interpretation
    assert not result.identified


def test_the_two_rater_bound_names_the_average_and_not_either_rater():
    """The claim that was false: `d/2` bounds the average, not an individual rater.

    Rater A is perfect and rater B is wrong on a fifth of the items. The disagreement rate is 0.20,
    `d/2` is 0.10, and A's true rate is 0.00. Any wording that claims every rater is at least `d/2`
    is refuted by this fixture, so the assertion is on what the module says as well as on what it
    computes.
    """
    truth = [0] * 1000
    a = list(truth)
    b = [1] * 200 + [0] * 800
    result = two_rater_bounds(a, b)
    e_a = sum(1 for x, t in zip(a, truth) if x != t) / len(truth)
    e_b = sum(1 for x, t in zip(b, truth) if x != t) / len(truth)
    d = result.error_rate.k / result.error_rate.n
    assert d == pytest.approx(0.20)
    assert e_a == 0.0 and e_b == pytest.approx(0.20)
    # The average is bounded below by d/2, exactly here because the two never agree on a wrong one.
    assert 0.5 * (e_a + e_b) == pytest.approx(d / 2.0)
    # An individual rater is not.
    assert e_a < d / 2.0
    assert "average of the two raters' error rates" in two_rater_bounds.__doc__
    assert "*average* of the two raters' error rates" in result.interpretation
    assert "says nothing about either rater on its own" in result.interpretation


def test_a_bounding_rung_does_not_license_a_scoring_read():
    """The export that matters: `LabelQuality.error_rate` gates `record.labels.adjudicate`."""
    a = [0] * 2000
    b = [0] * 1300 + [1] * 700
    result = two_rater_bounds(a, b)
    assert result.error_rate.point == pytest.approx(0.2261387212474169)
    quality = result.as_label_quality()
    assert quality.error_rate is None
    assert not quality.is_measured
    assert quality.n_audited == 2000
    assert "bounds the error rate rather than identifying it" in quality.method

    # Rung 0 still licenses it, which is the point of keeping the distinction.
    audited = audit_error_rate(AuditSample(n_audited=200, n_wrong=8)).as_label_quality()
    assert audited.is_measured and audited.error_rate == pytest.approx(0.04)


def test_the_independence_model_has_no_root_above_half_and_says_so():
    assert math.isnan(independent_rater_rate(0.6))
    assert independent_rater_rate(0.0) == 0.0
    assert independent_rater_rate(0.5) == pytest.approx(0.5)
    # And it is always at least d/2 and at most d on its own domain, so the point sits inside the
    # two bounds rather than outside them.
    for d in (0.01, 0.05, 0.1, 0.2, 0.35, 0.49):
        e = independent_rater_rate(d)
        assert d / 2.0 <= e <= d


def test_irt_surfaces_planted_mislabels_and_finds_none_when_there_are_none():
    rng = np.random.default_rng(7)
    n_raters, n_items = 40, 300
    ability = rng.normal(0.0, 1.0, n_raters)
    difficulty = rng.normal(0.0, 1.0, n_items)
    p = 1.0 / (1.0 + np.exp(-(ability[:, None] - difficulty[None, :])))
    clean = (rng.random((n_raters, n_items)) < p).astype(int)

    flipped = set(rng.choice(n_items, size=15, replace=False).tolist())
    dirty = clean.copy()
    dirty[:, sorted(flipped)] = 1 - dirty[:, sorted(flipped)]

    surfaced = irt_surface(dirty)
    negative = set(np.flatnonzero(np.asarray(surfaced.discrimination) < 0).tolist())
    assert surfaced.converged
    assert negative
    # Precision, which is what the rung claims: a surfaced item is a mislabel.
    assert len(negative & flipped) / len(negative) >= 0.9

    null = irt_surface(clean)
    assert null.n_negative <= 1


def test_label_error_audit_refuses_with_its_reason_and_remedy():
    reading = LabelErrorAudit(None, label_set="a-benchmark").estimate()
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.LABEL_QUALITY_UNKNOWN
    assert "AuditSample" in reading.remedy
    assert "assuming it is 1.0" in reading.remedy


@given(
    n=st.integers(min_value=1, max_value=2000),
    frac=st.floats(min_value=0.0, max_value=1.0),
    score=st.floats(min_value=0.0, max_value=1.0),
)
@SLOW
def test_ceiling_is_always_one_minus_the_rate(n, frac, score):
    k = int(round(frac * n))
    result = audit_error_rate(AuditSample(n_audited=n, n_wrong=k), reported_score=score)
    assert result.ceiling == pytest.approx(1.0 - result.error_rate.point)
    assert result.headroom == pytest.approx(score - result.ceiling)


# ---------------------------------------------------------------------------
# L3
# ---------------------------------------------------------------------------


def test_two_sided_error_uses_the_right_denominator_for_each_direction():
    verdicts = [
        Verdict(f"z{i}", verifier_score=0, agent_claimed_success=(i < 36)) for i in range(100)
    ]
    verdicts += [
        Verdict(f"o{i}", verifier_score=1, procedure_violated=(i < 27)) for i in range(100)
    ]
    result = two_sided_error(verdicts, corpus="test")
    assert result.n_zeros == 100 and result.n_ones == 100
    assert result.false_success_rate.point == pytest.approx(0.36)
    assert result.false_success_rate.n == 100
    assert result.corrupt_success_rate.point == pytest.approx(0.27)
    assert result.is_two_sided
    assert "wrong in both directions" in result.interpretation


def test_a_missing_audit_column_is_excluded_rather_than_scored_as_a_negative():
    verdicts = [Verdict(f"o{i}", verifier_score=1) for i in range(50)]
    verdicts += [Verdict("o-audited", verifier_score=1, procedure_violated=True)]
    result = two_sided_error(verdicts)
    assert result.corrupt_success_rate.n == 1
    assert result.corrupt_success_rate.point == pytest.approx(1.0)
    assert result.n_missing_violation == 50


def test_verifier_error_refuses_a_scalar_only_corpus():
    reading = TwoSidedVerifierError(
        [Verdict(f"t{i}", verifier_score=i % 2) for i in range(20)], corpus="scalar-only"
    ).estimate()
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.LABEL_QUALITY_UNKNOWN
    assert "agent_claimed_success" in reading.remedy
    assert "procedure_violated" in reading.remedy


def test_verifier_score_must_be_binary():
    with pytest.raises(ValueError, match="binary verdict"):
        Verdict("t", verifier_score=2)


# ---------------------------------------------------------------------------
# L4
# ---------------------------------------------------------------------------


def _locality_corpus(n: int = 200, *, terminal: bool, seed: int = 0):
    """A corpus whose tell is either only in the closing message or spread through the body."""
    rng = np.random.default_rng(seed)
    words = ("alpha", "beta", "gamma", "delta", "epsilon")
    texts, closings, labels = [], [], []
    for i in range(n):
        y = int(i % 2)
        body = " ".join(rng.choice(words, size=30))
        if y and not terminal:
            body += " " + " ".join(["telltale"] * 6)
        closing = "verdict: success" if y else "verdict: incomplete"
        texts.append(body + "\n\n" + closing)
        closings.append(closing)
        labels.append(y)
    return texts, closings, labels


def test_locality_reports_terminal_when_the_tell_is_only_at_the_end():
    texts, closings, labels = _locality_corpus(terminal=True)
    result = signal_locality(texts, labels, closings=closings, corpus="terminal")
    assert result.closing_only_auc > 0.95
    assert result.excluding_closing_auc < 0.6
    assert result.verdict == "terminal"
    assert result.delta < -0.3


def test_locality_reports_distributed_when_the_body_carries_it_too():
    texts, closings, labels = _locality_corpus(terminal=False)
    result = signal_locality(texts, labels, closings=closings, corpus="distributed")
    assert result.closing_only_auc > 0.95
    assert result.excluding_closing_auc > 0.95
    assert result.verdict == "distributed"
    assert abs(result.delta) < 0.1


def test_locality_refuses_a_single_class_corpus():
    reading = SignalLocalityText(["a\n\nb"] * 10, [1] * 10, corpus="one-class").estimate()
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.LABEL_QUALITY_UNKNOWN
    assert "both classes" in reading.remedy


def test_locality_refuses_an_empty_corpus():
    reading = SignalLocalityText([], [], corpus="empty").estimate()
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.LABEL_QUALITY_UNKNOWN
    assert "audit column rather than the verifier's scalar" in reading.remedy


def test_the_white_box_rung_is_registered_and_unimplemented():
    rungs = ladder("labels.fs_signal_locality")
    assert [e.rung for e in rungs] == [0, 1]
    assert rungs[0].run is not None
    assert rungs[1].run is None
    assert "residual_stream" in rungs[1].impl
    assert rungs[1].cost.gpu_seconds


# ---------------------------------------------------------------------------
# L5
# ---------------------------------------------------------------------------


def test_stratified_auc_by_hand():
    """Two strata. Stratum 0: one positive above one negative. Stratum 1: one positive below two."""
    scores = np.array([2.0, 1.0, 0.0, 1.0, 2.0])
    labels = np.array([1, 0, 1, 0, 0])
    strata = np.array([0, 0, 1, 1, 1])
    value, used, pairs = stratified_auc(scores, labels, strata)
    # stratum 0: 1 win of 1 pair. stratum 1: 0 wins of 2 pairs. pooled: 1/3.
    assert value == pytest.approx(1.0 / 3.0)
    assert used == 2
    assert pairs == 3


def test_stratified_auc_counts_ties_as_a_half():
    scores = np.array([1.0, 1.0])
    labels = np.array([1, 0])
    value, _, pairs = stratified_auc(scores, labels, np.array([0, 0]))
    assert value == pytest.approx(0.5)
    assert pairs == 1


@given(
    n=st.integers(min_value=6, max_value=60),
    seed=st.integers(min_value=0, max_value=1000),
)
@SLOW
def test_a_single_stratum_reproduces_the_pooled_auc(n, seed):
    from reward_lens.stats.roc import roc_pr

    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=int)
    labels[: max(1, n // 3)] = 1
    rng.shuffle(labels)
    assume(0 < labels.sum() < n)
    scores = rng.normal(size=n)
    value, _, _ = stratified_auc(scores, labels, np.zeros(n, dtype=int))
    assert value == pytest.approx(float(roc_pr(scores, labels).auc), abs=1e-12)


def _series(n_items: int = 60, *, seed: int = 0, position_only: bool = False):
    """A ragged localisation series with a planted error step per item."""
    rng = np.random.default_rng(seed)
    values, offsets, labels = [], [0], []
    for _ in range(n_items):
        k = int(rng.integers(5, 9))
        step = rng.normal(0.0, 0.3, k)
        err = int(rng.integers(0, k))
        if not position_only:
            step[err] -= 3.0
        values.extend(step.tolist())
        offsets.append(len(values))
        labels.append(err)
    return LocalisationSeries(
        values=np.asarray(values),
        offsets=np.asarray(offsets),
        labels=np.asarray(labels),
        higher_is_positive=False,
    )


def test_exact_position_strata_neutralise_position_by_construction():
    prior = rescore_against_position(_series(), n_boot=50)
    assert prior.exact_position_check == pytest.approx(0.5, abs=1e-12)


def test_a_real_localiser_survives_conditioning():
    prior = rescore_against_position(_series(n_items=200), n_boot=100)
    assert prior.uniform_auc > 0.8
    assert prior.stratified_auc > 0.8
    assert prior.verdict == "localises"


def test_inversion_is_exact_arithmetic():
    prior = rescore_against_position(_series(), n_boot=50)
    assert prior.inverted_stratified_auc == pytest.approx(1.0 - prior.stratified_auc)


def test_position_instrument_refuses_without_a_series():
    reading = PositionStratifiedNull().estimate(Context())
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "load_step_scores" in reading.remedy


def test_position_instrument_refuses_when_no_item_carries_a_label():
    series = LocalisationSeries(
        values=np.array([1.0, 2.0, 3.0]),
        offsets=np.array([0, 3]),
        labels=np.array([-1]),
        higher_is_positive=True,
    )
    reading = PositionStratifiedNull(series).estimate(Context())
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.LABEL_QUALITY_UNKNOWN
    assert "negative number for items that carry none" in reading.remedy


def test_ragged_layout_is_checked():
    with pytest.raises(ValueError, match="one more boundary"):
        LocalisationSeries(
            values=np.array([1.0, 2.0]),
            offsets=np.array([0, 2]),
            labels=np.array([0, 0]),
            higher_is_positive=True,
        )


# ---------------------------------------------------------------------------
# The generated invariance test, one per instrument
# ---------------------------------------------------------------------------


def test_every_instrument_declares_a_registered_quantity_and_passes_lint():
    for cls in INSTRUMENTS:
        inst = cls()
        assert inst.quantity in QUANTITIES, inst.name
        assert lint_instrument(inst) == [], inst.name


@pytest.mark.parametrize("cls", INSTRUMENTS, ids=lambda c: c.name)
def test_generated_invariance_check_for_each_instrument(cls):
    """The generated test. Four of the five declare `none`; L1 declares `units`."""
    inst = cls()
    group = "trivial" if inst.invariance == "none" else inst.invariance
    report = check_invariance(inst, group, InvariancePayload(), run=lambda i, p: 0.0)
    assert report.passed, report.render()
    assert report.skipped, "a group with no generators must say why it asserted nothing"


def test_the_units_group_assertion_holds_for_the_certificate_comparison():
    """L1's group is `units`, whose assertion is a refusal rather than a numeric relation."""
    doses = (0.6, 0.8, 1.0)
    a = certify(
        [Plant(d, d, seed=0) for d in doses],
        reference_id="a",
        assigned_dose=0.8,
        matrix=MatrixDescription(system="x"),
        dose_unit="dose",
    )
    b = certify(
        [Plant(d, d * 1.1, seed=0) for d in doses],
        reference_id="b",
        assigned_dose=0.8,
        matrix=MatrixDescription(system="x"),
        dose_unit="error_rate",
    )
    assert check_unit_refusal(compare_u_char, a, b)
    same = compare_u_char(a, a)
    assert isinstance(same, float) and same == pytest.approx(0.0)


def test_every_series_quantity_has_at_least_one_registered_estimator():
    for quantity in (
        "reference.u_characterisation",
        "reference.u_homogeneity",
        "reference.u_stability",
        "labels.error_rate",
        "labels.score_ceiling",
        "labels.false_success_rate",
        "labels.corrupt_success_rate",
        "labels.fs_signal_locality",
        "labels.position_prior",
    ):
        assert ladder(quantity), quantity


def test_registration_is_idempotent():
    register_all()
    register_all()
    assert len(ladder("labels.error_rate")) == 3

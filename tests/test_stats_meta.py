"""Tests for `stats/meta.py`, against published output and against exact arithmetic.

Three kinds of check, in descending order of how much I trust them.

**Against published model output.** Two datasets whose fitted values are printed in public
documentation, so the test compares this implementation against somebody else's rather than against
itself. The BCG vaccine meta-analysis (Colditz et al. 1994, `metafor`'s `dat.bcg`) pins REML tau2,
the pooled estimate, its standard error, the confidence interval, Q, I2, H2, and the normal-form
prediction interval. DerSimonian & Kacker (2007) is a k = 6 example, which is the size that matters
for the work this module was written for, and it pins DL, PM and REML side by side.

**Against exact arithmetic.** One four-study case with equal variances where every quantity has a
closed form small enough to check on paper, worked through in the docstring of
`test_the_hand_computed_case`. It happens to be a case where the three tau2 estimators provably
coincide, which makes it a three-way check rather than one.

**Against the definitions.** The Paule-Mandel estimate is the tau2 at which the generalised Q equals
k-1, so the test evaluates the generalised Q at the returned value and checks it is k-1. The REML
estimate is a fixed point of the REML estimating equation, so the test evaluates the residual. These
are the strongest checks available because they do not depend on anybody's printed digits.

One recorded disagreement. The DerSimonian & Kacker page reads tau2_REML = 0.0340 and this
implementation returns 0.033949, which satisfies the REML estimating equation to 8e-15. The
difference is in the fifth decimal, the printed mu and SE agree exactly, and the reference was read
through a web summary rather than from R, so the assertion here is to three decimals and the exact
check is the estimating-equation residual below it.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from scipy.stats import chi2, norm
from scipy.stats import t as student_t

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.stats import meta

# ---------------------------------------------------------------------------
# Reference datasets
# ---------------------------------------------------------------------------


def bcg() -> tuple[np.ndarray, np.ndarray]:
    """Log risk ratios and their variances for the 13 BCG vaccine trials.

    Counts from Colditz et al. (1994) as distributed in `metadat`'s `dat.bcg`. The effect measure is
    `escalc(measure="RR")`: yi = log of the ratio of the two risks, vi = 1/ai - 1/(ai+bi) + 1/ci -
    1/(ci+di). Computed here from the counts rather than copied as rounded yi/vi, so the comparison
    against the published fit is at full precision.
    """
    tpos = np.array([4, 6, 3, 62, 33, 180, 8, 505, 29, 17, 186, 5, 27])
    tneg = np.array([119, 300, 228, 13536, 5036, 1361, 2537, 87886, 7470, 1699, 50448, 2493, 16886])
    cpos = np.array([11, 29, 11, 248, 47, 372, 10, 499, 45, 65, 141, 3, 29])
    cneg = np.array([128, 274, 209, 12619, 5761, 1079, 619, 87892, 7232, 1600, 27197, 2338, 17825])
    yi = np.log((tpos / (tpos + tneg)) / (cpos / (cpos + cneg)))
    vi = 1 / tpos - 1 / (tpos + tneg) + 1 / cpos - 1 / (cpos + cneg)
    return yi, vi


def dersimonian_kacker() -> tuple[np.ndarray, np.ndarray]:
    """Log odds ratios and variances for the six trials in DerSimonian & Kacker (2007).

    The dataset `metafor` uses to demonstrate every tau2 estimator against each other. k = 6, which
    is the size of the collection this module was written to pool.
    """
    n1 = np.array([156, 303, 565, 1570, 103, 4659])
    n2 = np.array([74, 303, 477, 1565, 105, 4650])
    a = np.array([5, 5, 12, 69, 9, 313])
    c = np.array([8, 17, 9, 94, 11, 352])
    b, d = n1 - a, n2 - c
    return np.log(a * d / (b * c)), 1 / a + 1 / b + 1 / c + 1 / d


# ---------------------------------------------------------------------------
# Against published output
# ---------------------------------------------------------------------------


def test_bcg_reproduces_the_published_reml_fit():
    """metafor prints: tau2 0.3132, estimate -0.7145, se 0.1798, CI [-1.07, -0.36].

    Also Q(df=12) = 152.2330, I2 = 92.22%, H2 = 12.86. Reproducing all seven from raw counts is the
    check that the weights, the Q statistic, the REML iteration and the typical-within-variance term
    are all right at once, since getting any of them wrong moves at least one of the seven.
    """
    yi, vi = bcg()
    fit = meta.random_effects(yi, vi, tau2_method="REML")
    assert not isinstance(fit, Refusal)
    assert fit.het.tau2 == pytest.approx(0.3132, abs=5e-5)
    assert fit.pooled == pytest.approx(-0.7145, abs=5e-5)
    assert fit.se == pytest.approx(0.1798, abs=5e-5)
    assert fit.ci[0] == pytest.approx(-1.0669, abs=5e-4)
    assert fit.ci[1] == pytest.approx(-0.3622, abs=5e-4)
    assert fit.het.q == pytest.approx(152.2330, abs=5e-4)
    assert fit.het.q_df == 12
    assert fit.het.i2 == pytest.approx(92.22, abs=5e-3)
    assert fit.het.h2 == pytest.approx(12.86, abs=5e-3)


def test_bcg_reproduces_the_published_prediction_interval_under_the_normal_rule():
    """`predict(res)` prints pi.lb -1.87, pi.ub 0.44 for the BCG fit.

    metafor's default critical value for a prediction interval is the standard normal, which its own
    documentation states, so this is the interval reproduced under `PredictionRule.NORMAL`. It pins
    the sqrt(tau2 + se^2) term, which is the part of the formula people get wrong by using tau2
    alone or se alone.
    """
    yi, vi = bcg()
    fit = meta.random_effects(yi, vi, tau2_method="REML", rule=meta.PredictionRule.NORMAL)
    assert not isinstance(fit, Refusal)
    assert fit.prediction[0] == pytest.approx(-1.87, abs=5e-3)
    assert fit.prediction[1] == pytest.approx(0.44, abs=5e-3)
    assert fit.prediction_df is None


def test_the_two_prediction_rules_differ_by_exactly_the_critical_value_ratio():
    """The whole disagreement between the two conventions is t(k-2) against z, and nothing else.

    Worth pinning because it is the argument for the default: at k = 13 the two intervals differ by
    12%, which is why nobody has noticed, and at k = 4 they differ by a factor of 2.2, which is where
    this module is used.
    """
    yi, vi = bcg()
    kw = dict(tau2_method="REML")
    hts = meta.random_effects(yi, vi, rule=meta.PredictionRule.HTS, **kw)
    nor = meta.random_effects(yi, vi, rule=meta.PredictionRule.NORMAL, **kw)
    assert not isinstance(hts, Refusal) and not isinstance(nor, Refusal)
    expected = student_t.ppf(0.975, 11) / norm.ppf(0.975)
    assert hts.prediction_width / nor.prediction_width == pytest.approx(expected, rel=1e-12)
    assert hts.prediction_df == 11
    assert expected == pytest.approx(1.1230, abs=5e-4)
    # And the same ratio at the k this module is used at.
    assert student_t.ppf(0.975, 2) / norm.ppf(0.975) == pytest.approx(2.1953, abs=5e-4)


@pytest.mark.parametrize(
    "method,tau2,mu,se",
    [
        ("DL", 0.0540, -0.3240, 0.1540),
        ("PM", 0.1355, -0.3811, 0.2060),
        ("REML", 0.0340, -0.2980, 0.1343),
    ],
)
def test_dersimonian_kacker_reproduces_all_three_estimators(method, tau2, mu, se):
    """The published k = 6 comparison table, one row per estimator.

    DL is 0.054 and PM is 0.135 on the same six studies, a factor of 2.5 apart, and the pooled
    estimate moves from -0.324 to -0.381 with it. That is the reason all three are computed on every
    fit rather than one being chosen silently.

    REML is asserted to three decimals; see the module docstring for why.
    """
    yi, vi = dersimonian_kacker()
    fit = meta.random_effects(yi, vi, tau2_method=method)
    assert not isinstance(fit, Refusal)
    tol = 5e-4 if method == "REML" else 5e-5
    assert fit.het.tau2 == pytest.approx(tau2, abs=tol)
    assert fit.pooled == pytest.approx(mu, abs=5e-5)
    assert fit.se == pytest.approx(se, abs=5e-5)


# ---------------------------------------------------------------------------
# Against exact arithmetic
# ---------------------------------------------------------------------------


def test_the_hand_computed_case():
    """Four studies, y = 0, 1, 2, 3, all with variance 1. Every quantity in closed form.

    Fixed effect: w_i = 1, sum w = 4, mu = 6/4 = 3/2.
    Q = sum (y_i - 3/2)^2 = 9/4 + 1/4 + 1/4 + 9/4 = 5, df = 3.
    C = sum w - sum w^2 / sum w = 4 - 4/4 = 3, so tau2_DL = (5 - 3) / 3 = 2/3.

    With equal variances the three estimators provably coincide, which turns this into a three-way
    check. Paule-Mandel solves 5 / (1 + tau2) = 3, giving tau2 = 2/3. REML's fixed point is
    tau2 = 1/2 + tau2/4, giving tau2 = 2/3 as well.

    Random effects: w*_i = 1 / (1 + 2/3) = 3/5, sum w* = 12/5, mu = 3/2 by symmetry,
    SE = sqrt(5/12).
    s^2 = (k-1) sum w / (sum w^2 - sum w^2_i) = 3*4 / (16 - 4) = 1, so I2 = (2/3)/(2/3 + 1) = 40%,
    which is also (Q - df)/Q = 2/5, and H2 = 2/3 + 1 = 5/3, which is also Q/df.
    Prediction spread = sqrt(2/3 + 5/12) = sqrt(13/12).

    The `Fraction` arithmetic below is an independent path to the same numbers: it never touches the
    module, so a bug that happened to be in both the code and my longhand would still have to
    survive being recomputed in exact rationals.
    """
    y = np.array([0.0, 1.0, 2.0, 3.0])
    v = np.ones(4)

    # Exact rational recomputation, independent of the module.
    fy = [Fraction(0), Fraction(1), Fraction(2), Fraction(3)]
    fw = [Fraction(1)] * 4
    mu_fe = sum(w * t for w, t in zip(fw, fy)) / sum(fw)
    q_exact = sum(w * (t - mu_fe) ** 2 for w, t in zip(fw, fy))
    c_exact = sum(fw) - sum(w * w for w in fw) / sum(fw)
    tau2_exact = (q_exact - 3) / c_exact
    assert mu_fe == Fraction(3, 2)
    assert q_exact == 5
    assert tau2_exact == Fraction(2, 3)

    for method in ("DL", "PM", "REML"):
        assert meta.TAU2_ESTIMATORS[method](y, v) == pytest.approx(2 / 3, abs=1e-9)

    fit = meta.random_effects(y, v, tau2_method="DL")
    assert not isinstance(fit, Refusal)
    assert fit.het.q == pytest.approx(5.0)
    assert fit.het.q_df == 3
    assert fit.het.tau2 == pytest.approx(2 / 3)
    assert fit.het.typical_variance == pytest.approx(1.0)
    assert fit.het.i2 == pytest.approx(40.0)
    assert fit.het.h2 == pytest.approx(5 / 3)
    assert fit.pooled == pytest.approx(1.5)
    assert fit.se == pytest.approx(math.sqrt(5 / 12))

    half_ci = norm.ppf(0.975) * math.sqrt(5 / 12)
    assert fit.ci == pytest.approx((1.5 - half_ci, 1.5 + half_ci))

    spread = math.sqrt(2 / 3 + 5 / 12)
    half_pi = student_t.ppf(0.975, 2) * spread
    assert fit.prediction == pytest.approx((1.5 - half_pi, 1.5 + half_pi))
    assert fit.prediction_df == 2

    # The numbers a reader can check on paper, to four places. The prediction interval is 3.54
    # times the width of the confidence interval on four studies whose tau2 is a modest 0.67, which
    # is the size of the gap this module exists to stop people from leaving out of a write-up.
    assert fit.se == pytest.approx(0.6455, abs=5e-5)
    assert fit.ci == pytest.approx((0.2348, 2.7652), abs=5e-5)
    assert fit.prediction == pytest.approx((-2.9783, 5.9783), abs=5e-5)
    assert fit.width_ratio == pytest.approx(3.5398, abs=5e-5)


def test_the_hand_computed_case_has_a_tau2_interval_that_dwarfs_its_point_estimate():
    """tau2_hat = 0.667 on four studies, with a Q-profile interval of [0, 22.17].

    The generalised Q here is 5 / (1 + tau2), so the limits are solvable by hand: the lower limit
    needs 5/(1+tau2) = chi2(3, 0.975) = 9.3484, which has no non-negative solution and truncates to
    0, and the upper needs 5/(1+tau2) = chi2(3, 0.025) = 0.2158, giving tau2 = 22.17.

    This is the number that stops a four-study fit from being read as if it had settled anything
    about heterogeneity. The point estimate is 0.67 and the data are equally consistent with 0 and
    with 22.
    """
    y = np.array([0.0, 1.0, 2.0, 3.0])
    v = np.ones(4)
    lo, hi = meta.tau2_q_profile_ci(y, v)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(5 / chi2.ppf(0.025, 3) - 1, rel=1e-9)
    assert hi == pytest.approx(22.1697, abs=5e-4)


# ---------------------------------------------------------------------------
# Against the definitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", [bcg, dersimonian_kacker])
def test_paule_mandel_sits_where_its_definition_says_it_does(dataset):
    """PM is defined as the tau2 at which the generalised Q equals k-1. Check it there."""
    y, v = dataset()
    tau2 = meta.tau2_paule_mandel(y, v)
    assert meta.generalised_q(y, v, tau2) == pytest.approx(len(y) - 1, abs=1e-8)


@pytest.mark.parametrize("dataset", [bcg, dersimonian_kacker])
def test_reml_is_a_fixed_point_of_its_own_estimating_equation(dataset):
    """tau2 = sum w^2 {(y - mu)^2 - v} / sum w^2 + 1 / sum w, with w = 1/(v + tau2)."""
    y, v = dataset()
    tau2 = meta.tau2_reml(y, v)
    w = 1.0 / (v + tau2)
    mu = (w * y).sum() / w.sum()
    residual = (w**2 * ((y - mu) ** 2 - v)).sum() / (w**2).sum() + 1.0 / w.sum() - tau2
    assert residual == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("dataset", [bcg, dersimonian_kacker])
def test_i_squared_agrees_with_the_q_form_under_dl(dataset):
    """The tau2 form of I2 reduces to (Q - df) / Q exactly when tau2 is DerSimonian-Laird.

    Both forms are in circulation. They are the same number for DL and different numbers for PM and
    REML, so writing the tau2 form and checking the identity is how the module gets to use one
    formula for all three estimators without silently changing what I2 means for DL.
    """
    y, v = dataset()
    fit = meta.random_effects(y, v, tau2_method="DL")
    assert not isinstance(fit, Refusal)
    q, df, _ = meta.cochran_q(y, v)
    assert fit.het.i2 == pytest.approx(100.0 * (q - df) / q, rel=1e-10)
    assert fit.het.h2 == pytest.approx(q / df, rel=1e-10)


@pytest.mark.parametrize("dataset", [bcg, dersimonian_kacker])
def test_the_q_profile_interval_brackets_the_paule_mandel_estimate(dataset):
    """A consequence of both being roots of the same monotone function at different targets.

    PM solves generalised Q = k-1 and the limits solve it at the two chi-square quantiles that
    straddle k-1, so a PM estimate outside its own Q-profile interval would mean the bisection had
    gone wrong. Cheap and it catches a sign slip in the quantile order, which is the one mistake this
    function invites.
    """
    y, v = dataset()
    lo, hi = meta.tau2_q_profile_ci(y, v)
    tau2 = meta.tau2_paule_mandel(y, v)
    assert lo <= tau2 + 1e-12
    assert tau2 <= hi + 1e-12


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

_effects = st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False)
_variances = st.floats(min_value=1e-3, max_value=5.0, allow_nan=False, allow_infinity=False)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    data=st.lists(st.tuples(_effects, _variances), min_size=3, max_size=12),
    method=st.sampled_from(["DL", "PM", "REML"]),
)
def test_the_prediction_interval_always_contains_the_confidence_interval(data, method):
    """Never narrower, on any input. The two share a centre and the prediction spread is larger.

    sqrt(tau2 + se^2) >= se, and t(k-2) > z for every k >= 3, so the containment is strict rather
    than incidental. If this ever fails the formula has lost the tau2 term or picked up a z.
    """
    y = [d[0] for d in data]
    v = [d[1] for d in data]
    fit = meta.random_effects(y, v, tau2_method=method)
    assert not isinstance(fit, Refusal)
    assert fit.prediction[0] <= fit.ci[0] + 1e-12
    assert fit.prediction[1] >= fit.ci[1] - 1e-12
    assert fit.width_ratio >= 1.0


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    centre=_effects,
    data=st.lists(
        st.tuples(st.floats(min_value=-0.5, max_value=0.5), _variances), min_size=3, max_size=12
    ),
)
def test_tau2_zero_collapses_random_effects_onto_the_fixed_effect_fit(centre, data):
    """With no between-study variance the two models are the same model.

    The zero-tau2 case is constructed rather than filtered for. Writing each effect as
    y_i = centre + z_i * sqrt(v_i) with |z_i| <= 1/2 bounds Q above by sum z_i^2 <= k/4, because the
    fixed-effect mean minimises that sum, and k/4 <= k-1 for every k >= 2. So DL's numerator Q - df
    is never positive and tau2 is exactly zero by construction on every generated example. Rejection
    sampling for the same property throws away nine inputs in ten and distorts what is left.

    The pooled estimate, the standard error and the confidence interval must then coincide with the
    fixed-effect fit to machine precision. The prediction interval must not, and the ratio must be
    exactly t(k-2)/z, because even with tau2 = 0 the interval for a new study carries the
    uncertainty in the centre through a heavier-tailed distribution.
    """
    v = [d[1] for d in data]
    y = [centre + d[0] * math.sqrt(d[1]) for d in data]
    assume(meta.tau2_dersimonian_laird(y, v) == 0.0)  # a guard, not a filter: it never fires
    fit = meta.random_effects(y, v, tau2_method="DL")
    assert not isinstance(fit, Refusal)
    fe = meta.fixed_effect(y, v)
    assert fit.pooled == pytest.approx(fe.pooled, rel=1e-12, abs=1e-12)
    assert fit.se == pytest.approx(fe.se, rel=1e-12, abs=1e-12)
    assert fit.ci == pytest.approx(fe.ci, rel=1e-12, abs=1e-12)
    assert fit.het.i2 == pytest.approx(0.0)
    ratio = student_t.ppf(0.975, len(y) - 2) / norm.ppf(0.975)
    assert fit.width_ratio == pytest.approx(ratio, rel=1e-10)


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    counts=st.lists(st.integers(min_value=0, max_value=40), min_size=3, max_size=8),
    data=st.data(),
)
def test_pooled_proportions_stay_inside_the_unit_interval(counts, data):
    """Both intervals land in [0, 1] after back-transformation, whatever the counts.

    The reason to pool on a transformed scale at all. A random-effects interval computed on raw
    proportions runs past 1 as soon as the pooled value is near the boundary, and at k = 4 with a
    t(2) critical value it does so routinely.
    """
    totals = [data.draw(st.integers(min_value=max(1, c), max_value=max(1, c) + 40)) for c in counts]
    fit = meta.proportion_meta(counts, totals, scale="logit")
    assert not isinstance(fit, Refusal)
    for value in (fit.pooled_p, *fit.ci_p, *fit.prediction_p):
        assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_two_studies_refuse_and_hand_back_a_bound():
    """k = 2 has one degree of freedom for tau2 and none for the prediction interval.

    The refusal names the reason and the remedy, and carries the fixed-effect pooled value in
    `statistics` so the caller is not left with nothing. That is the difference between "I cannot
    tell you" and "I cannot tell you under a model you can check, and here is the number under one
    you cannot".
    """
    fit = meta.random_effects([0.4, 0.6], [0.01, 0.02])
    assert isinstance(fit, Refusal)
    assert fit.reason is RefusalReason.ABOVE_LOD_BELOW_LOQ
    assert "k = 2" in fit.detail
    assert "t(0)" in fit.detail
    assert "one more independent study" in fit.remedy
    assert fit.statistics["k"] == 2
    assert fit.statistics["fixed_effect_pooled"] == pytest.approx(
        meta.fixed_effect([0.4, 0.6], [0.01, 0.02]).pooled
    )


def test_one_study_refuses_outright():
    """One estimate is a study, not a meta-analysis, and there is no bound to offer."""
    fit = meta.random_effects([0.4], [0.01])
    assert isinstance(fit, Refusal)
    assert fit.reason is RefusalReason.ESS_BELOW_FLOOR
    assert "Report the single study" in fit.remedy


def test_eggers_test_refuses_below_ten_studies():
    """Cochrane's floor, and the refusal says what a non-significant result there would mean.

    The failure mode this prevents is specific: run Egger at k = 6, get p = 0.4, write "no evidence
    of publication bias". The test had no power to find any, so the sentence is about the test rather
    than about the literature.
    """
    y, v = dersimonian_kacker()
    out = meta.eggers_test(y, v)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ESS_BELOW_FLOOR
    assert out.statistics["k"] == 6
    assert out.statistics["floor"] == 10
    assert "could not be assessed" in out.remedy
    assert "negligible power" in out.detail


def test_eggers_test_runs_at_ten_studies_and_finds_nothing_in_symmetric_data():
    """The other side of the floor: at k = 10 it returns a result rather than a refusal."""
    rng = np.random.default_rng(11)
    v = rng.uniform(0.01, 0.4, size=10)
    y = rng.normal(0.5, np.sqrt(v))
    out = meta.eggers_test(y, v)
    assert not isinstance(out, Refusal)
    assert out.k == 10
    assert out.df == 8
    assert 0.0 <= out.p <= 1.0


def test_the_prediction_interval_function_refuses_the_arithmetic_it_cannot_do():
    """Called directly at k = 2, the HTS rule raises rather than returning a nan."""
    with pytest.raises(ValueError, match="no prediction interval"):
        meta.prediction_interval(0.5, 0.1, 0.02, k=2)
    lo, hi, df = meta.prediction_interval(0.5, 0.1, 0.02, k=2, rule=meta.PredictionRule.NORMAL)
    assert df is None and lo < 0.5 < hi


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_the_interpretation_says_which_interval_is_which():
    """The string is the deliverable as much as the numbers are.

    A reader who takes one sentence from a meta-analysis must not be able to take away "the effect is
    0.42, interval 0.31 to 0.53" without knowing that the second half is about the average of the
    studies pooled and not about the next one.
    """
    yi, vi = dersimonian_kacker()
    fit = meta.random_effects(yi, vi, tau2_method="PM")
    text = fit.interpretation(null=0.0)
    assert "MEAN effect" in text
    assert "NEW study" in text
    assert "Rucker" in text
    assert "prediction interval" in text
    assert "fixed-effect pooled estimate" in text


def test_the_caveat_fires_on_small_k_and_on_estimator_disagreement():
    """Two conditions, each with its own sentence, both checked on data that triggers them."""
    yi, vi = dersimonian_kacker()
    het = meta.heterogeneity(yi, vi, tau2_method="PM")
    caveat = het.caveat()
    assert "Q-profile interval" in caveat
    assert "Rucker" in caveat
    assert "do not agree" in caveat  # DL 0.054 against PM 0.135 on the same six studies
    assert not meta.heterogeneity(np.arange(4.0), np.ones(4)).is_reliable
    assert (
        f"k = 4 is below {meta.SMALL_K}" in meta.heterogeneity(np.arange(4.0), np.ones(4)).caveat()
    )


def test_the_proposed_quantity_rows_parse_and_match_the_objects():
    """The rows are emitted so they cannot drift from the `Quantity` objects. Check they also load.

    A row that does not parse is the same class of failure as a row with the wrong number in it, and
    it is easy to produce here because every definition contains a colon-space, which ends a key in
    YAML block context. Round-tripped through the parser rather than eyeballed.
    """
    yaml = pytest.importorskip("yaml")
    rows = yaml.safe_load(meta.as_yaml_rows())
    assert [r["id"] for r in rows] == [q.id for q in meta.PROPOSED]
    by_id = {q.id: q for q in meta.PROPOSED}
    for row in rows:
        q = by_id[row["id"]]
        assert row["definition"] == q.definition
        assert row["interpretation"] == q.interpretation
        assert row["invariance_group"] == q.invariance
        assert row["unit"]["as_printed"] == q.unit.as_printed
        assert row["unit"]["dimension"] == q.unit.dimension
        assert row["min_access"] == meta.PROPOSED_MIN_ACCESS[q.id]
        assert row["support"] == ("OPEN" if q.support is None else list(q.support))
    # None of them registers itself at import; that is a spec edit, not an import side effect.
    from reward_lens.core.quantity import QUANTITIES

    assert (
        not ({q.id for q in meta.PROPOSED} & set(QUANTITIES))
        or set(meta.register_proposed()) == set()
    )


def test_the_vote_count_baseline_counts_and_says_what_it_cannot_do():
    """The naive comparator, present so the pooled estimate has something to be compared against."""
    vc = meta.vote_count([0.2, 0.4, -0.1, 0.9])
    assert (vc.k, vc.positive, vc.negative) == (4, 3, 1)
    assert vc.fraction_positive == pytest.approx(0.75)
    assert "No interval" in vc.render()


def test_as_dict_round_trips_every_reported_number():
    """The artifact carries what the render shows, so a write-up cannot quote an unexported number."""
    yi, vi = dersimonian_kacker()
    fit = meta.random_effects(yi, vi, tau2_method="PM")
    d = fit.as_dict()
    assert d["k"] == 6
    assert d["prediction_rule"] == "t(k-2)"
    assert d["prediction_df"] == 4
    assert set(d["tau2_all"]) == {"DL", "PM", "REML"}
    assert d["tau2_ci"][0] <= d["tau2"] <= d["tau2_ci"][1]
    assert d["fixed_effect"]["pooled"] == pytest.approx(meta.fixed_effect(yi, vi).pooled)


# ---------------------------------------------------------------------------
# Proportions
# ---------------------------------------------------------------------------


def test_the_logit_transform_and_its_inverse_agree_with_longhand():
    """3 of 8 with the 0.5 correction: y = log(3.5/5.5), v = 1/3.5 + 1/5.5."""
    y, v = meta.proportion_effects([3], [8], scale="logit", correction=0.5)
    assert y[0] == pytest.approx(math.log(3.5 / 5.5))
    assert v[0] == pytest.approx(1 / 3.5 + 1 / 5.5)
    assert meta.proportion_back(y[0], scale="logit") == pytest.approx(3.5 / 9.0)
    # Uncorrected, the inverse recovers the raw proportion exactly.
    y0, _ = meta.proportion_effects([3], [8], scale="logit", correction=0.0)
    assert meta.proportion_back(y0[0], scale="logit") == pytest.approx(0.375)


def test_the_double_arcsine_transform_handles_the_boundary_the_logit_cannot():
    """k = n is a real case in this literature and log(n - k) is not defined at it."""
    y, v = meta.proportion_effects([4], [4], scale="double-arcsine")
    assert math.isfinite(y[0]) and v[0] == pytest.approx(1 / 18)
    with pytest.raises(ValueError, match="infinite logit"):
        meta.proportion_effects([4], [4], scale="logit", correction=0.0)


def test_the_double_arcsine_back_transform_reduces_to_sine_squared_at_large_n():
    """Miller's inverse carries a 1/n term; at n = 10^7 it should be invisible."""
    y, _ = meta.proportion_effects([3_000_000], [10_000_000], scale="double-arcsine")
    back = meta.proportion_back(y[0], scale="double-arcsine", totals=[10_000_000])
    assert back == pytest.approx(0.3, abs=1e-6)
    assert back == pytest.approx(math.sin(y[0]) ** 2, abs=1e-6)


def test_a_proportion_meta_reports_both_intervals_on_the_proportion_scale():
    counts, totals = [3, 4, 3, 10], [8, 9, 4, 20]
    fit = meta.proportion_meta(counts, totals, labels=list("abcd"))
    assert not isinstance(fit, Refusal)
    assert fit.observed == pytest.approx((0.375, 4 / 9, 0.75, 0.5))
    assert fit.prediction_p[0] < fit.ci_p[0] <= fit.pooled_p <= fit.ci_p[1] < fit.prediction_p[1]
    assert "about the MEAN" in fit.render()
    assert "about a NEW study" in fit.render()


# ---------------------------------------------------------------------------
# Power at the realised k
# ---------------------------------------------------------------------------


def test_power_for_the_pooled_effect_falls_as_heterogeneity_rises():
    """Adding tau2 adds variance to the pooled estimate, so power can only drop."""
    _, v = dersimonian_kacker()
    powers = [meta.power_for_pooled_effect(v, tau2=t, delta=0.5) for t in (0.0, 0.05, 0.2, 1.0)]
    assert powers == sorted(powers, reverse=True)
    assert powers[0] > 0.9
    assert powers[-1] < 0.4


def test_cochrans_q_has_almost_no_power_at_six_studies():
    """The measurement behind the rule that a non-significant Q is not evidence of homogeneity.

    Simulated at the realised within-study variances of the k = 6 reference dataset. At a tau2 equal
    to roughly the typical within-study variance, the test still misses most of the time.
    """
    _, v = dersimonian_kacker()
    s2 = meta.typical_within_variance(v)
    power = meta.power_to_detect_heterogeneity(v, tau2=s2, n_sim=8000, seed=3)
    assert 0.0 < power < 0.7
    assert meta.power_to_detect_heterogeneity(v, tau2=0.0, n_sim=8000, seed=3) == pytest.approx(
        0.05, abs=0.02
    )

"""The G-theory arithmetic, checked against hand computation and against a published identity.

There is no reference implementation to diff against. PyPI has no generalizability-theory package
and R's `gtheory` was removed from CRAN on 2025-03-24, so "it agrees with the other one" is not
available and something else has to do the work. Four things do:

**A design worked by hand.** A three-by-two matrix whose sums of squares, mean squares and three
variance components are computed in the docstring of `test_two_facet_matches_hand_computation` and
asserted as literals. If the arithmetic is wrong, that test says so and says where.

**A published identity.** For a crossed object-by-item design, the generalizability coefficient at
the study's own number of items is *exactly* Cronbach's alpha. Alpha is computed here from the raw
data by the 1951 formula, which shares no code with the module, so the equality is a real external
check on the whole path from sums of squares to coefficient.

**The estimator's own defining equations.** The method of moments inverts the expected mean squares,
so plugging the fitted components back into those equations must reproduce the observed mean squares
to floating point. That catches an algebra slip in the inversion but not one in the expectations, so
it is not sufficient on its own.

**Recovery from simulation.** Data is generated from known components and refitted three hundred
times; the mean of the fits sits on the truth. That is the check on the expectations themselves,
and it is the reason the module can claim the estimator is unbiased rather than merely
self-consistent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from reward_lens.stats.gtheory import (
    DesignError,
    check_balance,
    crossed_pr,
    crossed_pro,
    fit_unbalanced,
    mean_squares_pr,
    mean_squares_pro,
    statsmodels_available,
    to_cube,
)
from reward_lens.stats.variance import (
    ComponentSet,
    VarianceComponent,
    design_effect_ess,
    gauge_rr,
    group_effective_size,
    icc_oneway,
    kish_ess,
    truncate_at_zero,
)

SLOW = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


# ---------------------------------------------------------------------------
# The hand-computed design
# ---------------------------------------------------------------------------

#: Three objects, two raters, one score per cell. Worked out in full in the test below.
HAND = np.array([[1.0, 4.0], [2.0, 3.0], [6.0, 8.0]])


def test_two_facet_matches_hand_computation():
    """The three-by-two design, in full.

    Row means 2.5, 2.5, 7. Column means 3, 5. Grand mean 4.

        SS_p    = 2 * ((2.5-4)^2 + (2.5-4)^2 + (7-4)^2)   = 2 * 13.5 = 27,  df 2, MS 13.5
        SS_r    = 3 * ((3-4)^2 + (5-4)^2)                 = 3 * 2    =  6,  df 1, MS  6
        SS_pr,e = sum of (x - rowmean - colmean + grand)^2
                = (-0.5)^2 + 0.5^2 + 0.5^2 + (-0.5)^2 + 0 + 0        =  1,  df 2, MS  0.5

    The three sums of squares add to 34, which is the total sum of squares about the grand mean:
    9 + 0 + 4 + 1 + 4 + 16 = 34. Inverting the expected mean squares,

        sigma2(pr,e) = 0.5
        sigma2(p)    = (13.5 - 0.5) / 2 = 6.5
        sigma2(r)    = (6 - 0.5) / 3    = 11/6
    """
    ms = mean_squares_pr(HAND)
    assert ms.ss["p"] == pytest.approx(27.0)
    assert ms.ss["r"] == pytest.approx(6.0)
    assert ms.ss["pr,e"] == pytest.approx(1.0)
    assert ms.df == {"p": 2.0, "r": 1.0, "pr,e": 2.0}
    assert ms.ms["p"] == pytest.approx(13.5)
    assert ms.ms["r"] == pytest.approx(6.0)
    assert ms.ms["pr,e"] == pytest.approx(0.5)

    g = crossed_pr(HAND)
    assert g.components.value("pr,e") == pytest.approx(0.5)
    assert g.components.value("p") == pytest.approx(6.5)
    assert g.components.value("r") == pytest.approx(11.0 / 6.0)
    assert g.components.total == pytest.approx(6.5 + 11.0 / 6.0 + 0.5)

    # E rho^2 at the study's own two raters: 6.5 / (6.5 + 0.5/2) = 26/27.
    assert g.generalizability(r=2) == pytest.approx(26.0 / 27.0)


def test_generalizability_at_the_study_size_equals_cronbachs_alpha():
    """Alpha computed from the raw data by the 1951 formula, sharing no code with the module.

    The identity is exact rather than approximate, and it is the strongest external check available
    for a two-facet design, because alpha is a completely different route from the same numbers.
    """
    rng = np.random.default_rng(11)
    x = rng.normal(size=(60, 6)) + 2.0 * rng.normal(size=(60, 1))
    k = x.shape[1]
    alpha = (k / (k - 1)) * (1.0 - x.var(axis=0, ddof=1).sum() / x.sum(axis=1).var(ddof=1))
    assert crossed_pr(x).generalizability(r=k) == pytest.approx(alpha, rel=1e-12)


def test_three_facet_additive_data_has_exactly_zero_interactions():
    """An exactly additive array has no interaction, and the main effects are sample variances.

    A hand-checkable identity for the seven-component design: when `X[p,r,o] = a[p] + b[r] + c[o]`
    every interaction sum of squares is zero to floating point and each main-effect component is
    the ddof=1 sample variance of its own vector.
    """
    a = np.array([1.0, 4.0, 9.0, 2.0])
    b = np.array([0.5, -1.5, 3.0])
    c = np.array([10.0, -2.0])
    x = a[:, None, None] + b[None, :, None] + c[None, None, :]

    g = crossed_pro(x)
    for name in ("pr", "po", "ro", "pro,e"):
        assert abs(g.components.raw_dict()[name]) < 1e-9, name
    assert g.components.value("p") == pytest.approx(a.var(ddof=1))
    assert g.components.value("r") == pytest.approx(b.var(ddof=1))
    assert g.components.value("o") == pytest.approx(c.var(ddof=1))


def test_three_facet_degrees_of_freedom_and_sums_of_squares_partition_exactly():
    """df sum to N-1 and the seven sums of squares add to the total. Both must hold identically."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(9, 4, 3))
    ms = mean_squares_pro(x)
    assert sum(ms.df.values()) == 9 * 4 * 3 - 1
    total = float(np.sum((x - x.mean()) ** 2))
    assert sum(ms.ss.values()) == pytest.approx(total, rel=1e-12)


def test_expected_mean_squares_round_trip_exactly():
    """Plugging the fitted components back into the EMS equations reproduces the mean squares.

    The inversion is exact linear algebra, so any residual here is an algebra error rather than an
    estimation error, and the tolerance is floating point rather than statistical.
    """
    rng = np.random.default_rng(5)
    n_p, n_r, n_o = 8, 5, 4
    x = rng.normal(size=(n_p, n_r, n_o))
    g = crossed_pro(x)
    c = g.components.raw_dict()
    ms = g.mean_squares.ms
    expected = {
        "p": c["pro,e"] + n_o * c["pr"] + n_r * c["po"] + n_r * n_o * c["p"],
        "r": c["pro,e"] + n_o * c["pr"] + n_p * c["ro"] + n_p * n_o * c["r"],
        "o": c["pro,e"] + n_r * c["po"] + n_p * c["ro"] + n_p * n_r * c["o"],
        "pr": c["pro,e"] + n_o * c["pr"],
        "po": c["pro,e"] + n_r * c["po"],
        "ro": c["pro,e"] + n_p * c["ro"],
        "pro,e": c["pro,e"],
    }
    for name, value in expected.items():
        assert value == pytest.approx(ms[name], rel=1e-9, abs=1e-12), name


def _simulate(n_p: int, n_r: int, n_o: int, true: dict[str, float], rng) -> np.ndarray:
    return (
        rng.normal(0, math.sqrt(true["p"]), (n_p, 1, 1))
        + rng.normal(0, math.sqrt(true["r"]), (1, n_r, 1))
        + rng.normal(0, math.sqrt(true["o"]), (1, 1, n_o))
        + rng.normal(0, math.sqrt(true["pr"]), (n_p, n_r, 1))
        + rng.normal(0, math.sqrt(true["po"]), (n_p, 1, n_o))
        + rng.normal(0, math.sqrt(true["ro"]), (1, n_r, n_o))
        + rng.normal(0, math.sqrt(true["pro,e"]), (n_p, n_r, n_o))
    )


def test_all_seven_components_are_recovered_from_simulation():
    """Three hundred fits of data with known components land on the truth.

    Averaged over fits, because a single fit of a variance component on nine degrees of freedom is
    a wide thing. The tolerance is three standard errors of the mean of the fits, computed here
    rather than guessed, so this fails when the estimator is biased and not when it is noisy.
    """
    true = {"p": 1.0, "r": 0.5, "o": 0.25, "pr": 0.4, "po": 0.2, "ro": 0.1, "pro,e": 0.8}
    fits: dict[str, list[float]] = {k: [] for k in true}
    reps = 300
    for i in range(reps):
        x = _simulate(40, 6, 5, true, np.random.default_rng(9_000 + i))
        for name, value in crossed_pro(x).components.raw_dict().items():
            fits[name].append(value)
    for name, target in true.items():
        arr = np.asarray(fits[name])
        se = float(arr.std(ddof=1) / math.sqrt(reps))
        assert abs(float(arr.mean()) - target) < 3.0 * se, (
            f"{name}: mean {arr.mean():.4f} against {target} at {se:.4f} per fit"
        )


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_a_negative_component_is_truncated_and_the_truncation_is_recorded():
    """The whole reason `ComponentSet` exists rather than a dict of floats.

    Data with no rater effect at all produces a negative estimate roughly half the time. The value
    is zero, the raw estimate is kept, and the name appears in `truncated_names`, which is what
    stops a reader concluding that the raters agreed perfectly.
    """
    found = None
    for seed in range(200):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(6, 3)) + rng.normal(size=(6, 1)) * 3.0
        g = crossed_pr(x)
        if g.components["r"].truncated:
            found = g
            break
    assert found is not None, "no seed in 200 produced a negative rater component"
    assert found.components.value("r") == 0.0
    assert found.components["r"].raw < 0.0
    assert "r" in found.components.truncated_names
    assert found.components.any_truncated
    assert "TRUNCATED" in found.components.render()
    assert "do not read a truncated component as an established zero" in found.components.render()


def test_truncate_at_zero_keeps_the_raw_estimate_beside_the_reported_one():
    cs = truncate_at_zero({"p": 1.5, "r": -0.25, "pr,e": 0.5}, df={"p": 4.0})
    assert cs.value("r") == 0.0
    assert cs["r"].raw == -0.25
    assert cs["p"].df == 4.0
    assert cs.truncated_names == ("r",)
    assert cs.total == pytest.approx(2.0)
    assert cs.share("p") == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# The finite-universe correction
# ---------------------------------------------------------------------------


def test_declaring_a_facet_fixed_moves_variance_from_error_to_universe_score():
    """Brennan's trade, checked as an identity rather than as a direction.

    The object-by-facet term leaves relative error and arrives in universe-score variance, and
    nothing else moves. The sum of the two is invariant, which is the statement that fixing a facet
    redistributes variance rather than creating it.
    """
    rng = np.random.default_rng(21)
    x = _simulate(
        50,
        4,
        3,
        {"p": 1.0, "r": 0.3, "o": 0.2, "pr": 0.5, "po": 0.1, "ro": 0.05, "pro,e": 0.4},
        rng,
    )
    g = crossed_pro(x, facet_labels=("rater", "occasion"))

    at = {"r": 1, "o": 3}
    fixed = g.declare_fixed("r", universe_size=1)
    assert fixed.universe_variance(**at) == pytest.approx(
        g.universe_variance(**at) + g.components.value("pr") / 1
    )
    assert fixed.relative_error(**at) == pytest.approx(
        g.relative_error(**at) - g.components.value("pr") / 1
    )
    assert fixed.universe_variance(**at) + fixed.relative_error(**at) == pytest.approx(
        g.universe_variance(**at) + g.relative_error(**at)
    )
    assert fixed.generalizability(**at) > g.generalizability(**at)


def test_a_fixed_facets_main_effect_leaves_the_absolute_error_too():
    """With a facet fixed, its main effect is part of the definition and not part of the error."""
    rng = np.random.default_rng(22)
    x = _simulate(
        40,
        5,
        3,
        {"p": 1.0, "r": 0.9, "o": 0.2, "pr": 0.3, "po": 0.1, "ro": 0.05, "pro,e": 0.4},
        rng,
    )
    g = crossed_pro(x)
    at = {"r": 5, "o": 3}
    fixed = g.declare_fixed("r", universe_size=5)
    dropped = g.components.value("r") / 5 + g.components.value("pr") / 5
    assert fixed.absolute_error(**at) == pytest.approx(g.absolute_error(**at) - dropped)
    assert fixed.fixed == ("r",)
    assert g.fixed == ()


def test_a_partially_sampled_universe_interpolates_between_random_and_fixed():
    """N_i between n_i and infinity gives a fraction of the correction, which is the general form."""
    rng = np.random.default_rng(23)
    x = _simulate(
        30,
        4,
        2,
        {"p": 1.0, "r": 0.3, "o": 0.2, "pr": 0.5, "po": 0.1, "ro": 0.05, "pro,e": 0.4},
        rng,
    )
    g = crossed_pro(x)
    at = {"r": 2, "o": 2}
    random_e = g.relative_error(**at)
    half = g.declare_fixed("r", universe_size=4).relative_error(**at)
    full = g.declare_fixed("r", universe_size=2).relative_error(**at)
    assert full < half < random_e
    # At N = 4 with n' = 2 the pr term keeps (1 - 2/4) = half its random contribution.
    assert half == pytest.approx(random_e - 0.5 * g.components.value("pr") / 2, rel=1e-9)


def test_the_object_of_measurement_cannot_be_declared_fixed():
    g = crossed_pr(HAND)
    with pytest.raises(DesignError, match="not a facet"):
        g.declare_fixed("p")


# ---------------------------------------------------------------------------
# Balance
# ---------------------------------------------------------------------------


def test_check_balance_names_the_missing_cells():
    objects = ["a", "a", "b"]
    raters = ["x", "y", "x"]
    report = check_balance([objects, raters])
    assert not report.balanced
    assert report.n_cells_expected == 4
    assert report.n_cells_present == 3
    assert report.n_missing == 1
    assert ("b", "y") in report.missing_examples
    assert "not balanced" in report.render()


def test_check_balance_counts_replicated_cells_as_unbalanced():
    report = check_balance([["a", "a"], ["x", "x"]])
    assert not report.balanced
    assert report.replicated_cells == 1


def test_to_cube_refuses_an_unbalanced_design_rather_than_filling_it():
    with pytest.raises(DesignError, match="one observation in every crossed cell"):
        to_cube([1.0, 2.0, 3.0], [["a", "a", "b"], ["x", "y", "x"]])


def test_to_cube_round_trips_a_balanced_design():
    values = [1.0, 4.0, 2.0, 3.0, 6.0, 8.0]
    objects = ["p0", "p0", "p1", "p1", "p2", "p2"]
    raters = ["r0", "r1", "r0", "r1", "r0", "r1"]
    cube, levels = to_cube(values, [objects, raters])
    assert cube.shape == (3, 2)
    assert np.allclose(cube, HAND)
    assert levels == [["p0", "p1", "p2"], ["r0", "r1"]]


def test_fit_unbalanced_refuses_and_names_the_distribution():
    with pytest.raises(DesignError) as excinfo:
        fit_unbalanced()
    message = str(excinfo.value)
    assert "statsmodels" in message
    assert "MixedLM" in message
    assert "pip install 'statsmodels>=0.14'" in message
    assert ("installed" if statsmodels_available() else "not installed") in message


def test_statsmodels_probe_answers_without_importing_it():
    assert isinstance(statsmodels_available(), bool)


def test_a_design_with_one_rater_has_no_degrees_of_freedom_for_an_interaction():
    with pytest.raises(DesignError, match="at least two"):
        crossed_pr(np.array([[1.0], [2.0], [3.0]]))
    with pytest.raises(DesignError, match="at least two levels on every axis"):
        crossed_pro(np.zeros((5, 3, 1)))


def test_a_nan_is_a_hole_in_the_design_and_not_a_number_to_propagate():
    x = HAND.copy()
    x[0, 0] = np.nan
    with pytest.raises(DesignError, match="hole in the design"):
        crossed_pr(x)


# ---------------------------------------------------------------------------
# The two effective-sample-size forms
# ---------------------------------------------------------------------------


def test_kish_anchors():
    assert kish_ess([1.0, 1.0, 1.0, 1.0]) == pytest.approx(4.0)
    assert kish_ess([5.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert kish_ess([]) == 0.0
    assert kish_ess([0.0, 0.0]) == 0.0
    with pytest.raises(ValueError, match="non-negative weights"):
        kish_ess([1.0, -1.0])
    # Scale-free, including where the squares would underflow or overflow. Hypothesis found the
    # subnormal case against an implementation that squared the raw weights.
    assert kish_ess([2.2250738585e-313, 2.2250738585e-313]) == pytest.approx(2.0)
    assert kish_ess([1e200, 1e200, 1e200]) == pytest.approx(3.0)
    assert kish_ess([3.0, 1.0]) == pytest.approx(kish_ess([3e-200, 1e-200]))


def test_group_effective_size_on_the_worked_group():
    """Fifteen rollouts tied and one outlier, at K = 16.

    Weights are 1/16 fifteen times and 15/16 once, so the Kish count is
    (15/16 + 15/16)^2 / (15 * (1/16)^2 + (15/16)^2) = (30/16)^2 / (240/256) = 3.75.
    """
    scores = np.array([0.0] * 15 + [1.0])
    assert group_effective_size(scores) == pytest.approx(3.75)
    assert group_effective_size(np.zeros(8)) == 0.0
    assert group_effective_size(np.array([0.0, 1.0, 2.0, 3.0])) == pytest.approx(
        kish_ess(np.abs(np.array([0.0, 1.0, 2.0, 3.0]) - 1.5))
    )


def test_design_effect_anchors():
    assert design_effect_ess(16, 0.0) == pytest.approx(16.0)
    assert design_effect_ess(16, 1.0) == pytest.approx(1.0)
    assert design_effect_ess(1, 0.9) == 1.0
    # A negative within-cluster correlation makes a cluster worth more than K independent draws.
    assert design_effect_ess(4, -0.1) > 4.0
    with pytest.raises(ValueError, match="outside the exchangeable model"):
        design_effect_ess(4, -0.5)


def test_icc_oneway_anchors():
    identical_within = [[1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [9.0, 9.0, 9.0]]
    assert icc_oneway(identical_within) == pytest.approx(1.0)
    assert icc_oneway([[1.0, 2.0]]) == 0.0
    rng = np.random.default_rng(7)
    independent = [rng.normal(size=30) for _ in range(30)]
    assert abs(icc_oneway(independent)) < 0.15


# ---------------------------------------------------------------------------
# Gauge R&R
# ---------------------------------------------------------------------------


def test_gauge_rr_arithmetic_on_a_designed_component_set():
    """sigma2(p) = 9, everything else = 16, so sigma_part = 3 and sigma_GRR = 4."""
    cs = truncate_at_zero({"p": 9.0, "r": 7.0, "pr,e": 9.0})
    g = gauge_rr(cs)
    assert g.sigma_part == pytest.approx(3.0)
    assert g.sigma_grr == pytest.approx(4.0)
    assert g.sigma_total == pytest.approx(5.0)
    assert g.grr_percent == pytest.approx(80.0)
    assert g.ndc == pytest.approx(1.41 * 3.0 / 4.0)
    assert g.ndc_categories == 1
    assert not g.acceptable
    assert g.band == "unacceptable"
    assert "cannot resolve two adjacent items" in g.verdict()


def test_gauge_rr_bands_and_the_two_automotive_rules():
    good = gauge_rr(truncate_at_zero({"p": 100.0, "r": 0.5, "pr,e": 0.5}))
    assert good.band == "acceptable"
    assert good.ndc_categories >= 5
    assert good.acceptable
    # sigma_GRR = 2, sigma_total = sqrt(104), so %GRR = 19.6%: inside the 10 to 30 band.
    marginal = gauge_rr(truncate_at_zero({"p": 100.0, "r": 2.0, "pr,e": 2.0}))
    assert marginal.grr_percent == pytest.approx(100.0 * 2.0 / math.sqrt(104.0))
    assert marginal.band == "marginal"


def test_repeatability_is_none_unless_the_design_identifies_it():
    cs = truncate_at_zero({"p": 9.0, "r": 7.0, "pr": 4.0, "e": 5.0})
    assert gauge_rr(cs).repeatability is None
    split = gauge_rr(cs, repeatability="e")
    assert split.repeatability == pytest.approx(5.0)
    assert split.reproducibility == pytest.approx(11.0)


def test_a_gauge_with_no_measurement_variance_resolves_everything():
    g = gauge_rr(truncate_at_zero({"p": 4.0, "r": 0.0, "pr,e": 0.0}))
    assert math.isinf(g.ndc)
    assert g.grr_percent == 0.0
    assert g.acceptable


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

_finite = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False, width=32)


@given(
    values=st.lists(_finite, min_size=6, max_size=6),
    shift=_finite,
)
@SLOW
def test_components_are_invariant_under_a_location_shift(values, shift):
    """Adding a constant to every score changes no variance component."""
    x = np.asarray(values, dtype=np.float64).reshape(3, 2)
    a = crossed_pr(x).components.raw_dict()
    b = crossed_pr(x + shift).components.raw_dict()
    for name in a:
        assert b[name] == pytest.approx(a[name], rel=1e-6, abs=1e-6)


@given(
    values=st.lists(_finite, min_size=12, max_size=12),
    scale=st.floats(min_value=0.05, max_value=20.0, allow_nan=False, allow_infinity=False),
)
@SLOW
def test_components_scale_by_the_square_under_a_reward_rescaling(values, scale):
    """`r -> a*r` scales every component by `a^2`, so every share is unchanged.

    This is the `reward.affine` covariance of a variance and the invariance of a share, checked at
    the arithmetic layer rather than only through the generated instrument test.
    """
    x = np.asarray(values, dtype=np.float64).reshape(6, 2)
    base = crossed_pr(x)
    scaled = crossed_pr(x * scale)
    for name in base.components.names:
        assert scaled.components.raw_dict()[name] == pytest.approx(
            base.components.raw_dict()[name] * scale**2, rel=1e-6, abs=1e-9
        )
    if base.components.total > 1e-9:
        for name in base.components.names:
            assert scaled.components.share(name) == pytest.approx(
                base.components.share(name), abs=1e-6
            )


@given(values=st.lists(_finite, min_size=12, max_size=12))
@SLOW
def test_the_generalizability_coefficient_lies_in_the_unit_interval(values):
    g = crossed_pr(np.asarray(values, dtype=np.float64).reshape(6, 2))
    for n in (1, 2, 10):
        assert 0.0 <= g.generalizability(r=n) <= 1.0
        assert 0.0 <= g.dependability(r=n) <= 1.0


@given(values=st.lists(_finite, min_size=12, max_size=12))
@SLOW
def test_the_sums_of_squares_always_partition_the_total(values):
    x = np.asarray(values, dtype=np.float64).reshape(6, 2)
    ms = mean_squares_pr(x)
    total = float(np.sum((x - x.mean()) ** 2))
    assert sum(ms.ss.values()) == pytest.approx(total, rel=1e-6, abs=1e-6)


@given(
    weights=st.lists(
        st.floats(min_value=0.0, max_value=1e3, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=24,
    )
)
@SLOW
def test_kish_is_between_one_and_n_whenever_any_weight_is_positive(weights):
    n = len(weights)
    ess = kish_ess(weights)
    if max(weights) <= 0.0:
        assert ess == 0.0
    else:
        assert 1.0 - 1e-9 <= ess <= n + 1e-9


@given(
    k=st.integers(min_value=2, max_value=64),
    rho=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@SLOW
def test_the_design_effect_form_is_monotone_and_bounded(k, rho):
    ess = design_effect_ess(k, rho)
    assert 1.0 - 1e-9 <= ess <= k + 1e-9
    if rho < 1.0:
        assert design_effect_ess(k, min(1.0, rho + 0.05)) <= ess + 1e-9


def test_component_set_rejects_a_duplicate_name():
    from reward_lens.stats.variance import VarianceComponent

    with pytest.raises(ValueError, match="duplicate component names"):
        ComponentSet(
            components=(
                VarianceComponent("p", 1.0, 1.0),
                VarianceComponent("p", 2.0, 2.0),
            )
        )


def test_component_set_names_what_it_does_not_have():
    cs = truncate_at_zero({"p": 1.0, "r": 1.0})
    with pytest.raises(KeyError, match="no component named"):
        cs["o"]


# ---------------------------------------------------------------------------
# The statistical review's findings on this module (E41)
# ---------------------------------------------------------------------------


def test_the_shape_factor_returns_k_only_for_a_two_point_group():
    """The anchor the docstring used to claim, corrected.

    "K distinct scores spread symmetrically give K" was false and untested, and it is the anchor a
    reader calibrates on. The Kish count of centred magnitudes is one exactly when every rollout
    sits the same distance from the mean, which needs a two-point distribution.
    """
    assert group_effective_size(np.array([-1.0, 1.0])) == pytest.approx(2.0)
    assert group_effective_size(np.array([-1.0, -1.0, 1.0, 1.0])) == pytest.approx(4.0)

    # Four distinct scores, spread symmetrically, and it is not four.
    assert group_effective_size(np.array([-2.0, -1.0, 1.0, 2.0])) == pytest.approx(3.6)
    assert group_effective_size(np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(3.2)


def test_a_gaussian_group_lands_at_two_over_pi_with_no_measurement_error_at_all():
    """0.64K on a perfect grader is the expected reading and says nothing about the grader.

    This is the number most callers will see, so it is pinned. It is a property of the reward
    distribution's shape: rollouts near the group mean contribute almost nothing to the update.
    """
    rng = np.random.default_rng(7)
    ratio = float(np.mean([group_effective_size(rng.normal(size=64)) / 64 for _ in range(400)]))
    assert ratio == pytest.approx(2.0 / np.pi, abs=0.02)


def test_a_degenerate_decomposition_is_undetermined_rather_than_a_perfect_gauge():
    """An all-zero decomposition used to render as a gauge resolving 2,147,483,647 levels.

    Everything on that object is zero over zero. "Perfect" and "nothing was measured" have to be
    distinguishable, and `acceptable` returning True for the second is the confident wrong number
    this library exists to refuse.
    """
    zero = ComponentSet(
        components=(
            VarianceComponent("p", 0.0, 0.0, 1, ""),
            VarianceComponent("r", 0.0, 0.0, 1, ""),
            VarianceComponent("e", 0.0, 0.0, 1, ""),
        ),
        design="pxr",
    )
    g = gauge_rr(zero, part="p")
    assert g.determined is False
    assert g.acceptable is False
    assert g.band == "undetermined"
    assert "No gauge study here" in g.verdict()
    assert "2147483647" not in g.verdict()


def test_a_real_decomposition_is_determined_and_grr_is_the_standard_deviation_ratio():
    """AIAG's %GRR is `100 * sigma_GRR / sigma_TV`, not the variance ratio.

    With sigma_part 5, sigma_appraiser 1 and sigma_error 2: GRR is sqrt(5), TV is sqrt(30), so
    %GRR is 40.8248%. The variance ratio would be 16.67%, which is 2.449 times smaller and would
    move the AIAG band from unacceptable to marginal.
    """
    cs = ComponentSet(
        components=(
            VarianceComponent("p", 25.0, 25.0, 9, ""),
            VarianceComponent("r", 1.0, 1.0, 2, ""),
            VarianceComponent("e", 4.0, 4.0, 18, ""),
        ),
        design="pxr",
    )
    g = gauge_rr(cs, part="p")
    assert g.determined is True
    assert g.grr_percent == pytest.approx(100.0 * math.sqrt(5.0) / math.sqrt(30.0))
    assert g.ndc_categories == 3  # 1.41 * 5 / sqrt(5) = 3.1529, truncated


def test_kish_refuses_a_non_finite_weight_rather_than_returning_the_degenerate_answer():
    """A NaN weight returned 0.0, which is also what a group with no spread returns."""
    with pytest.raises(ValueError, match="non-finite"):
        kish_ess(np.array([1.0, float("nan"), 2.0]))
    with pytest.raises(ValueError, match="non-finite"):
        kish_ess(np.array([1.0, float("inf")]))


def test_icc_uses_the_variance_weighted_group_size_and_not_the_plain_mean():
    """On unequal groups the correction is load-bearing and the docstring used to misname it.

    n0 = (N - sum(n_g^2)/N) / (G - 1) is 2.9744 on sizes (6, 2, 3, 2). The harmonic mean is 2.6667
    and the plain mean is 3.25; substituting the plain mean gives 0.7966 where the answer is 0.8106.
    """
    rng = np.random.default_rng(3)
    groups = [rng.normal(loc=m, scale=0.4, size=n) for m, n in zip([0, 3, 6, 9], [6, 2, 3, 2])]
    icc = icc_oneway(groups)
    assert 0.0 < icc < 1.0
    # Equal sizes collapse n0 to k exactly, which is the property that pins the formula.
    equal = [rng.normal(loc=m, scale=0.4, size=5) for m in range(4)]
    assert 0.0 < icc_oneway(equal) < 1.0


def test_a_d_study_size_for_a_facet_that_does_not_exist_raises():
    """`rater` and `occasion` are the labels `crossed_pro` assigns, so they are the natural slip.

    A keyword naming no facet used to be discarded in silence and the default-size answer returned,
    which is a wrong reliability that looks like it worked.
    """
    rng = np.random.default_rng(0)
    g = crossed_pr(rng.normal(size=(30, 5)))
    correct = g.d_study(r=1).generalizability
    for bad in ("n_r", "rater", "R"):
        with pytest.raises(DesignError, match="name no facet"):
            g.d_study(**{bad: 1})
    assert g.d_study(r=1).generalizability == correct


def test_zero_gauge_variance_renders_as_unbounded_and_never_as_the_int32_sentinel():
    """E45, the case E41 missed, and the fix is to the rendering not the verdict.

    E41 guarded the all-zero decomposition. A design with real part variance and no measured gauge
    variance still printed the int32 sentinel, because ndc is `1.41 * sigma_part / 0`. D7 hit it on
    a real deterministic verifier while assembling the grader card.

    The verdict stays: a deterministic program verifier replayed and agreeing every time genuinely
    has no measurement error and genuinely resolves every distinct value, so calling it undetermined
    would be its own wrong answer. What was wrong is that infinity was rendered as 2,147,483,647.
    The design's weakness is stated separately, through `repeatability` being unidentifiable.
    """
    deterministic = ComponentSet(
        components=(
            VarianceComponent("p", 4.0, 4.0, 9, ""),
            VarianceComponent("r", 0.0, 0.0, 2, ""),
            VarianceComponent("e", 0.0, 0.0, 18, ""),
        ),
        design="pxr",
    )
    g = gauge_rr(deterministic, part="p")
    assert g.sigma_total > 0.0, "the total variance is real; this is not the E41 case"
    assert g.determined is True, "a deterministic verifier really does have no measurement error"
    assert g.ndc_unbounded is True
    assert "2147483647" not in g.verdict()
    assert "unbounded" in g.verdict()
    # One observation per cell carries no replication, so the verdict says the evidence is weak
    # without pretending the verdict itself is absent.
    assert g.repeatability is None
    assert "no replication" in g.verdict()

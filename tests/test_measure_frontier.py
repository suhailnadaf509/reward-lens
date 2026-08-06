"""Unit tests for Level 0: the closed forms, the identities, the refusals and the group.

A Gaussian reward is the best test this layer has, because every object at this level has a closed
form under it and none of them involve a fit. For ``r ~ N(mu, sigma^2)``:

    K(lambda)  = lambda mu + lambda^2 sigma^2 / 2
    K'(lambda) = mu + lambda sigma^2
    KL         = lambda^2 sigma^2 / 2
    n / ESS    = exp(lambda^2 sigma^2) = exp(t^2)

That last line is the one worth staring at. The visibility horizon at a floor `f` is at
``t = sqrt(-log f)`` and therefore at ``KL = -log(f) / 2`` nats, whatever `mu` and `sigma` are. At
the 5% default that is ``log(20) / 2 = 1.4979`` nats exactly, and it is the same number for every
Gaussian grader ever built. So the horizon has a value this file can assert against arithmetic
rather than against a stored expectation.

The second closed form is the turn. With ``r ~ N(0, 1)`` and ``g = -(r - c)^2``,

    E_lambda[g] = -(1 + (lambda - c)^2)

which peaks at ``lambda = c`` exactly, and ``Cov_lambda(g, r) = -2(lambda - c)`` is exactly linear,
so the closed-form cumulant expression and HedgeTune must both land on ``c`` and the first is exact
rather than approximate there. Choosing ``c = 1.2`` puts the turn inside the 5% horizon at
``t = 1.73``, and choosing ``c = 2`` puts it outside, which is how the refusal gets a case where
the right answer is known and is a refusal.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from reward_lens.core.envelope import RegimeReading
from reward_lens.core.evidence import ValueCodec
from reward_lens.core.invariance import (
    INVARIANT,
    GroupAction,
    InvarianceGroup,
    InvariancePayload,
    check_invariance,
)
from reward_lens.core.quantity import QUANTITIES, load_quantities
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.frontier import (
    FRONTIER,
    ChecklistReading,
    ConcomitantBestOfN,
    ConcomitantReading,
    FrontierReading,
    GoldVersusKL,
    Potential,
    RewardTailIndex,
    SurrogateChecklist,
    TailReading,
    VisibilityHorizon,
    concomitant_expectation,
    cumulant_turning_point,
    hedgetune,
    hill,
    horizon_lambda,
    measure_checklist,
    measure_concomitant,
    measure_frontier,
    measure_horizon,
    measure_tail_index,
    pickands,
    simulate_concomitant,
)

SLOW = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


def gaussian(n: int = 20000, mu: float = 0.0, sigma: float = 1.0, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(mu, sigma, n)


def quadratic_gold(r: np.ndarray, c: float) -> np.ndarray:
    """`g = -(r - c)^2`, whose tilted mean peaks at `lambda = c` for a standard normal `r`."""
    return -((r - c) ** 2)


# ===========================================================================
# The potential, against its closed forms
# ===========================================================================


@pytest.mark.parametrize("mu,sigma", [(0.0, 1.0), (2.5, 0.4), (-1.0, 3.0)])
def test_the_cumulant_generating_function_matches_the_gaussian_closed_form(mu, sigma):
    """Checked at dimensionless pressures inside the horizon, which is where the claim is made.

    The grid is `t = lambda sigma` in [0, 1.5] rather than a fixed lambda, for two reasons. A fixed
    lambda means a different amount of tilt on each of these three graders. And past the horizon at
    `t = 1.73` the self-normalised estimator is biased at order 1/ESS: at `t = 2` on 200,000 draws
    the tilted mean comes out 4% low, which is not a tolerance to loosen, it is the reading the
    horizon exists to refuse.
    """
    pot = Potential(gaussian(200_000, mu, sigma, seed=1))
    for t in (0.0, 0.3, 1.0, 1.5):
        lam = t / sigma
        want_k = lam * mu + 0.5 * lam * lam * sigma * sigma
        assert pot.log_mgf(lam) == pytest.approx(want_k, rel=0.02, abs=0.01)
        assert pot.reward_mean(lam) == pytest.approx(mu + lam * sigma * sigma, rel=0.02, abs=0.01)
        assert pot.reward_var(lam) == pytest.approx(sigma * sigma, rel=0.06)
        assert pot.kl(lam) == pytest.approx(0.5 * lam * lam * sigma * sigma, rel=0.05, abs=0.01)


def test_the_horizon_of_a_gaussian_grader_is_log_twenty_over_two_nats_whatever_its_scale():
    """The one number in this layer that is the same for every grader of a given shape.

    Two claims and they deserve different tolerances. That the three graders agree with **each
    other** is exact arithmetic and is asserted to 1e-9: they are the same draw rescaled, and the
    estimator is built so that a rescaling reparametrises lambda and moves nothing else. That they
    agree with the **closed form** is a sampling question, and the horizon is a tail functional: its
    own spread across seeds at n = 200,000 is a few percent and it shrinks slowly, because where
    ESS crosses 5% is decided by the largest handful of scores. Measured over four seeds at each of
    n = 50,000 to 4,000,000, the estimate sits within 10% of log(20)/2 and does not settle inside
    3% until well past a million draws. That is a fact about this quantity worth knowing before
    quoting a horizon to three figures.
    """
    want = math.log(20.0) / 2.0
    readings = [
        measure_horizon(gaussian(200_000, 3.0, sigma, seed=2)) for sigma in (0.25, 1.0, 7.0)
    ]
    for sigma, reading in zip((0.25, 1.0, 7.0), readings):
        assert not isinstance(reading, Refusal)
        assert reading.binding
        assert reading.coverage_at_horizon == pytest.approx(20.0, rel=1e-6)
        assert reading.kl_max == pytest.approx(want, rel=0.10), sigma
        assert reading.t_max == pytest.approx(math.sqrt(math.log(20.0)), rel=0.05)
        # 1e-7 rather than machine epsilon because the horizon is found by bisection to a stated
        # relative tolerance of 1e-9, so that is the precision lambda_max is determined to at all.
        assert reading.lambda_max == pytest.approx(readings[1].lambda_max / sigma, rel=1e-7)
    assert readings[0].kl_max == pytest.approx(readings[2].kl_max, rel=1e-7)
    assert readings[0].t_max == pytest.approx(readings[2].t_max, rel=1e-7)


def test_the_max_is_subtracted_so_a_huge_lambda_does_not_overflow():
    """`exp(lambda r)` overflows float64 at an exponent of 710 and the failure is silent nans."""
    pot = Potential(np.array([0.0, 1.0, 2.0, 3.0]))
    w = pot.weights(5_000.0)
    assert np.all(np.isfinite(w)) and w.sum() == pytest.approx(1.0)
    assert w[-1] == pytest.approx(1.0)  # the maximum takes all the mass, which is correct
    assert np.isfinite(pot.log_mgf(5_000.0)) and np.isfinite(pot.kl(5_000.0))


def test_the_coverage_identity_is_the_chi_squared_divergence_written_twice():
    """`n / ESS = exp(K(2 lambda) - 2 K(lambda))` is an identity, not a resemblance."""
    pot = Potential(gaussian(5000, 0.0, 1.3, seed=3))
    for lam in (0.0, 0.2, 0.7, 1.4):
        assert math.log(pot.coverage(lam)) == pytest.approx(pot.log_coverage(lam), abs=1e-9)


def test_the_reward_moves_one_over_lambda_nats_per_unit_of_kl():
    """`dE_lambda[r] / dKL = 1 / lambda`, checked numerically on the empirical potential."""
    pot = Potential(gaussian(50_000, 0.0, 1.0, seed=4))
    for lam in (0.3, 0.8, 1.5):
        h = 1e-4
        d_reward = pot.reward_mean(lam + h) - pot.reward_mean(lam - h)
        d_kl = pot.kl(lam + h) - pot.kl(lam - h)
        assert d_reward / d_kl == pytest.approx(1.0 / lam, rel=1e-4)


# ===========================================================================
# Properties, drawn rather than chosen
# ===========================================================================


@given(
    scores=st.lists(
        st.floats(-50, 50, allow_nan=False, allow_infinity=False), min_size=4, max_size=200
    ),
    lam=st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
)
@SLOW
def test_ess_is_bounded_by_one_and_n(scores, lam):
    pot = Potential(np.array(scores))
    ess = pot.ess(lam)
    assert 1.0 - 1e-9 <= ess <= pot.n + 1e-9


@given(
    scores=st.lists(
        st.floats(-20, 20, allow_nan=False, allow_infinity=False), min_size=40, max_size=300
    ),
    lo=st.floats(0.02, 0.4),
    gap=st.floats(0.01, 0.4),
)
@SLOW
def test_the_horizon_is_monotone_in_the_floor(scores, lo, gap):
    """A higher floor cannot see further. This is what makes the bisection legitimate."""
    pot = Potential(np.array(scores))
    assume(not pot.is_degenerate)
    hi = min(0.95, lo + gap)
    assume(hi > lo)
    lam_low, _ = horizon_lambda(pot, floor=lo)
    lam_high, _ = horizon_lambda(pot, floor=hi)
    assert lam_high <= lam_low + 1e-6


@given(
    scores=st.lists(
        st.floats(-10, 10, allow_nan=False, allow_infinity=False), min_size=8, max_size=120
    ),
    a=st.floats(0.05, 20.0),
    b=st.floats(-30.0, 30.0),
)
@SLOW
def test_both_axes_of_the_frontier_survive_an_affine_rescaling_of_the_reward(scores, a, b):
    """`r -> a r + b` reparametrises lambda to lambda/a and leaves KL and ESS where they were."""
    r = np.array(scores)
    base = Potential(r)
    assume(not base.is_degenerate)
    moved = Potential(a * r + b)
    for t in (0.0, 0.4, 1.0, 1.6):
        p0 = base.at(float(base.lam_of(t)))
        p1 = moved.at(float(moved.lam_of(t)))
        assert p1.kl == pytest.approx(p0.kl, rel=1e-7, abs=1e-9)
        assert p1.ess == pytest.approx(p0.ess, rel=1e-7)


# ===========================================================================
# The turn: HedgeTune and the closed form, against a known answer
# ===========================================================================


def test_hedgetune_and_the_cumulant_form_both_find_a_turn_whose_location_is_known():
    """`g = -(r - 1.2)^2` on a standard normal `r` turns at exactly `lambda = 1.2`."""
    r = gaussian(200_000, 0.0, 1.0, seed=5)
    pot = Potential(r, quadratic_gold(r, 1.2))
    tuned = hedgetune(pot, lam_hi=1.75)
    closed = cumulant_turning_point(pot, lam_hi=1.75)
    assert tuned.found and tuned.is_maximum
    assert tuned.lam == pytest.approx(1.2, rel=0.02), tuned.detail
    assert closed.found and closed.lam == pytest.approx(1.2, rel=0.02), closed.detail
    # The population Cov_lambda(g, r) is exactly linear here, so one Newton step from zero is the
    # whole answer and the two estimators are solving the same linear equation. The empirical
    # covariance is not exactly linear, so on 200,000 draws they land about 1% apart rather than on
    # the same float. That gap is the sampling error and not the linearisation.
    assert closed.lam == pytest.approx(tuned.lam, rel=0.02)
    assert pot.gold_mean(tuned.lam) == pytest.approx(-1.0, abs=0.02)
    assert pot.kl(tuned.lam) == pytest.approx(0.5 * 1.2**2, rel=0.03)


def test_a_jointly_gaussian_pair_never_turns_and_both_estimators_say_so():
    """`E_lambda[g] = mu_g + lambda rho sigma_g sigma_r` is linear, so there is no turn to find."""
    rng = np.random.default_rng(6)
    r = rng.normal(0.0, 1.0, 50_000)
    g = 0.7 * r + rng.normal(0.0, 0.8, 50_000)
    pot = Potential(r, g)
    tuned = hedgetune(pot, lam_hi=1.7)
    closed = cumulant_turning_point(pot, lam_hi=1.7)
    assert not tuned.found and "rising" in tuned.detail
    assert not closed.found
    assert "not on the positive axis" in closed.detail or "no root" in closed.detail


def test_a_turn_past_the_horizon_is_not_reported_as_a_turn():
    """`c = 2` puts the turn at `t = 2`, past the 5% horizon at `t = 1.73`. Silence is correct."""
    r = gaussian(100_000, 0.0, 1.0, seed=7)
    reading = measure_frontier(r, quadratic_gold(r, 2.0), resamples=0)
    assert not isinstance(reading, Refusal)
    assert not reading.peak_is_interior
    assert not reading.hedgetune.found
    assert "no turn" in reading.says.lower() or "still rising" in reading.says


# ===========================================================================
# The tail index, against three distributions whose index is known
# ===========================================================================


@pytest.mark.parametrize(
    "name,make,want",
    [
        ("pareto alpha=2", lambda rng: (1.0 - rng.random(200_000)) ** (-0.5), 0.5),
        ("gaussian", lambda rng: rng.normal(0.0, 1.0, 200_000), 0.0),
        ("uniform", lambda rng: rng.random(200_000), -1.0),
    ],
)
def test_the_invariant_tail_estimate_recovers_a_known_index(name, make, want):
    reading = measure_tail_index(make(np.random.default_rng(8)), resamples=120)
    assert not isinstance(reading, Refusal), name
    assert reading.invariant_gamma == pytest.approx(want, abs=0.35), name
    assert reading.plateau_found, name
    assert reading.light_tailed is (want <= 0.0), name


def test_hill_cannot_represent_the_light_tailed_case_it_is_being_asked_about():
    """Hill's support is `gamma > 0`, so a Gaussian gets a small positive number and not zero.

    This is the reason the light-tailed verdict is taken from Pickands. It also puts the one
    cited measurement in perspective: an exactly Gaussian tail produces a Hill estimate in the
    same neighbourhood as the 0.20 reported as "consistent with light-tailed error", so 0.20 is consistent with a tail that is not heavy at all.
    """
    reading = measure_tail_index(np.random.default_rng(9).normal(0.0, 1.0, 200_000), resamples=60)
    assert reading.hill_gamma > 0.0
    assert reading.hill_gamma < 0.35
    assert reading.invariant_gamma < reading.hill_gamma


def test_pickands_is_exactly_location_and_scale_invariant_and_hill_is_only_scale_invariant():
    """A finding about the estimators, pinned here so a future change to either is visible."""
    x = np.random.default_rng(10).normal(3.0, 1.0, 20_000)
    desc = np.sort(x)[::-1]
    scaled = np.sort(2.5 * x)[::-1]
    shifted = np.sort(2.5 * x + 1.7)[::-1]
    assert pickands(shifted, 200) == pytest.approx(pickands(desc, 200), rel=1e-9)
    assert hill(scaled, 200) == pytest.approx(hill(desc, 200), rel=1e-12)
    assert hill(shifted, 200) != pytest.approx(hill(desc, 200), rel=1e-3)


# ===========================================================================
# The concomitant, against simulation at the same n
# ===========================================================================


@pytest.mark.parametrize("n", [2, 4, 16, 64])
def test_the_exact_concomitant_expectation_agrees_with_simulation_at_the_same_n(n):
    """The exact expression and its listed baseline, on the same pairs and the same tie rule."""
    rng = np.random.default_rng(11)
    r = rng.normal(0.0, 1.0, 500)
    g = 0.6 * r + rng.normal(0.0, 0.8, 500)
    exact, _, _ = concomitant_expectation(r, g, n)
    sim, se = simulate_concomitant(r, g, n, replicates=40_000, seed=12)
    assert abs(exact - sim) < 4.0 * se, (exact, sim, se)


def test_the_concomitant_is_exact_on_a_tied_binary_proxy_where_the_answer_is_arithmetic():
    """A verifier scoring 0 or 1 is all ties, and the exact answer is one line of algebra.

    With a pass rate `p`, best-of-n picks a passing rollout unless all n fail, so
    ``E[G] = (1 - (1-p)^n) * mean(g | pass) + (1-p)^n * mean(g | fail)``.
    """
    r = np.array([1.0] * 30 + [0.0] * 70)
    g = np.array([0.9] * 30 + [0.2] * 70)
    p = 0.3
    for n in (1, 2, 5, 10):
        want = (1 - (1 - p) ** n) * 0.9 + (1 - p) ** n * 0.2
        got, _, _ = concomitant_expectation(r, g, n)
        assert got == pytest.approx(want, rel=1e-12), n


def test_the_concomitant_depends_only_on_the_rank_order_of_the_proxy():
    """Which is why it is exactly invariant under `r -> a r + b` for positive `a`."""
    rng = np.random.default_rng(13)
    r = rng.normal(0.0, 2.0, 300)
    g = rng.normal(0.0, 1.0, 300) + 0.4 * r
    base, _, _ = concomitant_expectation(r, g, 32)
    moved, _, _ = concomitant_expectation(3.7 * r - 11.0, g, 32)
    assert moved == pytest.approx(base, rel=1e-12)


# ===========================================================================
# The refusals, each with its reason and its remedy
# ===========================================================================


def _ctx() -> Context:
    """No signal. Level 0 reads injected scores, which is why it runs before anything exists."""
    return Context()


def test_the_horizon_refuses_when_the_floor_is_worth_less_than_one_sample():
    out = VisibilityHorizon(gaussian(10, seed=14)).estimate(_ctx())
    assert isinstance(out, Refusal) and out.reason is RefusalReason.ESS_BELOW_FLOOR
    assert "less than one" in out.detail
    assert "draw at least 20 rollouts" in out.remedy
    assert out.statistics["floor_in_samples"] == pytest.approx(0.5)


def test_the_horizon_refuses_on_a_grader_with_no_spread_at_all():
    """An all-pass verifier. The tilt family is a single point and there is no axis for a horizon."""
    out = VisibilityHorizon(np.full(500, 1.0)).estimate(_ctx())
    assert isinstance(out, Refusal) and out.reason is RefusalReason.BELOW_LOD
    assert "zero spread" in out.detail
    assert "score a bank the grader actually separates" in out.remedy


def test_the_frontier_refuses_past_the_horizon_and_hands_back_the_visible_part():
    """The headline refusal of this package, and it is a bounded one."""
    r = gaussian(20_000, 0.0, 1.0, seed=15)
    out = GoldVersusKL(r, quadratic_gold(r, 1.2), lambda_max=4.0, resamples=0).estimate(_ctx())
    assert isinstance(out, Refusal) and out.reason is RefusalReason.ESS_BELOW_FLOOR
    assert out.is_bounded, "a refusal that can still show the visible curve should show it"
    assert isinstance(out.partial.value, FrontierReading)
    assert "re-run with lambda_max <=" in out.remedy
    assert "draw more rollouts" in out.remedy
    assert out.statistics["requested_lambda"] == 4.0
    assert out.partial.value.kl_max == pytest.approx(out.statistics["kl_max"])
    assert "guess wearing an interval" in out.detail


def test_the_frontier_refuses_without_a_gold_channel_and_says_what_still_runs():
    out = GoldVersusKL(gaussian(500, seed=16)).estimate(_ctx())
    assert isinstance(out, Refusal) and out.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "GOLD:QUERY" in out.detail
    assert "`VisibilityHorizon` and `RewardTailIndex` still run" in out.remedy


def test_the_tail_index_refuses_below_the_exceedance_count_and_names_the_n_it_needs():
    """The stated number: a defensible index needs about 1,570 exceedances at q = 0.95."""
    out = RewardTailIndex(gaussian(200, seed=17), resamples=40).estimate(_ctx())
    assert isinstance(out, Refusal) and out.reason is RefusalReason.ABOVE_LOD_BELOW_LOQ
    assert out.is_bounded and isinstance(out.partial.value, TailReading)
    assert out.statistics["exceedances"] < 1570
    assert out.statistics["n_required"] == 31_400
    assert "31,400" in out.remedy
    assert "not worth quoting as a number" in out.remedy


def test_the_concomitant_refuses_when_best_of_n_collapses_onto_the_top_of_the_bank():
    rng = np.random.default_rng(18)
    r = rng.normal(0.0, 1.0, 100)
    out = ConcomitantBestOfN(r, 0.5 * r, best_of=5000).estimate(_ctx())
    assert isinstance(out, Refusal) and out.reason is RefusalReason.ESS_BELOW_FLOOR
    assert "effective distinct proxy values" in out.detail
    assert "ask for a smaller n" in out.remedy


def test_the_checklist_refuses_when_the_grader_moved_under_it():
    """N4's envelope: two channels measured across a grader change are two graders, not one."""
    rng = np.random.default_rng(19)
    r = rng.normal(0.0, 1.0, 400)
    out = SurrogateChecklist(r, 0.5 * r + rng.normal(0.0, 1.0, 400)).estimate(
        Context(regime_reading=RegimeReading.of(STATIONARY_GRADER=False))
    )
    assert isinstance(out, Refusal) and out.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "STATIONARY_GRADER" in out.detail
    assert out.remedy.strip()


def test_the_checklist_refuses_without_a_gold_channel():
    out = SurrogateChecklist(gaussian(400, seed=20)).estimate(_ctx())
    assert isinstance(out, Refusal) and out.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "GOLD:QUERY" in out.detail
    assert "a stronger grader" in out.remedy


# ===========================================================================
# The four criteria
# ===========================================================================


def test_every_criterion_returns_untestable_without_a_treatment_except_the_ones_that_cannot():
    """One arm gives no treatment effect to mediate, and saying so is the reading."""
    rng = np.random.default_rng(21)
    r = rng.normal(0.0, 1.0, 600)
    g = 0.5 * r + rng.normal(0.0, 1.0, 600)
    reading = measure_checklist(r, g, permutations=300, resamples=120)
    assert isinstance(reading, ChecklistReading)
    assert reading.verdict_of(2) == "untestable"
    assert reading.verdict_of(3) == "untestable"
    assert reading.verdict_of(4) == "untestable"
    assert reading.n_untestable >= 3
    for c in reading.criteria:
        if c.verdict == "untestable":
            assert c.testable_by.strip(), c.name


def test_a_proxy_that_is_anticorrelated_with_gold_fails_vanderweeles_necessary_condition():
    """The surrogate paradox, detected: pushing this proxy up pushes gold down."""
    rng = np.random.default_rng(22)
    r = rng.normal(0.0, 1.0, 800)
    g = -0.7 * r + rng.normal(0.0, 0.6, 800)
    reading = measure_checklist(r, g, permutations=300, resamples=120)
    assert reading.verdict_of(4) == "fail"
    assert "surrogate paradox is not excluded" in reading.render()


def test_a_proxy_that_does_not_mediate_the_treatment_fails_prentices_fourth_condition():
    """Treatment reaches gold by a route the proxy does not carry, which is the failure that matters."""
    rng = np.random.default_rng(23)
    n = 1200
    t = (rng.random(n) < 0.5).astype(float)
    r = 0.8 * t + rng.normal(0.0, 1.0, n)
    g = 0.3 * r + 1.5 * t + rng.normal(0.0, 1.0, n)  # a direct path from t to g
    reading = measure_checklist(r, g, treatment=t, permutations=300, resamples=200)
    fourth = reading.prentice_conditions[3]
    assert fourth.verdict == "fail"
    assert "does not fully mediate" in fourth.detail
    assert reading.verdict_of(1) == "fail"


def test_prentices_fourth_condition_can_pass_only_against_a_stated_equivalence_margin():
    """A non-significant coefficient is not evidence of absence, and the verdict says so."""
    rng = np.random.default_rng(24)
    n = 2000
    t = (rng.random(n) < 0.5).astype(float)
    r = 1.2 * t + rng.normal(0.0, 1.0, n)
    g = 0.9 * r + rng.normal(0.0, 1.0, n)  # fully mediated by construction
    bare = measure_checklist(r, g, treatment=t, permutations=300, resamples=200)
    assert bare.prentice_conditions[3].verdict == "untestable"
    assert "not evidence of full capture" in bare.prentice_conditions[3].detail
    assert "equivalence" in bare.prentice_conditions[3].testable_by

    stated = measure_checklist(
        r, g, treatment=t, equivalence_margin=0.5, permutations=300, resamples=200
    )
    assert stated.prentice_conditions[3].verdict == "pass"


def test_the_trial_level_coefficient_is_estimable_once_units_carry_both_arms():
    rng = np.random.default_rng(25)
    n_units, per = 20, 60
    unit = np.repeat(np.arange(n_units), per)
    t = np.tile(np.repeat([0.0, 1.0], per // 2), n_units)
    effect = rng.normal(1.0, 0.5, n_units)[unit]
    r = effect * t + rng.normal(0.0, 0.5, n_units * per)
    g = 0.9 * r + rng.normal(0.0, 0.3, n_units * per)
    reading = measure_checklist(
        r, g, treatment=t, unit=unit, permutations=200, resamples=120, r2_target=0.5
    )
    third = [c for c in reading.criteria if c.number == 3][0]
    assert third.verdict in {"pass", "fail"}
    assert "trial-level R^2" in third.detail


# ===========================================================================
# The generated invariance test, one per instrument
# ===========================================================================


def _affine_payload(seed: int, n: int = 4000) -> tuple[InvariancePayload, np.ndarray]:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 1.0, n)
    g = quadratic_gold(r, 1.2) + rng.normal(0.0, 0.3, n)
    return InvariancePayload(scores=r), g


def test_the_horizon_is_invariant_under_an_affine_rescaling_of_the_reward():
    payload, _ = _affine_payload(26)
    report = check_invariance(
        VisibilityHorizon(),
        "reward.affine",
        payload,
        n=12,
        relation=INVARIANT,
        run=lambda inst, p: measure_horizon(np.asarray(p.scores), grid=9).kl_max,
    )
    assert report.passed, report.render()
    assert report.n == 12


def test_the_frontier_is_invariant_under_an_affine_rescaling_of_the_reward():
    """Both axes. `lambda -> lambda / a` is what makes it true and the sweep is built to do it."""
    payload, g = _affine_payload(27)

    def run(inst, p):
        return measure_frontier(np.asarray(p.scores), g, grid=17, resamples=0).peak_kl

    report = check_invariance(
        GoldVersusKL(), "reward.affine", payload, n=10, relation=INVARIANT, run=run
    )
    assert report.passed, report.render()


def test_the_tail_index_is_invariant_under_an_affine_rescaling_of_the_reward():
    payload, _ = _affine_payload(28, n=6000)

    def run(inst, p):
        return measure_tail_index(
            np.asarray(p.scores), min_exceedances=1, resamples=0
        ).invariant_gamma

    report = check_invariance(
        RewardTailIndex(), "reward.affine", payload, n=10, relation=INVARIANT, run=run
    )
    assert report.passed, report.render()


def test_the_hill_estimator_alone_is_invariant_under_the_scale_subgroup_and_not_the_whole_group():
    """A finding about the estimator, recorded as a test rather than argued in a comment.

    The subgroup drawn here is `r -> a r` with no shift, which is a proper subgroup of
    `reward.affine` and is built locally rather than registered, because it is a diagnostic and not
    a restriction class any instrument declares itself under.
    """

    def scale_only() -> InvarianceGroup:
        def make(a: float) -> GroupAction:
            return GroupAction(
                name=f"r -> {a:.4g} r",
                apply=lambda p: p.replace(scores=a * np.asarray(p.scores)),
                params={"a": a},
                sample=lambda rng: make(math.exp(rng.uniform(math.log(0.2), math.log(5.0)))),
            )

        return InvarianceGroup(
            id="reward.scale",
            generators=(make(1.0),),
            acts_on="scores",
            admits=frozenset({"invariant", "covariant"}),
        )

    payload, _ = _affine_payload(29, n=6000)

    def run_hill(inst, p):
        return measure_tail_index(np.asarray(p.scores), min_exceedances=1, resamples=0).hill_gamma

    scaled = check_invariance(
        RewardTailIndex(), scale_only(), payload, n=8, relation=INVARIANT, run=run_hill
    )
    assert scaled.passed, scaled.render()

    shifted = check_invariance(
        RewardTailIndex(), "reward.affine", payload, n=8, relation=INVARIANT, run=run_hill
    )
    assert not shifted.passed, (
        "Hill is not location invariant and this assertion is the record of that. If it starts "
        "passing, either the estimator changed or the group stopped drawing shifts."
    )
    assert "moved under an affine rescaling" in shifted.interpretation


def test_the_checklist_is_invariant_under_an_affine_rescaling_of_the_reward():
    """The proportion explained is a ratio of two coefficients on `t`, and `r` only spans a column."""
    rng = np.random.default_rng(30)
    n = 900
    t = (rng.random(n) < 0.5).astype(float)
    base = 0.8 * t + rng.normal(0.0, 1.0, n)
    g = 0.4 * base + 0.9 * t + rng.normal(0.0, 1.0, n)

    def run(inst, p):
        reading = measure_checklist(
            np.asarray(p.scores), g, treatment=t, permutations=50, resamples=40
        )
        return [c for c in reading.criteria if c.number == 2][0].statistic

    report = check_invariance(
        SurrogateChecklist(),
        "reward.affine",
        InvariancePayload(scores=base),
        n=8,
        relation=INVARIANT,
        run=run,
    )
    assert report.passed, report.render()


def test_the_concomitant_is_invariant_under_an_affine_rescaling_of_the_reward():
    rng = np.random.default_rng(31)
    r = rng.normal(0.0, 1.0, 800)
    g = 0.5 * r + rng.normal(0.0, 1.0, 800)

    def run(inst, p):
        return measure_concomitant(np.asarray(p.scores), g, best_of=16, resamples=0).expected_gold

    report = check_invariance(
        ConcomitantBestOfN(),
        "reward.affine",
        InvariancePayload(scores=r),
        n=10,
        relation=INVARIANT,
        run=run,
    )
    assert report.passed, report.render()


# ===========================================================================
# Declarations and the store
# ===========================================================================


def test_all_five_instruments_pass_lint():
    load_quantities()
    for cls in FRONTIER:
        assert lint_instrument(cls()) == [], cls.__name__


def test_the_five_quantities_are_registered_and_carry_the_group_the_instruments_declare():
    load_quantities()
    ids = {
        "frontier.gold_vs_kl",
        "frontier.visibility_horizon",
        "frontier.tail_index",
        "frontier.prentice_checklist",
        "frontier.concomitant_bon",
    }
    assert ids <= set(QUANTITIES)
    for qid in ids:
        assert QUANTITIES.get(qid).invariance == "reward.affine"
        assert QUANTITIES.get(qid).wedge is True
    assert {cls.quantity for cls in FRONTIER} == ids


def test_every_instrument_declares_an_envelope_a_group_and_a_baseline():
    for cls in FRONTIER:
        assert cls.envelope is not None, cls.__name__
        assert cls.invariance and cls.baselines, cls.__name__
        assert cls.faithful_to and cls.deviations, cls.__name__
        assert cls.phases and cls.substrates, cls.__name__


def test_every_payload_round_trips_through_the_store_codec(tmp_path):
    """`ValueCodec` raises on an unregistered payload rather than degrading to a dict."""
    rng = np.random.default_rng(32)
    r = rng.normal(0.0, 1.0, 400)
    g = quadratic_gold(r, 1.0) + rng.normal(0.0, 0.2, 400)
    t = (rng.random(400) < 0.5).astype(float)
    payloads = [
        measure_horizon(r),
        measure_frontier(r, g, resamples=0),
        measure_tail_index(r, min_exceedances=1, resamples=0),
        measure_checklist(r, g, treatment=t, permutations=100, resamples=60),
        measure_concomitant(r, g, best_of=8, resamples=0),
    ]
    codec = ValueCodec()
    for payload in payloads:
        assert not isinstance(payload, Refusal), payload
        encoded = codec.encode(payload, tmp_path)
        back = codec.decode(encoded, tmp_path)
        assert type(back) is type(payload)
        assert back.baselines == pytest.approx(payload.baselines)


def test_the_readings_carry_their_baselines_where_a_card_can_find_them():
    rng = np.random.default_rng(33)
    r = rng.normal(0.0, 1.0, 600)
    g = quadratic_gold(r, 1.1)
    assert "baseline.hedgetune" in measure_frontier(r, g, resamples=0).baselines
    assert "baseline.raw_n" in measure_horizon(r).baselines
    assert (
        "baseline.exponential_tail_scale"
        in measure_tail_index(r, min_exceedances=1, resamples=0).baselines
    )
    concomitant = measure_concomitant(r, g, best_of=8, resamples=0, simulate=2000)
    assert "baseline.simulated_best_of_n" in concomitant.baselines
    assert isinstance(concomitant, ConcomitantReading)

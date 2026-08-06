"""Unit tests for the ARL design: three independent routes to the same number.

The Siegmund closed form, the renewal integral equation and direct Monte Carlo are three
implementations of the same quantity that share no code. Where they agree the number is established;
where they disagree the simulation is the arbiter, because it is the definition of an average run
length rather than an approximation to it.

Two checks here have closed forms the code does not know about, which makes them the strongest ones
in the file: at ``lam = 1`` the EWMA is a Shewhart chart and its ARL is ``1/(2 Phi(-L))``, and a
fixed ``L``-sigma threshold on independent normals has the same run length. Neither the EWMA
integral equation nor the CUSUM quadrature contains that formula.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import norm

from reward_lens.monitor.arl import (
    RHO,
    SIEGMUND_REFERENCE,
    arl_integral_equation,
    arl_monte_carlo,
    arl_siegmund,
    design_cusum,
    kl_gaussian,
    lorden_delay,
    shipped_ad_hoc_arl0,
    solve_h,
)
from reward_lens.monitor.ewma import arl_ewma, design_ewma, ewma_alarm, sigma_z

# ---------------------------------------------------------------------------
# The closed form, by hand
# ---------------------------------------------------------------------------


def test_siegmund_matches_the_formula_written_out() -> None:
    """`(exp(-2db) + 2db - 1) / (2 d^2)` with `d = -k`, `b = h + 1.166`, halved for two sides."""
    h, k = 4.77, 0.5
    b = h + RHO
    d = -k
    one_sided = (math.exp(-2 * d * b) + 2 * d * b - 1) / (2 * d * d)
    assert arl_siegmund(h, k, 0.0, 1) == pytest.approx(one_sided, rel=1e-12)
    assert arl_siegmund(h, k, 0.0, 2) == pytest.approx(one_sided / 2, rel=1e-12)


def test_the_removable_singularity_is_filled_by_its_limit() -> None:
    """At `shift == k` the formula is 0/0 and its limit is `b^2`."""
    h, k = 3.0, 0.5
    assert arl_siegmund(h, k, k, 1) == pytest.approx((h + RHO) ** 2)


def test_the_two_sided_chart_is_half_the_one_sided_one_in_control() -> None:
    for h in (3.0, 4.77, 6.0):
        assert arl_siegmund(h, 0.5, 0.0, 2) == pytest.approx(
            arl_siegmund(h, 0.5, 0.0, 1) / 2, rel=1e-12
        )


def test_siegmund_reproduces_the_textbook_pairs() -> None:
    """`k = 0.5, h = 4` gives 168 two-sided and 336 one-sided; `h = 5` gives 465 and 938."""
    assert arl_siegmund(4.0, 0.5, 0.0, 2) == pytest.approx(168, rel=0.01)
    assert arl_siegmund(4.0, 0.5, 0.0, 1) == pytest.approx(336, rel=0.01)
    assert arl_siegmund(5.0, 0.5, 0.0, 2) == pytest.approx(465, rel=0.01)
    assert arl_siegmund(5.0, 0.5, 0.0, 1) == pytest.approx(938, rel=0.01)


# ---------------------------------------------------------------------------
# The three routes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h", [4.0, 4.77, 5.0, 5.75])
def test_the_closed_form_and_the_integral_equation_agree_within_one_percent(h: float) -> None:
    closed = arl_siegmund(h, 0.5, 0.0, 2)
    exact = arl_integral_equation(h, 0.5, 0.0, 2)
    assert abs(closed - exact) / exact < 0.012, (h, closed, exact)


def test_monte_carlo_agrees_with_both_at_the_first_design_point() -> None:
    mean, se = arl_monte_carlo(4.77, 0.5, 0.0, 2, n_runs=20000, seed=5)
    assert abs(mean - 370.0) < 4 * se
    assert abs(mean - arl_siegmund(4.77, 0.5, 0.0, 2)) < 4 * se


def test_monte_carlo_contradicts_the_second_stated_design_point() -> None:
    """E47. The spec states `h = 5.71` for `ARL_0 = 1000`; simulation says 956."""
    mean, se = arl_monte_carlo(5.71, 0.5, 0.0, 2, n_runs=20000, seed=6)
    assert mean + 4 * se < 1000
    assert 900 < mean < 1000
    stated = {p.arl0_target: p for p in SIEGMUND_REFERENCE}
    assert stated[370.0].agrees and not stated[1000.0].agrees


# ---------------------------------------------------------------------------
# The design
# ---------------------------------------------------------------------------


def test_the_solver_inverts_the_closed_form() -> None:
    for target in (50.0, 100.0, 370.0, 1000.0, 10000.0):
        h = solve_h(target, 0.5, 2)
        assert arl_siegmund(h, 0.5, 0.0, 2) == pytest.approx(target, rel=1e-6)


@settings(max_examples=40, deadline=None)
@given(
    st.floats(min_value=20.0, max_value=100000.0),
    st.floats(min_value=0.1, max_value=2.0),
)
def test_the_solved_threshold_always_hits_its_target(arl0: float, k: float) -> None:
    h = solve_h(arl0, k, 2)
    assert arl_siegmund(h, k, 0.0, 2) == pytest.approx(arl0, rel=1e-5)


def test_the_design_sets_k_to_half_the_shift_and_nothing_else_is_free() -> None:
    d = design_cusum(1.4, 500.0)
    assert d.k == 0.7
    assert d.arl0_siegmund == pytest.approx(500.0, rel=1e-6)


def test_a_quieter_chart_costs_a_logarithm_rather_than_a_factor() -> None:
    """Lorden: `log(gamma)/KL`. Ten times quieter costs `log 10 / KL` extra steps."""
    assert kl_gaussian(1.0) == 0.5
    assert lorden_delay(1000.0, 1.0) - lorden_delay(100.0, 1.0) == pytest.approx(math.log(10) / 0.5)
    assert lorden_delay(1000.0, 1.0) / lorden_delay(100.0, 1.0) < 1.5


def test_the_shipped_ad_hoc_threshold_implies_an_arl_nobody_stated() -> None:
    implied = shipped_ad_hoc_arl0()
    assert implied == pytest.approx(469, rel=0.01)
    assert solve_h(implied, 0.5, 2) == pytest.approx(5.0, abs=0.005)


def test_a_target_below_one_observation_is_rejected() -> None:
    with pytest.raises(ValueError):
        solve_h(0.5, 0.5, 2)


def test_a_nonpositive_shift_is_rejected() -> None:
    with pytest.raises(ValueError):
        design_cusum(0.0, 370.0)


# ---------------------------------------------------------------------------
# EWMA, against a closed form the code does not contain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L", [2.0, 3.0, 3.09])
def test_the_ewma_at_lambda_one_reproduces_the_shewhart_identity(L: float) -> None:
    """`lam = 1` makes the EWMA a Shewhart chart, whose ARL is `1/(2 Phi(-L))` exactly.

    The integral equation does not contain that formula, so agreement to 12 digits is a check on
    the quadrature rather than a tautology.
    """
    got = arl_ewma(1.0, L * sigma_z(1.0, 1.0))
    want = 1.0 / (2.0 * norm.cdf(-L))
    assert got == pytest.approx(want, rel=1e-9)


def test_the_ewma_design_reproduces_lucas_and_saccucci() -> None:
    """Published: ARL_0 = 370 needs L = 2.703 at lam = 0.10 and 2.898 at lam = 0.25."""
    assert design_ewma(370.0, 0.10).multiplier == pytest.approx(2.703, abs=0.002)
    assert design_ewma(370.0, 0.25).multiplier == pytest.approx(2.898, abs=0.002)
    assert design_ewma(370.0, 1.00).multiplier == pytest.approx(3.000, abs=0.002)


def test_the_three_sigma_convention_is_wrong_at_a_small_lambda() -> None:
    """`L = 3` is a Shewhart reflex. At `lam = 0.1` it buys 842 steps, not 370."""
    assert arl_ewma(0.1, 3.0 * sigma_z(0.1)) == pytest.approx(842, rel=0.01)


def test_the_ewma_alarms_on_a_real_shift_and_not_on_noise() -> None:
    import numpy as np

    design = design_ewma(370.0, 0.2)
    rng = np.random.default_rng(2)
    quiet = rng.standard_normal(200)
    assert ewma_alarm(quiet, design) is None
    shifted = np.concatenate([rng.standard_normal(50), rng.standard_normal(150) + 2.0])
    where = ewma_alarm(shifted, design)
    assert where is not None and 50 <= where < 70


def test_a_lambda_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(ValueError):
        design_ewma(370.0, 1.5)

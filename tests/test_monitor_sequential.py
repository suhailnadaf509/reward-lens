"""Unit tests for `stats.sequential` and the vendored `cif.py`, against hand-computed values.

Everything here is checked against arithmetic that can be done on paper, a closed form the code does
not know about, or the vendored reference implementation. Nothing is checked against the code's own
previous output.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reward_lens.monitor._vendor import cif
from reward_lens.stats.sequential import (
    bounded_evalue_increment,
    ebh,
    fixed_sample_radius,
    merge_e,
    peeking_miscoverage,
    running_evalue,
    stitched_radius,
    ville_pvalue,
    ville_threshold,
    wilson_interval,
)

VENDORED = (
    Path(__file__).resolve().parents[1] / "src" / "reward_lens" / "monitor" / "_vendor" / "cif.py"
)
UPSTREAM_SHA256 = "4c9824c9d46b681dfca7688c42080a5f8dd701aad0bd9933105ea68325812a51"


# ---------------------------------------------------------------------------
# The vendored file
# ---------------------------------------------------------------------------


def test_the_vendored_body_is_byte_identical_to_what_was_fetched() -> None:
    """The header claims the body is unmodified. This is what makes that claim checkable."""
    text = VENDORED.read_text(encoding="utf-8")
    marker = "# --- begin vendored code/cif.py"
    start = text.index(marker)
    body = text[text.index("\n", start) + 1 :]
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == UPSTREAM_SHA256


def test_the_vendored_file_carries_its_licence_and_attribution() -> None:
    text = VENDORED.read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "Copyright (c) 2026 Amir Asiaee, AsiaeeLab" in text
    assert "AsiaeeLab/certified-interventional-fidelity" in text
    assert "Permission is hereby granted" in text
    assert UPSTREAM_SHA256 in text


def test_the_alpha_spending_schedule_sums_to_the_budget() -> None:
    """`sum_n 6d/(pi^2 n^2) = d` because `sum 1/n^2 = pi^2/6`. Hand-checkable, and it holds."""
    delta = 0.05
    total = sum(cif.spending_delta(n, delta) for n in range(1, 200000))
    assert abs(total - delta) < 1e-6, total


def test_the_alpha_spending_radius_matches_the_formula_by_hand() -> None:
    """`sqrt(log(2/d_n) / 2n)` at n = 100, delta = 0.05, computed independently."""
    delta_n = 6 * 0.05 / (math.pi**2 * 100**2)
    expected = math.sqrt(math.log(2.0 / delta_n) / 200.0)
    assert cif.cs_radius(100, 0.05) == pytest.approx(expected, rel=1e-15)


def test_the_local_fixed_sample_radius_matches_the_vendored_one() -> None:
    """The one duplicated function, pinned so it cannot drift."""
    for n in (1, 5, 100, 10000):
        for delta in (0.01, 0.05, 0.1):
            assert fixed_sample_radius(n, delta) == pytest.approx(
                cif.hoeffding_fixed_radius(n, delta), rel=0, abs=0
            )


# ---------------------------------------------------------------------------
# Ville
# ---------------------------------------------------------------------------


def test_ville_pvalue_and_threshold_are_reciprocals() -> None:
    assert ville_pvalue(20.0) == 0.05
    assert ville_threshold(0.05) == 20.0
    assert ville_pvalue(0.5) == 1.0, "a martingale that never got above 1 rejects nothing"
    assert ville_pvalue(float("inf")) == 1.0, "non-finite is not evidence"


@settings(max_examples=200, deadline=None)
@given(st.floats(min_value=1.0, max_value=1e12))
def test_ville_pvalue_is_a_probability(m: float) -> None:
    assert 0.0 <= ville_pvalue(m) <= 1.0


def test_the_ville_bound_holds_under_simulation() -> None:
    """Simulate the betting martingale under the null and count how often it ever crosses `1/alpha`.

    Ville says at most alpha. The realised rate should be at or below it, and it is well below,
    because the bound is not tight for a bet that is not tuned to the alternative.
    """
    alpha = 0.05
    rng = np.random.default_rng(3)
    crossings = 0
    n_runs = 800
    for _ in range(n_runs):
        xs = rng.random(400)  # Uniform[0,1], so the null mean 0.5 is true
        capital = running_evalue(xs, 0.5, adaptive=True)
        if np.max(capital) >= ville_threshold(alpha):
            crossings += 1
    rate = crossings / n_runs
    assert rate <= alpha, rate


# ---------------------------------------------------------------------------
# The betting increment
# ---------------------------------------------------------------------------


def test_the_betting_factor_is_hand_computable() -> None:
    """`1 + lam (x - m)` at lam = 0.5, x = 1, m = 0.2 is 1.4."""
    assert bounded_evalue_increment(1.0, 0.2, 0.5) == pytest.approx(1.4)
    assert bounded_evalue_increment(0.0, 0.2, 0.5) == pytest.approx(0.9)


def test_a_fixed_bet_gives_the_product_of_its_factors() -> None:
    """With `adaptive=False` the capital is exactly the running product, computable by hand."""
    xs = [1.0, 0.0, 1.0]
    got = running_evalue(xs, 0.5, lam=0.4, adaptive=False)
    expected = np.cumprod([1.2, 0.8, 1.2])
    assert np.allclose(got, expected)


def test_the_bet_is_clipped_so_the_factor_stays_positive() -> None:
    """A stake beyond `1/m` would make the factor negative at x = 0, which is bankruptcy."""
    factor = bounded_evalue_increment(0.0, 0.1, 50.0)
    assert factor > 0.0


@settings(max_examples=100, deadline=None)
@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=40),
    st.floats(min_value=0.05, max_value=0.95),
)
def test_the_capital_is_never_negative(xs: list[float], m: float) -> None:
    capital = running_evalue(xs, m)
    assert np.all(capital >= 0.0)


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------


def test_stitching_is_narrower_than_alpha_spending_and_the_gap_grows() -> None:
    """The rung-1 claim, measured. The docstring's numbers are these."""
    ratios = {}
    for n in (30, 100, 1000, 10000):
        ratios[n] = cif.cs_radius(n, 0.05) / stitched_radius(n, 0.05)
    assert ratios[30] == pytest.approx(1.33, abs=0.01)
    assert ratios[100] == pytest.approx(1.41, abs=0.01)
    assert ratios[1000] == pytest.approx(1.56, abs=0.01)
    assert ratios[10000] == pytest.approx(1.70, abs=0.01)
    assert ratios[30] < ratios[100] < ratios[1000] < ratios[10000]


def test_the_stitched_radius_is_wider_than_the_fixed_sample_one() -> None:
    """Anytime validity is not free. If it ever looks free, something is wrong."""
    for n in (10, 100, 10000):
        assert stitched_radius(n, 0.05) > fixed_sample_radius(n, 0.05)


def test_a_smaller_sigma_narrows_the_stitched_interval() -> None:
    """The one argument that can produce a confident wrong number, behaving as documented."""
    assert stitched_radius(100, 0.05, sigma=0.25) < stitched_radius(100, 0.05, sigma=0.5)


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def test_merging_under_arbitrary_dependence_is_the_mean_and_not_the_product() -> None:
    merged = merge_e([2.0, 8.0, 20.0], dependence="arbitrary")
    assert merged.value == pytest.approx(10.0)
    assert "arbitrary dependence" in merged.assumption
    product = merge_e([2.0, 8.0, 20.0], dependence="independent")
    assert product.value == pytest.approx(320.0)
    assert "not under arbitrary dependence" in product.assumption


def test_the_mean_merge_is_valid_under_perfect_dependence_and_the_product_is_not() -> None:
    """The concrete counterexample to the published claim, simulated.

    Three perfectly dependent e-values, each a valid e-value on its own. Under the null a valid
    merged e-value has expectation at most 1. The mean does; the product does not, by a wide margin.
    """
    rng = np.random.default_rng(9)
    # e = exp(z - 1/2) with z ~ N(0,1) has E[e] = 1 exactly, so it is a valid e-value.
    z = rng.standard_normal(200000)
    e = np.exp(z - 0.5)
    means = np.mean(np.vstack([e, e, e]), axis=0)
    products = e * e * e
    assert abs(np.mean(means) - 1.0) < 0.05, np.mean(means)
    assert np.mean(products) > 10.0, np.mean(products)


def test_merging_rejects_a_negative_evalue() -> None:
    with pytest.raises(ValueError):
        merge_e([1.0, -1.0])


# ---------------------------------------------------------------------------
# e-BH
# ---------------------------------------------------------------------------


def test_ebh_by_hand() -> None:
    """K = 4, alpha = 0.05. The cutoff at rank k is `K/(alpha k)` = 20, 40, 26.7, 20.

    Sorted descending: [100, 50, 30, 1]. Rank 1 needs 80, rank 2 needs 40, rank 3 needs 26.67,
    rank 4 needs 20. The largest admissible k is 3 (30 >= 26.67), so the top three are rejected.
    """
    result = ebh([100.0, 50.0, 1.0, 30.0], alpha=0.05)
    assert result.n_rejected == 3
    assert list(result.rejected) == [True, True, False, True]
    assert result.threshold == pytest.approx(4 / (0.05 * 3))


def test_ebh_rejects_nothing_when_no_evalue_is_large_enough() -> None:
    result = ebh([1.0, 2.0, 3.0], alpha=0.05)
    assert result.n_rejected == 0
    assert not result.rejected.any()


def test_ebh_treats_a_nonfinite_evalue_as_zero_rather_than_dropping_it() -> None:
    """Dropping it would shrink K and make the survivors easier to reject."""
    with_nan = ebh([100.0, float("nan")], alpha=0.05)
    assert with_nan.n_hypotheses == 2
    assert list(with_nan.rejected) == [True, False]


def test_ebh_controls_fdr_under_arbitrary_dependence() -> None:
    """Simulated under a global null with perfectly dependent e-values, which is the hard case."""
    rng = np.random.default_rng(21)
    false_discoveries = 0
    trials = 3000
    for _ in range(trials):
        z = rng.standard_normal()
        e = np.full(10, math.exp(z - 0.5))
        if ebh(e, alpha=0.05).n_rejected > 0:
            false_discoveries += 1
    assert false_discoveries / trials <= 0.05, false_discoveries / trials


# ---------------------------------------------------------------------------
# The peeking simulation
# ---------------------------------------------------------------------------


def test_the_wilson_interval_matches_a_hand_computation() -> None:
    """k = 5, n = 20, z = 1.96. Centre (5 + 1.9208)/(20 + 3.8416) = 0.29026."""
    lo, hi = wilson_interval(np.array([5.0]), np.array([20.0]))
    z = 1.959963984540054
    centre = (5 + z * z / 2) / (20 + z * z)
    half = (z * math.sqrt(20) / (20 + z * z)) * math.sqrt(0.25 * 0.75 + z * z / 80)
    assert lo[0] == pytest.approx(centre - half)
    assert hi[0] == pytest.approx(centre + half)


def test_the_wilson_interval_is_defined_at_zero_successes() -> None:
    """Where the normal approximation collapses to zero width and rigs the comparison."""
    lo, hi = wilson_interval(np.array([0.0]), np.array([20.0]))
    assert lo[0] == 0.0
    assert hi[0] > 0.0


def test_peeking_costs_the_fixed_sample_interval_its_level() -> None:
    cost = peeking_miscoverage(lambda n: cif.cs_radius(n, 0.05), n_runs=2000, n_steps=200, seed=1)
    assert cost.wilson_miscoverage > 0.25
    assert cost.anytime_miscoverage <= 0.05
    assert cost.hoeffding_miscoverage < cost.wilson_miscoverage


def test_a_burn_in_makes_the_fixed_interval_look_better_and_the_reading_records_it() -> None:
    early = peeking_miscoverage(lambda n: cif.cs_radius(n, 0.05), n_runs=2000, seed=1, burn_in=1)
    late = peeking_miscoverage(lambda n: cif.cs_radius(n, 0.05), n_runs=2000, seed=1, burn_in=20)
    assert late.wilson_miscoverage < early.wilson_miscoverage
    assert late.burn_in == 20 and early.burn_in == 1

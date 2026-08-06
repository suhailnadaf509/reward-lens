"""Unit tests for power, MDE and the resolution ratio (M10, `reward_lens.stats.power`).

The simulator is the ground truth in this module, so it is the thing checked first and against
something outside itself: at a zero effect it must reject at the nominal alpha, which is the size
of the test rather than its power. Everything else is checked against the simulator, including the
two calculators that are supposed to be right and the three that are not.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.stats.power import (
    ALPHA,
    CALCULATORS,
    DEFAULT_CLOSE_PAIR,
    DIMENSIONLESS,
    EFFECT,
    USES_CORRELATION,
    DetectionBand,
    PairedBinaryDesign,
    PositionBiasDesign,
    PowerQuantity,
    alpha_for_family,
    compare_calculators,
    detection_band,
    difference,
    minimum_detectable_effect,
    n_cohen_h,
    plan,
    required_n,
    resolution_from_lineage,
    resolution_ratio,
    rho_bounds,
    simulate_power,
)

# ---------------------------------------------------------------------------
# The design
# ---------------------------------------------------------------------------


def test_the_four_cells_are_a_distribution_and_reproduce_the_marginals():
    d = PairedBinaryDesign(n=100, accuracy_a=0.7, accuracy_b=0.8, rho=0.4)
    p11, p10, p01, p00 = d.cells
    assert sum(d.cells) == pytest.approx(1.0)
    assert all(c >= 0 for c in d.cells)
    assert p11 + p10 == pytest.approx(d.accuracy_a)
    assert p11 + p01 == pytest.approx(d.accuracy_b)


def test_a_correlation_outside_the_frechet_bounds_is_refused_at_construction():
    lo, hi = rho_bounds(0.5, 0.9)
    assert hi < 1.0
    with pytest.raises(ValueError, match="achievable range"):
        PairedBinaryDesign(n=100, accuracy_a=0.5, accuracy_b=0.9, rho=0.99)


def test_a_higher_correlation_means_fewer_disagreements():
    low = PairedBinaryDesign(n=100, accuracy_a=0.82, accuracy_b=0.85, rho=0.0)
    high = PairedBinaryDesign(n=100, accuracy_a=0.82, accuracy_b=0.85, rho=0.8)
    assert high.discordance < low.discordance


# ---------------------------------------------------------------------------
# The simulator, checked against something outside itself
# ---------------------------------------------------------------------------


def test_at_a_zero_effect_the_simulator_reproduces_the_nominal_size():
    """Not power: size. An exact test is conservative, so this rejects at or below alpha."""
    null = PairedBinaryDesign(n=400, accuracy_a=0.8, accuracy_b=0.8, rho=0.5, alpha=0.05)
    got = simulate_power(null, replicates=20_000, seed=0)
    assert got.power <= 0.05 + 4 * got.mc_se
    assert got.power > 0.02, "an exact McNemar should not be this conservative at n=400"


def test_power_rises_with_n_and_with_the_effect():
    base = PairedBinaryDesign(n=200, accuracy_a=0.82, accuracy_b=0.85, rho=0.5)
    assert (
        simulate_power(base.at_n(2000), replicates=4000).power
        > simulate_power(base, replicates=4000).power
    )
    assert (
        simulate_power(base.at_delta(0.10), replicates=4000).power
        > simulate_power(base.at_delta(0.03), replicates=4000).power
    )


def test_required_n_returns_the_smallest_n_reaching_the_target():
    d = PairedBinaryDesign(n=1, accuracy_a=0.82, accuracy_b=0.85, rho=0.5)
    n_star = required_n(d, target_power=0.8, replicates=6000, seed=0)
    assert simulate_power(d.at_n(n_star), replicates=20_000, seed=0).power >= 0.78
    assert simulate_power(d.at_n(int(n_star * 0.7)), replicates=20_000, seed=0).power < 0.8


def test_the_minimum_detectable_effect_is_the_smallest_effect_this_n_can_find():
    d = PairedBinaryDesign(n=400, accuracy_a=0.82, accuracy_b=0.85, rho=0.5)
    mde = minimum_detectable_effect(d, target_power=0.8, replicates=6000, seed=0)
    assert 0.0 < mde < 0.2
    assert simulate_power(d.at_delta(mde), replicates=20_000, seed=0).power >= 0.78
    assert simulate_power(d.at_delta(mde * 0.6), replicates=20_000, seed=0).power < 0.8


# ---------------------------------------------------------------------------
# The five calculators
# ---------------------------------------------------------------------------


def test_cohen_h_reproduces_the_published_table_entry_of_63_per_group():
    """The external anchor. Cohen's medium effect h = 0.5 needs 63 per group at 80% power."""
    phi_a = 2.0 * math.asin(math.sqrt(0.25))
    p_b = math.sin((phi_a + 0.5) / 2.0) ** 2
    d = PairedBinaryDesign(n=10, accuracy_a=0.25, accuracy_b=p_b, rho=0.0)
    assert n_cohen_h(d, 0.8) == pytest.approx(63.0, abs=1.0)


def test_with_no_correlation_all_five_calculators_agree_with_the_simulation():
    """The control for the finding: it is about pairing, so with no pairing there is no gap."""
    uncorrelated = PairedBinaryDesign(n=500, accuracy_a=0.82, accuracy_b=0.85, rho=0.0)
    checks = compare_calculators(uncorrelated, replicates=6000, seed=0)
    assert not [c.name for c in checks.values() if c.roughly_2x_wrong]


def test_three_of_five_calculators_are_roughly_2x_wrong_on_a_close_paired_comparison():
    """The finding, reproduced by simulation rather than quoted.

    Two close models on a shared benchmark at a per-item correctness correlation of 0.5. The three
    calculators that never look at the correlation ask for roughly twice the n the test actually
    needs; the two that use it land within a few percent.
    """
    checks = compare_calculators(DEFAULT_CLOSE_PAIR, replicates=8000, seed=0)
    off = {c.name for c in checks.values() if c.roughly_2x_wrong}
    assert off == {name for name, uses in USES_CORRELATION.items() if not uses}
    assert len(off) == 3
    for name in off:
        assert 1.6 <= checks[name].ratio <= 2.3, (name, checks[name].ratio)
    for name in (n for n, uses in USES_CORRELATION.items() if uses):
        assert 0.85 <= checks[name].ratio <= 1.15, (name, checks[name].ratio)


def test_the_inflation_grows_as_one_over_one_minus_rho():
    """At rho = 0.8 the same three calculators are near 5x, not near 2x."""
    strong = PairedBinaryDesign(n=500, accuracy_a=0.82, accuracy_b=0.85, rho=0.8)
    checks = compare_calculators(strong, replicates=6000, seed=0)
    for name, uses in USES_CORRELATION.items():
        if not uses:
            assert checks[name].ratio > 3.5, (name, checks[name].ratio)


def test_every_calculator_is_named_and_declares_whether_it_uses_the_correlation():
    assert set(CALCULATORS) == set(USES_CORRELATION)
    assert sum(USES_CORRELATION.values()) == 2


# ---------------------------------------------------------------------------
# The resolution ratio
# ---------------------------------------------------------------------------


def test_q_below_one_reports_not_resolved_rather_than_a_verdict():
    r = resolution_ratio(500, 1274)
    assert r.q == pytest.approx(500 / 1274)
    assert not r.resolved
    assert "NOT RESOLVED" in r.render()
    assert "short by 774" in r.render()


def test_q_at_or_above_one_reports_resolved():
    r = resolution_ratio(2000, 1274)
    assert r.resolved and "RESOLVED" in r.render() and "NOT RESOLVED" not in r.render()


def test_the_resolution_ratio_uses_lineages_when_the_rows_are_expansions():
    rows = [f"seed{i // 25}" for i in range(500)]  # 20 seeds, 25 rows each
    r = resolution_from_lineage(rows, 100.0)
    assert r.n == pytest.approx(20.0)
    assert r.basis == "effective sample size"
    assert not r.resolved  # 500 rows would have looked resolved; 20 lineages are not


def test_planning_alpha_for_a_family_is_the_bonferroni_level():
    assert alpha_for_family(0.05, 20) == pytest.approx(0.0025)
    assert alpha_for_family(0.05, 1) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# The detection band
# ---------------------------------------------------------------------------


def test_the_position_bias_effect_collapses_as_base_accuracy_approaches_one():
    """The mechanism behind the band: the bias is fixed, its expression in probability is not."""
    mid = PositionBiasDesign(n=400, base_accuracy=0.75, logit_advantage=0.5).accuracies
    high = PositionBiasDesign(n=400, base_accuracy=0.98, logit_advantage=0.5).accuracies
    assert (mid[0] - mid[1]) > 5.0 * (high[0] - high[1])


def test_the_band_has_a_computed_upper_edge_below_one():
    band = detection_band(
        n=400,
        logit_advantage=0.5,
        replicates=2500,
        grid=tuple(np.round(np.arange(0.55, 0.995, 0.05), 4)),
        seed=0,
    )
    assert not band.is_empty
    assert band.high < 0.98
    assert band.read(0.75) == "measurable"
    assert band.read(0.99) == "not_measurable_above"


def test_a_null_above_the_band_reads_as_not_measurable_and_never_as_unbiased():
    band = DetectionBand(
        low=0.60, high=0.90, target_power=0.8, n=400, logit_advantage=0.5, alpha=ALPHA
    )
    above = band.interpret(0.98, detected=False)
    assert "NOT MEASURABLE" in above
    assert "unbiased" in above  # only as the thing it must not be read as
    assert "Read it as not measurable, not as unbiased" in above
    inside = band.interpret(0.75, detected=False)
    assert "informative" in inside and "NOT MEASURABLE" not in inside


def test_a_band_whose_lower_edge_is_the_grid_edge_says_so():
    band = detection_band(
        n=1000,
        logit_advantage=0.6,
        replicates=1500,
        grid=(0.60, 0.70, 0.80, 0.90),
        seed=0,
    )
    assert band.low_is_grid_edge
    assert "no lower edge in this model" in band.render()


# ---------------------------------------------------------------------------
# The plan, and units
# ---------------------------------------------------------------------------


def test_the_plan_carries_every_number_and_says_what_validated_it():
    p = plan(DEFAULT_CLOSE_PAIR, replicates=4000, seed=0, with_calculators=True)
    assert p.validated_against == "simulation"
    assert p.n_star > DEFAULT_CLOSE_PAIR.n
    assert not p.adequate and not p.resolution.resolved
    assert len(p.calculators) == 5
    assert "NOT RESOLVED" in p.render()


def test_an_effective_n_overrides_the_numerator_of_q():
    p = plan(DEFAULT_CLOSE_PAIR, replicates=3000, seed=0, ess=40.0)
    assert p.resolution.n == 40.0
    assert p.resolution.basis == "effective sample size"


def test_subtracting_a_power_from_an_mde_refuses_rather_than_converting():
    power = PowerQuantity("study.power", 0.8, DIMENSIONLESS)
    mde = PowerQuantity("study.mde", 0.03, EFFECT)
    out = difference(power, mde)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.UNIT_MISMATCH


def test_subtracting_two_readings_of_the_same_quantity_is_allowed():
    a = PowerQuantity("study.power", 0.8, DIMENSIONLESS)
    b = PowerQuantity("study.power", 0.5, DIMENSIONLESS)
    assert difference(a, b).value == pytest.approx(0.3)

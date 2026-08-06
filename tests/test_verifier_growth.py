"""D6: Good-Turing, Chao1, and the Crow-AMSAA growth exponent on an exploit log.

Every number this module produces is arithmetic anyone can check on paper, so the tests are
written against numbers computed on paper rather than against a previous run's output. The planted
power law is the exception and it is the important one: a fit that cannot recover an exponent
somebody put there is not a fit.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reward_lens.core.envelope import RegimeCondition
from reward_lens.core.invariance import InvariancePayload, check_invariance
from reward_lens.core.quantity import ladder
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import lint_instrument
from reward_lens.verifier.growth import (
    MIN_FIT_POINTS,
    CrowFit,
    ExploitCoverage,
    ExploitFamilyCoverage,
    ExploitFind,
    ExploitLog,
    ReliabilityGrowth,
    chao1_unseen,
    chao1_unseen_bias_corrected,
    crow_amsaa,
    exploit_coverage,
    novelty_probability,
    observations_for_a_doubleton,
)

#: A log small enough to do by hand. Families A once, B twice, C three times, D once.
#: n = 7, S_obs = 4, f1 = 2 (A and D), f2 = 1 (B).
HAND_LOG = ExploitLog.of(
    [
        ("A", 1.0),
        ("B", 2.0),
        ("B", 3.0),
        ("C", 4.0),
        ("C", 5.0),
        ("C", 6.0),
        ("D", 7.0),
    ],
    total_effort=10.0,
    source="hand-checked",
)


# ---------------------------------------------------------------------------
# The frequency spectrum
# ---------------------------------------------------------------------------


def test_the_spectrum_of_the_hand_log_is_the_one_counted_by_hand() -> None:
    assert HAND_LOG.n == 7
    assert HAND_LOG.s_obs == 4
    assert HAND_LOG.counts() == {"A": 1, "B": 2, "C": 3, "D": 1}
    assert HAND_LOG.spectrum() == {1: 2, 2: 1, 3: 1}


def test_good_turing_and_chao1_match_the_arithmetic_done_on_paper() -> None:
    """f1/n = 2/7, f1^2/(2 f2) = 4/2 = 2, Chao1 = 4 + 2 = 6, unseen fraction = 2/6."""
    reading = exploit_coverage(HAND_LOG, rung=2)
    assert isinstance(reading, ExploitCoverage)
    assert reading.f1 == 2 and reading.f2 == 1
    assert reading.novelty_probability == pytest.approx(2 / 7)
    assert reading.unseen_families == pytest.approx(2.0)
    assert reading.chao1 == pytest.approx(6.0)
    assert reading.unseen_fraction == pytest.approx(2 / 6)
    assert reading.unseen_fraction_of_observed == pytest.approx(2 / 4)


def test_the_mandatory_baseline_is_the_raw_family_count_and_it_is_on_the_reading() -> None:
    """*Base: the raw count of families found.* It has to be printable beside the claim."""
    reading = exploit_coverage(HAND_LOG, rung=0)
    assert isinstance(reading, ExploitCoverage)
    assert reading.n_families == 4
    assert "baseline (the raw count of families found): 4" in reading.render()


def test_rung_zero_computes_no_growth_fit_and_no_chao1() -> None:
    """A lower rung is a smaller answer, not a worse one."""
    reading = exploit_coverage(HAND_LOG, rung=0)
    assert isinstance(reading, ExploitCoverage)
    assert reading.fit is None
    assert reading.chao1 is None
    assert reading.unseen_families == pytest.approx(2.0)


def test_a_log_where_nothing_repeats_twice_gives_zero_unseen_rather_than_a_refusal() -> None:
    """f1 = 0 is a real reading. Every family found more than once means nothing points outward."""
    log = ExploitLog.of([("A", 1.0), ("A", 2.0), ("A", 3.0), ("B", 4.0), ("B", 5.0), ("B", 6.0)])
    reading = exploit_coverage(log, rung=2)
    assert isinstance(reading, ExploitCoverage)
    assert reading.f1 == 0 and reading.f2 == 0
    assert reading.unseen_fraction == 0.0
    assert reading.novelty_probability == 0.0
    assert any("stopped turning up new things" in n for n in reading.notes)


# ---------------------------------------------------------------------------
# The f2 = 0 refusal, which is the canonical one in this package
# ---------------------------------------------------------------------------


def test_no_doubleton_refuses_with_a_bound_and_a_remedy_naming_the_n_needed() -> None:
    """*Refusal is a success.* f2 = 0 is not infinity and is not a very large number."""
    log = ExploitLog.of([("A", 1.0), ("B", 2.0), ("C", 3.0), ("D", 4.0), ("E", 5.0)])
    refusal = exploit_coverage(log, rung=2)
    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.ABOVE_LOD_BELOW_LOQ
    assert "f2 = 0" in refusal.detail
    assert refusal.statistics["f1"] == 5 and refusal.statistics["f2"] == 0

    needed = refusal.statistics["additional_finds_needed"]
    assert needed >= 1
    assert f"about {needed} more find" in refusal.remedy

    # The bound is the bias-corrected estimator, f1*(f1-1)/2 = 5*4/2 = 10, and it is attached
    # rather than lost, which is the difference between "I cannot tell you" and "at least ten".
    assert refusal.is_bounded
    assert refusal.partial is not None
    assert refusal.partial.value == pytest.approx(10.0)
    assert "f1*(f1-1)/2 = 10.0" in refusal.remedy

    # Good-Turing's novelty probability needs no doubletons, and the remedy says so rather than
    # leaving the caller with nothing.
    assert "f1/n = 1.000" in refusal.remedy


def test_the_bias_corrected_bound_is_zero_for_a_single_singleton() -> None:
    assert chao1_unseen_bias_corrected(0) == 0.0
    assert chao1_unseen_bias_corrected(1) == 0.0
    assert chao1_unseen_bias_corrected(5) == 10.0


def test_the_novelty_probability_needs_no_doubletons_and_is_nan_on_an_empty_log() -> None:
    """`f1/n` is the one number here that survives `f2 = 0`, which is why the remedy quotes it."""
    assert novelty_probability(3, 12) == pytest.approx(0.25)
    assert novelty_probability(0, 12) == 0.0
    assert math.isnan(novelty_probability(0, 0))


def test_chao1_unseen_is_nan_at_f2_zero_rather_than_infinity() -> None:
    """NaN propagates into a comparison as False; infinity propagates as a confident answer."""
    assert math.isnan(chao1_unseen(3, 0))
    assert chao1_unseen(0, 0) == 0.0
    assert chao1_unseen(4, 2) == pytest.approx(4.0)


def test_an_empty_log_refuses_rather_than_reporting_a_clean_grader() -> None:
    refusal = exploit_coverage(ExploitLog.of([]))
    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "not a claim that the grader is clean" in refusal.remedy


@pytest.mark.parametrize(("n", "f1", "s_obs"), [(5, 5, 5), (20, 20, 20), (10, 2, 4), (1, 1, 1)])
def test_the_n_needed_is_always_a_positive_integer(n: int, f1: int, s_obs: int) -> None:
    needed = observations_for_a_doubleton(n, f1, s_obs)
    assert isinstance(needed, int) and needed >= 1


# ---------------------------------------------------------------------------
# Crow-AMSAA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.5, 0.74, 1.0, 1.3])
def test_the_fit_recovers_a_planted_exponent(beta: float) -> None:
    """N(t) = lambda t^beta with lambda = 2, sampled exactly. The fit must return beta."""
    lam = 2.0
    t = [float(k) for k in range(1, 40)]
    n = [lam * ti**beta for ti in t]
    fit = crow_amsaa(t, n)
    assert isinstance(fit, CrowFit)
    assert fit.beta == pytest.approx(beta, abs=1e-9)
    assert fit.lam == pytest.approx(lam, rel=1e-9)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-12)
    # An exact power law has zero residual, so the interval collapses onto the point.
    assert fit.ci_low == pytest.approx(beta, abs=1e-6)
    assert fit.ci_high == pytest.approx(beta, abs=1e-6)


def test_a_noisy_planted_exponent_is_inside_its_own_interval() -> None:
    """The interval is the claim. A fit whose interval misses its own truth is not reporting one."""
    import numpy as np

    rng = np.random.default_rng(0)
    beta, lam = 0.7, 3.0
    t = np.arange(1, 60, dtype=float)
    n = lam * t**beta * np.exp(rng.normal(0.0, 0.05, t.size))
    fit = crow_amsaa(t, n)
    assert isinstance(fit, CrowFit)
    assert fit.ci_low < beta < fit.ci_high
    assert fit.ci_high - fit.ci_low < 0.1
    assert fit.converging_at_interval is True


def test_the_interval_spanning_one_is_reported_as_undecided_rather_than_as_converging() -> None:
    fit = crow_amsaa([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 2.0, 4.0])
    assert isinstance(fit, CrowFit)
    assert fit.ci_low < 1.0 < fit.ci_high
    assert fit.converging_at_interval is None
    assert "does not yet settle" in fit.render()


def test_the_crow_mle_is_reported_beside_the_least_squares_slope() -> None:
    """MIL-HDBK-189's estimator, as the cross-check the least-squares fit needs."""
    t = [1.0, 2.0, 4.0, 8.0, 16.0]
    fit = crow_amsaa(t, [1, 2, 3, 4, 5], total_effort=32.0)
    assert isinstance(fit, CrowFit)
    expected = 5 / sum(math.log(32.0 / ti) for ti in t)
    assert fit.beta_mle == pytest.approx(expected)


def test_two_points_refuse_with_the_slope_as_a_bound_because_an_interval_needs_a_residual() -> None:
    refusal = crow_amsaa([1.0, 2.0], [1, 2])
    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.ABOVE_LOD_BELOW_LOQ
    assert refusal.is_bounded
    assert refusal.partial is not None
    assert refusal.partial.value == pytest.approx(1.0)
    assert f"at least {MIN_FIT_POINTS} finds" in refusal.remedy


def test_one_point_refuses_outright() -> None:
    refusal = crow_amsaa([3.0], [1])
    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert not refusal.is_bounded


def test_a_constant_effort_axis_refuses_rather_than_dividing_by_zero() -> None:
    refusal = crow_amsaa([5.0, 5.0, 5.0, 5.0], [1, 2, 3, 4])
    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "an axis that moves" in refusal.remedy


# ---------------------------------------------------------------------------
# The degenerate axis, which is the trap this module exists to avoid
# ---------------------------------------------------------------------------


def test_fitting_finds_against_the_find_ordinal_is_refused_rather_than_returning_one() -> None:
    """N(t) = t returns beta = 1.0 for every log ever written. That is the axis, not a finding."""
    log = ExploitLog.of(["A", "B", "A", "C", "D", "B"])
    assert not log.has_effort
    reading = ReliabilityGrowth(log, fit_on="finds").estimate()
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "fitting N(t) = t" in reading.detail
    assert "fit_on='families'" in reading.remedy


def test_the_degenerate_axis_really_would_have_returned_exactly_one() -> None:
    """The refusal above is not defensive. This is the number it declines to print."""
    t = [float(i) for i in range(1, 12)]
    fit = crow_amsaa(t, list(range(1, 12)), fit_on="finds")
    assert isinstance(fit, CrowFit)
    assert fit.beta == pytest.approx(1.0)
    assert fit.r_squared == pytest.approx(1.0)


def test_fitting_families_against_the_find_ordinal_is_allowed_and_carries_the_caveat() -> None:
    # A:3, B:3, C:2, D:1 -> f1 = 1, f2 = 1, so the unseen bound is defined and the fit runs.
    log = ExploitLog.of(["A", "A", "B", "A", "B", "C", "D", "B", "C"])
    reading = exploit_coverage(log, rung=1, fit_on="families")
    assert isinstance(reading, ExploitCoverage)
    assert reading.fit is not None
    assert reading.fit.beta < 1.0
    assert any("against the find ordinal" in n for n in reading.notes)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@given(
    families=st.lists(st.sampled_from(["a", "b", "c", "d", "e", "f", "g"]), min_size=1, max_size=60)
)
@settings(max_examples=200, deadline=None)
def test_the_reading_is_never_an_exception_and_never_a_silent_nan(families: list[str]) -> None:
    """*A refusal is a value.* Every log either reads or refuses, with nothing in between."""
    log = ExploitLog.of([(f, float(i + 1)) for i, f in enumerate(families)])
    out = exploit_coverage(log, rung=2)
    if isinstance(out, Refusal):
        assert out.remedy.strip()
        assert out.reason in {
            RefusalReason.ABOVE_LOD_BELOW_LOQ,
            RefusalReason.ACCESS_INSUFFICIENT,
        }
        return
    assert 0.0 <= out.unseen_fraction < 1.0
    assert 0.0 <= out.novelty_probability <= 1.0
    assert out.n_families == len(set(families))
    assert out.n_finds == len(families)


@given(
    scale=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
    beta=st.floats(min_value=0.2, max_value=2.0),
)
@settings(max_examples=50, deadline=None)
def test_beta_does_not_move_when_the_effort_axis_is_rescaled(scale: float, beta: float) -> None:
    """The exponent is a property of the growth, not of what the odometer counts in.

    Rescaling `t` moves `lambda` and leaves `beta` alone, which is the whole reason a Duane plot is
    read off the slope: a team logging GPU-hours and a team logging rollouts get the same exponent
    from the same run.
    """
    t = [float(k) for k in range(1, 30)]
    n = [2.0 * ti**beta for ti in t]
    base = crow_amsaa(t, n)
    moved = crow_amsaa([ti * scale for ti in t], n)
    assert isinstance(base, CrowFit) and isinstance(moved, CrowFit)
    assert moved.beta == pytest.approx(base.beta, abs=1e-8)


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------


def test_both_instruments_pass_lint_and_declare_all_six() -> None:
    for inst in (ExploitFamilyCoverage(HAND_LOG), ReliabilityGrowth(HAND_LOG)):
        assert lint_instrument(inst) == [], inst.name
        assert inst.quantity and inst.requires and inst.substrates and inst.phases
        assert inst.envelope is not None and inst.invariance and inst.baselines


def test_the_envelope_requires_stationary_grader_and_names_d10_as_its_measurement() -> None:
    """A log spanning a grader patch mixes two populations, and D10 is what detects that."""
    inst = ExploitFamilyCoverage(HAND_LOG)
    assert inst.envelope is not None
    assert inst.envelope.requires == frozenset({RegimeCondition.STATIONARY_GRADER})
    assert inst.envelope.measured_by[RegimeCondition.STATIONARY_GRADER] == "env.replay_fidelity"


def test_the_registered_ladder_matches_what_the_quantity_registry_declares() -> None:
    """`rungs: 3` for the unseen mass and `rungs: 1` for beta, as `spec/QUANTITIES.yaml` says."""
    assert [e.rung for e in ladder("verifier.unseen_exploit_mass")] == [0, 1, 2]
    assert [e.rung for e in ladder("verifier.reliability_growth_beta")] == [1]


def test_the_generated_invariance_test_passes_for_both() -> None:
    """*No instrument merges without its generated invariance test passing.*

    Both declare `none`, which resolves to the trivial group: no affine rescaling of the reward
    acts on a count of exploit families or on the exponent of their arrival process, and that is
    an answer rather than an omission. See E11.
    """
    for inst in (ExploitFamilyCoverage(HAND_LOG), ReliabilityGrowth(HAND_LOG)):
        # `none` is the registry's answer and `trivial` is the group it resolves to; the kernel's
        # `get_group` takes the group id, so the same translation D1 and D9's tests make is made
        # here. See E11.
        group = inst.invariance if inst.invariance != "none" else "trivial"
        report = check_invariance(inst, group, InvariancePayload(), n=4)
        assert report.passed
        assert "trivial group" in report.skipped


def test_a_find_without_a_family_is_a_construction_error() -> None:
    with pytest.raises(ValueError, match="needs a family name"):
        ExploitFind(family="")


def test_a_non_finite_effort_is_a_construction_error() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        ExploitFind(family="a", effort=float("inf"))


def test_two_different_logs_get_two_different_checksums() -> None:
    """The log is the subject, so the same instrument on two logs is two evidence rows."""
    other = ExploitLog.of([("A", 1.0), ("B", 2.0), ("B", 3.0)])
    assert HAND_LOG.checksum() != other.checksum()
    assert HAND_LOG.checksum() == ExploitLog.of(list(HAND_LOG.finds), total_effort=10.0).checksum()


def test_the_crow_exponent_is_the_mle_under_both_censoring_schemes():
    """The filter `ti < horizon` changes what `len(ratios)` means and the numerator must not follow.

    With `total_effort` the test is time-truncated, the horizon sits beyond every failure and all n
    terms survive. Without it the horizon is the last failure, whose term is log(1) = 0 and is
    filtered out, so the sum runs to n-1 while the MLE's numerator stays n. Reporting
    `len(ratios)/sum` gave the MLE in the first case and one short of it in the second, and
    `total_effort` defaults to None so the second is the path most logs take. E42.
    """
    import math

    t = [1.0, 2.0, 4.0, 8.0, 16.0]
    n = [1, 2, 3, 4, 5]

    failure_truncated = crow_amsaa(t, n, fit_on="families")
    s = sum(math.log(16.0 / x) for x in t if x < 16.0)
    assert failure_truncated.beta_mle == pytest.approx(5 / s)
    assert failure_truncated.beta_unbiased == pytest.approx(3 / s)

    time_truncated = crow_amsaa(t, n, fit_on="families", total_effort=32.0)
    s2 = sum(math.log(32.0 / x) for x in t)
    assert time_truncated.beta_mle == pytest.approx(5 / s2)
    assert time_truncated.beta_unbiased == pytest.approx(4 / s2)


def test_the_bias_corrected_exponent_recovers_a_planted_beta_where_the_mle_does_not():
    """The MLE is 32% high at eight failures, which is the size an exploit log actually has."""
    import math

    import numpy as np

    rng = np.random.default_rng(1)
    mle, unbiased = [], []
    for _ in range(1500):
        arrivals = np.cumsum(rng.exponential(size=8)) ** (1.0 / 0.7)
        horizon = float(arrivals[-1])
        s = sum(math.log(horizon / x) for x in arrivals if 0 < x < horizon)
        if s <= 0:
            continue
        mle.append(8 / s)
        unbiased.append(6 / s)

    assert float(np.mean(unbiased)) == pytest.approx(0.70, abs=0.03)
    assert float(np.mean(mle)) > 0.85, "the raw MLE should be visibly biased upward at n = 8"

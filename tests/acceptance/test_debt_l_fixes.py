"""Regressions for the series-L defects the eight-reviewer statistical review found.

Five defects, in three modules, and every one of them is pinned here with the number it changed.
The rule these follow is that a regression test is worth having only if it fails on the old code,
so each test says in its docstring what the old code returned and asserts against that value rather
than only against the new one.

    L2 rung 2   the two-rater bound claimed for either rater what holds only for their average,
                and the point estimate sat at the infimum of the identified set and was exported
                through the gate that licenses a scoring read.
    L5          the position-confound verdict could not be reached above chance, because the sign
                test ran before the confound test.
    L5          `uniform_auc` is computed on within-item standardised scores and was documented as
                the pooled statistic a localisation study publishes, which is a different number.
    D6          a Chao1 richness fraction was printed under Good-Turing's name, and Good-Turing had
                no interval.

The `declare_fixed` and Hill-slope findings are not here, because both live outside this package
and the fix belongs to whoever owns those files. They are recorded with their numbers instead.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reward_lens.measure.labels import (
    LocalisationSeries,
    independent_rater_rate,
    rescore_against_position,
    two_rater_bounds,
)
from reward_lens.measure.labels.error_rate import AuditSample, audit_error_rate
from reward_lens.measure.labels.position import CHANCE
from reward_lens.verifier.growth import (
    ExploitLog,
    chao1_interval,
    exploit_coverage,
    good_turing_interval,
    novelty_probability,
)

# ---------------------------------------------------------------------------
# L5: the position-confound verdict above chance
# ---------------------------------------------------------------------------


def _confounded_localiser(
    n_items: int = 400, *, slope: float = 3.0, bump: float = 0.45, seed: int = 0
) -> LocalisationSeries:
    """A localiser that is real and is mostly reading the position, above chance.

    The mirror image of the fixture in `test_w3_6_labels.py`: labelled positions sit late, the
    statistic rises with position, and a genuine bump sits on the labelled step. The pooled AUC is
    high mostly because two position distributions agree; a smaller real signal survives
    conditioning.
    """
    rng = np.random.default_rng(seed)
    values: list[float] = []
    offsets: list[int] = [0]
    labels: list[int] = []
    for _ in range(n_items):
        k = int(rng.integers(6, 10))
        position = np.arange(k) / (k - 1)
        err = int(rng.integers(max(1, k - 3), k))
        step = slope * position + rng.normal(size=k)
        step[err] += bump
        values.extend(step.tolist())
        offsets.append(len(values))
        labels.append(err)
    return LocalisationSeries(
        values=np.asarray(values),
        offsets=np.asarray(offsets),
        labels=np.asarray(labels),
        higher_is_positive=True,
    )


def test_a_confound_that_inflates_an_above_chance_reading_is_named() -> None:
    """The old rule returned "localises" here and said the discrimination was the localiser's own.

    Measured on this fixture: pooled 0.8291, position alone 0.8400, conditioned 0.6298 with a 95%
    cluster-bootstrap interval of [0.5949, 0.6685]. Five sixths of the pooled reading's distance
    from chance is gone once position is controlled, and the old verdict said none of it was
    position's, because `if stratified > CHANCE: return "localises"` ran before the confound test.
    """
    prior = rescore_against_position(_confounded_localiser(), n_boot=400, seed=0)

    assert prior.uniform_auc == pytest.approx(0.8291, abs=5e-4)
    assert prior.position_only_auc == pytest.approx(0.8400, abs=5e-4)
    assert prior.stratified_auc == pytest.approx(0.6298, abs=5e-4)
    assert (prior.ci_low, prior.ci_high) == pytest.approx((0.5949, 0.6685), abs=5e-4)

    # The three facts the old ordering made unreachable together.
    assert prior.stratified_auc > CHANCE
    assert not (prior.ci_low <= CHANCE <= prior.ci_high)
    assert abs(prior.stratified_auc - CHANCE) < abs(prior.uniform_auc - CHANCE)

    assert prior.verdict == "localises, and position inflated the pooled reading"
    assert prior.verdict != "localises", "the old answer"
    assert "0.6298 is the one to publish" in prior.interpretation


def test_a_clean_localiser_is_still_called_localises() -> None:
    """The fix must not turn every above-chance reading into a confound.

    Same construction with no position slope. The labels still sit late, so position alone scores
    0.8400 and the pooled reading is 0.0078 above the conditioned one, which has the *sign* of a
    confound. The bootstrap interval on that move is [-0.0178, +0.0043] and contains zero, so it is
    resampling noise and the plain verdict stands. This is the case a sign-only rule gets wrong,
    and it is why the move is judged against its own interval.
    """
    prior = rescore_against_position(_confounded_localiser(slope=0.0, bump=1.5), n_boot=400, seed=0)
    assert prior.stratified_auc > CHANCE
    assert abs(prior.stratified_auc - CHANCE) < abs(prior.uniform_auc - CHANCE), "the sign alone"
    assert prior.confound_size == pytest.approx(-0.0078, abs=5e-4)
    assert prior.confound_ci[0] < 0.0 < prior.confound_ci[1], "and the interval that overrules it"
    assert prior.verdict == "localises"


# ---------------------------------------------------------------------------
# L5: which scores the pooled AUC is computed on
# ---------------------------------------------------------------------------


def test_the_pooled_auc_says_which_scores_it_was_computed_on() -> None:
    """`uniform_auc` is on standardised scores; the number a paper publishes is on raw ones.

    Both are reported, and the difference is a measurement rather than an assertion. On this
    fixture the standardisation is the whole of the pooled reading: the raw scores carry a large
    between-item level difference, so pooling them measures that instead.
    """
    rng = np.random.default_rng(1)
    values: list[float] = []
    offsets: list[int] = [0]
    labels: list[int] = []
    for i in range(300):
        k = int(rng.integers(6, 10))
        level = 10.0 * rng.normal()  # a per-item level the pooled raw AUC cannot see past
        step = level + rng.normal(0.0, 0.3, k)
        err = int(rng.integers(0, k))
        step[err] += 1.5
        values.extend(step.tolist())
        offsets.append(len(values))
        labels.append(err)
    series = LocalisationSeries(
        values=np.asarray(values),
        offsets=np.asarray(offsets),
        labels=np.asarray(labels),
        higher_is_positive=True,
    )
    prior = rescore_against_position(series, n_boot=100, seed=0)

    assert prior.uniform_auc > 0.9, "within-item standardised, the localiser is obvious"
    assert abs(prior.uniform_auc_raw - CHANCE) < 0.1, "pooled raw, it is nearly invisible"
    assert prior.uniform_auc_raw != prior.uniform_auc
    assert "on raw scores, unstandardised" in prior.render()

    # And with the standardisation off, the two are the same statistic and say so.
    unstandardised = rescore_against_position(series, standardise=False, n_boot=100, seed=0)
    assert unstandardised.uniform_auc_raw == pytest.approx(unstandardised.uniform_auc)
    assert "on raw scores, unstandardised" not in unstandardised.render()


# ---------------------------------------------------------------------------
# L2 rung 2: what the two-rater design bounds, and what it exports
# ---------------------------------------------------------------------------


def test_the_two_rater_identity_holds_and_the_individual_claim_does_not() -> None:
    """`mean(e) = d/2 + P(both wrong)`, and `d/2` is not a floor under an individual rater.

    Simulated on 400,000 binary items. The identity closes to 1.4e-17 at equal rater rates and
    2.8e-17 at unequal ones. The claim the module used to make, that a single rater's rate is at
    least half the disagreement rate, is false in two of the three cases below: with rater A
    perfect and rater B at 0.20 the measured disagreement is 0.19998 and `d/2` asserts 0.09999
    against A's 0.00000.
    """
    rng = np.random.default_rng(20260805)
    n = 400_000
    truth = rng.integers(0, 2, n)

    def run(e_a: float, e_b: float) -> tuple[float, float, float, float]:
        a, b = truth.copy(), truth.copy()
        fa, fb = rng.random(n) < e_a, rng.random(n) < e_b
        a[fa] = 1 - a[fa]
        b[fb] = 1 - b[fb]
        d = float(np.mean(a != b))
        rate_a = float(np.mean(a != truth))
        rate_b = float(np.mean(b != truth))
        both = float(np.mean((a != truth) & (b != truth)))
        return d, rate_a, rate_b, both

    d, rate_a, rate_b, both = run(0.0, 0.20)
    assert abs(0.5 * (rate_a + rate_b) - (d / 2.0 + both)) < 1e-12
    assert rate_a == 0.0
    assert d / 2.0 > rate_a, "the false claim, refuted: A is below the supposed floor"

    d, rate_a, rate_b, both = run(0.10, 0.10)
    assert abs(0.5 * (rate_a + rate_b) - (d / 2.0 + both)) < 1e-12
    assert 0.5 * (rate_a + rate_b) >= d / 2.0 - 1e-12, "the true claim: it bounds the average"

    d, rate_a, rate_b, both = run(0.05, 0.25)
    assert abs(0.5 * (rate_a + rate_b) - (d / 2.0 + both)) < 1e-12
    assert d / 2.0 > rate_a


def test_the_rung_two_point_estimate_left_the_optimistic_end() -> None:
    """At `d = 0.35` the shipped point was 0.175, the infimum of its own identified set.

    It is now the independence model at 0.226139, and the ceiling moves with it from 0.8250 to
    0.7739. The lower bound survives as `low`, where it is labelled as a bound on the average of
    the two raters rather than on either of them.
    """
    result = two_rater_bounds([0] * 2000, [0] * 1300 + [1] * 700)
    d = result.error_rate.k / result.error_rate.n
    assert d == pytest.approx(0.35)

    assert result.error_rate.point == pytest.approx(0.2261387212474169)
    assert result.error_rate.point == pytest.approx(independent_rater_rate(0.35))
    assert result.error_rate.point != pytest.approx(0.175), "the old answer, d/2"
    assert result.ceiling == pytest.approx(0.7738612787525831)
    assert d / 2.0 <= result.error_rate.point <= d


def test_a_rung_that_only_bounds_the_rate_does_not_license_a_scoring_read() -> None:
    """`as_label_quality().error_rate` gates `record.labels.adjudicate` under `SCORING`.

    Rung 2 exported 0.175 into that gate on a design that identifies nothing tighter than
    `mean(e) >= 0.175`. It now exports None and the method string says why, so the scoring path
    refuses with `LABEL_QUALITY_UNKNOWN` instead of proceeding on the most flattering value in the
    identified set. Rung 0, which does identify the rate, is unaffected.
    """
    two_rater = two_rater_bounds([0] * 2000, [0] * 1300 + [1] * 700).as_label_quality()
    assert two_rater.error_rate is None
    assert not two_rater.is_measured
    assert two_rater.n_audited == 2000
    assert "bounds the error rate rather than identifying it" in two_rater.method

    audited = audit_error_rate(AuditSample(n_audited=200, n_wrong=8)).as_label_quality()
    assert audited.is_measured
    assert audited.error_rate == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# D6: two statistics under one name, and the intervals neither had
# ---------------------------------------------------------------------------


def _log_with(f1: int, f2: int, s_obs: int, n: int) -> ExploitLog:
    """A log with a chosen frequency spectrum, so the arithmetic is checkable by hand."""
    families: list[str] = [f"single{i}" for i in range(f1)]
    for i in range(f2):
        families += [f"double{i}"] * 2
    remaining = n - f1 - 2 * f2
    others = s_obs - f1 - f2
    for i in range(others):
        k = remaining // others + (1 if i < remaining % others else 0)
        families += [f"many{i}"] * k
    assert len(families) == n
    return ExploitLog.of(
        [(fam, float(j + 1)) for j, fam in enumerate(families)],
        total_effort=float(n + 1),
        source="hand-built spectrum",
    )


def test_the_chao1_fraction_is_not_printed_as_a_good_turing_mass() -> None:
    """They differ by a factor of 3.7 on this log, and the render used to name the wrong one.

    `f1 = 7` singletons in `n = 100` finds gives Good-Turing's unseen mass as 0.070, and
    `f1**2/(2*f2) = 8.167` unseen families against 23 seen gives a Chao1 richness fraction of
    0.262. The reading printed "Good-Turing bounds the unseen mass at 26.2%".
    """
    log = _log_with(f1=7, f2=3, s_obs=23, n=100)
    reading = exploit_coverage(log, rung=2)
    assert reading.f1 == 7 and reading.f2 == 3 and reading.n_families == 23

    assert novelty_probability(7, 100) == pytest.approx(0.070)
    assert reading.unseen_families == pytest.approx(49.0 / 6.0)
    assert reading.unseen_fraction == pytest.approx(0.2620320855614973)
    assert reading.unseen_fraction / reading.novelty_probability == pytest.approx(3.743, abs=1e-3)

    rendered = reading.render()
    assert "Chao1 puts 26.2% of the families still unseen" in rendered
    assert "Good-Turing puts the probability that the next find is novel at 0.070" in rendered
    assert "Good-Turing bounds the unseen mass" not in rendered


def test_good_turing_carries_estys_interval_and_it_is_most_of_the_range() -> None:
    """[0.0007, 0.1393] around 0.070, hand-checkable from Esty's variance.

    `Var = (f1 + 2*f2 - f1**2/n) / n**2 = (7 + 6 - 0.49) / 10000 = 1.251e-4 * 10`, so the standard
    error is 0.035369 and the 95% half-width is 0.069323. A point estimate of 0.070 whose interval
    reaches 0.139 does not distinguish "the search is nearly exhausted" from "one find in seven is
    still novel", and the reading used to carry no interval at all.
    """
    lo, hi = good_turing_interval(7, 3, 100)
    assert (lo, hi) == pytest.approx((0.0007, 0.1393), abs=5e-5)

    var = (7 + 2 * 3 - 7 * 7 / 100) / 100**2
    assert math.sqrt(var) == pytest.approx(0.0353694787, abs=1e-9)
    assert hi - lo == pytest.approx(2 * 1.959963984540054 * math.sqrt(var), rel=1e-9)

    reading = exploit_coverage(_log_with(f1=7, f2=3, s_obs=23, n=100), rung=2)
    assert reading.novelty_ci == pytest.approx((lo, hi))
    assert "[0.001, 0.139]" in reading.render()


def test_the_chao1_fraction_carries_an_interval_and_it_is_a_floors_precision() -> None:
    """Chao (1987) with the log-normal transform, mapped onto the fraction.

    The interval on the unseen count is `[f0/K, f0*K]`, which is asymmetric and stays positive.
    Mapped through `f0 / (S_obs + f0)`, which is increasing in `f0`, the endpoints map in order.
    What it does not cover is Chao1's own downward bias: simulated on communities with
    gamma-distributed abundances it contains the true unseen count 0.847 of the time at moderate
    unevenness and 0.611 at high unevenness, and the misses are almost all the truth sitting above
    the interval, which is why the reading calls the fraction a floor.
    """
    f0_lo, f0_hi = chao1_interval(7, 3)
    f0 = 49.0 / 6.0
    assert f0_lo < f0 < f0_hi
    assert f0_lo > 0.0
    assert f0_hi - f0 > f0 - f0_lo, "log-normal, so the upper arm is the longer one"

    reading = exploit_coverage(_log_with(f1=7, f2=3, s_obs=23, n=100), rung=2)
    assert reading.unseen_families_ci == pytest.approx((f0_lo, f0_hi))
    lo, hi = reading.unseen_fraction_ci
    assert lo == pytest.approx(f0_lo / (23 + f0_lo))
    assert hi == pytest.approx(f0_hi / (23 + f0_hi))
    assert lo < reading.unseen_fraction < hi
    assert "floor rather than an estimate" in reading.render()


def test_the_intervals_reach_the_evidence_row() -> None:
    """The headline quantity is the Chao1 fraction, so its interval is the one on `Uncertainty`."""
    from reward_lens.verifier.growth import ExploitFamilyCoverage

    log = _log_with(f1=7, f2=3, s_obs=23, n=100)
    reading = ExploitFamilyCoverage(log).estimate()
    assert reading.uncertainty is not None
    assert reading.uncertainty.ci_low == pytest.approx(reading.value.unseen_fraction_ci[0])
    assert reading.uncertainty.ci_high == pytest.approx(reading.value.unseen_fraction_ci[1])
    assert "Chao1" in reading.uncertainty.method
    assert "no interval" not in reading.uncertainty.method

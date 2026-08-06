"""Acceptance: the compute-gated rate package, proved on planted subjects.

**Nothing here runs on a real subject and that is the whole point of the package.** Phase 6 is
compute-gated: the module, the frozen study, this test, the runbook and the price exist so that a
maintainer can decide with a number in front of them whether to buy the arms. So what this file can
establish is exactly one thing, and it says so rather than implying more: **the arithmetic is
right.** Each instrument is pointed at a planted system whose answer is known before the fit runs,
and the assertion is that it recovers the planted answer and refuses on the planted failure.

It proves nothing about a language policy. The subject that carries a real reward-hacking transition
in this project is the AISI series, 25,664 rollouts over 401 steps with a fitted midpoint at step
106.0 and a width of 23.9 steps, and it is a published rollout table rather than a loop: it cannot
be re-annealed at a second rate, it cannot be perturbed and held, and it cannot be swept backwards.
Every claim these instruments make needs arms that do not exist, and `studies/w6_rate/RUNBOOK.md`
says how to make them and what they cost.

The planted subjects, and why each is the right one:

**A first-order tracker crossing a fixed critical point**, for H2. Its relaxation time is the knob
and its critical point does not move, so any separation between two anneal rates is lag and nothing
else. That is the only construction in which "the curves separated" has an unambiguous cause.

**A single exponential recovery**, for H1 rung 1, at hold lengths above and below the floor.

**A tilted double well against a single well**, for H3, and this pair is the load-bearing one. The
single well has no second stable state at all and still traces a loop of area 0.0514 at the fastest
of four sweep rates. If the rate extrapolation did not exist, that number is what would be published
as hysteresis.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import kendalltau

from reward_lens.core.quantity import ladder
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.loops.anneal import double_well_responder
from reward_lens.measure.base import lint_instrument
from reward_lens.measure.rate.collapse import (
    CollapseCriteria,
    RateArm,
    RateDependenceTest,
    two_run_rate_test,
)
from reward_lens.measure.rate.hysteresis import (
    HysteresisCriteria,
    RateExtrapolatedHysteresis,
    SweepArea,
    rate_extrapolated_area,
    sweep_areas,
)
from reward_lens.measure.rate.perturb import (
    PERTURB_AND_HOLD_IMPL,
    PerturbAndHold,
    PerturbCriteria,
    Recovery,
    _kendall_p_floor,
    relaxation_time_from_hold,
    rung_transfer,
)
from reward_lens.measure.rate.transition import TransitionCriteria
from reward_lens.measure.rate.warning import (
    WarningCriteria,
    driver_comparison,
    early_warning,
    flickering,
    trend_significance,
)

#: Cheaper resamples than the shipped defaults, so the suite stays quick. Every verdict asserted
#: below is checked to be stable at these settings; the shipped defaults are wider, not different.
FAST_BAND = CollapseCriteria(n_boot=80, n_grid=120)
FAST_FIT = TransitionCriteria(n_boot=30)
FAST_WARN = WarningCriteria(n_surrogates=120, n_flicker_boot=60)


# ---------------------------------------------------------------------------
# The planted subjects
# ---------------------------------------------------------------------------


def tracker(
    label: str,
    n_steps: int,
    *,
    tau: float,
    seed: int,
    noise: float = 0.03,
    critical: float = 0.5,
) -> RateArm:
    """A first-order tracker crossing a fixed critical point on an exponential lambda schedule.

    The equilibrium is a logistic in `lambda` whose midpoint is `critical` whatever `tau` and
    whatever the rate. The order parameter relaxes toward it with time constant `tau`, so a large
    `tau` makes the observed transition lag the equilibrium one, and the lag is larger on the arm
    whose `lambda` is moving faster. That is rate-induced tipping, planted.
    """
    lam = np.exp(np.linspace(math.log(0.05), math.log(1.0), n_steps))
    rng = np.random.default_rng(seed)
    m = np.zeros(n_steps)
    for i in range(1, n_steps):
        eq = 1.0 / (1.0 + np.exp(-(lam[i] - critical) / 0.05))
        m[i] = m[i - 1] + (eq - m[i - 1]) / tau + rng.normal(0.0, noise)
    return RateArm.from_series(
        label, lam, m, np.arange(n_steps, dtype=float), series="order_parameter"
    )


def recovery(
    label: str, *, tau: float, magnitude: float, hold: int, seed: int, noise: float = 0.01
) -> Recovery:
    """One planted perturb-and-hold episode: a single exponential return to a known level."""
    rng = np.random.default_rng(seed)
    pre = 0.5 + rng.normal(0.0, noise, 20)
    s = np.arange(hold, dtype=float)
    return Recovery.from_series(
        label, 0.5 + magnitude * np.exp(-s / tau) + rng.normal(0.0, noise, hold), pre=pre
    )


def single_well(beta: float, m: float, *, n_iter: int = 60, lr: float = 0.02) -> float:
    """One minimum, at `beta / 2`, and no bistability anywhere. The lag-only control for H3."""
    x = float(m)
    for _ in range(n_iter):
        x -= lr * (2.0 * x - beta)
    return x


# ---------------------------------------------------------------------------
# Rule 3: lint everything this package ships. E56.
# ---------------------------------------------------------------------------


def test_every_instrument_this_package_ships_passes_lint():
    """E56: four instruments shipped failing lint while their package read `done`.

    They failed because the acceptance test rendered readings and never linted. So this runs first
    in the file and it covers all three, constructed with empty inputs on purpose: lint reads the
    declaration and never the data, and an instrument that needs a subject before it can be linted
    is an instrument whose declaration depends on its input.
    """
    instruments = (
        RateDependenceTest(tracker("a", 20, tau=2.0, seed=0), tracker("b", 20, tau=2.0, seed=0)),
        PerturbAndHold(()),
        RateExtrapolatedHysteresis(()),
    )
    findings = {i.name: [f.render() for f in lint_instrument(i)] for i in instruments}
    assert all(not v for v in findings.values()), findings


def test_the_quantities_this_package_claims_are_registered_with_their_rungs():
    """A quantity with no estimator reads as an open research target; these are no longer open."""
    rate = {e.rung: e.impl for e in ladder("run.rate_dependence")}
    assert rate == {0: "run.rate_dependence.two_arm_lambda_collapse"}

    hyst = {e.rung: e.impl for e in ladder("run.hysteresis_area")}
    assert hyst == {
        0: "run.hysteresis_area.raw_single_rate",
        1: "run.hysteresis_area.rate_extrapolated",
    }

    # H1 rung 1 was registered elsewhere with `run=None` and this package implements it. The entry
    # is in a file this package does not own, so it is left as registered and the wiring is a
    # request; this assertion is what will fail the day it is wired, which is the point.
    tau = {e.rung: e for e in ladder("run.tau_relax")}
    assert tau[1].impl == PERTURB_AND_HOLD_IMPL
    assert tau[1].run is None, (
        "run.tau_relax rung 1 now carries a callable. Point it at "
        "`perturb.relaxation_time_from_hold` and delete this assertion."
    )


# ---------------------------------------------------------------------------
# H2, the two-run rate test
# ---------------------------------------------------------------------------


def test_a_quasi_static_tracker_collapses_and_the_step_axis_does_not():
    """The licensing case. Both arms track their equilibrium, so against lambda they are one curve.

    Against `t` the same two arms separate on more than a third of the shared support, because the
    fast arm reaches every lambda at an earlier step. That number is the baseline and it is what
    makes the collapse mean something: the test had every opportunity to separate and did not.
    """
    slow, fast = tracker("slow", 400, tau=3.0, seed=7), tracker("fast", 100, tau=3.0, seed=7)
    out = two_run_rate_test(fast, slow, criteria=FAST_BAND, fit_criteria=FAST_FIT, seed=1)
    assert not isinstance(out, Refusal), getattr(out, "render", lambda: out)()

    assert out.rate_ratio == pytest.approx(4.0, abs=0.1)
    assert out.separated_fraction == 0.0
    assert out.collapses and not out.rate_induced
    assert out.against_t_separated_fraction > 0.3, (
        "the against-t baseline has to separate, or the collapse against lambda is not evidence "
        "that the reparametrisation did anything"
    )
    assert "bifurcation-induced" in out.says()


@pytest.mark.parametrize("tau", [10.0, 20.0, 40.0])
def test_a_lagging_tracker_separates_and_the_faster_arm_transitions_later_in_lambda(tau):
    """The killing case, at three relaxation times. The sign is asserted, not just the magnitude.

    Rate-induced tipping displaces the transition **later** in lambda on the faster arm, because the
    system has not finished responding when the driver arrives at the critical point. A separation
    in the other direction would be a real effect and would not be lag, so the sign is part of the
    claim.
    """
    slow, fast = tracker("slow", 400, tau=tau, seed=7), tracker("fast", 100, tau=tau, seed=7)
    out = two_run_rate_test(fast, slow, criteria=FAST_BAND, fit_criteria=FAST_FIT, seed=1)
    assert not isinstance(out, Refusal), getattr(out, "render", lambda: out)()

    assert out.rate_induced and not out.collapses
    assert out.separated_fraction >= out.criteria.min_separated_fraction
    assert out.shift_in_widths > 0.0, "the faster arm should transition later in lambda"
    assert out.fast.identified and out.slow.identified
    assert "rate-induced" in out.says()
    assert "which is the lag a rate-induced transition produces" in out.render()


def test_the_shift_is_reported_in_h4_transition_widths_and_not_in_a_second_unit():
    """H4's own argument, applied to this instrument.

    Lead time is defined as a fraction of the fitted transition width because four incommensurable
    conventions were in circulation. A displacement between two arms is the same kind of quantity,
    so it is divided by the same denominator, and the denominator comes from H4's fit rather than
    from a width this module computes for itself.
    """
    slow, fast = tracker("slow", 400, tau=20.0, seed=11), tracker("fast", 100, tau=20.0, seed=11)
    out = two_run_rate_test(fast, slow, criteria=FAST_BAND, fit_criteria=FAST_FIT, seed=1)
    assert not isinstance(out, Refusal)

    pooled = 0.5 * (out.fast.width + out.slow.width)
    assert out.shift_in_lambda == pytest.approx(out.fast.midpoint - out.slow.midpoint)
    assert out.shift_in_widths == pytest.approx(out.shift_in_lambda / pooled)


def test_two_arms_at_nearly_one_rate_refuse_rather_than_collapsing_by_construction():
    """The refusal most likely to fire on a real pair of arms, and the one that matters most.

    Arms sized to the same wall-clock budget end up at similar rates, they collapse whatever the
    physics, and a collapse that cannot fail would license the whole early-warning layer for free.
    """
    slow, fast = tracker("slow", 400, tau=3.0, seed=3), tracker("fast", 320, tau=3.0, seed=3)
    out = two_run_rate_test(fast, slow, criteria=FAST_BAND, fit_criteria=FAST_FIT)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ENVELOPE_VIOLATED
    assert out.statistics["ratio"] < 2.0
    assert "re-run one arm at a rate at least" in out.remedy
    assert "licenses nothing" in out.remedy


def test_arms_that_swept_different_lambda_ranges_refuse_rather_than_comparing_nothing():
    """Curves cannot be compared where only one of them was measured."""
    a = tracker("a", 200, tau=3.0, seed=5)
    b = tracker("b", 50, tau=3.0, seed=5)
    shifted = RateArm.from_series("b-shifted", b.lambdas * 40.0, b.observable, b.steps)
    out = two_run_rate_test(shifted, a, criteria=FAST_BAND, fit_criteria=FAST_FIT)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "anneal both arms over the same lambda interval" in out.remedy


def test_a_constant_schedule_is_not_an_arm_of_an_anneal():
    flat = RateArm.from_series(
        "flat", np.full(40, 0.5), np.linspace(0, 1, 40), np.arange(40, dtype=float)
    )
    out = two_run_rate_test(flat, tracker("slow", 200, tau=3.0, seed=1), criteria=FAST_BAND)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "has no rate" in out.remedy


def test_the_baselines_are_scored_in_the_readings_own_unit():
    """A claim without a baseline is sent back, and both of H2's are real numbers on every run."""
    slow, fast = tracker("slow", 400, tau=20.0, seed=7), tracker("fast", 100, tau=20.0, seed=7)
    instrument = RateDependenceTest(fast, slow, criteria=FAST_BAND, fit_criteria=FAST_FIT, seed=1)
    out = instrument.compute()
    assert not isinstance(out, Refusal)
    scores = instrument.baseline_scores(out)
    assert scores["baseline.single_rate_run"] == 0.0
    assert scores["baseline.against_t"] == out.against_t_separated_fraction
    assert scores["baseline.against_t"] > 0.0


# ---------------------------------------------------------------------------
# H1 rung 1, perturb-and-hold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("planted", [4.0, 8.0, 20.0])
def test_a_planted_relaxation_time_is_recovered_from_held_recoveries(planted):
    """Five episodes at magnitudes spanning a factor of five, each held past the floor.

    Held for twelve planted time constants so the fit is in the regime the truncation table says is
    unbiased. The tolerance is ten percent and it comes from a measurement rather than from
    loosening until green: over 200 replicate protocols of exactly this design the pooled geometric
    mean lands at -0.50, -0.21 and -0.27 percent of the planted value at 4, 8 and 20 steps, so the
    estimator is unbiased, and its protocol-to-protocol spread is 4.4, 3.3 and 2.4 percent with a
    5th-to-95th band of [3.72, 4.25] at a planted 4. A five-percent tolerance is tighter than the
    estimator's own 90 percent interval at five episodes and would flake.
    """
    hold = int(12 * planted)
    episodes = [
        recovery(f"m{i}", tau=planted, magnitude=m, hold=hold, seed=100 + i)
        for i, m in enumerate((0.08, 0.12, 0.18, 0.27, 0.40))
    ]
    out = relaxation_time_from_hold(episodes, seed=1)
    assert not isinstance(out, Refusal), getattr(out, "render", lambda: out)()

    assert out.tau == pytest.approx(planted, rel=0.10)
    assert out.tau_low < out.tau < out.tau_high
    assert out.n_quantitative == 5
    assert out.identified
    assert all(f.returned_to_level and f.hold_adequate for f in out.episodes)


def test_the_interval_under_covers_at_five_episodes_and_the_module_says_by_how_much():
    """Coverage is a property of the estimator over protocols, not of one interval.

    Asserting that a single 95 percent interval contains the planted value is a test that fails one
    time in twenty by design, and loosening it until it stops would be hiding the estimator's real
    behaviour. So it is measured: 60 protocols of five episodes each at a planted 8 steps. Over 100
    protocols the coverage is 0.84 at five episodes, 0.90 at eight and 0.93 at twelve, and the
    module docstring carries those numbers with the direction they err in. The floor here is 0.75,
    which is two Monte Carlo standard errors below the measured 0.84 at this replicate count.
    """
    planted, n_protocols = 8.0, 60
    criteria = PerturbCriteria(n_boot=400)
    mags = np.geomspace(0.08, 0.40, 5)
    covered = 0
    for p in range(n_protocols):
        episodes = [
            recovery(f"m{i}", tau=planted, magnitude=m, hold=96, seed=9000 + p * 50 + i)
            for i, m in enumerate(mags)
        ]
        out = relaxation_time_from_hold(episodes, criteria=criteria, seed=p)
        assert not isinstance(out, Refusal)
        covered += int(out.tau_low <= planted <= out.tau_high)
    coverage = covered / n_protocols
    assert 0.75 <= coverage < 0.95, (
        f"nominal 95 percent interval covered {coverage:.2f} of {n_protocols} protocols. The "
        f"module docstring records 0.84 at five episodes; a value at or above 0.95 would mean the "
        f"under-coverage has been fixed and the docstring is now wrong."
    )


def test_the_linear_response_check_cannot_fire_at_three_episodes_and_says_so():
    """The defect this module had before it was run on a linear system.

    Kendall's tau over three pairs takes only four values, a perfectly monotone ordering arises by
    chance one time in three, and a threshold on the statistic declared a third of linear systems
    nonlinear. The check is now read on its p-value and the reading states that at three episodes
    the smallest available p-value is 0.333.
    """
    for n in (3, 4, 5, 6):
        x = np.arange(n, dtype=float)
        assert _kendall_p_floor(n) == pytest.approx(float(kendalltau(x, x).pvalue))

    three = [
        recovery(f"m{i}", tau=8.0, magnitude=m, hold=96, seed=200 + i)
        for i, m in enumerate((0.10, 0.20, 0.40))
    ]
    out = relaxation_time_from_hold(three, seed=1)
    assert not isinstance(out, Refusal)
    assert not out.magnitude_test_powered
    assert out.linear_response, "the check must not fire when it could not have fired"
    assert out.magnitude_p >= 1.0 / 3.0
    assert "could not have fired" in out.render()

    five = [
        recovery(f"m{i}", tau=8.0, magnitude=m, hold=96, seed=300 + i)
        for i, m in enumerate((0.08, 0.12, 0.18, 0.27, 0.40))
    ]
    powered = relaxation_time_from_hold(five, seed=1)
    assert not isinstance(powered, Refusal)
    assert powered.magnitude_test_powered
    assert "so the response is linear over the range probed" in powered.render()


def test_a_truncated_hold_returns_a_fit_free_lower_bound_rather_than_a_plausible_number():
    """The dangerous failure, and the reason the bound is not the fit.

    Measured on planted exponentials at a planted time constant of 20 steps, 300 replicates per
    cell: below three time constants of hold the median fitted value runs down to 3.99 at an
    amplitude-to-noise ratio of 3 while the geometric mean runs up, so the fit is unstable rather
    than biased in a knowable direction and is not a bound in either. What is established without a
    fit is that the observable had not come back within the hold, so the return time exceeds the
    hold divided by three.
    """
    episodes = [
        recovery(f"s{i}", tau=60.0, magnitude=m, hold=25, seed=400 + i)
        for i, m in enumerate((0.10, 0.20, 0.40))
    ]
    out = relaxation_time_from_hold(episodes, seed=1)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ABOVE_LOD_BELOW_LOQ
    assert out.is_bounded

    bound = out.partial.value
    assert bound == pytest.approx(25.0 / 3.0)
    assert bound < 60.0, "it has to be a bound the planted value actually respects"
    assert out.statistics["longest_hold"] == 25
    assert "end the hold on the observable returning" in out.remedy


def test_a_hold_that_did_not_hold_refuses_with_the_schedule_named():
    """A recovery measured while lambda anneals is the relaxation time and the drift added up."""
    episodes = [
        Recovery(
            label=f"d{i}",
            values=np.asarray(recovery(f"d{i}", tau=8.0, magnitude=0.2, hold=96, seed=i).values),
            pre_level=0.5,
            pre_sd=0.01,
            magnitude=0.2,
            held_lambda_span=0.15,
        )
        for i in range(3)
    ]
    out = relaxation_time_from_hold(episodes, seed=1)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ENVELOPE_VIOLATED
    assert out.statistics["worst_span"] == pytest.approx(0.15)
    assert "freeze the schedule for the whole hold" in out.remedy


def test_a_nudge_inside_the_noise_refuses_rather_than_fitting_the_noises_decay():
    episodes = [
        recovery(f"t{i}", tau=8.0, magnitude=0.005, hold=96, seed=500 + i, noise=0.05)
        for i in range(3)
    ]
    out = relaxation_time_from_hold(episodes, seed=1)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.BELOW_LOD
    assert "increase the nudge" in out.remedy


def test_the_two_rungs_of_tau_relax_disagree_and_it_is_published_as_a_transfer():
    """One quantity, two rungs, one run: M11's case, and the reason the ladder exists."""
    episodes = [
        recovery(f"m{i}", tau=8.0, magnitude=m, hold=96, seed=600 + i)
        for i, m in enumerate((0.08, 0.12, 0.18, 0.27, 0.40))
    ]
    rung1 = relaxation_time_from_hold(episodes, seed=1)
    assert not isinstance(rung1, Refusal)

    transfer = rung_transfer(3.0, rung1)
    assert transfer.value == pytest.approx(abs(3.0 - rung1.tau))
    assert transfer.name == "t21"
    assert "against the perturb-and-hold protocol" in transfer.method
    assert transfer.as_term().name == "t21"


def test_too_few_episodes_refuse_because_three_is_where_the_linearity_check_starts():
    out = relaxation_time_from_hold(
        [recovery("only", tau=8.0, magnitude=0.2, hold=96, seed=1)], seed=1
    )
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "three perturbation magnitudes" in out.remedy


# ---------------------------------------------------------------------------
# H3, rate-extrapolated hysteresis
# ---------------------------------------------------------------------------


def test_a_single_well_with_no_bistability_traces_a_loop_and_the_extrapolation_kills_it():
    """The load-bearing test in this file.

    One minimum, at `beta / 2`, and no second stable state anywhere. Swept up and back at four
    rates it still encloses a loop, and the raw area at the fastest rate is what a single-rate
    protocol would publish as hysteresis. The extrapolated area's interval contains zero, so the
    instrument reports what is true: all of it was lag.
    """
    sweeps = sweep_areas(
        single_well, lam0=0.0, lam1=3.0, n_points=(20, 40, 80, 160), seeds=(0, 1, 2), perturb=0.02
    )
    out = rate_extrapolated_area(sweeps)
    assert not isinstance(out, Refusal), getattr(out, "render", lambda: out)()

    assert out.raw_area_fastest > 0.04, "the confounded rung-0 number has to be visibly nonzero"
    assert not out.genuine
    assert out.ci[0] <= 0.0 <= out.ci[1]
    assert abs(out.area_zero) < 0.01 * out.raw_area_fastest
    assert "consistent with lag" in out.says()
    assert "no evidence here of a second stable state" in out.says()


def test_a_tilted_double_well_survives_the_extrapolation():
    """The other half of the pair. Two stable states, and the loop is still there at zero rate."""
    sweeps = sweep_areas(
        double_well_responder(n_iter=60, lr=0.02),
        lam0=0.0,
        lam1=3.0,
        n_points=(20, 40, 80, 160),
        seeds=(0, 1, 2),
        perturb=0.02,
    )
    out = rate_extrapolated_area(sweeps)
    assert not isinstance(out, Refusal), getattr(out, "render", lambda: out)()

    assert out.genuine
    assert out.ci[0] > 0.0
    assert out.area_zero < out.raw_area_slowest < out.raw_area_fastest, (
        "the raw area has to fall with the rate, or there is no lag to remove"
    )
    assert "genuine rather than lag" in out.says()


def test_the_interval_widens_when_the_seed_spread_does_not_explain_the_scatter():
    """The correction that stopped a fit artifact being called genuine.

    A deterministic responder's seeds agree to nine decimal places while the residuals about the
    rate model sit orders of magnitude above that. The textbook weighted interval is then a width
    near zero around whatever the fit's intercept happened to be, which is how the single-well
    responder was first reported as having genuine hysteresis at an area of 0.00015.
    """
    sweeps = sweep_areas(
        single_well, lam0=0.0, lam1=3.0, n_points=(20, 40, 80, 160), seeds=(0, 1, 2), perturb=0.02
    )
    out = rate_extrapolated_area(sweeps)
    assert not isinstance(out, Refusal)
    assert out.weighted
    assert out.birge > 1.0
    assert "widened" in out.render()

    unwidened_half_width = (out.ci[1] - out.ci[0]) / (2.0 * out.birge)
    assert out.area_zero - unwidened_half_width > 0.0, (
        "without the widening this interval excludes zero, which is the failure being guarded"
    )


def test_fewer_than_three_rates_refuse_because_two_points_fit_two_parameters_exactly():
    out = rate_extrapolated_area(
        [SweepArea.from_seeds(0.1, [0.5, 0.51]), SweepArea.from_seeds(0.05, [0.4, 0.41])]
    )
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "Two rates fit two parameters exactly" in out.remedy


def test_rates_within_a_factor_of_two_refuse_because_the_lever_is_too_short():
    sweeps = [SweepArea.from_seeds(v, [0.3 + 2.0 * v, 0.31 + 2.0 * v]) for v in (0.10, 0.13, 0.16)]
    out = rate_extrapolated_area(sweeps)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ENVELOPE_VIOLATED
    assert out.statistics["rate_span"] < 4.0
    assert "times slower than the fastest one" in out.remedy


def test_one_sweep_per_rate_at_three_rates_leaves_one_degree_of_freedom_and_a_wide_interval():
    """The design floor, and what it actually buys.

    Three rates against two parameters is one residual degree of freedom, not zero, so the fit
    returns rather than refusing. What it returns is an interval whose t-multiplier is 12.7, and
    that is the honest cost of a design with no seeds: the answer exists and is barely constrained.
    The runbook asks for seeds before it asks for rates because of this number.
    """
    out = rate_extrapolated_area([SweepArea(rate=v, area=0.2 + 2.0 * v) for v in (0.02, 0.06, 0.2)])
    assert not isinstance(out, Refusal), getattr(out, "render", lambda: out)()
    assert out.dof == 1
    assert not out.weighted
    assert math.isnan(out.birge)
    assert out.area_zero == pytest.approx(0.2)
    assert "no rate was run at more than one seed" in out.render()


def test_a_fit_with_no_degrees_of_freedom_returns_a_bound_because_the_kill_needs_an_interval():
    """Two rates and two parameters. Reachable only by lowering `min_rates`, and it is guarded."""
    out = rate_extrapolated_area(
        [SweepArea(rate=v, area=0.2 + 2.0 * v) for v in (0.02, 0.2)],
        criteria=HysteresisCriteria(min_rates=2),
    )
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ABOVE_LOD_BELOW_LOQ
    assert out.is_bounded
    assert out.statistics["dof"] == 0
    assert "run each rate at three or more seeds" in out.remedy


def test_the_baselines_are_the_raw_single_rate_areas_the_literature_reports():
    sweeps = sweep_areas(
        single_well, lam0=0.0, lam1=3.0, n_points=(20, 40, 80, 160), seeds=(0, 1, 2), perturb=0.02
    )
    instrument = RateExtrapolatedHysteresis(sweeps)
    out = instrument.compute()
    assert not isinstance(out, Refusal)
    scores = instrument.baseline_scores(out)
    assert scores["baseline.raw_area_fastest_rate"] == out.raw_area_fastest
    assert scores["baseline.raw_area_slowest_rate"] == out.raw_area_slowest
    assert scores["baseline.raw_area_fastest_rate"] > scores["baseline.raw_area_slowest_rate"]


# ---------------------------------------------------------------------------
# The early-warning statistics, and the nulls that make them evidence
# ---------------------------------------------------------------------------


def _rising_ar1(n: int, seed: int, phi0: float = 0.2, phi1: float = 0.95) -> np.ndarray:
    """Slowing down, planted: an autoregressive series whose coefficient climbs toward one."""
    rng = np.random.default_rng(seed)
    phi = np.linspace(phi0, phi1, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi[i] * x[i - 1] + rng.normal(0.0, 0.1)
    return x


def _stationary_ar1(n: int, seed: int, phi: float = 0.7) -> np.ndarray:
    """No slowing down anywhere, and a rolling autocorrelation that trends anyway."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0.0, 0.1)
    return x


def test_a_rising_autocorrelation_without_its_null_is_not_evidence():
    """The published claim, demonstrated on a series that has no slowing down in it.

    A stationary first-order series at a fixed coefficient of 0.7 produces a rolling
    autocorrelation whose Kendall tau against time is strongly positive. Reported on its own that
    reads as textbook critical slowing down. Against 120 phase-randomised surrogates it does not
    clear the null, because the null's own 95th percentile is nearly as large: overlapping windows
    make the rolling statistic heavily autocorrelated and a big tau is ordinary.
    """
    stationary = trend_significance(
        _stationary_ar1(300, seed=3), criteria=FAST_WARN, name="stationary", seed=1
    )
    assert not isinstance(stationary, Refusal), getattr(stationary, "render", lambda: stationary)()

    assert stationary.tau > 0.4, "the bare statistic looks like slowing down"
    assert not stationary.significant, "and the null says it is not"
    assert stationary.null_q95 > 0.5, (
        "the null's own upper tail has to be large, or this demonstration proves nothing about why "
        "the null is needed"
    )
    assert "does not beat its null" in stationary.render()

    planted = trend_significance(
        _rising_ar1(300, seed=21), criteria=FAST_WARN, name="planted", seed=1
    )
    assert not isinstance(planted, Refusal)
    assert planted.significant and planted.tau > stationary.tau
    assert "beats its null" in planted.render()


def test_both_surrogate_constructions_run_and_are_named_on_the_reading():
    """Two nulls that fail differently, and the reading says which one produced its p-value."""
    series = _rising_ar1(240, seed=5)
    for method in ("fourier", "ar1"):
        out = trend_significance(series, criteria=FAST_WARN, method=method, name="s", seed=2)
        assert not isinstance(out, Refusal)
        assert out.method == method
        assert 0.0 < out.p <= 1.0
        assert method in out.render()


def test_flickering_is_bimodality_with_a_null_and_it_says_where_that_null_is_wrong():
    """Two states are found, one state is not, and the documented failure is demonstrated.

    A Gamma(2, 1) sample is unimodal and skewed. A two-component Gaussian mixture fits it better
    than one component and the parametric bootstrap under a single Gaussian confirms that at p
    below 0.05, because the null is Gaussian and the data is not. The separation guard is what
    catches it, and the module docstring says so rather than leaving the caveat to be discovered.
    """
    rng = np.random.default_rng(9)
    two_states = np.concatenate([rng.normal(0.0, 0.3, 120), rng.normal(3.0, 0.3, 120)])
    rng.shuffle(two_states)
    out = flickering(two_states, criteria=FAST_WARN, name="two_states", seed=2)
    assert not isinstance(out, Refusal)
    assert out.flickering and out.separation > 2.0
    assert out.delta_bic > 0

    one_state = flickering(rng.normal(0.0, 1.0, 240), criteria=FAST_WARN, name="one", seed=2)
    assert not isinstance(one_state, Refusal)
    assert not one_state.flickering
    assert "not detectably bimodal" in one_state.render()

    skewed = flickering(rng.gamma(2.0, 1.0, 240), criteria=FAST_WARN, name="gamma", seed=2)
    assert not isinstance(skewed, Refusal)
    assert skewed.p <= 0.05, "the Gaussian null is beaten by skew, which is the documented failure"
    assert skewed.separation < 2.0
    assert not skewed.flickering, "and the separation guard is what stops it being called flicker"
    assert "a better likelihood rather than a second state" in skewed.render()


def test_the_driver_check_is_the_adiabaticity_number_reached_from_the_other_side():
    """This is one of the warning checks and it is H1's quantity, not a new one."""
    series = _stationary_ar1(300, seed=11, phi=0.8)
    slow = driver_comparison(series, 0.02, name="obs", seed=1)
    fast = driver_comparison(series, 0.9, name="obs", seed=1)
    assert not isinstance(slow, Refusal) and not isinstance(fast, Refusal)

    assert slow.tau_relax == pytest.approx(fast.tau_relax), "one relaxation time, two drivers"
    assert slow.ad == pytest.approx(slow.tau_relax * 0.02)
    assert slow.licensed and not fast.licensed
    assert "an early warning has room to be early" in slow.render()
    assert "measuring the transition rather than anticipating it" in fast.render()


def test_the_assembled_reading_needs_all_four_checks_and_an_absent_control_is_not_a_quiet_one():
    """`credible` is a conjunction, and the missing-control case is the one worth asserting.

    An early-warning reading with no noise control has nothing separating a slowing-down signal
    from something that moved every channel at once, so it is not credible. Treating an absent
    check as a passed check is the failure the study runner's void machinery exists to prevent one
    layer up, and it is the same failure here.
    """
    series = _rising_ar1(240, seed=31)
    without = early_warning(series, criteria=FAST_WARN, name="obs", seed=1)
    assert not isinstance(without, Refusal), getattr(without, "render", lambda: without)()
    assert not without.control_quiet and not without.credible
    assert "no noise control was supplied" in without.render()

    with_control = early_warning(
        series,
        control=_stationary_ar1(240, seed=32, phi=0.3),
        drive_rate=0.005,
        criteria=FAST_WARN,
        name="obs",
        control_name="unrelated",
        seed=1,
    )
    assert not isinstance(with_control, Refusal)
    assert with_control.control is not None and with_control.control_quiet
    assert with_control.driver is not None and with_control.driver.licensed
    assert with_control.sensitivity.n_ok >= 3

    # The two caveats travel with every reading rather than living in a page nobody opens.
    text = with_control.render()
    assert "Systems with no critical transition in them show these signals" in text
    assert "a warning arrives too late when the driver moves faster than the system relaxes" in text


def test_a_series_too_short_for_a_rolling_window_refuses_with_the_floor_named():
    out = trend_significance(np.arange(12, dtype=float), criteria=FAST_WARN, name="short")
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert out.statistics["floor"] == FAST_WARN.min_points


# ---------------------------------------------------------------------------
# The frozen studies, the price, and what a compute-gated study returns
# ---------------------------------------------------------------------------


def test_both_studies_freeze_before_any_arm_exists_and_their_ids_are_content_hashes():
    """Gate 3. A prediction hashed before the run is a prediction; after it, a description."""
    from studies.w6_rate import freeze_all

    frozen = freeze_all()
    assert set(frozen) == {"w6-1-two-run-rate-test", "w6-2-relaxation-and-hysteresis"}
    for study in frozen.values():
        assert study.study_id.startswith("study:")
        assert study.spec_hash and study.git_sha and study.frozen_at
        assert study.spec.hypotheses and study.spec.kill_criteria
        for h in study.spec.hypotheses:
            assert h.prediction.rationale, "every prediction names the baseline it must beat"

    again = freeze_all()
    assert {k: v.study_id for k, v in again.items()} == {
        k: v.study_id for k, v in frozen.items()
    }, "the id is a content hash, so re-freezing an unedited spec has to reproduce it"


def test_w6_1_registers_p7s_own_resolution_rule_and_its_direction():
    """P7 is already frozen and this spec is written to it rather than to a rule that reads better.

    P7 predicts that the curves separate, which is the outcome that kills the early-warning layer.
    A spec that quietly registered the collapse instead would be predicting the result its author
    would prefer, so the direction is asserted here.
    """
    from studies.w6_rate import w6_1_spec

    spec = w6_1_spec()
    separate = next(h for h in spec.hypotheses if h.id == "H2-curves-separate")
    assert separate.prediction.metric == "separated_fraction"
    assert separate.prediction.comparator == ">="
    assert separate.prediction.threshold == 0.05
    assert "assume bifurcation" in separate.prediction.rationale

    kill = next(k for k in spec.kill_criteria if k.id == "K-collapse-licenses-the-layer")
    assert kill.fired(0.0) and not kill.fired(0.2)
    assert "good news for the library" in kill.description


def test_a_compute_gated_study_voids_every_prediction_by_name_and_passes_nothing():
    """The one thing a study with no subjects must never do is look like it ran and was fine.

    Every prediction and every kill criterion comes back VOID with its arc named, `killed` is False
    because nothing was evaluated, and the recorded Evidence says what is missing and what it costs.
    """
    from reward_lens.core.store import EvidenceStore
    from reward_lens.studies.runner import run_study
    from studies.w6_rate import METRIC_ARCS, freeze_all

    for key, frozen in freeze_all().items():
        store = EvidenceStore()
        _, result = run_study(frozen, store=store, metric_arcs=METRIC_ARCS)
        assert result.outcome.name == "VOID"
        assert not result.killed
        assert set(result.outcomes.values()) == {"void"}
        assert set(result.kill_outcomes.values()) == {"void"}
        assert all(v.arc for v in result.voids.values()), (
            "a void with no arc is an absence rather than a work item"
        )
        gated = [e for e in store if e.observable == "W6.GatedArm"]
        assert gated, f"{key} recorded no gated-arm evidence"
        for e in gated:
            assert e.value["status"] == "inconclusive-because-gated"
            assert "GPU-hours" in e.value["price"]


def test_the_analysis_frozen_before_the_arms_exist_scores_them_when_they_do():
    """The point of writing the analysis now: the day the arms exist, this is what scores them."""
    from reward_lens.core.store import EvidenceStore
    from reward_lens.studies.runner import run_study
    from studies.w6_rate import METRIC_ARCS, freeze_all

    frozen = freeze_all()["w6-1-two-run-rate-test"]
    subjects = {
        "fast": tracker("fast", 100, tau=20.0, seed=7),
        "slow": tracker("slow", 400, tau=20.0, seed=7),
    }
    _, result = run_study(frozen, subjects=subjects, store=EvidenceStore(), metric_arcs=METRIC_ARCS)
    assert result.outcome.name != "VOID"
    assert result.outcomes["H2-curves-separate"] == "confirmed"
    assert result.outcomes["H2-lambda-beats-t"] in ("confirmed", "refuted")
    assert result.kill_outcomes["K-collapse-licenses-the-layer"] == "passed"
    assert result.metrics["separated_fraction"] >= 0.05
    assert result.evidence


def test_the_price_is_arithmetic_and_the_specs_own_three_numbers_do_not_close():
    """Every number in the price sheet comes from code, including the discrepancy in its anchor.

    The specification prices ten seeds by three conditions at "roughly 11,520 GPU-hours and $17,000 to $23,000
    at neocloud rates, against a floor of about $2.15 per GPU-hour preemptible". Those do not
    reconcile, and the price module carries all three rather than picking one silently.
    """
    from studies.w6_rate.price import (
        PRICE_POINTS,
        PRODUCTION_ARM_GPU_HOURS,
        ranked,
        w6_1_price,
    )

    assert PRODUCTION_ARM_GPU_HOURS == pytest.approx(384.0)
    assert 11_520.0 * 2.15 == pytest.approx(24_768.0)
    assert 11_520.0 * 2.15 > 23_000.0, (
        "the specification's stated preemptible floor puts its own GPU-hour figure above the top "
        "of its own dollar range"
    )
    assert PRICE_POINTS["spec_range_low_implied"] == pytest.approx(1.4757, abs=1e-4)
    assert PRICE_POINTS["spec_range_high_implied"] == pytest.approx(1.9965, abs=1e-4)

    one = w6_1_price()
    assert one.gpu_hours == pytest.approx(3 * (10.0 + 2.5))
    assert one.dollars() == pytest.approx(one.gpu_hours * 2.15)
    assert one.in_campaigns == pytest.approx(one.gpu_hours / 4.465)

    order = [p.package for p in ranked()]
    assert order[0].startswith("W6.1"), (
        "the two-run rate test is the cheapest decisive item and it is the one to buy first"
    )
    assert len(order) == 3

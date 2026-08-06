"""The clause for series I.

    A McCrary test and a bunching elasticity are computed for two hard gates in a real run and the
    elasticity responds when the gate is synthetically moved; I5 is scored against the
    gradient-norm peak in transition-width units.

Read the second half of this docstring before quoting the first half. **The real run in
`tests/fixtures/grpo_run/long` has no hard reward gate in it.** It is 200 steps of a real
`GRPOTrainer` against a length grader, its composition tree is a single leaf, and `recorded_gates`
on it returns nothing. So the two gates below are **installed** on that run's own recorded leaves
and features by `install_gate`, which is the counterfactual mechanism instrument B3 uses, pointed
at a rule the policy never saw. That makes them a validation of the estimators against cutoffs
whose location is known. It does not make them evidence about anybody's run, and the tests here
assert that the readings say so.

The one hard threshold this run really has is the sampler's completion cap at 12 tokens, and it
binds on 100 percent of rollouts, so the token-count running variable is a point mass and I1
refuses on it. That refusal is asserted too, because it is the honest half of the clause and
because it is the confound I3's second control exists for.

The gate-move half of the clause has two directions and both are tested. **Specificity** is on the
real record: the estimator is pointed at cutoffs all across a density with no gate anywhere in it,
and it must report nothing at all of them. **Sensitivity** needs a subject where a policy really
did bunch, and this run is not one, so it is tested on the real record's own lengths with a
response planted into them at a known cutoff. That plant is labelled where it is built.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from reward_lens.core.evidence import Evidence
from reward_lens.core.invariance import InvariancePayload, check_invariance, check_unit_refusal
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.threshold import (
    BunchingElasticity,
    DeadZoneFraction,
    DensityDiscontinuity,
    Gate,
    RunSeries,
    ThresholdQuantity,
    VarianceDerivative,
    bunching_elasticity,
    completion_lengths,
    decode_length,
    density_discontinuity,
    gate_response,
    install_gate,
    penalty_fraction,
    planted_run,
    rank,
    recorded_gates,
    run_series,
    score_contexts,
    variance_derivative,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "grpo_run" / "long"

#: The two cutoffs the gates are installed at. Both sit inside the support of the real
#: character-length distribution, which runs 31 to 168 with a median of 61, and both leave enough
#: rollouts on each side for a local linear density fit. A cutoff at 90 does not: the fitted density
#: there goes negative and I1 refuses, which is asserted in its own test.
CUTOFFS = (55.0, 75.0)

#: The constant a token-budget override writes. Kimi K3's rule is -1 for a trajectory over budget.
OVERRIDE_CONSTANT = -1.0


# ---------------------------------------------------------------------------
# the real subject
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_run():
    if not (FIXTURE / "runs").is_dir():
        pytest.skip(f"the 200-step GRPO record is not at {FIXTURE}")
    from reward_lens.record.reader import open_run

    run_dir = next(p.name for p in (FIXTURE / "runs").iterdir())
    return open_run(FIXTURE, run_dir.replace("run_", "run:"))


@pytest.fixture(scope="module")
def subject(real_run):
    """The running variable, the decode length, and the recorded trees, off the real record."""
    running = completion_lengths(real_run, unit="characters")
    trees, contexts, values = score_contexts(real_run, running=running)
    return {
        "run": real_run,
        "running": running,
        "tokens": completion_lengths(real_run, unit="tokens"),
        "decode": decode_length(real_run),
        "trees": trees,
        "contexts": contexts,
        "values": values,
    }


def installed(subject, cutoff: float):
    """Install a budget gate at ``cutoff`` on every recorded tree, and measure what it costs."""
    trees = subject["trees"]
    name = f"budget_gate_{int(cutoff)}"
    gated = [
        None
        if t is None
        else install_gate(t, cutoff=cutoff, constant=OVERRIDE_CONSTANT, name=name)[0]
        for t in trees
    ]
    _, gate = install_gate(trees[0], cutoff=cutoff, constant=OVERRIDE_CONSTANT, name=name)
    cost = penalty_fraction(
        trees, subject["contexts"], subject["values"], cutoff=cutoff, constant=OVERRIDE_CONSTANT
    )
    return gated, Gate(**{**gate.__dict__, "penalty_fraction": cost})


def test_the_real_run_has_no_recorded_hard_gate(subject):
    """The premise of everything below, asserted rather than assumed."""
    assert recorded_gates(subject["trees"], unit="characters") == ()
    assert subject["running"].n == 1600
    assert subject["running"].n_distinct > 50


# ---------------------------------------------------------------------------
# clause, first half: a McCrary test and a bunching elasticity for two gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_mccrary_is_computed_for_each_gate_on_the_real_run(subject, cutoff):
    _, gate = installed(subject, cutoff)
    reading = DensityDiscontinuity(
        subject["running"], gate, n_null=150, n_placebos=25, n_boot=150, decode=subject["decode"]
    ).estimate(Context())
    assert isinstance(reading, Evidence), reading
    assert reading.observable == "DensityDiscontinuity"
    assert reading.quantity == "gate.mccrary_statistic"
    value = reading.value
    assert value.n == 1600
    assert math.isfinite(value.theta) and math.isfinite(value.se) and value.se > 0
    assert math.isfinite(value.z) and 0.0 <= value.p <= 1.0
    assert value.rung == 1
    assert value.bins_left >= 3 and value.bins_right >= 3
    # both mandatory baselines are on the reading and both produced a band
    assert value.smooth_null is not None and value.smooth_null.n_draws >= 20
    assert value.placebo is not None and len(value.placebo_z) >= 5
    # the gate is installed and the reading says so
    assert value.gate.installed and "INSTALLED" in value.render()
    # and the decode length travels with it
    assert value.decode is not None and value.decode.cap == 12.0


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_bunching_elasticity_is_computed_for_each_gate_on_the_real_run(subject, cutoff):
    _, gate = installed(subject, cutoff)
    assert gate.penalty_fraction is not None and gate.penalty_fraction > 1.0
    reading = BunchingElasticity(
        subject["running"], gate, n_boot=150, n_placebos=25, decode=subject["decode"]
    ).estimate(Context())
    assert isinstance(reading, Evidence), reading
    assert reading.observable == "BunchingElasticity"
    assert reading.quantity == "gate.bunching_elasticity"
    value = reading.value
    assert value.n == 1600
    assert math.isfinite(value.excess_mass) and math.isfinite(value.dz_star)
    assert value.elasticity is not None and math.isfinite(value.elasticity)
    assert value.rung == 1
    assert value.window_chosen_from_data
    assert len(value.placebo_excess) >= 5
    assert value.smooth_null is not None
    assert value.sweep is not None and len(value.sweep.window_bins) >= 6
    assert value.gate.installed


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_the_dead_zone_fraction_carries_both_of_its_controls(subject, cutoff):
    gated, gate = installed(subject, cutoff)
    reading = DeadZoneFraction(
        gated, subject["contexts"], subject["running"], gate, subject["decode"]
    ).estimate(Context())
    assert isinstance(reading, Evidence), reading
    value = reading.value
    assert 0.0 < value.fraction < 1.0
    assert value.zone.n == 1600
    assert value.zone.n_binding == value.zone.n_region
    # control 1: the graded penalty, with the ordering it preserved or destroyed
    assert value.contrast is not None
    assert value.contrast.deadzone_fraction == 0.0
    assert value.contrast.distinct_realised_override == 1
    assert value.contrast.distinct_realised_graded > 1
    # control 2: the decode length, reported separately from the training budget
    assert value.decode.cap == 12.0
    assert value.decode.cap_key == "max_completion_length"
    assert value.decode.clipped_fraction == pytest.approx(1.0)
    assert "decode length cap 12 tokens" in value.render()


def test_a_looser_gate_leaves_a_smaller_dead_zone(subject):
    """The one monotone fact a dead-zone fraction must satisfy, on the real record."""
    fractions = []
    for cutoff in CUTOFFS:
        gated, gate = installed(subject, cutoff)
        reading = DeadZoneFraction(
            gated, subject["contexts"], subject["running"], gate, subject["decode"]
        ).estimate(Context())
        fractions.append(reading.value.fraction)
    assert fractions[0] > fractions[1]


# ---------------------------------------------------------------------------
# clause, second half: the elasticity responds when the gate moves
# ---------------------------------------------------------------------------


def bunched_at(
    lengths: np.ndarray, cutoff: float, *, share: float = 0.5, reach: float = 12.0, seed: int = 0
) -> np.ndarray:
    """The real run's lengths with a response to a gate at ``cutoff`` PLANTED into them.

    This is a plant and it is labelled one. A record re-scored under an installed gate cannot show
    bunching, because the rollouts on it were generated without the gate: bunching is a change in
    what the policy produced, and no counterfactual re-scoring of fixed rollouts can produce one. So
    the sensitivity half of I2's kill condition is tested on the real density with a known response
    written into it, and the specificity half is tested on the real density untouched.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(lengths, dtype=np.float64).copy()
    region = np.where((x >= cutoff) & (x < cutoff + reach))[0]
    if region.size == 0:
        return x
    moved = rng.choice(region, size=int(share * region.size), replace=False)
    x[moved] = cutoff - rng.uniform(0.0, 2.0, size=moved.size)
    return x


#: The three cutoffs the plant is placed at. All three sit where the real density has mass: the
#: reach window [c, c+12) holds 610, 583 and 276 of the 1600 rollouts. A plant at 80 was tried and
#: dropped, and dropping it is a statement about the plant rather than about the estimator: only 70
#: rollouts sit in [80, 92), so moving half of them produces an excess mass of 1.8 out of 1600,
#: which is not a gate response, it is noise wearing one's name. The estimator reports it as such.
PLANT_CUTOFFS = (50.0, 60.0, 70.0)

#: What a real gate does to this density, so "small" below has something to be small against. The
#: three plants produce excess masses of 291, 346 and 155 rollouts.
PLANTED_RESPONSE_FLOOR = 100.0


def test_the_elasticity_responds_when_the_gate_is_synthetically_moved(subject):
    """I2's kill condition from the catalogue: the estimate must follow the gate."""
    lengths = subject["running"].values
    samples = {c: bunched_at(lengths, c, seed=int(c)) for c in PLANT_CUTOFFS}
    gate = Gate(
        name="moved",
        cutoff=PLANT_CUTOFFS[0],
        unit="characters",
        penalty_fraction=1.9,
        installed=True,
    )
    response = gate_response(samples, gate=gate, binsize=2.0, window_bins=3)
    assert not isinstance(response, Refusal), response

    # first: the plant landed. A gate-move check on a response too small to see says nothing.
    for own in response.at_true_cutoff:
        assert own > PLANTED_RESPONSE_FLOOR, response.render()

    # then: the estimate follows the gate, and by a real margin at every position.
    assert response.tracks, response.render()
    assert response.margin > PLANTED_RESPONSE_FLOOR
    for own, others in zip(response.at_true_cutoff, response.at_other_cutoffs):
        assert own > max(others)
    assert all(math.isfinite(e) and e > 0 for e in response.elasticity)


def test_the_elasticity_does_not_respond_where_there_is_no_gate(subject):
    """The other direction, on the real record untouched: an estimator that always fires is useless.

    The claim that matters is not that the estimate is exactly zero. It is that the estimate is
    **two orders of magnitude** below what the same estimator reports on the same density when a gate
    really is there: 11 to 17 rollouts of excess mass across four cutoffs with no gate, against 155
    to 346 at the three planted gates. An estimator measuring its own exclusion window would report
    the same thing in both cases.

    A tighter containment test was tried and it is worth recording why it was dropped rather than
    quietly loosened. Requiring the observed excess mass to sit inside the 95th percentile of the
    smooth-density null failed at 75 characters, where the estimator reads -16.7 against a null of
    mean -5.1 and spread 5.3, a 2.2-sigma deficit with an empirical p of 0.007. That is a real
    property of the estimator: its bias depends on where the cutoff sits on the density, the smooth
    null measures that bias but does not fully absorb it, and one cutoff in four landing at two sigma
    is what that looks like. It is recorded in `bunching.bunching_elasticity` as a limitation. What
    it is not is a gate: 16.7 rollouts out of 1600 is 1 percent of the smallest planted response.
    """
    running = subject["running"]
    for cutoff in (45.0, 55.0, 65.0, 75.0):
        gate = Gate(
            name="none", cutoff=cutoff, unit="characters", penalty_fraction=1.9, installed=True
        )
        reading = bunching_elasticity(running, gate, n_boot=0, n_placebos=20, n_null=200, seed=1)
        assert not isinstance(reading, Refusal), reading
        band = reading.smooth_null
        assert band is not None and band.n_draws >= 100

        # nowhere near a gate-sized response
        assert abs(reading.excess_mass) < 0.2 * PLANTED_RESPONSE_FLOOR, reading.render()
        assert reading.elasticity is not None and reading.elasticity < 1e-4, reading.render()

        # and inside three standard deviations of what a smooth density of this shape produces
        deviation = (reading.excess_mass - band.mean) / band.sd
        assert abs(deviation) < 3.0, (cutoff, deviation, reading.render())


def test_moving_a_gate_that_changes_nothing_is_void_rather_than_a_failed_kill(subject):
    """E24, void condition 8, in the place series I can reach it.

    An installed gate does not change the running variable, so a gate-move check run on one sample
    re-scored twice compares two identical arrays. The honest answer is that the contrast never
    applied, not that the elasticity failed to respond, because the second reads as a kill.
    """
    lengths = subject["running"].values
    gate = Gate(name="inert", cutoff=55.0, unit="characters", penalty_fraction=1.9, installed=True)
    out = gate_response({55.0: lengths, 75.0: lengths.copy()}, gate=gate)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.VOID
    assert out.statistics["void_condition"] == "contrast_inert"
    assert "check that moving the gate actually changed the rollouts" in out.remedy


# ---------------------------------------------------------------------------
# clause, third half: I5 against the gradient-norm peak, in transition widths
# ---------------------------------------------------------------------------


def test_i5_is_scored_against_the_gradient_norm_peak_in_transition_width_units():
    """The scoring, on a planted run whose onset and width are known in closed form.

    Planted, and labelled planted. The real 200-step record has no behavioural transition in it, so
    there is no width to divide by there; that case is the next test. What this asserts is that when
    a transition exists, every detector including the gradient-norm peak is scored in the same unit
    and against the same onset.
    """
    plant = planted_run()
    series = RunSeries(
        steps=plant["steps"],
        variance=plant["variance"],
        grad_norm=plant["grad_norm"],
        outcome=plant["outcome"],
        outcome_source="planted logistic",
        grad_norm_source="planted",
        degenerate_fraction=0.0,
        n_groups=int(plant["steps"].size),
    )
    reading = variance_derivative(series, baseline_steps=80)
    assert not isinstance(reading, Refusal), reading

    # the width the leads are divided by is the planted one
    assert reading.fit.valid
    assert reading.fit.midpoint == pytest.approx(float(plant["planted_onset"]), abs=1.0)
    assert reading.fit.width == pytest.approx(float(plant["planted_width"]), rel=0.05)

    # every detector, including both readings of the gradient norm, is scored in widths
    names = {d.name for d in reading.detectors}
    assert names == {
        "variance derivative, CUSUM",
        "variance level, CUSUM",
        "gradient norm, CUSUM",
        "gradient norm, peak",
    }
    for d in reading.detectors:
        assert d.lead_in_widths is not None and math.isfinite(d.lead_in_widths)

    # the catalogue's kill condition, answered
    assert reading.beats_gradnorm is True
    mine = reading.detector("variance derivative, CUSUM")
    peak = reading.detector("gradient norm, peak")
    assert mine.lead_in_widths > 0.5 > peak.lead_in_widths

    # and the alarms are distinguishable from this series' own false-alarm rate
    for c in reading.calibration:
        assert c.informative, c.render()


def test_i5_on_the_real_run_refuses_the_width_and_bounds_the_alarms(subject):
    """The honest reading on the real record: no transition, so no unit, so no lead time.

    The alarms are still worth having and they are in the bound. What is refused is the conversion
    to transition widths, which is the only unit two runs can be compared in, and refusing it is
    the difference between this and a confident wrong number: the logistic converges on this series,
    on a step through the noise, and the width it returns would have made every lead about twelve
    transition windows.
    """
    series = run_series(subject["run"])
    assert series.grad_norm is not None
    assert series.outcome is not None
    reading = variance_derivative(series, baseline_steps=60, n_surrogates=100)
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.RECORD_INCOMPLETE
    assert reading.is_bounded
    assert "does not beat a constant" in reading.detail
    assert reading.statistics["fit_p_value"] > 0.001
    assert "outcome series that contains a behavioural transition" in reading.remedy
    # the bound carries the alarm steps and nothing derived from the width it refused
    for d in reading.partial.value:
        assert d.lead_in_widths is None and d.lead_steps is None
    assert any(d.alarm_step is not None for d in reading.partial.value)
    # and the surrogate alarm rates that say what an alarm on this run is worth
    rates = reading.statistics["surrogate_alarm_rate"]
    assert set(rates) == {
        "variance derivative, CUSUM",
        "variance level, CUSUM",
        "gradient norm, CUSUM",
    }
    assert all(0.0 <= v <= 1.0 for v in rates.values())


# ---------------------------------------------------------------------------
# the honest half: the one real hard threshold this run has
# ---------------------------------------------------------------------------


def test_the_real_decode_cap_is_a_hard_threshold_the_density_test_refuses_on(subject):
    """The sampler's cap at 12 tokens is a real hard threshold, and it has no density to test.

    Every completion in this run is exactly 12 tokens, so the token-count running variable is a
    point mass on its own cutoff. There is no density on either side, the fix is upstream, and the
    remedy says so rather than telling the reader to get more access.
    """
    decode = subject["decode"]
    assert decode.cap == 12.0
    assert decode.binds_everywhere

    gate = Gate(
        name="decode_cap",
        cutoff=12.0,
        unit="tokens",
        installed=False,
        provenance="max_completion_length in the recorded GRPOConfig",
    )
    out = density_discontinuity(subject["tokens"], gate, instrument="DensityDiscontinuity")
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "record completion length (tokens) per rollout" in out.remedy


def test_a_cutoff_in_the_thin_tail_refuses_rather_than_reporting_a_negative_density(subject):
    """At 90 characters the local linear fit extrapolates past the data and goes negative."""
    gate = Gate(name="thin", cutoff=90.0, unit="characters", installed=True)
    out = density_discontinuity(subject["running"], gate)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert out.statistics["density_right"] <= 0 or out.statistics["density_left"] <= 0
    assert "test a cutoff further inside the support" in out.remedy


# ---------------------------------------------------------------------------
# the four declarations every instrument has to satisfy
# ---------------------------------------------------------------------------


INSTRUMENTS = [
    DensityDiscontinuity(),
    BunchingElasticity(),
    DeadZoneFraction(),
    VarianceDerivative(),
]


@pytest.mark.parametrize("instrument", INSTRUMENTS, ids=lambda i: i.name)
def test_every_instrument_lints_clean(instrument):
    assert lint_instrument(instrument) == []


@pytest.mark.parametrize("instrument", INSTRUMENTS, ids=lambda i: i.name)
def test_the_generated_invariance_test_passes(instrument):
    """`units` routes to a refusal rather than to a value relation, and the refusal is asserted."""
    report = check_invariance(instrument, "units", InvariancePayload(), run=lambda i, p: 1.0)
    assert report.passed and "refusal" in report.skipped

    # The real assertion: a McCrary z and a bunching elasticity do not rank against each other.
    assert check_unit_refusal(
        rank,
        ThresholdQuantity("gate.mccrary_statistic", 8.0),
        ThresholdQuantity("gate.bunching_elasticity", 0.02),
    )
    assert check_unit_refusal(
        rank,
        ThresholdQuantity("gate.deadzone_fraction", 0.31),
        ThresholdQuantity("run.variance_derivative", 0.004),
    )
    out = rank(
        ThresholdQuantity("gate.mccrary_statistic", 8.0),
        ThresholdQuantity("gate.bunching_elasticity", 0.02),
    )
    assert isinstance(out, Refusal) and out.reason is RefusalReason.UNIT_MISMATCH
    assert "the larger number is not the worse gate" in out.remedy


@pytest.mark.parametrize("instrument", INSTRUMENTS, ids=lambda i: i.name)
def test_an_instrument_with_nothing_to_read_refuses_with_a_remedy(instrument):
    """Rule 1: a refusal is a value carrying a reason and an instruction, never an exception."""
    out = instrument.estimate(Context())
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert out.remedy.strip() and out.remedy != out.detail
    assert "pass `" in out.remedy

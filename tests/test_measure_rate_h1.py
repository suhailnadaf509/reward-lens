"""H1 rung 0, the adiabaticity number, against dynamics whose relaxation time is known.

`Ad = tau_relax * |d log lambda / dt|` is a product of two things measured in different ways, so
they are checked separately before they are checked together. The driving rate is exact: an
exponential schedule has a constant log derivative and the estimator either returns it or is wrong.
The relaxation time is statistical, so it is checked against first-order autoregressive series with
a planted coefficient, and what is asserted is that the bias correction moves the estimate toward
the planted value rather than that any single fit lands on it.

The disagreement with `rate/regime.py` is asserted rather than avoided. Both estimators are rung 0
for `run.tau_relax`, they read the same series over the same early window, and the uncorrected one
is short by construction. That difference is the reading `tau_transfer` publishes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reward_lens.core.envelope import RegimeCondition
from reward_lens.core.invariance import (
    INVARIANT,
    InvariancePayload,
    check_invariance,
    check_unit_refusal,
)
from reward_lens.core.quantity import ladder
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Access, Component
from reward_lens.measure.base import lint_instrument
from reward_lens.measure.rate.adiabaticity import (
    Adiabaticity,
    DriveRate,
    RelaxationFloors,
    RelaxationTime,
    _phi_ols,
    adiabaticity,
    adiabaticity_number,
    drive_rates,
    regime_inputs,
    relaxation_time,
    tau_of,
    tau_transfer,
)
from reward_lens.measure.rate.regime import measure_regime
from reward_lens.record.provenance import SegmentProvenance
from reward_lens.record.schema import (
    EstimatorSpec,
    Group,
    GroupID,
    GroupStats,
    InMemoryStepStream,
    OptimizerTelemetry,
    RegimeDeclaration,
    Run,
    RunID,
    Step,
    TaskID,
    make_trajectory,
)
from reward_lens.record.tensors import Engine
from reward_lens.record.turns import Turn

ENGINE = Engine(name="vllm", revision="0.11.2", dtype="bfloat16")

#: A cheaper bootstrap than the shipped 400, so the suite stays quick. The bias it estimates is
#: stable well below this; the two tests that measure coverage say what they used.
FAST = RelaxationFloors(n_boot=120)


# ---------------------------------------------------------------------------
# Synthetic records
# ---------------------------------------------------------------------------


def _step(index: int, *, lam: float, mean: float) -> Step:
    trajectory = make_trajectory(
        id=f"t{index}",
        task_ref="task0",
        turns=[Turn(index=0, role="assistant", token_ids=(1, 2, 3, 4))],
        advantage=0.0,
        provenance=[
            SegmentProvenance(
                turn_range=(0, 1),
                policy_version="p0",  # type: ignore[arg-type]
                staleness_steps=0,
                engine=ENGINE,
            )
        ],
    )
    return Step(
        index=index,
        groups=(
            Group(
                id=GroupID(f"g{index}"),
                task_ref=TaskID("task0"),
                trajectories=(trajectory,),
                estimator=EstimatorSpec(family="grpo"),
                group_stats=GroupStats(k=4, mean=mean, std=0.3),
            ),
        ),
        schedule={"kl_coefficient": lam},
        optimizer=OptimizerTelemetry(),
    )


def annealed_run(*, rate: float, means, run_id: str = "synthetic") -> Run:
    """A record whose only schedule parameter moves at exactly ``rate`` per step in log units.

    Exponential rather than linear, because `Ad` is defined on `d log lambda / dt` and an
    exponential schedule makes that derivative a constant. On a linear schedule the measured rate
    depends on which pair of steps the window happened to contain, which is a property of the
    window rather than of the run.
    """
    return Run(
        id=RunID(run_id),
        kind="train",
        components={},
        access={Component.RECORD: Access.RECORD},
        regime=RegimeDeclaration(),
        steps=InMemoryStepStream(
            [_step(i, lam=math.exp(rate * i), mean=float(m)) for i, m in enumerate(means)]
        ),
    )


def ar1(phi: float, n: int, seed: int, *, level: float = 0.5, sd: float = 0.05):
    """A first-order autoregressive series about ``level``, which is what a relaxing observable is."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sd)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0.0, sd)
    return level + x


# ---------------------------------------------------------------------------
# The mapping from a coefficient to a time
# ---------------------------------------------------------------------------


def test_the_coefficient_maps_onto_a_time_with_both_ends_defined():
    """Hand-computed. ``phi = exp(-1)`` is one step of relaxation, by definition."""
    assert tau_of(math.exp(-1.0)) == pytest.approx(1.0)
    assert tau_of(math.exp(-0.5)) == pytest.approx(2.0)
    assert tau_of(0.0) == 0.0
    assert tau_of(-0.5) == 0.0, "a series with no positive memory relaxes inside one step"
    assert tau_of(1.0) == float("inf")
    assert tau_of(float("nan")) == 0.0


@settings(deadline=None, max_examples=50)
@given(
    a=st.floats(min_value=-0.99, max_value=0.999),
    b=st.floats(min_value=-0.99, max_value=0.999),
)
def test_the_relaxation_time_is_monotone_in_the_coefficient(a, b):
    """More memory is never less time. If this fails, an interval's ends can cross."""
    lo, hi = (a, b) if a <= b else (b, a)
    assert tau_of(lo) <= tau_of(hi)


# ---------------------------------------------------------------------------
# The relaxation time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phi_true", [0.0, 0.3, 0.6, 0.85])
def test_the_correction_lengthens_the_relaxation_time_on_every_replicate(phi_true):
    """The direction claim, asserted per replicate rather than on an average.

    This is the whole reason the module exists. `rate/regime.py` reports the uncorrected
    coefficient, a short coefficient is a short relaxation time, a short relaxation time is a small
    `Ad`, and a small `Ad` licenses treating the run as quasi-static. So the cheap estimator errs
    toward licensing on every run, not on average, and the correction has to move every one of them
    the same way or it is not a correction.
    """
    for s in range(30):
        x = ar1(phi_true, 50, seed=500 + s, level=0.0, sd=1.0)
        out = relaxation_time(x, name="planted", floors=FAST, seed=s)
        assert not isinstance(out, Refusal)
        assert out.bias < 0, "the least-squares lag-1 fit is biased low"
        assert out.phi > out.phi_ols
        assert out.tau >= tau_of(out.phi_ols)


@pytest.mark.parametrize("phi_true", [0.6, 0.85])
def test_the_bias_correction_reduces_the_error_where_the_bias_is_large(phi_true):
    """Measured over 40 replicates at n = 50, at the coefficients where 40 can resolve it.

    At phi = 0.3 the uncorrected bias is -0.056 and the estimator's own spread is 0.15, so forty
    replicates cannot separate the two and asserting it there would flake. Over 200 replicates the
    correction takes the bias from -0.035, -0.056, -0.083 and -0.124 at phi = 0, 0.3, 0.6 and 0.85
    down to +0.004, +0.007, +0.006 and -0.010, and it costs a little variance at low phi: root mean
    squared error improves from 0.148 to 0.137 at phi = 0.6 and worsens from 0.147 to 0.156 at
    phi = 0. Those numbers come from a replicate count this test does not run.
    """
    raw, corrected = [], []
    for s in range(40):
        x = ar1(phi_true, 50, seed=500 + s, level=0.0, sd=1.0)
        raw.append(_phi_ols(x))
        out = relaxation_time(x, name="planted", floors=FAST, seed=s)
        assert not isinstance(out, Refusal)
        corrected.append(out.phi)
    raw_bias = float(np.mean(raw)) - phi_true
    corrected_bias = float(np.mean(corrected)) - phi_true
    assert raw_bias < 0
    assert abs(corrected_bias) < abs(raw_bias)


def test_the_measured_bias_is_reported_beside_the_corrected_coefficient():
    x = ar1(0.6, 50, seed=11, level=0.0, sd=1.0)
    out = relaxation_time(x, name="planted", floors=FAST, seed=0)
    assert not isinstance(out, Refusal)
    assert out.phi == pytest.approx(out.phi_ols - out.bias)
    assert out.bias < 0
    assert out.phi_ci[0] <= out.phi <= out.phi_ci[1]
    assert out.tau_low <= out.tau <= out.tau_high
    assert "after removing a measured bias" in out.render()


def test_the_relaxation_time_interval_covers_the_planted_one():
    """30 replicates at n = 50 and phi = 0.6, nominal 0.95. Asserted at 0.80.

    The wider measurement is separate: over 200 replicates the realised coverage of this
    interval is 0.86 to 0.93 at n = 20 and 0.93 to 0.98 at n = 120, so it under-covers at small n.
    Under-coverage makes the upper bound on `Ad` optimistic, which is stated on the instrument.
    """
    target = tau_of(0.6)
    hits = 0
    for s in range(30):
        x = ar1(0.6, 50, seed=700 + s, level=0.0, sd=1.0)
        out = relaxation_time(x, name="planted", floors=FAST, seed=s)
        assert not isinstance(out, Refusal)
        hits += int(out.tau_low <= target <= out.tau_high)
    assert hits >= 24, f"only {hits}/30 intervals covered tau = {target:.4g}"


def test_a_series_shorter_than_the_floor_refuses():
    out = relaxation_time([0.1, 0.2, 0.3], name="group_mean", floors=FAST)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert out.statistics["n"] == 3
    assert "nothing that short shows a return" in out.remedy


def test_a_series_lying_exactly_on_its_trend_refuses():
    out = relaxation_time(np.arange(40, dtype=float), name="ramp", floors=FAST)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "exactly linear" in out.detail


# ---------------------------------------------------------------------------
# The driving rate
# ---------------------------------------------------------------------------


def test_the_driving_rate_is_exact_on_an_exponential_schedule():
    run = annealed_run(rate=0.25, means=ar1(0.5, 20, seed=1))
    steps = sorted(run.steps.slice(0, 20), key=lambda s: s.index)
    rates = drive_rates(steps)
    assert len(rates) == 19
    assert all(r.rate == pytest.approx(0.25) for r in rates)
    assert all(r.parameter == "kl_coefficient" for r in rates)


def test_a_flat_schedule_gives_a_rate_of_exactly_zero_rather_than_nothing():
    run = annealed_run(rate=0.0, means=ar1(0.5, 20, seed=2))
    steps = sorted(run.steps.slice(0, 20), key=lambda s: s.index)
    rates = drive_rates(steps)
    assert len(rates) == 19
    assert all(r.rate == 0.0 for r in rates)


# ---------------------------------------------------------------------------
# Ad, per step
# ---------------------------------------------------------------------------


def test_ad_is_computed_per_step_from_a_record_alone():
    run = annealed_run(rate=0.02, means=ar1(0.6, 60, seed=3))
    out = adiabaticity(run, floors=FAST)
    assert not isinstance(out, Refusal)
    assert len(out.per_step) == 59, "one Ad per consecutive pair of recorded steps"
    assert all(s.ad == pytest.approx(out.tau.tau * s.rate) for s in out.per_step)
    assert all(s.ad_low <= s.ad <= s.ad_high for s in out.per_step)
    assert out.window == (0, 60)


def test_the_worst_step_is_not_the_typical_step_on_a_schedule_that_decays_to_zero():
    """A linear decay to zero has a small log derivative everywhere and an unbounded one at the end.

    This is the case a window maximum hides, and it is not hypothetical: it is what the shipped
    learning-rate schedule of the GRPO fixture does. Reporting one number for the window would say
    "fast driving" about a run that was quasi-static for all but its final step.
    """
    n = 40
    lam = [1e-6 * (1.0 - i / n) for i in range(n)]
    means = ar1(0.5, n, seed=4)
    steps = [_step(i, lam=lam[i], mean=float(means[i])) for i in range(n)]
    run = Run(
        id=RunID("linear-decay"),
        kind="train",
        components={},
        access={Component.RECORD: Access.RECORD},
        regime=RegimeDeclaration(),
        steps=InMemoryStepStream(steps),
    )
    out = adiabaticity(run, floors=FAST)
    assert not isinstance(out, Refusal)
    rates = [s.rate for s in out.per_step]
    # 14x over 40 steps. On the 200-step GRPO fixture, whose learning rate decays the same way, the
    # ratio is 70: the last step moves at ln 2 per step and the median step at 0.00995.
    assert max(rates) > 10.0 * float(np.median(rates))
    assert out.worst_step == n - 1
    assert out.median_ad < out.ad


def test_a_flat_schedule_is_quasi_static_whatever_the_relaxation_time_is():
    run = annealed_run(rate=0.0, means=ar1(0.9, 60, seed=5))
    out = adiabaticity(run, floors=FAST)
    assert not isinstance(out, Refusal)
    assert out.flat_schedule
    assert out.ad == 0.0 and out.ad_high == 0.0
    assert out.holds is True
    assert "the driver is not moving" in out.says()


def test_a_fast_driven_run_fails_the_condition_on_the_lower_end_of_its_interval():
    """Driven at 3.0 per step in log units against a relaxation time near one step.

    `holds` is False only when the *lower* end of the interval exceeds the threshold, so a run that
    fails here fails whatever the relaxation time is inside its interval. The coefficient is planted
    at 0.5 rather than higher because a coefficient near one gives an interval reaching past one,
    where the honest answer is the bounded refusal rather than a verdict.
    """
    run = annealed_run(rate=3.0, means=ar1(0.5, 60, seed=9))
    out = adiabaticity(run, floors=FAST)
    assert not isinstance(out, Refusal)
    assert out.ad_low > out.threshold
    assert out.holds is False
    assert "QUASI_STATIC fails" in out.says()


def test_a_run_with_no_schedule_refuses_rather_than_reading_a_rate_of_zero():
    """An empty schedule is not a flat one, and the difference is the whole refusal."""
    means = ar1(0.5, 20, seed=7)
    steps = [
        Step(
            index=i,
            groups=_step(i, lam=1.0, mean=float(means[i])).groups,
            schedule={},
            optimizer=OptimizerTelemetry(),
        )
        for i in range(20)
    ]
    run = Run(
        id=RunID("no-schedule"),
        kind="train",
        components={},
        access={Component.RECORD: Access.RECORD},
        regime=RegimeDeclaration(),
        steps=InMemoryStepStream(steps),
    )
    out = adiabaticity(run, floors=FAST)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "An empty schedule is not a flat one" in out.remedy


def test_a_window_of_one_step_refuses_rather_than_differentiating_nothing():
    run = annealed_run(rate=0.1, means=ar1(0.5, 20, seed=8))
    out = adiabaticity(run, window=(3, 4), floors=FAST)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "a window of one step has none" in out.remedy


def test_a_run_with_no_steps_is_void_rather_than_quasi_static():
    run = Run(
        id=RunID("empty"),
        kind="train",
        components={},
        access={Component.RECORD: Access.RECORD},
        regime=RegimeDeclaration(),
        steps=InMemoryStepStream([]),
    )
    out = adiabaticity(run, floors=FAST)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.VOID


def test_a_non_stationary_early_fit_returns_a_bound_rather_than_a_number():
    """A random walk does not return, so its relaxation time has no upper end and neither does Ad.

    The bound is the honest part of the answer: `Ad` is at least the lower end, and that is a
    statement a reader can act on where "the fit failed" is not.
    """
    walk = np.cumsum(np.random.default_rng(9).normal(0.0, 1.0, 60))
    run = annealed_run(rate=0.5, means=walk)
    out = adiabaticity(run, floors=FAST)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ABOVE_LOD_BELOW_LOQ
    assert out.is_bounded and out.partial is not None
    assert out.statistics["phi_ci_high"] >= 1.0
    assert "perturb-and-hold" in out.remedy


# ---------------------------------------------------------------------------
# The unit
# ---------------------------------------------------------------------------


def test_multiplying_a_time_in_steps_by_a_rate_per_epoch_refuses():
    x = ar1(0.6, 50, seed=12, level=0.0, sd=1.0)
    tau = relaxation_time(x, name="planted", floors=FAST)
    assert not isinstance(tau, Refusal)
    per_step = DriveRate(from_step=0, to_step=1, rate=0.4, parameter="kl_coefficient")
    assert adiabaticity_number(tau, per_step) == pytest.approx(tau.tau * 0.4)

    per_epoch = DriveRate(
        from_step=0, to_step=1, rate=0.4, parameter="kl_coefficient", axis="epoch"
    )
    out = adiabaticity_number(tau, per_epoch)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.UNIT_MISMATCH
    assert "not dimensionless" in out.detail
    assert check_unit_refusal(adiabaticity_number, tau, per_epoch)


def test_the_generated_invariance_check_for_the_units_group():
    report = check_invariance(
        Adiabaticity(None),
        "units",
        InvariancePayload(),
        relation=INVARIANT,
        run=lambda i, p: 1.0,
    )
    assert report.passed
    assert "refusal" in report.skipped


# ---------------------------------------------------------------------------
# The handoff and the disagreement
# ---------------------------------------------------------------------------


def test_the_handoff_lets_the_regime_reading_answer_quasi_static():
    """The series is white noise about a level, which is where `rate/regime.py` returns nothing.

    Its uncorrected coefficient lands at or below zero, so it reports no relaxation time and
    `QUASI_STATIC` comes back `None`. The corrected coefficient is positive, its interval has a
    finite upper end, and the condition gets a verdict.
    """
    run = annealed_run(rate=0.01, means=ar1(0.0, 60, seed=13))
    before = measure_regime(run).conditions[RegimeCondition.QUASI_STATIC]
    assert before.holds is None
    assert "at or below zero" in before.detail

    out = adiabaticity(run, floors=FAST)
    assert not isinstance(out, Refusal)
    after = measure_regime(run, inputs=regime_inputs(out)).conditions[RegimeCondition.QUASI_STATIC]
    assert after.holds is not None
    assert math.isfinite(after.statistic)


def test_the_handoff_defaults_to_the_upper_bound_and_says_so():
    run = annealed_run(rate=0.01, means=ar1(0.4, 60, seed=14))
    out = adiabaticity(run, floors=FAST)
    assert not isinstance(out, Refusal)
    assert regime_inputs(out).tau_relax == pytest.approx(out.tau.tau_high)
    assert regime_inputs(out, bound="point").tau_relax == pytest.approx(out.tau.tau)
    assert regime_inputs(out).relaxation_series == "group_mean"


def test_the_handoff_on_a_flat_schedule_passes_no_relaxation_time_at_all():
    """There is nothing to hand over and `measure_regime` needs nothing: an exact zero times
    anything is zero, and its own flat-schedule branch answers before it looks for a time.
    """
    run = annealed_run(rate=0.0, means=ar1(0.9, 60, seed=17))
    out = adiabaticity(run, floors=FAST)
    assert not isinstance(out, Refusal) and out.flat_schedule
    assert regime_inputs(out).tau_relax is None
    answered = measure_regime(run, inputs=regime_inputs(out)).conditions[
        RegimeCondition.QUASI_STATIC
    ]
    assert answered.holds is True
    assert answered.statistic == 0.0


def test_the_two_rung_zero_estimators_disagree_and_the_difference_is_a_transfer():
    run = annealed_run(rate=0.01, means=ar1(0.6, 60, seed=15))
    out = adiabaticity(run, floors=FAST)
    assert not isinstance(out, Refusal)
    transfer = tau_transfer(out)
    assert transfer.value == pytest.approx(abs(tau_of(out.tau.phi_ols) - out.tau.tau))
    assert transfer.value > 0
    assert transfer.name == "t21"
    assert "rate/regime.py" in transfer.method
    assert out.tau.tau > tau_of(out.tau.phi_ols), "the corrected time is the longer one"


def test_the_transfer_says_so_when_the_cheap_estimator_declined_entirely():
    run = annealed_run(rate=0.01, means=ar1(0.0, 60, seed=13))
    out = adiabaticity(run, floors=FAST)
    assert not isinstance(out, Refusal)
    assert out.tau.phi_ols <= 0
    transfer = tau_transfer(out)
    assert "an absence rather than a zero" in transfer.method
    assert transfer.value == pytest.approx(out.tau.tau)


# ---------------------------------------------------------------------------
# The instrument's own declarations
# ---------------------------------------------------------------------------


def test_the_instrument_passes_the_section_four_two_lint():
    assert lint_instrument(Adiabaticity(None)) == []


def test_the_ladder_carries_a_rung_one_that_is_priced_and_not_built():
    """A rung nobody has built is a research target the capability report can price, not a gap."""
    for quantity in ("run.tau_relax", "run.adiabaticity"):
        rungs = {e.rung: e for e in ladder(quantity)}
        assert set(rungs) == {0, 1}
        assert rungs[1].run is None
        assert "Not built." in rungs[1].cost.note
        assert rungs[1].requires[Component.POLICY] is Access.MUTATE
        assert rungs[1].requires[Component.OPTIMIZER] is Access.CONTROL
        assert rungs[0].requires == {Component.RECORD: Access.RECORD}


def test_the_envelope_cannot_require_the_condition_this_instrument_measures():
    envelope = Adiabaticity.envelope
    assert RegimeCondition.QUASI_STATIC not in envelope.requires
    assert envelope.requires == frozenset({RegimeCondition.STATIONARY_GRADER})
    assert envelope.on_violation == "downgrade"


def test_the_reading_round_trips_through_the_evidence_codec():
    from reward_lens.core.evidence import ValueCodec

    run = annealed_run(rate=0.05, means=ar1(0.5, 40, seed=16))
    out = adiabaticity(run, floors=FAST)
    assert not isinstance(out, Refusal)
    codec = ValueCodec()
    back = codec.decode(codec.encode(out))
    assert back.ad == pytest.approx(out.ad)
    assert isinstance(back.tau, RelaxationTime)
    assert len(back.per_step) == len(out.per_step)

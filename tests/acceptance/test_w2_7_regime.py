"""Acceptance: an instrument declaring QUASI_STATIC is refused at Ad = 3.

The clause: *an instrument declaring `QUASI_STATIC` is refused on a synthetic record with `Ad = 3`,
and the refusal carries the statistic and the threshold.*

The kernel already discharged that clause against a hand-built `RegimeReading`, which proved the
envelope machinery works and proved nothing about whether anything can produce the reading. This
file discharges it the other way round: the record is synthetic, the reading is measured from it by
`measure_regime`, and the refusal comes out of `BaseObservable.preflight` with the measured
statistic in it. Nobody types 3.0 anywhere; the schedule and the relaxation time are what make it
three.

The other three tests here are the surrounding claims that make the first one worth having. All
three of the regime violation behaviours are driven by the same measured reading, so `bound` and
`downgrade` are not dead branches. An undetermined condition refuses as well as a failed one,
because unknown is not a pass. And the capability report renders the measured block, which is what
the reading is for.
"""

from __future__ import annotations

import math

import pytest

from reward_lens.access.report import capability_report
from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    Phase,
    Substrate,
    TrustLevel,
)
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.rate.regime import MEASURED_BY, RegimeInputs, measure_regime
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


# ---------------------------------------------------------------------------
# One synthetic record, annealed at a stated rate
# ---------------------------------------------------------------------------


def _step(index: int, *, kl_coefficient: float, mean: float) -> Step:
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
        schedule={"kl_coefficient": kl_coefficient},
        optimizer=OptimizerTelemetry(),
    )


def annealed_run(*, rate: float, n: int = 8, means=None) -> Run:
    """A record whose only schedule parameter moves at exactly ``rate`` in log units per step.

    An exponential schedule rather than a linear one, because the adiabaticity number is defined on
    ``d log lambda / dt`` and an exponential schedule makes that derivative a constant. Anything
    else would make the measured rate depend on which pair of steps the window happened to contain,
    and then the number in the refusal would be an artifact of the window rather than a property of
    the run.
    """
    return Run(
        id=RunID("synthetic-fast-anneal"),
        kind="train",
        components={},
        access={Component.RECORD: Access.RECORD},
        regime=RegimeDeclaration(),
        steps=InMemoryStepStream(
            [
                _step(
                    i,
                    kl_coefficient=math.exp(rate * i),
                    mean=0.5 if means is None else means[i],
                )
                for i in range(n)
            ]
        ),
    )


# ---------------------------------------------------------------------------
# Instruments that declare the condition. Three, one per violation behaviour.
# ---------------------------------------------------------------------------


_RECORD_ONLY: AccessMatrix = {Component.RECORD: Access.RECORD}


class _NeedsQuasiStatic(BaseObservable):
    """Any Level 0 extrapolation. Outside the envelope the quantity is undefined, so: refuse."""

    name = "NeedsQuasiStatic"
    version = "1.0"
    capabilities = Capability.NONE
    quantity = "run.adiabaticity"
    requires: AccessMatrix = _RECORD_ONLY
    substrates = frozenset(Substrate)
    phases = frozenset({Phase.IN_RUN, Phase.POST_RUN})
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.QUASI_STATIC}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    invariance = "units"
    baselines = ("baseline.length",)


class _BoundsOutsideQuasiStatic(_NeedsQuasiStatic):
    name = "BoundsOutsideQuasiStatic"
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.QUASI_STATIC}),
        measured_by=MEASURED_BY,
        on_violation="bound",
        bound_estimator="run.adiabaticity.rate_bound",
    )


class _DowngradesOutsideQuasiStatic(_NeedsQuasiStatic):
    name = "DowngradesOutsideQuasiStatic"
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.QUASI_STATIC}),
        measured_by=MEASURED_BY,
        on_violation="downgrade",
    )


def _context(reading) -> Context:
    return Context(
        signal=None,
        access=_RECORD_ONLY,
        substrate=Substrate.NEURAL_GEN,
        phase=Phase.POST_RUN,
        regime_reading=reading,
    )


# ---------------------------------------------------------------------------
# The clause
# ---------------------------------------------------------------------------


def test_an_instrument_declaring_quasi_static_is_refused_at_ad_three():
    """The clause, against a reading measured from a record.

    `tau_relax = 3` steps and a schedule moving at 1.0 in log units per step give `Ad = 3`. The
    relaxation time is supplied rather than fitted because that is what makes the number exact: a
    fitted `tau` on a finite series carries its own sampling error, and a clause about the value 3
    should not be discharged by a number that happens to round there. The rung-1 method measures
    `tau` by perturb-and-hold and hands it over exactly like this. The record-derived path is the
    test immediately below.
    """
    run = annealed_run(rate=1.0)
    reading = measure_regime(run, inputs=RegimeInputs(tau_relax=3.0))

    measured = reading.conditions[RegimeCondition.QUASI_STATIC]
    assert measured.statistic == pytest.approx(3.0)
    assert measured.threshold == 1.0
    assert measured.holds is False

    result = _NeedsQuasiStatic().preflight(_context(reading))
    assert result.ok is False
    refusal = result.refusal
    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.ENVELOPE_VIOLATED

    # The statistic and the threshold, both machine-readable and both in the rendered text.
    assert refusal.statistics["condition"] == "QUASI_STATIC"
    assert refusal.statistics["statistic"] == pytest.approx(3.0)
    assert refusal.statistics["threshold"] == 1.0
    assert "3 (threshold 1)" in refusal.detail
    assert refusal.remedy.strip()


def test_the_same_refusal_arrives_with_the_relaxation_time_read_off_the_record():
    """No supplied `tau`: the AR(1) fit supplies it, and the anneal is fast.

    The number is not 3 here and it is not supposed to be. What this checks is that the whole path
    from a record with nothing but a schedule and a reward series to a refusal carrying real
    numbers closes without anybody handing it a relaxation time.

    The anneal rate is well clear of the boundary on purpose. The fitted relaxation time is biased
    low, by the OLS lag-1 bias and by detrending on top of it, so a case built to sit near `Ad = 1`
    would be testing the bias rather than the path. Building this is what measured that bias: a
    planted `phi = 0.75` over forty points fitted 0.398 here, which is a relaxation time of 1.09
    steps against a planted 3.48.
    """
    run = annealed_run(rate=2.0, n=40, means=_ar1_series(n=40, phi=0.75, seed=4))
    reading = measure_regime(run)

    measured = reading.conditions[RegimeCondition.QUASI_STATIC]
    assert measured.holds is False, measured.render()
    assert measured.statistic > 1.0
    assert "early AR(1) coefficient" in measured.detail
    assert "errs toward licensing" in measured.detail

    refusal = _NeedsQuasiStatic().preflight(_context(reading)).refusal
    assert isinstance(refusal, Refusal)
    assert refusal.statistics["statistic"] == pytest.approx(measured.statistic)


def _ar1_series(*, n: int, phi: float, seed: int) -> list[float]:
    import numpy as np

    rng = np.random.default_rng(seed)
    x, out = 0.0, []
    for _ in range(n):
        x = phi * x + float(rng.normal(0, 0.1))
        out.append(0.5 + x)
    return out


# ---------------------------------------------------------------------------
# The other two violation behaviours, on the same measured reading
# ---------------------------------------------------------------------------


def test_bound_and_downgrade_are_driven_by_the_same_measured_reading():
    """The three violation behaviours are not interchangeable and all three need the reading."""
    reading = measure_regime(annealed_run(rate=1.0), inputs=RegimeInputs(tau_relax=3.0))
    ctx = _context(reading)

    bounded = _BoundsOutsideQuasiStatic().preflight(ctx)
    assert bounded.ok is True
    assert bounded.bound_estimator == "run.adiabaticity.rate_bound"
    assert any("bound estimator" in note for note in bounded.notes)

    downgraded = _DowngradesOutsideQuasiStatic().preflight(ctx)
    assert downgraded.ok is True
    assert downgraded.trust_cap is TrustLevel.EXPLORATORY


def test_a_quasi_static_run_lets_the_same_instrument_through():
    """The counterfactual, so the refusal above is about the regime and not about the plumbing."""
    reading = measure_regime(annealed_run(rate=0.01), inputs=RegimeInputs(tau_relax=3.0))
    assert reading.conditions[RegimeCondition.QUASI_STATIC].statistic == pytest.approx(0.03)
    result = _NeedsQuasiStatic().preflight(_context(reading))
    assert result.ok is True
    assert result.refusal is None


# ---------------------------------------------------------------------------
# Unknown is not a pass
# ---------------------------------------------------------------------------


def test_a_condition_nobody_could_determine_refuses_as_well_as_one_that_failed():
    """A record with no schedule cannot give an adiabaticity number, and that is not a licence."""
    run = Run(
        id=RunID("no-schedule"),
        kind="train",
        components={},
        access={Component.RECORD: Access.RECORD},
        regime=RegimeDeclaration(declared={RegimeCondition.QUASI_STATIC: True}),
        steps=InMemoryStepStream(
            [
                Step(index=i, groups=(), schedule={}, optimizer=OptimizerTelemetry())
                for i in range(4)
            ]
        ),
    )
    reading = measure_regime(run)
    measured = reading.conditions[RegimeCondition.QUASI_STATIC]
    assert measured.holds is None
    assert math.isnan(measured.statistic)
    # The run said it was quasi-static and nobody could check, which is the disagreement the
    # declaration exists to surface rather than a match.
    assert "the run declared this true" in measured.detail

    refusal = _NeedsQuasiStatic().preflight(_context(reading)).refusal
    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "unknown" in refusal.detail
    assert "An empty schedule is not a flat one" in refusal.detail
    assert math.isnan(refusal.statistics["statistic"])

    # The capability report is the path with the per-condition remedy on it, and it distinguishes
    # a condition that failed from one nobody could settle.
    report = capability_report(
        _RECORD_ONLY,
        Substrate.NEURAL_GEN,
        Phase.POST_RUN,
        reading,
        instruments=[_NeedsQuasiStatic()],
    )
    assert "unknown" in report.render()


# ---------------------------------------------------------------------------
# The first consumer
# ---------------------------------------------------------------------------


def test_the_capability_report_prints_the_measured_regime_and_refuses_on_it():
    reading = measure_regime(annealed_run(rate=1.0), inputs=RegimeInputs(tau_relax=3.0))
    report = capability_report(
        _RECORD_ONLY,
        Substrate.NEURAL_GEN,
        Phase.POST_RUN,
        reading,
        instruments=[_NeedsQuasiStatic()],
    )
    text = report.render()
    assert "REGIME MEASURED" in text
    assert "QUASI_STATIC" in text
    assert "3 (threshold 1)" in text
    assert any(
        a.preflight.refusal is not None and a.quantity == "run.adiabaticity"
        for a in report.assessments
    )

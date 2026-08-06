"""M2, the instrument effect: what measuring this run cost the run.

No competitor has published an overhead number of any kind. That is the whole opportunity and it is
also the whole risk: a monitor that changed the run it was watching produced a reading of a
perturbed system, and without a number nobody can tell whether it did. The tap already measures
what it costs, per call, and carries the result as `tap.InstrumentEffect`. What was missing is the
step that turns that into a reading with a unit, a budget term and a refusal when the conversion is
not available.

**The refusal is the interesting part.** `instrument.effect` is registered `per: step`. The tap can
time the callable it wraps and nothing else: it does not know what a step costs, or how many grader
calls a step makes. So the number it holds is per call, the registered quantity is per step, and the
conversion factor is a property of the run rather than of the unit. That is exactly the condition
`UNIT_MISMATCH` names, and converting silently would be the same class of error as comparing a
per-token KL against a per-sequence one. Supply a `StepBasis` and the reading is produced; do not,
and the refusal names what a `StepBasis` is and where to get one.

**The per-step number is a difference of two snapshots.** The guard's counters are cumulative with
rolling quantiles, so the effect of one step is the difference between the snapshot at its start and
the snapshot at its end, which is what `per_step` computes. Dividing a cumulative total by a step
count instead gives the mean over the whole run, which is a different and usually smaller number on
any run whose tap warmed up or was disabled partway.

**The unit is two units and the registry says so.** `instrument.effect` prints `ms, bytes` and
carries `dimension: OPEN` for that reason. Both halves are measured
here and reported separately rather than combined into a composite nobody can interpret, and the
time half is what composes into an uncertainty budget, because it is the half that has a stated
sensitivity: `InstrumentEffect.as_term` turns the p99 fraction of grader time into a Type B
rectangular term.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from reward_lens.core.budget import BudgetTerm
from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import (
    BaselineID,
    BiasStatement,
    CostModel,
    EstimatorEntry,
    register_estimator,
)
from reward_lens.core.reading import Refusal, RefusalReason, refuse_incomplete
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.meta._base import MetaInstrument

#: A tap on the grader, and the record it writes. Both are needed: the counters live in the guard
#: and the step accounting lives in the record.
EFFECT_ACCESS: dict[Component, Access] = {
    Component.GRADER: Access.RECORD,
    Component.RECORD: Access.RECORD,
}

#: What an overhead reading is being compared against. The first is what every uninstrumented run
#: implicitly claims; the second is the same run's own step time, which is the denominator that
#: turns a duration into a judgement.
EFFECT_BASELINES: tuple[BaselineID, ...] = (
    "baseline.uninstrumented_run",
    "baseline.step_wall_time",
)

EFFECT_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "a paired difference between two snapshots of one process taken at two step boundaries. It "
        "counts elapsed nanoseconds and resident bytes and asserts nothing about the optimisation, "
        "so no regime of the run can make the difference wrong. What it cannot see is machine load "
        "outside the process, which is why the p99 travels beside the median on every reading."
    ),
)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepBasis:
    """What converts the tap's per-call numbers into the per-step unit the registry uses.

    ``calls`` is how many wrapped calls happened in the window, ``steps`` how many optimisation
    steps, and ``wall_seconds`` how long those steps took in total. The last is optional and is what
    turns an absolute overhead into a fraction of the run, which is the number anyone actually wants
    to know.
    """

    steps: int
    calls: int
    wall_seconds: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError(f"a step basis needs at least one step; got {self.steps}")
        if self.calls < 0:
            raise ValueError(f"call count cannot be negative; got {self.calls}")
        if self.wall_seconds is not None and self.wall_seconds <= 0:
            raise ValueError(
                f"wall_seconds must be positive when supplied; got {self.wall_seconds}"
            )

    @property
    def calls_per_step(self) -> float:
        return self.calls / self.steps

    @property
    def seconds_per_step(self) -> float | None:
        return None if self.wall_seconds is None else self.wall_seconds / self.steps


def per_step(before: Any, after: Any, basis: StepBasis) -> "StepDelta":
    """The effect of the window between two snapshots, which is how a per-step rate is got.

    The guard's counters are cumulative, so subtracting two snapshots is the only way to get the
    cost of a bounded window. A cumulative total divided by a step count is the mean over the whole
    run and is a different number whenever the tap warmed up, was disabled, or was added late.
    """
    calls = int(after.calls) - int(before.calls)
    if calls < 0:
        raise ValueError(
            f"the later snapshot has fewer calls ({after.calls}) than the earlier one "
            f"({before.calls}); the two are not from one guard in order"
        )
    return StepDelta(
        calls=calls,
        added_ns=int(after.added_ns_total) - int(before.added_ns_total),
        inner_ns=int(after.inner_ns_total) - int(before.inner_ns_total),
        basis=basis,
    )


@dataclass(frozen=True)
class StepDelta:
    """The difference between two snapshots, over a stated number of steps."""

    calls: int
    added_ns: int
    inner_ns: int
    basis: StepBasis

    @property
    def added_ms_per_step(self) -> float:
        return (self.added_ns / self.basis.steps) / 1e6


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class Overhead:
    """What the instrument cost, per step, in both of the units the registry names."""

    tap: str
    steps: int
    calls: int
    calls_per_step: float
    added_ms_per_step: float
    added_ms_per_call_p50: float
    added_ms_per_call_p99: float
    fraction_of_grader_time_mean: float
    fraction_of_grader_time_p99: float
    fraction_of_step_time: float | None
    resident_bytes: int
    added_alloc_bytes_per_step: float | None
    window_n: int
    enabled: bool
    recorder_exceptions: int
    unchecked: tuple[str, ...]
    baselines: Mapping[str, float] = field(default_factory=dict)

    def as_term(self, name: str = "instrument_overhead", *, sensitivity: float = 1.0) -> BudgetTerm:
        """The overhead as a Type B rectangular term, from the p99 fraction of grader time.

        Type B because the perturbation is an uncorrected bias: the tap cannot give back the time it
        took, so the GUM treatment is a bound divided by root three rather than a standard
        deviation. The sensitivity is the caller's to state, because how much a given reading moves
        per unit of timing perturbation is a property of that reading and not of the tap.
        """
        return BudgetTerm.from_half_width(
            name,
            half_width=self.fraction_of_grader_time_p99,
            distribution="rectangular",
            sensitivity=sensitivity,
            note=(
                f"p99 of the per-call fraction of grader time added by {self.tap}, over "
                f"{self.window_n:,} sampled calls"
            ),
        )

    def says(self) -> str:
        head = (
            f"Instrumenting {self.tap} cost {self.added_ms_per_step:.4g} ms per step over "
            f"{self.steps:,} steps at {self.calls_per_step:.1f} grader calls each, which is "
            f"{self.fraction_of_grader_time_mean:.2%} of the time spent inside the grader itself "
            f"and {self.added_ms_per_call_p99:.4g} ms at the p99 call."
        )
        if self.fraction_of_step_time is not None:
            head += f" That is {self.fraction_of_step_time:.3%} of a step."
        if self.added_alloc_bytes_per_step is not None:
            head += f" It allocated {self.added_alloc_bytes_per_step:,.0f} bytes per step."
        head += f" It holds {self.resident_bytes:,} bytes resident."
        if self.unchecked:
            head += (
                f" Not established: {', '.join(self.unchecked)}. Those are unmeasured rather than "
                f"zero."
            )
        return head


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


class InstrumentEffectReading(MetaInstrument):
    """M2. The overhead this measurement imposed, per step, as a term in the uncertainty budget.

    Takes either a single `tap.InstrumentEffect` snapshot with a `StepBasis`, or a pair of snapshots
    with the basis for the window between them, which is the better of the two and the one `per_step`
    exists for.
    """

    name = "InstrumentEffect"
    version = "1.0"
    quantity = "instrument.effect"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    requires = EFFECT_ACCESS
    substrates = frozenset(
        {
            Substrate.NEURAL_SCALAR,
            Substrate.NEURAL_GEN,
            Substrate.PROGRAM,
            Substrate.PROCEDURAL,
            Substrate.HUMAN,
            Substrate.COMPOSITE,
        }
    )
    phases = frozenset({Phase.IN_RUN, Phase.POST_RUN})
    envelope = EFFECT_ENVELOPE
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = EFFECT_BASELINES
    rung = 0
    faithful_to = "M2"
    deviations = (
        "the fraction reported is added time over the *grader's* own time, not over step time, "
        "because the tap can only see the callable it wraps. It is an upper bound on the "
        "perturbation to the run and errs conservatively. `fraction_of_step_time` is reported "
        "instead whenever the step basis carries wall-clock seconds, and is the number to quote",
        "a cheap grader reports a large fraction, and that is a true statement about that setup "
        "rather than an artefact: wrapping a callable that does one comparison, the tap is most of "
        "the call. The reading carries the grader's own time so a reader can see which case they "
        "are in",
    )

    def __init__(
        self,
        effect: Any = None,
        basis: StepBasis | None = None,
        *,
        before: Any = None,
        after: Any = None,
    ) -> None:
        if before is not None and after is not None:
            self.effect = after
            self.window = per_step(before, after, basis) if basis is not None else None
        else:
            self.effect = effect
            self.window = None
        self.basis = basis

    def compute(self) -> Any:
        effect = self.effect
        if effect is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no instrument effect was supplied, so there is no overhead to report",
                remedy=(
                    "wrap the grader with `reward_lens.tap.instrument_grader` and pass "
                    "`wrapped.effect()`. The wrapper carries the counters; nothing else in the "
                    "process knows what the tap cost. Two snapshots and `before=`/`after=` give the "
                    "per-step number directly."
                ),
                statistics={"calls": 0},
            )
        if int(getattr(effect, "calls", 0)) <= 0:
            return refuse_incomplete(
                self.name,
                field="any recorded call",
                subject=f"the tap {getattr(effect, 'tap_name', 'unnamed')!r}",
                remedy=(
                    "run the wrapped grader at least once before asking what it cost. A tap that "
                    "was installed and never invoked has an overhead of zero for a reason that is "
                    "not a property of the tap, and reporting it as a measurement would put a zero "
                    "into a budget where an unmeasured term belongs."
                ),
                calls=int(getattr(effect, "calls", 0)),
            )
        if self.basis is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.UNIT_MISMATCH,
                detail=(
                    f"instrument.effect is registered per step and the tap measured "
                    f"{effect.calls:,} calls. A call is not a step, and the factor between them is "
                    f"how many grader calls a step makes, which is a property of the run rather "
                    f"than of the unit, so it is not applied silently"
                ),
                remedy=(
                    "pass `basis=StepBasis(steps=..., calls=..., wall_seconds=...)`. The step count "
                    "and the call count are both in the training loop; the wall time is optional "
                    "and is what turns the overhead into a fraction of a step, which is the number "
                    "worth quoting."
                ),
                statistics={"calls": int(effect.calls), "basis": None},
            )

        basis = self.basis
        window = self.window
        added_ns = window.added_ns if window is not None else int(effect.added_ns_total)
        inner_ns = window.inner_ns if window is not None else int(effect.inner_ns_total)
        calls = window.calls if window is not None else int(effect.calls)
        seconds_per_step = basis.seconds_per_step
        alloc = getattr(effect, "added_alloc_bytes_this_step", None)
        return Overhead(
            tap=str(getattr(effect, "tap_name", "")),
            steps=basis.steps,
            calls=calls,
            calls_per_step=calls / basis.steps,
            added_ms_per_step=(added_ns / basis.steps) / 1e6,
            added_ms_per_call_p50=float(effect.added_ns_p50) / 1e6,
            added_ms_per_call_p99=float(effect.added_ns_p99) / 1e6,
            fraction_of_grader_time_mean=(added_ns / inner_ns) if inner_ns > 0 else float("nan"),
            fraction_of_grader_time_p99=float(effect.fraction_p99),
            fraction_of_step_time=(
                None
                if seconds_per_step is None
                else (added_ns / basis.steps) / (seconds_per_step * 1e9)
            ),
            resident_bytes=int(effect.resident_bytes),
            added_alloc_bytes_per_step=(None if alloc is None else float(alloc)),
            window_n=int(getattr(effect, "window_n", 0)),
            enabled=bool(getattr(effect, "enabled", True)),
            recorder_exceptions=int(getattr(effect, "recorder_exceptions", 0)),
            unchecked=tuple(getattr(effect, "unchecked", ())),
            baselines={
                # An uninstrumented run adds nothing, which is what every run without one of these
                # numbers is implicitly claiming about itself.
                "baseline.uninstrumented_run": 0.0,
                # The denominator that turns a duration into a judgement, in milliseconds.
                "baseline.step_wall_time": (
                    float("nan") if seconds_per_step is None else seconds_per_step * 1e3
                ),
            },
        )

    def uncertainty(self, computed: Overhead) -> Uncertainty | None:
        """The p50-to-p99 spread of the per-call cost, scaled to a step. Not a confidence interval.

        A latency distribution is right-skewed and its mean is not the number that hurts, so the
        interval quoted here is the observed spread of the per-call cost rather than an interval on
        its mean. `method` says which it is, because a reader who assumes the other one will
        conclude that the overhead is known far more precisely than it is.
        """
        return Uncertainty(
            ci_low=computed.added_ms_per_call_p50 * computed.calls_per_step,
            ci_high=computed.added_ms_per_call_p99 * computed.calls_per_step,
            ci_level=0.99,
            n=computed.window_n or computed.calls,
            method="observed p50 to p99 of the per-call cost, scaled to one step; not an interval on a mean",
        )

    def payload(self, computed: Overhead) -> dict[str, Any]:
        term = computed.as_term()
        return {
            "tap": computed.tap,
            "steps": computed.steps,
            "calls": computed.calls,
            "calls_per_step": computed.calls_per_step,
            "added_ms_per_step": computed.added_ms_per_step,
            "added_ms_per_call_p50": computed.added_ms_per_call_p50,
            "added_ms_per_call_p99": computed.added_ms_per_call_p99,
            "fraction_of_grader_time_mean": computed.fraction_of_grader_time_mean,
            "fraction_of_grader_time_p99": computed.fraction_of_grader_time_p99,
            "fraction_of_step_time": computed.fraction_of_step_time,
            "resident_bytes": computed.resident_bytes,
            "added_alloc_bytes_per_step": computed.added_alloc_bytes_per_step,
            "window_n": computed.window_n,
            "enabled": computed.enabled,
            "recorder_exceptions": computed.recorder_exceptions,
            "unchecked": list(computed.unchecked),
            "budget_term": {
                "name": term.name,
                "value": term.value,
                "kind": term.kind,
                "distribution": term.distribution,
            },
            "baselines": dict(computed.baselines),
            "says": computed.says(),
        }


def register_ladder() -> list[str]:
    """Register M2's rungs. Not called at import, by design."""
    entries = [
        EstimatorEntry(
            quantity="instrument.effect",
            impl="m2.snapshot_difference",
            requires=EFFECT_ACCESS,
            envelope=EFFECT_ENVELOPE,
            rung=1,
            bias=BiasStatement(
                direction="downward",
                why=(
                    "the guard cannot time its own ring offer, which the tap's own documentation "
                    "puts at roughly 131 ns per call on its reference machine, so the measured "
                    "overhead is short by that much per call"
                ),
            ),
            cost=CostModel(note="two snapshots at step boundaries"),
            run=None,
        ),
        EstimatorEntry(
            quantity="instrument.effect",
            impl="m2.cumulative_over_steps",
            requires=EFFECT_ACCESS,
            envelope=EFFECT_ENVELOPE,
            rung=0,
            bias=BiasStatement(
                direction="downward",
                why=(
                    "a cumulative total divided by a step count is the mean over the whole run, so "
                    "it understates the cost of any window in which the tap was actually on "
                    "whenever the tap warmed up, was added late, or disabled itself partway"
                ),
            ),
            cost=CostModel(note="one snapshot"),
            run=None,
        ),
    ]
    for e in entries:
        register_estimator(e)
    return [e.impl for e in entries]


__all__ = [
    "EFFECT_ACCESS",
    "EFFECT_BASELINES",
    "EFFECT_ENVELOPE",
    "InstrumentEffectReading",
    "Overhead",
    "StepBasis",
    "StepDelta",
    "per_step",
    "register_ladder",
]

"""Preflight, compute, refuse or emit: the shape series I's four instruments share.

`Observable.measure` returns `Evidence` by contract and `Instrument.estimate` returns `Reading`,
which is `Evidence | Refusal`. Every instrument here decides to refuse partway through the
arithmetic rather than in preflight, because none of them can know whether it will refuse until it
has looked at the running variable: I1 cannot know whether the density has support on both sides of
the cutoff until it has binned it, and I5 cannot know whether the outcome series has a transition
until it has tried to fit one. So the seam sits here, once, and a subclass writes `compute`.

`measure.estimator._base.EstimatorInstrument`, `measure.controls._base.ControlInstrument` and
`measure.frontier._base.FrontierInstrument` have the same shape for the same reason. None is
imported: they are private modules of other packages, and one instrument family reaching into
another's underscore is a dependency that is invisible at the point where it breaks. The duplicated
part is thirty lines of dispatch.

Every instrument in this package declares `Capability.NONE` and reads a record through the access
matrix, so `run()`'s `CapabilityError` branch is unreachable from `estimate`.
"""

from __future__ import annotations

from typing import Any

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Evidence
from reward_lens.core.gates import require_frame_for_comparison
from reward_lens.core.quantity import CostModel
from reward_lens.core.reading import Reading, Refusal
from reward_lens.core.types import Access, Component, Phase, Substrate
from reward_lens.measure.base import BaseObservable, Context, run
from reward_lens.measure.rate.regime import MEASURED_BY

#: Every substrate. A threshold is a rule applied to a recorded number, and the rule does not care
#: what kind of object produced the number it is applied to. A program verifier with a token budget
#: and a generative judge with one are the same gate.
ALL_SUBSTRATES = frozenset(
    {
        Substrate.NEURAL_SCALAR,
        Substrate.NEURAL_GEN,
        Substrate.PROGRAM,
        Substrate.PROCEDURAL,
        Substrate.HUMAN,
        Substrate.COMPOSITE,
    }
)

#: What every instrument in this series needs. `Access.RECORD` is "read logged values that already
#: exist", which is all of it: nothing here calls a grader, runs a policy or differentiates
#: anything. I3 additionally needs the composition tree, which lives on the record under
#: `Trajectory.scores`, so it is the same component at the same rung.
RECORD_ACCESS: dict[Component, Access] = {Component.RECORD: Access.RECORD}

#: The gate instruments read a per-rollout running variable and a per-rollout score, both of which
#: are written by the grader wrap, so the grader is named as a second component at RECORD.
GATE_ACCESS: dict[Component, Access] = {
    Component.RECORD: Access.RECORD,
    Component.GRADER: Access.RECORD,
}

#: Reading a record costs the time to walk it. No calls, no GPU.
FREE_READ = CostModel(note="free; reads per-rollout numbers already on the record")

#: The phase a threshold question can be asked in. The record has to exist, so PRE_RUN is out; a
#: deployed artifact has no rollout record, so DEPLOYED is out.
RECORD_PHASES = frozenset({Phase.IN_RUN, Phase.POST_RUN})


#: What I1 and I2 need to be true, not merely available.
#:
#: `STATIONARY_GRADER` is the one that bites. McCrary's identifying assumption is that the
#: counterfactual density of the running variable is continuous at the cutoff. If the gate moved
#: inside the measurement window, the pooled density is a mixture over two cutoffs: mass that
#: bunched at the earlier one sits inside the bandwidth of the later one, and the test can report a
#: discontinuity at a place where none exists at either cutoff. That is the direction that
#: manufactures a finding rather than the direction that hides one.
#:
#: `on_violation` is "downgrade" rather than "refuse", and the choice is deliberate. A drifting
#: grader does not make the density uncomputable, and refusing on it would withhold the reading on
#: exactly the runs where a gate is most likely to have been retuned, which is refusing where the
#: quantity is still defined. The quantity stays defined, its trust caps at EXPLORATORY, and the
#: violated condition is recorded on the reading.
GATE_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
    measured_by={RegimeCondition.STATIONARY_GRADER: MEASURED_BY[RegimeCondition.STATIONARY_GRADER]},
    on_violation="downgrade",
)

#: What I5 needs. Within-group reward variance is undefined when every group is degenerate, and it
#: means something different when the grader changed under it: a grader retune moves the variance
#: for a reason that has nothing to do with the policy's behaviour, which is precisely the change a
#: derivative detector is designed to alarm on.
#:
#: Both are "downgrade" for the same reason as `GATE_ENVELOPE`, with one exception handled in the
#: arithmetic rather than in the envelope: a run in which *every* group is degenerate has a
#: within-group variance that is identically zero, and its derivative is not a weak reading but an
#: undefined one. That case refuses in `compute`, on the measured fraction being exactly 1.0 rather
#: than on a threshold somebody picked.
VARIANCE_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.GROUP_NONDEGENERATE, RegimeCondition.STATIONARY_GRADER}),
    measured_by={
        RegimeCondition.GROUP_NONDEGENERATE: MEASURED_BY[RegimeCondition.GROUP_NONDEGENERATE],
        RegimeCondition.STATIONARY_GRADER: MEASURED_BY[RegimeCondition.STATIONARY_GRADER],
    },
    on_violation="downgrade",
)

#: What I3 needs, which is nothing. The dead-zone fraction counts which recorded rollouts fall
#: inside a recorded override's region, by re-evaluating leaves that are already on the record. No
#: regime can make a count of recorded rollouts wrong.
#:
#: What a regime can make wrong is what the count *means*, and that is not an envelope's job. The
#: two controls this instrument runs are where it is checked: the graded-penalty contrast, which
#: says whether the suppression is the override's or the penalty's, and the decode-length report,
#: which says whether the region was drawn by the training budget or by the sampler.
DEADZONE_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "a census over a record: it counts which recorded rollouts fall inside a recorded "
        "override's region and re-evaluates leaves that are already there, calling nothing, so no "
        "regime can make the count wrong."
    ),
)


class ThresholdInstrument(BaseObservable):
    """Series I's runner: preflight, compute once, return the refusal or emit the Evidence."""

    #: Set by `estimate` for the duration of one call so `measure` does not recompute.
    _computed: Any = None

    def compute(self) -> Any:  # pragma: no cover - abstract
        """The instrument's own arithmetic, with no `Context`. Returns a payload or a `Refusal`.

        Written without a Context on purpose. Every reading in this series is a pure function of
        numbers the caller already holds, which is what makes the whole series testable against a
        density whose discontinuity you put there yourself.
        """
        raise NotImplementedError

    def gated_emit(self, ctx: Context, computed: Any) -> Evidence:
        """Hand a computed payload to the runner, or apply the runner's gates by hand.

        `run` resolves `ctx.signal.caps` to enforce the capability check, and nothing here touches
        a signal. The no-signal branch does what `run` would do minus the check that has nothing to
        check against. Gate 2 is applied in both branches because it depends on the instrument's
        gauge status and the context's frame rather than on the signal.

        **``ctx._observable`` is set around both branches, and it is the whole reason this is not a
        two-line method.** `Context.emit` reads the observable's name, version, gauge status and
        quantity off `ctx._observable`, and `run()` is the only place in the kernel that sets it. A
        no-signal branch that calls `measure` directly therefore emits `observable="anonymous"`,
        `observable_version="0"`, `gauge=INVARIANT` and `quantity=""`, whatever the instrument
        declared. That is a known defect arriving by a different road: `emit` was made to forward
        the quantity, and the path every record-only instrument actually takes never gives it one
        to forward. Measured on the shipped `ClipAccounting`, which declares
        `estimator.clip_fraction_effect` and emits `observable='anonymous', quantity=''`. Reported
        for the three sibling runners in `estimator/`, `controls/` and `frontier/`, which have the
        same branch; fixed here.
        """
        self._computed = computed
        previous = ctx._observable
        ctx._observable = self
        try:
            if ctx.signal is not None:
                return run(self, ctx)
            if ctx.is_comparison:
                require_frame_for_comparison(self.gauge_status, ctx.frame)
            return self.measure(ctx)
        finally:
            ctx._observable = previous
            self._computed = None

    def estimate(self, ctx: Context) -> Reading:
        """Preflight, compute, refuse or emit.

        The violated condition reaches the reading through `Context.emit`, which forwards
        ``ctx.regime_reading`` onto the Evidence, so a number produced outside `GATE_ENVELOPE`
        carries the reading that says so. What does **not** reach it is `PreflightResult.trust_cap`:
        `BaseObservable.preflight` computes the cap for `on_violation="downgrade"` and nothing in
        the library applies it, so the emitted trust is whatever `compute_trust` decides from
        calibration and registration alone. Grepped across the tree, the only consumer is
        `measure.card.card._envelope_note`, which renders a sentence about it. Recorded here rather
        than worked around, because capping a trust level from inside an instrument would mean
        bypassing `make_evidence`, which is the one place that decides it.
        """
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        out = self.compute()
        if isinstance(out, Refusal):
            return out
        return self.gated_emit(ctx, out)

    def measure(self, ctx: Context) -> Evidence:
        out = self._computed if self._computed is not None else self.compute()
        if isinstance(out, Refusal):
            raise ValueError(
                f"{self.name}.measure was called on a measurement that declines to produce "
                f"Evidence: {out.reason.name}. Call `estimate`, which returns the refusal as a "
                f"value carrying its remedy."
            )
        return ctx.emit(out)


__all__ = [
    "ALL_SUBSTRATES",
    "DEADZONE_ENVELOPE",
    "FREE_READ",
    "GATE_ACCESS",
    "GATE_ENVELOPE",
    "RECORD_ACCESS",
    "RECORD_PHASES",
    "ThresholdInstrument",
    "VARIANCE_ENVELOPE",
]

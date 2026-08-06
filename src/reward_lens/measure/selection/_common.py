"""Shared declarations for series C's white-box instruments: access, envelopes, emit.

Four things every instrument in this package needs and none of them belongs in six copies.

`SELECTION_SUBSTRATES` and the two access matrices are the instrument declarations that genuinely
are the same across the package. Anything with a plausible default is left to the subclass, because
a plausible default is indistinguishable from a decision and `lint_instrument` is what turns an
undeclared field into a finding.

`emit_white_box` is the one piece of real machinery. An `IncrementalValidity` record is mandatory
on every white-box reading and `lint_reading` enforces it, so an instrument in this package that
forgets one does not merge. `Context.emit` takes the record directly, so this is a thin wrapper
whose whole job is to make forgetting hard: it takes the record as a required argument rather than
an optional one.
"""

from __future__ import annotations

from typing import Any, Mapping

from reward_lens.core.budget import IncrementalValidity
from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Evidence, Uncertainty
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Access, AccessMatrix, Component, GaugeStatus, Phase, Substrate
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.rate.regime import MEASURED_BY

#: Reading a planted organism's internals and being able to make another plant. C3's access line is
#: "ORGANISM:MUTATE, a certified reference", and an organism is the `GOLD` component: the answer key
#: and the thing that produced it are the same object when the organism *is* the ground truth.
ACCESS_ORGANISM_MUTATE: AccessMatrix = {
    Component.GOLD: Access.MUTATE,
    Component.GRADER: Access.FORWARD,
}

#: C4's line, verbatim: GRADER:MUTATE. An erasure edits the grader's forward pass, and `MUTATE` is
#: the flag that says so.
ACCESS_GRADER_MUTATE: AccessMatrix = {Component.GRADER: Access.MUTATE}

#: C8's line: GRADER:FORWARD plus a fitted lens. `FORWARD` is "run it and read internal activations",
#: which is exactly what a lens does and is strictly less than `MUTATE`, so C8 is the one instrument
#: in this package that needs no write access to anything.
ACCESS_GRADER_FORWARD: AccessMatrix = {Component.GRADER: Access.FORWARD}

#: C6 and C7's line: POLICY:MUTATE. An ablation and its rescue both edit a forward pass.
ACCESS_POLICY_MUTATE: AccessMatrix = {Component.POLICY: Access.MUTATE}

#: C5's line: POLICY:MUTATE + CONTROL. `CONTROL` is "stand up a counterfactual arm of the whole
#: loop", which is what continued training after an ablation is, and it is the flag that makes C5
#: compute-gated rather than merely expensive.
ACCESS_POLICY_MUTATE_CONTROL: AccessMatrix = {
    Component.POLICY: Access.MUTATE | Access.CONTROL,
}

#: A network with activations to read. The other four substrates have none, so declaring them would
#: be a claim about objects this package cannot point at.
SELECTION_SUBSTRATES: frozenset[Substrate] = frozenset(
    {Substrate.NEURAL_SCALAR, Substrate.NEURAL_GEN}
)

#: Before a run or after one. None of these is an in-run measurement: every one of them intervenes
#: on a checkpoint and reads what changed, which is a question you ask of a model rather than of a
#: run in flight.
SELECTION_PHASES: frozenset[Phase] = frozenset({Phase.PRE_RUN, Phase.POST_RUN})

#: An intervention effect smaller than the substrate's disagreement with itself is not attributable
#: to the intervention. M1 measures the floor; this is the envelope that consults it.
ABOVE_LOD_ONLY = EnvelopeSpec(
    requires=frozenset({RegimeCondition.ABOVE_LOD}),
    measured_by={RegimeCondition.ABOVE_LOD: MEASURED_BY[RegimeCondition.ABOVE_LOD]},
    on_violation="refuse",
)


class SelectionInstrument(BaseObservable):
    """The declarations series C shares, so six instruments do not restate them.

    Exactly three: the substrates, the phases and the gauge status. Everything else differs per
    instrument and is left undeclared here on purpose, so that a subclass which forgets one gets a
    lint finding rather than inheriting something that looked reasonable.
    """

    substrates = SELECTION_SUBSTRATES
    phases = SELECTION_PHASES
    gauge_status = GaugeStatus.INVARIANT


def emit_white_box(
    ctx: Context,
    value: Any,
    *,
    incremental: IncrementalValidity,
    baselines: Mapping[str, float],
    uncertainty: Uncertainty | None = None,
    reference: Any = None,
    subject_extra: dict[str, Any] | None = None,
) -> Evidence:
    """`Context.emit` with the two fields this package must never omit, made required.

    ``incremental`` is positional-only in spirit: the record is mandatory on every white-box
    reading, `lint_reading` fails the instrument without one, and the way to stop that being a
    recurring review comment is to make the argument impossible to leave out. ``baselines``
    is the same rule one lint rule earlier: a claim with no dumb baseline is not a claim.

    ``reference`` is forwarded because C3 calibrates against a reference material and the trust cap
    for an uncertified one lives inside the Evidence content id, so it cannot be applied afterwards.
    """
    return ctx.emit(
        value,
        uncertainty=uncertainty,
        baselines=dict(baselines),
        incremental=incremental,
        reference=reference,
        subject_extra=subject_extra,
    )


def refuse_unmeasured_control(
    instrument: str, *, what: str, remedy: str, **statistics: Any
) -> Refusal:
    """`NO_MATCHED_CONTROL`: a control that was never run is not a control that passed.

    The distinction this carries is the one series C exists to enforce. A knockout with no rescue, a
    single dissociation, an acute effect with no chronic arm and an ablation with no placebo are all
    the same shape of mistake: an experiment whose missing arm would have been the one that could
    have contradicted it. Refusing rather than reporting is what makes the missing arm visible.
    """
    return Refusal(
        instrument=instrument,
        reason=RefusalReason.NO_MATCHED_CONTROL,
        detail=(
            f"{what}. An experiment whose control arm was never run cannot be distinguished from "
            f"one whose control arm would have failed, and the reading would carry a confidence "
            f"the design does not support."
        ),
        remedy=remedy,
        statistics=dict(statistics),
    )


__all__ = [
    "ABOVE_LOD_ONLY",
    "ACCESS_GRADER_FORWARD",
    "ACCESS_GRADER_MUTATE",
    "ACCESS_ORGANISM_MUTATE",
    "ACCESS_POLICY_MUTATE",
    "ACCESS_POLICY_MUTATE_CONTROL",
    "SELECTION_PHASES",
    "SELECTION_SUBSTRATES",
    "SelectionInstrument",
    "emit_white_box",
    "refuse_unmeasured_control",
]

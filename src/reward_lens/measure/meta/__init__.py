"""The meta-instruments: M1, M2, M7, M8, M9 and M11, the six that read the library's own readings.

Every other instrument in the catalogue reads a grader, a record or a policy. These six read a
finished reading, a ladder, or a set of instruments, which makes them the ones that keep the rest
honest and gives them a failure mode of their own: **a meta-instrument reporting a clean number
about a measurement nobody made is worse than one that refuses.** Every refusal below exists because
of that.

    M1   SubstrateNoiseFloor         what the apparatus disagrees with itself by, and the LOD and LOQ
    M2   InstrumentEffectReading     what measuring the run cost the run, per step
    M7   UncertaintyBudgetReading    the GUM table, composed and linted, with the largest term named
    M8   InterlaboratoryComparison   s_L and the Birge ratio, and no comparison without a control
    M9   IncrementalValidityReading  what an instrument adds over the ones already run
    M11  RungDisagreement            two rungs of one ladder disagreeing, published rather than resolved

Three of the six compose into each other and that is the point of shipping them together. M1's
noise floor is a Type B term in M7's budget. M11's rung disagreement is a `Transfer` term in the
same budget. M2's overhead is a third. A reading whose budget carries all three can say which of
the apparatus, the estimator ladder and the instrumentation is responsible for its uncertainty,
which is a sentence nothing in this field currently produces.

**Four decisions were open in the catalogue and are recorded rather than left implicit.** M7's and
M8's quantity lists were `OPEN` and are argued in `meta/quantities.py`; every other `OPEN` field on
these six records is filled by the declarations on the instrument classes, which is where a reader
should look for what was decided. `quantities.as_yaml_rows()` emits the amendment for
`spec/QUANTITIES.yaml`, which this package does not write to.

**What is deliberately absent.** `RungDisagreement` has no field that picks a winner: no
`preferred`, no `resolved_value`, no rule that prefers the higher rung. Two rungs disagreeing is
the reading, and an instrument that resolved it would be throwing away the measurement to keep an
estimate.
"""

from __future__ import annotations

from reward_lens.measure.meta._base import MetaInstrument
from reward_lens.measure.meta.effect import (
    EFFECT_ACCESS,
    EFFECT_BASELINES,
    EFFECT_ENVELOPE,
    InstrumentEffectReading,
    Overhead,
    StepBasis,
    StepDelta,
    per_step,
)
from reward_lens.measure.meta.effect import register_ladder as register_effect_ladder
from reward_lens.measure.meta.floor import (
    MIN_BLANK_REPLICATES,
    MIN_SWEEP_POINTS,
    NOISE_FLOOR_ACCESS,
    NOISE_FLOOR_BASELINES,
    NOISE_FLOOR_ENVELOPE,
    BlankReplicates,
    DoseSweep,
    HillFit,
    NoiseFloor,
    SubstrateNoiseFloor,
    fit_hill,
    limits_from,
    two_arm_blanks,
    verdict_for,
)
from reward_lens.measure.meta.floor import register_ladder as register_floor_ladder
from reward_lens.measure.meta.gum import (
    BUDGET_ACCESS,
    BUDGET_BASELINES,
    BUDGET_ENVELOPE,
    BudgetAudit,
    BudgetFinding,
    UncertaintyBudgetReading,
    compose,
    lint_budget,
)
from reward_lens.measure.meta.incremental import (
    INCREMENTAL_ACCESS,
    INCREMENTAL_BASELINES,
    INCREMENTAL_ENVELOPE,
    Combiner,
    Detector,
    Increment,
    IncrementalValidityReading,
    mean_margin,
    phi,
    standardised_margin,
)
from reward_lens.measure.meta.incremental import register_ladder as register_incremental_ladder
from reward_lens.measure.meta.interlab import (
    CONTROL_N_TOLERANCE,
    INTERLAB_ACCESS,
    INTERLAB_BASELINES,
    INTERLAB_ENVELOPE,
    MIN_LABS,
    ControlPanel,
    Interlaboratory,
    InterlaboratoryComparison,
    Lab,
    bootstrap_control,
)
from reward_lens.measure.meta.interlab import register_ladder as register_interlab_ladder
from reward_lens.measure.meta.quantities import (
    DECIDED,
    DEFINITIONS,
    INSTRUMENT_LISTS,
    as_yaml_rows,
    definition_of,
)
from reward_lens.measure.meta.rungs import (
    RUNG_ACCESS,
    RUNG_BASELINES,
    RUNG_ENVELOPE,
    Disagreement,
    RungDisagreement,
    RungReading,
    compare_rungs,
    rung_from_effective_size,
)
from reward_lens.measure.meta.rungs import register_ladder as register_rung_ladder

#: The six instruments of this package, so a test or a registry can enumerate them.
META = (
    SubstrateNoiseFloor,
    InstrumentEffectReading,
    UncertaintyBudgetReading,
    InterlaboratoryComparison,
    IncrementalValidityReading,
    RungDisagreement,
)

#: Which catalogue record each class is, for a reader holding the catalogue rather than the code.
CATALOGUE_IDS: dict[str, str] = {
    "SubstrateNoiseFloor": "M1",
    "InstrumentEffect": "M2",
    "UncertaintyBudget": "M7",
    "InterlaboratoryComparison": "M8",
    "IncrementalValidity": "M9",
    "RungDisagreement": "M11",
}


def instances() -> tuple[MetaInstrument, ...]:
    """One instance of each, carrying the minimum each needs to be lint-clean.

    Only M7 needs an argument, and it needs one for a reason rather than for convenience: its
    quantity is its subject's, so an M7 with no subject genuinely has no quantity and `lint_instrument`
    is right to say so. Enumerating the six for a lint test therefore has to say what each budget is
    a budget of, and this is the shortest honest way to do that.
    """
    return (
        SubstrateNoiseFloor(),
        InstrumentEffectReading(),
        UncertaintyBudgetReading(quantity_id="grader.effective_group_size"),
        InterlaboratoryComparison(),
        IncrementalValidityReading(),
        RungDisagreement(),
    )


def register_ladders() -> list[str]:
    """Register every estimator this package specifies. Not called at import, by design.

    Some of the rungs registered here have no implementation, and that is the documented way to say
    that a quantity has a better estimator than the one built: `EstimatorEntry.run` stays None and
    the capability report names the rung and what it would need rather than pretending the ladder
    stops where this build does.
    """
    out: list[str] = []
    for register in (
        register_floor_ladder,
        register_effect_ladder,
        register_interlab_ladder,
        register_incremental_ladder,
        register_rung_ladder,
    ):
        out.extend(register())
    return out


__all__ = [
    "BUDGET_ACCESS",
    "BUDGET_BASELINES",
    "BUDGET_ENVELOPE",
    "CATALOGUE_IDS",
    "CONTROL_N_TOLERANCE",
    "DECIDED",
    "DEFINITIONS",
    "EFFECT_ACCESS",
    "EFFECT_BASELINES",
    "EFFECT_ENVELOPE",
    "INCREMENTAL_ACCESS",
    "INCREMENTAL_BASELINES",
    "INCREMENTAL_ENVELOPE",
    "INSTRUMENT_LISTS",
    "INTERLAB_ACCESS",
    "INTERLAB_BASELINES",
    "INTERLAB_ENVELOPE",
    "META",
    "MIN_BLANK_REPLICATES",
    "MIN_LABS",
    "MIN_SWEEP_POINTS",
    "NOISE_FLOOR_ACCESS",
    "NOISE_FLOOR_BASELINES",
    "NOISE_FLOOR_ENVELOPE",
    "RUNG_ACCESS",
    "RUNG_BASELINES",
    "RUNG_ENVELOPE",
    "BlankReplicates",
    "BudgetAudit",
    "BudgetFinding",
    "Combiner",
    "ControlPanel",
    "Detector",
    "Disagreement",
    "DoseSweep",
    "HillFit",
    "Increment",
    "IncrementalValidityReading",
    "InstrumentEffectReading",
    "Interlaboratory",
    "InterlaboratoryComparison",
    "Lab",
    "MetaInstrument",
    "NoiseFloor",
    "Overhead",
    "RungDisagreement",
    "RungReading",
    "StepBasis",
    "StepDelta",
    "SubstrateNoiseFloor",
    "UncertaintyBudgetReading",
    "as_yaml_rows",
    "bootstrap_control",
    "compare_rungs",
    "compose",
    "definition_of",
    "fit_hill",
    "instances",
    "limits_from",
    "lint_budget",
    "mean_margin",
    "per_step",
    "phi",
    "register_ladders",
    "rung_from_effective_size",
    "standardised_margin",
    "two_arm_blanks",
    "verdict_for",
]

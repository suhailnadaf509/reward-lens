"""Series C's white-box selection instruments: C3 through C8.

Six instruments and one shared idea: a claim about a direction is worth what its controls are worth.

    C3  InstrumentRecoveryTable   every localisation method against one planted key, losses included
    C4  ErasureCost               what an erasure removes, what it costs, and the dose window
    C5  AcuteChronic              the effect that survives continued training (compute-gated)
    C6  RescueFraction            put the ablated component back and check the behaviour returns
    C7  DoubleDissociation        necessity is not sufficiency, and one dissociation is not two
    C8  VerdictDirection          when the judge's verdict direction stopped moving

The estimators these are built on live in `reward_lens.policy.selection`, which takes arrays rather
than subjects so that the same code reads a grader and a policy without knowing which it has. The
admission protocol lives there too, as a gate that refuses rather than a checklist: a direction
carrying a claim must be decodable, used, unmatched by a dumb baseline, and unmatched by a coherent
irrelevant semantic direction, and a condition nobody measured is not a condition that passed.
"""

from __future__ import annotations

from reward_lens.measure.selection._common import (
    ABOVE_LOD_ONLY,
    ACCESS_GRADER_FORWARD,
    ACCESS_GRADER_MUTATE,
    ACCESS_ORGANISM_MUTATE,
    ACCESS_POLICY_MUTATE,
    ACCESS_POLICY_MUTATE_CONTROL,
    SelectionInstrument,
    emit_white_box,
    refuse_unmeasured_control,
)
from reward_lens.measure.selection.controls import (
    CHRONIC_COST_NOTE,
    AcuteChronic,
    AcuteChronicReading,
    Dissociation,
    DoubleDissociation,
    Rescue,
    RescueFraction,
    rescue_fraction,
)
from reward_lens.measure.selection.erasure import (
    NAMED_DIFFERENCES,
    PUBLISHED_ALTERNATIVE,
    BenchmarkFloor,
    ErasureCost,
    ErasureReading,
    Reconciliation,
    dose_eraser,
    reconcile,
    rewardbench2_floor,
    surgery_result,
)
from reward_lens.measure.selection.recovery import (
    RECOVERY_BASELINES,
    RECOVERY_ENVELOPE,
    InstrumentRecoveryTable,
    campaign_rows,
)
from reward_lens.measure.selection.table import (
    MEASURED,
    RecoveryRow,
    RecoveryTable,
    recovery_auc,
    score_row,
)
from reward_lens.measure.selection.transport import (
    SKIP_FIRST_N_POSITIONS,
    AveragedJacobianTransport,
    IdentityTransport,
    VerdictTransport,
    fit_averaged_jacobian,
    jlens_transport,
)
from reward_lens.measure.selection.verdict import (
    COMMITMENT_FRACTION,
    Commitment,
    Controls,
    VerdictDirection,
    VerdictReading,
    commitment,
    settles_at,
    verdict_direction,
)

#: The six instruments, so a battery can enumerate them without importing six modules.
SELECTION_INSTRUMENTS: tuple[type, ...] = (
    InstrumentRecoveryTable,
    ErasureCost,
    AcuteChronic,
    RescueFraction,
    DoubleDissociation,
    VerdictDirection,
)

__all__ = [
    "ABOVE_LOD_ONLY",
    "ACCESS_GRADER_FORWARD",
    "ACCESS_GRADER_MUTATE",
    "ACCESS_ORGANISM_MUTATE",
    "ACCESS_POLICY_MUTATE",
    "ACCESS_POLICY_MUTATE_CONTROL",
    "CHRONIC_COST_NOTE",
    "COMMITMENT_FRACTION",
    "MEASURED",
    "NAMED_DIFFERENCES",
    "PUBLISHED_ALTERNATIVE",
    "RECOVERY_BASELINES",
    "RECOVERY_ENVELOPE",
    "SELECTION_INSTRUMENTS",
    "SKIP_FIRST_N_POSITIONS",
    "AcuteChronic",
    "AcuteChronicReading",
    "AveragedJacobianTransport",
    "BenchmarkFloor",
    "Commitment",
    "Controls",
    "Dissociation",
    "DoubleDissociation",
    "ErasureCost",
    "ErasureReading",
    "IdentityTransport",
    "InstrumentRecoveryTable",
    "Reconciliation",
    "RecoveryRow",
    "RecoveryTable",
    "Rescue",
    "RescueFraction",
    "SelectionInstrument",
    "VerdictDirection",
    "VerdictReading",
    "VerdictTransport",
    "campaign_rows",
    "commitment",
    "dose_eraser",
    "emit_white_box",
    "fit_averaged_jacobian",
    "jlens_transport",
    "reconcile",
    "recovery_auc",
    "refuse_unmeasured_control",
    "rescue_fraction",
    "rewardbench2_floor",
    "score_row",
    "settles_at",
    "surgery_result",
    "verdict_direction",
]

"""The controls bank: M3, M4, M5 and M10, the four gates every other series ships against.

These four are here rather than at the end of Phase 3 for one reason. A baseline bank that arrives
after the claims it was supposed to gate is a baseline bank nobody used. Building them first is
what turns "every claim ships against six dumb baselines" from an instruction in a document into a
property of the codebase.

    M3  DumbBaselineBank        the six numbers a claim has to beat
    M4  SemanticPlacebo         a coherent irrelevant direction, on every steering claim
    M5  MatchedPositiveControl  a null with no identically-powered control refuses
    M10 PowerAndMDE             simulated power, the MDE, and q = N/N*, before the run

The three that gate rather than report do it with values, not exceptions. `gate_null` returns a
`Refusal`; `compare_to_placebo` returns one when the placebo arm is missing; `resolve_row` returns
one when `q < 1`. Each carries a remedy that is an instruction rather than a diagnosis.

**Two decisions in here were mine to propose and not to make, and they are recorded so a reviewer
can find them rather than discover them.**

`RefusalReason.NO_MATCHED_CONTROL` is defined for a null with no *positive* control. M4's missing
placebo is a missing *negative* control, which is the same failure in the other direction, and it
reuses the same reason because the fifteen members are a fixed contract and a
sixteenth is an amendment rather than an implementation detail. The refusal's detail says which
kind of control is missing, so nothing is ambiguous at the point of reading.

`resolve_row` refuses an unresolved leaderboard row with `BELOW_LOD`, on the argument that the
limit of detection and a minimum detectable effect are the same construction over different noise
sources. Both are open for a reviewer to overturn in favour of a new reason.
"""

from __future__ import annotations

from reward_lens.measure.controls._base import ControlInstrument
from reward_lens.measure.controls.baselines import BASELINE_ENVELOPE, CHANCE, DumbBaselineBank
from reward_lens.measure.controls.matched import (
    MATCHED_CONTROL_ENVELOPE,
    NOMINAL_POWER,
    ControlDesign,
    ControlVerdict,
    GuardedInstrument,
    MatchedControl,
    MatchedPositiveControl,
    NullClaim,
    default_null_extractor,
    gate_null,
    guard_null,
)
from reward_lens.measure.controls.placebo import (
    BANK_BY_ID,
    PLACEBO_BANK,
    PLACEBO_ENVELOPE,
    RANDOM_GAUSSIAN,
    InterventionArm,
    PlaceboComparison,
    PlaceboContrast,
    PlaceboDirection,
    SemanticPlacebo,
    compare_to_placebo,
    contrast,
    default_contrast,
    random_gaussian_direction,
    semantic_placebo,
)
from reward_lens.measure.controls.power import (
    CALCULATOR_BASELINES,
    POWER_ENVELOPE,
    PowerAndMDE,
    resolve_row,
)
from reward_lens.measure.controls.quantities import PROPOSED, as_yaml_rows, register_proposed

#: The four instruments of this package, so a test or a registry can enumerate them.
CONTROLS = (DumbBaselineBank, SemanticPlacebo, MatchedPositiveControl, PowerAndMDE)

__all__ = [
    "BANK_BY_ID",
    "BASELINE_ENVELOPE",
    "CALCULATOR_BASELINES",
    "CHANCE",
    "CONTROLS",
    "MATCHED_CONTROL_ENVELOPE",
    "NOMINAL_POWER",
    "PLACEBO_BANK",
    "PLACEBO_ENVELOPE",
    "POWER_ENVELOPE",
    "PROPOSED",
    "RANDOM_GAUSSIAN",
    "ControlDesign",
    "ControlInstrument",
    "ControlVerdict",
    "DumbBaselineBank",
    "GuardedInstrument",
    "InterventionArm",
    "MatchedControl",
    "MatchedPositiveControl",
    "NullClaim",
    "PlaceboComparison",
    "PlaceboContrast",
    "PlaceboDirection",
    "PowerAndMDE",
    "SemanticPlacebo",
    "as_yaml_rows",
    "compare_to_placebo",
    "contrast",
    "default_contrast",
    "default_null_extractor",
    "gate_null",
    "guard_null",
    "random_gaussian_direction",
    "register_proposed",
    "resolve_row",
    "semantic_placebo",
]

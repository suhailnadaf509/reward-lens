"""``reward_lens.measure.battery`` — the measurement battery.

The battery is v1's primitive corpus, ported behind the frozen ``Observable`` protocol so every
instrument declares what it requires (R3), how its value transforms under the gauge group, which
theory object or E-report object it instantiates (``faithful_to``), and where it departs from that
(``deviations``). Every Observable here runs on any ``RewardSignal`` that declares the required
capability, returns gated Evidence, and is checked for faithful reproduction of the v1 primitive it
ports (the E-parity suite) or, where it is a working port of a science the v1 corpus only sketched,
carries a property test on the tiny model.

Each module is one Observable. The four that carry the E-parity (``LensCrystallization``,
``DirectLinearAttribution``, ``PatchGrid``, ``MultiObjectiveGeometry``) are proven byte-for-byte
against the v1 primitives on the tiny model; the rest are working ports with property tests. Head- and
component-level attribution route through the one canonical implementation in
``reward_lens.attribution.dla``, and the patching mechanics live in ``reward_lens.interventions.patch``
as Interventions, so a measurement and an intervention never diverge in how they read the model.
"""

from __future__ import annotations

from reward_lens.core.extras import require_extra

require_extra("white-box", subsystem="reward_lens.measure.battery")

from reward_lens.measure.battery.bias import BiasBattery
from reward_lens.measure.battery.circuit import CircuitJaccard
from reward_lens.measure.battery.concept import ConceptDoseResponse
from reward_lens.measure.battery.conflict import ConflictMatrix
from reward_lens.measure.battery.dla import DirectLinearAttribution
from reward_lens.measure.battery.feature import FeatureRewardAlignment
from reward_lens.measure.battery.geometry import MultiObjectiveGeometry
from reward_lens.measure.battery.lens import LensCrystallization
from reward_lens.measure.battery.patch import PatchGrid
from reward_lens.measure.battery.path import PathEffect
from reward_lens.measure.battery.snr import PromptSNR

# The battery in a stable order, so a card or a scoreboard can enumerate it deterministically.
BATTERY = (
    LensCrystallization,
    DirectLinearAttribution,
    PatchGrid,
    PathEffect,
    ConceptDoseResponse,
    BiasBattery,
    PromptSNR,
    ConflictMatrix,
    CircuitJaccard,
    FeatureRewardAlignment,
    MultiObjectiveGeometry,
)

__all__ = [
    "LensCrystallization",
    "DirectLinearAttribution",
    "PatchGrid",
    "PathEffect",
    "ConceptDoseResponse",
    "BiasBattery",
    "PromptSNR",
    "ConflictMatrix",
    "CircuitJaccard",
    "FeatureRewardAlignment",
    "MultiObjectiveGeometry",
    "BATTERY",
]

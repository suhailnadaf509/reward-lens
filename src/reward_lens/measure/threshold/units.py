"""The `units` group's assertion, made executable for series I.

All four instruments here are registered under `units`, whose registered assertion is not a numeric
relation at all: a comparison across a unit boundary raises `UNIT_MISMATCH` rather than silently
converting. `check_invariance` routes that group to `check_unit_refusal`, which needs a comparison
to assert on, and this is it.

The pair that matters in this series is real rather than contrived, and it is the mistake this
package is most likely to invite. `gate.mccrary_statistic` is a z and `gate.bunching_elasticity` is
dimensionless, and both of them answer a question a reader will phrase as "how hard is the policy
pushing on this gate". Ranking one against the other, or reporting the larger of the two as the
worse gate, is the unit error that is the most common silent failure in this literature.
A z of 8 and an elasticity of 0.02 are not a large number and a small one, and there is no factor
between them: one says the density is discontinuous and the other says what the discontinuity
implies about behaviour.

The third case is sharper still and it is the one a length gate produces every time.
`gate.deadzone_fraction` is a share of rollouts and `run.variance_derivative` is a variance per
step, so they cannot be ranked either, and nothing in a record supplies the conversion.

The units are read out of the registry rather than restated here. A unit written down twice is a
unit that can disagree with itself, and the registry is the contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reward_lens.core.quantity import QUANTITIES, Unit
from reward_lens.core.reading import Refusal, RefusalReason


def unit_of(quantity: str) -> Unit:
    """The registered unit for a quantity id, raising if nothing is registered under it.

    Raising rather than returning a placeholder: a comparison against a unit nobody registered is
    exactly the case where a silent pass is most damaging, and a `KeyError` at the call site names
    the id that is missing.
    """
    from reward_lens.core.quantity import load_quantities

    if quantity not in QUANTITIES:
        load_quantities()
    return QUANTITIES.get(quantity).unit  # type: ignore[no-any-return]


@dataclass(frozen=True)
class ThresholdQuantity:
    """One reading from series I, carrying the unit that decides what it may be compared to."""

    quantity: str
    value: float

    @property
    def unit(self) -> Unit:
        return unit_of(self.quantity)


def rank(a: ThresholdQuantity, b: ThresholdQuantity) -> Any:
    """The worse of two gate readings, or the refusal that says they do not rank.

    Ranking rather than subtracting, because ranking is what a reader actually wants to do with two
    gate statistics and it is the operation that looks harmless. Subtracting a z from an elasticity
    is obviously wrong; asking which of two gates is worse and getting an answer is not.
    """
    if not a.unit.compatible_with(b.unit):
        return Refusal(
            instrument="measure.threshold.units.rank",
            reason=RefusalReason.UNIT_MISMATCH,
            detail=(
                f"{a.quantity} is in {a.unit} and {b.quantity} is in {b.unit}; these are different "
                f"quantities, not one quantity in two clothes"
            ),
            remedy=(
                "rank gates on the same quantity. A McCrary z says the density is discontinuous "
                "and a bunching elasticity says what the discontinuity implies about behaviour; "
                "there is no factor between them, so the larger number is not the worse gate."
            ),
            statistics={"unit_a": str(a.unit), "unit_b": str(b.unit)},
        )
    return a if a.value >= b.value else b


__all__ = ["ThresholdQuantity", "rank", "unit_of"]

"""The `units` group's assertion, made executable for series E.

Two of the six instruments here are registered under `units`, whose assertion is not a
numeric relation at all: a comparison across a unit boundary raises `UNIT_MISMATCH` rather than
silently converting. `check_invariance` routes that group to `check_unit_refusal`, which needs a
comparison to assert on, and this is it.

The pair that matters in this series is real rather than contrived. `estimator.noise_share` is a
dimensionless share of gradient power and `policy.train_infer_logprob_mismatch` is in nats per
token. Subtracting one from the other, or ranking them against each other, is the most common
silent failure in this literature. The second of those two was once mis-decomposed as per-sequence
in the registry, which had made it byte-identical to `update.kl_spent`.

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
class EstimatorQuantity:
    """One reading from series E, carrying the unit that decides what it may be compared to."""

    quantity: str
    value: float

    @property
    def unit(self) -> Unit:
        return unit_of(self.quantity)


def difference(a: EstimatorQuantity, b: EstimatorQuantity) -> Any:
    """Subtract two series E readings, or refuse when the units do not admit it.

    There is nothing to convert. The factor between a per-token quantity and a dimensionless share
    is a property of the data (how many tokens?) rather than of the unit, so a conversion would need
    information the comparison does not have.
    """
    if not a.unit.compatible_with(b.unit):
        return Refusal(
            instrument="measure.estimator.units.difference",
            reason=RefusalReason.UNIT_MISMATCH,
            detail=(
                f"{a.quantity} is in {a.unit} and {b.quantity} is in {b.unit}; these are different "
                f"quantities, not one quantity in two clothes"
            ),
            remedy=(
                "compare readings of the same quantity. A share of gradient power and a per-token "
                "logprob gap do not subtract, and the conversion factor between them is a property "
                "of the batch rather than of the unit."
            ),
            statistics={"unit_a": str(a.unit), "unit_b": str(b.unit)},
        )
    return EstimatorQuantity(quantity=a.quantity, value=a.value - b.value)


__all__ = ["EstimatorQuantity", "difference", "unit_of"]

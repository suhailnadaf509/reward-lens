"""K2's three-rung calibration ladder, assembled from parts that already exist.

A calibration chain is a primary method, a reference method, a working method, and a
transfer coefficient for each step down. `core/reference.py` holds the types and composes the budget;
`measure/labels/reference.py` measures the reference material's own certificate; `measure/meta/rungs.py`
turns any two rungs measured on the same data into a transfer term. None of them says what K2's three
rungs *are*, and that is all this module does.

    rung 0   the raw gap. Calibrate on a clean planted organism, evaluate on the real corpus, take
             the difference. This is what everyone reports and it is what X3 measured.
    rung 1   standard addition. Dose the target itself and calibrate inside its own matrix, so the
             sensitivity is the target's. `organisms/standard_addition.py`.
    rung 2   a selection diagram, stating which quantities are entitled to cross at all.
             `organisms/transport.py`.

The rungs are ordered by what they cost and by what they remove, and the two orders agree here,
which is unusual and worth saying: rung 1 costs a fine-tune of the target and removes the
multiplicative matrix bias, rung 2 costs an assumption written down and removes the class of
comparisons that were never licensed.

Why this is thin
----------------

Everything numerical is delegated. The transfer between two rungs is `core.reference.ladder_disagreement`,
which is M11 in one call. The composed budget is `CalibrationChain.as_budget`, which is the GUM
table with degrees of freedom plumbed through. The reference certificate is L1's. What this module
contributes is the bookkeeping that keeps a ladder honest: rungs that estimate different quantities
refuse rather than subtract, and a ladder whose reference material is uncertified reports its total
as a lower bound rather than as a total.

The limit, stated here: a transfer coefficient between two rungs is only a *transfer* when both
rungs ran on the same items. Two rungs measured on different corpora differ for reasons that
include the corpora, and the difference is then a mixture with no named parts. `TransferLadder`
checks the item count and refuses on a mismatch, which catches the accident and not the case where
two disjoint corpora happen to be the same size.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from reward_lens.core.evidence import register_payload
from reward_lens.core.quantity import QUANTITIES
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.reference import (
    CalibrationChain,
    ChainLevel,
    MatrixDescription,
    ReferenceMaterial,
    Transfer,
    ladder_disagreement,
)

#: Which chain level each K2 rung sits at. Rung 0 and rung 1 both estimate the working method's
#: disagreement with the reference method; rung 2 is not a level at all, it is a licence, and it
#: appears here mapped to the rung it licenses rather than to a level of its own.
RUNG_LEVELS: dict[int, tuple[ChainLevel, ChainLevel]] = {
    0: ("primary", "reference_method"),
    1: ("primary", "reference_method"),
}


@dataclass(frozen=True)
class LadderRung:
    """One estimate of the transfer coefficient, and what it took to get it.

    ``quantity`` is required rather than optional. Two rungs are comparable because they estimate
    the same registered quantity, and a rung that names none is a rung whose comparability is taken
    on trust, which is the case `measure/meta/rungs.py` refuses on and this type declines to create.
    """

    rung: int
    value: float
    n: int
    quantity: str
    estimator: str
    access: str = ""
    u: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    cost: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not self.quantity:
            raise ValueError(
                f"rung {self.rung} names no quantity. Two rungs of a ladder are comparable because "
                f"they estimate the same registered thing; without the id there is nothing to "
                f"check that against and the difference is arithmetic rather than a transfer."
            )
        if self.n < 1:
            raise ValueError(f"rung {self.rung} reports n = {self.n}; a rung ran on some items")

    @property
    def has_interval(self) -> bool:
        return math.isfinite(self.ci_low) and math.isfinite(self.ci_high)

    def render(self) -> str:
        interval = f" [{self.ci_low:.4g}, {self.ci_high:.4g}]" if self.has_interval else ""
        tail = f", {self.cost}" if self.cost else ""
        return f"rung {self.rung}: {self.value:.4g}{interval}  ({self.estimator}{tail})"

    def __canonical__(self) -> dict[str, Any]:
        return {
            "rung": self.rung,
            "value": self.value,
            "n": self.n,
            "quantity": self.quantity,
            "estimator": self.estimator,
            "access": self.access,
            "u": self.u,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "cost": self.cost,
            "note": self.note,
        }


@register_payload
@dataclass(frozen=True)
class TransferLadder:
    """K2's rungs side by side, with the disagreements between them as budget terms.

    ``improvement`` is the number the row exists to produce: how far rung 1 moves the coefficient
    from rung 0. It is signed, and negative is the interesting direction, because the prediction is
    that dosing the target drives the coefficient down.
    """

    rungs: tuple[LadderRung, ...]
    transfers: tuple[Transfer, ...]
    quantity: str
    note: str = ""

    @property
    def by_rung(self) -> dict[int, LadderRung]:
        return {r.rung: r for r in self.rungs}

    @property
    def improvement(self) -> float:
        """`rung1 - rung0`, or NaN when either is absent. Negative means standard addition helped."""
        table = self.by_rung
        if 0 not in table or 1 not in table:
            return float("nan")
        return float(table[1].value - table[0].value)

    @property
    def improvement_fraction(self) -> float:
        """`improvement / rung0`: what fraction of the raw gap standard addition removed."""
        table = self.by_rung
        if 0 not in table or table[0].value == 0.0:
            return float("nan")
        return float(self.improvement / table[0].value)

    def render(self) -> str:
        lines = [f"{self.quantity} over {len(self.rungs)} rung(s)"]
        lines += [f"  {r.render()}" for r in sorted(self.rungs, key=lambda r: r.rung)]
        lines += [f"  {t.render()}" for t in self.transfers]
        if math.isfinite(self.improvement):
            lines.append(
                f"  standard addition moves it by {self.improvement:+.4g} "
                f"({self.improvement_fraction:+.1%} of the raw gap)"
            )
        if self.note:
            lines.append(f"  {self.note}")
        return "\n".join(lines)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "quantity": self.quantity,
            "rungs": [r.__canonical__() for r in self.rungs],
            "transfers": [
                {
                    "name": t.name,
                    "value": t.value,
                    "method": t.method,
                    "n": t.n,
                    "note": t.note,
                }
                for t in self.transfers
            ],
            "improvement": self.improvement,
            "improvement_fraction": self.improvement_fraction,
            "note": self.note,
        }


def build_ladder(
    rungs: Sequence[LadderRung], *, instrument: str = "organisms.chain", note: str = ""
) -> TransferLadder | Refusal:
    """Order the rungs, check they are rungs of one ladder, and form the transfers between them.

    Three refusals, in the order they are cheapest to act on. A ladder with one rung is not a
    ladder. Rungs estimating different quantities refuse with `UNIT_MISMATCH`, because their
    difference is a number in no unit. Rungs measured on different item counts refuse with
    `RECORD_INCOMPLETE`, because a difference across two corpora is a mixture and the mixing
    weights are not recoverable from the two numbers.
    """
    ordered = sorted(rungs, key=lambda r: r.rung)
    if len(ordered) < 2:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"{len(ordered)} rung(s) supplied. A transfer term is the disagreement between two "
                f"rungs measured on the same data, so one rung produces no transfer and a ladder "
                f"of one is a single estimate wearing a ladder's name."
            ),
            remedy=(
                "measure the same quantity at two rungs on the same items. For K2 that is the raw "
                "planted-organism gap and the standard-addition gap on one corpus."
            ),
            statistics={"n_rungs": len(ordered)},
        )

    quantities = {r.quantity for r in ordered}
    if len(quantities) > 1:
        names = sorted(quantities)
        units = {
            q: str(QUANTITIES.get(q).unit) if QUANTITIES.get(q) is not None else "unregistered"
            for q in names
        }
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.UNIT_MISMATCH,
            detail=(
                f"the rungs estimate {len(names)} different quantities: "
                + "; ".join(f"{q} in {units[q]}" for q in names)
                + ". Their differences are not transfer terms and composing them into one budget "
                "would add quantities that are not the same quantity."
            ),
            remedy=(
                "give every rung the same registered quantity id, or publish them as separate "
                "readings. Two rungs measuring different things are two facts, not a disagreement."
            ),
            statistics={"quantities": names},
        )

    counts = {r.n for r in ordered}
    if len(counts) > 1:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"the rungs ran on {sorted(counts)} items respectively. A transfer coefficient is "
                f"the disagreement between two estimators on one dataset; across two datasets the "
                f"difference also contains whatever the datasets differ by, and nothing in the two "
                f"numbers separates the parts."
            ),
            remedy=(
                "score every rung on the same items. Where a rung genuinely cannot reach all of "
                "them, restrict every rung to the intersection and report the reduced n, rather "
                "than comparing a rung on 25,664 rollouts with one on the 4,000 it could reach."
            ),
            statistics={"item_counts": sorted(counts)},
        )

    transfers = []
    for cheap, expensive in zip(ordered, ordered[1:]):
        levels = RUNG_LEVELS.get(expensive.rung, ("primary", "reference_method"))
        transfers.append(
            ladder_disagreement(
                cheap.value,
                expensive.value,
                from_level=levels[0],
                to_level=levels[1],
                n=cheap.n,
                method=(
                    f"rung {cheap.rung} ({cheap.estimator}) against rung {expensive.rung} "
                    f"({expensive.estimator}) on the same {cheap.n:,} items"
                ),
            )
        )
    return TransferLadder(
        rungs=tuple(ordered),
        transfers=tuple(transfers),
        quantity=ordered[0].quantity,
        note=note,
    )


def compose(
    ladder: TransferLadder,
    reference: ReferenceMaterial,
    *,
    working_matrix: MatrixDescription | None = None,
    u_working: float = 0.0,
    u_instrument: float = 0.0,
) -> CalibrationChain:
    """The ladder's transfers plus a reference certificate, as the kernel's chain.

    A pass-through to `CalibrationChain`, kept here so a caller assembling K2 does not have to know
    that the transfers and the certificate compose in the kernel rather than in either of the
    modules that produced them. The kernel is what refuses to publish `u_total` against an
    uncertified reference, and that refusal is the whole reason to route through it.
    """
    return CalibrationChain(
        reference=reference,
        transfers=ladder.transfers,
        u_working=u_working,
        u_instrument=u_instrument,
        working_matrix=working_matrix,
    )


def as_rung_readings(ladder: TransferLadder) -> list[Any]:
    """The ladder as M11 `RungReading`s, so `measure/meta/rungs.py` can publish the disagreement.

    M11 already knows how to render two rungs, check their units and emit the transfer as evidence.
    This adapter exists so K2 does not grow a second copy of that, and it is the same shape as
    `rungs.rung_from_effective_size`, which does the job for A1.
    """
    from reward_lens.measure.meta.rungs import RungReading

    return [
        RungReading(
            rung=r.rung,
            value=r.value,
            n=r.n,
            access=r.access or "organisms + a labelled real corpus",
            estimator=r.estimator,
            ci_low=r.ci_low,
            ci_high=r.ci_high,
            cost=r.cost,
            quantity=r.quantity,
            note=r.note,
        )
        for r in sorted(ladder.rungs, key=lambda r: r.rung)
    ]


__all__ = [
    "RUNG_LEVELS",
    "LadderRung",
    "TransferLadder",
    "as_rung_readings",
    "build_ladder",
    "compose",
]

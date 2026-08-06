"""Three quantities M3 and M4 need, proposed here and **not registered**.

M3 to M8 carry `quantities: OPEN` in `spec/CATALOGUE.yaml` as a bare string rather than a list. So
two of the four instruments in this package estimate something the registry has no id for, and
`lint_instrument` reports that by name, which is the lint doing its job.

Registering a quantity is a decision about what the library claims to measure and it belongs in
`spec/QUANTITIES.yaml`, which this package does not write to. So the proposals live here as data.
Nothing in this module runs at import: `register_proposed()` exists so a test can show that the
instruments are lint-clean apart from the missing registration, and `as_yaml_rows()` emits the
exact shape `spec/QUANTITIES.yaml` uses so the rows can be lifted across without retyping.

The other two instruments need nothing new. M5 estimates `study.power`, which is already
registered: a matched positive control is a *measurement* of the same quantity M10 computes from a
design, at a higher rung, and when the two disagree the disagreement is the deliverable rather
than a discrepancy to reconcile. M10 estimates `study.power`, `study.mde` and
`study.resolution_ratio`, all three registered and all three already attributed to M10.

`spec/QUANTITIES.yaml` attributes `study.power` and `study.mde` to M10 alone. M5 estimates the
first of those and the catalogue's `instrument` list for it should grow an M5, which is likewise a
spec edit rather than one this package makes.
"""

from __future__ import annotations

from typing import Any

from reward_lens.core.quantity import QUANTITIES, Quantity, Unit, register_quantity

#: Dimensionless, matching how `study.power` is registered: `as_printed: '1'`.
_ONE = Unit(dimension="1", per=None, scale=None, as_printed="1")


#: M3's headline: the floor a claim has to clear. Proposed group `reward.affine`, because every
#: baseline in the bank reads the transcript and the label and not the reward, so an affine
#: rescaling of the reward leaves the reading exactly unchanged. That is a real generated test
#: rather than a vacuous one: an M3 reading that moves under a reward rescaling is reading the
#: reward it is supposed to be a control for.
BASELINE_BEST_SCORE = Quantity(
    id="baseline.best_score",
    definition=(
        "The discrimination achieved on a claim's own task by the strongest member of the "
        "six-baseline bank: the area under the ROC curve of that baseline's per-item scores "
        "against the claim's labels."
    ),
    unit=_ONE,
    invariance="reward.affine",
    interpretation=(
        "The floor a claim has to clear. A claim whose own discrimination does not exceed this "
        "has not been shown to detect anything a zero-parameter comparator could not."
    ),
    support=(0.0, 1.0),
    wedge=True,
)

#: The second reading M3 produces. Proposed, and the one to drop if fewer registrations are
#: wanted: it is derivable from the claim's own score and `baseline.best_score`, and it is
#: registered separately only because the paired interval on the difference is not derivable from
#: the two point estimates.
BASELINE_MARGIN = Quantity(
    id="baseline.margin",
    definition=(
        "A claim's own discrimination minus the strongest baseline's, computed on the same items "
        "with the pairing preserved, so the interval on the difference is narrower than the two "
        "marginal intervals would suggest."
    ),
    unit=_ONE,
    invariance="reward.affine",
    interpretation=(
        "Positive with an interval excluding zero is a win. An interval containing zero is a "
        "match, and a match reported as a win is the failure the bank exists to catch."
    ),
    support=(-1.0, 1.0),
    wedge=True,
)

#: M4's headline. Proposed group `reward.affine`, because the ratio is of two effects measured in
#: the same units on the same outcome: the additive part of the group cancels in each difference
#: and the multiplicative part cancels in the ratio.
PLACEBO_EFFECT_RATIO = Quantity(
    id="placebo.effect_ratio",
    definition=(
        "The effect of a coherent semantically irrelevant direction, applied at the same norm and "
        "the same dose, divided by the effect of the claimed direction on the same outcome."
    ),
    unit=_ONE,
    invariance="reward.affine",
    interpretation=(
        "1.0 means a direction with no relation to the claim did the intervention's work, so the "
        "claim does not describe what happened. Near 0 means the effect is specific to the "
        "claimed direction. Values above 1 happen and mean the placebo did more."
    ),
    support=None,
    wedge=True,
)

PROPOSED: tuple[Quantity, ...] = (
    BASELINE_BEST_SCORE,
    BASELINE_MARGIN,
    PLACEBO_EFFECT_RATIO,
)

#: Which instrument each proposal belongs to, for the catalogue's `instrument` list.
PROPOSED_INSTRUMENTS: dict[str, tuple[str, ...]] = {
    "baseline.best_score": ("M3",),
    "baseline.margin": ("M3",),
    "placebo.effect_ratio": ("M4",),
}

#: The minimum access each proposal needs, in the catalogue's `min_access` spelling.
PROPOSED_MIN_ACCESS: dict[str, str] = {
    "baseline.best_score": "RECORD:RECORD",
    "baseline.margin": "RECORD:RECORD",
    "placebo.effect_ratio": "GRADER:RECORD",
}


def register_proposed() -> list[str]:
    """Register the proposals in this process only. Not called at import, by design.

    A test uses this to show that the instruments are otherwise lint-clean, which separates "this
    instrument has an unregistered quantity" from "this instrument has four other problems". It is
    not a substitute for the rows landing in `spec/QUANTITIES.yaml`, and nothing in the package
    calls it.
    """
    added = []
    for q in PROPOSED:
        if q.id not in QUANTITIES:
            register_quantity(q)
            added.append(q.id)
    return added


def as_yaml_rows() -> str:
    """The proposals in `spec/QUANTITIES.yaml`'s own field order, ready to paste.

    Emitted rather than hand-written so the rows cannot drift from the `Quantity` objects the
    instruments are linted against.
    """
    lines: list[str] = []
    for q in PROPOSED:
        support: Any = q.support
        lines.append(f"- id: {q.id}")
        lines.append("  unit:")
        lines.append(f"    as_printed: '{q.unit.as_printed}'")
        lines.append(f"    dimension: '{q.unit.dimension}'")
        lines.append(f"    per: {q.unit.per if q.unit.per else 'null'}")
        lines.append(f"    scale: {q.unit.scale if q.unit.scale else 'null'}")
        lines.append(f"  invariance_group: {q.invariance}")
        lines.append(f"  min_access: {PROPOSED_MIN_ACCESS[q.id]}")
        lines.append("  rungs: 1")
        lines.append("  instrument:")
        for inst in PROPOSED_INSTRUMENTS[q.id]:
            lines.append(f"  - {inst}")
        lines.append(f"  wedge: {'true' if q.wedge else 'false'}")
        lines.append(f"  definition: {q.definition}")
        lines.append(f"  interpretation: {q.interpretation}")
        lines.append(
            "  support: " + ("OPEN" if support is None else f"[{support[0]}, {support[1]}]")
        )
    return "\n".join(lines)


__all__ = [
    "BASELINE_BEST_SCORE",
    "BASELINE_MARGIN",
    "PLACEBO_EFFECT_RATIO",
    "PROPOSED",
    "PROPOSED_INSTRUMENTS",
    "PROPOSED_MIN_ACCESS",
    "as_yaml_rows",
    "register_proposed",
]

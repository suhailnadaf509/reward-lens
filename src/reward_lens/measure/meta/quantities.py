"""The eight registry rows this package estimates, written out because all eight say `definition: OPEN`.

Every id here is already registered. `spec/QUANTITIES.yaml` carries the row, the unit and the
invariance group, and carries `definition: OPEN`, `interpretation: OPEN` and `support: OPEN` for
each one. So `Quantity.definition` resolves to the empty string, and a quantity with no definition
cannot do its one job: two rungs of a ladder are comparable because they estimate the
same defined thing, and nothing is comparable to a blank.

Nothing here registers anything. `register_quantity` refuses to redefine a live name to something
different, which is correct and is why filling these in is a `spec/QUANTITIES.yaml` edit rather than
an import-time side effect. The definitions live here as data so that the instruments can be linted
against them and so `as_yaml_rows` can emit the amendment in the file's own field order rather than
having somebody retype it.

**Two of the six instruments in this package had `quantities: OPEN` in the catalogue and needed a
decision.** Both decisions are recorded in `DECIDED` and argued in the module that makes them:

`M8` estimates `study.tau2`. An interlaboratory comparison is a random-effects meta-analysis whose
studies are laboratories, `s_L` is the square root of the between-study variance, and the row's own
definition ("the between-study variance of a random-effects meta-analysis, by DerSimonian-Laird,
Paule-Mandel or restricted maximum likelihood, reported with a Q-profile interval") is already
exactly what M8 computes. That row's `instrument` field is `OPEN`, so naming M8 in it adds an
instrument to a quantity that has none rather than contesting one that does.

`M7` takes the quantity of the reading whose uncertainty it decomposes. A GUM budget is not a
separate measurand: the combined standard uncertainty is in the units of the reading, every term is
a contribution to that reading, and giving the budget an id of its own would create a quantity whose
unit is the unit of whatever it happens to be pointed at. So M7's `quantity` is set per instance
from its subject, and an M7 with no subject refuses rather than emitting an unattributed table.
"""

from __future__ import annotations

from typing import Any, Mapping

from reward_lens.core.quantity import QUANTITIES, Quantity, Unit, load_quantities


def _ensure_quantities() -> None:
    """Load the shipped registry if nobody has yet. Reads the catalogue; registers nothing new."""
    if "substrate.noise_floor" not in QUANTITIES:
        load_quantities()


_ensure_quantities()


#: `rel L2`, from the registry row. Kept undecomposed on purpose: `dimension: '1'` and
#: `scale: relative` are what the row prints and this is not the module that changes them.
_REL_L2 = Unit(dimension="1", per=None, scale="relative", as_printed="rel L2")
_EFFECT = Unit(dimension="effect", per=None, scale=None, as_printed="effect")
_DOSE = Unit(dimension="dose", per=None, scale=None, as_printed="dose")
_RHO = Unit(dimension="correlation", per=None, scale=None, as_printed="ρ")
_DELTA = Unit(dimension="difference", per=None, scale=None, as_printed="Δ")


SUBSTRATE_NOISE_FLOOR = Quantity(
    id="substrate.noise_floor",
    definition=(
        "The standard deviation of a stated reading over replicate measurements of a blank: a "
        "sample of the same substrate, prepared and read by the same procedure, in which the "
        "quantity being looked for is absent by construction. Expressed relative to the scale of "
        "the reading itself, so a floor measured on one grader's score units is comparable with "
        "one measured on another's. It is a property of a configuration rather than of a model: "
        "engine, revision, dtype, attention implementation and layer each change it, which is what "
        "`SubstrateKey` is keyed on."
    ),
    unit=_REL_L2,
    invariance="repr.basis",
    interpretation=(
        "The substrate's disagreement with itself. Any effect smaller than a few multiples of this "
        "is not attributable to the thing being measured, however many samples it was averaged "
        "over, because averaging reduces the sampling term and leaves this one where it was."
    ),
    support=(0.0, float("inf")),
    wedge=True,
)

SUBSTRATE_LOD = Quantity(
    id="substrate.lod",
    definition=(
        "The smallest dose whose reading is distinguishable from a blank, `LOD = 3.3 * "
        "sigma_blank / S`, where `sigma_blank` is `substrate.noise_floor` on the same "
        "configuration and `S` is the slope of the calibration curve of reading against dose at a "
        "stated dose. Undefined, rather than infinite by convention, when `S` is not positive: a "
        "reading that does not respond to dose has no detection limit because it has no "
        "calibration."
    ),
    unit=_DOSE,
    invariance="units",
    interpretation=(
        "Below this, refuse. The three-outcome rule is refuse below the LOD, return a "
        "bound between the LOD and the LOQ, and report with a budget above the LOQ."
    ),
    support=(0.0, float("inf")),
    wedge=True,
)

SUBSTRATE_LOQ = Quantity(
    id="substrate.loq",
    definition=(
        "The smallest dose whose reading can carry a point estimate rather than a bound, "
        "`LOQ = 10 * sigma_blank / S`, on the same configuration and the same calibration curve as "
        "`substrate.lod`."
    ),
    unit=_DOSE,
    invariance="units",
    interpretation=(
        "Between the LOD and this, a reading is detected and not quantifiable, and the honest "
        "return is a bound. A point estimate in that band is false precision with an interval "
        "drawn around it."
    ),
    support=(0.0, float("inf")),
    wedge=True,
)

INTERVENTION_EC50 = Quantity(
    id="intervention.ec50",
    definition=(
        "The dose at which a fitted Hill curve `E(c) = E_max * c^n / (EC50^n + c^n)` reaches half "
        "its own maximum effect, from a dose sweep of at least four doses including a blank. It is "
        "a fitted parameter of a curve and not a reading at a dose, so it is quoted with the fit "
        "that produced it and with the cooperativity `n` beside it."
    ),
    unit=_DOSE,
    invariance="units",
    interpretation=(
        "Where the substrate starts to respond. Read it with `n`: a large `n` is switch-like, which "
        "means a threshold, which is the shape worth finding and the one a single-dose intervention "
        "report cannot see."
    ),
    support=(0.0, float("inf")),
    wedge=False,
)

INSTRUMENT_EFFECT = Quantity(
    id="instrument.effect",
    definition=(
        "What a measurement cost the run it measured, per optimisation step: the wall-clock time "
        "the instrument added and the resident bytes it held, both differenced between two "
        "snapshots taken at step boundaries so the number is a per-step rate rather than a "
        "cumulative total."
    ),
    unit=Unit(dimension="OPEN", per="step", scale="OPEN", as_printed="ms, bytes"),
    invariance="units",
    interpretation=(
        "An instrument whose effect is a material fraction of a step has changed the run it was "
        "measuring, and the reading it produced is of a perturbed system. The number is small for "
        "any real grader and it is worth having precisely because it can then be shown to be small."
    ),
    support=(0.0, float("inf")),
    wedge=True,
)

INSTRUMENT_INCREMENTAL_VALIDITY = Quantity(
    id="instrument.incremental_validity",
    definition=(
        "What an instrument adds to a set of instruments already run on the same items: the score "
        "of the ensemble of the set plus this instrument, minus the score of the ensemble of the "
        "set alone, on one fixed item set with the pairing preserved. Reported with an interval "
        "from a paired resample over items and with the combining rule named, because the "
        "increment is a property of the combining rule as well as of the instrument."
    ),
    unit=Unit(dimension="OPEN", per="OPEN", scale="OPEN", as_printed="4-tuple"),
    invariance="units",
    interpretation=(
        "The bar is decorrelation plus signal, not superiority. An instrument ten points worse than "
        "the set and uncorrelated with it is worth more than one two points better and redundant. "
        "An increment whose interval covers zero is a finding about that instrument."
    ),
    support=None,
    wedge=False,
)

INSTRUMENT_ERROR_CORRELATION = Quantity(
    id="instrument.error_correlation",
    definition=(
        "The correlation between two instruments' per-item **errors** on one fixed item set: phi "
        "for binary outcomes, Pearson for continuous residuals. Between errors and not between "
        "scores, because two instruments that agree about which items are hard have correlated "
        "errors whatever their scores do."
    ),
    unit=_RHO,
    invariance="units",
    interpretation=(
        "Near 1 means the two fail on the same items, so the second adds nothing an ensemble can "
        "use. Near 0 means they fail independently, which is what makes a pair worth more than "
        "either member. This is the statistic nobody reports, and complementarity inferred from "
        "ensemble gain alone is inferred rather than measured."
    ),
    support=(-1.0, 1.0),
    wedge=False,
)

LADDER_RUNG_DISAGREEMENT = Quantity(
    id="ladder.rung_disagreement",
    definition=(
        "The absolute difference between two rungs of one estimator ladder for one quantity, "
        "computed on identical data. It is a `Transfer` term of the calibration chain: the "
        "cheap rung's transfer uncertainty against the expensive one, on this substrate."
    ),
    unit=_DELTA,
    invariance="units",
    interpretation=(
        "It is the reading, not a discrepancy to reconcile. Two rungs disagreeing on one dataset "
        "is what the cheap rung costs in accuracy, measured rather than assumed, and it composes "
        "into the calibration chain of anything that used the cheap rung. Picking the higher rung "
        "and discarding the difference throws away the only free measurement of that cost anyone "
        "gets."
    ),
    support=(0.0, float("inf")),
    wedge=True,
)


#: Every definition this package writes out, by id.
DEFINITIONS: Mapping[str, Quantity] = {
    q.id: q
    for q in (
        SUBSTRATE_NOISE_FLOOR,
        SUBSTRATE_LOD,
        SUBSTRATE_LOQ,
        INTERVENTION_EC50,
        INSTRUMENT_EFFECT,
        INSTRUMENT_INCREMENTAL_VALIDITY,
        INSTRUMENT_ERROR_CORRELATION,
        LADDER_RUNG_DISAGREEMENT,
    )
}

#: The two catalogue fields that were `OPEN` and are decided here, with what closes each.
DECIDED: Mapping[str, str] = {
    "M8.quantities": (
        "study.tau2. An interlaboratory comparison is a random-effects meta-analysis over "
        "laboratories and s_L is the square root of its between-study variance. The row's "
        "`instrument` field is OPEN, so this adds an instrument to a quantity that has none."
    ),
    "M7.quantities": (
        "the quantity of the reading being decomposed, set per instance. A combined standard "
        "uncertainty is in the units of its own reading, so a budget has no measurand separate "
        "from the one it is a budget for, and an M7 with no subject refuses."
    ),
}

#: Which instrument each row should name once the amendment lands, including the two additions.
INSTRUMENT_LISTS: Mapping[str, tuple[str, ...]] = {
    "substrate.noise_floor": ("M1", "M8"),
    "substrate.lod": ("M1",),
    "substrate.loq": ("M1",),
    "intervention.ec50": ("C4", "M1"),
    "instrument.effect": ("M2",),
    "instrument.incremental_validity": ("M9",),
    "instrument.error_correlation": ("M9",),
    "ladder.rung_disagreement": ("M11",),
    "study.tau2": ("M8", "X8"),
}


def _scalar(text: str) -> str:
    """A YAML scalar for a sentence that may contain a colon, quoted the way the file quotes."""
    if ":" in text or "#" in text:
        return "'" + text.replace("'", "''") + "'"
    return text


def as_yaml_rows() -> str:
    """The three OPEN fields of each row, in `spec/QUANTITIES.yaml`'s own order, ready to paste.

    Emitted rather than hand-written so the text cannot drift from the `Quantity` objects the
    instruments are documented and linted against. Only the fields that are OPEN are emitted: the
    unit, the invariance group and the access level are already in the file and are not this
    package's to change.
    """
    lines: list[str] = []
    for qid, q in DEFINITIONS.items():
        lines.append(f"# {qid}")
        lines.append(f"  definition: {_scalar(q.definition)}")
        lines.append(f"  interpretation: {_scalar(q.interpretation)}")
        support: Any = q.support
        if support is None:
            lines.append("  support: OPEN")
        else:
            lo, hi = support
            hi_txt = ".inf" if hi == float("inf") else repr(hi)
            lines.append(f"  support: [{lo}, {hi_txt}]")
        lines.append("  instrument:")
        for inst in INSTRUMENT_LISTS[qid]:
            lines.append(f"  - {inst}")
    lines.append("# study.tau2")
    lines.append("  instrument:")
    for inst in INSTRUMENT_LISTS["study.tau2"]:
        lines.append(f"  - {inst}")
    return "\n".join(lines)


def definition_of(qid: str) -> str:
    """This package's written-out definition for a registered id, or the registry's if it has one."""
    registered = QUANTITIES.get(qid) if qid in QUANTITIES else None
    if registered is not None and registered.definition:
        return str(registered.definition)
    q = DEFINITIONS.get(qid)
    return q.definition if q is not None else ""


__all__ = [
    "DECIDED",
    "DEFINITIONS",
    "INSTRUMENT_EFFECT",
    "INSTRUMENT_ERROR_CORRELATION",
    "INSTRUMENT_INCREMENTAL_VALIDITY",
    "INSTRUMENT_LISTS",
    "INTERVENTION_EC50",
    "LADDER_RUNG_DISAGREEMENT",
    "SUBSTRATE_LOD",
    "SUBSTRATE_LOQ",
    "SUBSTRATE_NOISE_FLOOR",
    "as_yaml_rows",
    "definition_of",
]

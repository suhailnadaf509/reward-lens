"""The capability report: what you could learn about this run, and what more costs.

This is the product for most users. Somebody with a rollout record and a grader endpoint runs one
command and gets four things: what access they turned out to have, whether the estimators'
assumptions hold on their run, which quantities are reachable right now with the rung and the price
attached, and for everything else a refusal that carries an instruction rather than a diagnosis.

It is producible from `preflight` alone, before any compute, which is what makes it cheap enough to
be the front door. Nothing here loads a model, runs an estimator, or touches the network.

Three commitments shape the code more than anything else.

**Availability is only ever claimed for a registered estimator.** `spec/CATALOGUE.yaml` describes 95
instruments and its `access_min` column is free prose ("all four books", "an exploit log",
"GRADER:RECORD (r0), REPLICATE (r2+)"). Parsing that into an access
requirement would be a guess, and a guess in the generous direction produces the one output this
library exists to prevent. So the catalogue's prose is printed verbatim and never parsed, and an
instrument with no registered estimator appears under SPECIFIED, NOT YET BUILT.

That rule is right and its implementation was silently wrong for a long time, which is worth
recording here. This paragraph used to end "Today that is all 85 of
them, and the report says so rather than looking richer than the library is", which was true when it
was written and became false as forty-three packages closed around it. Nothing re-read it. The
report's `covered` set is built from `ESTIMATORS`, and estimators register at module import, so in
any process that had not imported the leaves the set was empty and **every** row fell through to
SPECIFIED, NOT YET BUILT. Measured on the 3.0.0rc1 wheel in a clean environment: 0 registered, 95 of
95 rows reported unbuilt. `_ensure_estimators_loaded` is the fix, and the lesson is that a docstring
describing a temporary state as though it were a design decision is how a defect acquires a
justification.

**A check that could not run is named.** `PreflightResult.unchecked` is the field this report was
built around. An unmeasured regime is not a satisfied one, an unresolved substrate is not a matching
one, and both appear as text on the row they affect.

**A refusal carries an instruction.** "Envelope violated" is a diagnosis. "Restrict the window to
steps 0-239, or supply the weight schedule so the composition can be held fixed counterfactually"
is something to do on Monday. `REMEDY_FOR_CONDITION` and `_access_remedy` are where that lives, and
they are the most user-facing prose in the library.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from reward_lens.access.estimators import load_estimator_ladder
from reward_lens.access.matrix import (
    AccessResolution,
    render_access,
    resolution_from_matrix,
)
from reward_lens.access.phase import PhaseResolution
from reward_lens.access.substrate import SubstrateReading
from reward_lens.core.budget import LimitOfDetection
from reward_lens.core.envelope import RegimeCondition, RegimeReading
from reward_lens.core.quantity import (
    ESTIMATORS,
    QUANTITIES,
    EstimatorEntry,
    QuantityID,
    catalogue_path,
    ladder,
    load_quantities,
    what_would_it_take,
)
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.reference import ReferenceMaterial, uncertified_refusal
from reward_lens.core.types import (
    AccessMatrix,
    Component,
    Phase,
    Substrate,
    satisfies,
)
from reward_lens.measure.base import Context, PreflightResult

# ---------------------------------------------------------------------------
# Column geometry for the printed report block
# ---------------------------------------------------------------------------

_LABEL_COL = 10  # "SUBSTRATE " is the longest label and it is exactly this wide
_VALUE_COL = 29
_QUANTITY_COL = 48
_RUNG_COL = 3
_U_COL = 9
_REFUSAL_ID_COL = 33
#: Wide enough for `planned`, the longest status a not-yet-built row can carry, and a gap after it.
_STATUS_COL = 10
_WRAP = 79
#: The ACCESS RESOLVED rows wrap wider than the refusal prose, because their notes start at
#: column 41 and the longest line in that block is 90 characters.
_ROW_WRAP = 100

#: Refusals print in the order `preflight` checks them, which is the order of how actionable the
#: remedy is. "Supply the policy checkpoint" is something to do this afternoon; "restrict the window
#: to a span where the grader was stationary" is a re-analysis; a substrate mismatch is neither,
#: it is a category error and the reader should see it before they start reading thresholds.
_REASON_ORDER: dict[RefusalReason, int] = {
    RefusalReason.ACCESS_INSUFFICIENT: 0,
    RefusalReason.SUBSTRATE_MISMATCH: 1,
    RefusalReason.PHASE_MISMATCH: 2,
    RefusalReason.ENVELOPE_VIOLATED: 3,
    RefusalReason.BELOW_LOD: 4,
    RefusalReason.REFERENCE_UNCERTIFIED: 5,
    RefusalReason.LABEL_QUALITY_UNKNOWN: 6,
}


# ---------------------------------------------------------------------------
# Remedies
# ---------------------------------------------------------------------------

#: One instruction per regime condition. These are the sentences somebody reads while holding a
#: failed run, so each says what to change rather than what went wrong, and each says what it costs
#: you to take the shortcut where there is one.
REMEDY_FOR_CONDITION: dict[RegimeCondition, str] = {
    RegimeCondition.QUASI_STATIC: (
        "slow the anneal, or restrict the window to a span where the schedule is flat and "
        "re-measure tau_relax inside it. A reading taken while the schedule is moving is a "
        "transient, and more samples will not turn it into a steady-state number."
    ),
    RegimeCondition.LINEAR_RESPONSE: (
        "restrict the window to a span with a smaller step, or ask for a rung that does not expand "
        "to first order. At this step size the second-order term is the size of the effect rather "
        "than a correction to it."
    ),
    RegimeCondition.GROUP_NONDEGENERATE: (
        "raise K, or restrict to the prompts whose groups have spread. A group where every rollout "
        "scored the same contributes no gradient, so including it dilutes the estimate instead of "
        "adding to it."
    ),
    RegimeCondition.NEAR_POLICY: (
        "restrict the window to steps whose rollouts came from one policy version, or shrink the "
        "off-policy buffer and re-run. If the record carries the behaviour policy's logprobs, ask "
        "for the importance-weighted rung instead, which survives staleness at a stated cost in "
        "variance."
    ),
    RegimeCondition.STATIONARY_GRADER: (
        "restrict the window to the span before the grader changed, or supply the weight schedule "
        "so the composition can be held fixed counterfactually."
    ),
    RegimeCondition.EXOGENOUS_CURRICULUM: (
        "hold the task distribution fixed across the window, or measure on a held-out slice the "
        "curriculum does not touch. A curriculum that responds to the score puts the score on both "
        "sides of the estimate."
    ),
    RegimeCondition.NO_COMPACTION: (
        "restrict the window to a span with no prefix rewrite, or supply the compaction events so "
        "the affected trajectories can be dropped. After a rewrite the tokens that were scored are "
        "not the tokens that were generated."
    ),
    RegimeCondition.ABOVE_LOD: (
        "raise the dose, or measure on a configuration with a lower noise floor. A reading below "
        "the floor is not attributable to the thing being measured, so averaging more of them will "
        "not help."
    ),
    RegimeCondition.ESS_ADEQUATE: (
        "shorten the window until the importance weights hold up, or re-collect on policy. Past "
        "the visibility horizon any number is a guess wearing an interval."
    ),
    RegimeCondition.LIGHT_TAILED: (
        "trim or winsorise at a stated quantile and report the quantile, or ask for the "
        "median-of-means rung. A mean over a heavy tail is a statement about the largest "
        "observation."
    ),
    RegimeCondition.SCALAR_REPRESENTABLE: (
        "measure the curl mass and report it beside this reading, or ask for the pairwise rung "
        "that does not assume a scalar potential exists. A grader with curl is not approximating "
        "any single scalar."
    ),
    RegimeCondition.MASK_STABLE: (
        "restrict the window to a span with one loss-mask policy, or supply the masks so the "
        "per-token quantities can be put on a common basis."
    ),
}

#: What supplies each component, as the flag you would actually type. A remedy such as
#: "Remedy: --policy ckpt/step-*/" is this table with one row filled in.
_SUPPLIES: dict[Component, str] = {
    Component.TASK: "--env <environment source tree>",
    Component.GRADER: "--grader <endpoint>",
    Component.POLICY: "--policy <checkpoint directory>",
    Component.RECORD: "--record <run directory>",
    Component.ARTIFACT: "--artifact <served endpoint>",
    Component.GOLD: "--gold <labelled corpus>",
}


def _access_remedy(needs: Mapping[str, str], cheapest: EstimatorEntry | None) -> str:
    """Turn an access gap into an instruction, with the price of the rung it would unlock."""
    parts: list[str] = []
    for name, flags in sorted(needs.items()):
        try:
            component = Component[name]
        except KeyError:
            parts.append(f"{name} at {flags}")
            continue
        how = _SUPPLIES.get(component, f"supply {component.name.lower()}")
        extra = ""
        if component is Component.GRADER and "REPLICATE" in flags:
            extra = (
                ", and a probe budget so the seed can be tested: --probe 3. REPLICATE is not "
                "implied by a reachable endpoint"
            )
        parts.append(f"{name} at {flags} ({how}{extra})")
    joined = "; ".join(parts)
    if cheapest is not None:
        return (
            f"supply {joined}. That would put rung {cheapest.rung} ({cheapest.impl}) within reach "
            f"at {cheapest.cost.render()}."
        )
    return f"supply {joined}."


def _condition_status(
    requires: Sequence[RegimeCondition] | frozenset[RegimeCondition],
    regime: RegimeReading | None,
) -> tuple[list[RegimeCondition], list[RegimeCondition]]:
    """Split an envelope's conditions into the ones that failed and the ones nobody measured.

    `EnvelopeSpec.admits` folds these together and returns False for both, which is the right
    behaviour for a gate: an instrument must not run on an unchecked precondition. A report is not
    a gate. Folding them together here would bury the two conditions that actually failed under the
    ten nobody has measured yet, and the user would stop reading. So the split is made here, a
    measured failure becomes a refusal, and an unmeasured condition becomes an `unchecked` line on
    the row it affects. Neither is ever counted as a pass.

    A condition that is present in the reading with ``holds is None`` counts as failed rather than
    unmeasured. Somebody tried to determine it and could not, and that is a finding.
    """
    failed: list[RegimeCondition] = []
    unmeasured: list[RegimeCondition] = []
    for condition in sorted(requires, key=lambda c: c.name):
        if regime is None or condition not in regime.conditions:
            unmeasured.append(condition)
        elif regime.holds(condition) is not True:
            failed.append(condition)
    return failed, unmeasured


def _condition_detail(regime: RegimeReading | None, conditions: Sequence[RegimeCondition]) -> str:
    """The measured conditions behind a refusal, with the column padding squeezed out."""
    bits = []
    for condition in conditions:
        reading = regime.conditions.get(condition) if regime is not None else None
        bits.append(
            " ".join(reading.render().split())
            if reading is not None
            else f"{condition.name} unknown"
        )
    return "; ".join(bits)


def _envelope_remedy(reading: RegimeReading | None, conditions: Sequence[RegimeCondition]) -> str:
    """One remedy per failed condition, in the order the conditions failed."""
    out: list[str] = []
    for condition in conditions:
        holds = reading.holds(condition) if reading is not None else None
        instruction = REMEDY_FOR_CONDITION.get(condition, "")
        if holds is None:
            out.append(
                f"{condition.name} could not be determined, and an unchecked precondition is not "
                f"a satisfied one. Measure it first, then, if it fails: {instruction}"
                if instruction
                else f"{condition.name} could not be determined. Measure it before relying on this."
            )
        else:
            out.append(f"{condition.name} fails: {instruction}" if instruction else condition.name)
    return " ".join(out)


# ---------------------------------------------------------------------------
# The catalogue: read, never parsed for access
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogueInstrument:
    """One row of `spec/CATALOGUE.yaml`, with its declarations normalised.

    The polymorphic-field trap is handled here and nowhere else: six instruments
    store `quantities` as the bare string ``OPEN``, and iterating that yields four single-character
    ids that resolve to nothing and are reported as nothing. `_as_names` normalises a string to an
    empty list explicitly, so an undeclared field reads as undeclared rather than as four phantom
    entries.
    """

    id: str
    name: str
    quantities: tuple[str, ...] = ()
    substrates: frozenset[Substrate] = frozenset()
    phases: frozenset[Phase] = frozenset()
    envelope_requires: frozenset[RegimeCondition] = frozenset()
    access_min: str = ""
    wedge: bool = False
    status: str = ""

    @property
    def headline_quantity(self) -> str:
        return self.quantities[0] if self.quantities else self.id


def _as_names(value: Any) -> tuple[str, ...]:
    """A list field, with the catalogue's bare ``OPEN`` string normalised to nothing."""
    if value is None or isinstance(value, str):
        return ()
    return tuple(str(v) for v in value)


def _status_word(status: str) -> str:
    """A row's own `status`, which is the whole of what it can say about work nobody has done.

    Three values, and they are the only three: `built`, `planned`, and `OPEN` for a row that is
    registered but unscheduled. A row carrying no status at all reads as `OPEN` rather than as
    blank, because a blank field and a deliberate "nobody has committed to this" look identical
    and mean different things.
    """
    return (status or "").strip() or "OPEN"


def _members(names: Sequence[str], enum_cls: Any) -> frozenset[Any]:
    out = []
    for name in names:
        try:
            out.append(enum_cls[str(name).strip().upper()])
        except KeyError:
            continue
    return frozenset(out)


_CATALOGUE: tuple[CatalogueInstrument, ...] | None = None


def load_instrument_catalogue(path: Any = None) -> tuple[CatalogueInstrument, ...]:
    """Read the 85 catalogue records. Cached, because nothing about them changes at runtime."""
    global _CATALOGUE
    if path is None and _CATALOGUE is not None:
        return _CATALOGUE
    source = path if path is not None else catalogue_path("CATALOGUE.json")
    if source is None:
        return ()
    with open(source, encoding="utf-8") as fh:
        doc = json.load(fh)
    rows = tuple(
        CatalogueInstrument(
            id=str(row.get("id", "?")),
            name=str(row.get("name", "")),
            quantities=_as_names(row.get("quantities")),
            substrates=_members(_as_names(row.get("substrates")), Substrate),
            phases=_members(_as_names(row.get("phases")), Phase),
            envelope_requires=_members(_as_names(row.get("envelope_requires")), RegimeCondition),
            access_min=""
            if row.get("access_min") in (None, "OPEN")
            else str(row.get("access_min")),
            wedge=bool(row.get("wedge", False)),
            status=str(row.get("status", "")),
        )
        for row in doc.get("instruments", [])
    )
    if path is None:
        _CATALOGUE = rows
    return rows


def _ensure_quantities_loaded() -> None:
    """The quantity registry is loaded from `spec/` on first use rather than at import."""
    if len(QUANTITIES) == 0:
        load_quantities()


def _ensure_estimators_loaded() -> None:
    """Import the tree so the estimator ladder exists before anyone asks what is built.

    The two registries fill from different places and only the first was ever hooked here.
    Quantities come from a data file, so `_ensure_quantities_loaded` can load them on demand.
    Estimators come from module import: `verifier/coverage.py` ends in a bare `_register()` call
    and nothing registers them centrally, so `ESTIMATORS` is empty in a process that has done
    `import reward_lens` and stays empty after `import reward_lens.measure`, because a package's
    `__init__` does not pull its leaves.

    Without this, `covered` below is empty, so **every** catalogue row falls through to
    `SPECIFIED, NOT YET BUILT` and the report tells a reader that a library of ninety-five
    instruments contains none. Measured on the 3.0.0rc1 wheel in a clean environment before this
    call existed: 0 estimators registered, 0 quantities covered, 95 of 95 rows reported as not yet
    built.

    Failures are not raised. A base install genuinely cannot see the modules behind an extra, and
    reporting their quantities as unavailable is correct there; `load_estimator_ladder` names which
    extras were missing so a caller can tell that apart from an instrument nobody wrote. A module
    that is broken for any other reason is collected in `LadderLoad.broken`, which the docs build
    fails on and this report does not, because refusing to answer at all is the one thing the
    capability report must never do.
    """
    load_estimator_ladder()


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Assessment:
    """One quantity, and the preflight that decided whether it is reachable here.

    ``built`` is False for a quantity whose only instrument is a catalogue record. The distinction
    matters in both directions: a not-built instrument is never counted as available, and a
    not-built instrument whose declared envelope already fails on this run is still worth saying
    out loud, because it tells you the reading would not have been valid even once it is built.
    """

    quantity: QuantityID
    preflight: PreflightResult
    instrument_id: str = ""
    built: bool = True

    @property
    def ok(self) -> bool:
        return self.preflight.ok

    @property
    def refusal(self) -> Refusal | None:
        return self.preflight.refusal


@dataclass(frozen=True)
class NotBuilt:
    """A catalogue instrument with no registered estimator and nothing against it on this run.

    `built` separates the two very different reasons a row lands here, and the separation is the
    whole point of the field. A row with `built=False` has no implementation: "not yet built" is
    the truth about it. A row with `built=True` has a shipped, linting instrument and no registered
    `EstimatorEntry`, so the report cannot quote its rung, its expected uncertainty or its price,
    and that is the only thing it cannot do. Calling the second one "not yet built" is false about
    the library.

    The obvious shortcut is to fold built rows into `covered` instead. That is worse than the
    defect: the loop that builds this list skips a covered quantity outright, on the assumption it
    was already assessed against a registered estimator. A row with no estimator entry never was,
    so it would be skipped without being assessed and **disappear from the report entirely**. A
    mislabelled row is visible and a dropped one is not.
    """

    id: str
    name: str
    quantity: str
    status: str
    access_min: str
    wedge: bool = False
    built: bool = False


@dataclass(frozen=True)
class CapabilityReport:
    """The four sections of the report, plus the two the honest version of it needs.

    SPECIFIED, NOT YET BUILT exists because 85 instruments are catalogued and none of them has a
    registered estimator yet. Folding those into the refusals would misreport what is implemented
    as an access problem, and hiding them would make the library look smaller than its plan.

    NOT CHECKED exists because `PreflightResult.unchecked` has to land somewhere. A report that
    silently omits the checks it could not run is the same failure as an instrument that treats an
    unmeasured precondition as a satisfied one.
    """

    access: AccessResolution
    substrate: SubstrateReading
    phase: PhaseResolution
    regime: RegimeReading | None = None
    assessments: tuple[Assessment, ...] = ()
    not_built: tuple[NotBuilt, ...] = ()
    unchecked: tuple[str, ...] = ()
    catalogue_size: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def available(self) -> tuple[Assessment, ...]:
        return tuple(a for a in self.assessments if a.ok)

    @property
    def refused(self) -> tuple[Refusal, ...]:
        return tuple(a.refusal for a in self.assessments if a.refusal is not None)

    # -- rendering ---------------------------------------------------------

    def render(self, *, show_not_built: bool = False) -> str:
        return "\n".join(
            self._access_block()
            + [""]
            + self._regime_block()
            + [""]
            + self._available_block()
            + [""]
            + self._refused_block()
            + [""]
            + self._not_built_block(show_all=show_not_built)
            + [""]
            + self._unchecked_block()
        )

    def _row(self, label: str, value: str, note: str) -> str:
        """One ACCESS RESOLVED line: label, value, and the note in parentheses beside it.

        The note goes in its own column and wraps back to that column, because a resolved access
        is usually a sentence rather than a word ("no weights supplied; --policy would add
        FORWARD") and the value it explains is usually one token.
        """
        pad = " " * (2 + _LABEL_COL + _VALUE_COL)
        head = f"  {label:<{_LABEL_COL}}{value}"
        if not note:
            return head.rstrip()
        first = (head + f"  ({note})") if len(head) > len(pad) else f"{head:<{len(pad)}}({note})"
        return "\n".join(textwrap.wrap(first, width=_ROW_WRAP, subsequent_indent=pad))

    def _access_block(self) -> list[str]:
        lines = ["ACCESS RESOLVED"]
        for component in self.access.components:
            lines.append(
                self._row(component.component.name, render_access(component.access), component.note)
            )
        lines.append(self._row("SUBSTRATE", self.substrate.render(), self.substrate.note))
        lines.append(self._row("PHASE", self.phase.render(), self.phase.note))
        if self.access.calls_made:
            lines.append(f"  {self.access.calls_made} grader calls were made to resolve this.")
        return lines

    def _regime_block(self) -> list[str]:
        lines = ["REGIME MEASURED"]
        if self.regime is None or not self.regime.conditions:
            lines.append(
                "  not measured. Every estimator with a non-empty envelope is reported below with"
            )
            lines.append(
                "  its envelope unchecked rather than satisfied; measure the regime to settle them."
            )
            return lines
        lines.extend(
            f"  {self.regime.conditions[c].render()}".rstrip()
            for c in sorted(self.regime.conditions, key=lambda c: c.name)
        )
        return lines

    def _available_block(self) -> list[str]:
        header = "AVAILABLE NOW" + " " * 36 + "rung   est. u   cost"
        lines = [header]
        rows = self.available
        if not rows:
            lines.append(
                "  nothing. No registered estimator resolves at this access, substrate and phase."
            )
            return lines
        for item in sorted(rows, key=lambda a: a.quantity):
            pre = item.preflight
            lines.append(
                f"  {item.quantity:<{_QUANTITY_COL}}"
                f"{pre.rung:>{_RUNG_COL}}   "
                f"{_render_uncertainty(pre.expected_uncertainty):<{_U_COL}}"
                f"{pre.cost.render()}".rstrip()
            )
            if pre.unchecked:
                lines.extend(_wrap(f"not checked: {', '.join(pre.unchecked)}"))
            for note in pre.notes:
                lines.extend(_wrap(f"note: {note}"))
        return lines

    def _refused_block(self) -> list[str]:
        lines = ["REFUSED, WITH REMEDY"]
        refusals = [(a.quantity, a.refusal) for a in self.assessments if a.refusal is not None]
        if not refusals:
            lines.append("  nothing was refused.")
            return lines
        for quantity, refusal in sorted(
            refusals, key=lambda kv: (_REASON_ORDER.get(kv[1].reason, 99), kv[0])
        ):
            lines.append(f"  {quantity:<{_REFUSAL_ID_COL}}{refusal.reason.name}")
            lines.extend(_wrap(refusal.detail))
            if refusal.partial is not None:
                lines.extend(_wrap(f"Bound: {refusal.partial.value}"))
            lines.extend(_wrap(f"Remedy: {refusal.remedy}"))
        return lines

    def _not_built_block(self, *, show_all: bool) -> list[str]:
        """Two groups, because "nobody wrote it" and "it ships and has no price" are not the same.

        These were one list until a correction. Sixty-three shipped, linting instruments were
        being described to users as not yet built, because the only signal consulted was whether a
        registered `EstimatorEntry` existed and most instruments do not register one. The rows are
        still both here, and neither is hidden: what changed is that each is now told the truth
        about itself.
        """
        priced = [n for n in self.not_built if n.built]
        unwritten = [n for n in self.not_built if not n.built]

        lines: list[str] = []
        if priced:
            lines.append("BUILT, WITH NO REGISTERED COST")
            lines.extend(
                _wrap(
                    "These ship and pass their own lint. What is missing is an estimator entry, so "
                    "this report cannot quote a rung, an expected uncertainty or a price for them. "
                    "Call the instrument directly and it will answer or refuse on its own terms.",
                    indent=2,
                )
            )
            for item in sorted(priced, key=lambda n: _natural(n.id)):
                lines.append(f"  {item.id:<6}{item.quantity}")
            lines.append("")

        lines.append("SPECIFIED, NOT YET BUILT")
        if not unwritten:
            lines.append("  nothing; every catalogued instrument has an implementation.")
            return lines
        by_status: dict[str, list[NotBuilt]] = {}
        for item in unwritten:
            by_status.setdefault(_status_word(item.status), []).append(item)
        if show_all:
            for item in sorted(self.not_built, key=lambda n: _natural(n.id)):
                lines.append(
                    f"  {item.id:<6}{item.quantity:<{_REFUSAL_ID_COL}}"
                    f"{_status_word(item.status):<{_STATUS_COL}}"
                    f"{item.access_min}".rstrip()
                )
        else:
            for status in sorted(by_status):
                items = by_status[status]
                names = ", ".join(i.id for i in sorted(items, key=lambda n: _natural(n.id)))
                lines.extend(
                    _wrap(f"{status}: {_plural(len(items), 'instrument')} ({names})", indent=2)
                )
            lines.append(
                "  Pass --all to list each one with the access its specification asks for."
            )
        return lines

    def _unchecked_block(self) -> list[str]:
        lines = ["NOT CHECKED"]
        if not self.unchecked:
            lines.append("  nothing; every check this report can make, it made.")
        else:
            for item in self.unchecked:
                lines.extend(_wrap(item, indent=2))
        lines.append("")
        lines.append(
            f"  {self.catalogue_size} catalogued instruments: {len(self.available)} available, "
            f"{len(self.refused)} refused, {len(self.not_built)} not yet built."
        )
        return lines

    # -- machine form ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "access": {
                c.component.name: {
                    "access": render_access(c.access),
                    "note": c.note,
                    "unresolved": list(c.unresolved),
                }
                for c in self.access.components
            },
            "grader_calls_made": self.access.calls_made,
            "substrate": {
                "value": self.substrate.substrate.name if self.substrate.substrate else None,
                "render": self.substrate.render(),
                "leaves": [
                    {
                        "name": leaf.name,
                        "substrate": leaf.substrate.name if leaf.substrate else None,
                        "enabled": leaf.enabled,
                        "why": leaf.why,
                    }
                    for leaf in self.substrate.leaves
                ],
            },
            "phase": {
                "value": self.phase.phase.name if self.phase.enforced else None,
                "note": self.phase.note,
            },
            "regime": (
                {c.name: self.regime.conditions[c].holds for c in self.regime.conditions}
                if self.regime is not None
                else None
            ),
            "available": [
                {
                    "quantity": a.quantity,
                    "rung": a.preflight.rung,
                    "expected_uncertainty": a.preflight.expected_uncertainty,
                    "cost": a.preflight.cost.render(),
                    "unchecked": list(a.preflight.unchecked),
                }
                for a in self.available
            ],
            "refused": [
                {
                    "quantity": a.quantity,
                    "instrument": a.instrument_id or a.preflight.instrument,
                    "reason": a.refusal.reason.name,
                    "detail": a.refusal.detail,
                    "remedy": a.refusal.remedy,
                    "built": a.built,
                }
                for a in self.assessments
                if a.refusal is not None
            ],
            "not_built": [
                {
                    "id": n.id,
                    "name": n.name,
                    "quantity": n.quantity,
                    "access_min": n.access_min,
                }
                for n in self.not_built
            ],
            "unchecked": list(self.unchecked),
            "catalogue_size": self.catalogue_size,
        }


def _render_uncertainty(u: float | None) -> str:
    """``±0.4``, ``exact``, or ``unknown``. Zero means an exact count, not a suspiciously good one."""
    if u is None:
        return "unknown"
    if u == 0:
        return "exact"
    return f"±{u:.2g}"


def _wrap(text: str, indent: int = 6) -> list[str]:
    pad = " " * indent
    return textwrap.wrap(text, width=_WRAP, initial_indent=pad, subsequent_indent=pad) or [pad]


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _natural(instrument_id: str) -> tuple[str, int]:
    """Sort D2 before D10. The catalogue's ids are a letter and a number, not strings."""
    head = instrument_id.rstrip("0123456789")
    tail = instrument_id[len(head) :]
    return head, int(tail) if tail else 0


def _leaves_matching(
    declared: frozenset[Substrate] | None, reading: SubstrateReading
) -> tuple[str, ...]:
    """Which live leaves of a composite grader an instrument's declared substrates cover.

    `COMPOSITE` admits all the other substrates *on its leaves*, plus the composition
    instruments on the tree itself. So a `PROGRAM` instrument in front of a tree
    with a verifier leaf is not a category error, it is an instrument pointed at the wrong node,
    and those call for different sentences. Without this the report tells someone with a perfectly
    ordinary verifier-plus-judge grader that a coverage instrument will never apply to them, which
    is both wrong and the exact opposite of the advice they need.
    """
    if not declared or reading.substrate is not Substrate.COMPOSITE:
        return ()
    return tuple(leaf.name for leaf in reading.live_leaves if leaf.substrate in declared)


def _join(items: Sequence[str], conjunction: str = "and") -> str:
    """``a``, ``a and b``, ``a, b and c``. Remedies are read as sentences."""
    items = list(items)
    if len(items) <= 1:
        return items[0] if items else ""
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


def _substrate_remedy(declared: frozenset[Substrate], reading: SubstrateReading, what: str) -> str:
    """The remedy for a substrate mismatch, which depends on whether a leaf would take it."""
    leaves = _leaves_matching(declared, reading)
    if leaves:
        kinds = _join(sorted(s.name for s in declared), "or")
        subject = (
            f"{leaves[0]} is a {kinds} leaf and would take it"
            if len(leaves) == 1
            else f"{_join(leaves)} are {kinds} leaves and any of them would take it"
        )
        return (
            f"point it at a leaf rather than at the tree. {subject}. For the tree itself, ask for "
            f"the composition instruments, which are the ones that measure how the leaves are "
            f"combined rather than what any one of them says."
        )
    kind = reading.substrate.name if reading.substrate is not None else "unresolved"
    return (
        f"ask for {what} declared for {kind}. A {kind} grader is a different kind of object, not a "
        f"harder case of the same one, so no rung of this ladder gets easier by being asked twice."
    )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _unchecked_for(
    regime: RegimeReading | None,
    substrate: Substrate | None,
    phase: Phase | None,
    lod: LimitOfDetection | None,
    entry: EstimatorEntry | None,
    unmeasured: Sequence[RegimeCondition] = (),
    reference: ReferenceMaterial | None = None,
) -> tuple[str, ...]:
    """What this row could not check, in the wording `BaseObservable.preflight` uses."""
    out: list[str] = []
    if substrate is None:
        out.append("substrate")
    if phase is None:
        out.append("phase")
    if entry is not None and entry.envelope.requires and regime is None:
        out.append("envelope (regime not measured)")
    elif unmeasured:
        out.append("envelope (" + ", ".join(c.name for c in unmeasured) + " not measured)")
    if lod is None:
        out.append("limit of detection")
    if entry is not None and entry.calibrated_against and reference is None:
        out.append(f"reference ({entry.calibrated_against} not supplied)")
    return tuple(out)


def _assess_registered(
    quantity: QuantityID,
    access: AccessMatrix,
    substrate_reading: SubstrateReading,
    phase: Phase | None,
    regime: RegimeReading | None,
    lod: LimitOfDetection | None,
    reference: ReferenceMaterial | None,
) -> Assessment:
    """Resolve one quantity against its registered ladder, in preflight's order.

    Access first, because "supply the policy checkpoint" is the most actionable remedy there is.
    Substrate and phase next, because those are category errors. The envelope last among the hard
    checks, because it is the only one whose failure needs a measurement to establish, and an
    unmeasured regime is reported as unchecked rather than refused, which is what
    `BaseObservable.preflight` does with the same situation.
    """
    substrate = substrate_reading.substrate
    rungs = ladder(quantity)
    by_access = [e for e in rungs if satisfies(access, e.requires)]
    if not by_access:
        gap = what_would_it_take(quantity, access)
        cheapest = rungs[0] if rungs else None
        needs = gap.get("needs", {})
        return Assessment(
            quantity=quantity,
            preflight=PreflightResult(
                instrument=cheapest.impl if cheapest else quantity,
                ok=False,
                refusal=Refusal(
                    instrument=quantity,
                    reason=RefusalReason.ACCESS_INSUFFICIENT,
                    detail=(
                        "needs "
                        + ", ".join(f"{k}: {v}" for k, v in sorted(needs.items()))
                        + "; you have "
                        + (
                            ", ".join(
                                f"{c.name}: {render_access(a)}"
                                for c, a in sorted(access.items(), key=lambda kv: kv[0].name)
                                if a
                            )
                            or "nothing"
                        )
                    ),
                    remedy=_access_remedy(needs, cheapest),
                    statistics=dict(gap),
                ),
            ),
        )

    by_substrate = [e for e in by_access if e.applies_to(substrate, None)]
    if not by_substrate and substrate is not None:
        declared = frozenset(s for e in by_access if e.substrates for s in e.substrates)
        leaves = _leaves_matching(declared, substrate_reading)
        names = ", ".join(sorted(s.name for s in declared))
        return Assessment(
            quantity=quantity,
            preflight=PreflightResult(
                instrument=by_access[0].impl,
                ok=False,
                refusal=Refusal(
                    instrument=quantity,
                    reason=RefusalReason.SUBSTRATE_MISMATCH,
                    detail=(
                        f"every estimator for this quantity applies to {names}; the grader is "
                        f"{substrate.name}"
                        + (f", whose leaves include {_join(leaves)}" if leaves else "")
                    ),
                    remedy=_substrate_remedy(declared, substrate_reading, "a quantity"),
                ),
            ),
        )

    by_phase = [e for e in by_substrate if e.applies_to(None, phase)]
    if not by_phase and phase is not None:
        declared = sorted({p.name for e in by_substrate if e.phases for p in e.phases})
        return Assessment(
            quantity=quantity,
            preflight=PreflightResult(
                instrument=by_substrate[0].impl,
                ok=False,
                refusal=Refusal(
                    instrument=quantity,
                    reason=RefusalReason.PHASE_MISMATCH,
                    detail=(
                        f"this quantity is estimated at {'/'.join(declared)}; you are at "
                        f"{phase.name}"
                    ),
                    remedy=(
                        f"measure it during {' or '.join(declared)}. If the run is over, ask "
                        f"whether a record-only quantity answers the same question: "
                        f"`reward-lens capabilities` lists what this record still supports."
                    ),
                ),
            ),
        )

    # The envelope. A condition nobody measured is unchecked; a condition that was measured and
    # did not hold is a refusal. See `_condition_status` for why those are kept apart here.
    status = {e.impl: _condition_status(e.envelope.requires, regime) for e in by_phase}
    admitted = [e for e in by_phase if not status[e.impl][0]]
    if not admitted:
        worst = by_phase[-1]
        failed = status[worst.impl][0]
        return Assessment(
            quantity=quantity,
            preflight=PreflightResult(
                instrument=worst.impl,
                ok=False,
                regime=regime,
                refusal=Refusal(
                    instrument=quantity,
                    reason=RefusalReason.ENVELOPE_VIOLATED,
                    detail=_condition_detail(regime, failed),
                    remedy=_envelope_remedy(regime, failed),
                    statistics={"conditions": [c.name for c in failed], "rung": worst.rung},
                ),
            ),
        )
    chosen = admitted[-1]
    unmeasured = status[chosen.impl][1]

    if reference is not None and chosen.calibrated_against and not reference.is_certified:
        return Assessment(
            quantity=quantity,
            preflight=PreflightResult(
                instrument=chosen.impl,
                ok=False,
                refusal=uncertified_refusal(quantity, reference),
            ),
        )

    return Assessment(
        quantity=quantity,
        preflight=PreflightResult(
            instrument=chosen.impl,
            ok=True,
            rung=chosen.rung,
            cost=chosen.cost,
            regime=regime,
            unchecked=_unchecked_for(regime, substrate, phase, lod, chosen, unmeasured, reference),
            notes=(
                ()
                if chosen.run is not None
                else (
                    f"the rung is registered and {chosen.impl} has no implementation yet, so this "
                    f"row is what the estimator would cost rather than a result you can collect "
                    f"today",
                )
            ),
        ),
    )


def _assess_catalogue(
    instrument: CatalogueInstrument,
    substrate_reading: SubstrateReading,
    phase: Phase | None,
    regime: RegimeReading | None,
) -> Assessment | None:
    """What a not-yet-built instrument's own declarations already rule out on this run.

    Only the three structured fields are consulted: `substrates`, `phases` and `envelope_requires`.
    The catalogue's `access_min` is prose and is never parsed, so no access refusal is produced
    here and no availability is ever claimed. Returning None means "nothing to say yet", and the
    instrument lands under SPECIFIED, NOT YET BUILT instead.
    """
    substrate = substrate_reading.substrate
    quantity = instrument.headline_quantity
    stub = f"{instrument.id} ({instrument.name}) is specified and not yet built"
    # `planned` is the one status worth adding to the stub: the row is scheduled and nobody has run
    # it. `OPEN` adds nothing the stub has not already said, and `built` would contradict it.
    schedule = " (planned)" if _status_word(instrument.status) == "planned" else ""

    if instrument.substrates and substrate is not None and substrate not in instrument.substrates:
        leaves = _leaves_matching(instrument.substrates, substrate_reading)
        return Assessment(
            quantity=quantity,
            built=False,
            instrument_id=instrument.id,
            preflight=PreflightResult(
                instrument=instrument.id,
                ok=False,
                refusal=Refusal(
                    instrument=instrument.id,
                    reason=RefusalReason.SUBSTRATE_MISMATCH,
                    detail=(
                        f"{stub}{schedule}, and it declares "
                        f"{', '.join(sorted(s.name for s in instrument.substrates))}; the grader "
                        f"is {substrate.name}"
                        + (
                            f", whose leaves include {_join(leaves)}"
                            if leaves
                            else ", so building it would not make this reachable"
                        )
                    ),
                    remedy=_substrate_remedy(
                        instrument.substrates, substrate_reading, "an instrument"
                    ),
                ),
            ),
        )

    if instrument.phases and phase is not None and phase not in instrument.phases:
        return Assessment(
            quantity=quantity,
            built=False,
            instrument_id=instrument.id,
            preflight=PreflightResult(
                instrument=instrument.id,
                ok=False,
                refusal=Refusal(
                    instrument=instrument.id,
                    reason=RefusalReason.PHASE_MISMATCH,
                    detail=(
                        f"{stub}{schedule}, and it answers a "
                        f"{'/'.join(sorted(p.name for p in instrument.phases))} question; you are "
                        f"at {phase.name}"
                    ),
                    remedy=(
                        f"ask it during {' or '.join(sorted(p.name for p in instrument.phases))}. "
                        f"Phase decides whether a reading can still change a decision, so this one "
                        f"would be late rather than wrong."
                    ),
                ),
            ),
        )

    failed, unmeasured = _condition_status(instrument.envelope_requires, regime)
    if failed:
        return Assessment(
            quantity=quantity,
            built=False,
            instrument_id=instrument.id,
            preflight=PreflightResult(
                instrument=instrument.id,
                ok=False,
                regime=regime,
                unchecked=_unchecked_for(regime, substrate, phase, None, None, unmeasured),
                refusal=Refusal(
                    instrument=instrument.id,
                    reason=RefusalReason.ENVELOPE_VIOLATED,
                    detail=f"{stub}{schedule}, and its envelope already fails on this run: "
                    + _condition_detail(regime, failed),
                    remedy=_envelope_remedy(regime, failed),
                    statistics={"conditions": [c.name for c in failed], "built": False},
                ),
            ),
        )
    return None


def capability_report(
    access: AccessMatrix | AccessResolution,
    substrate: Substrate | SubstrateReading | str | None = None,
    phase: Phase | PhaseResolution | str | None = None,
    regime: RegimeReading | None = None,
    *,
    lod: LimitOfDetection | None = None,
    reference: ReferenceMaterial | None = None,
    instruments: Sequence[Any] = (),
    context: Context | None = None,
    catalogue: Sequence[CatalogueInstrument] | None = None,
) -> CapabilityReport:
    """What this access, substrate, phase and regime can measure, and what it would cost.

    The signature is widened from the plain matrix form so the resolvers in this package can hand
    over what they know. `access` takes an `AccessMatrix` or an `AccessResolution`; the second
    carries the per-component notes the report prints in parentheses, and the first renders with
    "supplied by the caller" against every row, which is the truth about a bare matrix.

    ``instruments`` accepts real `Instrument` objects and calls their own `preflight`, so an
    instrument that computes a genuine expected uncertainty reports it here rather than "unknown".
    Nothing is computed either way: `preflight` is defined to do no work.
    """
    _ensure_quantities_loaded()
    _ensure_estimators_loaded()

    resolution = access if isinstance(access, AccessResolution) else resolution_from_matrix(access)
    matrix = resolution.matrix

    substrate_reading = _as_substrate_reading(substrate)
    phase_resolution = _as_phase_resolution(phase)
    substrate_value = substrate_reading.substrate
    phase_value = phase_resolution.phase if phase is not None else None

    rows = tuple(catalogue) if catalogue is not None else load_instrument_catalogue()

    assessments: list[Assessment] = []
    covered: set[str] = set()

    for quantity in sorted({e.quantity for e in ESTIMATORS.values()}):
        assessments.append(
            _assess_registered(
                quantity, matrix, substrate_reading, phase_value, regime, lod, reference
            )
        )
        covered.add(quantity)

    for inst in instruments:
        ctx = context or Context(
            signal=None,  # type: ignore[arg-type]
            access=matrix,
            substrate=substrate_value,
            phase=phase_value,
            regime_reading=regime,
            lod=lod,
        )
        pre = inst.preflight(ctx)
        quantity = getattr(inst, "quantity", "") or getattr(inst, "name", "instrument")
        assessments.append(Assessment(quantity=quantity, preflight=pre))
        covered.add(quantity)

    not_built: list[NotBuilt] = []
    for row in rows:
        if any(q in covered for q in row.quantities):
            continue
        assessment = _assess_catalogue(row, substrate_reading, phase_value, regime)
        if assessment is not None:
            assessments.append(assessment)
            continue
        not_built.append(
            NotBuilt(
                id=row.id,
                name=row.name,
                quantity=row.headline_quantity,
                status=row.status,
                access_min=row.access_min,
                wedge=row.wedge,
                # The catalogue's own verdict on whether an implementation exists, which is the
                # only thing that separates "nobody wrote this" from "this ships and has no
                # registered cost model". Reading it here is also what stops `status` being a
                # field the file carries and nothing consults.
                built=(row.status or "").strip().lower() == "built",
            )
        )

    unchecked: list[str] = []
    if substrate_value is None:
        unchecked.append(
            "substrate: unresolved, so no instrument's substrate declaration was enforced. "
            + (
                substrate_reading.refusal.remedy
                if substrate_reading.refusal is not None
                else "Pass the grader or declare it."
            )
        )
    if phase is None:
        unchecked.append(
            "phase: not supplied, so no instrument's phase declaration was enforced. Pass "
            "--phase, or let the resolver read it from the record."
        )
    if regime is None:
        unchecked.append(
            "regime: not measured, so every envelope in the report is unchecked rather than "
            "satisfied. An unchecked precondition is the one failure mode access cannot see."
        )
    if lod is None:
        unchecked.append(
            "limit of detection: no floor was supplied for this configuration, so no reading was "
            "compared against the substrate's disagreement with itself. Characterise one with a "
            "blank sweep before quoting a small effect."
        )
    if reference is None:
        unchecked.append(
            "reference material: none supplied, so nothing here is calibrated and every reading "
            "it produces will be EXPLORATORY."
        )
    for component in resolution.components:
        if component.unresolved:
            unchecked.append(
                f"access {component.component.name}: {' and '.join(component.unresolved)} not "
                f"established. The note beside {component.component.name} above says what would."
            )
    if substrate_reading.truncated:
        unchecked.append(
            "score tree: the walk stopped early, either at the depth limit or on a cycle, so the "
            "leaf list below is partial."
        )

    return CapabilityReport(
        access=resolution,
        substrate=substrate_reading,
        phase=phase_resolution,
        regime=regime,
        assessments=tuple(assessments),
        not_built=tuple(not_built),
        unchecked=tuple(unchecked),
        catalogue_size=len(rows),
    )


def _as_substrate_reading(value: Any) -> SubstrateReading:
    if isinstance(value, SubstrateReading):
        return value
    if isinstance(value, Substrate):
        return SubstrateReading(substrate=value, note="supplied by the caller")
    if isinstance(value, str):
        try:
            return SubstrateReading(
                substrate=Substrate[value.strip().upper()], note="supplied by the caller"
            )
        except KeyError:
            pass
    return SubstrateReading(
        substrate=None,
        note="not supplied",
        refusal=Refusal(
            instrument="access.substrate",
            reason=RefusalReason.SUBSTRATE_MISMATCH,
            detail="no substrate was supplied or resolved",
            remedy=(
                "pass the grader so it can be classified, or name the substrate directly. Series "
                "A, B and D are all gated on it."
            ),
        ),
    )


def _as_phase_resolution(value: Any) -> PhaseResolution:
    if isinstance(value, PhaseResolution):
        return value
    if isinstance(value, Phase):
        return PhaseResolution(value, "supplied by the caller", declared=True)
    if isinstance(value, str):
        try:
            return PhaseResolution(
                Phase[value.strip().upper()], "supplied by the caller", declared=True
            )
        except KeyError:
            pass
    return PhaseResolution(
        Phase.POST_RUN,
        "not supplied, so no instrument's phase declaration was enforced",
        enforced=False,
    )


__all__ = [
    "REMEDY_FOR_CONDITION",
    "Assessment",
    "CapabilityReport",
    "CatalogueInstrument",
    "NotBuilt",
    "capability_report",
    "load_instrument_catalogue",
]

"""D7, the grader card: thirteen readings about one grader, on one page.

A card is a composition, and the thing it composes is `Reading`, which is `Evidence | Refusal`.
That is the whole design and every other decision in this module follows from it.

**Most graders will refuse most fields, and that is the artifact.** A grader nobody instrumented
has no exploit log, no replicated scoring design and no recorded abstention channel, so eight or
nine of the thirteen come back as refusals. A card that rendered those as blanks, zeros, or
omissions would be indistinguishable from a card about a grader that had been measured and found
clean, and that confusion is the thing this project exists to remove. So every field renders
either its reading or its refusal, and the refusal renders its remedy, because the remedy is what
turns "this is unknown" into a list of what to go and record.

**The card states its own trust and no caller can set it.** `Evidence.trust` is computed by
`compute_trust` from the calibration reference and the registration status, and this module has no
argument, field or keyword that touches it. What it does add is the *constituent floor*: the
lowest trust among the readings the card is built from, which is a fact about the composition and
is printed beside the card's own level. A card whose own level is higher than its floor says so in
words, because a composition is not more trustworthy than the weakest thing in it and nothing in
the kernel currently expresses that as a cap.

**Dual use is a property of the row.** D2's surviving-mutant list and D5's false-positive
catalogue are reproducible instructions for making the grader wrong. They arrive carrying
``sensitive=True`` on the payload itself, `CardField.sensitive` is read off the payload rather than
declared here, the rendered card shows the redacted form by default, and the card's own subject is
stamped with the same flag so a store row cannot be mistaken for a publishable one.

**The capability report is the product.** `capability_report` resolves every field, asks each
instrument what it would refuse and what it would cost, and answers "what would this card contain,
and what would it take" without a single grader call. For a buyer deciding whether to pay for a
measurement, that report is the whole transaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Evidence, register_payload
from reward_lens.core.extras import ExtraRequiredError
from reward_lens.core.gates import require_frame_for_comparison
from reward_lens.core.invariance import INVARIANT, TRIVIAL_GROUP
from reward_lens.core.quantity import FREE, CostModel, QuantityID
from reward_lens.core.reading import (
    Reading,
    Refusal,
    RefusalReason,
    refuse_incomplete,
)
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
    TrustLevel,
    missing_access,
)
from reward_lens.measure.base import BaseObservable, Context, PreflightResult, run
from reward_lens.measure.card.fields import CARD_FIELDS, CardInputs, FieldSpec

# ---------------------------------------------------------------------------
# What the card asks for
# ---------------------------------------------------------------------------

#: D7's `access_min`, verbatim: "GRADER:QUERY+REPLICATE and source where available".
#:
#: This is **reported against and not gated on**, and the distinction is the point of the
#: instrument. A reader below this line still gets a card; what they get is a card whose fields
#: refuse, each naming what would let it read. Turning the minimum into a wholesale refusal would
#: make the card unavailable to exactly the reader called Profile A, whose entire
#: deliverable this is. So `requires` is empty, every field declares its own access, and the
#: shortfall against this line is a rendered row rather than a gate.
D7_ACCESS_MIN: AccessMatrix = {Component.GRADER: Access.QUERY | Access.REPLICATE}

#: A card aggregates readings taken over a window, and if the grader was edited inside that window
#: half the card describes a program the other half does not. That is `STATIONARY_GRADER`, and D10
#: is what measures it. The violation downgrades rather than refuses: the readings are still each
#: valid about whichever program produced them, the composition is what stops being one statement,
#: and withholding the page would destroy more information than it protects.
D7_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
    measured_by={RegimeCondition.STATIONARY_GRADER: "env.replay_fidelity"},
    on_violation="downgrade",
)

#: The two dumb baselines a card is a claim against, and both are zero.
#:
#: A published grader card, model card or marketplace listing states none of these thirteen
#: quantities, so the number of them a buyer has today is zero. And an all-refusals card is what
#: this instrument returns for a grader nobody instrumented, which is the floor every filled field
#: has to be read against: three fields filled out of thirteen is a real result and it is also
#: three, not thirteen.
CARD_BASELINES: tuple[str, ...] = ("baseline.published_grader_card", "baseline.all_refusals")

#: What a card is for, in the sentence the rendered header prints.
CARD_PURPOSE = "what a buyer gets before spending money on a grader, and what an auditor gets after"


# ---------------------------------------------------------------------------
# One row
# ---------------------------------------------------------------------------


#: How many lines of an instrument's own report the card carries when that instrument offers no
#: one-line summary of its own. Truncating is the only editing the card does to a reading, and it
#: is visible: the row says how many lines it dropped and where the rest is.
_CONTINUATION_LINES = 3


def _summary_of(value: Any) -> tuple[str, tuple[str, ...]]:
    """The payload's own words, as a headline and whatever the headline left out.

    Each composed payload already knows how to describe itself and the vocabularies differ: some
    expose a `headline`, series A and B emit a mapping carrying a `says` key, and the rest have
    only a `render`. Asking each for its own sentence keeps the card a router. Writing thirteen
    summaries here would be thirteen places for the card's description of a number to drift away
    from the instrument's.

    ``headline`` is checked for being a string rather than for existing, because the name is used
    for two things across the packages this composes: D6 and D10 mean a sentence by it and D1
    means the coverage fraction. A card that printed the float would be labelling one number with
    another instrument's word for it.

    A payload that declines to summarise itself gets the first line of its report plus the next
    few, because the first line of a multi-line report is usually a title and not a finding. That
    is truncation and not selection: the lines are consecutive, in the payload's order, and the row
    says how many it cut.
    """
    headline = getattr(value, "headline", None)
    if isinstance(headline, str) and headline.strip():
        return headline.strip().splitlines()[0], ()
    if isinstance(value, Mapping):
        says = value.get("says")
        if isinstance(says, str) and says.strip():
            return says.strip().splitlines()[0], ()
        return _mapping_summary(value)
    render = getattr(value, "render", None)
    if callable(render):
        lines = [line.strip() for line in str(render()).strip().splitlines() if line.strip()]
        if lines:
            rest = lines[1 : 1 + _CONTINUATION_LINES]
            dropped = len(lines) - 1 - len(rest)
            if dropped > 0:
                rest = [*rest, f"(+{dropped} more lines in this instrument's own report)"]
            return lines[0], tuple(rest)
    if isinstance(value, (int, float, str)):
        return str(value), ()
    return type(value).__name__, ()


#: How many `key=value` pairs go on one line of a mapping payload's summary.
_PAIRS_PER_LINE = 4


def _mapping_summary(value: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    """A mapping payload with no `says` key, printed as its own scalar entries.

    B1 and B2 emit a plain dict rather than a dataclass, so there is no `render` to ask and no
    sentence to quote. Printing the type name instead gave a card row reading "curl mass  dict",
    which is a field that looks filled and says nothing, and is the exact shape this artifact
    exists to remove. Every scalar entry is printed, in the payload's own order, with the
    non-scalar entries counted rather than dropped silently.
    """
    scalars = [
        f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
        for k, v in value.items()
        if isinstance(v, (bool, int, float, str)) and k != "says"
    ]
    nested = [k for k, v in value.items() if not isinstance(v, (bool, int, float, str))]
    chunks = [
        ", ".join(scalars[i : i + _PAIRS_PER_LINE]) for i in range(0, len(scalars), _PAIRS_PER_LINE)
    ]
    if not chunks:
        chunks = ["(no scalar entries)"]
    rest = list(chunks[1 : 1 + _CONTINUATION_LINES])
    dropped = len(chunks) - 1 - len(rest)
    if dropped > 0:
        rest.append(f"(+{dropped} more line(s) of scalar entries on this reading)")
    if nested:
        rest.append(f"structured entries not summarised here: {', '.join(sorted(nested))}")
    return chunks[0], tuple(rest)


#: The keyword each sensitive payload uses for "print the withheld content too". They disagree:
#: `FPCatalogue` calls it `include_entries` and `AttackSurface` calls it `include_targets`, and
#: `MutationReading` has none because its plain `render` already lists the survivors it holds.
#: Listed rather than discovered so a payload that grows a third spelling is a card that has to be
#: told, not a card that quietly stops expanding.
_EXPAND_KEYWORDS: tuple[str, ...] = ("include_entries", "include_targets")


def _expanded_render(value: Any) -> str:
    """The payload's full report, including whatever it withholds by default.

    Only reached when a reader has explicitly asked for the sensitive content and the payload's own
    disclosure gate has let it through.
    """
    import inspect

    render = getattr(value, "render", None)
    if not callable(render):
        return str(value)
    params = inspect.signature(render).parameters
    for keyword in _EXPAND_KEYWORDS:
        if keyword in params:
            return str(render(**{keyword: True}))
    return str(render())


@register_payload
@dataclass(frozen=True)
class CardField:
    """One field of the card: a reading, or the refusal that stands in its place.

    Both cases are represented on one type rather than on a union, because the card's contract is
    that the field is *there* either way. A union would let a renderer drop the refusals by
    pattern-matching on the reading case, which is the failure this artifact exists to prevent.

    The constituent `Evidence` is not carried whole. Its id is, and its value is, and its trust is,
    and those are what the card prints and what a reader follows back into the store. Carrying the
    object itself would put an unregistered payload type inside a registered one, so a card written
    to a store would fail to decode.
    """

    name: str
    quantity: QuantityID
    instrument: str
    #: True when this field carries a reading. False when it carries a refusal.
    read: bool

    # -- the reading
    headline: str = ""
    #: What the headline left out, when the instrument offered no one-line summary of its own.
    continues: tuple[str, ...] = ()
    value: Any = None
    evidence: str = ""
    #: `TrustLevel` as its integer, so a stored card decodes to the same value it was written with.
    trust: int | None = None
    #: Checks the instrument could not run, named. Never counted as passes.
    unchecked: tuple[str, ...] = ()

    # -- the refusal
    reason: str = ""
    detail: str = ""
    remedy: str = ""
    statistics: Mapping[str, Any] = field(default_factory=dict)
    #: The value of `Refusal.partial` when the refusal still carried an honest bound.
    bound: Any = None

    # -- dual use, read off the payload rather than declared here
    sensitive: bool = False
    sensitive_note: str = ""
    #: True on a row whose value has already been swapped for its publishable form.
    redacted_here: bool = False

    #: Grader calls this field's instrument said it would make, from its own preflight.
    calls: int = 0
    #: Whether the instrument priced itself at all. Most return the `FREE` default, which means
    #: "this instrument does not model its cost" and not "this instrument is free". Reporting the
    #: two the same way would let a card total four figures of grader calls and print `free`.
    priced: bool = False

    def __post_init__(self) -> None:
        if not self.read and not self.remedy.strip():
            raise ValueError(
                f"card field {self.name!r} refuses and carries no remedy. A refusal without one is "
                f"a blank with a label on it, which is the artifact this card replaces."
            )

    @property
    def trust_level(self) -> TrustLevel | None:
        return None if self.trust is None else TrustLevel(self.trust)

    @property
    def refusal(self) -> Refusal | None:
        """The refusal, rebuilt.

        ``partial`` is not restored as `Evidence`: the bound's *value* survives on `bound` and the
        Evidence wrapper does not, because carrying it would make this payload undecodable. A
        caller who needs the bound as Evidence should read the constituent reading from the store
        by its id.
        """
        if self.read:
            return None
        return Refusal(
            instrument=self.instrument,
            reason=RefusalReason[self.reason],
            detail=self.detail,
            remedy=self.remedy,
            statistics=dict(self.statistics),
        )

    def redacted(self) -> "CardField":
        """The publishable form of this row: the counts survive, the reproducers do not."""
        if not self.sensitive or self.value is None:
            return self
        from reward_lens.verifier import redact

        return replace(self, value=redact(self.value), sensitive=False, redacted_here=True)

    def for_publication(self) -> "CardField":
        """The unredacted row, and only where the payload's own disclosure gate allows it.

        A sensitive payload decides its own publication rule. `FPCatalogue` and `AttackSurface`
        raise `DisclosureRequired` without a signed `DisclosureDecision`; a payload that defines no
        such gate is returned as it is. Asking the payload rather than deciding here is what keeps
        the rule with the data instead of in a renderer nobody audits.
        """
        gate = getattr(self.value, "for_publication", None)
        if callable(gate):
            return replace(self, value=gate())
        return self

    def render(self, *, width: int = 26, unchecked: bool = False) -> str:
        """This row.

        ``unchecked`` prints the per-field skipped checks, which the one-page card aggregates into
        its header instead of repeating on every row.

        A sensitive row that has *not* been redacted is one a reader asked for and a disclosure
        gate released, so it prints the payload's full report rather than the summary. Anything
        else would make the explicit request a no-op that looks like it worked.
        """
        head = f"  {self.name:<{width}}"
        pad = f"{'':<{width + 2}}"
        if self.read:
            level = self.trust_level
            trust = f"  [{level.name.lower()}]" if level is not None else ""
            if self.sensitive and not self.redacted_here and self.value is not None:
                body = _expanded_render(self.value).splitlines()
                lines = [f"{head}{body[0] if body else self.headline}{trust}"]
                lines.extend(f"{pad}{line.strip()}" for line in body[1:])
            else:
                lines = [f"{head}{self.headline}{trust}"]
                lines.extend(f"{pad}{line}" for line in self.continues)
            if self.redacted_here:
                lines.append(f"{pad}reproducers withheld. {self.sensitive_note}")
            if unchecked:
                lines.extend(f"{pad}not checked: {name}" for name in self.unchecked)
            return "\n".join(lines)
        lines = [f"{head}REFUSED  {self.reason}", f"{pad}{self.detail}"]
        if self.bound is not None:
            lines.append(f"{pad}Bound: {self.bound}")
        lines.append(f"{pad}Remedy: {self.remedy}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class CardReading:
    """The card: thirteen fields, the access they were read at, and what they cost.

    The card's own trust level is deliberately absent from this payload. It is computed by the
    gates and lives on the `Evidence` this is the value of, and putting a copy here would create a
    second place for it to be set. `render_card` reads it off the Evidence.
    """

    grader: str
    subject: str
    substrate: str
    phase: str
    access: str
    fields: tuple[CardField, ...]
    #: Access D7 asks for and this reader does not have, per component. Empty means the reader
    #: meets D7's stated minimum.
    access_shortfall: Mapping[str, str] = field(default_factory=dict)
    access_min: str = ""
    #: What the card's own envelope check found, in words.
    envelope: str = ""
    #: Set when the envelope downgraded rather than refused, naming the ceiling that applies.
    trust_cap: str = ""
    unchecked: tuple[str, ...] = ()
    calls: int = 0
    baselines: Mapping[str, float] = field(default_factory=dict)

    # -- census -------------------------------------------------------------

    @property
    def n_fields(self) -> int:
        return len(self.fields)

    @property
    def read_fields(self) -> tuple[CardField, ...]:
        return tuple(f for f in self.fields if f.read)

    @property
    def refused_fields(self) -> tuple[CardField, ...]:
        return tuple(f for f in self.fields if not f.read)

    @property
    def by_reason(self) -> dict[str, int]:
        """How the refusals divide. The shape of a card is mostly this histogram."""
        out: dict[str, int] = {}
        for f in self.refused_fields:
            out[f.reason] = out.get(f.reason, 0) + 1
        return dict(sorted(out.items()))

    @property
    def constituent_trust(self) -> TrustLevel | None:
        """The lowest trust among the readings this card is built from.

        None when nothing read, which is a different statement from EXPLORATORY and is why this is
        not defaulted to the bottom of the ladder.
        """
        levels = [f.trust for f in self.read_fields if f.trust is not None]
        return TrustLevel(min(levels)) if levels else None

    @property
    def cost(self) -> CostModel:
        """What the card cost, and how much of that is a real total.

        Only the instruments that model their own cost contribute a number. The rest return the
        `FREE` default from `BaseObservable.preflight`, which means nobody wrote a cost model and
        not that the field is free: D2 re-grades the whole corpus once per mutant and prices itself
        at nothing. So the total is reported as a floor, with the count of fields it does not cover.
        """
        priced = [f for f in self.read_fields if f.priced]
        silent = len(self.read_fields) - len(priced)
        note = f"{self.calls:,} grader calls" if self.calls else "no field priced itself"
        if silent:
            note += (
                f", which is a floor: {silent} of the {len(self.read_fields)} fields that read do "
                f"not model their own cost"
            )
        return CostModel(calls=self.calls or None, note=note)

    @property
    def unchecked_census(self) -> dict[str, tuple[str, ...]]:
        """Which checks were skipped, and on which fields. Never counted as passes."""
        out: dict[str, list[str]] = {}
        for f in self.read_fields:
            for name in f.unchecked:
                out.setdefault(name, []).append(f.name)
        return {k: tuple(v) for k, v in sorted(out.items())}

    @property
    def says(self) -> str:
        n_read, n = len(self.read_fields), self.n_fields
        floor = self.constituent_trust
        tail = (
            f" The lowest trust among them is {floor.name.lower()}."
            if floor is not None
            else " Nothing read, so this card is a list of what it would take."
        )
        return f"{n_read} of {n} fields read and {n - n_read} refused.{tail}"

    # -- dual use -----------------------------------------------------------

    @property
    def sensitive(self) -> bool:
        return any(f.sensitive for f in self.fields)

    @property
    def sensitive_fields(self) -> tuple[CardField, ...]:
        return tuple(f for f in self.fields if f.sensitive)

    def redacted(self) -> "CardReading":
        """Every sensitive field swapped for its publishable form. What a rendered card gets."""
        return replace(self, fields=tuple(f.redacted() for f in self.fields))

    def for_publication(self) -> "CardReading":
        """The unredacted card, subject to each sensitive payload's own disclosure gate."""
        return replace(self, fields=tuple(f.for_publication() for f in self.fields))

    # -- presentation -------------------------------------------------------

    def render(self, *, include_sensitive: bool = False, trust: TrustLevel | None = None) -> str:
        """One page.

        ``include_sensitive`` is the explicit request the dual-use rule requires. It does
        not override the payload's own gate: a `FPCatalogue` with no recorded `DisclosureDecision`
        still raises `DisclosureRequired`, because the flag on this call says the reader wants the
        exploit content and the decision says somebody accountable agreed to release it, and those
        are different questions.
        """
        card = self.for_publication() if include_sensitive else self.redacted()
        width = max((len(f.name) for f in card.fields), default=10) + 2
        lines = [
            f"GRADER CARD  {card.grader}",
            f"  {CARD_PURPOSE}",
            "",
            f"  subject        {card.subject}",
            f"  substrate      {card.substrate}    phase  {card.phase}",
            f"  access         {card.access}",
        ]
        if card.access_shortfall:
            gap = ", ".join(f"{c}: {a}" for c, a in sorted(card.access_shortfall.items()))
            lines.append(
                f"                 short of D7's stated minimum ({card.access_min}) by {gap}"
            )
        else:
            lines.append(f"                 meets D7's stated minimum ({card.access_min})")
        if trust is not None:
            lines.append(f"  trust          {trust.name.lower()}, computed by the gates")
        floor = card.constituent_trust
        if floor is not None:
            lines.append(f"  lowest reading {floor.name.lower()}")
            if trust is not None and floor < trust:
                lines.append(
                    f"                 no claim on this card is stronger than "
                    f"{floor.name.lower()}, whatever the card's own level says"
                )
        lines.append(f"  envelope       {card.envelope}")
        if card.trust_cap:
            lines.append(f"                 {card.trust_cap}")
        lines.append(f"  cost           {card.cost.render()}")
        census = dict(card.unchecked_census)
        skipped = []
        for name in sorted(set(census) | set(card.unchecked)):
            where = []
            if name in census:
                where.append(f"{len(census[name])} field(s)")
            if name in card.unchecked:
                where.append("the card itself")
            skipped.append(f"{name} on {' and '.join(where)}")
        if skipped:
            lines.append(f"  not checked    {'; '.join(skipped)}")
        lines.append("")
        lines.append(f"  {card.says}")
        lines.append("")
        for f in card.fields:
            lines.append(f.render(width=width))
        withheld = [f for f in card.fields if f.redacted_here]
        if withheld:
            lines.append("")
            lines.append(
                f"  {len(withheld)} field(s) withheld their reproducers: "
                f"{', '.join(f.name for f in withheld)}. Pass include_sensitive=True with a "
                f"recorded DisclosureDecision on the payload to see them."
            )
        return "\n".join(lines)


def render_card(reading: Reading, *, include_sensitive: bool = False) -> str:
    """Render an `Evidence[CardReading]` or the `Refusal` that came back instead.

    Takes the reading rather than the payload so the card's gate-computed trust level appears on
    the page. That level lives on the Evidence and nowhere else, which is what stops a caller
    handing a card a trust level of its own choosing.
    """
    if isinstance(reading, Refusal):
        return reading.render()
    value = getattr(reading, "value", None)
    if not isinstance(value, CardReading):
        raise TypeError(
            f"render_card takes the reading returned by GraderCard.estimate, whose value is a "
            f"CardReading; got {type(value).__name__}"
        )
    return value.render(include_sensitive=include_sensitive, trust=reading.trust)


# ---------------------------------------------------------------------------
# The capability report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldPlan:
    """What one field would contain and what it would cost, before anything runs."""

    name: str
    quantity: QuantityID
    instrument: str
    available: bool
    rung: int = 0
    calls: int = 0
    #: Whether the instrument models its own cost. False means unpriced, not free.
    priced: bool = False
    reason: str = ""
    detail: str = ""
    remedy: str = ""
    unchecked: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def render(self, *, width: int = 26) -> str:
        head = f"  {self.name:<{width}}"
        if self.available:
            cost = (
                CostModel(calls=self.calls or None).render()
                if self.priced
                else "cost not modelled by this instrument"
            )
            lines = [f"{head}available at rung {self.rung}, {cost}"]
            for note in self.notes:
                lines.append(f"{'':<{width + 2}}{note}")
            for name in self.unchecked:
                lines.append(f"{'':<{width + 2}}not checked: {name}")
            return "\n".join(lines)
        return (
            f"{head}would refuse: {self.reason}\n"
            f"{'':<{width + 2}}{self.detail}\n"
            f"{'':<{width + 2}}Remedy: {self.remedy}"
        )


@dataclass(frozen=True)
class CardPlan:
    """What this card would contain and what it would cost, with no grader called.

    For most readers this is the product. A marketplace buyer deciding whether to pay for a
    measurement wants to know which of the thirteen their access reaches, what the rest would need,
    and how many grader calls the answer costs, and every one of those is answerable before the
    first call.
    """

    grader: str
    fields: tuple[FieldPlan, ...]
    access_shortfall: Mapping[str, str] = field(default_factory=dict)
    access_min: str = ""
    unchecked: tuple[str, ...] = ()

    @property
    def available(self) -> tuple[FieldPlan, ...]:
        return tuple(f for f in self.fields if f.available)

    @property
    def refused(self) -> tuple[FieldPlan, ...]:
        return tuple(f for f in self.fields if not f.available)

    @property
    def cost(self) -> CostModel:
        """A floor, and it says so. Only the instruments that model their cost contribute."""
        priced = [f for f in self.available if f.priced]
        calls = sum(f.calls for f in priced)
        silent = len(self.available) - len(priced)
        note = f"at least {calls:,} grader calls, no GPU and no model"
        if silent:
            note += (
                f". {silent} of the {len(self.available)} available fields do not model their own "
                f"cost, so this is a floor and not a total"
            )
        return CostModel(calls=calls or None, note=note)

    def render(self) -> str:
        width = max((len(f.name) for f in self.fields), default=10) + 2
        lines = [
            f"CARD PLAN  {self.grader}",
            f"  {len(self.available)} of {len(self.fields)} fields would read; "
            f"{len(self.refused)} would refuse.",
            f"  cost  {self.cost.render()}",
        ]
        if self.access_shortfall:
            gap = ", ".join(f"{c}: {a}" for c, a in sorted(self.access_shortfall.items()))
            lines.append(f"  short of D7's stated minimum ({self.access_min}) by {gap}")
        for name in self.unchecked:
            lines.append(f"  not checked  {name}")
        lines.append("")
        for f in self.fields:
            lines.append(f.render(width=width))
        lines.append("")
        lines.append(
            "  A field whose payload declares itself sensitive is withheld from the rendered "
            "card. Which fields those are is a property of the reading, so it is known once the "
            "reading is taken and not before."
        )
        lines.append(
            "  A field that would read here can still refuse once it runs, when the data turns "
            "out not to support the estimator. That is a smaller number of readings than this "
            "plan promises and it is the honest one."
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Refusals the card itself produces
# ---------------------------------------------------------------------------


def _extra_refusal(spec: FieldSpec, exc: ExtraRequiredError) -> Refusal:
    """The instrument's module needs an optional extra that is not installed.

    `ACCESS_INSUFFICIENT` is the least wrong of the sixteen and it is not a comfortable fit. The
    reader's access to the *grader* is fine; what is missing is a package. It is used because
    the test for which reason applies is whether the remedy is answerable where the reader is
    standing or upstream, and one pip install is as answerable-where-you-stand as a remedy gets.
    The statistics carry ``missing_extra`` so a card can group on the real cause rather than on the
    reason name.
    """
    return Refusal(
        instrument=spec.attr,
        reason=RefusalReason.ACCESS_INSUFFICIENT,
        detail=(
            f"{spec.attr} is in {spec.module}, which is not importable here: {exc}. No estimator "
            f"for {spec.quantity} exists on this install, so nothing was computed."
        ),
        remedy=(
            f"install the optional extra: pip install 'reward-lens[{spec.extra}]'. Every other "
            f"field on this card reads without it."
        ),
        statistics={"missing_extra": spec.extra, "quantity": spec.quantity},
    )


def _applicability_refusal(cls: Any, ctx: Context) -> Refusal | None:
    """Substrate, phase, access and envelope, asked of the class rather than of an instance.

    Three of the composed instruments take their subject as a required positional argument, so
    when that subject is the very thing missing there is no instance to run `preflight` on, and the
    card still has to say whether the field was applicable in the first place. Answering
    "you did not supply a comparison flow" to a reader whose grader is a program, when B1 does not
    apply to programs at all, sends them to collect data that will not help.

    The check is the kernel's own, run against an uninitialised instance. `BaseObservable.preflight`
    reads only class attributes and the context, so this asks the class a question about the reader
    and never touches instrument state. `BaseObservable.preflight` is named explicitly rather than
    reached through the instance, because two of the three overrides read constructor state to
    price themselves and would raise here; all three call this first and add only a cost estimate.
    """
    probe = cls.__new__(cls)
    result = BaseObservable.preflight(probe, ctx)
    return result.refusal if not result.ok else None


def _absent_input_refusal(spec: FieldSpec, cls: Any, inputs: CardInputs, ctx: Context) -> Refusal:
    """What is in the way, in the order the reader can act on.

    Applicability first, because a substrate mismatch is a category error and no amount of data
    fixes it, and an access gap is answerable by getting access rather than by collecting
    something. Only when the reader could have supplied the input does this reach
    `RECORD_INCOMPLETE`, which is the honest reason for the remaining case: the access is
    sufficient, the instrument applies, and the field was never recorded, so nothing the reader
    does to what they are holding recovers it and the fix is upstream in whatever produced it.
    """
    blocked = _applicability_refusal(cls, ctx)
    if blocked is not None:
        return blocked
    gaps = inputs.missing(spec.needs)
    return refuse_incomplete(
        cls.name,
        field=spec.describe_missing(inputs),
        subject="the record this card was built from",
        remedy=spec.remedy,
        quantity=spec.quantity,
        missing=list(gaps),
        needs=list(spec.needs),
    )


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


class GraderCard(BaseObservable):
    """D7 `grader.card`: the composite artifact and the wedge product.

    Construct it with a `CardInputs` holding whatever you have about one grader, and call
    `estimate` with a `Context` saying what you can reach. Call `capability_report` first if you
    want to know what it would contain and what it would cost, which needs no grader call.

    Kill condition, from the catalogue: **if nobody reads one.** That is a fact about adoption
    rather than about the instrument, and what the instrument itself can control is that the page
    is short, that no field is silently absent, and that every refusal on it names something the
    reader can go and do.
    """

    name = "GraderCard"
    version = "1.0"
    quantity = "grader.card"
    #: The card calls no signal itself; each composed instrument declares what it needs of one.
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    #: Nothing, and this is a decision rather than an omission. See `D7_ACCESS_MIN`: a card that
    #: refuses wholesale below D7's access minimum is unavailable to the reader it is for.
    requires: AccessMatrix = {}
    #: The catalogue prints OPEN for both of these. A card is a composition of readings and every
    #: substrate admits some subset of the thirteen, so all six are declared rather than left
    #: unrestricted, and the same for the four phases: a card is asked before a purchase, during a
    #: run, after one, and of a deployed artifact. Both are proposed as catalogue fills rather than
    #: left as an empty set meaning "not stated".
    substrates = frozenset(Substrate)
    phases = frozenset(Phase)
    envelope = D7_ENVELOPE
    #: `spec/QUANTITIES.yaml` declares `none` for `grader.card`, and this is what that resolves to.
    #: It is a decision: relabelling the items or rescaling the reward changes what the individual
    #: readings say and changes nothing about which thirteen fields a card has.
    invariance = TRIVIAL_GROUP
    invariance_relation = INVARIANT
    baselines = CARD_BASELINES
    rung = 0
    #: There is no theory object a composite artifact instantiates. Saying so is the honest
    #: answer; naming the catalogue record here would make a provenance claim into a citation.
    faithful_to: str | None = None
    deviations: tuple[str, ...] = (
        "the access matrix is empty where the catalogue prints `GRADER:QUERY+REPLICATE and source "
        "where available`. The minimum is reported on the card and gated per field rather than "
        "gated on the card, because a wholesale refusal below it withholds the artifact from the "
        "access profile it exists for",
        "the catalogue's `substrates` and `phases` are OPEN and this declares all six and all "
        "four, which is a positive claim rather than a transcription",
    )

    def __init__(
        self,
        inputs: CardInputs | None = None,
        *,
        fields: Sequence[FieldSpec] = CARD_FIELDS,
    ) -> None:
        self.inputs = inputs if inputs is not None else CardInputs()
        self.fields = tuple(fields)
        self._computed: CardReading | None = None

    # -- the capability report ----------------------------------------------

    def capability_report(self, ctx: Context) -> CardPlan:
        """What each field would contain and what it would cost. No grader is called.

        Every instrument that can be built is built and asked its own `preflight`, which is where
        the three costed instruments in series D price themselves: D3 at two calls per
        rollout-relation pair, D4 at `N(2D+2)`, D5 at its family and search draws. Building an
        instrument imports the grader's module where the grader is a source file, which executes
        its module body and calls nothing.
        """
        plans: list[FieldPlan] = []
        for spec in self.fields:
            plans.append(self._plan_field(spec, ctx))
        return CardPlan(
            grader=self._grader_name(ctx),
            fields=tuple(plans),
            access_shortfall=self._shortfall(ctx),
            access_min=_render_access(D7_ACCESS_MIN),
            unchecked=self._own_preflight(ctx).unchecked,
        )

    def _plan_field(self, spec: FieldSpec, ctx: Context) -> FieldPlan:
        field_ctx = _field_context(ctx)
        try:
            cls = spec.resolve()
        except ExtraRequiredError as exc:
            return _plan_refused(spec, spec.attr, _extra_refusal(spec, exc))
        if self.inputs.missing(spec.needs):
            return _plan_refused(
                spec, cls.name, _absent_input_refusal(spec, cls, self.inputs, field_ctx)
            )
        instrument = spec.build(cls, self.inputs)
        pre = instrument.preflight(field_ctx)
        if not pre.ok and pre.refusal is not None:
            return _plan_refused(spec, cls.name, pre.refusal)
        return FieldPlan(
            name=spec.name,
            quantity=spec.quantity,
            instrument=cls.name,
            available=True,
            rung=pre.rung,
            calls=int(pre.cost.calls or 0),
            priced=_is_priced(pre),
            unchecked=pre.unchecked,
            notes=pre.notes,
        )

    def preflight(self, ctx: Context) -> PreflightResult:
        """The card's own preflight, carrying the aggregated cost of the fields that would read.

        The card itself never refuses for want of access, because it declares none. It refuses on
        substrate, phase or a violated envelope exactly as any other instrument does, and its
        envelope downgrades rather than refuses, so in practice the only wholesale refusal a card
        can produce is a category error about what kind of thing it was pointed at.
        """
        base = self._own_preflight(ctx)
        if not base.ok:
            return base
        plan = self.capability_report(ctx)
        available = plan.available
        return PreflightResult(
            instrument=self.name,
            ok=True,
            rung=self.rung,
            cost=plan.cost,
            regime=base.regime,
            unchecked=base.unchecked,
            notes=(
                f"{len(available)} of {len(plan.fields)} fields would read; "
                f"{len(plan.refused)} would refuse.",
                *base.notes,
            ),
            bound_estimator=base.bound_estimator,
            trust_cap=base.trust_cap,
        )

    def _own_preflight(self, ctx: Context) -> PreflightResult:
        return BaseObservable.preflight(self, ctx)

    # -- the card -----------------------------------------------------------

    def compose(self, ctx: Context) -> CardReading:
        """Every field, read or refused, with nothing dropped.

        This is where the one rule that outranks the rest is enforced structurally rather than by
        discipline: the loop appends a `CardField` on every iteration and there is no branch that
        continues without one, so a field cannot go missing by being skipped.
        """
        base = self._own_preflight(ctx)
        rows = [self._read_field(spec, ctx) for spec in self.fields]
        return CardReading(
            grader=self._grader_name(ctx),
            subject=_subject_of(ctx),
            substrate=ctx.substrate.name if ctx.substrate is not None else "not stated",
            phase=ctx.phase.name if ctx.phase is not None else "not stated",
            access=_render_access(ctx.access) if ctx.access is not None else "not stated",
            fields=tuple(rows),
            access_shortfall=self._shortfall(ctx),
            access_min=_render_access(D7_ACCESS_MIN),
            envelope=_envelope_note(base),
            trust_cap=(
                f"the envelope downgraded rather than refused, so nothing on this card is above "
                f"{base.trust_cap.name.lower()}"
                if base.trust_cap is not None
                else ""
            ),
            unchecked=base.unchecked,
            calls=sum(r.calls for r in rows),
            baselines={name: 0.0 for name in CARD_BASELINES},
        )

    def _read_field(self, spec: FieldSpec, ctx: Context) -> CardField:
        field_ctx = _field_context(ctx)
        try:
            cls = spec.resolve()
        except ExtraRequiredError as exc:
            return _field_from_refusal(spec, spec.attr, _extra_refusal(spec, exc))
        if self.inputs.missing(spec.needs):
            return _field_from_refusal(
                spec, cls.name, _absent_input_refusal(spec, cls, self.inputs, field_ctx)
            )
        instrument = spec.build(cls, self.inputs)
        pre = instrument.preflight(field_ctx)
        if not pre.ok and pre.refusal is not None:
            return _field_from_refusal(spec, cls.name, pre.refusal)
        reading = instrument.estimate(field_ctx)
        calls, priced = int(pre.cost.calls or 0), _is_priced(pre)
        if isinstance(reading, Refusal):
            return _field_from_refusal(spec, cls.name, reading, calls=calls, priced=priced)
        return _field_from_evidence(
            spec,
            cls.name,
            reading,
            unchecked=pre.unchecked,
            calls=calls,
            priced=priced,
        )

    # -- the estimate path --------------------------------------------------

    def measure(self, ctx: Context) -> Evidence:
        card = self._computed if self._computed is not None else self.compose(ctx)
        extra: dict[str, str] = {"grader": card.grader, "fields_read": str(len(card.read_fields))}
        if card.sensitive:
            from reward_lens.verifier import SENSITIVE_SUBJECT_EXTRA

            extra.update(SENSITIVE_SUBJECT_EXTRA)
        # No uncertainty. A card has no scalar to put an interval on, and the intervals that do
        # exist are on the constituent readings where they belong. Manufacturing one from the
        # field count would be a number about the card's completeness wearing the name of a
        # measurement error.
        return ctx.emit(card, subject_extra=extra, baselines=card.baselines)

    def estimate(self, ctx: Context) -> Reading:
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        self._computed = self.compose(ctx)
        try:
            if ctx.signal is not None:
                return run(self, ctx)
            if ctx.is_comparison:
                require_frame_for_comparison(self.gauge_status, ctx.frame)
            # Set explicitly on the no-signal path. `run` does this for the signal path, and an
            # instrument that emits without it gets `observable="anonymous"` and `quantity=""` on
            # its row, which is a card nobody can find in a store.
            ctx._observable = self
            try:
                return self.measure(ctx)
            finally:
                ctx._observable = None
        finally:
            self._computed = None

    # -- helpers ------------------------------------------------------------

    def _shortfall(self, ctx: Context) -> dict[str, str]:
        if ctx.access is None:
            return {}
        gap = missing_access(ctx.access, D7_ACCESS_MIN)
        return {
            c.name: a.name or str(int(a)) for c, a in sorted(gap.items(), key=lambda kv: kv[0].name)
        }

    def _grader_name(self, ctx: Context) -> str:
        if self.inputs.grader_name:
            return self.inputs.grader_name
        verifier = self.inputs.verifier
        if verifier is not None:
            return f"{verifier.source_path.name}:{verifier.entrypoint}"
        signal = ctx.signal
        inner = getattr(signal, "verifier", None)
        if inner is not None:
            return f"{inner.source_path.name}:{inner.entrypoint}"
        return getattr(signal, "name", None) or "unnamed grader"


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------


def _is_priced(pre: PreflightResult) -> bool:
    """Whether this instrument modelled its own cost, or took the default and said nothing.

    Identity against the `FREE` singleton rather than `CostModel.is_free`, because the two
    questions are different and only this one is answerable: `PreflightResult.cost` defaults to
    exactly that object, so an instrument that never set it is distinguishable from one that
    costed itself and got zero.
    """
    return pre.cost is not FREE


def _field_context(ctx: Context) -> Context:
    """A per-field context, so one instrument cannot leave state on another's.

    Two of the composed runners emit without setting `Context._observable` when the context carries
    no signal, and one restores it to None rather than to what it found. Giving each field its own
    context makes both harmless instead of order-dependent.
    """
    return replace(ctx, _observable=None)


def _plan_refused(spec: FieldSpec, instrument: str, refusal: Refusal) -> FieldPlan:
    return FieldPlan(
        name=spec.name,
        quantity=spec.quantity,
        instrument=instrument,
        available=False,
        reason=refusal.reason.name,
        detail=refusal.detail,
        remedy=refusal.remedy,
    )


def _field_from_refusal(
    spec: FieldSpec, instrument: str, refusal: Refusal, *, calls: int = 0, priced: bool = False
) -> CardField:
    return CardField(
        name=spec.name,
        quantity=spec.quantity,
        instrument=instrument,
        read=False,
        reason=refusal.reason.name,
        detail=refusal.detail,
        remedy=refusal.remedy,
        statistics=dict(refusal.statistics),
        bound=None if refusal.partial is None else refusal.partial.value,
        calls=calls,
        priced=priced,
    )


def _field_from_evidence(
    spec: FieldSpec,
    instrument: str,
    evidence: Evidence[Any],
    *,
    unchecked: tuple[str, ...] = (),
    calls: int = 0,
    priced: bool = False,
) -> CardField:
    """One row from one reading, with the dual-use flag read off the payload.

    The headline is computed from the *publishable* form of a sensitive payload, so the summary
    line a card carries is safe to render whatever the reader asked for. A headline derived from
    the unredacted object and stored on the row would put exploit content on the publishable path
    and nothing downstream would know.
    """
    from reward_lens.verifier import SENSITIVE_NOTE, is_sensitive, redact

    value = evidence.value
    sensitive = is_sensitive(evidence)
    shown = redact(value) if sensitive else value
    headline, continues = _summary_of(shown)
    return CardField(
        name=spec.name,
        quantity=spec.quantity,
        instrument=instrument,
        read=True,
        headline=headline,
        continues=continues,
        value=value,
        evidence=str(evidence.id),
        trust=int(evidence.trust),
        unchecked=tuple(unchecked),
        sensitive=sensitive,
        # A payload that flags itself and states no reason gets the series-D sentence, which is
        # where the rule is written down. An empty note beside a withheld field would read as an
        # unexplained omission, which is the shape this card exists to remove.
        sensitive_note=str(getattr(value, "sensitive_note", "") or "") or SENSITIVE_NOTE,
        calls=calls,
        priced=priced,
    )


def _render_access(access: AccessMatrix | None) -> str:
    if not access:
        return "nothing"
    return ", ".join(
        f"{c.name}: {a.name or str(int(a))}"
        for c, a in sorted(access.items(), key=lambda kv: kv[0].name)
    )


def _subject_of(ctx: Context) -> str:
    signal = ctx.signal
    meta = getattr(signal, "meta", None)
    fingerprint = getattr(meta, "fingerprint", None)
    return str(fingerprint) if fingerprint else "no subject fingerprint on the context"


def _envelope_note(base: PreflightResult) -> str:
    """What the card's own envelope check found, in a sentence a reader can act on."""
    unmeasured = any(name.startswith("envelope") for name in base.unchecked)
    tail = (
        "A card assembled across a grader edit describes two programs rather than one. D10 "
        "(`env.replay_fidelity`) is what measures whether it was one; pass its ConditionReading "
        "as `regime_reading` to have this checked."
    )
    if unmeasured:
        return f"STATIONARY_GRADER was not measured, so this check did not run. {tail}"
    if base.trust_cap is not None:
        return f"STATIONARY_GRADER does not hold over this window. {tail}"
    return "STATIONARY_GRADER holds over the window this card covers."


# ---------------------------------------------------------------------------
# The one-call form
# ---------------------------------------------------------------------------


def card_context(
    inputs: CardInputs,
    *,
    access: AccessMatrix | None = None,
    substrate: Substrate | None = None,
    phase: Phase | None = None,
    regime_reading: Any = None,
) -> Context:
    """A `Context` describing the grader this card is about.

    Everything is optional and nothing is guessed. A preflight that could not check something names
    it rather than counting it as a pass, so leaving `access` out produces a card that says its
    access was never checked, which is different from a card that says the access was sufficient.

    The substrate defaults to PROGRAM only when a verifier source was supplied, because that is the
    one case where the subject's kind is established by the input rather than asserted by the
    caller.
    """
    signal: Any = None
    if inputs.verifier is not None:
        from reward_lens.verifier import ProgramSubject

        signal = ProgramSubject(inputs.verifier)
        if substrate is None:
            substrate = Substrate.PROGRAM
    return Context(
        signal=signal,
        readout="score",
        access=access,
        substrate=substrate,
        phase=phase,
        regime_reading=regime_reading,
    )


def grader_card(
    inputs: CardInputs,
    *,
    ctx: Context | None = None,
    fields: Sequence[FieldSpec] = CARD_FIELDS,
    **context_kwargs: Any,
) -> Reading:
    """Build a card in one call. The form a report script uses."""
    context = ctx if ctx is not None else card_context(inputs, **context_kwargs)
    return GraderCard(inputs, fields=fields).estimate(context)


def card_plan(
    inputs: CardInputs,
    *,
    ctx: Context | None = None,
    fields: Sequence[FieldSpec] = CARD_FIELDS,
    **context_kwargs: Any,
) -> CardPlan:
    """What the card would contain and what it would cost. No grader is called."""
    context = ctx if ctx is not None else card_context(inputs, **context_kwargs)
    return GraderCard(inputs, fields=fields).capability_report(context)


def refusal_reasons(card: CardReading) -> Iterable[str]:
    """Every reason this card refused for, sorted. What an ordering test compares."""
    return sorted({f.reason for f in card.refused_fields})


__all__ = [
    "CARD_BASELINES",
    "CARD_PURPOSE",
    "D7_ACCESS_MIN",
    "D7_ENVELOPE",
    "CardField",
    "CardPlan",
    "CardReading",
    "FieldPlan",
    "GraderCard",
    "card_context",
    "card_plan",
    "grader_card",
    "refusal_reasons",
    "render_card",
]

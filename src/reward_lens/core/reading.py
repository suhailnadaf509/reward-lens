"""Reading = Evidence | Refusal. A refusal is a value, not an exception.

The rule this module enforces is the one that outranks the rest of the design: **a confident wrong
number is the only unforgivable output.** A `Refusal` carrying a reason, the numbers that produced
it, and a remedy is a correct and valuable return value. It is never an exception, never a silent
downgrade to a worse estimator, never a `None`, and never a zero.

Refusals and exceptions are different things and the distinction is enforced by use, not by
convention. A `Refusal` is for a condition the instrument anticipated: insufficient access, a
violated envelope, an effect below the limit of detection, an uncertified reference. An exception
is for a condition it did not: a corrupt file, a shape mismatch, a bug. Never catch a broad
exception and turn it into a refusal, because that is precisely the ``except Exception: ans = 0.0``
pattern that instrument B4 exists to count.

The field that makes this work in practice is `partial`. "I cannot give you the effective group
size, but I can bound it above by 6.1" is a supported answer, and `bounded_refusal` constructs one
in a single call so it is the easy path rather than an aspiration.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar

if TYPE_CHECKING:  # avoid a cycle: evidence imports nothing from here
    from reward_lens.core.evidence import Evidence
    from reward_lens.core.provenance import Provenance

T = TypeVar("T")


class RefusalReason(enum.Enum):
    """The seventeen anticipated conditions an instrument may refuse on.

    Fifteen were there from the start, and two were added later.

    **The two amendments share one test, and it is the test to apply before proposing a third:
    where is the remedy answerable?**

    `RECORD_INCOMPLETE` is the sixteenth. `ACCESS_INSUFFICIENT` means "no estimator works at
    the access you have", answerable **where the reader is standing** by getting more access or
    dropping a rung. This one means "the access is sufficient and the field was never written",
    answerable **upstream**, where the record was produced. Five sites across two packages were
    using one reason for both, and every one of those remedies had the form "record X and re-run".

    `QUANTITY_UNDEFINED` is the seventeenth, and it is the case answerable **nowhere**: the
    question does not apply to the object. An estimator that does not z-score has no amplification
    to measure, and no access and no rewriting of the record gives it one. It is not
    `SUBSTRATE_MISMATCH` either, which is about the grader's kind rather than the estimator's; the
    two live on different axes and an instrument can be refused on either. Three packages reached
    for it independently before it existed, which is the signal no single report carries and is how
    `Access.SOURCE` was found.
    """

    #: No estimator exists at this access level. Carries the rung that would work.
    ACCESS_INSUFFICIENT = enum.auto()
    #: The access is sufficient and the record does not carry the field.
    RECORD_INCOMPLETE = enum.auto()
    #: Asked a program for its activations. A category error, not a hard case.
    SUBSTRATE_MISMATCH = enum.auto()
    #: Asked an in-run question of a finished artifact.
    PHASE_MISMATCH = enum.auto()
    #: A regime condition failed. Carries WHICH condition, its statistic and its threshold.
    ENVELOPE_VIOLATED = enum.auto()
    #: The effect is smaller than the substrate's disagreement with itself.
    BELOW_LOD = enum.auto()
    #: Detected but not quantifiable. A bound is returned in `partial`.
    ABOVE_LOD_BELOW_LOQ = enum.auto()
    #: An extrapolation past its visibility horizon is a guess wearing a number.
    ESS_BELOW_FLOOR = enum.auto()
    #: A null with no identically-powered positive control is indistinguishable from an
    #: underpowered experiment.
    NO_MATCHED_CONTROL = enum.auto()
    #: A covariant comparison with no shared frame. Coordinate artifacts.
    GAUGE_MISMATCH = enum.auto()
    #: Per-token compared against per-sequence. The most common silent error in this literature.
    UNIT_MISMATCH = enum.auto()
    #: Calibrating against a reference with no stated uncertainty of its own.
    REFERENCE_UNCERTIFIED = enum.auto()
    #: Scoring against labels with no measured error rate measures the labels.
    LABEL_QUALITY_UNKNOWN = enum.auto()
    #: A registered prediction's metric is produced by no arc in the plan.
    PLAN_NOT_CLOSED = enum.auto()
    #: The costed plan exceeds the declared budget.
    BUDGET_EXCEEDED = enum.auto()
    #: The run is not readable at all.
    VOID = enum.auto()
    #: The question does not apply to this object.
    QUANTITY_UNDEFINED = enum.auto()


#: What each reason means, in one sentence, for the refusal reference page. Users open that page
#: more than any other, so these are written for someone holding a failure rather than for someone
#: reading the architecture.
REASON_MEANING: dict[RefusalReason, str] = {
    RefusalReason.ACCESS_INSUFFICIENT: (
        "No estimator for this quantity works at the access you have. Silent degradation to a "
        "worse one is how a number becomes uninterpretable, so nothing was computed."
    ),
    RefusalReason.RECORD_INCOMPLETE: (
        "Your access is sufficient and the record does not carry the field this estimator reads. "
        "Nothing more can be recovered from this record; the fix is upstream, where it was written."
    ),
    RefusalReason.SUBSTRATE_MISMATCH: (
        "This instrument does not apply to this kind of grader. A program has no activations; "
        "that is a category error rather than a hard case."
    ),
    RefusalReason.PHASE_MISMATCH: (
        "This is an in-run question and the run is over, or a pre-run question and it has started."
    ),
    RefusalReason.ENVELOPE_VIOLATED: (
        "The estimator's assumptions do not hold on this run. An instrument that is available and "
        "invalid is worse than one that is unavailable."
    ),
    RefusalReason.BELOW_LOD: (
        "The effect is smaller than the measurement substrate's disagreement with itself, so it is "
        "not attributable to the thing being measured."
    ),
    RefusalReason.ABOVE_LOD_BELOW_LOQ: (
        "Detected but not quantifiable. A bound is returned; a point estimate would be false "
        "precision."
    ),
    RefusalReason.ESS_BELOW_FLOOR: (
        "The importance weights have degenerated, so this is past the visibility horizon and any "
        "number would be a guess wearing an interval."
    ),
    RefusalReason.NO_MATCHED_CONTROL: (
        "A null with no identically-powered positive control cannot be distinguished from an "
        "underpowered experiment."
    ),
    RefusalReason.GAUGE_MISMATCH: (
        "A covariant quantity was compared across frames with no shared basis, so the difference "
        "would be a coordinate artifact."
    ),
    RefusalReason.UNIT_MISMATCH: (
        "Two quantities in incompatible units were compared. The conversion factor is a property "
        "of the data, not of the unit, so this is not converted silently."
    ),
    RefusalReason.REFERENCE_UNCERTIFIED: (
        "The reference material carries no uncertainty of its own. You cannot calibrate against an "
        "uncalibrated ruler."
    ),
    RefusalReason.LABEL_QUALITY_UNKNOWN: (
        "The labels have no measured error rate, so scoring against them measures the labels."
    ),
    RefusalReason.PLAN_NOT_CLOSED: (
        "A registered prediction names a metric that no arc in this plan produces. Found before "
        "anything ran."
    ),
    RefusalReason.BUDGET_EXCEEDED: "The costed plan exceeds the declared budget.",
    RefusalReason.VOID: "The run is not readable, which is different from a negative result.",
    RefusalReason.QUANTITY_UNDEFINED: (
        "This quantity is not defined for this object, so there is nothing here to measure at any "
        "access and from any record. The remedy names the question that does apply instead."
    ),
}


@dataclass(frozen=True)
class Refusal:
    """A declined measurement, with the numbers and what to do about it.

    ``detail`` carries what failed, with the numbers. ``remedy`` is the sentence the user acts on
    and it is a user interface: write it as an instruction. "Restrict the window to steps 0-239,
    or supply the weight schedule so the composition can be held fixed counterfactually" is a
    remedy. "Envelope violated" is not.
    """

    instrument: str
    reason: RefusalReason
    detail: str
    remedy: str
    partial: "Evidence[Any] | None" = None
    provenance: "Provenance | None" = None
    #: Free-form numbers behind the refusal, so the reason is auditable rather than asserted.
    statistics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.remedy.strip():
            raise ValueError(
                f"refusal {self.reason.name} from {self.instrument!r} carries no remedy. A refusal "
                f"without one is a tool that looks broken instead of a tool that looks careful."
            )

    @property
    def is_bounded(self) -> bool:
        """Whether this refusal still carries an honest bound."""
        return self.partial is not None

    @property
    def meaning(self) -> str:
        return REASON_MEANING[self.reason]

    def render(self) -> str:
        lines = [f"{self.instrument}  {self.reason.name}", f"    {self.detail}"]
        if self.partial is not None:
            lines.append(f"    Bound: {self.partial.value}")
        lines.append(f"    Remedy: {self.remedy}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


class ReadingResult(Generic[T]):
    """Marker for documentation; the runtime type is the union below."""


if TYPE_CHECKING:
    #: The return type of every instrument. Never a bare float, never a silent degradation.
    #:
    #: A real union under the type checker, so `-> Reading` on an instrument that returns a float
    #: is an error mypy catches. It was `Any`, which made every such annotation vacuous and left
    #: the one rule this module exists to state unenforced by the one tool that could enforce it.
    #: At runtime it stays `Any`: `evidence` imports nothing from this module and this module
    #: must not start importing `evidence`, because `Refusal.partial` holds an `Evidence` and the
    #: dependency would run both ways.
    Reading: TypeAlias = "Evidence[Any] | Refusal"
else:
    Reading = Any


def is_refusal(reading: Any) -> bool:
    return isinstance(reading, Refusal)


def value_or_none(reading: Any) -> Any:
    """The value if this is Evidence, else None.

    Deliberately not called ``unwrap``. A caller that reaches for this is discarding a remedy, so
    the name should not read like the normal path.
    """
    return None if isinstance(reading, Refusal) else getattr(reading, "value", None)


def bounded_refusal(
    instrument: str,
    reason: RefusalReason,
    *,
    detail: str,
    remedy: str,
    bound: "Evidence[Any]",
    **statistics: Any,
) -> Refusal:
    """A refusal that still answers, partially.

    The single-call constructor exists so that returning a bound is the easy path. If it were
    three lines of assembly, instruments would return a bare refusal and the bound would be lost,
    which is the difference between "I cannot tell you" and "I cannot tell you exactly, but it is
    at most 6.1".
    """
    return Refusal(
        instrument=instrument,
        reason=reason,
        detail=detail,
        remedy=remedy,
        partial=bound,
        statistics=statistics,
    )


def refuse_access(instrument: str, *, needs: dict[str, str], have: str, remedy: str) -> Refusal:
    """The commonest refusal, with its detail assembled from the access gap."""
    gap = ", ".join(f"{component}: {access}" for component, access in sorted(needs.items()))
    return Refusal(
        instrument=instrument,
        reason=RefusalReason.ACCESS_INSUFFICIENT,
        detail=f"needs {gap}; you have {have}",
        remedy=remedy,
        statistics={"missing": needs},
    )


def refuse_incomplete(
    instrument: str,
    *,
    field: str,
    subject: str,
    remedy: str,
    **statistics: Any,
) -> Refusal:
    """The record has the shape and not the field.

    Kept separate from `refuse_access` because the two send the reader in opposite directions. An
    access refusal is answerable where the reader is standing, by getting more access or dropping a
    rung. This one is not: the field was never written, so nothing the reader does to this record
    recovers it and the fix is upstream in whatever produced it. A remedy that says "get more
    access" when the honest answer is "your framework does not dump this" costs somebody an
    afternoon and then still does not work.

    ``field`` is what is missing and ``subject`` is what it is missing from, because "no
    `logprobs_sampling`" is half a sentence and "no `logprobs_sampling` on 412 of 512 trajectories"
    is the half a reader can act on.
    """
    return Refusal(
        instrument=instrument,
        reason=RefusalReason.RECORD_INCOMPLETE,
        detail=f"{subject} carries no {field}",
        remedy=remedy,
        statistics={"field": field, "subject": subject, **statistics},
    )


def refuse_undefined(
    instrument: str,
    *,
    quantity: str,
    subject: str,
    instead: str,
    remedy: str,
    **statistics: Any,
) -> Refusal:
    """The question does not apply to this object.

    Kept separate from `refuse_access` and `refuse_incomplete` because the three send the reader in
    three different directions, and one test separates them: **is the remedy answerable where the
    reader is standing?** An access refusal is answerable there, by getting
    more access or dropping a rung. A record-incomplete refusal is answerable upstream, where the
    record was written. This one is answerable **nowhere**, because no amount of access and no
    rewriting of the record gives a mean-centred estimator an amplification ratio: the quantity is
    not defined for that object.

    It is also not `SUBSTRATE_MISMATCH`, which is about the grader's kind (a program has no
    activations) rather than about the estimator's. The two live on different axes and an instrument
    can be refused on either.

    ``instead`` is what makes this refusal worth returning rather than merely correct. When the
    asked question does not apply, the only useful sentence available is the name of the question
    that does, so it is a required argument rather than an optional courtesy.

    Three packages reached for this independently before it existed, which is the signal no single
    report carries and is exactly how `Access.SOURCE` was found.
    """
    return Refusal(
        instrument=instrument,
        reason=RefusalReason.QUANTITY_UNDEFINED,
        detail=f"{quantity} is not defined for {subject}",
        remedy=remedy,
        statistics={"quantity": quantity, "subject": subject, "instead": instead, **statistics},
    )


__all__ = [
    "REASON_MEANING",
    "Reading",
    "ReadingResult",
    "Refusal",
    "RefusalReason",
    "bounded_refusal",
    "is_refusal",
    "refuse_access",
    "refuse_incomplete",
    "refuse_undefined",
    "value_or_none",
]

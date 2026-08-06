"""M11, rung disagreement, published.

When two rungs of one estimator ladder run on the same data and give different answers, the
difference is not a discrepancy to reconcile. It is the cheap rung's transfer uncertainty against
the expensive one, measured on this substrate, and it is the one term of the calibration chain
that falls out of an experiment somebody was going to run anyway. Nobody publishes it.

**This instrument must not resolve the disagreement, and that is the whole design.** There is no
`preferred`, no `best`, no `resolved_value` and no rule that picks the higher rung. A reader who
wants one number after reading this has misunderstood what the reading is: the deliverable is both
numbers, both intervals, the access each rung required, and the difference between them as a
`Transfer` that composes into every later budget built on the cheap rung. Picking a winner throws
away the measurement and keeps the estimate.

The three ways this can be got wrong, each of which is a refusal here:

**Different quantities.** Two rungs are comparable because they estimate the same defined thing.
Two estimators of different quantities differing is not a transfer term, it is two facts, and
subtracting them produces a number in no unit at all.

**Different data.** The disagreement is a transfer term only if the two rungs saw identical data.
Run them on different item sets and the difference contains a sampling term nobody can separate out.

**No access statement.** The point of publishing a disagreement is that a reader can decide whether
the expensive rung is worth buying. A difference with no statement of what each rung cost in access
does not answer that, so it is incomplete rather than cheap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import (
    QUANTITIES,
    BaselineID,
    BiasStatement,
    CostModel,
    EstimatorEntry,
    register_estimator,
    what_would_it_take,
)
from reward_lens.core.reading import Refusal, RefusalReason, refuse_incomplete
from reward_lens.core.reference import Transfer, ladder_disagreement
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.meta._base import MetaInstrument

#: Reading two finished readings needs the record they are in and nothing else. That is the whole
#: economic argument for M11: the expensive rung was already paid for by whoever ran it.
RUNG_ACCESS: dict[Component, Access] = {Component.RECORD: Access.RECORD}

#: Two numbers a published disagreement has to be read against. The first is what a paper that
#: quotes only the cheap rung is implicitly claiming; the second is what the cheap rung's own
#: interval would have suggested the disagreement could be.
RUNG_BASELINES: tuple[BaselineID, ...] = (
    "baseline.rungs_agree",
    "baseline.cheap_rung_interval_width",
)

RUNG_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "a difference between two readings that were each taken under their own envelope. This "
        "instrument adds no assumption about the run: whatever conditions the two rungs required "
        "travel on the two readings, and the subtraction cannot be made wrong by a regime that "
        "left both of them valid."
    ),
)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RungReading:
    """One rung's answer, with the access it needed and what it cost to get.

    ``access`` is a sentence rather than an `AccessMatrix` because it has to be readable on a card:
    "GRADER: RECORD, free" and "GRADER: REPLICATE, 11 raters x 7,052 objects" are what a reader
    compares. ``requires`` carries the machine-readable form alongside for anything that wants to
    check reachability.
    """

    rung: int
    value: float
    #: The item count both rungs must share. Two rungs on different data do not make a transfer.
    n: int
    access: str
    estimator: str = ""
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    ci_level: float = 0.95
    cost: str = ""
    bias: str = ""
    requires: AccessMatrix = field(default_factory=dict)
    #: Which registered quantity this rung estimates. Two rungs are comparable because they
    #: estimate the same defined thing, so a rung that names one is checkable against its partner
    #: and a rung that names none is taken on the instrument's word.
    quantity: str = ""
    note: str = ""

    @property
    def has_interval(self) -> bool:
        return math.isfinite(self.ci_low) and math.isfinite(self.ci_high)

    @property
    def half_width(self) -> float:
        return (self.ci_high - self.ci_low) / 2.0 if self.has_interval else float("nan")

    def render(self) -> str:
        interval = f" [{self.ci_low:.4g}, {self.ci_high:.4g}]" if self.has_interval else ""
        return (
            f"rung {self.rung}: {self.value:.4g}{interval}  ({self.access}"
            + (f", {self.cost}" if self.cost else "")
            + ")"
        )


def rung_from_effective_size(reading: Any, *, access: str = "", estimator: str = "") -> RungReading:
    """Adapt one of A1's effective-group-size readings into a rung of this ladder.

    A1 is the ladder with the most rungs in the catalogue and the one whose rungs demonstrably
    disagree, so the adapter is worth shipping rather than leaving every caller to write it. The
    access sentence defaults to what the catalogue prints for the rung: `GRADER: RECORD` at rung 0
    and `GRADER: REPLICATE` from rung 2 up.
    """
    rung = int(reading.rung)
    default_access = "GRADER: RECORD" if rung == 0 else "GRADER: REPLICATE"
    return RungReading(
        rung=rung,
        value=float(reading.n_eff),
        n=int(reading.n_groups),
        access=access or default_access,
        estimator=estimator or reading.method,
        ci_low=float(reading.ci_low),
        ci_high=float(reading.ci_high),
        ci_level=float(reading.ci_level),
        cost="free" if rung == 0 else "one scored cell per object per rater",
        bias=str(reading.bias),
        requires={Component.GRADER: Access.RECORD if rung == 0 else Access.REPLICATE},
        quantity="grader.effective_group_size",
        note=reading.universe,
    )


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class Disagreement:
    """Two rungs, both published, and the transfer term their difference is.

    Every field is symmetric in the two rungs or is explicitly labelled cheap or expensive by the
    access each required, never by which number is larger and never by which one is preferred.
    """

    quantity: str
    cheap_rung: int
    expensive_rung: int
    cheap_value: float
    expensive_value: float
    cheap_ci: tuple[float, float]
    expensive_ci: tuple[float, float]
    cheap_access: str
    expensive_access: str
    cheap_estimator: str
    expensive_estimator: str
    cheap_cost: str
    expensive_cost: str
    delta: float
    relative_delta: float
    intervals_overlap: bool
    n: int
    transfer_name: str
    transfer_method: str
    baselines: Mapping[str, float] = field(default_factory=dict)

    def transfer(self) -> Transfer:
        """The difference as a chain term, so it composes into every budget downstream."""
        return ladder_disagreement(
            self.cheap_value,
            self.expensive_value,
            from_level="working_method",
            to_level="reference_method",
            n=self.n,
            method=self.transfer_method,
        )

    def says(self) -> str:
        return (
            f"{self.quantity}: rung {self.cheap_rung} reads {self.cheap_value:.4g} and rung "
            f"{self.expensive_rung} reads {self.expensive_value:.4g} on the same {self.n:,} "
            f"observations. The disagreement is {self.delta:.4g}, which is "
            f"{self.relative_delta:.0%} of the cheap rung's own reading, and it is what rung "
            f"{self.cheap_rung} costs in accuracy on this substrate. Rung {self.cheap_rung} needs "
            f"{self.cheap_access} and rung {self.expensive_rung} needs {self.expensive_access}; "
            f"both numbers stand, and the difference between them is the reading."
        )

    def render(self) -> str:
        lines = [
            self.says(),
            f"    rung {self.cheap_rung}: {self.cheap_value:.4g} "
            f"[{self.cheap_ci[0]:.4g}, {self.cheap_ci[1]:.4g}]  ({self.cheap_access}, "
            f"{self.cheap_cost})",
            f"    rung {self.expensive_rung}: {self.expensive_value:.4g} "
            f"[{self.expensive_ci[0]:.4g}, {self.expensive_ci[1]:.4g}]  "
            f"({self.expensive_access}, {self.expensive_cost})",
            f"    {self.transfer().render()}",
        ]
        if self.intervals_overlap:
            lines.append(
                "    the two intervals overlap, so the disagreement is not established at this n. "
                "That is a reading about the ladder and it is what the kill condition turns on."
            )
        return "\n".join(lines)


def _unit_mismatch(instrument: str, cheap: RungReading, expensive: RungReading) -> Refusal | None:
    """Whether the two rungs estimate one defined quantity, or the refusal saying they do not.

    Two rungs are comparable because they estimate the same thing. Subtracting readings of two
    different quantities produces a number in no unit at all, and it is exactly the case where the
    arithmetic looks fine, which is why this is checked before anything is computed.
    """
    a, b = cheap.quantity, expensive.quantity
    if not a or not b or a == b:
        return None
    qa = QUANTITIES.get(a) if a in QUANTITIES else None
    qb = QUANTITIES.get(b) if b in QUANTITIES else None
    units = ""
    if qa is not None and qb is not None:
        units = (
            f" Their units are {qa.unit} and {qb.unit}, which are "
            f"{'compatible, which does not make them one quantity' if qa.unit.compatible_with(qb.unit) else 'not compatible'}."
        )
    return Refusal(
        instrument=instrument,
        reason=RefusalReason.UNIT_MISMATCH,
        detail=(
            f"rung {cheap.rung} estimates {a!r} and rung {expensive.rung} estimates {b!r}, so the "
            f"two are not rungs of one ladder and their difference is not a transfer term.{units}"
        ),
        remedy=(
            "compare two rungs of the same quantity. If the two readings are genuinely of different "
            "quantities, what you have is two facts rather than a disagreement, and the difference "
            "between them is not a calibration term and must not be composed into a chain as one."
        ),
        statistics={"cheap_quantity": a, "expensive_quantity": b},
    )


def compare_rungs(cheap: RungReading, expensive: RungReading) -> Any:
    """Two rungs in, a `Reading` out. The one-call form, and the subject of the `units` check.

    Exists as a named function rather than only as a constructor call because the `units` group's
    assertion is about a *comparison*: `check_unit_refusal` needs something it can hand two objects
    to and watch refuse.
    """
    return RungDisagreement(cheap, expensive).compute()


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


class RungDisagreement(MetaInstrument):
    """M11. Two rungs of one ladder disagreeing, published as a transfer term.

    Constructed from two `RungReading`s of the same quantity on the same data, or from a whole
    ladder with `rungs=[...]`, in which case the cheapest and the most expensive are compared and
    the intermediate rungs are carried in the payload rather than dropped.
    """

    name = "RungDisagreement"
    version = "1.0"
    quantity = "ladder.rung_disagreement"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    requires = RUNG_ACCESS
    substrates = frozenset(
        {
            Substrate.NEURAL_SCALAR,
            Substrate.NEURAL_GEN,
            Substrate.PROGRAM,
            Substrate.PROCEDURAL,
            Substrate.HUMAN,
            Substrate.COMPOSITE,
        }
    )
    phases = frozenset({Phase.PRE_RUN, Phase.IN_RUN, Phase.POST_RUN, Phase.DEPLOYED})
    envelope = RUNG_ENVELOPE
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = RUNG_BASELINES
    rung = 0
    faithful_to = "M11"
    deviations = (
        "the interval on the disagreement combines the two rungs' half-widths in quadrature, which "
        "assumes they are independent. They are not: both rungs read the same data, so their "
        "errors are positively correlated and the true interval is narrower. The quadrature "
        "interval is therefore an upper bound and is labelled as one on the reading",
    )

    def __init__(
        self,
        cheap: RungReading | None = None,
        expensive: RungReading | None = None,
        *,
        quantity_id: str = "",
        rungs: Sequence[RungReading] = (),
        subject: str = "",
    ) -> None:
        ordered = sorted(rungs, key=lambda r: r.rung) if rungs else []
        if ordered and cheap is None and expensive is None:
            cheap = ordered[0]
            expensive = ordered[-1]
        self.cheap = cheap
        self.expensive = expensive
        self.rungs = tuple(ordered)
        declared = {r.quantity for r in (cheap, expensive) if r is not None and r.quantity}
        self.quantity_id = quantity_id or (declared.pop() if len(declared) == 1 else "")
        self.subject = subject

    def compute(self) -> Any:
        cheap, expensive = self.cheap, self.expensive
        if cheap is None or expensive is None:
            return self._one_rung_refusal(cheap or expensive)
        mismatch = _unit_mismatch(self.name, cheap, expensive)
        if mismatch is not None:
            return mismatch
        if cheap.rung == expensive.rung:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    f"both readings are rung {cheap.rung}, so there is no ladder step between them "
                    f"and no transfer to publish"
                ),
                remedy=(
                    "run a second rung of the same ladder on the same data. `ladder(quantity)` "
                    "lists every registered estimator with the access each needs, and the point of "
                    "this instrument is that the step you can afford is worth measuring against "
                    "the one you cannot."
                ),
                statistics={"rung": cheap.rung},
            )
        for label, reading in (("cheap", cheap), ("expensive", expensive)):
            if not reading.access.strip():
                return refuse_incomplete(
                    self.name,
                    field="an access statement",
                    subject=f"the {label} rung (rung {reading.rung})",
                    remedy=(
                        "record what access the rung needed before publishing its disagreement. A "
                        "difference with no statement of what each rung cost does not answer the "
                        "question a reader has, which is whether the expensive rung is worth "
                        "buying. Set `access=` on the RungReading."
                    ),
                    rung=reading.rung,
                )
        if cheap.n != expensive.n:
            return refuse_incomplete(
                self.name,
                field="a common item set",
                subject=(
                    f"the two rungs (rung {cheap.rung} on {cheap.n:,}, rung {expensive.rung} on "
                    f"{expensive.n:,})"
                ),
                remedy=(
                    "re-run both rungs on identical data. A difference between rungs on different "
                    "item sets carries a sampling term that cannot be separated from the transfer "
                    "term, so it is not a calibration quantity and must not be composed into a "
                    "chain as one."
                ),
                cheap_n=cheap.n,
                expensive_n=expensive.n,
            )

        qid = self.quantity_id
        if qid and qid not in QUANTITIES:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.UNIT_MISMATCH,
                detail=(
                    f"the two rungs are declared to estimate {qid!r}, which spec/QUANTITIES.yaml "
                    f"does not carry, so there is no unit in which their difference is expressed"
                ),
                remedy=(
                    "name the registered id the two rungs estimate, or register it. Two rungs are "
                    "comparable because they estimate the same defined quantity, and an id that "
                    "resolves to nothing is not a definition."
                ),
                statistics={"quantity": qid},
            )

        delta = abs(cheap.value - expensive.value)
        denominator = abs(cheap.value)
        overlap = bool(
            cheap.has_interval
            and expensive.has_interval
            and cheap.ci_low <= expensive.ci_high
            and expensive.ci_low <= cheap.ci_high
        )
        method = (
            f"{qid or 'the same quantity'}, rung {cheap.rung} against rung {expensive.rung}, same "
            f"{cheap.n:,} observations" + (f", {self.subject}" if self.subject else "")
        )
        return Disagreement(
            quantity=qid,
            cheap_rung=cheap.rung,
            expensive_rung=expensive.rung,
            cheap_value=cheap.value,
            expensive_value=expensive.value,
            cheap_ci=(cheap.ci_low, cheap.ci_high),
            expensive_ci=(expensive.ci_low, expensive.ci_high),
            cheap_access=cheap.access,
            expensive_access=expensive.access,
            cheap_estimator=cheap.estimator,
            expensive_estimator=expensive.estimator,
            cheap_cost=cheap.cost,
            expensive_cost=expensive.cost,
            delta=delta,
            relative_delta=(delta / denominator) if denominator > 0 else float("nan"),
            intervals_overlap=overlap,
            n=cheap.n,
            transfer_name=ladder_disagreement(cheap.value, expensive.value).name,
            transfer_method=method,
            baselines={
                # What a paper quoting only the cheap rung is implicitly claiming.
                "baseline.rungs_agree": 0.0,
                # What the cheap rung's own interval would have suggested the gap could be.
                "baseline.cheap_rung_interval_width": (
                    cheap.ci_high - cheap.ci_low if cheap.has_interval else float("nan")
                ),
            },
        )

    def _one_rung_refusal(self, only: RungReading | None) -> Refusal:
        """One rung is not a disagreement, and the remedy names the rung that would make it one."""
        stats: dict[str, Any] = {"rungs_supplied": 1 if only is not None else 0}
        remedy = (
            "run a second rung of the same ladder on the same data and pass both. "
            "`ladder(quantity)` lists every registered estimator with its rung, its access and its "
            "cost, and `what_would_it_take(quantity, access)` names the cheapest rung your access "
            "reaches."
        )
        if self.quantity_id and self.quantity_id in QUANTITIES:
            gap = what_would_it_take(self.quantity_id, dict(only.requires) if only else {})
            stats["what_would_it_take"] = gap
            if gap.get("needs"):
                needs = ", ".join(f"{c}: {a}" for c, a in sorted(gap["needs"].items()))
                remedy = (
                    f"the next rung is {gap.get('impl')} at rung {gap.get('rung')} and it needs "
                    f"{needs} ({gap.get('cost')}). Get that access, run it on the same data, and "
                    f"pass both readings. Until then there is one number and no measurement of what "
                    f"it costs."
                )
        return Refusal(
            instrument=self.name,
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=(
                "a rung disagreement needs two rungs of one ladder on identical data, and "
                f"{stats['rungs_supplied']} was supplied"
            ),
            remedy=remedy,
            statistics=stats,
        )

    def uncertainty(self, computed: Disagreement) -> Uncertainty | None:
        """The two half-widths in quadrature, labelled as the upper bound it is.

        The two rungs read the same data, so their errors are positively correlated and the honest
        interval on the difference is narrower than this. Quoting the narrower one would need the
        correlation, which neither rung reports, so the bound is what gets published and the method
        string says so rather than leaving a reader to assume independence.
        """
        lo_c, hi_c = computed.cheap_ci
        lo_e, hi_e = computed.expensive_ci
        if not all(math.isfinite(x) for x in (lo_c, hi_c, lo_e, hi_e)):
            return Uncertainty(n=computed.n, method="no interval on at least one rung")
        half = math.hypot((hi_c - lo_c) / 2.0, (hi_e - lo_e) / 2.0)
        return Uncertainty(
            ci_low=max(0.0, computed.delta - half),
            ci_high=computed.delta + half,
            ci_level=0.95,
            n=computed.n,
            method=(
                "the two rungs' half-widths in quadrature; an upper bound, because both rungs read "
                "the same data and their errors are positively correlated"
            ),
        )

    def payload(self, computed: Disagreement) -> dict[str, Any]:
        out: dict[str, Any] = {
            "quantity": computed.quantity,
            "cheap_rung": computed.cheap_rung,
            "expensive_rung": computed.expensive_rung,
            "cheap_value": computed.cheap_value,
            "expensive_value": computed.expensive_value,
            "cheap_ci_low": computed.cheap_ci[0],
            "cheap_ci_high": computed.cheap_ci[1],
            "expensive_ci_low": computed.expensive_ci[0],
            "expensive_ci_high": computed.expensive_ci[1],
            "cheap_access": computed.cheap_access,
            "expensive_access": computed.expensive_access,
            "cheap_estimator": computed.cheap_estimator,
            "expensive_estimator": computed.expensive_estimator,
            "cheap_cost": computed.cheap_cost,
            "expensive_cost": computed.expensive_cost,
            "delta": computed.delta,
            "relative_delta": computed.relative_delta,
            "intervals_overlap": computed.intervals_overlap,
            "n": computed.n,
            "transfer_name": computed.transfer_name,
            "transfer_method": computed.transfer_method,
            "baselines": dict(computed.baselines),
            "says": computed.says(),
        }
        if len(self.rungs) > 2:
            out["intermediate_rungs"] = [
                {
                    "rung": r.rung,
                    "value": r.value,
                    "ci_low": r.ci_low,
                    "ci_high": r.ci_high,
                    "access": r.access,
                }
                for r in self.rungs[1:-1]
            ]
        return out


def register_ladder() -> list[str]:
    """Register M11's single rung. Not called at import, by design."""
    entry = EstimatorEntry(
        quantity="ladder.rung_disagreement",
        impl="m11.paired_rung_difference",
        requires=RUNG_ACCESS,
        envelope=RUNG_ENVELOPE,
        rung=0,
        bias=BiasStatement(
            direction="approximately_unbiased",
            why=(
                "the difference of two readings taken on identical data. It inherits whatever bias "
                "each rung has, which is the point: the quantity is the gap between the two, and a "
                "gap between two biased estimators is exactly what the cheap one costs"
            ),
        ),
        cost=CostModel(note="free once both rungs have run"),
        run=None,
    )
    register_estimator(entry)
    return [entry.impl]


__all__ = [
    "RUNG_ACCESS",
    "RUNG_BASELINES",
    "RUNG_ENVELOPE",
    "Disagreement",
    "RungDisagreement",
    "RungReading",
    "compare_rungs",
    "register_ladder",
    "rung_from_effective_size",
]

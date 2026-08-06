"""Quantities and estimators are different things.

A **quantity** is what you want to know. An **estimator** is one way to get it, at a stated access
level, with a stated bias, at a stated cost. Conflating them is what forces a library to have one
architecture per access profile, and it is why the shipped 2.0.1 either computes a thing or does
not: there is no vocabulary in it for "I can give you this, cheaply, biased upward, and here is
what the better answer would cost."

Four consequences follow from separating them, and the third is the one nobody publishes.

A closed lab gets a real answer at rung 0 with an honest bias direction, rather than a refusal.
Claims become comparable, because "ESS = 4.2 (rung 0)" and "ESS = 2.1 (rung 3)" are two claims
about one quantity and the ladder says so. **Rung disagreement is a deliverable**: when two rungs
disagree on the same data, that difference is the cheap method's transfer uncertainty, and it
composes into the calibration chain. And a quantity registered with no estimator is an open
research target rather than a bug, which is a thing the docs build should say out loud.

`Unit` does real work here rather than being a label. A per-token KL compared against a
per-sequence KL is the most common silent error in this literature, and putting `Unit` inside
`Quantity` is what turns it into a type error instead of a plausible number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.types import (
    AccessMatrix,
    Phase,
    Substrate,
    missing_access,
    satisfies,
)

QuantityID = str
EstimatorID = str
InvarianceGroupID = str
ReferenceID = str
BaselineID = str


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Unit:
    """Not a string. A unit that knows what it cannot be compared to.

    Three axes, because they fail independently. ``dimension`` is what is being counted
    ("nats", "count", "probability", "correlation", "reward", "1"). ``per`` is what it is counted
    over ("token", "sequence", "step", "group", or None for an extensive quantity). ``scale`` is
    the convention it is expressed in ("nats", "bits", "raw", "normalised").

    A quantity per token and the same quantity per sequence are **different quantities**, not the
    same quantity in different clothes, and `compatible_with` returning False is what raises
    `UNIT_MISMATCH` rather than silently converting. Silent conversion is the wrong behaviour even
    when the conversion is arithmetically available, because the factor is a property of the data
    (how many tokens?) rather than of the unit, so converting requires information the comparison
    does not have.
    """

    dimension: str
    per: str | None = None
    scale: str | None = None
    #: The token as printed in the registry, kept so the decomposition is checkable.
    as_printed: str = ""

    @property
    def is_decided(self) -> bool:
        """Whether every axis has actually been decided.

        24 of the 128 registry rows carry `OPEN` on all three axes, because the registry carries one
        undecomposed token and some tokens (`tree`, `matrix`, `spectrum`, `graph`) do not decompose
        unambiguously into dimension, per and scale. That is honest, and it is why `as_printed`
        travels with every row.
        """
        return "OPEN" not in (self.dimension, self.per, self.scale)

    def compatible_with(self, other: "Unit") -> bool:
        """Whether a value in this unit may be compared with, added to, or ranked against another.

        **An undecided axis makes a unit incomparable, including with another undecided one.**
        Unknown is not a value, and encoding it as the string `OPEN` made it one: before this
        check, the 24 rows whose decomposition nobody has settled were all mutually compatible, so
        a score tree compared cleanly against an exploit transfer matrix and a confidence interval.
        276 pairs, every one of them wrong, and invisibly so.

        Refusing is the right direction to err in. A quantity whose unit has not been decided is
        exactly the case where a silent comparison is most likely to be meaningless, and the remedy
        is one a user can act on: decide the decomposition and record it.
        """
        if not (self.is_decided and other.is_decided):
            return False
        return (
            self.dimension == other.dimension
            and self.per == other.per
            and self.scale == other.scale
        )

    def __str__(self) -> str:
        s = self.dimension
        if self.per:
            s += f"/{self.per}"
        if self.scale and self.scale != self.dimension:
            s += f" [{self.scale}]"
        return s


DIMENSIONLESS = Unit(dimension="1", as_printed="1")

#: What a deliberate answer of "no group acts on this" resolves to. Distinct from a missing
#: declaration, which is a lint failure.
TRIVIAL_GROUP: str = "trivial"


# ---------------------------------------------------------------------------
# Bias and cost, so a ladder rung can say what it is giving up
# ---------------------------------------------------------------------------

BiasDirection = Literal["upward", "downward", "unknown", "approximately_unbiased"]


@dataclass(frozen=True)
class BiasStatement:
    """Which way a rung is wrong, and why.

    The ``why`` is the field that matters. "Biased upward because it cannot see correlated grader
    error at all" is the sentence the whole ladder exists to be able to produce; "biased upward"
    on its own tells a reader nothing they can act on.
    """

    direction: BiasDirection
    why: str
    magnitude: float | None = None

    def __str__(self) -> str:
        mag = f" by about {self.magnitude:g}" if self.magnitude is not None else ""
        return f"{self.direction}{mag}: {self.why}"


@dataclass(frozen=True)
class CostModel:
    """What a rung costs, in every currency it can cost anything in.

    All optional, because a cost model that can express only one of them is useless: the capability
    report prints "420 calls, $1.10" and a rung that costs eighteen minutes of CPU and nothing else
    has to be able to say that without inventing a dollar figure.
    """

    calls: int | None = None
    gpu_seconds: float | None = None
    cpu_seconds: float | None = None
    dollars: float | None = None
    note: str = ""

    @property
    def is_free(self) -> bool:
        return not any((self.calls, self.gpu_seconds, self.cpu_seconds, self.dollars))

    def render(self) -> str:
        if self.note:
            return self.note
        if self.is_free:
            return "free"
        bits = []
        if self.calls:
            bits.append(f"{self.calls:,} calls")
        if self.gpu_seconds:
            bits.append(f"{self.gpu_seconds:.0f}s GPU")
        if self.cpu_seconds:
            bits.append(f"{self.cpu_seconds / 60:.0f} min CPU")
        if self.dollars:
            bits.append(f"${self.dollars:,.2f}")
        return ", ".join(bits)


FREE = CostModel()


# ---------------------------------------------------------------------------
# The two registered types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Quantity:
    """What you want to know, defined once, unambiguously.

    ``definition`` is the mathematical object and it has to be unambiguous, because the entire
    point of separating quantity from estimator is that two rungs estimate *the same thing*. Two
    rungs that silently estimate different things are two quantities wearing one name, and the
    ladder becomes a way of hiding that rather than a way of exposing it.
    """

    id: QuantityID
    definition: str
    unit: Unit
    invariance: InvarianceGroupID
    interpretation: str = ""
    support: tuple[float, float] | None = None
    wedge: bool = False

    def __post_init__(self) -> None:
        if not self.id or " " in self.id:
            raise ValueError(f"quantity id must be a dotted token with no spaces; got {self.id!r}")


@dataclass(frozen=True)
class EstimatorEntry:
    """One way to get a quantity, at a stated access level, bias and cost."""

    quantity: QuantityID
    impl: EstimatorID
    requires: AccessMatrix
    envelope: EnvelopeSpec
    rung: int
    bias: BiasStatement
    cost: CostModel = FREE
    substrates: frozenset[Substrate] | None = None  # None means every substrate
    phases: frozenset[Phase] | None = None  # None means every phase
    calibrated_against: ReferenceID | None = None
    #: Filled in by the implementation when there is one; None means the rung is specified and
    #: not yet built, which the capability report reports as such rather than hiding.
    run: Callable[..., Any] | None = None

    def applies_to(self, substrate: Substrate | None, phase: Phase | None) -> bool:
        if (
            substrate is not None
            and self.substrates is not None
            and substrate not in self.substrates
        ):
            return False
        if phase is not None and self.phases is not None and phase not in self.phases:
            return False
        return True


# ---------------------------------------------------------------------------
# The registries
# ---------------------------------------------------------------------------


class Registry:
    """A name-keyed registry that refuses to redefine a name to something different.

    Re-registering an identical entry is a no-op, so importing a module twice is safe. Registering
    a *different* entry under a live name raises, because a quantity whose definition depends on
    import order is not a quantity.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Any] = {}

    def register(self, key: str, value: Any) -> Any:
        existing = self._items.get(key)
        if existing is not None and existing != value:
            raise ValueError(
                f"{self.kind} {key!r} is already registered with a different definition. "
                f"A name that means two things depending on import order is not a name."
            )
        self._items[key] = value
        return value

    def get(self, key: str) -> Any:
        try:
            return self._items[key]
        except KeyError:
            raise KeyError(f"no {self.kind} registered as {key!r}") from None

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def values(self) -> Any:
        return self._items.values()

    def items(self) -> Any:
        return self._items.items()


QUANTITIES = Registry("quantity")

#: Estimators are many-per-quantity, so this is keyed by EstimatorID and indexed by quantity below.
ESTIMATORS = Registry("estimator")


def register_quantity(q: Quantity) -> Quantity:
    return QUANTITIES.register(q.id, q)  # type: ignore[no-any-return]


def register_estimator(e: EstimatorEntry) -> EstimatorEntry:
    if e.quantity not in QUANTITIES:
        raise ValueError(
            f"estimator {e.impl!r} estimates {e.quantity!r}, which is not a registered quantity. "
            f"Register the quantity first, or fix the id: an instrument whose quantity is not "
            f"registered fails at import, by design."
        )
    return ESTIMATORS.register(e.impl, e)  # type: ignore[no-any-return]


def ladder(q: QuantityID) -> list[EstimatorEntry]:
    """Every registered estimator for a quantity, cheapest rung first.

    An empty list is not an error. It means the quantity is an open research target, and the docs
    build says so by name rather than treating it as a bug.
    """
    return sorted(
        (e for e in ESTIMATORS.values() if e.quantity == q), key=lambda e: (e.rung, e.impl)
    )


def open_quantities() -> list[QuantityID]:
    """Registered quantities with no estimator: the roadmap, not the defect list."""
    return sorted(q for q in QUANTITIES if not ladder(q))


def best_estimator(
    q: QuantityID,
    access: AccessMatrix,
    *,
    substrate: Substrate | None = None,
    phase: Phase | None = None,
    regime: Any = None,
) -> EstimatorEntry | None:
    """The highest rung this access, substrate, phase and regime can actually support.

    Highest rather than cheapest: the ladder is ordered by how much it needs, so the best answer
    available is the last one whose requirements are met. Returns None when nothing is reachable,
    which the caller turns into a `Refusal` naming the rung that *would* work and what it needs.
    """
    reachable = [
        e for e in ladder(q) if satisfies(access, e.requires) and e.applies_to(substrate, phase)
    ]
    if regime is not None:
        reachable = [e for e in reachable if e.envelope.admits(regime)]
    return reachable[-1] if reachable else None


def what_would_it_take(q: QuantityID, access: AccessMatrix) -> dict[str, Any]:
    """For a quantity this access cannot reach: the cheapest rung that would, and the gap.

    This is the remedy half of a refusal. "needs POLICY: BACKWARD. Remedy: --policy ckpt/step-*/"
    is a sentence a user can act on; "access insufficient" is not.
    """
    rungs = ladder(q)
    if not rungs:
        return {"reachable": False, "reason": "no estimator is registered for this quantity"}
    for e in rungs:
        gap = missing_access(access, e.requires)
        if not gap:
            return {"reachable": True, "rung": e.rung, "impl": e.impl, "cost": e.cost.render()}
    cheapest = rungs[0]
    gap = missing_access(access, cheapest.requires)
    return {
        "reachable": False,
        "rung": cheapest.rung,
        "impl": cheapest.impl,
        "cost": cheapest.cost.render(),
        "needs": {c.name: a.name for c, a in gap.items()},
    }


# ---------------------------------------------------------------------------
# Loading the registry from the catalogue
# ---------------------------------------------------------------------------

_SPEC_DIR_CANDIDATES = (
    # A source checkout: src/reward_lens/core/quantity.py -> the repo root.
    Path(__file__).resolve().parents[3] / "spec",
    # An installed wheel: site-packages/reward_lens/core/quantity.py -> the packaged data.
    Path(__file__).resolve().parents[1] / "spec",
)


def catalogue_path(name: str) -> Path | None:
    for base in _SPEC_DIR_CANDIDATES:
        p = base / name
        if p.exists():
            return p
    return None


@dataclass
class LoadReport:
    """What loading the catalogue produced, including what it declined to produce."""

    loaded: int = 0
    skipped_open: list[str] = field(default_factory=list)
    trivial_group: list[str] = field(default_factory=list)
    source: str = ""

    def render(self) -> str:
        return (
            f"{self.loaded} quantities loaded from {self.source}; "
            f"{len(self.trivial_group)} declare the trivial group; "
            f"{len(self.skipped_open)} skipped with an OPEN invariance group."
        )


def _read_catalogue(stem: str) -> tuple[Any, str]:
    """Load a catalogue file, preferring the generated JSON over the YAML source.

    The YAML is the human-editable source of truth: it holds the comments that say where each row
    came from and what an OPEN means, and nobody should be hand-editing JSON. But PyYAML ships a
    compiled extension, and the core of this library installs with no compiled dependency, so the
    registry cannot import yaml at load time. The JSON is generated from the YAML and a test
    asserts the two agree, which keeps one source of truth and costs the base install nothing.
    """
    js = catalogue_path(f"{stem}.json")
    if js is not None:
        import json

        return json.loads(js.read_text(encoding="utf-8")), str(js)
    ya = catalogue_path(f"{stem}.yaml")
    if ya is not None:
        import yaml  # only reachable in a dev checkout, where [dev] is installed

        return yaml.safe_load(ya.read_text(encoding="utf-8")), str(ya)
    return None, "(not found)"


def load_quantities(path: Path | None = None) -> LoadReport:
    """Populate `QUANTITIES` from `spec/QUANTITIES.yaml`.

    A quantity whose invariance group is literally OPEN is **not** registered, because every
    registered instrument gets one generated invariance property test and a placeholder group
    would manufacture a test that asserts nothing and then report it as coverage.

    The group ``"none"`` is a different thing and it **is** registered. It is carried for 28
    quantities and it is a declaration, not an omission: a count of grader exceptions is not
    affine-covariant in any useful sense, and saying so is the honest answer rather than a gap.
    The rule that "a new instrument that does not declare a group does not merge" targets
    omission; `TRIVIAL_GROUP` is what a deliberate answer of "none" resolves to, so the two are
    distinguishable in the registry instead of collapsing into each other.
    """
    if path is not None:
        import yaml

        doc, source = yaml.safe_load(path.read_text(encoding="utf-8")), str(path)
    else:
        doc, source = _read_catalogue("QUANTITIES")
    report = LoadReport(source=source)
    if doc is None:
        return report

    for row in doc.get("quantities", []):
        group = row.get("invariance_group", "OPEN")
        if group in (None, "OPEN"):
            report.skipped_open.append(row["id"])
            continue
        if str(group) == "none":
            group = TRIVIAL_GROUP
            report.trivial_group.append(row["id"])
        u = row.get("unit") or {}
        unit = Unit(
            dimension=str(u.get("dimension", "OPEN")),
            per=u.get("per"),
            scale=u.get("scale"),
            as_printed=str(u.get("as_printed", "")),
        )
        definition = row.get("definition") or ""
        register_quantity(
            Quantity(
                id=row["id"],
                definition="" if definition == "OPEN" else str(definition),
                unit=unit,
                invariance=str(group),
                interpretation=""
                if row.get("interpretation") in (None, "OPEN")
                else str(row["interpretation"]),
                wedge=bool(row.get("wedge", False)),
            )
        )
        report.loaded += 1
    return report


__all__ = [
    "DIMENSIONLESS",
    "ESTIMATORS",
    "TRIVIAL_GROUP",
    "FREE",
    "QUANTITIES",
    "BaselineID",
    "BiasDirection",
    "BiasStatement",
    "CostModel",
    "EstimatorEntry",
    "EstimatorID",
    "InvarianceGroupID",
    "LoadReport",
    "Quantity",
    "QuantityID",
    "ReferenceID",
    "Registry",
    "Unit",
    "best_estimator",
    "catalogue_path",
    "ladder",
    "load_quantities",
    "open_quantities",
    "register_estimator",
    "register_quantity",
    "what_would_it_take",
]

"""Identity types, enums, and addressing primitives for the kernel.

Everything comparable across runs gets a stable, content-derived id. Ids are
BLAKE2b-128 digests over canonical serializations, carried with human-readable prefixes so a
glance at a string tells you what kind of thing it names. Content derivation is what lets the
evidence store be a DAG: two runs that computed the same thing from the same inputs land on the
same id, and a derived quantity can point at the leaf measurements it consumed.

The enums here are load-bearing policy, not decoration. `Capability` is the declared contract
that replaces v1's duck-typed `hasattr` discovery. `TrustLevel` is the ladder the three
gates climb; it is an `IntEnum` so "the highest applicable rung" is a max and card
rendering can sort by it. `GaugeStatus` is the typing that makes a raw-coordinate cross-model
number impossible to mistake for an invariant one (I3, gate 2).
"""

from __future__ import annotations

import enum
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, NewType

# ---------------------------------------------------------------------------
# Content-derived identity
# ---------------------------------------------------------------------------

_HASH_BYTES = 16  # 128-bit digest


def canonical_bytes(obj: Any) -> bytes:
    """Serialize an object to canonical bytes for hashing.

    The canonical form is JSON with sorted keys, no insignificant whitespace, and a small
    set of extensions for objects JSON does not natively handle (bytes, sets, tuples via the
    default list coercion, and objects exposing ``__canonical__`` or ``_asdict``). Floats are
    emitted with ``repr`` semantics via json, which is stable within a platform; ids that must
    survive across platforms should hash structural content (shapes, names, integer counts),
    not raw float payloads, and the callers in this codebase do exactly that.
    """

    def _default(o: Any) -> Any:
        if isinstance(o, bytes):
            return {"__bytes__": o.hex()}
        if isinstance(o, (set, frozenset)):
            return {"__set__": sorted(_default(x) if not _json_native(x) else x for x in o)}
        if hasattr(o, "__canonical__"):
            return o.__canonical__()
        if hasattr(o, "_asdict"):  # namedtuple-like
            return o._asdict()
        if hasattr(o, "__dict__"):
            return {k: v for k, v in sorted(vars(o).items()) if not k.startswith("_")}
        raise TypeError(f"cannot canonicalize {type(o).__name__}")

    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_default
    ).encode("utf-8")


def _json_native(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool)) or x is None


def content_hash(obj: Any, prefix: str) -> str:
    """Return ``"{prefix}:{hexdigest}"`` for the canonical serialization of ``obj``.

    The prefix is the human-readable tag (``mfp``, ``ds``, ``ev`` and so on). The digest is
    BLAKE2b truncated to 128 bits, which is collision-safe at the scale of any evidence store
    and short enough to read.
    """
    digest = hashlib.blake2b(canonical_bytes(obj), digest_size=_HASH_BYTES).hexdigest()
    return f"{prefix}:{digest}"


def hash_bytes(data: bytes, prefix: str) -> str:
    """Content hash of raw bytes (streamed file content, tensor buffers)."""
    digest = hashlib.blake2b(data, digest_size=_HASH_BYTES).hexdigest()
    return f"{prefix}:{digest}"


# Stable identifier NewTypes. These are ``str`` at runtime; the NewType is documentation and a
# mypy guard so a DatasetID is never accidentally passed where a ModelFP is expected.
ModelFP = NewType("ModelFP", str)  # "mfp:..."   weights+config+tokenizer hash
DatasetID = NewType("DatasetID", str)  # "ds:..."    dataset card hash (content + builder version)
DirectionID = NewType("DirectionID", str)  # "dir:..." persisted direction/probe hash
FrameID = NewType("FrameID", str)  # "frame:..." gauge frame hash
EvidenceID = NewType("EvidenceID", str)  # "ev:..."   assigned at store append
StudyID = NewType("StudyID", str)  # "study:name@vN#hash"
OrganismID = NewType("OrganismID", str)  # "org:..."


# ---------------------------------------------------------------------------
# Enums: capabilities, trust, gauge
# ---------------------------------------------------------------------------


class Capability(enum.Flag):
    """Declared capabilities of a reward signal.

    Instruments declare `capabilities: Capability`; the runner checks compatibility before any
    GPU work and fails with a precise message. This replaces v1's `hasattr(adapter, ...)`
    duck typing, where a missing method surfaced as a deep AttributeError or, worse, a
    silently skipped code path.

    The field was called `requires` until that name was needed for the access matrix. Both
    concepts survive, so one of them had to move.
    """

    NONE = 0
    SCORES = enum.auto()
    PREFIX_SCORES = enum.auto()
    ACTIVATIONS = enum.auto()
    GRADIENTS = enum.auto()
    HVP = enum.auto()
    LINEAR_READOUT = enum.auto()
    MULTI_READOUT = enum.auto()
    STEP_SCORES = enum.auto()
    DISTRIBUTIONAL = enum.auto()
    SPAN_TYPES = enum.auto()
    GENERATIVE = enum.auto()
    PAIRED_MODELS = enum.auto()

    def missing_from(self, available: "Capability") -> "Capability":
        """Return the subset of ``self`` not present in ``available`` (empty if satisfied)."""
        return self & ~available


class TrustLevel(enum.IntEnum):
    """The trust ladder the three gates compute.

    Ordered so that comparisons and ``max`` express "the highest applicable rung". The level
    is never set by a caller; it is computed from whether the Evidence carries a calibration
    reference, whether it was produced under a frozen study, and whether it survived its kill
    criteria and review. See `reward_lens.core.gates.compute_trust`.
    """

    EXPLORATORY = 0  # default; anything computed ad hoc
    CALIBRATED = 1  # the observable has a scorecard entry covering this signal family + regime
    REGISTERED = 2  # computed under a frozen Study whose predictions predate the run
    ADJUDICATED = 3  # registered + calibrated + survived its own kill criteria and review

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


class GaugeStatus(enum.Enum):
    """How an Observable's value transforms under the reward gauge group (I3, gate 2).

    INVARIANT quantities are safe to compare across signals directly. COVARIANT quantities
    (directions, angles, subspace overlaps) require a shared `Frame` to compare and the
    comparison APIs take a frame argument with no default. RAW_ONLY quantities are computable
    and scientifically interesting (E19 proved this) but are typed as raw coordinates and
    rendered as such, never mistaken for invariant.
    """

    INVARIANT = "invariant"
    COVARIANT = "covariant"
    RAW_ONLY = "raw_only"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# ---------------------------------------------------------------------------
# Addressing: Site and Span
# ---------------------------------------------------------------------------

SitePoint = Literal["resid_pre", "resid_post", "attn_out", "mlp_out", "head_out", "embed"]


@dataclass(frozen=True, order=True)
class Site:
    """A location in a network.

    ``layer`` indexes the transformer block; ``point`` names the read/write surface within it;
    ``head`` selects an attention head where ``point == "head_out"`` and is None otherwise. The
    type is frozen and ordered so it can key the activation cache and sort deterministically in
    reports.
    """

    layer: int
    point: SitePoint = "resid_post"
    head: int | None = None

    def __canonical__(self) -> dict[str, Any]:
        return {"layer": self.layer, "point": self.point, "head": self.head}

    def __str__(self) -> str:
        h = f".h{self.head}" if self.head is not None else ""
        return f"L{self.layer}.{self.point}{h}"


@dataclass(frozen=True)
class Span:
    """A typed token interval ``[start, end)``.

    The ``kind`` tag is what makes span-level patching and attribution meaningful: a receipt
    span, an error step, a critique sentence, a verdict token. Core defines the primitive; the
    data plane's `spans.py` defines the vocabulary of kinds and the character-to-token mapping
    that produces exact spans.
    """

    start: int
    end: int
    kind: str = "text"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"span end {self.end} precedes start {self.start}")

    def __len__(self) -> int:
        return self.end - self.start

    def __canonical__(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "kind": self.kind, "meta": self.meta}


# ---------------------------------------------------------------------------
# SubjectRef: what an Evidence is about
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectRef:
    """The subject of a measurement.

    Names the signal(s) by fingerprint, the dataset, the readout, the frame (for covariant
    quantities), and any interventions applied, by fingerprint. Recording intervention
    fingerprints here is what makes an erased-model card structurally unable to masquerade as a
    base-model card: the interventions are part of the subject's identity.
    """

    signals: tuple[ModelFP, ...] = ()
    dataset: DatasetID | None = None
    readout: str | None = None
    frame: FrameID | None = None
    interventions: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "signals": list(self.signals),
            "dataset": self.dataset,
            "readout": self.readout,
            "frame": self.frame,
            "interventions": list(self.interventions),
            "extra": self.extra,
        }


# ---------------------------------------------------------------------------
# The four typing dimensions
# ---------------------------------------------------------------------------
#
# Which instrument applies is decided by four independent things, and all four belong in the type
# system because all four vary per user, per component and per run. `Capability` above is the
# 2.0 answer to one of them and it is kept: it describes what a *signal* offers. These describe
# what the *analyst* can reach, which is a different question and the one that decides whether an
# instrument exists at all for a given person.


class Component(enum.Enum):
    """The nodes of the loop. Every quantity is a property of one, or of a channel between two."""

    TASK = enum.auto()
    GRADER = enum.auto()
    POLICY = enum.auto()
    ESTIMATOR = enum.auto()
    OPTIMIZER = enum.auto()
    ARTIFACT = enum.auto()
    GOLD = enum.auto()
    RECORD = enum.auto()


class Access(enum.IntFlag):
    """What can be touched, per component.

    Ordered so that ``BACKWARD > FORWARD > QUERY > RECORD`` as a containment claim, and MUTATE and
    CONTROL are orthogonal to that ladder rather than above it.

    ``SOURCE`` is the second member that earns its place, and it is not on the ladder either. The
    catalogue asks for "Access source" on D1, D2 and D9, and none of the other seven flags says it:
    ``FORWARD`` means run it and read activations, which a program does not have, and ``MUTATE``
    means modify it, which reading does not. A verifier's source code, control-flow graph and test
    suite are the substrate the entire D series measures, so without this flag ten of the eighty-five
    catalogued instruments cannot declare what they need.

    The member that earns its place is ``REPLICATE``, and it does **not** follow from ``QUERY``. A
    hosted judge with a fixed internal seed is callable and not facet-varyable, and without facet
    variation there is no variance decomposition, no effective group size and no attenuation
    factor. Collapsing the two would silently delete half of series A for anyone behind an API
    that will not take a seed.
    """

    NONE = 0
    RECORD = 1 << 0  # read logged values that already exist
    QUERY = 1 << 1  # call it again on inputs of my choosing
    REPLICATE = 1 << 2  # call it again under CONTROLLED facet variation
    FORWARD = 1 << 3  # run it and read internal activations
    BACKWARD = 1 << 4  # differentiate through it
    SOURCE = 1 << 5  # read its source code, control-flow graph and test suite
    MUTATE = 1 << 6  # modify it: patch, ablate, edit, plant, recompile
    CONTROL = 1 << 7  # stand up a counterfactual arm of the whole loop

    def missing_from(self, available: "Access") -> "Access":
        """The subset of ``self`` not present in ``available`` (empty if satisfied)."""
        return self & ~available


#: What an analyst can reach, per component. Absent means `Access.NONE`.
AccessMatrix = Mapping[Component, Access]

#: The containment ladder: `BACKWARD > FORWARD > QUERY > RECORD`. Holding a rung
#: implies holding every rung below it, because you cannot differentiate through a model you cannot
#: run, and running it produces a value you could have logged.
#:
#: `REPLICATE`, `MUTATE` and `CONTROL` are deliberately absent from the ladder. They are orthogonal
#: to it, and `REPLICATE` is the one that earns the distinction: a hosted judge with a fixed
#: internal seed is callable and not facet-varyable, so `QUERY` does not give it. Putting it on the
#: ladder would silently delete half of series A for anyone behind an API that will not take a seed.
_LADDER: tuple["Access", ...] = (Access.RECORD, Access.QUERY, Access.FORWARD, Access.BACKWARD)


def expand_access(access: "Access") -> "Access":
    """Add every rung the containment ladder implies.

    `Access` is an `IntFlag` with independent bits, so the containment ordering is not
    something the type gives for free: without this, `have BACKWARD, need RECORD` is False, which
    is wrong and was wrong until it was measured. Expansion happens at comparison time rather than
    at construction so a capability report can still say "you have QUERY" rather than reciting
    every rung underneath it.
    """
    out = access
    highest = -1
    for i, rung in enumerate(_LADDER):
        if access & rung:
            highest = i
    for rung in _LADDER[: highest + 1]:
        out |= rung
    return out


def access_of(matrix: AccessMatrix, component: Component) -> Access:
    """Look up a component's access as declared, treating an absent entry as NONE.

    Deliberately not expanded: this returns what the caller said they have, which is what gets
    rendered. `satisfies` and `missing_access` expand before comparing.
    """
    return matrix.get(component, Access.NONE)


def satisfies(available: AccessMatrix, required: AccessMatrix) -> bool:
    """Whether ``available`` meets every requirement in ``required``, honouring the ladder."""
    return all(
        not required_access.missing_from(expand_access(access_of(available, component)))
        for component, required_access in required.items()
    )


def missing_access(available: AccessMatrix, required: AccessMatrix) -> dict[Component, "Access"]:
    """What is missing, per component, so a refusal can say exactly what to supply."""
    out: dict[Component, Access] = {}
    for component, required_access in required.items():
        gap = required_access.missing_from(expand_access(access_of(available, component)))
        if gap:
            out[component] = gap
    return out


class Phase(enum.Enum):
    """When the question is being asked.

    Not derivable from access, which is the point. An auditor holding a complete record is
    POST_RUN with RECORD access; a lab replaying its own record is also POST_RUN and can stand up
    arms. Phase decides whether a reading can still change a decision.
    """

    PRE_RUN = enum.auto()  # before any optimisation: grader study, frontier forecast
    IN_RUN = enum.auto()  # during: the hot path, the tap, monitoring
    POST_RUN = enum.auto()  # after: the record exists, the model is finished
    DEPLOYED = enum.auto()  # the artifact is serving; only the artifact exists


class Substrate(enum.Enum):
    """What kind of thing the grader is.

    "A grader is anything with score()" is too coarse, because each of these admits a different
    instrument family and forbids others. Only the two neural kinds have activations to read.
    PROGRAM has none, and has source code, a control-flow graph and forty years of analysis
    tooling instead. COMPOSITE is the frontier default and its leaves are any of the rest.
    """

    NEURAL_SCALAR = enum.auto()  # classifier RM with a head; w_r exists
    NEURAL_GEN = enum.auto()  # GenRM or judge; a verdict direction exists, w_r does not
    PROGRAM = enum.auto()  # verifier, unit tests, sandbox; no weights, HAS SOURCE CODE
    PROCEDURAL = enum.auto()  # tournament, rubric ensemble, aggregation rule
    HUMAN = enum.auto()  # labels, with rater identity
    COMPOSITE = enum.auto()  # a ScoreTree over any of the above


#: The four access profiles, as concrete matrices. Profile A alone is a shippable
#: product, which is why the wedge is defined by what it can reach.
PROFILE_MARKETPLACE_BUYER: AccessMatrix = {
    Component.TASK: Access.QUERY,
    Component.GRADER: Access.QUERY | Access.REPLICATE,
}
PROFILE_OPEN_WEIGHTS: AccessMatrix = {
    Component.TASK: Access.RECORD,
    Component.GRADER: Access.FORWARD | Access.BACKWARD | Access.QUERY | Access.RECORD,
    Component.POLICY: Access.FORWARD | Access.BACKWARD | Access.QUERY | Access.RECORD,
    Component.RECORD: Access.RECORD,
}
PROFILE_AUDITOR: AccessMatrix = {
    Component.TASK: Access.RECORD,
    Component.GRADER: Access.RECORD,
    Component.RECORD: Access.RECORD,
}


__all__ = [
    "canonical_bytes",
    "content_hash",
    "hash_bytes",
    "ModelFP",
    "DatasetID",
    "DirectionID",
    "FrameID",
    "EvidenceID",
    "StudyID",
    "OrganismID",
    "Capability",
    "TrustLevel",
    "GaugeStatus",
    "SitePoint",
    "Site",
    "Span",
    "SubjectRef",
    "Component",
    "Access",
    "AccessMatrix",
    "access_of",
    "expand_access",
    "satisfies",
    "missing_access",
    "Phase",
    "Substrate",
    "PROFILE_MARKETPLACE_BUYER",
    "PROFILE_OPEN_WEIGHTS",
    "PROFILE_AUDITOR",
]

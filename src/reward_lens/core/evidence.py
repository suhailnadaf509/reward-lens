"""Evidence: the universal measurement return type.

Every measurement API in the kernel returns ``Evidence[T]``, never a bare float. The Evidence
carries the typed value, its uncertainty, its gauge status, its calibration reference, its
trust level (computed by the gates, never set by a caller), and its provenance including the
parent Evidence it was derived from. This is the atom of the store and the reason a card and a
paper can be guaranteed to cite the same number.

The value payload ``T`` is a typed dataclass (or a primitive). It is serialized by the
`ValueCodec` below: primitives and small arrays inline into the JSON envelope, bulk arrays go to
content-addressed ``.npy`` sidecars so the store stays a diffable directory of files while large
tensors do not bloat the JSONL. Payload dataclasses register themselves with `register_payload`
so they round-trip exactly.
"""

from __future__ import annotations

import base64
import enum
import importlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

import numpy as np

from reward_lens.core.budget import IncrementalValidity, LimitOfDetection, UncertaintyBudget
from reward_lens.core.envelope import RegimeReading
from reward_lens.core.gates import CalibrationRef, compute_trust
from reward_lens.core.migrations import SCHEMA_VERSION
from reward_lens.core.provenance import Provenance
from reward_lens.core.types import (
    EvidenceID,
    GaugeStatus,
    SubjectRef,
    TrustLevel,
    content_hash,
)

T = TypeVar("T")

# Arrays with more elements than this go to a sidecar rather than inlining into the envelope.
_INLINE_ARRAY_MAX = 64


@dataclass(frozen=True)
class Uncertainty:
    """The uncertainty of a measurement.

    ``n`` is the nominal row count; ``n_effective`` is the lineage-aware effective sample size,
    which, on a dataset of clones, is far smaller than ``n`` and is the
    structural death of v1's fake-n failure class. ``seed_spread`` is the cross-seed standard
    deviation where a quantity is measured over multiple seeds. ``method`` names how the
    interval was produced ("bootstrap-bca", "analytic", "conformal", and crucially
    "bootstrap-CLONE-INFLATED" when a caller opted into resampling across clones).
    """

    ci_low: float | None = None
    ci_high: float | None = None
    ci_level: float | None = None
    n: int | None = None
    n_effective: float | None = None
    seed_spread: float | None = None
    method: str = "none"
    #: The GUM table this interval summarises, where one was built.
    #:
    #: The budget carries alongside the interval rather than replacing it, because an interval and
    #: a table answer different questions and both get asked: a plot wants the interval, and a card
    #: wants to know which term dominates. Replacing the type outright would also have rewritten
    #: every row in every existing store for no gain, since the interval is recoverable from the
    #: table and not the other way round.
    budget: "UncertaintyBudget | None" = None

    @classmethod
    def from_budget(
        cls, budget: "UncertaintyBudget", *, n: int | None = None, method: str = "gum"
    ) -> "Uncertainty":
        """Build the interval from the table, so the two cannot disagree.

        The half-width is the expanded uncertainty `U = k·u_c` at the coverage factor the table
        supports, centred on zero: the caller adds the value. Deriving it rather than accepting it
        separately is what stops a reading carrying a table that says one thing and an interval
        that says another.

        **``ci_level`` is the level the interval really has, not the level anybody wanted.** It
        used to be stamped 0.95 unconditionally, which was wrong in both directions at once: an
        interval built with `k = 2` on a table with no degrees of freedom covers 95.45% under the
        normal it assumes, and the same `k = 2` on a table with four effective degrees of freedom
        covers 88.39%. Both were labelled 95%. `UncertaintyBudget.coverage_achieved` computes the
        real number and this stamps that.
        """
        u = budget.expanded
        return cls(
            ci_low=-u,
            ci_high=u,
            ci_level=budget.coverage_achieved,
            n=n,
            method=method,
            budget=budget,
        )

    @property
    def dominant_term(self) -> str:
        """Which term the budget blames, or empty when there is no budget. Rarely sampling noise."""
        if self.budget is None:
            return ""
        term = self.budget.dominant
        return term.name if term else ""

    def __canonical__(self) -> dict[str, Any]:
        return {
            "ci_low": _num(self.ci_low),
            "ci_high": _num(self.ci_high),
            "ci_level": self.ci_level,
            "n": self.n,
            "n_effective": _num(self.n_effective),
            "seed_spread": _num(self.seed_spread),
            "method": self.method,
            "budget": self.budget.__canonical__() if self.budget is not None else None,
        }


def _num(x: float | None) -> float | str | None:
    """JSON cannot represent NaN/Inf portably; encode them as tagged strings."""
    if x is None:
        return None
    if isinstance(x, float):
        if np.isnan(x):
            return "__nan__"
        if np.isposinf(x):
            return "__inf__"
        if np.isneginf(x):
            return "__-inf__"
    return float(x)


def _unnum(x: float | str | None) -> float | None:
    if isinstance(x, str):
        return {"__nan__": float("nan"), "__inf__": float("inf"), "__-inf__": float("-inf")}[x]
    return x


# ---------------------------------------------------------------------------
# Value codec
# ---------------------------------------------------------------------------

_PAYLOAD_REGISTRY: dict[str, type] = {}


def register_payload(cls: type) -> type:
    """Register a dataclass as an Evidence value payload so it round-trips exactly.

    Decorate any dataclass used as ``Evidence.value``. The codec tags the encoded form with the
    fully qualified name and reconstructs the instance on read. Payloads must be dataclasses;
    their fields may be primitives, lists, dicts, numpy arrays, or other registered payloads.
    """
    if not is_dataclass(cls):
        raise TypeError(f"payload {cls.__name__} must be a dataclass")
    key = f"{cls.__module__}.{cls.__qualname__}"
    _PAYLOAD_REGISTRY[key] = cls
    return cls


class PayloadTypeUnregistered(TypeError):
    """A stored payload names a dataclass nothing has registered.

    Raised rather than degraded, because the degraded result is a plain dict holding the payload's
    fields and that is indistinguishable from a legitimate mapping value. `Blind[T]` is the
    case that makes it unacceptable: its whole guarantee is that a held-out label cannot reach a
    detector, and a `Blind` decoding to a dict hands over the label with the wrapper removed.
    """


class ValueCodec:
    """Encode/decode Evidence value payloads to a JSON-compatible envelope plus array sidecars.

    ``encode`` returns a JSON-serializable object; any array larger than the inline threshold is
    written to ``sidecar_dir`` as a content-addressed ``.npy`` file and referenced by name.
    ``decode`` inverts this, reconstructing registered dataclasses and loading sidecars.

    ``strict`` decides what happens when a stored payload names a type nothing has registered.
    The default raises. Passing ``strict=False`` returns the field dict instead, which is
    occasionally what a forensic reader of an old store wants and is never what a measurement
    path wants.
    """

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict

    def encode(self, value: Any, sidecar_dir: Any = None) -> Any:
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            return _num(value)
        if isinstance(value, np.floating):
            return _num(float(value))
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, (list, tuple)):
            return {"__seq__": [self.encode(v, sidecar_dir) for v in value]}
        if isinstance(value, dict):
            return {"__map__": {str(k): self.encode(v, sidecar_dir) for k, v in value.items()}}
        if isinstance(value, np.ndarray):
            return self._encode_array(value, sidecar_dir)
        if isinstance(value, enum.Enum):
            # By name rather than by value, because a member's name is the stable identity and its
            # value is often an implementation detail somebody renumbers. Reached only by enums that
            # are not already int or str subclasses: `Access` and `Capability` are IntFlags and are
            # caught by the primitive branch above, so their on-disk encoding is unchanged and no
            # existing store has to migrate.
            return {
                "__enum__": f"{type(value).__module__}.{type(value).__qualname__}",
                "name": value.name,
            }
        if is_dataclass(value) and not isinstance(value, type):
            key = f"{type(value).__module__}.{type(value).__qualname__}"
            return {
                "__type__": key,
                "fields": {
                    f.name: self.encode(getattr(value, f.name), sidecar_dir) for f in fields(value)
                },
            }
        raise TypeError(f"cannot encode value of type {type(value).__name__}")

    def decode(self, obj: Any, sidecar_dir: Any = None) -> Any:
        if obj is None or isinstance(obj, (bool, int)):
            return obj
        if isinstance(obj, float):
            return obj
        if isinstance(obj, str):
            return _unnum(obj) if obj.startswith("__") and obj.endswith("__") else obj
        if isinstance(obj, dict):
            if "__seq__" in obj:
                return [self.decode(v, sidecar_dir) for v in obj["__seq__"]]
            if "__map__" in obj:
                return {k: self.decode(v, sidecar_dir) for k, v in obj["__map__"].items()}
            if "__ndarray__" in obj:
                return self._decode_array(obj["__ndarray__"], sidecar_dir)
            if "__enum__" in obj:
                return self._decode_enum(obj)
            if "__type__" in obj:
                return self._decode_dataclass(obj, sidecar_dir)
        return obj

    def _decode_enum(self, obj: dict[str, Any]) -> Any:
        """Resolve an encoded enum member, or leave it encoded rather than guess.

        In non-strict mode an unimportable enum stays a dict, which is what lets a converter read a
        store whose writer's package is not installed. In strict mode it raises, because silently
        handing back a dict where an enum member was written is a wrong value rather than a missing
        one.
        """
        key, name = obj["__enum__"], obj.get("name")
        module, _, qualname = key.rpartition(".")
        try:
            cls = getattr(importlib.import_module(module), qualname)
            return cls[name]
        except (ImportError, AttributeError, KeyError) as exc:
            if self.strict:
                raise PayloadTypeUnregistered(
                    f"cannot resolve enum {key}.{name}. Install the package that wrote this row, "
                    f"or read the store with ValueCodec(strict=False) to keep the member as a dict."
                ) from exc
            return obj

    def _encode_array(self, arr: np.ndarray, sidecar_dir: Any) -> dict[str, Any]:
        arr = np.ascontiguousarray(arr)
        if arr.size <= _INLINE_ARRAY_MAX or sidecar_dir is None:
            return {
                "__ndarray__": {
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                    "b64": base64.b64encode(arr.tobytes()).decode("ascii"),
                }
            }
        from pathlib import Path

        sidecar_dir = Path(sidecar_dir)
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        digest = content_hash(
            {"dtype": str(arr.dtype), "shape": list(arr.shape), "bytes": arr.tobytes()}, "arr"
        ).split(":")[1]
        name = f"{digest}.npy"
        path = sidecar_dir / name
        if not path.exists():
            # Write-then-rename: existence at the content-addressed name is what every later
            # run trusts, so a writer killed mid-save must never leave a truncated file there.
            # The temp name carries the pid so concurrent shard writers cannot collide.
            tmp = sidecar_dir / f"{name}.tmp{os.getpid()}"
            with open(tmp, "wb") as fh:
                np.save(fh, arr)
            os.replace(tmp, path)
        return {"__ndarray__": {"sidecar": name, "dtype": str(arr.dtype), "shape": list(arr.shape)}}

    def _decode_array(self, spec: dict[str, Any], sidecar_dir: Any) -> np.ndarray:
        if "b64" in spec:
            raw = base64.b64decode(spec["b64"])
            return np.frombuffer(raw, dtype=np.dtype(spec["dtype"])).reshape(spec["shape"]).copy()
        from pathlib import Path

        if sidecar_dir is None:
            raise ValueError("array sidecar referenced but no sidecar_dir supplied")
        return np.asarray(np.load(Path(sidecar_dir) / spec["sidecar"]))

    def _decode_dataclass(self, obj: dict[str, Any], sidecar_dir: Any) -> Any:
        key = obj["__type__"]
        cls = _PAYLOAD_REGISTRY.get(key)
        decoded = {k: self.decode(v, sidecar_dir) for k, v in obj["fields"].items()}
        if cls is None:
            # Try to import the module so the decorator runs, then retry.
            module_name = key.rsplit(".", 1)[0]
            try:
                importlib.import_module(module_name)
            except ImportError:
                pass
            cls = _PAYLOAD_REGISTRY.get(key)
        if cls is None and not self.strict:
            return decoded
        if cls is None:
            raise PayloadTypeUnregistered(
                f"payload type {key!r} is not registered, so decoding it would return a plain dict "
                f"carrying its fields. Import the module that defines it, or decorate it with "
                f"@register_payload.\n\n"
                f"This raises rather than degrading because a dict that looks like the value is the "
                f"worst possible failure here: a `Blind[T]` whose type is unregistered would decode "
                f"to a mapping holding the very label the type exists to keep out of a detector's "
                f"reach, and nothing downstream would notice. Pass strict=False to the codec if you "
                f"genuinely want the field dict, and know that you are opting out of that guarantee."
            )
        return cls(**decoded)


_CODEC = ValueCodec()


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Evidence(Generic[T]):
    """The universal typed measurement return value.

    Construct via `make_evidence`, which computes the content-derived id and the gate-computed
    trust level. The trust level is not a constructor argument on purpose: it is a function of
    the calibration reference and the registration status, and allowing a caller to set it would
    defeat the gates.
    """

    id: EvidenceID
    observable: str
    observable_version: str
    subject: SubjectRef
    value: T
    uncertainty: Uncertainty
    gauge: GaugeStatus
    calibration: CalibrationRef | None
    trust: TrustLevel
    provenance: Provenance
    created_at: str
    #: Deliberately excluded from the content id. Two envelopes describing the same measurement
    #: at different schema versions are the same measurement, and hashing the version would give
    #: them different ids and break every parent reference in a store on the first migration.
    schema_version: int = SCHEMA_VERSION

    # -- the 3.0 envelope. Every one is optional, so a 2.0.1 envelope on disk reads
    # -- without migration and a reading that has not been retrofitted is visibly bare.

    #: Which registered quantity this estimates. Empty means the instrument has not been
    #: retrofitted; `lint_instrument` reports that.
    quantity: str = ""
    #: The substrate's disagreement with itself when this was measured. Absent means no floor was
    #: characterised, which is why the value cannot be checked against one.
    lod: "LimitOfDetection | None" = None
    #: What was true about the run when this was measured. Absent means the envelope was never
    #: checked, which is different from checked and passed.
    regime: "RegimeReading | None" = None
    #: The reference material this was calibrated against. Its certificate is what
    #: caps the trust level.
    reference: str | None = None
    #: The dumb baselines this reading beat, or did not. Mandatory by lint: a claim with no
    #: baseline is not a claim.
    baselines: Mapping[str, float] = field(default_factory=dict)
    #: What a white-box reading adds over the best black-box method. Lint requires
    #: it on white-box readings, because the bar is decorrelation plus signal, not superiority.
    incremental: "IncrementalValidity | None" = None
    #: **The third clock**, and it is not derivable from the other two. A checkpoint
    #: produced at step 200 has run position 200 and information time equal to whenever it became
    #: available to the forecaster, and those differ whenever anyone reanalyses an archive. The
    #: forecast barrier reads this field and nothing else. Defaults to `created_at`, which is right
    #: for a fresh measurement and wrong for a reanalysis, so a reanalysis sets it.
    information_time: str = ""

    @property
    def is_calibrated(self) -> bool:
        return self.calibration is not None

    def envelope(self, sidecar_dir: Any = None) -> dict[str, Any]:
        """The JSON-serializable store envelope for this Evidence."""
        return {
            "id": self.id,
            "observable": self.observable,
            "observable_version": self.observable_version,
            "subject": self.subject.__canonical__(),
            "value": _CODEC.encode(self.value, sidecar_dir),
            "uncertainty": self.uncertainty.__canonical__(),
            "gauge": self.gauge.value,
            "calibration": self.calibration.__canonical__() if self.calibration else None,
            "trust": int(self.trust),
            "provenance": self.provenance.__canonical__(),
            "created_at": self.created_at,
            "schema_version": self.schema_version,
            "quantity": self.quantity,
            "lod": self.lod.__canonical__() if self.lod is not None else None,
            "regime": (
                {
                    c.name: {
                        "holds": r.holds,
                        "statistic": _num(r.statistic),
                        "threshold": _num(r.threshold),
                        "detail": r.detail,
                    }
                    for c, r in self.regime.conditions.items()
                }
                if self.regime is not None
                else None
            ),
            "reference": self.reference,
            "baselines": dict(self.baselines),
            "incremental": self.incremental.__canonical__()
            if self.incremental is not None
            else None,
            "information_time": self.information_time or self.created_at,
        }


def make_evidence(
    *,
    observable: str,
    observable_version: str,
    subject: SubjectRef,
    value: T,
    uncertainty: Uncertainty | None = None,
    gauge: GaugeStatus = GaugeStatus.INVARIANT,
    calibration: CalibrationRef | None = None,
    provenance: Provenance | None = None,
    registered: bool = False,
    adjudicated: bool = False,
    created_at: str | None = None,
    quantity: str = "",
    lod: LimitOfDetection | None = None,
    regime: RegimeReading | None = None,
    reference: Any = None,
    baselines: Mapping[str, float] | None = None,
    incremental: IncrementalValidity | None = None,
    information_time: str | None = None,
) -> "Evidence[T]":
    """Build an Evidence, computing its content id and gate-derived trust level.

    The id hashes the observable, subject, value, gauge, calibration, and provenance (excluding
    the wall-clock timestamp) so identical measurements from identical inputs share an id, which
    is what makes the store a deduplicating DAG. Trust is computed by `compute_trust`; passing
    ``registered=True`` (the study runner does this) yields REGISTERED, a calibration reference
    yields at least CALIBRATED, and the two together with ``adjudicated=True`` yield ADJUDICATED.
    """
    unc = uncertainty or Uncertainty()
    prov = provenance or Provenance()
    ref_id = getattr(reference, "id", reference)
    trust = compute_trust(
        calibration=calibration,
        registered=registered,
        adjudicated=adjudicated,
        reference=reference if hasattr(reference, "trust_cap") else None,
    )
    id_material = {
        "observable": observable,
        "observable_version": observable_version,
        "subject": subject.__canonical__(),
        "value": _CODEC.encode(value, None),
        "uncertainty": unc.__canonical__(),
        "gauge": gauge.value,
        "calibration": calibration.__canonical__() if calibration else None,
        "trust": int(trust),
        "provenance": prov.__canonical__(),
    }
    # The 3.0 additions go into the id under one key that is omitted entirely when none is
    # set, so every id already written stays byte-identical and no parent reference breaks.
    #
    # What is in here and what is not is a real distinction rather than an implementation detail.
    # Identity is over **what was measured and against what**: the quantity, the reference, the
    # baselines the claim is made against, and the information time, because two readings that
    # became available at different moments are different claims and the forecast barrier turns on
    # exactly that. The quality annotations are **descriptions of** the measurement rather than
    # part of it: a budget, a detection limit, a regime reading and an incremental-validity record
    # can all be attached to an existing reading later without making it a different reading.
    ext = {
        k: v
        for k, v in (
            ("quantity", quantity or None),
            ("reference", ref_id),
            ("baselines", dict(baselines) if baselines else None),
            ("information_time", information_time or None),
        )
        if v is not None
    }
    if ext:
        id_material["ext"] = ext
    ev_id = EvidenceID(content_hash(id_material, "ev"))
    ts = created_at or datetime.now(timezone.utc).isoformat()
    return Evidence(
        id=ev_id,
        observable=observable,
        observable_version=observable_version,
        subject=subject,
        value=value,
        uncertainty=unc,
        gauge=gauge,
        calibration=calibration,
        trust=trust,
        provenance=prov,
        created_at=ts,
        quantity=quantity,
        lod=lod,
        regime=regime,
        reference=ref_id,
        baselines=dict(baselines) if baselines else {},
        incremental=incremental,
        information_time=information_time or ts,
    )


def evidence_from_envelope(env: dict[str, Any], sidecar_dir: Any = None) -> "Evidence[Any]":
    """Reconstruct an Evidence from its store envelope, migrating it forward if it is older.

    Every read goes through `migrate`, which is a no-op on a current envelope, so an old store
    opens without a conversion step and without anyone having to remember one.
    """
    from reward_lens.core.migrations import migrate
    from reward_lens.core.provenance import Cost

    env = migrate(env)

    subj = env["subject"]
    subject = SubjectRef(
        signals=tuple(subj.get("signals", [])),
        dataset=subj.get("dataset"),
        readout=subj.get("readout"),
        frame=subj.get("frame"),
        interventions=tuple(subj.get("interventions", [])),
        extra=subj.get("extra", {}),
    )
    u = env["uncertainty"]
    unc = Uncertainty(
        ci_low=_unnum(u.get("ci_low")),
        ci_high=_unnum(u.get("ci_high")),
        ci_level=u.get("ci_level"),
        n=u.get("n"),
        n_effective=_unnum(u.get("n_effective")),
        seed_spread=_unnum(u.get("seed_spread")),
        method=u.get("method", "none"),
    )
    p = env["provenance"]
    c = p.get("cost", {})
    prov = Provenance(
        git_sha=p.get("git_sha", "unknown"),
        config_hash=p.get("config_hash"),
        seeds=tuple(p.get("seeds", [])),
        cost=Cost(
            gpu_seconds=c.get("gpu_seconds", 0.0),
            tokens=c.get("tokens", 0),
            wall_seconds=c.get("wall_seconds", 0.0),
        ),
        oracle_calls=tuple(p.get("oracle_calls", [])),
        parents=tuple(p.get("parents", [])),
        study=p.get("study"),
        extra=p.get("extra", {}),
    )
    cal = env.get("calibration")
    calibration = (
        CalibrationRef(
            scorecard_entry=cal["scorecard_entry"],
            organism_family=cal["organism_family"],
            regime_match=cal.get("regime_match", "exact"),
            operating_point=cal.get("operating_point"),
        )
        if cal
        else None
    )
    return Evidence(
        id=EvidenceID(env["id"]),
        observable=env["observable"],
        observable_version=env["observable_version"],
        subject=subject,
        value=_CODEC.decode(env["value"], sidecar_dir),
        uncertainty=unc,
        gauge=GaugeStatus(env["gauge"]),
        calibration=calibration,
        trust=TrustLevel(env["trust"]),
        provenance=prov,
        created_at=env["created_at"],
        quantity=env.get("quantity", ""),
        reference=env.get("reference"),
        baselines=env.get("baselines") or {},
        information_time=env.get("information_time") or env["created_at"],
    )


__all__ = [
    "PayloadTypeUnregistered",
    "Uncertainty",
    "Evidence",
    "make_evidence",
    "evidence_from_envelope",
    "register_payload",
    "ValueCodec",
]

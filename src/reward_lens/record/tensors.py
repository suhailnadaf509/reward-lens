"""The tensor store, and the three-way reference that makes non-storage the default.

A `TensorRef` is one of three things and the third one is the important one:

- `StoredRef` — the bytes are on disk, content-addressed, and can be loaded.
- `RecomputeRef` — the bytes are not stored, and here is exactly what to run to get them back.
- `AbsentRef` — the bytes are not available, with the reason and what to do about it.

`AbsentRef` is a value, not a `None`, for the same reason `Refusal` is a value and not an
exception: the caller has to handle it, and the thing it must never quietly become is a zero. A
`RecomputeRef` that cannot be honoured resolves to an `AbsentRef` carrying which of the four ways
it failed, never to `np.zeros(shape)`.

**Why non-storage is the default rather than a fallback.** `residual_bytes` below is the whole
argument, executable. A bf16 residual stream costs ``2 * d_model`` bytes per token per layer. On
Llama-3.1-70B (``d_model = 8192``) that is 16.00 KiB per token per layer, so one layer over 10^9
rollout tokens is 16.4 TB and all eighty layers is 1.31 PB. Storage is not even the binding
constraint: the wall is the egress path off the accelerator. The SGLang Kimi-K2 measurement is the
proof, at 300 to 600 ms of stall on the scheduler main thread against a 30 to 80 ms decode step,
with 31 of 32 ranks blocked inside the collective while one rank pickles `routed_experts`.

So the honest schema records what it would take to reproduce a tensor and admits when it does not
have it, and every instrument that wants activations has to be written against a union whose third
member says no.
"""

from __future__ import annotations

import enum
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Protocol, Union, runtime_checkable

import numpy as np

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import ModelFP, Site, content_hash, hash_bytes

#: Bumped in the same commit that changes the on-disk manifest layout.
TENSOR_MANIFEST_VERSION = 1

Container = Literal["npy", "safetensors"]


# ---------------------------------------------------------------------------
# The arithmetic that forces the default
# ---------------------------------------------------------------------------


def residual_bytes(*, d_model: int, tokens: int = 1, layers: int = 1, dtype_bytes: int = 2) -> int:
    """Bytes of residual stream for ``tokens`` tokens over ``layers`` layers.

    The rule is ``dtype_bytes * d_model`` per token per layer, and ``dtype_bytes`` defaults to 2
    because bf16 is what training runs hold activations in. This is a function rather than a
    constant in a docstring so the numbers in the module docstring are checkable: at
    ``d_model = 8192`` it returns 16,384 bytes per token per layer, 1.6384e13 over 10^9 tokens on
    one layer, and 1.31072e15 over eighty.
    """
    if d_model <= 0 or tokens < 0 or layers < 0 or dtype_bytes <= 0:
        raise ValueError(
            f"residual_bytes needs a positive d_model and dtype_bytes and non-negative counts; "
            f"got d_model={d_model}, tokens={tokens}, layers={layers}, dtype_bytes={dtype_bytes}"
        )
    return dtype_bytes * d_model * tokens * layers


# ---------------------------------------------------------------------------
# What to capture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureSpec:
    """What a capture would read, precisely enough to reproduce it.

    ``sites`` are kernel `Site` values, so a capture spec addresses the same surfaces the rest of
    the library addresses. ``stride`` and ``max_tokens`` exist because the egress cost is linear
    in captured tokens and a run that captures every tenth token pays a tenth of the stall; the
    stride is recorded rather than assumed so a downstream estimator knows the sample it is
    looking at is not the sequence.
    """

    sites: tuple[Site, ...] = ()
    dtype: str = "bfloat16"
    stride: int = 1
    max_tokens: int | None = None
    include_logits: bool = False
    include_routing: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stride < 1:
            raise ValueError(f"capture stride must be at least 1; got {self.stride}")
        if self.max_tokens is not None and self.max_tokens < 0:
            raise ValueError(f"capture max_tokens cannot be negative; got {self.max_tokens}")

    def __canonical__(self) -> dict[str, Any]:
        return {
            "sites": [s.__canonical__() for s in self.sites],
            "dtype": self.dtype,
            "stride": self.stride,
            "max_tokens": self.max_tokens,
            "include_logits": self.include_logits,
            "include_routing": self.include_routing,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "CaptureSpec":
        return cls(
            sites=tuple(
                Site(layer=s["layer"], point=s.get("point", "resid_post"), head=s.get("head"))
                for s in obj.get("sites", ())
            ),
            dtype=obj.get("dtype", "bfloat16"),
            stride=obj.get("stride", 1),
            max_tokens=obj.get("max_tokens"),
            include_logits=obj.get("include_logits", False),
            include_routing=obj.get("include_routing", False),
            extra=dict(obj.get("extra", {})),
        )


# ---------------------------------------------------------------------------
# The three-way union
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredRef:
    """The bytes exist. ``digest`` is the content address and is the only identity that matters.

    Two runs that captured the same tensor land on the same digest and share one shard, which is
    the only reason storing anything at this scale is tractable at all.
    """

    digest: str
    shard: str
    key: str
    dtype: str
    shape: tuple[int, ...]
    container: Container = "npy"
    nbytes: int = 0

    def __canonical__(self) -> dict[str, Any]:
        return {
            "kind": "stored",
            "digest": self.digest,
            "shard": self.shard,
            "key": self.key,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "container": self.container,
            "nbytes": self.nbytes,
        }


@dataclass(frozen=True)
class RecomputeRef:
    """Not stored. Here is exactly what to run to get it back.

    Every field is load-bearing for reproduction, and ``engine`` is the one people leave out.
    Measuring HuggingFace against vLLM on pinned revisions with identical prompt token ids in BF16
    found 0.18% relative L2 on a small dense model and 3.28% on a hybrid MoE, with compiled and
    eager vLLM disagreeing with each other by about as much as either disagrees with HuggingFace.
    Two configurations of the same weights are two instruments, so a recompute recipe without the
    engine is not a recipe.

    ``expected_numerics_floor`` is the relative disagreement to expect when this is honoured, in
    the same unit as those measurements (relative L2, so 0.0033 is 0.33%). It is not a
    tolerance the caller may widen quietly: `honour` refuses when the achieved disagreement
    exceeds it, because a recomputed tensor that disagrees with the original by more than the
    substrate's own noise floor is a different tensor.
    """

    model_fp: ModelFP
    engine: "Engine"
    token_ids: tuple[int, ...]
    sampling_seed: int
    capture_spec: CaptureSpec
    expected_numerics_floor: float

    def __post_init__(self) -> None:
        if self.expected_numerics_floor < 0:
            raise ValueError(
                f"expected_numerics_floor is a relative disagreement and cannot be negative; got "
                f"{self.expected_numerics_floor}"
            )

    @property
    def id(self) -> str:
        """Content address of the recipe, so two identical recipes are one recipe."""
        return content_hash(self.__canonical__(), "rcp")

    def __canonical__(self) -> dict[str, Any]:
        return {
            "kind": "recompute",
            "model_fp": str(self.model_fp),
            "engine": self.engine.__canonical__(),
            "token_ids": list(self.token_ids),
            "sampling_seed": self.sampling_seed,
            "capture_spec": self.capture_spec.__canonical__(),
            "expected_numerics_floor": self.expected_numerics_floor,
        }


class AbsenceReason(enum.Enum):
    """Why a tensor is not here. Seven ways, and none of them is a zero.

    These are deliberately not `RefusalReason` members. A `RefusalReason` is a statement about an
    instrument's reading; this is a statement about one tensor, and several tensors can be absent
    for different reasons inside a single reading that still succeeds. `AbsentRef.as_refusal`
    converts one when an instrument has to give up because of it.
    """

    #: The capture or record-sampling policy excluded it. This is the honest default at scale,
    #: not a failure: see the module docstring for what capturing it would have cost.
    NOT_CAPTURED = enum.auto()
    #: A tap was disabled by its budget rather than stalling the scheduler. The SGLang Kimi-K2
    #: case: 300 to 600 ms of main-thread stall against a 30 to 80 ms decode step.
    EGRESS_REFUSED = enum.auto()
    #: A `StoredRef` whose shard is not in this store. The record travelled and the tensors did not.
    SHARD_MISSING = enum.auto()
    #: A `RecomputeRef` and no recomputer was supplied.
    RECOMPUTE_UNSUPPORTED = enum.auto()
    #: A recomputer was supplied and declined: weights gone, engine unavailable, revision unpinned.
    RECOMPUTE_UNAVAILABLE = enum.auto()
    #: Recomputed, and the achieved disagreement exceeded `expected_numerics_floor`.
    NUMERICS_FLOOR_EXCEEDED = enum.auto()
    #: The prefix was rewritten by a compaction, so this tensor is not defined on this prefix at
    #: all. Different from missing: there is no recipe that would bring it back.
    COMPACTED = enum.auto()


#: What to do about each absence, as an instruction. Same discipline as `Refusal.remedy`: these
#: are read by someone holding a failure, so they say what to change rather than what happened.
DEFAULT_REMEDY: dict[AbsenceReason, str] = {
    AbsenceReason.NOT_CAPTURED: (
        "Re-run with a CaptureSpec naming this site, or accept the estimator that works from "
        "scores alone. Check the cost first: residual_bytes() prices the capture you are asking for."
    ),
    AbsenceReason.EGRESS_REFUSED: (
        "Raise the tap budget, or lower the capture stride so the per-step egress fits inside it, "
        "then re-run. The recorded InstrumentEffect says which budget term was breached."
    ),
    AbsenceReason.SHARD_MISSING: (
        "Point the reader at the tensor store that accompanies this record, or copy the shard "
        "named in the StoredRef into the store's shards/ directory."
    ),
    AbsenceReason.RECOMPUTE_UNSUPPORTED: (
        "Pass a Recomputer to resolve(). The RecomputeRef carries the model fingerprint, engine, "
        "token ids and seed needed to build one."
    ),
    AbsenceReason.RECOMPUTE_UNAVAILABLE: (
        "Make the recipe's model and engine reachable: the exact revision and attention "
        "implementation named in the RecomputeRef, not a near equivalent. Eager and compiled vLLM "
        "disagree with each other as much as either disagrees with HuggingFace, so a near "
        "equivalent is a different instrument."
    ),
    AbsenceReason.NUMERICS_FLOOR_EXCEEDED: (
        "Recompute on the engine, revision, dtype and attention implementation named in the "
        "RecomputeRef. If that is what you did, the floor recorded at capture time was too "
        "optimistic and re-measuring it is the fix, not widening the tolerance."
    ),
    AbsenceReason.COMPACTED: (
        "Restrict the window to steps on one side of the compaction event, or use a quantity that "
        "does not condition on the rewritten prefix. There is no recipe that recovers this one."
    ),
}


#: Which `RefusalReason` each absence becomes when an instrument has to give up because of it.
#: One row per reason, decided by reading that reason's own remedy above and asking one question:
#: is the remedy answerable where the reader is standing, or only upstream where the record was
#: written? The two point in opposite directions, and a refusal that says "get more access" when
#: the honest answer is "your run did not dump this" costs somebody an afternoon and then still
#: does not work.
#:
#: `NOT_CAPTURED` and `EGRESS_REFUSED`: the remedies are "re-run with a CaptureSpec naming this
#: site" and "raise the tap budget, then re-run". The tensor was never written. No access the
#: reader can acquire and nothing they do to this record recovers it.
#:
#: `COMPACTED`: the prefix was rewritten during the run, so the tensor is not defined on the prefix
#: this record holds. The remedy says outright that there is no recipe that recovers it. That is
#: upstream in the strongest sense, since the fix is a run that does not compact there.
#:
#: `SHARD_MISSING`, `RECOMPUTE_UNSUPPORTED` and `RECOMPUTE_UNAVAILABLE`: the record is complete and
#: the reader is short of something. A shard that travelled separately, a `Recomputer` that was
#: never passed, a model and engine that are not reachable from here. Every one of those is
#: obtained where the reader is standing, which is what `ACCESS_INSUFFICIENT` means.
#:
#: `NUMERICS_FLOOR_EXCEEDED`: a recompute ran and disagreed with the recorded floor. The remedy's
#: first branch is the reader's ("recompute on the engine, revision, dtype and attention
#: implementation named in the RecomputeRef"), so the reason follows that branch. Its second branch
#: is upstream, and it is the one the reader reaches only after exhausting the first, so it stays
#: in the remedy string rather than splitting the row.
#:
#: `COMPACTED` is the row that would move if a `QUANTITY_UNDEFINED` reason were ever added. An
#: importance ratio across a rewritten prefix is not a quantity that a better record answers, it
#: is a quantity the object does not have. `RECORD_INCOMPLETE` is the closer of the sixteen because
#: a run recorded without that compaction does carry the tensor, and adding a seventeenth member to
#: the kernel's refusal vocabulary is not a decision this module makes.
ABSENCE_REFUSAL: dict[AbsenceReason, RefusalReason] = {
    AbsenceReason.NOT_CAPTURED: RefusalReason.RECORD_INCOMPLETE,
    AbsenceReason.EGRESS_REFUSED: RefusalReason.RECORD_INCOMPLETE,
    AbsenceReason.SHARD_MISSING: RefusalReason.ACCESS_INSUFFICIENT,
    AbsenceReason.RECOMPUTE_UNSUPPORTED: RefusalReason.ACCESS_INSUFFICIENT,
    AbsenceReason.RECOMPUTE_UNAVAILABLE: RefusalReason.ACCESS_INSUFFICIENT,
    AbsenceReason.NUMERICS_FLOOR_EXCEEDED: RefusalReason.ACCESS_INSUFFICIENT,
    AbsenceReason.COMPACTED: RefusalReason.RECORD_INCOMPLETE,
}


@dataclass(frozen=True)
class AbsentRef:
    """The tensor is not available, and this says why and what to do.

    Constructed through `of` almost everywhere, which fills the remedy from `DEFAULT_REMEDY`. The
    empty-remedy check is the same rule `Refusal` enforces: an absence with no remedy is a tool
    that looks broken rather than a tool that looks careful.
    """

    reason: AbsenceReason
    detail: str = ""
    remedy: str = ""
    statistics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.remedy.strip():
            raise ValueError(
                f"AbsentRef({self.reason.name}) carries no remedy. Use AbsentRef.of(), which "
                f"fills the default, or write one."
            )

    @classmethod
    def of(cls, reason: AbsenceReason, *, detail: str = "", **statistics: float) -> "AbsentRef":
        return cls(
            reason=reason, detail=detail, remedy=DEFAULT_REMEDY[reason], statistics=statistics
        )

    def as_refusal(self, instrument: str) -> Refusal:
        """Convert to a kernel `Refusal` when an instrument cannot proceed without this tensor.

        The mapping is per reason and it is decided by each reason's own remedy, which is the
        test: `ACCESS_INSUFFICIENT` when the remedy is answerable where the reader is standing,
        `RECORD_INCOMPLETE` when it is answerable only upstream, where the record was produced.
        See `ABSENCE_REFUSAL` for the table and the argument for each row. This used to be one
        constant for all seven, chosen so it was one decision rather than seventy; the cost of that
        was three reasons whose remedy says "re-run" telling the reader to get more access.
        """
        return Refusal(
            instrument=instrument,
            reason=ABSENCE_REFUSAL[self.reason],
            detail=f"tensor absent ({self.reason.name}): {self.detail}".strip(),
            remedy=self.remedy,
            statistics={"absence": self.reason.name, **dict(self.statistics)},
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "kind": "absent",
            "reason": self.reason.name,
            "detail": self.detail,
            "remedy": self.remedy,
            "statistics": dict(self.statistics),
        }


TensorRef = Union[StoredRef, RecomputeRef, AbsentRef]

#: A `TensorRef` whose leading axis is tokens. The alias exists because
#: `Trajectory.advantage_tokens` carries this type and the leading-axis convention is the whole
#: content of the distinction: a per-token quantity compared against a per-sequence one is
#: `UNIT_MISMATCH`, the commonest silent error in this literature.
TokenTensorRef = TensorRef


def ref_from_canonical(obj: Mapping[str, Any]) -> TensorRef:
    """Rebuild a `TensorRef` from its canonical form. Dispatches on the ``kind`` tag."""
    kind = obj["kind"]
    if kind == "stored":
        return StoredRef(
            digest=obj["digest"],
            shard=obj["shard"],
            key=obj["key"],
            dtype=obj["dtype"],
            shape=tuple(obj["shape"]),
            container=obj.get("container", "npy"),
            nbytes=obj.get("nbytes", 0),
        )
    if kind == "recompute":
        return RecomputeRef(
            model_fp=ModelFP(obj["model_fp"]),
            engine=Engine.from_canonical(obj["engine"]),
            token_ids=tuple(obj["token_ids"]),
            sampling_seed=obj["sampling_seed"],
            capture_spec=CaptureSpec.from_canonical(obj["capture_spec"]),
            expected_numerics_floor=obj["expected_numerics_floor"],
        )
    if kind == "absent":
        return AbsentRef(
            reason=AbsenceReason[obj["reason"]],
            detail=obj.get("detail", ""),
            remedy=obj.get("remedy") or DEFAULT_REMEDY[AbsenceReason[obj["reason"]]],
            statistics=dict(obj.get("statistics", {})),
        )
    raise ValueError(f"unknown tensor ref kind {kind!r}")


@dataclass(frozen=True)
class CaptureRef:
    """A trajectory's captured tensors, keyed by name, each a full three-way `TensorRef`.

    The spec is what was asked for; the mapping is what came back. They differ whenever a tap was
    disabled or a capture was sampled out, and the difference is exactly the information an
    estimator needs to know it is looking at a subsample.
    """

    spec: CaptureSpec
    tensors: Mapping[str, TensorRef] = field(default_factory=dict)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "spec": self.spec.__canonical__(),
            "tensors": {k: v.__canonical__() for k, v in self.tensors.items()},
        }

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "CaptureRef":
        return cls(
            spec=CaptureSpec.from_canonical(obj["spec"]),
            tensors={k: ref_from_canonical(v) for k, v in obj.get("tensors", {}).items()},
        )


# ---------------------------------------------------------------------------
# Engine identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Engine:
    """The serving or training stack a tensor came out of, identified well enough to reproduce.

    The canonical schema types this field as an opaque `EngineID`. It is a structure here because
    the numerics floor needs the components and an opaque id cannot supply them: the limits of
    detection are cached per ``(model, engine, revision, dtype, attention_impl, layer)``, and
    "vLLM" is not one instrument (compiled and eager disagree with each other about as much as
    either disagrees with HuggingFace). `Engine.id` is the printed `EngineID`, so anything typed
    against the id is unaffected.
    """

    name: str
    revision: str = "unknown"
    attention_impl: str = "unknown"
    dtype: str = "unknown"
    compiled: bool | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return content_hash(self.__canonical__(), "eng")

    def __canonical__(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "revision": self.revision,
            "attention_impl": self.attention_impl,
            "dtype": self.dtype,
            "compiled": self.compiled,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_canonical(cls, obj: Mapping[str, Any]) -> "Engine":
        return cls(
            name=obj["name"],
            revision=obj.get("revision", "unknown"),
            attention_impl=obj.get("attention_impl", "unknown"),
            dtype=obj.get("dtype", "unknown"),
            compiled=obj.get("compiled"),
            extra=dict(obj.get("extra", {})),
        )

    def __str__(self) -> str:
        bits = [self.name, self.revision, self.attention_impl, self.dtype]
        if self.compiled is not None:
            bits.append("compiled" if self.compiled else "eager")
        return "/".join(bits)


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


@runtime_checkable
class Recomputer(Protocol):
    """What honours a `RecomputeRef`.

    Implemented by the white-box side, which needs torch; nothing here does. `recompute` returns
    the array and the relative disagreement it achieved against whatever it can compare to. A
    recomputer that has no way to measure its own agreement returns ``float('nan')``, which
    `honour` treats as a failure rather than as a pass: an unmeasured floor is not a met floor.
    """

    def can_honour(self, ref: RecomputeRef) -> bool: ...

    def recompute(self, ref: RecomputeRef) -> tuple[np.ndarray, float]: ...


def honour(ref: RecomputeRef, recomputer: Recomputer | None) -> "np.ndarray | AbsentRef":
    """Run a recompute recipe, or say honestly why it did not run.

    Four outcomes and three of them are an `AbsentRef`. Never a zero array, which is the failure
    mode this whole module is shaped around: a silent zero flows into a mean and a standard
    deviation and an advantage and nothing downstream can tell it apart from a measurement.
    """
    if recomputer is None:
        return AbsentRef.of(
            AbsenceReason.RECOMPUTE_UNSUPPORTED,
            detail=f"recipe {ref.id} for {ref.model_fp} on {ref.engine} was never run",
        )
    if not recomputer.can_honour(ref):
        return AbsentRef.of(
            AbsenceReason.RECOMPUTE_UNAVAILABLE,
            detail=(
                f"the recomputer declined recipe {ref.id}: it cannot reach {ref.model_fp} on "
                f"{ref.engine}"
            ),
        )
    array, achieved = recomputer.recompute(ref)
    if not np.isfinite(achieved) or achieved > ref.expected_numerics_floor:
        return AbsentRef.of(
            AbsenceReason.NUMERICS_FLOOR_EXCEEDED,
            detail=(
                f"recompute of {ref.id} disagreed by {achieved:.6g} against an expected floor of "
                f"{ref.expected_numerics_floor:.6g}"
            ),
            achieved=float(achieved),
            expected=float(ref.expected_numerics_floor),
        )
    return array


class TensorStore:
    """Content-addressed tensor shards plus a JSON manifest.

    One shard per content address, holding one array. Deduplication is then free and exact: two
    runs that captured identical bytes share one file, and re-putting an array already present is
    a no-op that returns the same `StoredRef`.

    **Container formats.** safetensors is implemented, but the default is ``.npy`` because numpy is
    a base dependency and safetensors is behind the ``white-box`` extra, and a record whose tensors
    cannot be read on a base install is useless to the auditor profile that only ever gets handed a
    record. The manifest records which container each shard uses, so a store can hold both and a
    reader never has to guess.
    """

    def __init__(self, root: str | Path, *, container: Container = "npy") -> None:
        self.root = Path(root)
        self.shards = self.root / "shards"
        self.manifest_path = self.root / "manifest.json"
        self.container: Container = container
        self.entries: dict[str, dict[str, Any]] = {}
        if self.manifest_path.exists():
            loaded = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.entries = dict(loaded.get("entries", {}))

    # -- write ---------------------------------------------------------------

    def put(self, array: np.ndarray, *, name: str = "tensor") -> StoredRef:
        """Store an array, returning its `StoredRef`. Idempotent on content."""
        arr = np.ascontiguousarray(array)
        digest = hash_bytes(
            f"{arr.dtype}|{tuple(arr.shape)}|".encode("utf-8") + arr.tobytes(), "tns"
        )
        short = digest.split(":", 1)[1]
        existing = self.entries.get(digest)
        if existing is not None:
            return StoredRef(
                digest=digest,
                shard=existing["shard"],
                key=existing["key"],
                dtype=existing["dtype"],
                shape=tuple(existing["shape"]),
                container=existing["container"],
                nbytes=existing["nbytes"],
            )
        suffix = ".safetensors" if self.container == "safetensors" else ".npy"
        shard = f"{short}{suffix}"
        self.shards.mkdir(parents=True, exist_ok=True)
        path = self.shards / shard
        if not path.exists():
            # Write-then-rename, for the same reason the evidence store's sidecars do it:
            # existence at the content-addressed name is what every later read trusts, so a
            # writer killed mid-save must never leave a truncated file sitting there.
            tmp = self.shards / f"{shard}.tmp{os.getpid()}"
            if self.container == "safetensors":
                from safetensors.numpy import save_file

                save_file({name: arr}, str(tmp))
            else:
                with open(tmp, "wb") as fh:
                    np.save(fh, arr)
            os.replace(tmp, path)
        ref = StoredRef(
            digest=digest,
            shard=shard,
            key=name,
            dtype=str(arr.dtype),
            shape=tuple(arr.shape),
            container=self.container,
            nbytes=int(arr.nbytes),
        )
        entry: dict[str, Any] = ref.__canonical__()
        entry.pop("kind")
        entry.pop("digest")
        self.entries[digest] = entry
        return ref

    def flush(self) -> Path:
        """Write the manifest atomically. Cheap; call it after a batch of puts."""
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "manifest_version": TENSOR_MANIFEST_VERSION,
            "default_container": self.container,
            "entries": self.entries,
        }
        tmp = self.manifest_path.with_suffix(f".json.tmp{os.getpid()}")
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.manifest_path)
        return self.manifest_path

    # -- read ----------------------------------------------------------------

    def get(self, ref: StoredRef) -> "np.ndarray | AbsentRef":
        """Load a stored array, or say why it is not here."""
        path = self.shards / ref.shard
        if not path.exists():
            return AbsentRef.of(
                AbsenceReason.SHARD_MISSING,
                detail=f"shard {ref.shard} for {ref.digest} is not in {self.shards}",
            )
        if ref.container == "safetensors":
            from safetensors.numpy import load_file

            return np.asarray(load_file(str(path))[ref.key])
        return np.asarray(np.load(path))

    def total_bytes(self) -> int:
        return sum(int(e["nbytes"]) for e in self.entries.values())


def resolve(
    ref: TensorRef,
    *,
    store: TensorStore | None = None,
    recomputer: Recomputer | None = None,
) -> "np.ndarray | AbsentRef":
    """The one entry point: turn any of the three references into bytes, or into an honest absence.

    Callers should switch on ``isinstance(result, AbsentRef)`` rather than on the ref type. That
    is the point of the union: an instrument written against `resolve` handles the not-stored case
    once, at the top, instead of discovering it at the bottom as a shape error.
    """
    if isinstance(ref, AbsentRef):
        return ref
    if isinstance(ref, StoredRef):
        if store is None:
            return AbsentRef.of(
                AbsenceReason.SHARD_MISSING,
                detail=f"{ref.digest} is stored but no tensor store was supplied to resolve()",
            )
        return store.get(ref)
    return honour(ref, recomputer)


def resolve_all(
    refs: Mapping[str, TensorRef],
    *,
    store: TensorStore | None = None,
    recomputer: Recomputer | None = None,
) -> dict[str, "np.ndarray | AbsentRef"]:
    """Resolve a mapping of refs, keeping absences in place rather than dropping the keys.

    Dropping an absent key would turn "we did not capture the routing trace" into "there is no
    routing trace", and those call for different responses.
    """
    return {k: resolve(v, store=store, recomputer=recomputer) for k, v in refs.items()}


def absences(results: Mapping[str, Any]) -> dict[str, AbsentRef]:
    """The subset of a `resolve_all` result that came back absent."""
    return {k: v for k, v in results.items() if isinstance(v, AbsentRef)}


def compacted_refs(names: Iterable[str], *, at_turn: int, method: str) -> dict[str, AbsentRef]:
    """Mark a set of tensors absent because a compaction rewrote the prefix they lived on.

    Used by the converters when a `CompactionEvent` lands inside the window a capture covers.
    The distinction from NOT_CAPTURED is worth the separate reason: this one has no recipe.
    """
    return {
        name: AbsentRef.of(
            AbsenceReason.COMPACTED,
            detail=f"prefix rewritten at turn {at_turn} by {method}; not defined on this prefix",
            at_turn=float(at_turn),
        )
        for name in names
    }


def with_absence(ref: TensorRef, reason: AbsenceReason, *, detail: str = "") -> AbsentRef:
    """Downgrade any ref to an absence, preserving what it used to be in the detail string."""
    was = ref.__canonical__().get("kind", "unknown")
    return AbsentRef.of(reason, detail=(f"was {was}; {detail}" if detail else f"was {was}"))


__all__ = [
    "ABSENCE_REFUSAL",
    "AbsenceReason",
    "AbsentRef",
    "CaptureRef",
    "CaptureSpec",
    "Container",
    "DEFAULT_REMEDY",
    "Engine",
    "RecomputeRef",
    "Recomputer",
    "StoredRef",
    "TENSOR_MANIFEST_VERSION",
    "TensorRef",
    "TensorStore",
    "TokenTensorRef",
    "absences",
    "compacted_refs",
    "honour",
    "ref_from_canonical",
    "residual_bytes",
    "resolve",
    "resolve_all",
    "with_absence",
]

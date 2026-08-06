"""Reading the campaign's evidence store without writing to it and without decoding it wrong.

The campaign store is the only copy of a $17.73 experiment, so everything here opens it through
`EvidenceStore(path, readonly=True)`. That mode is not decoration: it makes `append` raise, and it
makes the torn-tail scan skip a partial final line rather than truncating the file. Neither applies
to this store (it has no torn tail) but a converter that opens an archive for writing is one
mistake away from being the reason the archive needs restoring.

**Why this module decodes payloads itself.** The campaign wrote fifteen payload dataclasses under
``campaign.payloads.*``, and that package is not installed here and is not a dependency of this
library. `evidence_from_envelope` decodes through a module-level `ValueCodec` with ``strict=True``,
so it raises `PayloadTypeUnregistered` on the first row. Measured, on the shipped store::

    EvidenceStore('campaign-results/runs/campaign', readonly=True).find(observable='campaign.scores')
    -> PayloadTypeUnregistered: payload type 'campaign.payloads.ScoreBank' is not registered

That is the correct default and it is the reason `Blind` cannot be silently unwrapped. It is also
the reason the shipped store reader cannot read the shipped store, which is worth stating plainly:
a store is only readable by a process that can import every payload class that wrote it. This
module opts out with ``strict=False`` and gets the field dict, which is exactly what a converter
wants and is safe here for one checkable reason: the census below finds no `Blind` and no
`record.labels` type anywhere in the store, because the store predates that type by a release. The
census runs, rather than being asserted, in `CampaignStore.payload_types`.

**Sidecar resolution across two stores.** The campaign left two stores. The one that
carries the adjudications, ``runs/campaign``, is referentially exact: 1,076 sidecar names
referenced, 1,076 files present, 0 missing and 0 orphaned. The other, ``store/``, holds arrays the
first does not, so `CampaignStore` takes a search path rather than a single directory. Resolution
uses one property of `pathlib`: ``Path(a) / "/abs/b"`` discards ``a`` and yields ``/abs/b``. So a
sidecar found in a fallback directory is rewritten to its absolute path in place and the kernel's
own codec loads it, rather than this module growing a second copy of the array loader.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from reward_lens.core.evidence import ValueCodec
from reward_lens.core.store import EvidenceStore

#: Observable prefixes the campaign used for its two close-out row families. The adjudication rows
#: carry the per-card verdict and the metrics it was computed from; the result rows carry the
#: report payload. Nineteen of the twenty-seven cards have both; eight have only the adjudication,
#: and those eight are exactly the cards that have to be re-read.
ADJUDICATION_PREFIX = "campaign.adjudication."
RESULT_PREFIX = "campaign.result."


@dataclass(frozen=True)
class CampaignRow:
    """One evidence envelope from the campaign store, with the fields a converter joins on.

    The envelope is kept whole in ``envelope`` rather than being flattened, because the converter
    needs the raw ``value`` and the value is the part that cannot be decoded without a codec. The
    named fields are the join keys: ``roster_key`` is which reward model, ``slice_name`` is which
    bank, and the pair is the campaign's primary key on almost every row.
    """

    id: str
    observable: str
    signals: tuple[str, ...]
    extra: Mapping[str, Any]
    created_at: str
    study: str | None
    arc: str
    gpu: str
    envelope: Mapping[str, Any]

    @property
    def roster_key(self) -> str | None:
        """Which reward model produced this row, as the campaign named it."""
        key = self.extra.get("roster_key")
        if key is None and len(self.signals) == 1:
            return self.signals[0]
        return key

    @property
    def slice_name(self) -> str | None:
        """Which bank, including the ``::partNNNN`` suffix a chunked arc appended."""
        return self.extra.get("slice")

    @property
    def bank(self) -> str | None:
        """The bank without its part suffix, which is what a study's subject query names."""
        s = self.slice_name
        return None if s is None else s.split("::", 1)[0]

    @property
    def card(self) -> str | None:
        """The preregistered card this row belongs to, on the close-out rows that name one."""
        return self.extra.get("card")


def _row_from_envelope(env: Mapping[str, Any]) -> CampaignRow:
    subject = env.get("subject", {})
    provenance = env.get("provenance", {})
    prov_extra = provenance.get("extra", {}) or {}
    return CampaignRow(
        id=env["id"],
        observable=env["observable"],
        signals=tuple(subject.get("signals", ()) or ()),
        extra=dict(subject.get("extra", {}) or {}),
        created_at=env.get("created_at", ""),
        study=provenance.get("study"),
        arc=str(prov_extra.get("arc", "")),
        gpu=str(prov_extra.get("gpu", "")),
        envelope=env,
    )


def _envelopes(store: EvidenceStore) -> Iterator[Mapping[str, Any]]:
    """Every envelope in a store, in file order.

    `EvidenceStore` exposes no public iterator over raw envelopes: `__iter__` and `find` both go
    through `evidence_from_envelope`, which is what raises on this store. The private index is read
    here through a guarded accessor so that adding a public one upstream removes this branch rather
    than breaking it.
    """
    public = getattr(store, "envelopes", None)
    if callable(public):
        yield from public()
        return
    yield from store._index.values()


def _as_fields(value: Any) -> Any:
    """A foreign payload dataclass as its field mapping; anything else untouched.

    Whether a campaign payload decodes to a dataclass or to a dict depends on whether
    `campaign.payloads` happens to be importable in this process, which depends on what else has run.
    The converter reads these as mappings, so it normalises here rather than at forty call sites.

    Payloads defined by this library are returned as they are. That exclusion is the point: `Blind[T]`
    is a registered payload whose whole guarantee is that a held-out label cannot be reached except
    through the adjudication path that records the read, and flattening one into its fields would
    hand over the label with the wrapper removed.
    """
    if not is_dataclass(value) or isinstance(value, type):
        return value
    if type(value).__module__.startswith("reward_lens."):
        return value
    return {f.name: getattr(value, f.name) for f in fields(value)}


class CampaignStore:
    """The campaign's evidence store, opened read-only and decoded permissively.

    Construct it with the directory holding ``evidence.jsonl`` and ``payloads/``. Extra sidecar
    directories are searched in order after the store's own, which is how the campaign's second
    store is reached without copying anything out of it.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        sidecar_dirs: Sequence[str | Path] = (),
    ) -> None:
        self.path = Path(path)
        self.store = EvidenceStore(self.path, readonly=True)
        self.payloads = self.path / "payloads"
        self.sidecar_dirs: tuple[Path, ...] = (self.payloads, *(Path(p) for p in sidecar_dirs))
        # strict=False on purpose, and only after `payload_types` has been checked for a Blind.
        # See the module docstring: the campaign's payload classes are not importable here, and
        # the field dict is what the converter wants from them.
        self._codec = ValueCodec(strict=False)
        self.rows: tuple[CampaignRow, ...] = tuple(
            sorted(
                (_row_from_envelope(env) for env in _envelopes(self.store)),
                key=lambda r: (r.created_at, r.id),
            )
        )

    # -- census ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.rows)

    def observables(self) -> dict[str, int]:
        """Row count per observable, which is the first thing anyone wants from a strange store."""
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row.observable] = counts.get(row.observable, 0) + 1
        return dict(sorted(counts.items()))

    def payload_types(self) -> dict[str, int]:
        """Every ``__type__`` tag in the store, with counts.

        This is the check that licenses ``strict=False``. A store containing a
        `reward_lens.record.labels.Blind` must not be decoded permissively, because a `Blind` that
        decodes to a field dict hands over the held-out label with the wrapper removed. This
        counts them so the licence is measured rather than assumed, and `assert_no_blind_payloads`
        turns the count into a refusal to proceed.
        """
        counts: dict[str, int] = {}

        def walk(obj: Any) -> None:
            if isinstance(obj, dict):
                tag = obj.get("__type__")
                if isinstance(tag, str):
                    counts[tag] = counts.get(tag, 0) + 1
                for value in obj.values():
                    walk(value)
            elif isinstance(obj, list):
                for value in obj:
                    walk(value)

        for row in self.rows:
            walk(row.envelope.get("value"))
        return dict(sorted(counts.items()))

    def assert_no_blind_payloads(self) -> None:
        """Raise if the store holds a blinded label, because permissive decoding would open it.

        An exception rather than a refusal, deliberately. A refusal is a value the caller weighs;
        this is a condition under which the whole module's decoding contract is wrong, and
        continuing would leak an oracle into a detector's reach with nothing downstream able to
        tell.
        """
        offenders = {
            name: n
            for name, n in self.payload_types().items()
            if name.endswith(".Blind") or ".labels." in name
        }
        if offenders:
            raise ValueError(
                f"this store holds blinded payloads {sorted(offenders)}, and CampaignStore decodes "
                f"with strict=False, which would return each Blind as a field dict carrying the "
                f"label it exists to withhold. Import the module that registers them and decode "
                f"strictly, or read this store through reward_lens.record.labels."
            )

    # -- rows -----------------------------------------------------------------

    def by_observable(self, name: str) -> tuple[CampaignRow, ...]:
        return tuple(r for r in self.rows if r.observable == name)

    def by_prefix(self, prefix: str) -> tuple[CampaignRow, ...]:
        return tuple(r for r in self.rows if r.observable.startswith(prefix))

    def has(self, name: str) -> bool:
        return any(r.observable == name for r in self.rows)

    # -- values ---------------------------------------------------------------

    def value(self, row: CampaignRow) -> Any:
        """The decoded payload of one row, as a field mapping for anything foreign.

        Arrays load from whichever sidecar directory holds them. Nothing is cached: a bank of five
        hundred float32 scores is 2 KB and a caller that wants them twice can hold them, whereas a
        cache here would keep every bank the converter has ever touched alive for the life of the
        process.

        A campaign payload comes back as its field dict whether or not the class that wrote it
        happens to be importable, and that normalisation is load-bearing rather than tidiness.
        `_PAYLOAD_REGISTRY` is a process-global that nothing ever cleans, so a single earlier
        import of `campaign.payloads` anywhere in the process, by another test, by a user who has
        the campaign package installed, or by the codec's own import-and-retry, silently flips every
        row in this store from a dict to a dataclass. The converter read `.get` on all of them and
        broke on `AttributeError` in the full suite while passing alone. Normalising here fixes
        every call site at once and makes the converter independent of what else is loaded.

        Payloads defined by this library are returned untouched. That exclusion is the important
        half: `Blind[T]` is a registered payload whose entire guarantee is that the held-out label
        cannot be reached without going through the audited adjudication path, and shallow-converting
        one to its field dict would hand over the label with the wrapper removed.
        """
        decoded = self._codec.decode(self._resolved(row.envelope["value"]), self.payloads)
        return _as_fields(decoded)

    def _resolved(self, obj: Any) -> Any:
        """Rewrite sidecar names the primary directory does not hold to absolute paths.

        ``Path(primary) / "/abs/name.npy"`` is ``/abs/name.npy``, so the kernel's array loader
        follows the rewrite with no change. Anything the search path cannot find is left alone and
        fails in the loader, naming the file, which is the right place for that error: a missing
        sidecar is a corrupt store rather than a case a converter should paper over.
        """
        if isinstance(obj, dict):
            spec = obj.get("__ndarray__")
            if isinstance(spec, dict) and "sidecar" in spec:
                name = spec["sidecar"]
                if not (self.payloads / name).exists():
                    for candidate in self.sidecar_dirs[1:]:
                        if (candidate / name).exists():
                            spec = {**spec, "sidecar": str((candidate / name).resolve())}
                            return {"__ndarray__": spec}
                return obj
            return {k: self._resolved(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._resolved(v) for v in obj]
        return obj

    def sidecar_report(self) -> dict[str, Any]:
        """Referential integrity between rows and payload files, counted rather than trusted.

        Returns the referenced count, how many resolve in the primary directory, how many resolve
        only through the search path, how many resolve nowhere, and how many files in the primary
        directory nothing references.
        """
        referenced: set[str] = set()

        def walk(obj: Any) -> None:
            if isinstance(obj, dict):
                spec = obj.get("__ndarray__")
                if isinstance(spec, dict) and "sidecar" in spec:
                    referenced.add(spec["sidecar"])
                for value in obj.values():
                    walk(value)
            elif isinstance(obj, list):
                for value in obj:
                    walk(value)

        for row in self.rows:
            walk(row.envelope.get("value"))
        present = {p.name for p in self.payloads.glob("*.npy")} if self.payloads.exists() else set()
        fallback = 0
        unresolved = 0
        for name in sorted(referenced - present):
            if any((d / name).exists() for d in self.sidecar_dirs[1:]):
                fallback += 1
            else:
                unresolved += 1
        return {
            "referenced": len(referenced),
            "in_primary": len(referenced & present),
            "via_search_path": fallback,
            "unresolved": unresolved,
            "orphaned_in_primary": len(present - referenced),
        }


__all__ = [
    "ADJUDICATION_PREFIX",
    "RESULT_PREFIX",
    "CampaignRow",
    "CampaignStore",
]

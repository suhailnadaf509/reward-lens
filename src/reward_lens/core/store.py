"""The evidence store.

Append-only, file-backed, trivially inspectable. Envelopes are JSON Lines in ``evidence.jsonl``;
bulk arrays are content-addressed ``.npy`` sidecars under ``payloads/``. There is no database
server; the files are the interface, so the store is diffable and a human can read it. Cards,
the Atlas, papers, and safety cases are views over this store and never compute fresh numbers,
which is what guarantees a card and a paper cite identical values (I5).

The store is a DAG: a derived Evidence names its parents in provenance, and the store refuses to
append a derived Evidence whose parents it cannot resolve (I5). That refusal is the mechanism
behind "every result must compose": you cannot record a forecast that consumed a KUI number the
store never saw.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import warnings
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from reward_lens.core.config import get_settings
from reward_lens.core.errors import ProvenanceError
from reward_lens.core.evidence import Evidence, evidence_from_envelope
from reward_lens.core.types import EvidenceID, TrustLevel


def _quarantine_torn_tail(jsonl: Path, raw: bytes, offset: int) -> None:
    """Move a torn final line out of ``jsonl`` and truncate the file back to good state.

    The partial bytes are appended to ``<name>.partial`` (fsynced before the truncate, so the
    record cannot be lost to the very failure mode it documents) and the JSONL is cut at the end
    of the last newline-terminated good line. The warning is loud on purpose: a torn tail is the
    expected residue of a process killed mid-append, but the operator should know an envelope
    was dropped and will be re-recorded by the re-run.
    """
    partial = jsonl.with_name(jsonl.name + ".partial")
    with partial.open("ab") as fh:
        fh.write(raw if raw.endswith(b"\n") else raw + b"\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.truncate(jsonl, offset)
    warnings.warn(
        f"quarantined a torn final line ({len(raw)} bytes) in {jsonl}; the partial bytes were "
        f"preserved in {partial.name} and the store resumes at the last complete envelope. "
        f"This is the expected residue of a writer killed mid-append.",
        RuntimeWarning,
        stacklevel=3,
    )


def _scan_envelopes(jsonl: Path, *, repair: bool = True) -> Iterator[dict[str, Any]]:
    """Yield envelopes from a store JSONL, tolerating exactly one torn final line.

    An append is not atomic at the filesystem level: a large envelope spans many write
    syscalls, so a process killed mid-append leaves a truncated last line. That torn tail is
    the one corruption an append-only writer can produce by itself, so it is tolerated here
    rather than allowed to block every subsequent load of the store. A line that fails to
    parse anywhere other than the tail cannot be a torn append and still raises: real
    corruption must stay loud.

    ``repair`` controls what happens to the torn tail. A writer opening its own store
    quarantines and truncates it (the file must end at a complete envelope before the next
    append). A reader looking at a store some other process may still be writing must not
    mutate the file at all: a torn tail under a live writer is simply an append in flight,
    and truncating it from the read side corrupts the writer's next append. With
    ``repair=False`` the torn line is skipped and left in place for the store's own writer
    to resolve.
    """
    torn: tuple[bytes, int] | None = None
    with jsonl.open("rb") as fh:
        line_start = 0
        while True:
            raw = fh.readline()
            if not raw:
                break
            next_start = line_start + len(raw)
            stripped = raw.strip()
            if stripped:
                try:
                    env = json.loads(stripped.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if fh.readline():
                        raise
                    torn = (raw, line_start)
                    break
                yield env
            line_start = next_start
    if torn is not None:
        if repair:
            _quarantine_torn_tail(jsonl, torn[0], torn[1])
        else:
            warnings.warn(
                f"skipped a torn final line ({len(torn[0])} bytes) in {jsonl} without "
                f"touching the file; it is either an append in flight or residue the "
                f"store's own writer will quarantine on its next run.",
                RuntimeWarning,
                stacklevel=3,
            )


class EvidenceStore:
    """A directory-backed, append-only store of Evidence.

    Not thread-safe across processes (files are the interface; use one writer), but guarded by a
    lock within a process. The in-memory index maps id to envelope for O(1) lookup; it is built
    once on construction by streaming the JSONL.
    """

    def __init__(self, path: str | Path | None = None, *, readonly: bool = False):
        """``readonly=True`` opens the store for reading without ever mutating its files.

        A reader must pass it whenever the store's writer may still be alive (a merge or a
        spend meter looking at live shards): the torn-tail repair then skips the tail in
        place instead of truncating a file another process is appending to, and ``append``
        refuses outright.
        """
        self.path = Path(path) if path is not None else get_settings().resolved_store()
        self.readonly = readonly
        if not readonly:
            self.path.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.path / "evidence.jsonl"
        self.payloads = self.path / "payloads"
        self._lock = threading.RLock()
        self._index: dict[str, dict[str, Any]] = {}
        # Populated by `merge` on the store it returns; None on an ordinary store.
        self.merge_report: dict[str, Any] | None = None
        self._load_index()

    def _load_index(self) -> None:
        if not self.jsonl.exists():
            return
        for env in _scan_envelopes(self.jsonl, repair=not self.readonly):
            self._index[env["id"]] = env

    # -- write ---------------------------------------------------------------

    def append(self, evidence: Evidence[Any]) -> EvidenceID:
        """Append an Evidence, returning its id. Idempotent on content-derived ids.

        Enforces the DAG invariant: if the Evidence declares parents, every parent must already
        be in the store, else `ProvenanceError`. Re-appending an id that is already present is a
        no-op (the content is identical by construction), so replaying a study is safe.
        """
        if self.readonly:
            raise RuntimeError(
                f"store at {self.path} was opened readonly; appending would race the "
                f"writer this mode exists to protect"
            )
        with self._lock:
            for parent in evidence.provenance.parents:
                if parent not in self._index:
                    raise ProvenanceError(
                        f"Evidence {evidence.id} declares parent {parent} not present in the "
                        f"store; a derived quantity whose parents are missing is an error (I5). "
                        f"Append the parent measurements first."
                    )
            if evidence.id in self._index:
                return evidence.id
            env = evidence.envelope(sidecar_dir=self.payloads)
            # A kill can persist an envelope's bytes but drop its trailing newline; that
            # tail parses fine, so the torn-tail quarantine never sees it, and appending
            # straight onto it would weld two envelopes into one unparseable line that
            # poisons every later load. Heal the terminator before writing.
            heal = b""
            if self.jsonl.exists() and self.jsonl.stat().st_size > 0:
                with self.jsonl.open("rb") as check:
                    check.seek(-1, os.SEEK_END)
                    if check.read(1) != b"\n":
                        heal = b"\n"
            with self.jsonl.open("ab") as fh:
                fh.write(heal + (json.dumps(env, ensure_ascii=False) + "\n").encode("utf-8"))
                fh.flush()
                # fsync narrows the window in which a crash leaves this line torn or unwritten;
                # it cannot close it (a kill mid-write still truncates), so the torn-tail
                # tolerance in _scan_envelopes is the actual recovery guarantee.
                os.fsync(fh.fileno())
            self._index[evidence.id] = env
            return evidence.id

    @classmethod
    def merge(cls, shard_paths: Sequence[str | Path], dest: str | Path) -> "EvidenceStore":
        """Fold per-process shard stores into one store at ``dest``.

        The store is single-writer, so a fan-out run writes one shard store per worker and the
        report step folds them back into a single DAG here. Envelopes are appended in ascending
        ``created_at`` order across all shards and are written to ``dest`` exactly as recorded:
        ids are content-derived, so rebuilding an Evidence on the way through would be redundant
        at best and a provenance break at worst. Payload sidecars are content-addressed files,
        so the union is a copy of whichever ``.npy`` files ``dest`` does not already hold.

        The same invariants ``append`` enforces hold over the merged whole. An envelope whose id
        is already in ``dest`` is skipped, which makes re-merging idempotent and tolerates the
        same envelope reaching the merge twice. Note the limit of that guarantee: ids are
        content-derived over the whole envelope including provenance, so the same logical
        measurement re-recorded with different provenance (a different wall time) carries a
        different id and is NOT deduplicated here. Single-writer shards (the writer lock)
        are what keep that case from arising. A parent id that resolves in no shard and is
        not already in ``dest`` raises `ProvenanceError` before anything is written, naming the
        child, the parent, and the shard: a merge must not admit a derived quantity whose
        parents are missing (I5). A shard path that does not exist is skipped and reported
        rather than raised, so a partially completed run still merges; the returned store
        carries the accounting in ``merge_report`` as
        ``{"merged": n, "skipped_existing": n, "missing_shards": [...]}``. A shard whose
        writer was killed mid-append carries a torn final line; the merge skips that tail
        without touching the shard file (the writer may still be alive), so a crashed or
        in-flight worker blocks neither the merge nor its own re-run.
        """
        dest_store = cls(dest)
        missing_shards: list[str] = []
        arrivals: list[tuple[str, int, dict[str, Any], Path]] = []
        shard_ids: set[str] = set()
        order = 0
        for raw in shard_paths:
            shard = Path(raw)
            if not shard.exists():
                missing_shards.append(str(shard))
                continue
            jsonl = shard / "evidence.jsonl"
            if not jsonl.exists():
                # A shard directory that was created but never written is an empty store,
                # which contributes nothing and is not an error.
                continue
            # Shards are read without mutating them: a torn tail is skipped, never
            # truncated, because the shard's writer may still be alive (a detached arc) and
            # a read-side truncate would corrupt its next append. The skipped envelope is
            # picked up by a later re-merge once the writer finishes or its re-run
            # quarantines the tail itself.
            for env in _scan_envelopes(jsonl, repair=False):
                arrivals.append((env["created_at"], order, env, shard))
                shard_ids.add(env["id"])
                order += 1

        with dest_store._lock:
            # Validate parent resolution over the merged whole before writing anything: a
            # dangling parent means a corrupt shard set, and a half-written merge would only
            # smear that corruption into dest.
            for _, _, env, shard in arrivals:
                for parent in env["provenance"].get("parents", []):
                    if parent not in shard_ids and parent not in dest_store._index:
                        raise ProvenanceError(
                            f"evidence {env['id']} in shard {shard} declares parent {parent}, "
                            f"which resolves in no shard and is not already in the destination "
                            f"store; a derived quantity whose parents are missing must not be "
                            f"merged (I5)."
                        )

            arrivals.sort(key=lambda item: (item[0], item[1]))
            merged = 0
            skipped = 0
            pending = arrivals
            with dest_store.jsonl.open("a", encoding="utf-8") as fh:
                # Timestamp order almost always is dependency order (a parent is created
                # before anything derived from it), but equal timestamps across shards can
                # present a child first, so anything whose parents have not landed yet is
                # deferred to another pass rather than appended out of order.
                while pending:
                    deferred: list[tuple[str, int, dict[str, Any], Path]] = []
                    progressed = False
                    for item in pending:
                        _, _, env, shard = item
                        if env["id"] in dest_store._index:
                            skipped += 1
                            progressed = True
                            continue
                        parents = env["provenance"].get("parents", [])
                        if any(p not in dest_store._index for p in parents):
                            deferred.append(item)
                            continue
                        cls._copy_sidecars(env, shard / "payloads", dest_store.payloads)
                        fh.write(json.dumps(env, ensure_ascii=False) + "\n")
                        dest_store._index[env["id"]] = env
                        merged += 1
                        progressed = True
                    if deferred and not progressed:
                        env = deferred[0][2]
                        raise ProvenanceError(
                            f"could not order evidence {env['id']} after its parents; the "
                            f"shard set contains a parent cycle, which no valid store can."
                        )
                    pending = deferred
                fh.flush()
                # A merge killed mid-write leaves the destination with a torn tail; the load
                # tolerance quarantines it and a re-merge is idempotent, so one fsync at the
                # end (rather than per envelope) is enough.
                os.fsync(fh.fileno())

        dest_store.merge_report = {
            "merged": merged,
            "skipped_existing": skipped,
            "missing_shards": missing_shards,
        }
        return dest_store

    @staticmethod
    def _copy_sidecars(env: dict[str, Any], src_dir: Path, dst_dir: Path) -> None:
        """Copy the ``.npy`` sidecars an envelope references from a shard into ``dst_dir``.

        Sidecar files are content-addressed, so a name that already exists at the destination
        is the same bytes and is left alone. A referenced sidecar missing from its shard is a
        corrupt shard (the envelope's payload can never be read) and raises.
        """

        def walk(obj: Any) -> Iterator[str]:
            if isinstance(obj, dict):
                nd = obj.get("__ndarray__")
                if isinstance(nd, dict) and "sidecar" in nd:
                    yield nd["sidecar"]
                for value in obj.values():
                    yield from walk(value)
            elif isinstance(obj, list):
                for value in obj:
                    yield from walk(value)

        for name in walk(env):
            dst = dst_dir / name
            if dst.exists():
                continue
            src = src_dir / name
            if not src.exists():
                raise ProvenanceError(
                    f"evidence {env['id']} references payload sidecar {name}, which is missing "
                    f"from its shard's payloads directory; the shard is corrupt."
                )
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # -- read ----------------------------------------------------------------

    def __contains__(self, ev_id: str) -> bool:
        return ev_id in self._index

    def __len__(self) -> int:
        return len(self._index)

    def get(self, ev_id: str) -> Evidence[Any]:
        """Load an Evidence by id, reconstructing its typed payload."""
        env = self._index.get(ev_id)
        if env is None:
            raise KeyError(f"no evidence with id {ev_id}")
        return evidence_from_envelope(env, sidecar_dir=self.payloads)

    def __iter__(self) -> Iterator[Evidence[Any]]:
        for ev_id in list(self._index):
            yield self.get(ev_id)

    def envelopes(self) -> Iterator[Mapping[str, Any]]:
        """Every stored envelope as it sits on disk, in file order, decoding nothing.

        `__iter__` and `find` go through `evidence_from_envelope`, which reconstructs the typed
        payload and raises when the payload's class is not importable in this process. That is
        right for a reader who asked for an `Evidence` and wrong for one surveying a store somebody
        else wrote: a converter reading the campaign's store cannot import the campaign's payload
        classes and does not need to, because it reads the envelope's own metadata rather than the
        payload. Without this the converter reached `_index` through a guarded private accessor,
        which is a public need met privately.

        The mappings are the store's own and not copies, so a caller that mutates one corrupts the
        index. Read them.
        """
        yield from self._index.values()

    def find(
        self,
        observable: str | None = None,
        signal: str | None = None,
        dataset: str | None = None,
        readout: str | None = None,
        study: str | None = None,
        min_trust: TrustLevel | None = None,
        latest: bool = False,
    ) -> list[Evidence[Any]]:
        """Query the store with simple structural filters.

        ``signal`` matches an Evidence whose subject names that model fingerprint. ``latest``
        collapses to the most recently created Evidence per (observable, subject) key, which is
        the common "give me the current value" query. All filtering is over the in-memory index;
        for ad hoc analysis use `frame` to get a pandas DataFrame of envelopes.
        """
        out: list[dict[str, Any]] = []
        for env in self._index.values():
            if observable is not None and env["observable"] != observable:
                continue
            subj = env["subject"]
            if signal is not None and signal not in subj.get("signals", []):
                continue
            if dataset is not None and subj.get("dataset") != dataset:
                continue
            if readout is not None and subj.get("readout") != readout:
                continue
            if study is not None and env["provenance"].get("study") != study:
                continue
            if min_trust is not None and env["trust"] < int(min_trust):
                continue
            out.append(env)
        if latest:
            keyed: dict[tuple[str, str], dict[str, Any]] = {}
            for env in out:
                key = (env["observable"], json.dumps(env["subject"], sort_keys=True))
                cur = keyed.get(key)
                if cur is None or env["created_at"] > cur["created_at"]:
                    keyed[key] = env
            out = list(keyed.values())
        out.sort(key=lambda e: e["created_at"])
        return [evidence_from_envelope(e, sidecar_dir=self.payloads) for e in out]

    def frame(self) -> Any:
        """Return a pandas DataFrame of flattened envelopes for ad hoc queries."""
        import pandas as pd

        rows = []
        for env in self._index.values():
            rows.append(
                {
                    "id": env["id"],
                    "observable": env["observable"],
                    "version": env["observable_version"],
                    "signals": ",".join(env["subject"].get("signals", [])),
                    "dataset": env["subject"].get("dataset"),
                    "readout": env["subject"].get("readout"),
                    "trust": TrustLevel(env["trust"]).name,
                    "gauge": env["gauge"],
                    "calibrated": env.get("calibration") is not None,
                    "study": env["provenance"].get("study"),
                    "gpu_seconds": env["provenance"].get("cost", {}).get("gpu_seconds", 0.0),
                    "created_at": env["created_at"],
                }
            )
        return pd.DataFrame(rows)

    # -- DAG -----------------------------------------------------------------

    def parents(self, evidence: Evidence[Any]) -> list[Evidence[Any]]:
        """Return the immediate parent Evidence, raising if any is unresolvable."""
        out = []
        for pid in evidence.provenance.parents:
            if pid not in self._index:
                raise ProvenanceError(f"parent {pid} of {evidence.id} is not in the store")
            out.append(self.get(pid))
        return out

    def ancestors(self, evidence: Evidence[Any]) -> list[Evidence[Any]]:
        """Return the transitive closure of parents (the full derivation DAG of a quantity)."""
        seen: dict[str, Evidence[Any]] = {}
        frontier = list(evidence.provenance.parents)
        while frontier:
            pid = frontier.pop()
            if pid in seen:
                continue
            if pid not in self._index:
                raise ProvenanceError(f"ancestor {pid} of {evidence.id} is not in the store")
            anc = self.get(pid)
            seen[pid] = anc
            frontier.extend(anc.provenance.parents)
        return list(seen.values())


_DEFAULT_STORE: EvidenceStore | None = None


def default_store() -> EvidenceStore:
    """Return the process-wide default store (under the configured store path)."""
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = EvidenceStore()
    return _DEFAULT_STORE


def set_default_store(store: EvidenceStore) -> None:
    global _DEFAULT_STORE
    _DEFAULT_STORE = store


__all__ = ["EvidenceStore", "default_store", "set_default_store"]

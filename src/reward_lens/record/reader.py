"""Reading a record, lazily, without materialising the parts nobody asked for.

The design constraint is one sentence of the record schema: `Run.steps` is a `StepStream` and a
stream is "lazy, chunked, resumable". The case that forces it is the auditor. They are handed a
record of a run that took four hundred optimizer steps, they want steps 200 to 210 because that is
where the reward curve bent, and building the other 390 in memory to hand back eleven is the
difference between an analysis that runs on a laptop and one that does not.

Laziness here is structural rather than incidental. Partitioning by step range is what makes it
real: reading steps 200 to 210 of a 401-step run written in partitions of a hundred opens one
partition directory out of five and never touches the bytes of the other four. Inside the
partition, rows are filtered on their step index before any `Turn`, `Trajectory` or `Group` object
is constructed.

`ReadStats` is on the reader for exactly this reason. "It is lazy" is a claim, and a claim about
what code did not do is only checkable if the code counts. Every file the reader opens, every byte
it reads, and every object it constructs is counted, and the acceptance test asserts on the counts
rather than on the answer being right.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from reward_lens.core.types import Component
from reward_lens.record.compaction import CompactionEvent
from reward_lens.record.provenance import SegmentProvenance
from reward_lens.record.schema import (
    ComponentRef,
    EstimatorSpec,
    FeatureID,
    Group,
    GroupID,
    GroupStats,
    InstrumentEffect,
    OptimizerTelemetry,
    ProbeResult,
    RecordSamplingPolicy,
    RegimeDeclaration,
    Run,
    RunID,
    RunLineage,
    Step,
    StepStream,
    TaskID,
    Trajectory,
    TrajectoryID,
    decode_access,
    decode_foreign,
    decode_regime_reading,
    decode_tensor_ref,
)
from reward_lens.record.tensors import CaptureRef, TensorStore
from reward_lens.record.turns import Turn
from reward_lens.record.writer import partition_lo, partition_name, safe_run_dir


class RecordNotFound(FileNotFoundError):
    """No record at that path, or no manifest in it. A distinct type so a CLI can say which."""


@dataclass
class ReadStats:
    """What the reader actually touched. Mutable on purpose: it accumulates across calls.

    The counters are the evidence for the laziness claim. ``files_opened`` holds paths rather than
    a count so a failing assertion says which extra file was read, which is the only useful form of
    that failure.
    """

    files_opened: list[str] = field(default_factory=list)
    bytes_read: int = 0
    rows_scanned: int = 0
    rows_decoded: int = 0
    steps_materialised: int = 0
    groups_materialised: int = 0
    trajectories_materialised: int = 0
    turns_materialised: int = 0

    def reset(self) -> None:
        self.files_opened.clear()
        self.bytes_read = 0
        self.rows_scanned = 0
        self.rows_decoded = 0
        self.steps_materialised = 0
        self.groups_materialised = 0
        self.trajectories_materialised = 0
        self.turns_materialised = 0

    @property
    def partitions_opened(self) -> tuple[str, ...]:
        """Distinct partition directories touched, in first-touch order."""
        seen: list[str] = []
        for path in self.files_opened:
            part = Path(path).parent.name
            if part not in seen:
                seen.append(part)
        return tuple(seen)


# ---------------------------------------------------------------------------
# Row decoding
# ---------------------------------------------------------------------------


def _decode_turn(row: dict[str, Any]) -> Turn:
    return Turn.from_canonical(json.loads(row["payload"]))


def _decode_trajectory(row: dict[str, Any], turns: Sequence[Turn], sidecar: Path) -> Trajectory:
    payload = json.loads(row["payload"])
    capture = payload.get("capture")
    return Trajectory(
        id=TrajectoryID(row["trajectory_id"]),
        task_ref=TaskID(row["task_ref"]),
        turns=tuple(sorted(turns, key=lambda t: t.index)),
        scores=decode_foreign(payload.get("scores"), sidecar),
        advantage=row.get("advantage"),
        advantage_tokens=decode_tensor_ref(payload.get("advantage_tokens")),
        provenance=tuple(
            SegmentProvenance.from_canonical(p) for p in payload.get("provenance", [])
        ),
        compaction=tuple(CompactionEvent.from_canonical(c) for c in payload.get("compaction", [])),
        labels={k: decode_foreign(v, sidecar) for k, v in payload.get("labels", {}).items()},
        features={FeatureID(k): float(v) for k, v in payload.get("features", {}).items()},
        capture=None if capture is None else CaptureRef.from_canonical(capture),
    )


def _decode_group(row: dict[str, Any], trajectories: Sequence[Trajectory]) -> Group:
    payload = json.loads(row["payload"])
    return Group(
        id=GroupID(row["group_id"]),
        task_ref=TaskID(row["task_ref"]),
        trajectories=tuple(trajectories),
        estimator=EstimatorSpec.from_canonical(payload["estimator"]),
        group_stats=GroupStats.from_canonical(payload["group_stats"]),
    )


def _decode_step(row: dict[str, Any], groups: Sequence[Group]) -> Step:
    payload = json.loads(row["payload"])
    return Step(
        index=row["step_index"],
        groups=tuple(groups),
        schedule={k: float(v) for k, v in payload.get("schedule", {}).items()},
        optimizer=OptimizerTelemetry.from_canonical(payload.get("optimizer", {})),
        probes=tuple(ProbeResult.from_canonical(p) for p in payload.get("probes", [])),
        instrument=InstrumentEffect.from_canonical(payload.get("instrument", {})),
        regime_measured=decode_regime_reading(payload.get("regime_measured", {})),
    )


# ---------------------------------------------------------------------------
# Table backends
# ---------------------------------------------------------------------------


_STEP_INDEX_RE = re.compile(rb'"step_index":\s*(\d+)')


def _read_jsonl(path: Path, stats: ReadStats, keep: set[int] | None) -> list[dict[str, Any]]:
    """Rows from a JSONL table, decoding only those whose step index is wanted.

    The filter runs on the raw line before `json.loads`, matching the ``step_index`` column the
    writer puts in every row of every table. That is not only an optimisation: it is what makes
    ``rows_decoded`` an honest counter, because a reader that parses every row and then discards
    most of them has already paid for the whole partition and the count would say otherwise.
    """
    stats.files_opened.append(str(path))
    out: list[dict[str, Any]] = []
    with path.open("rb") as fh:
        for line in fh:
            stats.bytes_read += len(line)
            stats.rows_scanned += 1
            if keep is not None:
                m = _STEP_INDEX_RE.search(line)
                if m is None or int(m.group(1)) not in keep:
                    continue
            row = json.loads(line.decode("utf-8"))
            stats.rows_decoded += 1
            out.append(row)
    return out


def _read_parquet(path: Path, stats: ReadStats, keep: set[int] | None) -> list[dict[str, Any]]:
    """Rows from a Parquet table, with the step filter pushed into the read."""
    import pyarrow.parquet as pq

    stats.files_opened.append(str(path))
    stats.bytes_read += path.stat().st_size
    filters = None if keep is None else [("step_index", "in", sorted(keep))]
    table = pq.read_table(path, filters=filters)
    rows: list[dict[str, Any]] = table.to_pylist()
    stats.rows_scanned += len(rows)
    stats.rows_decoded += len(rows)
    return rows


# ---------------------------------------------------------------------------
# The stream
# ---------------------------------------------------------------------------


class PartitionedStepStream(StepStream):
    """The lazy `StepStream` over a written record.

    Holds the manifest and a path, and nothing else. Iterating decodes one partition at a time and
    releases it, so peak memory is one partition regardless of run length; slicing decodes only the
    partitions the slice intersects.
    """

    def __init__(self, reader: "RecordReader") -> None:
        self._reader = reader

    def __iter__(self) -> Iterator[Step]:
        return self.slice(0, (max(self.indices) + 1) if self.indices else 0)

    def __len__(self) -> int:
        return len(self._reader.step_indices)

    @property
    def indices(self) -> tuple[int, ...]:
        return self._reader.step_indices

    def slice(self, lo: int, hi: int) -> Iterator[Step]:
        return self._reader.iter_steps(lo, hi)

    def chunks(self, size: int | None = None) -> Iterator[tuple[Step, ...]]:
        """Partition-sized batches by default, which is the chunking the layout already has."""
        return super().chunks(size if size is not None else self._reader.chunk_steps)


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------


class RecordReader:
    """Opens a written record and hands back a `Run` whose steps are lazy.

    ``RecordReader(root, run_id)`` or ``RecordReader.at(run_directory)``. The manifest decides the
    table format, so a record written as Parquet on a machine with pyarrow reads as Parquet and one
    written as JSONL reads as JSONL, with no guessing and no environment sniffing.
    """

    def __init__(self, root: str | Path, run_id: str) -> None:
        self.dir = Path(root) / "runs" / safe_run_dir(run_id)
        self._load()

    @classmethod
    def at(cls, run_dir: str | Path) -> "RecordReader":
        """Open a record by its own directory, for a record that travelled without its root."""
        obj = cls.__new__(cls)
        obj.dir = Path(run_dir)
        obj._load()
        return obj

    def _load(self) -> None:
        manifest_path = self.dir / "manifest.json"
        if not manifest_path.exists():
            raise RecordNotFound(
                f"no record manifest at {manifest_path}. A run directory without one was written "
                f"by a writer that never called close(), so the partitions on disk may be "
                f"incomplete; the last complete partition is the one to resume from."
            )
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.header: dict[str, Any] = json.loads(
            (self.dir / "run.json").read_text(encoding="utf-8")
        )
        self.fmt: str = self.manifest.get("format", "jsonl")
        self.chunk_steps: int = self.manifest.get("chunk_steps", 100)
        self.step_indices: tuple[int, ...] = tuple(self.manifest.get("step_indices", []))
        self.stats = ReadStats()
        self.payloads = self.dir / "payloads"
        self.tensors = TensorStore(self.dir / "tensors")

    # -- the run --------------------------------------------------------------

    def run(self) -> Run:
        """The run header plus a lazy stream. Reads two small JSON files and no step data."""
        h = self.header
        return Run(
            id=RunID(h["id"]),
            kind=h["kind"],
            components={
                Component[k]: ComponentRef.from_canonical(v)
                for k, v in h.get("components", {}).items()
            },
            access=decode_access(h.get("access", {})),
            regime=RegimeDeclaration.from_canonical(h.get("regime", {})),
            steps=PartitionedStepStream(self),
            arms=tuple(RunID(a) for a in h.get("arms", [])),
            coupling=decode_foreign(h.get("coupling"), self.payloads),
            lineage=RunLineage.from_canonical(h.get("lineage", {})),
            sampling_policy=RecordSamplingPolicy.from_canonical(h.get("sampling_policy", {})),
        )

    # -- steps ----------------------------------------------------------------

    def partitions_for(self, lo: int, hi: int) -> list[int]:
        """The partition starts a half-open step range ``[lo, hi)`` intersects.

        Filtered out of the partitions that exist rather than enumerated across the range that was
        asked for. The other way round reads more naturally and hangs: a caller asking for every
        step with a sentinel bound, which is the obvious way to say "the whole run", walks
        `range(-2**62, 2**62, chunk)` before it ever reaches the membership test. It does not raise
        and it does not return, which is the worst shape a bug can have. The partitions on disk are
        a handful; the range asked for is unbounded. Iterate the small one.
        """
        if hi <= lo:
            return []
        first = partition_lo(lo, self.chunk_steps)
        last = partition_lo(hi - 1, self.chunk_steps)
        known = {partition_lo(i, self.chunk_steps) for i in self.step_indices}
        return sorted(p for p in known if first <= p <= last)

    def iter_steps(self, lo: int, hi: int) -> Iterator[Step]:
        """Steps with index in ``[lo, hi)``, one partition at a time."""
        for start in self.partitions_for(lo, hi):
            top = start + self.chunk_steps
            wanted = {i for i in self.step_indices if start <= i < top and lo <= i < hi}
            if not wanted:
                continue
            yield from self._decode_partition(start, wanted)

    def step(self, index: int) -> Step:
        """One step. Opens one partition."""
        for step in self.iter_steps(index, index + 1):
            return step
        raise KeyError(f"step {index} is not in this record; it has {len(self.step_indices)} steps")

    def _table(self, start: int, table: str, keep: set[int] | None) -> list[dict[str, Any]]:
        part = self.dir / "steps" / partition_name(start, self.chunk_steps)
        suffix = "parquet" if self.fmt == "parquet" else "jsonl"
        path = part / f"{table}.{suffix}"
        if not path.exists():
            return []
        if self.fmt == "parquet":
            return _read_parquet(path, self.stats, keep)
        return _read_jsonl(path, self.stats, keep)

    def _decode_partition(self, start: int, wanted: set[int]) -> Iterator[Step]:
        """Assemble the wanted steps of one partition from its four tables, bottom up.

        Rows are grouped by their foreign keys before anything is constructed, so a turn belonging
        to a step outside ``wanted`` is never turned into a `Turn`. That is what the
        ``*_materialised`` counters measure.

        The grouping key is the full path (step, group, trajectory) rather than the id alone. Ids
        are unique in a well-formed record and a reader that assumes so silently welds together the
        turns of two trajectories that happen to share an id, which is a plausible state for a
        synthetic record or a converter mid-development and produces a tiling error four frames
        away from the cause.
        """
        turn_rows = self._table(start, "turns", wanted)
        traj_rows = self._table(start, "trajectories", wanted)
        group_rows = self._table(start, "groups", wanted)
        step_rows = self._table(start, "steps", wanted)

        turns_by_traj: dict[tuple[int, str, str], list[Turn]] = {}
        for row in turn_rows:
            key = (row["step_index"], row["group_id"], row["trajectory_id"])
            turns_by_traj.setdefault(key, []).append(_decode_turn(row))
            self.stats.turns_materialised += 1

        trajs_by_group: dict[tuple[int, str], list[Trajectory]] = {}
        for row in traj_rows:
            key = (row["step_index"], row["group_id"], row["trajectory_id"])
            traj = _decode_trajectory(row, turns_by_traj.get(key, []), self.payloads)
            trajs_by_group.setdefault((row["step_index"], row["group_id"]), []).append(traj)
            self.stats.trajectories_materialised += 1

        groups_by_step: dict[int, list[Group]] = {}
        for row in group_rows:
            group = _decode_group(row, trajs_by_group.get((row["step_index"], row["group_id"]), []))
            groups_by_step.setdefault(row["step_index"], []).append(group)
            self.stats.groups_materialised += 1

        for row in sorted(step_rows, key=lambda r: r["step_index"]):
            self.stats.steps_materialised += 1
            yield _decode_step(row, groups_by_step.get(row["step_index"], []))

    # -- accounting -----------------------------------------------------------

    def total_bytes(self) -> int:
        """Bytes of step data on disk, so a laziness claim can be stated as a fraction."""
        return sum(p.stat().st_size for p in (self.dir / "steps").rglob("*") if p.is_file())


def open_run(root: str | Path, run_id: str) -> Run:
    """Convenience: open a record and return its `Run`. The stream stays lazy."""
    return RecordReader(root, run_id).run()


__all__ = [
    "PartitionedStepStream",
    "ReadStats",
    "RecordNotFound",
    "RecordReader",
    "open_run",
]

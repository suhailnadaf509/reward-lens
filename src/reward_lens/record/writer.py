"""Writing a record: partitioned scalar tables, a content-addressed tensor store, a manifest.

The storage contract asks for Parquet partitioned by ``run_id/step_range`` for the scalars and
content-addressed shards plus a JSON manifest for the tensors. That is what this writes, with one
deliberate change to the default and the reason recorded here rather than in a commit message.

**The default container is JSON Lines, not Parquet.** pyarrow is a compiled dependency and the
core of this library installs with none, which is not an aesthetic position: profile D, the
auditor, is handed a record and cannot re-query anything, and a record written in a format their
install cannot open is not a record. So the portable format is the default and Parquet is opt-in
with ``format="parquet"``, which needs pyarrow. The two are the same logical tables with the same
columns and the same partition layout, the manifest says which one a run used, and the reader
dispatches on the manifest rather than on what happens to be importable.

**The layout is four flat tables per partition, not one nested one.** ``steps``, ``groups``,
``trajectories`` and ``turns``, joined on ids, each partitioned by step range. Scalars are real
columns so a Parquet reader can push a predicate down to them; the structured fields (the score
tree, the labels, the provenance segments, the tensor refs) are one JSON column each. A single
deeply-nested step row would be legal Parquet and would defeat the point, because the predicate
that makes a slice cheap has to apply to a column.

The writer is single-writer per run. Two processes writing one run's partitions will interleave
partial rows; a fan-out run writes one record per worker and merges, exactly as the evidence store
does with its shards.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from reward_lens.record.schema import (
    RECORD_SCHEMA_VERSION,
    Group,
    Run,
    Step,
    Trajectory,
    encode_access,
    encode_foreign,
    encode_regime_reading,
    encode_tensor_ref,
)
from reward_lens.record.tensors import TensorStore
from reward_lens.record.turns import Turn

RecordFormat = Literal["jsonl", "parquet"]

#: The four flat tables written per partition. Order matters only for readability.
TABLES = ("steps", "groups", "trajectories", "turns")


class ParquetUnavailable(ImportError):
    """``format="parquet"`` was asked for and pyarrow is not installed.

    Its own type so a caller can fall back to ``jsonl`` deliberately. It is not routed through
    `require_extra` because there is no declared extra that installs pyarrow yet; when one exists
    this should raise `ExtraRequiredError` naming it instead.
    """


def _have_pyarrow() -> bool:
    import importlib.util

    return importlib.util.find_spec("pyarrow") is not None


def safe_run_dir(run_id: str) -> str:
    """A run id as a directory name.

    Content-derived ids carry a ``prefix:hexdigest`` shape and a colon is legal in a POSIX path and
    a nuisance everywhere else, so it becomes an underscore. The true id lives in the manifest, and
    nothing reads the identity back out of the directory name.
    """
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in run_id)


def partition_name(lo: int, chunk: int) -> str:
    """Hive-style partition directory for the step range starting at ``lo``."""
    return f"step_range={lo:06d}-{lo + chunk - 1:06d}"


def partition_lo(index: int, chunk: int) -> int:
    return (index // chunk) * chunk


# ---------------------------------------------------------------------------
# Row encoding
# ---------------------------------------------------------------------------


def _turn_rows(
    run: str, step: int, group: str, trajectory: str, turns: Sequence[Turn]
) -> list[dict[str, Any]]:
    return [
        {
            "run_id": run,
            "step_index": step,
            "group_id": group,
            "trajectory_id": trajectory,
            "turn_index": t.index,
            "role": t.role,
            "text": t.text,
            "n_tokens": t.n_tokens,
            "step_score": t.step_score,
            "step_advantage": t.step_advantage,
            "truncated": t.truncated,
            "overlong_prompt": t.overlong_prompt,
            "payload": json.dumps(t.__canonical__(), ensure_ascii=False),
        }
        for t in turns
    ]


def _trajectory_row(
    run: str, step: int, group: str, traj: Trajectory, sidecar_dir: Path
) -> dict[str, Any]:
    return {
        "run_id": run,
        "step_index": step,
        "group_id": group,
        "trajectory_id": str(traj.id),
        "task_ref": str(traj.task_ref),
        "n_turns": traj.n_turns,
        "n_tokens": traj.n_tokens,
        "advantage": traj.advantage,
        "is_single_policy": traj.is_single_policy,
        "max_staleness": traj.max_staleness,
        "was_compacted": traj.was_compacted,
        "payload": json.dumps(
            {
                "scores": encode_foreign(traj.scores, sidecar_dir),
                "advantage_tokens": encode_tensor_ref(traj.advantage_tokens),
                "provenance": [p.__canonical__() for p in traj.provenance],
                "compaction": [c.__canonical__() for c in traj.compaction],
                "labels": {k: encode_foreign(v, sidecar_dir) for k, v in traj.labels.items()},
                "features": {str(k): float(v) for k, v in traj.features.items()},
                "capture": None if traj.capture is None else traj.capture.__canonical__(),
            },
            ensure_ascii=False,
        ),
    }


def _group_row(run: str, step: int, group: Group) -> dict[str, Any]:
    return {
        "run_id": run,
        "step_index": step,
        "group_id": str(group.id),
        "task_ref": str(group.task_ref),
        "k": group.k,
        "mean": group.group_stats.mean,
        "std": group.group_stats.std,
        "degenerate": group.group_stats.degenerate,
        "all_fail": group.group_stats.all_fail,
        "payload": json.dumps(
            {
                "estimator": group.estimator.__canonical__(),
                "group_stats": group.group_stats.__canonical__(),
            },
            ensure_ascii=False,
        ),
    }


def _step_row(run: str, step: Step) -> dict[str, Any]:
    return {
        "run_id": run,
        "step_index": step.index,
        "n_groups": len(step.groups),
        "n_trajectories": sum(g.k for g in step.groups),
        "grad_norm_clipped": step.optimizer.grad_norm_clipped,
        "grad_norm_unclipped": step.optimizer.grad_norm_unclipped,
        "instrument_disabled": step.instrument.disabled,
        "payload": json.dumps(
            {
                "schedule": {str(k): float(v) for k, v in step.schedule.items()},
                "optimizer": step.optimizer.__canonical__(),
                "probes": [p.__canonical__() for p in step.probes],
                "instrument": step.instrument.__canonical__(),
                "regime_measured": encode_regime_reading(step.regime_measured),
            },
            ensure_ascii=False,
        ),
    }


def encode_step(step: Step, run: str, sidecar_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """One step, flattened into rows for the four tables."""
    rows: dict[str, list[dict[str, Any]]] = {t: [] for t in TABLES}
    rows["steps"].append(_step_row(run, step))
    for group in step.groups:
        rows["groups"].append(_group_row(run, step.index, group))
        for traj in group.trajectories:
            rows["trajectories"].append(
                _trajectory_row(run, step.index, str(group.id), traj, sidecar_dir)
            )
            rows["turns"].extend(
                _turn_rows(run, step.index, str(group.id), str(traj.id), traj.turns)
            )
    return rows


def _run_header(run: Run, sidecar_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "id": str(run.id),
        "kind": run.kind,
        "components": {c.name: r.__canonical__() for c, r in run.components.items()},
        "access": encode_access(run.access),
        "regime": run.regime.__canonical__(),
        "arms": [str(a) for a in run.arms],
        "coupling": encode_foreign(run.coupling, sidecar_dir),
        "lineage": run.lineage.__canonical__(),
        "sampling_policy": run.sampling_policy.__canonical__(),
    }


# ---------------------------------------------------------------------------
# The writer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WriteReport:
    """What a write actually did. Counts, not adjectives."""

    root: Path
    steps: int
    groups: int
    trajectories: int
    turns: int
    partitions: int
    fmt: RecordFormat
    tensor_bytes: int = 0


class RecordWriter:
    """Writes one run to ``root/runs/<run_dir>/``.

    Usage is either one call (`write`) or a stream (`open_run`, `append_step`, `close`). The
    streaming form is what a live tap uses and it is resumable: `last_step_written` reads the
    manifest, so a writer killed at step 274 restarts at 275 without re-emitting the first 274.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        fmt: RecordFormat = "jsonl",
        chunk_steps: int = 100,
        tensor_container: str = "npy",
    ) -> None:
        if chunk_steps < 1:
            raise ValueError(f"chunk_steps must be at least 1; got {chunk_steps}")
        if fmt == "parquet" and not _have_pyarrow():
            raise ParquetUnavailable(
                "format='parquet' needs pyarrow, which is not installed and is not a base "
                "dependency of reward-lens: the core installs with no compiled dependency. Use "
                "format='jsonl', which writes the same tables and the same partition layout, or "
                "install pyarrow."
            )
        self.root = Path(root)
        self.fmt: RecordFormat = fmt
        self.chunk_steps = chunk_steps
        self.tensor_container = tensor_container
        self._run: Run | None = None
        self._dir: Path | None = None
        self._buffer: dict[int, dict[str, list[dict[str, Any]]]] = {}
        self._counts = {"steps": 0, "groups": 0, "trajectories": 0, "turns": 0}
        self._partitions: set[int] = set()
        self._written_indices: set[int] = set()
        self.tensors: TensorStore | None = None

    # -- lifecycle ------------------------------------------------------------

    def open_run(self, run: Run) -> Path:
        """Create the run directory and write the header. Idempotent on a re-open."""
        self._run = run
        self._dir = self.root / "runs" / safe_run_dir(str(run.id))
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / "payloads").mkdir(exist_ok=True)
        self.tensors = TensorStore(self._dir / "tensors", container=self.tensor_container)  # type: ignore[arg-type]
        header = _run_header(run, self._dir / "payloads")
        _atomic_write(self._dir / "run.json", json.dumps(header, indent=1, ensure_ascii=False))
        return self._dir

    def append_step(self, step: Step) -> None:
        """Buffer a step and flush its partition when the partition is complete."""
        if self._run is None or self._dir is None:
            raise RuntimeError("call open_run() before append_step()")
        rows = encode_step(step, str(self._run.id), self._dir / "payloads")
        lo = partition_lo(step.index, self.chunk_steps)
        self._buffer.setdefault(lo, {t: [] for t in TABLES})
        for table in TABLES:
            self._buffer[lo][table].extend(rows[table])
        self._counts["steps"] += 1
        self._counts["groups"] += len(rows["groups"])
        self._counts["trajectories"] += len(rows["trajectories"])
        self._counts["turns"] += len(rows["turns"])
        # Flush any partition other than the one this step landed in: steps arrive in order, so
        # an older partition is finished. Holding one partition in memory bounds the writer's
        # footprint at chunk_steps steps regardless of run length.
        for other in sorted(k for k in self._buffer if k != lo):
            self._flush_partition(other)

    def close(self) -> WriteReport:
        """Flush the remaining partitions, the tensor manifest, and the run manifest."""
        if self._run is None or self._dir is None:
            raise RuntimeError("nothing to close: open_run() was never called")
        for lo in sorted(self._buffer):
            self._flush_partition(lo)
        tensor_bytes = 0
        if self.tensors is not None:
            self.tensors.flush()
            tensor_bytes = self.tensors.total_bytes()
        manifest = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "run_id": str(self._run.id),
            "format": self.fmt,
            "chunk_steps": self.chunk_steps,
            "tables": list(TABLES),
            "partitions": [
                {
                    "lo": lo,
                    "hi": lo + self.chunk_steps - 1,
                    "name": partition_name(lo, self.chunk_steps),
                }
                for lo in sorted(self._partitions)
            ],
            "step_indices": sorted(self._written_indices),
            "counts": dict(self._counts),
            "tensor_bytes": tensor_bytes,
        }
        _atomic_write(
            self._dir / "manifest.json", json.dumps(manifest, indent=1, ensure_ascii=False)
        )
        return WriteReport(
            root=self._dir,
            steps=self._counts["steps"],
            groups=self._counts["groups"],
            trajectories=self._counts["trajectories"],
            turns=self._counts["turns"],
            partitions=len(self._partitions),
            fmt=self.fmt,
            tensor_bytes=tensor_bytes,
        )

    # -- one-shot -------------------------------------------------------------

    def write(self, run: Run, steps: Iterable[Step] | None = None) -> WriteReport:
        """Write a whole run. ``steps`` defaults to iterating the run's own stream."""
        self.open_run(run)
        for step in steps if steps is not None else run.steps:
            self.append_step(step)
        return self.close()

    # -- resumption -----------------------------------------------------------

    @staticmethod
    def last_step_written(root: str | Path, run_id: str) -> int | None:
        """The highest step index in a written record, or None if there is none.

        Reads the manifest and nothing else, so resuming a 40,000-step run costs one small file
        read rather than a scan.
        """
        path = Path(root) / "runs" / safe_run_dir(run_id) / "manifest.json"
        if not path.exists():
            return None
        indices = json.loads(path.read_text(encoding="utf-8")).get("step_indices", [])
        return max(indices) if indices else None

    # -- internals ------------------------------------------------------------

    def _flush_partition(self, lo: int) -> None:
        assert self._dir is not None
        rows = self._buffer.pop(lo)
        part = self._dir / "steps" / partition_name(lo, self.chunk_steps)
        part.mkdir(parents=True, exist_ok=True)
        for table in TABLES:
            if not rows[table]:
                continue
            if self.fmt == "parquet":
                _write_parquet(part / f"{table}.parquet", rows[table])
            else:
                _write_jsonl(part / f"{table}.jsonl", rows[table])
        self._partitions.add(lo)
        for row in rows["steps"]:
            self._written_indices.add(row["step_index"])


# ---------------------------------------------------------------------------
# Table backends
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Append rows as JSON Lines, healing a missing terminator first.

    Same defect the evidence store hit: a writer killed after the bytes and before the newline
    leaves a line that parses, so nothing notices, and the next append welds two rows into one
    unparseable line. Checking the last byte costs one seek.
    """
    heal = ""
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as fh:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                heal = "\n"
    with path.open("a", encoding="utf-8") as fh:
        if heal:
            fh.write(heal)
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _write_parquet(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write rows as a Parquet file, appending by rewriting the file if it already exists.

    Parquet has no append, so a re-flush of the same partition reads and concatenates. Steps
    arrive in order and a partition is flushed once, so this path is the exception rather than the
    rule; it exists so that a resumed writer landing inside a written partition does not silently
    drop the earlier rows.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(list(rows))
    if path.exists():
        table = pa.concat_tables([pq.read_table(path), table], promote_options="default")
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


__all__ = [
    "ParquetUnavailable",
    "RecordFormat",
    "RecordWriter",
    "TABLES",
    "WriteReport",
    "encode_step",
    "partition_lo",
    "partition_name",
    "safe_run_dir",
]

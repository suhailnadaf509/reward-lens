"""Sweeping an Observable or index across a checkpoint sequence, cached and resumable.

A developmental measurement is the same Observable run at every checkpoint, so this module is thin by
design: it walks a verified `CheckpointSequence`, runs the measurement at each step, writes the
per-checkpoint Evidence to the store, and returns the trajectory with the step as the covariate. The
two properties that make it usable at RM-Pythia scale are caching and resumability. The training run
is paid for once (a few hundred GPU-hours); a sweep over it must never recompute a
checkpoint it has already measured, whether because a previous sweep crashed halfway or because a new
index is being added to an existing study. So each checkpoint's result is committed to the evidence
store, and a small per-sweep manifest records which steps are done. A second run reads the manifest,
loads the already-computed Evidence, and appends nothing new.

The store is content-addressed and idempotent on append, which is the safety net: even if two
processes computed the same checkpoint, the second append is a no-op. The manifest is the optimization
on top of that safety net, and it is what turns "append is idempotent" into "the expensive forward
passes are skipped entirely". The manifest lives beside the store as a plain JSON file, so a sweep is
as inspectable and diffable as the store it rides on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from reward_lens.core.evidence import Evidence
from reward_lens.core.store import EvidenceStore, default_store
from reward_lens.core.types import ModelFP, content_hash
from reward_lens.measure.base import Context, Observable, run

if TYPE_CHECKING:
    from reward_lens.dynamics.checkpoints import CheckpointSequence

# A sweep target is either a full Observable (run under the gates via `measure.run`) or a plain
# callable that takes the loaded signal and the view and returns an Evidence. The callable form is the
# escape hatch for a bare score sweep or a bespoke per-checkpoint reduction that is not worth wrapping
# as an Observable; both flow through the same caching and manifest machinery.
SweepCallable = Callable[[Any, Any], Evidence[Any]]


@dataclass
class SweepPoint:
    """One checkpoint's measurement in a trajectory.

    ``step`` is the training-time covariate; ``model_fp`` names the checkpoint; ``evidence`` is the
    measurement (freshly computed or loaded from cache); ``from_cache`` records which, so a caller can
    assert resumability directly (a resumed sweep has every point ``from_cache=True``).
    """

    step: int
    model_fp: ModelFP
    evidence: Evidence[Any]
    from_cache: bool


@dataclass
class Trajectory:
    """A developmental trajectory: the per-checkpoint Evidence indexed by training step.

    Returned by `sweep_over_checkpoints`. ``points`` is ordered by step; ``sweep_id`` keys the sweep to
    its (observable, view, chain) triple so a resumed run finds the same manifest. The helpers pull the
    covariate and a chosen scalar out of the Evidence so a curve module can plot a value against step
    without re-loading the store.
    """

    points: list[SweepPoint]
    observable: str
    sweep_id: str
    n_computed: int = 0
    n_cached: int = 0

    @property
    def steps(self) -> list[int]:
        """The training steps in order (the developmental covariate)."""
        return [p.step for p in self.points]

    @property
    def evidence(self) -> list[Evidence[Any]]:
        """The per-checkpoint Evidence in step order."""
        return [p.evidence for p in self.points]

    def values(self, extract: Callable[[Any], float]) -> list[float]:
        """Pull one scalar per checkpoint by applying ``extract`` to each Evidence value payload."""
        return [float(extract(p.evidence.value)) for p in self.points]


class SweepManifest:
    """A per-sweep record of which checkpoint steps are computed, for resumability.

    A plain JSON sidecar next to the evidence store mapping each step to the Evidence id computed for
    it. It is deliberately not part of the store's DAG: it is bookkeeping that lets a sweep skip work,
    and losing it only costs recomputation, never correctness (the store append stays idempotent). The
    file is keyed by ``sweep_id`` so distinct sweeps (different observable, view, or chain) never share
    a manifest.
    """

    def __init__(self, store: EvidenceStore, sweep_id: str):
        self.store = store
        self.sweep_id = sweep_id
        self.dir = Path(store.path) / "dynamics_sweeps"
        self.path = self.dir / f"{sweep_id}.json"
        self._entries: dict[str, str] = {}
        self._meta: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = dict(raw.get("steps", {}))
            self._meta = dict(raw.get("meta", {}))

    def _flush(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = {"sweep_id": self.sweep_id, "meta": self._meta, "steps": self._entries}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def set_meta(self, **meta: Any) -> None:
        self._meta.update(meta)
        self._flush()

    def get(self, step: int) -> str | None:
        """The Evidence id computed for ``step``, or None if the step is not yet done."""
        return self._entries.get(str(step))

    def record(self, step: int, evidence_id: str) -> None:
        self._entries[str(step)] = evidence_id
        self._flush()


def _view_id(view: Any) -> str:
    """A stable id for a data view: its declared id or checksum, else a hash of its repr."""
    for attr in ("dataset_id",):
        value = getattr(view, attr, None)
        if value:
            return str(value)
    checksum = getattr(view, "checksum", None)
    if callable(checksum):
        return str(checksum())
    return content_hash({"repr": repr(view)}, "dv")


def _observable_identity(observable: Observable | SweepCallable) -> tuple[str, str]:
    """Return ``(name, version)`` for either an Observable or a bare callable sweep target."""
    name = getattr(observable, "name", None)
    if name is not None:
        return str(name), str(getattr(observable, "version", "0"))
    qual = getattr(observable, "__qualname__", None) or getattr(observable, "__name__", "callable")
    return f"callable:{qual}", "0"


def compute_sweep_id(
    sequence: "CheckpointSequence",
    observable: Observable | SweepCallable,
    view: Any,
    readout: str,
) -> str:
    """The content id keying a resumable sweep to its (observable, view, chain, readout).

    Derived from the observable identity, the view id, the chain signature (its head link, which
    commits to every checkpoint), and the readout name. Two invocations with the same four land on the
    same id and therefore the same manifest, which is what makes a sweep resumable; a change to any of
    them starts a fresh sweep rather than silently reusing stale results.
    """
    name, version = _observable_identity(observable)
    material = {
        "observable": name,
        "version": version,
        "view": _view_id(view),
        "chain": sequence.signature(),
        "readout": readout,
    }
    return content_hash(material, "sweep").split(":")[1]


def sweep_over_checkpoints(
    sequence: "CheckpointSequence",
    observable: Observable | SweepCallable,
    *,
    view: Any = None,
    readout: str = "reward",
    store: EvidenceStore | None = None,
    frame: str | None = None,
    study: str | None = None,
    regime: dict[str, Any] | None = None,
    resume: bool = True,
    verify: bool = True,
) -> Trajectory:
    """Run an Observable or index across a checkpoint sequence, cached and resumable.

    For each checkpoint in order: if ``resume`` and the sweep manifest already has this step (and its
    Evidence is in the store), the stored Evidence is loaded and the step is skipped without loading
    the model; otherwise the checkpoint's signal is materialized, the measurement is run (an Observable
    goes through `measure.run` so the gates apply; a callable is invoked directly), the resulting
    Evidence is appended to the store, and the manifest records it. Either way the point carries the
    training step as its covariate.

    The sweep verifies the chain first (``verify``), because a developmental trajectory read off an
    untrusted chain is meaningless. It returns a `Trajectory` whose ``n_computed`` and
    ``n_cached`` counts make resumability checkable: a second identical call recomputes nothing,
    appends nothing to the store, and reports every point as cached.
    """
    # An empty EvidenceStore is falsy (``__len__`` is 0), so this must test ``is None`` rather than
    # truthiness or a caller's fresh store would be silently swapped for the default singleton.
    store = store if store is not None else default_store()
    if verify:
        sequence.verify()  # chain-only; deep fingerprint verification is an explicit opt-in

    name, _version = _observable_identity(observable)
    sweep_id = compute_sweep_id(sequence, observable, view, readout)
    manifest = SweepManifest(store, sweep_id)
    manifest.set_meta(observable=name, view=_view_id(view), chain=sequence.signature())

    is_observable = isinstance(observable, Observable)
    points: list[SweepPoint] = []
    n_computed = 0
    n_cached = 0

    for checkpoint in sequence:
        step = checkpoint.step
        cached_id = manifest.get(step) if resume else None
        if cached_id is not None and cached_id in store:
            points.append(
                SweepPoint(
                    step=step,
                    model_fp=checkpoint.model_fp,
                    evidence=store.get(cached_id),
                    from_cache=True,
                )
            )
            n_cached += 1
            continue

        signal = checkpoint.load()
        if is_observable:
            ctx = Context(
                signal=signal,
                view=view,
                readout=readout,
                frame=frame,
                study=study,
                regime={**(regime or {}), "checkpoint_step": step},
            )
            evidence = run(observable, ctx)  # type: ignore[arg-type]
        else:
            evidence = observable(signal, view)  # type: ignore[operator]

        store.append(evidence)
        manifest.record(step, evidence.id)
        points.append(
            SweepPoint(
                step=step,
                model_fp=checkpoint.model_fp,
                evidence=evidence,
                from_cache=False,
            )
        )
        n_computed += 1

    return Trajectory(
        points=points,
        observable=name,
        sweep_id=sweep_id,
        n_computed=n_computed,
        n_cached=n_cached,
    )


__all__ = [
    "SweepCallable",
    "SweepPoint",
    "Trajectory",
    "SweepManifest",
    "compute_sweep_id",
    "sweep_over_checkpoints",
]

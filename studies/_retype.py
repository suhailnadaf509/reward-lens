"""The retype layer: what a science declares once it stands on the kernel rather than beside it.

The sixteen sciences were written before the kernel existed. Each is a frozen `StudySpec` plus an
`analyze(StudyRun) -> StudyResult`, and each one plants its own organism, computes its own numbers
and hands back a bare `dict[str, float]`. That worked, and it produced every confirmed result this
project has. What it cannot do is answer three questions the kernel asks of everything else: what
registered quantity is this number, can the plan that promises it actually produce it, and what
happens when the subject in front of you does not contain the phenomenon.

This module is the retype, and it is an addition rather than a rewrite. `analyze` keeps its
signature and keeps running under `reward_lens.studies.runner`, so no confirmed result moves. Beside
it each science now declares:

- a `ScienceRetype`, which binds every metric the frozen spec registers to a **registered** quantity
  id and names the arc that produces it, so `check_closure` has something to check;
- a `read(run: Run) -> Reading`, which takes a `record/` object rather than a planted organism and
  returns `Evidence | Refusal` rather than a metrics dict.

**The refusals are the point of the second one.** Nine of the fifteen sciences cannot be computed
from a record that does not carry activations, and saying so with a reason and a remedy is a better
answer than the synthetic number the science would otherwise return. Pointed at the 12-step GRPO
fixture, `s02_gauge.read` refuses `ACCESS_INSUFFICIENT` because the record is RECORD-access and the
canonical cosine needs a readout vector; `s05_capacity.read` refuses because a one-leaf score tree
has no Welch floor to test; `s03_thermo.read` computes what it can and refuses the rest. Those are
three different sentences and a caller can act on each.

**What is not decided here.** Registering a quantity id is a maintainer's decision, so where a
science computes something with no registered id this module records the gap in `MetricSpec.gap`
rather than inventing one, and `unregistered_metrics()` enumerates them for the report. A metric
with a gap does not close, which is correct: a plan that promises a number nothing can name is not
a plan.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from reward_lens.core.closure import UNBUDGETED, ArcSpec, CostBudget, MetricBinding, Output
from reward_lens.core.evidence import make_evidence
from reward_lens.core.provenance import Provenance
from reward_lens.core.quantity import QUANTITIES, CostModel, QuantityID, load_quantities
from reward_lens.core.reading import Reading, Refusal, RefusalReason, refuse_access
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Component,
    GaugeStatus,
    SubjectRef,
    missing_access,
)
from reward_lens.studies.plan import Plan
from reward_lens.studies.spec import StudySpec

#: Bumped when the shape of what `read` emits changes, not when a science changes its arithmetic.
RETYPE_VERSION = "1.0"


# ---------------------------------------------------------------------------
# One metric, bound
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """One frozen metric name, bound to a registered quantity on a named subject.

    `metric` is the string the science's `Prediction` already froze, so it cannot be changed to suit
    the binding: the binding has to reach the metric where it is. `quantity` is a registered id.
    `arc` is the unit of work that produces it, which is what makes the plan a graph rather than a
    list.

    `frame`, `arm` and `dataset` exist because two metrics in one science are often the same
    quantity read under different conditions. `canonical_cos` and `raw_cos` are both
    `grader.objective_geometry`; what separates them is the frame, and `SubjectRef.frame` is the
    field for that. Folding the difference into the metric name instead would make the subject a
    lie, and closure resolves on the subject.

    `gap` is set instead of `quantity` when the science computes something the registry has no id
    for. It carries the unit, the invariance group and the definition, which is what a registration
    request needs. A gapped metric fails closure on purpose.

    **`source` keeps two different jobs apart, and conflating them would put a unit error in a
    payload.** A binding is a plan-time statement about arcs: it answers "which unit of work
    produces the number this prediction names", and `chi_recovery_corr` is answered by the arc that
    measures `selection.differential_S` on the planted organism even though a recovery correlation
    is not itself a covariance. The quantity stamped on an emitted `Evidence` is a run-time
    statement about a number, and that one has to be exact, because it is what the unit machinery
    keys on. So `source="organism"` marks a metric that only the planted
    calibration arm produces: it is bound for closure and `read` never reports it, because `read`
    takes a record and a record contains no plant.
    """

    metric: str
    quantity: QuantityID = ""
    arc: str = ""
    frame: str = ""
    arm: str = ""
    dataset: str = ""
    source: str = "record"
    note: str = ""
    gap: str = ""

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("a metric spec needs the metric name the study froze")
        if bool(self.quantity) == bool(self.gap):
            raise ValueError(
                f"metric {self.metric!r} must name exactly one of a registered quantity or a gap. "
                f"Naming both hides the request; naming neither hides the metric."
            )
        if self.quantity and not self.arc:
            raise ValueError(
                f"metric {self.metric!r} binds to {self.quantity} and names no arc. Closure asks "
                f"which unit of work produces it, and 'somewhere in the analysis' is not an answer."
            )

    @property
    def bound(self) -> bool:
        return bool(self.quantity)

    @property
    def from_record(self) -> bool:
        """Whether `read` can produce this metric at all, or only the planted calibration arm can."""
        return self.source == "record"

    def subject(self, science: str, **extra: Any) -> SubjectRef:
        return SubjectRef(
            dataset=self.dataset or None,
            frame=self.frame or None,
            interventions=(self.arm,) if self.arm else (),
            extra={"science": science, **extra},
        )

    def output(self, science: str, **extra: Any) -> Output:
        return Output(self.quantity, self.subject(science, **extra))


# ---------------------------------------------------------------------------
# One science, retyped
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScienceRetype:
    """Everything a science declares to the kernel, in one object the acceptance test can walk.

    `needs` is the access the record must carry for `read` to compute anything at all. It is checked
    before any arithmetic runs and it is the commonest refusal by a wide margin, because the
    sciences were written against organisms that could be probed and a record is a file.

    `headline` is the quantity the emitted Evidence carries. One Evidence carries one quantity, so
    a science that reports eight metrics under six quantity ids reports the six in the value payload
    under `quantities` and stamps the Evidence with the one its scoreboard row is about. That is
    auditable rather than asserted, and the rule it follows is: assert the emitted quantity, never
    the declared one.

    `destination` names where the science lands, in prose, because the mapping is otherwise a table
    and a table is not executable.
    """

    science: str
    spec: StudySpec
    metrics: tuple[MetricSpec, ...]
    headline: QuantityID
    destination: str
    needs: Mapping[Component, Access] = field(default_factory=dict)
    arc_requires: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    arc_costs: Mapping[str, CostModel] = field(default_factory=dict)
    waiting_on: str = ""

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for m in self.metrics:
            if m.metric in seen:
                raise ValueError(
                    f"{self.science}: metric {m.metric!r} is declared twice. One metric name inside "
                    f"one study is one quantity on one subject."
                )
            seen.add(m.metric)
        frozen = frozen_metrics(self.spec)
        missing = sorted(frozen - seen)
        if missing:
            raise ValueError(
                f"{self.science}: the frozen spec registers {missing} and the retype binds none of "
                f"them. A prediction whose metric is unbound cannot be adjudicated, which is the "
                f"failure plan closure exists to find."
            )

    # -- the three things the acceptance test asks for ------------------------

    def by_metric(self) -> Mapping[str, MetricSpec]:
        return {m.metric: m for m in self.metrics}

    def quantity_of(self, metric: str) -> QuantityID | None:
        spec = self.by_metric().get(metric)
        return spec.quantity if spec is not None and spec.bound else None

    def bindings(self, **extra: Any) -> tuple[MetricBinding, ...]:
        """One `MetricBinding` per bound metric. Gapped metrics are deliberately left out."""
        return tuple(
            MetricBinding(
                study=self.spec.id,
                metric=m.metric,
                quantity=m.quantity,
                subject=m.subject(self.science, **extra),
            )
            for m in self.metrics
            if m.bound
        )

    def arcs(self, **extra: Any) -> tuple[ArcSpec, ...]:
        """The arc graph, derived from the metric table rather than written twice.

        An arc produces every bound metric that names it and requires every output of the arcs named
        in `arc_requires`. Deriving `produces` from the table is the same argument `ArcSpec.subjects`
        makes: a second place to say what an arc produces is a second place to say it differently.
        """
        produces: dict[str, set[Output]] = {}
        for m in self.metrics:
            if m.bound:
                produces.setdefault(m.arc, set()).add(m.output(self.science, **extra))
        arcs: list[ArcSpec] = []
        for arc_id in sorted(produces):
            needs: set[Output] = set()
            for upstream in self.arc_requires.get(arc_id, ()):
                needs |= produces.get(upstream, set())
            arcs.append(
                ArcSpec(
                    id=arc_id,
                    produces=frozenset(produces[arc_id]),
                    requires=frozenset(needs - produces[arc_id]),
                    cost=self.arc_costs.get(arc_id, CostModel()),
                )
            )
        return tuple(arcs)

    def plan(self, budget: CostBudget = UNBUDGETED, **extra: Any) -> Plan:
        """The science as a plan: one study, its arcs, its bindings. `check_closure` takes this."""
        return Plan(
            studies=(self.spec,),
            arcs=self.arcs(**extra),
            bindings=self.bindings(**extra),
            budget=budget,
            name=f"{self.science} (retyped)",
        )

    # -- what is still open --------------------------------------------------

    def gaps(self) -> tuple[MetricSpec, ...]:
        """Metrics computing something the registry cannot name. These are the id requests."""
        return tuple(m for m in self.metrics if not m.bound)

    def unregistered(self) -> tuple[str, ...]:
        """Bound metrics whose quantity id is not in the registry. Should always be empty."""
        load_quantities()
        # `in` rather than `get(...) is None`: the registry's `get` raises on a missing key, which
        # is the right behaviour for a lookup and the wrong one for a membership test.
        return tuple(
            sorted(m.quantity for m in self.metrics if m.bound and m.quantity not in QUANTITIES)
        )

    # -- emitting ------------------------------------------------------------

    def subject_of(self, run: Any) -> SubjectRef:
        return SubjectRef(
            dataset=str(getattr(run, "id", "")) or None,
            extra={"science": self.science, "study": self.spec.id},
        )

    def evidence(
        self,
        run: Any,
        metrics: Mapping[str, float],
        *,
        summary: str,
        measured: Mapping[str, tuple[float, QuantityID]] | None = None,
        quantity: QuantityID = "",
        gauge: GaugeStatus = GaugeStatus.INVARIANT,
        baselines: Mapping[str, float] | None = None,
        refusals: Mapping[str, str] | None = None,
        **payload: Any,
    ) -> Any:
        """The reading a retyped science emits: every number it computed, each named by its quantity.

        `metrics` are frozen spec metrics the record could answer. `measured` is everything else the
        record supports, as `{name: (value, quantity_id)}`; a science pointed at a training record
        usually measures more than its spec froze, because the spec was written against a planted
        organism. Both go through the same rule: **the id has to be in the registry.** A number with
        no registered quantity is not reported at all, it is refused by name in `refusals`.

        `quantities` in the payload is the per-number resolution, carried rather than implied, so a
        reader can check what `chi_bon_spearman` was taken to be instead of trusting that somebody
        got it right. `refusals` names what the record could not support, so a partial reading says
        which part is missing rather than being silently short.

        `quantity` overrides the science's headline, and a caller should use it whenever the record
        did not support the headline measurement. Stamping a reading with the quantity the science
        wishes it had measured is the same failure as an unwired field that nothing emits: the
        declaration is right and the emitted value is wrong, and only the emitted one is what a
        downstream unit check sees.
        """
        table = self.by_metric()
        unknown = sorted(set(metrics) - set(table))
        if unknown:
            raise ValueError(
                f"{self.science}: read reported {unknown}, which the retype does not bind. Every "
                f"number a science reports has to name the quantity it is, or the unit machinery "
                f"has nothing to key on."
            )
        planted = sorted(m for m in metrics if not table[m].from_record)
        if planted:
            raise ValueError(
                f"{self.science}: read reported {planted}, which are declared source='organism'. "
                f"Those come from the planted calibration arm and a record contains no plant, so "
                f"reporting one here would stamp a record reading with a number that did not come "
                f"from the record."
            )
        load_quantities()
        resolved: dict[str, str] = {m: str(self.quantity_of(m)) for m in metrics}
        values: dict[str, float] = {m: float(v) for m, v in metrics.items()}
        for name, (value, measured_quantity) in (measured or {}).items():
            # Named `measured_quantity` rather than `quantity`: the loop variable shadowed the
            # parameter here, so every reading was stamped with whichever id the last measured
            # entry happened to carry. The declaration was right and the emitted value was wrong,
            # which is exactly why the check below asserts the emitted id rather than the declared.
            if measured_quantity not in QUANTITIES:
                raise ValueError(
                    f"{self.science}: read reported {name!r} as {measured_quantity!r}, which is not "
                    f"in the quantity registry. Registering an id is the maintainer's decision; "
                    f"until it is taken, refuse the number rather than shipping it under a name "
                    f"nobody agreed to."
                )
            values[name] = float(value)
            resolved[name] = str(measured_quantity)
        stamped = quantity or self.headline
        if str(stamped) not in set(resolved.values()):
            raise ValueError(
                f"{self.science}: the reading would be stamped {stamped!r} and reports none of "
                f"{sorted(set(resolved.values()))}. Stamp the quantity the record actually "
                f"supported, or refuse."
            )
        return make_evidence(
            observable=f"{self.science}.read",
            observable_version=RETYPE_VERSION,
            subject=self.subject_of(run),
            value={
                "metrics": values,
                "quantities": resolved,
                "refusals": dict(refusals or {}),
                "summary": summary,
                **payload,
            },
            gauge=gauge,
            quantity=stamped,
            baselines=dict(baselines) if baselines else None,
            provenance=Provenance(study=self.spec.id, extra={"science": self.science}),
        )

    # -- the standing refusals ----------------------------------------------

    def access_refusal(self, run: Any, *, remedy: str) -> Refusal | None:
        """`ACCESS_INSUFFICIENT` when the record cannot reach what the science probes.

        Returns None when the access is there, so the caller reads `if (r := ...) is not None`.
        """
        available: AccessMatrix = getattr(run, "access", {}) or {}
        gap = missing_access(available, dict(self.needs))
        if not gap:
            return None
        have = ", ".join(f"{c.name}: {a.name}" for c, a in sorted(available.items(), key=_cname))
        return refuse_access(
            f"{self.science}.read",
            needs={c.name: a.name for c, a in gap.items()},
            have=have or "nothing declared",
            remedy=remedy,
        )

    def incomplete(self, *, field: str, subject: str, remedy: str, **statistics: Any) -> Refusal:
        """`RECORD_INCOMPLETE`: the access is there and the field was never written.

        E30's test is whether the remedy is answerable where the reader is standing. It is not, for
        any of these: the fix is upstream in whatever produced the record.
        """
        from reward_lens.core.reading import refuse_incomplete

        return refuse_incomplete(
            f"{self.science}.read",
            field=field,
            subject=subject,
            remedy=remedy,
            **statistics,
        )

    def undefined(self, *, detail: str, remedy: str, **statistics: Any) -> Refusal:
        """The quantity does not apply to this object at all.

        This wants `QUANTITY_UNDEFINED`, which is agreed but not yet a member of `RefusalReason`.
        The count assertion is pinned at sixteen, so it carries `ACCESS_INSUFFICIENT` with an honest
        paragraph until that member lands, and the detail says plainly that no access fixes it.
        """
        return Refusal(
            instrument=f"{self.science}.read",
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=(
                f"{detail} This is not an access problem and no rung of access changes it: the "
                f"quantity is not defined on this object. Carried as ACCESS_INSUFFICIENT because "
                f"RefusalReason has no member for 'answerable nowhere' yet."
            ),
            remedy=remedy,
            statistics=dict(statistics),
        )


def _cname(pair: tuple[Component, Access]) -> str:
    return pair[0].name


# ---------------------------------------------------------------------------
# Reading the frozen spec
# ---------------------------------------------------------------------------


def frozen_metrics(spec: StudySpec) -> set[str]:
    """Every metric name the spec registers, hypotheses and kill criteria alike.

    Kill criteria are collected on the same footing for the reason `demands_of` gives: a kill whose
    metric cannot be produced is the more dangerous of the two, because it looks exactly like a kill
    that was evaluated and did not fire.
    """
    return {h.prediction.metric for h in spec.hypotheses} | {k.metric for k in spec.kill_criteria}


# ---------------------------------------------------------------------------
# The population
# ---------------------------------------------------------------------------

#: The directory this package lives in. The registry enumerates it rather than listing names, so a
#: science added later is caught by the acceptance test instead of being silently exempt.
STUDIES_DIR = Path(__file__).resolve().parent


def science_modules() -> tuple[str, ...]:
    """Every module under `studies/` that could hold a retype, found by walking the directory."""
    found: list[str] = []
    for entry in sorted(STUDIES_DIR.iterdir()):
        if not entry.is_dir() or entry.name.startswith((".", "_")) or entry.name == "__pycache__":
            continue
        for mod in sorted(pkgutil.iter_modules([str(entry)])):
            if mod.name.startswith("_"):
                continue
            found.append(f"studies.{entry.name}.{mod.name}")
    return tuple(found)


def retypes() -> Mapping[str, ScienceRetype]:
    """Every `RETYPE` declared anywhere under `studies/`, keyed by its science id.

    Enumerated from the filesystem, not from a list. A science that ships a spec and no retype is
    absent here and the acceptance test names it, which is the whole reason this walks.
    """
    out: dict[str, ScienceRetype] = {}
    for path in science_modules():
        module = importlib.import_module(path)
        retype = getattr(module, "RETYPE", None)
        if isinstance(retype, ScienceRetype):
            out[retype.science] = retype
    return dict(sorted(out.items()))


def readers() -> Mapping[str, Callable[[Any], Reading]]:
    """The `read` function beside every declared `RETYPE`."""
    out: dict[str, Callable[[Any], Reading]] = {}
    for path in science_modules():
        module = importlib.import_module(path)
        retype = getattr(module, "RETYPE", None)
        reader = getattr(module, "read", None)
        if isinstance(retype, ScienceRetype) and callable(reader):
            out[retype.science] = reader
    return dict(sorted(out.items()))


def specs() -> Mapping[str, StudySpec]:
    """Every `build_spec()` under `studies/`, retyped or not. The denominator."""
    out: dict[str, StudySpec] = {}
    for path in science_modules():
        module = importlib.import_module(path)
        builder = getattr(module, "build_spec", None)
        if callable(builder):
            out[path.split(".")[1] if path.count(".") == 2 else path] = builder()
    return out


def unregistered_metrics() -> Mapping[str, tuple[str, ...]]:
    """Per science, the metrics that compute something the registry has no id for.

    This is the maintainer's work list, and it is deliberately data rather than prose: each request
    is adjudicated on its own. Of the six raised so far, four were registered, one was split, and
    one was refused because its value moved 24-fold on an arbitrary tie-break.
    """
    return {name: tuple(m.metric for m in r.gaps()) for name, r in retypes().items() if r.gaps()}


# ---------------------------------------------------------------------------
# Small numeric helpers the sciences share when they read a record
# ---------------------------------------------------------------------------


def leaf_scores(run: Any, *, limit: int | None = None) -> dict[str, list[float]]:
    """Every scalar leaf of every trajectory's score tree, keyed by leaf name.

    Walks the record once. `limit` caps the number of trajectories read, because a 200-step run with
    four rollouts per group is 800 trees and a science that only wants a tail index does not need
    all of them.
    """
    out: dict[str, list[float]] = {}
    seen = 0
    for step in run.steps:
        for group in step.groups:
            for traj in group.trajectories:
                tree = traj.scores
                if tree is None:
                    continue
                for name, value in _walk_leaves(tree):
                    out.setdefault(name, []).append(value)
                seen += 1
                if limit is not None and seen >= limit:
                    return out
    return out


def _walk_leaves(node: Any) -> Iterable[tuple[str, float]]:
    name = getattr(node, "name", None)
    value = getattr(node, "value", None)
    children = getattr(node, "children", None)
    if children:
        for child in children:
            yield from _walk_leaves(child)
    elif name is not None and value is not None:
        try:
            yield str(name), float(value)
        except (TypeError, ValueError):
            return


def step_series(run: Any, key: str) -> tuple[list[int], list[float]]:
    """A per-step scalar series off the optimizer telemetry, with the step indices beside it.

    `key` is either a named `OptimizerTelemetry` field or a key of its `extra` dict, which is where
    a framework's own logging lands. Steps that do not carry it are skipped rather than filled,
    because a gap in a series and a zero in a series are different facts.
    """
    idx: list[int] = []
    vals: list[float] = []
    for step in run.steps:
        telem = step.optimizer
        raw = getattr(telem, key, None)
        if raw is None:
            raw = (telem.extra or {}).get(key)
        if raw is None:
            continue
        idx.append(int(step.index))
        vals.append(float(raw))
    return idx, vals


def trajectory_features(run: Any) -> tuple[list[str], list[list[float]], list[float]]:
    """The feature bank, the per-trajectory feature rows, and the realised reward beside them.

    Returns `(names, rows, rewards)` with `rows[i][j]` the value of feature `names[j]` on trajectory
    `i`. Trajectories missing any feature in the bank are dropped rather than imputed: an imputed
    feature enters a covariance as a constant and pulls every susceptibility toward zero.
    """
    names: list[str] = []
    for step in run.steps:
        for group in step.groups:
            for traj in group.trajectories:
                for key in traj.features or {}:
                    if key not in names:
                        names.append(str(key))
        if names:
            break
    rows: list[list[float]] = []
    rewards: list[float] = []
    for step in run.steps:
        for group in step.groups:
            for traj in group.trajectories:
                feats = traj.features or {}
                if any(n not in feats for n in names):
                    continue
                total = _tree_total(traj.scores)
                if total is None:
                    continue
                rows.append([float(feats[n]) for n in names])
                rewards.append(total)
    return names, rows, rewards


def _tree_total(tree: Any) -> float | None:
    if tree is None:
        return None
    leaves = list(_walk_leaves(tree))
    if not leaves:
        return None
    return float(sum(v for _, v in leaves))


def advantages(run: Any) -> list[float]:
    """Every recorded advantage in the run, in step order."""
    out: list[float] = []
    for step in run.steps:
        for group in step.groups:
            for traj in group.trajectories:
                if traj.advantage is not None:
                    out.append(float(traj.advantage))
    return out


def group_rankings(run: Any) -> list[tuple[int, ...]]:
    """The recorded within-group rankings, for the sciences that need a tournament.

    `GroupStats.ranks` is the record's own ordering of the group's rollouts. A group that recorded
    none is skipped; a group of one has no comparison in it and is skipped too.
    """
    out: list[tuple[int, ...]] = []
    for step in run.steps:
        for group in step.groups:
            ranks = group.group_stats.ranks
            if ranks and len(ranks) > 1:
                out.append(tuple(int(r) for r in ranks))
    return out


def count_trajectories(run: Any) -> int:
    return sum(len(g.trajectories) for s in run.steps for g in s.groups)


def declared_conditions(run: Any) -> Mapping[str, bool]:
    """The regime the recorder declared, keyed by condition name rather than by enum member."""
    declared = getattr(getattr(run, "regime", None), "declared", {}) or {}
    return {c.name: bool(v) for c, v in declared.items()}


__all__ = [
    "RETYPE_VERSION",
    "STUDIES_DIR",
    "MetricSpec",
    "ScienceRetype",
    "advantages",
    "count_trajectories",
    "declared_conditions",
    "frozen_metrics",
    "group_rankings",
    "leaf_scores",
    "readers",
    "retypes",
    "science_modules",
    "specs",
    "step_series",
    "trajectory_features",
    "unregistered_metrics",
]

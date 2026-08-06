"""The plan: studies, the arcs that answer them, and the gate between them.

A `Plan` is the whole of what a run intends to do. It holds the frozen studies, whose predictions
are already uneditable; the arcs, which are the units of work and the only things that produce
anything; a binding from each study's metric names to the quantities and subjects they stand for;
and the budget. `check_closure` takes that and either returns a report or refuses to let the run
start.

The order matters and it is the point. Freeze first, so the predictions predate the data. Bind
second, so every metric name means something checkable. Close third, so a prediction nothing can
answer is found now rather than after the bill. Only then does anything run.

The campaign this replaces did the first step and none of the others. Its metric names were
strings that its analysis functions happened to emit, its arcs declared nothing, and the check
that a prediction was answerable happened at adjudication time, by way of `result.metrics.get`
returning `None`. Eight of twenty-seven cards found out that way.

`check_closure` raises. The argument for that, against a library whose first rule is that a
refusal is a value, is in `reward_lens.core.closure`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from reward_lens.core.closure import (
    UNBUDGETED,
    ArcSpec,
    ClosureError,
    ClosureReport,
    CostBudget,
    Demand,
    MetricBinding,
    Output,
    analyse,
)
from reward_lens.studies.freeze import FrozenStudy
from reward_lens.studies.spec import StudySpec

#: A plan may hold specs that have not been frozen yet, so closure can be checked while a study is
#: still being written. Nothing runs off an unfrozen spec; the freeze is still what makes evidence
#: REGISTERED.
StudyLike = FrozenStudy | StudySpec


def _spec_of(study: StudyLike) -> StudySpec:
    return study.spec if isinstance(study, FrozenStudy) else study


@dataclass(frozen=True)
class Plan:
    """Everything a run intends to do, before any of it happens.

    `bindings` is the field the plan cannot work without. A `Prediction.metric` is a string like
    `spearman_biasbattery_vs_rmbench_hard`; it is not a quantity id and it names no subject. Until
    something says which registered quantity, measured on what, that string refers to, there is no
    question for closure to answer. Leaving a metric unbound is itself a closure gap rather than a
    silent pass, because "nobody said what this metric is" and "nobody produces it" are the same
    problem seen at different depths.
    """

    studies: tuple[StudyLike, ...] = ()
    arcs: tuple[ArcSpec, ...] = ()
    bindings: tuple[MetricBinding, ...] = ()
    budget: CostBudget = UNBUDGETED
    name: str = ""

    @property
    def binding_index(self) -> Mapping[tuple[str, str], Output]:
        """`(spec id, metric) -> Output`, refusing two different meanings for one metric."""
        index: dict[tuple[str, str], Output] = {}
        for binding in self.bindings:
            key = (binding.study, binding.metric)
            existing = index.get(key)
            if existing is not None and existing != binding.output:
                raise ValueError(
                    f"study {binding.study!r} binds the metric {binding.metric!r} to both "
                    f"{existing} and {binding.output}. One metric name inside one study is one "
                    f"quantity on one subject, or the adjudication is reading whichever landed "
                    f"first."
                )
            index[key] = binding.output
        return index

    @property
    def arcs_by_id(self) -> Mapping[str, ArcSpec]:
        return {arc.id: arc for arc in self.arcs}


def demands_of(plan: Plan) -> tuple[Demand, ...]:
    """Every registered prediction and kill criterion in the plan, resolved against the bindings.

    Kill criteria are collected on exactly the same footing as hypotheses. That is not symmetry for
    its own sake: a kill criterion whose metric cannot be produced is the more dangerous of the
    two, because under the adjudication this project replaced it was indistinguishable from a
    criterion that was evaluated and did not fire.
    """
    index = plan.binding_index
    demands: list[Demand] = []
    for study in plan.studies:
        spec = _spec_of(study)
        for hypothesis in spec.hypotheses:
            prediction = hypothesis.prediction
            demands.append(
                Demand(
                    study=spec.id,
                    owner=hypothesis.id,
                    kind="hypothesis",
                    metric=prediction.metric,
                    statement=(
                        f"{prediction.metric} {prediction.comparator} {prediction.threshold:g}"
                    ),
                    output=index.get((spec.id, prediction.metric)),
                )
            )
        for kill in spec.kill_criteria:
            demands.append(
                Demand(
                    study=spec.id,
                    owner=kill.id,
                    kind="kill",
                    metric=kill.metric,
                    statement=f"{kill.metric} {kill.comparator} {kill.threshold:g}",
                    output=index.get((spec.id, kill.metric)),
                )
            )
    return tuple(demands)


def closure_report(plan: Plan) -> ClosureReport:
    """The static check as data. Never raises on a plan that does not close.

    Use this when the caller wants to show someone what is missing. Use `check_closure` when the
    caller is about to spend money.
    """
    return analyse(plan.arcs, demands_of(plan), plan.budget)


def check_closure(plan: Plan) -> ClosureReport:
    """Refuse to let a run start unless every registered prediction is reachable.

    For every hypothesis and every kill criterion in every study: resolve the metric to a
    (QuantityID, SubjectRef) pair, find an arc that produces it on that subject, and verify that
    arc's own requirements are transitively satisfied. Any unreachable metric raises
    `ClosureError` naming the prediction, the metric and the gap. Cycles in the arc graph and a
    cost that exceeds the declared budget raise the same way.

    Returns the report on success, and the report is worth keeping: `report.metric_arcs` is the
    per-metric arc attribution the runner records, so that a metric which does go missing at run
    time produces a void naming the arc rather than only the metric.
    """
    report = closure_report(plan)
    if not report.closed:
        raise ClosureError(report)
    return report


def metric_arcs_for(plan: Plan, study: StudyLike) -> Mapping[str, str]:
    """The `metric_arcs` mapping for one study, ready to hand to `run_study`.

    Scoped per study because two studies may use the same metric name for different quantities;
    the plan-wide mapping in `ClosureReport.metric_arcs` is keyed on the metric alone and would
    collapse them.
    """
    spec = _spec_of(study)
    index = plan.binding_index
    producers: dict[Output, list[str]] = {}
    for arc in plan.arcs:
        for out in arc.produces:
            producers.setdefault(out, []).append(arc.id)

    out_map: dict[str, str] = {}
    metrics = [h.prediction.metric for h in spec.hypotheses] + [
        k.metric for k in spec.kill_criteria
    ]
    for metric in metrics:
        output = index.get((spec.id, metric))
        if output is None:
            continue
        arc_ids = sorted(producers.get(output, []))
        if arc_ids:
            out_map[metric] = arc_ids[0]
    return out_map


def bind(study: StudyLike, quantity: str, **subject_extra: object) -> tuple[MetricBinding, ...]:
    """Bind every metric a study registers to one quantity on one subject.

    The common shape for a study whose analysis is a single arc: one card, one result row, every
    metric on it. Written as a helper because a study with eight hypotheses would otherwise need
    eight near-identical bindings, and eight chances to mistype one.
    """
    from reward_lens.core.types import SubjectRef

    spec = _spec_of(study)
    metrics: list[str] = []
    for hypothesis in spec.hypotheses:
        metrics.append(hypothesis.prediction.metric)
    for kill in spec.kill_criteria:
        metrics.append(kill.metric)
    seen: dict[str, None] = {}
    for metric in metrics:
        seen.setdefault(metric)
    return tuple(
        MetricBinding(
            study=spec.id,
            metric=metric,
            quantity=quantity,
            subject=SubjectRef(extra={**subject_extra, "metric": metric}),
        )
        for metric in seen
    )


@dataclass(frozen=True)
class PlanBuilder:
    """A small mutable-free accumulator, so a plan can be assembled in pieces.

    Plans are built by several modules at once in practice (one per science), and a frozen
    dataclass is awkward to grow. This keeps the plan immutable while making the assembly
    readable.
    """

    studies: tuple[StudyLike, ...] = ()
    arcs: tuple[ArcSpec, ...] = ()
    bindings: tuple[MetricBinding, ...] = ()
    budget: CostBudget = UNBUDGETED
    name: str = ""

    def with_studies(self, *studies: StudyLike) -> "PlanBuilder":
        return PlanBuilder(
            self.studies + tuple(studies), self.arcs, self.bindings, self.budget, self.name
        )

    def with_arcs(self, *arcs: ArcSpec) -> "PlanBuilder":
        return PlanBuilder(
            self.studies, self.arcs + tuple(arcs), self.bindings, self.budget, self.name
        )

    def with_bindings(self, *bindings: MetricBinding) -> "PlanBuilder":
        return PlanBuilder(
            self.studies, self.arcs, self.bindings + tuple(bindings), self.budget, self.name
        )

    def with_budget(self, budget: CostBudget) -> "PlanBuilder":
        return PlanBuilder(self.studies, self.arcs, self.bindings, budget, self.name)

    def build(self) -> Plan:
        return Plan(
            studies=self.studies,
            arcs=self.arcs,
            bindings=self.bindings,
            budget=self.budget,
            name=self.name,
        )


def plan_of(
    studies: Sequence[StudyLike],
    arcs: Sequence[ArcSpec],
    bindings: Sequence[MetricBinding] = (),
    budget: CostBudget = UNBUDGETED,
    name: str = "",
) -> Plan:
    """The one-call constructor, for callers that have everything at hand."""
    return Plan(
        studies=tuple(studies),
        arcs=tuple(arcs),
        bindings=tuple(bindings),
        budget=budget,
        name=name,
    )


__all__ = [
    "Plan",
    "PlanBuilder",
    "StudyLike",
    "bind",
    "check_closure",
    "closure_report",
    "demands_of",
    "metric_arcs_for",
    "plan_of",
]

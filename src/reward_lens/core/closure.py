"""Plan closure: prove every registered prediction is reachable before any work starts.

A frozen prediction names a metric. Something has to compute that metric. If nothing in the plan
does, the prediction cannot be adjudicated, and the campaign that produced this library's evidence
base found that out one card at a time, after the GPU bill. Eight of its twenty-seven cards came
back "inconclusive" and in every case the cause was infrastructural: an arc had not run, or its
shard had not merged, or a resolver hit a ``PermissionError``. The check in this module is the
static version of that discovery, and it costs nothing.

The shape is a graph reachability question. Arcs are the units of work. Each one declares what it
produces, as exact quantity ids on named subjects, and what it requires from other arcs. Each
registered prediction and each kill criterion resolves to one quantity on one subject. A plan
closes when every one of those is produced by some arc whose own inputs are transitively produced
too, when the arc graph has no cycle, and when the declared budget covers the arcs the demands
actually reach.

**Why this raises where the rest of the library returns a value.** The rule everywhere else is that
a refusal is a value: an instrument that cannot measure returns a `Refusal` carrying the numbers and a
remedy, never an exception, because a caller who gets a refusal still has a run to look at and a
next step to take. Plan closure is the one place where that reasoning inverts. There is no
measurement to hand back and no partial answer to bound, because nothing has run yet. What the
caller has is a plan with a hole in it, and the only useful behaviour is to stop before spending
anything. A return value can be ignored by a caller who does not check it, and the failure this
module exists to prevent is precisely a check that nobody noticed had failed to happen. So
`check_closure` raises. `RefusalReason.PLAN_NOT_CLOSED` still exists and is still the right thing
to record downstream, in a capability report or a card, where the plan is data being described
rather than work about to start.

For the same reason there are two entry points. `analyse` computes the report and never raises, so
a tool that wants to show a user what is missing can. `check_closure` in `reward_lens.studies.plan`
raises on that report, and that is the one a runner calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Literal, Mapping, Sequence

from reward_lens.core.errors import RewardLensError
from reward_lens.core.quantity import FREE, CostModel, QuantityID
from reward_lens.core.types import SubjectRef, content_hash

# ---------------------------------------------------------------------------
# Subjects, made comparable
# ---------------------------------------------------------------------------


def subject_key(subject: SubjectRef) -> str:
    """A stable, hashable identity for a subject.

    `SubjectRef` carries an `extra` dict, so it is a frozen dataclass that cannot actually be
    hashed, and the campaign puts its entire join key (`roster_key`, `slice`) in exactly that
    field. Hashing the canonical form is what lets a subject be a set member without either
    dropping the extras or inventing a second, poorer notion of subject identity.
    """
    return content_hash(subject.__canonical__(), "subj")


def render_subject(subject: SubjectRef) -> str:
    """A subject in one line, written for whoever is reading the closure failure.

    Extras are printed because on real plans that is where the join key lives: "roster_key=armorm
    slice=diagnostic-v3-degradation" is the sentence that tells someone which arc to go and run.
    """
    bits: list[str] = []
    if subject.signals:
        bits.append("signals=" + ",".join(str(s) for s in subject.signals))
    if subject.dataset:
        bits.append(f"dataset={subject.dataset}")
    if subject.readout:
        bits.append(f"readout={subject.readout}")
    if subject.frame:
        bits.append(f"frame={subject.frame}")
    if subject.interventions:
        bits.append("interventions=" + ",".join(subject.interventions))
    for k, v in sorted((subject.extra or {}).items()):
        bits.append(f"{k}={v}")
    return " ".join(bits) if bits else "(unnamed subject)"


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n} {singular}" if n == 1 else f"{n} {plural or singular + 's'}"


@dataclass(frozen=True, eq=False)
class Output:
    """One thing an arc makes: a quantity, on one subject.

    Equality and hashing run on the quantity plus the canonical subject, so two `Output`s built
    from separately-constructed but identical `SubjectRef`s are the same output. That is the whole
    point: the producer and the consumer are written in different modules by different people, and
    they have to be able to name the same thing without sharing an object.

    The obvious alternative is `produces: frozenset[QuantityID]` beside a separate `subjects`
    tuple. Pairing them here rather than crossing them is deliberate: an arc that
    declares three quantities and four subjects does not produce twelve things, and reading it as
    if it did is exactly the permissive resolver that would let a plan pass closure and then fail
    at run time.
    """

    quantity: QuantityID
    subject: SubjectRef = field(default_factory=SubjectRef)

    def __post_init__(self) -> None:
        if not self.quantity or " " in self.quantity:
            raise ValueError(
                f"an output's quantity must be a token with no spaces; got {self.quantity!r}"
            )
        object.__setattr__(self, "_key", (self.quantity, subject_key(self.subject)))

    @property
    def key(self) -> tuple[str, str]:
        return self._key  # type: ignore[attr-defined,no-any-return]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Output) and other.key == self.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __str__(self) -> str:
        return f"{self.quantity} on {render_subject(self.subject)}"


def on(quantity: QuantityID, **extra: Any) -> Output:
    """An output on a subject named entirely by its extras.

    Most real plans join on a couple of string keys (a roster key and a slice, a run id and a
    step range) rather than on a fingerprint, and writing `SubjectRef(extra={...})` out at every
    site makes a plan unreadable. `on("campaign.scores", roster_key="armorm", slice="rb2-full")`
    is the same object.
    """
    return Output(quantity, SubjectRef(extra=dict(extra)))


# ---------------------------------------------------------------------------
# The units of work, and the budget they run against
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArcSpec:
    """One unit of work, declaring what it makes and what it needs.

    `produces` and `requires` are sets of exact `Output`s. `cost` is what running it costs, and it
    is what makes the budget half of closure computable: the cost of a plan is the sum over the
    arcs its demands actually reach, which is not the same as the sum over every arc in the file.
    """

    id: str
    produces: frozenset[Output] = frozenset()
    requires: frozenset[Output] = frozenset()
    cost: CostModel = FREE

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("an arc needs an id; the id is what a closure failure names")
        overlap = self.produces & self.requires
        if overlap:
            raise ValueError(
                f"arc {self.id!r} both produces and requires {sorted(str(o) for o in overlap)}. "
                f"An arc that is its own input is a cycle of length one and it will never run."
            )

    @property
    def subjects(self) -> tuple[SubjectRef, ...]:
        """Every subject this arc touches, on either side.

        `subjects` could be a field. It is derived here instead, because a field that repeats
        what `produces` already says is a field that can disagree with it, and the disagreement
        would be silent.
        """
        seen: dict[str, SubjectRef] = {}
        for out in sorted(self.produces | self.requires, key=lambda o: o.key):
            seen.setdefault(subject_key(out.subject), out.subject)
        return tuple(seen.values())


@dataclass(frozen=True)
class CostBudget:
    """A ceiling on what a plan may spend, per currency.

    A currency left as `None` is not budgeted rather than budgeted at zero. The distinction
    matters: most plans have a dollar ceiling and no opinion about CPU seconds, and treating the
    absent opinion as a limit of zero would refuse every plan.
    """

    calls: int | None = None
    gpu_seconds: float | None = None
    cpu_seconds: float | None = None
    dollars: float | None = None
    note: str = ""

    def overruns(self, cost: CostModel) -> dict[str, tuple[float, float]]:
        """Each budgeted currency the cost exceeds, as ``{currency: (spend, ceiling)}``."""
        pairs = (
            ("calls", cost.calls, self.calls),
            ("gpu_seconds", cost.gpu_seconds, self.gpu_seconds),
            ("cpu_seconds", cost.cpu_seconds, self.cpu_seconds),
            ("dollars", cost.dollars, self.dollars),
        )
        return {
            name: (float(spend), float(ceiling))
            for name, spend, ceiling in pairs
            if ceiling is not None and spend is not None and float(spend) > float(ceiling)
        }

    def render(self) -> str:
        bits = []
        if self.calls is not None:
            bits.append(f"{self.calls:,} calls")
        if self.gpu_seconds is not None:
            bits.append(f"{self.gpu_seconds:.0f}s GPU")
        if self.cpu_seconds is not None:
            bits.append(f"{self.cpu_seconds / 60:.0f} min CPU")
        if self.dollars is not None:
            bits.append(f"${self.dollars:,.2f}")
        return ", ".join(bits) if bits else "unbudgeted"


UNBUDGETED = CostBudget()


def total_cost(arcs: Iterable[ArcSpec]) -> CostModel:
    """Add up the arcs' costs, keeping a currency `None` when nobody declared it.

    Summing `None` as zero everywhere would turn "nobody costed this in dollars" into "$0.00",
    which reads like a measurement and is not one.
    """
    totals: dict[str, float | None] = {
        "calls": None,
        "gpu_seconds": None,
        "cpu_seconds": None,
        "dollars": None,
    }
    for arc in arcs:
        for name in totals:
            value = getattr(arc.cost, name)
            if value is None:
                continue
            totals[name] = (totals[name] or 0) + value
    calls = totals["calls"]
    return CostModel(
        calls=int(calls) if calls is not None else None,
        gpu_seconds=totals["gpu_seconds"],
        cpu_seconds=totals["cpu_seconds"],
        dollars=totals["dollars"],
    )


# ---------------------------------------------------------------------------
# What the plan has to answer for
# ---------------------------------------------------------------------------

DemandKind = Literal["hypothesis", "kill"]


@dataclass(frozen=True)
class Demand:
    """One registered prediction's claim on the plan.

    A hypothesis and a kill criterion make the same demand and are kept apart because the
    consequences of failing to answer them differ. An unanswered hypothesis is a study that learns
    nothing. An unanswered kill criterion is a safety check that did not run and, under the
    adjudication this project replaced, looked exactly like one that ran and passed.
    """

    study: str
    owner: str
    kind: DemandKind
    metric: str
    statement: str
    output: Output | None = None

    def describe(self) -> str:
        noun = "hypothesis" if self.kind == "hypothesis" else "kill criterion"
        return f"study {self.study}, {noun} {self.owner}"


@dataclass(frozen=True)
class MetricBinding:
    """What a study's metric name means, as a quantity on a subject.

    A `Prediction.metric` is a bare string chosen by whoever wrote the study, like
    `spearman_biasbattery_vs_rmbench_hard`. It is not a quantity id and it names no subject, so
    something has to say which registered quantity, measured on what, that string refers to.
    That is this. Binding is a plan-time act and it is the step the campaign never performed,
    which is why nothing could check its plan.

    `study` is the spec id (`StudySpec.id`), not the frozen `StudyID`. Bindings are written
    alongside the specs and before the freeze, so keying them on a hash that does not exist yet
    would be circular.
    """

    study: str
    metric: str
    quantity: QuantityID
    subject: SubjectRef = field(default_factory=SubjectRef)

    @property
    def output(self) -> Output:
        return Output(self.quantity, self.subject)


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


class GapKind(Enum):
    """The five ways a plan fails to close. Each one has a different fix."""

    #: The plan never said what quantity, on what subject, this metric is. Nothing to look for.
    UNBOUND_METRIC = "unbound_metric"
    #: The metric resolves, and no arc in the plan produces it.
    NO_PRODUCER = "no_producer"
    #: An arc produces it, and that arc needs an input nothing produces.
    UNSATISFIED_INPUT = "unsatisfied_input"
    #: The arcs that would produce it depend on each other in a loop, so none of them can start.
    CYCLE = "cycle"
    #: The plan closes and costs more than the declared budget allows.
    OVER_BUDGET = "over_budget"


@dataclass(frozen=True)
class Gap:
    """One reason the plan does not close, written as a work item.

    `chain` is the arcs walked from the demand's producer down to the missing thing, so a failure
    three arcs deep names the path rather than only the leaf. `remedy` is an instruction; a gap
    that says "unsatisfied input" and stops has told the reader nothing they can act on.
    """

    kind: GapKind
    detail: str
    remedy: str
    demand: Demand | None = None
    missing: Output | None = None
    arc: str | None = None
    chain: tuple[str, ...] = ()

    def render(self) -> str:
        lines: list[str] = []
        if self.demand is not None:
            lines.append(f"  {self.demand.describe()}")
            lines.append(f"      prediction  {self.demand.statement}")
            lines.append(f"      metric      {self.demand.metric}")
        else:
            lines.append(f"  {self.kind.name}")
        if self.arc:
            lines.append(f"      arc         {self.arc}")
        if len(self.chain) > 1:
            lines.append(f"      through     {' -> '.join(self.chain)}")
        lines.append(f"      gap         {self.detail}")
        lines.append(f"      do          {self.remedy}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


@dataclass(frozen=True)
class ClosureReport:
    """What the static check found, whether or not it found anything wrong.

    `metric_arcs` is the field the runner wants. A run should record, per
    prediction, which arc was supposed to produce its metric, so that a metric which goes missing
    at run time names the arc rather than only itself. That mapping is a by-product of closing the
    plan, and `run_study(..., metric_arcs=report.metric_arcs)` is what carries it through.
    """

    gaps: tuple[Gap, ...] = ()
    order: tuple[str, ...] = ()
    required_arcs: frozenset[str] = frozenset()
    unused_arcs: tuple[str, ...] = ()
    metric_arcs: Mapping[str, str] = field(default_factory=dict)
    cost: CostModel = FREE
    budget: CostBudget = UNBUDGETED
    demands: tuple[Demand, ...] = ()
    ambiguous: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def closed(self) -> bool:
        return not self.gaps

    @property
    def blocked_studies(self) -> tuple[str, ...]:
        seen = {g.demand.study for g in self.gaps if g.demand is not None}
        return tuple(sorted(seen))

    def gaps_of(self, kind: GapKind) -> tuple[Gap, ...]:
        return tuple(g for g in self.gaps if g.kind is kind)

    def render(self) -> str:
        if self.closed:
            return (
                f"plan closes: {len(self.demands)} registered predictions, "
                f"{len(self.required_arcs)} arcs, {self.cost.render()} against "
                f"{self.budget.render()}"
            )
        blocked = tuple(g for g in self.gaps if g.demand is not None)
        parts: list[str] = []
        if blocked:
            parts.append(
                f"{_plural(len(blocked), 'registered prediction')} in "
                f"{_plural(len(self.blocked_studies), 'study', 'studies')} "
                f"name a metric this plan cannot produce."
            )
        other = len(self.gaps) - len(blocked)
        if other:
            parts.append(f"{_plural(other, 'further problem')} with the plan itself.")
        return "\n\n".join([" ".join(parts), *(g.render() for g in self.gaps)])

    def __str__(self) -> str:
        return self.render()


class ClosureError(RewardLensError):
    """Raised at plan time. Names the prediction, the metric, and the gap.

    An exception rather than a `Refusal` because there is no measurement to return and nothing to
    bound: the run has not started, and the correct behaviour is that it does not. See the module
    docstring for the full argument. The report is attached so a caller that catches this can show
    every gap rather than only the first.
    """

    def __init__(self, report: ClosureReport) -> None:
        super().__init__(report.render())
        self.report = report


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def _producer_index(arcs: Sequence[ArcSpec]) -> dict[Output, tuple[str, ...]]:
    index: dict[Output, list[str]] = {}
    for arc in arcs:
        for out in arc.produces:
            index.setdefault(out, []).append(arc.id)
    return {out: tuple(sorted(ids)) for out, ids in index.items()}


def _cyclic_arcs(arcs: Sequence[ArcSpec], index: Mapping[Output, tuple[str, ...]]) -> set[str]:
    """The arcs that sit in a dependency loop, by Tarjan's strongly connected components.

    Doing this first is what makes the reachability walk below a plain memoised DFS over a DAG. A
    memo taken during a walk that can revisit its own stack is not sound, and the resulting bug
    (an arc reported satisfiable because the walk happened to reach it from a different branch
    first) is the kind that shows up once, on someone else's plan.
    """
    edges: dict[str, set[str]] = {arc.id: set() for arc in arcs}
    for arc in arcs:
        for need in arc.requires:
            edges[arc.id].update(index.get(need, ()))

    order: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    cyclic: set[str] = set()
    counter = 0

    def strongconnect(start: str) -> None:
        nonlocal counter
        # Iterative, because a deep plan would blow the recursion limit and a plan is user data.
        work: list[tuple[str, list[str]]] = [(start, sorted(edges[start]))]
        order[start] = low[start] = counter
        counter += 1
        stack.append(start)
        on_stack.add(start)
        while work:
            node, pending = work[-1]
            if pending:
                nxt = pending.pop()
                if nxt not in order:
                    order[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, sorted(edges[nxt])))
                elif nxt in on_stack:
                    low[node] = min(low[node], order[nxt])
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == order[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1 or node in edges[node]:
                    cyclic.update(component)

    for arc in arcs:
        if arc.id not in order:
            strongconnect(arc.id)
    return cyclic


@dataclass
class _Walk:
    """Whether an arc can actually run, and if not, the first reason it cannot."""

    ok: bool
    chain: tuple[str, ...] = ()
    missing: Output | None = None
    kind: GapKind = GapKind.UNSATISFIED_INPUT


def _satisfiable(
    arc_id: str,
    by_id: Mapping[str, ArcSpec],
    index: Mapping[Output, tuple[str, ...]],
    cyclic: set[str],
    memo: dict[str, _Walk],
) -> _Walk:
    cached = memo.get(arc_id)
    if cached is not None:
        return cached
    if arc_id in cyclic:
        result = _Walk(False, (arc_id,), None, GapKind.CYCLE)
        memo[arc_id] = result
        return result
    memo[arc_id] = _Walk(True)  # provisional; the DAG guarantees we do not revisit this arc
    arc = by_id[arc_id]
    for need in sorted(arc.requires, key=lambda o: o.key):
        producers = index.get(need, ())
        if not producers:
            result = _Walk(False, (arc_id,), need, GapKind.NO_PRODUCER)
            memo[arc_id] = result
            return result
        attempts = [_satisfiable(p, by_id, index, cyclic, memo) for p in producers]
        if not any(attempt.ok for attempt in attempts):
            # Report the first producer's reason rather than the last one tried. Producers are
            # sorted, so this makes the message a function of the plan and not of iteration order.
            failed = attempts[0]
            result = _Walk(False, (arc_id, *failed.chain), failed.missing, failed.kind)
            memo[arc_id] = result
            return result
    memo[arc_id] = _Walk(True)
    return memo[arc_id]


def _reachable(
    roots: Iterable[str],
    by_id: Mapping[str, ArcSpec],
    index: Mapping[Output, tuple[str, ...]],
) -> set[str]:
    """Every arc some demand depends on, directly or through another arc.

    Iterative for the same reason the SCC pass is: a plan is user data and a deep dependency chain
    would otherwise be a recursion-limit crash rather than a report.
    """
    seen: set[str] = set()
    todo = [r for r in roots if r in by_id]
    while todo:
        arc_id = todo.pop()
        if arc_id in seen:
            continue
        seen.add(arc_id)
        for need in by_id[arc_id].requires:
            todo.extend(p for p in index.get(need, ()) if p in by_id and p not in seen)
    return seen


def _topological(
    arcs: Sequence[ArcSpec], index: Mapping[Output, tuple[str, ...]]
) -> tuple[str, ...]:
    """Arcs in an order that runs, producers first. Arcs in a cycle are left out."""
    by_id = {arc.id: arc for arc in arcs}
    depends: dict[str, set[str]] = {}
    for arc in arcs:
        needs: set[str] = set()
        for need in arc.requires:
            needs.update(pid for pid in index.get(need, ()) if pid in by_id)
        depends[arc.id] = needs - {arc.id}

    emitted: list[str] = []
    done: set[str] = set()
    ready = sorted(a for a, d in depends.items() if not d)
    while ready:
        node = ready.pop(0)
        if node in done:
            continue
        emitted.append(node)
        done.add(node)
        freed = sorted(
            a for a, d in depends.items() if a not in done and d <= done and a not in ready
        )
        ready.extend(freed)
    return tuple(emitted)


def analyse(
    arcs: Sequence[ArcSpec],
    demands: Sequence[Demand],
    budget: CostBudget = UNBUDGETED,
) -> ClosureReport:
    """The static check, as a report. Never raises on a plan that does not close.

    For every demand: resolve its metric to a quantity on a subject, find an arc that produces
    exactly that, and verify that arc's own requirements are transitively satisfied. Detect cycles
    over the whole arc graph first, because an arc inside one can never run whatever else is true
    of it. Then cost the arcs the demands actually reach and compare that with the budget.

    Costing the reachable arcs rather than every arc in the plan is deliberate. An arc nothing
    needs should not run, so charging the plan for it would refuse plans that are affordable. It
    is listed in `unused_arcs` instead, because an arc nobody needs is usually a mistake in one
    direction or the other.
    """
    by_id: dict[str, ArcSpec] = {}
    for arc in arcs:
        if arc.id in by_id and by_id[arc.id] != arc:
            raise ValueError(
                f"two different arcs are both called {arc.id!r}. An arc id is what a closure "
                f"failure names, so it has to mean one thing."
            )
        by_id[arc.id] = arc
    unique = tuple(by_id.values())

    index = _producer_index(unique)
    cyclic = _cyclic_arcs(unique, index)
    memo: dict[str, _Walk] = {}

    gaps: list[Gap] = []
    metric_arcs: dict[str, str] = {}
    ambiguous: dict[str, tuple[str, ...]] = {}
    roots: set[str] = set()

    for demand in demands:
        if demand.output is None:
            gaps.append(
                Gap(
                    kind=GapKind.UNBOUND_METRIC,
                    demand=demand,
                    detail=(
                        f"the plan binds no quantity to the metric {demand.metric!r}, so there is "
                        f"nothing to look for a producer of."
                    ),
                    remedy=(
                        f"add a MetricBinding for {demand.metric!r} in study {demand.study!r} "
                        f"naming the quantity and the subject it is measured on, or drop the "
                        f"prediction."
                    ),
                )
            )
            continue

        producers = index.get(demand.output, ())
        if not producers:
            gaps.append(
                Gap(
                    kind=GapKind.NO_PRODUCER,
                    demand=demand,
                    missing=demand.output,
                    detail=(
                        f"no arc in this plan produces {demand.output}. The prediction cannot be "
                        f"adjudicated by any work this plan contains."
                    ),
                    remedy=(
                        f"add an arc that produces {demand.output.quantity} on "
                        f"{render_subject(demand.output.subject)}, or drop the prediction before "
                        f"the run starts."
                    ),
                )
            )
            continue

        walk = _Walk(False)
        chosen = producers[0]
        for producer in producers:
            walk = _satisfiable(producer, by_id, index, cyclic, memo)
            if walk.ok:
                chosen = producer
                break
        if len(producers) > 1:
            ambiguous[str(demand.output)] = producers

        if walk.ok:
            metric_arcs[demand.metric] = chosen
            roots.add(chosen)
            continue

        if walk.kind is GapKind.CYCLE:
            gaps.append(
                Gap(
                    kind=GapKind.CYCLE,
                    demand=demand,
                    missing=demand.output,
                    arc=chosen,
                    chain=walk.chain,
                    detail=(
                        f"arc {walk.chain[-1]!r} sits in a dependency loop, so it can never start "
                        f"and {demand.output} is never produced."
                    ),
                    remedy=(
                        f"break the loop through {' -> '.join(walk.chain)}: one of those arcs has "
                        f"to stop requiring an output of the others."
                    ),
                )
            )
            continue

        blocked = walk.chain[-1] if walk.chain else chosen
        gaps.append(
            Gap(
                kind=GapKind.UNSATISFIED_INPUT,
                demand=demand,
                missing=walk.missing,
                arc=blocked,
                chain=walk.chain,
                detail=(
                    f"arc {blocked!r} requires {walk.missing}, and no arc in this plan produces "
                    f"it, so {chosen!r} never runs."
                ),
                remedy=(
                    f"add an arc that produces {walk.missing.quantity if walk.missing else '?'} "
                    f"on {render_subject(walk.missing.subject) if walk.missing else '?'}, or "
                    f"remove that requirement from {blocked!r}."
                ),
            )
        )

    reached = _reachable(roots, by_id, index)
    cost = total_cost(by_id[a] for a in sorted(reached))
    overruns = budget.overruns(cost)
    if overruns:
        terms = "; ".join(
            f"{name} {spend:,.4g} against a ceiling of {ceiling:,.4g}"
            for name, (spend, ceiling) in sorted(overruns.items())
        )
        gaps.append(
            Gap(
                kind=GapKind.OVER_BUDGET,
                detail=(
                    f"the {len(reached)} arcs these predictions reach cost more than the plan "
                    f"declares: {terms}."
                ),
                remedy=(
                    "raise the budget, or drop predictions until the arcs they reach fit inside "
                    "it. Deciding which predictions to drop after seeing the data is not a "
                    "budget decision."
                ),
            )
        )

    return ClosureReport(
        gaps=tuple(gaps),
        order=_topological(unique, index),
        required_arcs=frozenset(reached),
        unused_arcs=tuple(sorted(set(by_id) - reached)),
        metric_arcs=dict(sorted(metric_arcs.items())),
        cost=cost,
        budget=budget,
        demands=tuple(demands),
        ambiguous=dict(sorted(ambiguous.items())),
    )


__all__ = [
    "UNBUDGETED",
    "ArcSpec",
    "CostBudget",
    "ClosureError",
    "ClosureReport",
    "Demand",
    "DemandKind",
    "Gap",
    "GapKind",
    "MetricBinding",
    "Output",
    "analyse",
    "on",
    "render_subject",
    "subject_key",
    "total_cost",
]

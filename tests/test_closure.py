"""Unit and property tests for plan closure.

The algorithm is small, so most of these are hand-built plans with one known gap each. The two
property tests cover the parts a hand-built example cannot: that a cycle anywhere in the arc graph
is always found, and that a plan whose arcs form a DAG covering every registered metric always
closes. Those two together are what stop the check from being either vacuous or superstitious.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from reward_lens.core.closure import (
    ArcSpec,
    ClosureError,
    ClosureReport,
    CostBudget,
    Demand,
    GapKind,
    MetricBinding,
    Output,
    _cyclic_arcs,
    _producer_index,
    analyse,
    on,
    render_subject,
    subject_key,
    total_cost,
)
from reward_lens.core.quantity import CostModel
from reward_lens.core.types import SubjectRef
from reward_lens.studies.plan import (
    Plan,
    PlanBuilder,
    bind,
    check_closure,
    closure_report,
    demands_of,
    metric_arcs_for,
    plan_of,
)
from reward_lens.studies.spec import Hypothesis, KillCriterion, Prediction, StudySpec

# ---------------------------------------------------------------------------
# Fixtures: one study, two arcs, one gap to move around
# ---------------------------------------------------------------------------


def a_study(
    study_id: str = "demo",
    metrics: tuple[str, ...] = ("drift",),
    kill_metrics: tuple[str, ...] = (),
) -> StudySpec:
    return StudySpec(
        id=study_id,
        title="a study that registers something",
        science="S03-thermo",
        hypotheses=tuple(
            Hypothesis(
                id=f"H{i}",
                statement=f"{m} exceeds the registered threshold",
                prediction=Prediction(metric=m, comparator=">", threshold=0.3),
            )
            for i, m in enumerate(metrics)
        ),
        analysis="tests.test_closure.nothing",
        kill_criteria=tuple(
            KillCriterion(id=f"K{i}", metric=m, comparator=">", threshold=0.1)
            for i, m in enumerate(kill_metrics)
        ),
    )


def analysis_arc(*needs: Output, arc_id: str = "arc:analysis", card: str = "demo") -> ArcSpec:
    """The arc that turns intermediates into the metrics a study registered."""
    return ArcSpec(
        id=arc_id,
        produces=frozenset({on("result", card=card, metric="drift")}),
        requires=frozenset(needs),
    )


# ---------------------------------------------------------------------------
# Subjects and outputs
# ---------------------------------------------------------------------------


def test_a_subject_with_extras_is_hashable_through_its_key():
    """`SubjectRef` carries a dict, so it cannot be hashed, and the campaign's join key is in it."""
    subject = SubjectRef(extra={"roster_key": "armorm", "slice": "diagnostic-v3"})
    with pytest.raises(TypeError):
        hash(subject)
    assert subject_key(subject) == subject_key(
        SubjectRef(extra={"slice": "diagnostic-v3", "roster_key": "armorm"})
    )


def test_two_outputs_built_separately_are_the_same_output():
    """The producer and the consumer are written in different modules and never share an object."""
    assert on("campaign.scores", roster_key="armorm") == on("campaign.scores", roster_key="armorm")
    assert len({on("q", a=1), on("q", a=1)}) == 1
    assert on("q", a=1) != on("q", a=2)
    assert on("q", a=1) != on("other", a=1)


def test_the_same_quantity_on_a_different_subject_is_a_different_output():
    """The permissive reading crosses quantities with subjects. It is the failure mode to avoid."""
    battery = on("campaign.bias.battery", roster_key="armorm", slice="diagnostic-v3")
    degradation = on(
        "campaign.bias.battery", roster_key="armorm", slice="diagnostic-v3-degradation"
    )
    assert battery != degradation
    arc = ArcSpec(id="arc:model", produces=frozenset({battery}))
    assert degradation not in arc.produces


def test_a_subject_renders_its_extras_because_that_is_where_the_join_key_lives():
    rendered = render_subject(SubjectRef(extra={"roster_key": "armorm", "slice": "rb2-full"}))
    assert "roster_key=armorm" in rendered
    assert "slice=rb2-full" in rendered


def test_an_arc_that_requires_its_own_output_is_refused_at_construction():
    same = on("q", a=1)
    with pytest.raises(ValueError, match="cycle of length one"):
        ArcSpec(id="arc:self", produces=frozenset({same}), requires=frozenset({same}))


def test_arc_subjects_are_derived_rather_than_declared_twice():
    arc = ArcSpec(
        id="arc:x",
        produces=frozenset({on("q", slice="a")}),
        requires=frozenset({on("r", slice="b")}),
    )
    assert {s.extra["slice"] for s in arc.subjects} == {"a", "b"}


# ---------------------------------------------------------------------------
# The five gap kinds
# ---------------------------------------------------------------------------


def test_a_metric_no_arc_produces_raises_naming_prediction_metric_and_gap():
    """The acceptance shape: the gap is found before anything runs, and it names all three."""
    spec = a_study()
    plan = plan_of(studies=[spec], arcs=[], bindings=bind(spec, "result", card="demo"))

    with pytest.raises(ClosureError) as caught:
        check_closure(plan)

    message = str(caught.value)
    assert "H0" in message  # the prediction
    assert "drift" in message  # the metric
    assert "no arc in this plan produces" in message  # the gap
    gap = caught.value.report.gaps[0]
    assert gap.kind is GapKind.NO_PRODUCER
    assert gap.remedy.startswith("add an arc")


def test_a_metric_the_plan_never_bound_is_a_gap_rather_than_a_pass():
    """ "Nobody said what this metric is" and "nobody produces it" are the same problem."""
    spec = a_study()
    plan = plan_of(studies=[spec], arcs=[analysis_arc()], bindings=())

    report = closure_report(plan)

    assert not report.closed
    assert report.gaps[0].kind is GapKind.UNBOUND_METRIC
    assert "drift" in report.gaps[0].detail


def test_an_arc_whose_own_input_is_missing_blocks_the_prediction_and_names_the_arc():
    """The campaign's shape: the analysis exists, the intermediate it reads does not."""
    spec = a_study()
    missing = on("campaign.bias.battery", roster_key="armorm", slice="diagnostic-v3-degradation")
    plan = plan_of(
        studies=[spec],
        arcs=[analysis_arc(missing)],
        bindings=bind(spec, "result", card="demo"),
    )

    report = closure_report(plan)

    gap = report.gaps[0]
    assert gap.kind is GapKind.UNSATISFIED_INPUT
    assert gap.arc == "arc:analysis"
    assert gap.missing == missing
    assert "roster_key=armorm" in gap.detail
    assert "diagnostic-v3-degradation" in gap.detail


def test_a_gap_three_arcs_deep_names_the_whole_chain():
    """A failure at the leaf is useless if the reader cannot see which path reached it."""
    spec = a_study()
    plan = plan_of(
        studies=[spec],
        arcs=[
            analysis_arc(on("mid", step=1)),
            ArcSpec(
                id="arc:mid",
                produces=frozenset({on("mid", step=1)}),
                requires=frozenset({on("deep", step=2)}),
            ),
            ArcSpec(
                id="arc:deep",
                produces=frozenset({on("deep", step=2)}),
                requires=frozenset({on("absent", step=3)}),
            ),
        ],
        bindings=bind(spec, "result", card="demo"),
    )

    gap = closure_report(plan).gaps[0]

    assert gap.chain == ("arc:analysis", "arc:mid", "arc:deep")
    assert gap.arc == "arc:deep"
    assert gap.missing == on("absent", step=3)
    assert "arc:analysis -> arc:mid -> arc:deep" in gap.render()


def test_a_cycle_is_a_gap_and_the_remedy_names_the_loop():
    spec = a_study()
    plan = plan_of(
        studies=[spec],
        arcs=[
            analysis_arc(on("a", i=1)),
            ArcSpec(
                id="arc:a",
                produces=frozenset({on("a", i=1)}),
                requires=frozenset({on("b", i=2)}),
            ),
            ArcSpec(
                id="arc:b",
                produces=frozenset({on("b", i=2)}),
                requires=frozenset({on("a", i=1)}),
            ),
        ],
        bindings=bind(spec, "result", card="demo"),
    )

    gap = closure_report(plan).gaps[0]

    assert gap.kind is GapKind.CYCLE
    assert set(gap.chain) >= {"arc:a"}
    assert "loop" in gap.detail


def test_a_plan_that_closes_but_costs_too_much_is_still_refused():
    spec = a_study()
    plan = plan_of(
        studies=[spec],
        arcs=[
            ArcSpec(
                id="arc:analysis",
                produces=frozenset({on("result", card="demo", metric="drift")}),
                cost=CostModel(dollars=42.0, gpu_seconds=600.0),
            )
        ],
        bindings=bind(spec, "result", card="demo"),
        budget=CostBudget(dollars=30.0),
    )

    report = closure_report(plan)

    assert [g.kind for g in report.gaps] == [GapKind.OVER_BUDGET]
    assert "dollars 42 against a ceiling of 30" in report.gaps[0].detail
    # The unbudgeted currency is not silently treated as a ceiling of zero.
    assert "gpu_seconds" not in report.gaps[0].detail


def test_the_budget_counts_only_the_arcs_the_predictions_reach():
    """An arc nothing needs should not run, so charging the plan for it would refuse good plans."""
    spec = a_study()
    plan = plan_of(
        studies=[spec],
        arcs=[
            ArcSpec(
                id="arc:analysis",
                produces=frozenset({on("result", card="demo", metric="drift")}),
                cost=CostModel(dollars=5.0),
            ),
            ArcSpec(
                id="arc:spare",
                produces=frozenset({on("nobody.wants", x=1)}),
                cost=CostModel(dollars=1000.0),
            ),
        ],
        bindings=bind(spec, "result", card="demo"),
        budget=CostBudget(dollars=10.0),
    )

    report = check_closure(plan)

    assert report.closed
    assert report.cost.dollars == 5.0
    assert report.unused_arcs == ("arc:spare",)


# ---------------------------------------------------------------------------
# Kill criteria are first-class demands
# ---------------------------------------------------------------------------


def test_a_kill_criterion_with_no_producer_is_a_gap_on_the_same_footing_as_a_hypothesis():
    """This is the dangerous one: an unevaluated kill used to look like one that passed."""
    spec = a_study(metrics=("drift",), kill_metrics=("collapse",))
    plan = plan_of(
        studies=[spec],
        arcs=[
            ArcSpec(
                id="arc:analysis",
                produces=frozenset({on("result", card="demo", metric="drift")}),
            )
        ],
        bindings=bind(spec, "result", card="demo"),
    )

    report = closure_report(plan)

    kill_gaps = [g for g in report.gaps if g.demand is not None and g.demand.kind == "kill"]
    assert len(kill_gaps) == 1
    assert kill_gaps[0].demand.owner == "K0"
    assert "kill criterion K0" in kill_gaps[0].render()


def test_demands_cover_every_hypothesis_and_every_kill_criterion():
    spec = a_study(metrics=("m1", "m2"), kill_metrics=("k1",))
    plan = plan_of(studies=[spec], arcs=[], bindings=bind(spec, "result", card="demo"))

    demands = demands_of(plan)

    assert [d.owner for d in demands] == ["H0", "H1", "K0"]
    assert [d.kind for d in demands] == ["hypothesis", "hypothesis", "kill"]
    assert demands[0].statement == "m1 > 0.3"


# ---------------------------------------------------------------------------
# The closed case, and what it hands the runner
# ---------------------------------------------------------------------------


def test_a_closed_plan_returns_a_report_with_the_per_metric_arc_attribution():
    """Closure rule 3: the runner records which arc was supposed to produce each metric."""
    spec = a_study(metrics=("drift",), kill_metrics=("collapse",))
    bank = on("campaign.scores", roster_key="a", slice="s")
    plan = plan_of(
        studies=[spec],
        arcs=[
            ArcSpec(id="arc:model", produces=frozenset({bank}), cost=CostModel(gpu_seconds=90.0)),
            ArcSpec(
                id="arc:analysis",
                produces=frozenset(
                    {
                        on("result", card="demo", metric="drift"),
                        on("result", card="demo", metric="collapse"),
                    }
                ),
                requires=frozenset({bank}),
            ),
        ],
        bindings=bind(spec, "result", card="demo"),
    )

    report = check_closure(plan)

    assert report.closed
    assert report.metric_arcs == {"collapse": "arc:analysis", "drift": "arc:analysis"}
    assert report.order == ("arc:model", "arc:analysis")
    assert report.required_arcs == {"arc:model", "arc:analysis"}
    assert report.cost.gpu_seconds == 90.0
    assert "plan closes" in report.render()


def test_the_metric_arc_mapping_is_what_the_runner_takes(tmp_path):
    """The mapping feeds `run_study(metric_arcs=...)`, which turns a void into a work item."""
    from reward_lens.core.store import EvidenceStore
    from reward_lens.studies.runner import run_study
    from reward_lens.studies.spec import StudyResult

    spec = a_study(metrics=("drift",))
    plan = plan_of(
        studies=[spec],
        arcs=[
            ArcSpec(
                id="arc:battery",
                produces=frozenset({on("result", card="demo", metric="drift")}),
            )
        ],
        bindings=bind(spec, "result", card="demo"),
    )
    check_closure(plan)

    _, result = run_study(
        spec,
        store=EvidenceStore(tmp_path / "store"),
        analysis_fn=lambda run: StudyResult(outcomes={}, metrics={}),
        metric_arcs=metric_arcs_for(plan, spec),
    )

    assert result.voids["H0"].arc == "arc:battery"


def test_two_studies_may_use_one_metric_name_for_different_things():
    """Per-study scoping is why `metric_arcs_for` exists beside the plan-wide mapping."""
    first, second = a_study("first"), a_study("second")
    plan = plan_of(
        studies=[first, second],
        arcs=[
            ArcSpec(id="arc:one", produces=frozenset({on("result", card="first", metric="drift")})),
            ArcSpec(
                id="arc:two", produces=frozenset({on("result", card="second", metric="drift")})
            ),
        ],
        bindings=bind(first, "result", card="first") + bind(second, "result", card="second"),
    )

    check_closure(plan)

    assert metric_arcs_for(plan, first) == {"drift": "arc:one"}
    assert metric_arcs_for(plan, second) == {"drift": "arc:two"}


def test_one_metric_bound_two_ways_inside_one_study_is_refused():
    spec = a_study()
    plan = plan_of(
        studies=[spec],
        arcs=[],
        bindings=(
            MetricBinding(study="demo", metric="drift", quantity="a"),
            MetricBinding(study="demo", metric="drift", quantity="b"),
        ),
    )
    with pytest.raises(ValueError, match="binds the metric"):
        closure_report(plan)


def test_two_arcs_producing_one_output_are_recorded_rather_than_refused():
    """Redundancy is legal. Silent ambiguity in the attribution is not, so it is reported."""
    spec = a_study()
    out = on("result", card="demo", metric="drift")
    plan = plan_of(
        studies=[spec],
        arcs=[
            ArcSpec(id="arc:a", produces=frozenset({out})),
            ArcSpec(id="arc:b", produces=frozenset({out})),
        ],
        bindings=bind(spec, "result", card="demo"),
    )

    report = check_closure(plan)

    assert report.closed
    assert report.ambiguous[str(out)] == ("arc:a", "arc:b")


def test_a_second_producer_rescues_a_prediction_whose_first_producer_is_blocked():
    """The walk has to try every producer, not only the first one it indexed."""
    spec = a_study()
    out = on("result", card="demo", metric="drift")
    plan = plan_of(
        studies=[spec],
        arcs=[
            ArcSpec(id="arc:a", produces=frozenset({out}), requires=frozenset({on("absent")})),
            ArcSpec(id="arc:b", produces=frozenset({out})),
        ],
        bindings=bind(spec, "result", card="demo"),
    )

    report = check_closure(plan)

    assert report.closed
    assert report.metric_arcs == {"drift": "arc:b"}


def test_an_empty_plan_closes_because_it_promises_nothing():
    assert check_closure(Plan()).closed


def test_the_builder_assembles_the_same_plan_in_pieces():
    spec = a_study()
    arc = ArcSpec(id="arc:x", produces=frozenset({on("result", card="demo", metric="drift")}))
    built = (
        PlanBuilder(name="p")
        .with_studies(spec)
        .with_arcs(arc)
        .with_bindings(*bind(spec, "result", card="demo"))
        .with_budget(CostBudget(dollars=1.0))
        .build()
    )
    assert built == plan_of(
        [spec], [arc], bind(spec, "result", card="demo"), CostBudget(dollars=1.0), "p"
    )


def test_two_different_arcs_under_one_id_are_refused():
    with pytest.raises(ValueError, match="mean one thing"):
        analyse(
            [
                ArcSpec(id="arc:x", produces=frozenset({on("a")})),
                ArcSpec(id="arc:x", produces=frozenset({on("b")})),
            ],
            [],
        )


def test_costs_add_and_an_undeclared_currency_stays_undeclared():
    """Summing None as zero would turn "nobody costed this" into "$0.00", which reads as a number."""
    total = total_cost(
        [
            ArcSpec(id="a", cost=CostModel(gpu_seconds=10.0)),
            ArcSpec(id="b", cost=CostModel(gpu_seconds=5.0, calls=3)),
        ]
    )
    assert total.gpu_seconds == 15.0
    assert total.calls == 3
    assert total.dollars is None


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(size=st.integers(min_value=2, max_value=7), seed=st.integers(min_value=0, max_value=10**6))
def test_a_cycle_is_always_detected(size: int, seed: int):
    """Whatever the shape, an arc inside a loop can never run and the check has to say so.

    The plan is a ring of `size` arcs, each requiring the previous one's output, with the demand
    pointing at the first. Every arc in it is unreachable, so a report that closes would be a
    report that licensed a run which could not start.
    """
    outs = [on("q", i=i) for i in range(size)]
    arcs = [
        ArcSpec(
            id=f"arc:{i}",
            produces=frozenset({outs[i]}),
            requires=frozenset({outs[(i - 1) % size]}),
        )
        for i in range(size)
    ]
    rotated = arcs[seed % size :] + arcs[: seed % size]
    demand = Demand(
        study="s", owner="H0", kind="hypothesis", metric="m", statement="m > 0", output=outs[0]
    )

    report = analyse(rotated, [demand])

    assert not report.closed
    assert report.gaps[0].kind is GapKind.CYCLE
    assert report.order == ()  # nothing in a ring can be ordered


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    edges=st.lists(
        st.tuples(st.integers(min_value=0, max_value=6), st.integers(min_value=0, max_value=6)),
        max_size=14,
    ),
    n=st.integers(min_value=1, max_value=7),
)
def test_a_dag_covering_every_metric_always_closes(edges, n: int):
    """The other half. A check that never passes is as useless as one that never fails.

    Arcs are numbered and an arc may only require the output of a strictly lower-numbered arc, so
    the graph is a DAG by construction whatever the drawn edges are. Every arc produces its own
    quantity and one demand points at each, so the plan covers every metric.
    """
    outs = [on("q", i=i) for i in range(n)]
    requires: list[set[Output]] = [set() for _ in range(n)]
    for a, b in edges:
        lo, hi = min(a, b), max(a, b)
        if lo == hi or hi >= n:
            continue
        requires[hi].add(outs[lo])
    arcs = [
        ArcSpec(id=f"arc:{i}", produces=frozenset({outs[i]}), requires=frozenset(requires[i]))
        for i in range(n)
    ]
    demands = [
        Demand(
            study="s",
            owner=f"H{i}",
            kind="hypothesis",
            metric=f"m{i}",
            statement=f"m{i} > 0",
            output=outs[i],
        )
        for i in range(n)
    ]

    report = analyse(arcs, demands)

    assert report.closed, report.render()
    assert len(report.order) == n
    position = {arc_id: k for k, arc_id in enumerate(report.order)}
    for i in range(n):
        for need in requires[i]:
            producer = f"arc:{need.subject.extra['i']}"
            assert position[producer] < position[f"arc:{i}"]


@settings(max_examples=100, deadline=None)
@given(missing=st.integers(min_value=0, max_value=5))
def test_removing_any_single_producer_from_a_closed_plan_opens_a_gap(missing: int):
    """Closure is not a property of the plan's size. Dropping one producer has to be visible."""
    n = 6
    outs = [on("q", i=i) for i in range(n)]
    arcs = [ArcSpec(id=f"arc:{i}", produces=frozenset({outs[i]})) for i in range(n)]
    demands = [
        Demand(
            study="s",
            owner=f"H{i}",
            kind="hypothesis",
            metric=f"m{i}",
            statement=f"m{i} > 0",
            output=outs[i],
        )
        for i in range(n)
    ]
    assert analyse(arcs, demands).closed

    report = analyse([a for k, a in enumerate(arcs) if k != missing], demands)

    assert [g.kind for g in report.gaps] == [GapKind.NO_PRODUCER]
    assert report.gaps[0].demand.owner == f"H{missing}"


def test_the_report_of_a_closed_plan_is_still_worth_reading():
    """A gate that can only speak when it refuses gives the operator nothing on the happy path."""
    report: ClosureReport = analyse([ArcSpec(id="arc:x", produces=frozenset({on("q")}))], [])
    assert report.closed
    assert report.unused_arcs == ("arc:x",)
    assert report.render().startswith("plan closes")


def _on_a_cycle(edges: dict[str, set[str]]) -> set[str]:
    """Transitive closure, then the nodes that reach themselves. Slow and obviously correct."""
    reach = {node: set(targets) for node, targets in edges.items()}
    changed = True
    while changed:
        changed = False
        for node in edges:
            grown = set(reach[node])
            for target in list(reach[node]):
                grown |= reach[target]
            if grown != reach[node]:
                reach[node] = grown
                changed = True
    return {node for node in edges if node in reach[node]}


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=2, max_value=6),
    drawn=st.lists(
        st.tuples(st.integers(min_value=0, max_value=5), st.integers(min_value=0, max_value=5)),
        max_size=20,
    ),
)
def test_cycle_detection_agrees_with_a_brute_force_oracle(n: int, drawn):
    """Tarjan is worth checking against something obviously right, on graphs nobody chose.

    The fast implementation exists so a deep plan does not hit the recursion limit. That is a
    performance reason for a correctness-critical routine, which is the combination worth pinning
    against a transitive-closure oracle rather than against three hand-drawn examples.
    """
    outs = [on("q", i=i) for i in range(n)]
    edges: dict[str, set[str]] = {f"arc:{i}": set() for i in range(n)}
    requires: list[set[Output]] = [set() for _ in range(n)]
    for a, b in drawn:
        if a >= n or b >= n or a == b:
            continue
        requires[a].add(outs[b])
        edges[f"arc:{a}"].add(f"arc:{b}")
    arcs = [
        ArcSpec(id=f"arc:{i}", produces=frozenset({outs[i]}), requires=frozenset(requires[i]))
        for i in range(n)
    ]

    detected = _cyclic_arcs(arcs, _producer_index(arcs))

    assert detected == _on_a_cycle(edges)
    # And nothing on a cycle survives into the run order.
    assert not (set(analyse(arcs, []).order) & detected)

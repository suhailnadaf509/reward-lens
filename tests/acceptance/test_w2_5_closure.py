"""Acceptance: a plan that cannot answer its own predictions does not start.

The clause this file discharges: *a plan whose study registers a metric no arc produces raises
`ClosureError` naming the prediction, the metric and the gap, before any work runs.*

The second half is the regression, and it is the reason the package exists. The campaign that
produced this library's evidence base ran 27 preregistered cards and eight of them came back
"inconclusive", every time for an infrastructural reason. This file rebuilds that campaign's plan
from its own source and its own run, runs closure over it, and checks that the gate fires on the
right cards and names the arc for each.

What that regression establishes, stated exactly, because the difference matters:

* Closure names five of the eight cards and thirteen of their registered predictions, from the
  campaign's own resolvers and its own arc stamps, before any adjudication. Nine of the thirteen
  are hypotheses and **four are kill criteria**, and those four are the sharp end: the campaign's
  scoreboard shows a blank in the "Kill fired" column for each of them, which is what an
  unevaluated safety check looked like under the adjudication this project replaced.
* It does not name `GAUGE-E19` or `GAUGE-XFAM`, and it should not. Those two failed with
  `PermissionError(13, 'Permission denied')` inside `resolve_subjects`. That is a filesystem fault
  rather than an unproducible metric, and a gate that claimed it would have caught them would be
  overclaiming.
* It does not name `HUMP` because `campaign.hump` does not import against this branch at all, so
  its resolver's read list cannot be recovered. Its missing intermediate,
  `campaign.recorder.drift` on the policy seat, is the same shape as the five that are caught.

And the finding that does not fit in an assertion, so it is written here. The campaign could not
have been closed in advance whatever gate existed, because **it never declared what its arcs
produce.** `default_menu` derives a seat's slice list from the cards that name it, so on that axis
the producer set is defined by the consumer set and cannot fail. `_vocab_targets` is computed
inside the arc, after the seat has loaded. And a static read of `campaign/arcs.py` finds record
sites whose observable or subject is an f-string or a local variable, which no checker can resolve.
The gate in `reward_lens.core.closure` is worth having because it makes the declaration a
precondition of running, not because the old code was one annotation away from being safe.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from reward_lens.core.closure import ArcSpec, ClosureError, GapKind, Output, on
from reward_lens.core.types import SubjectRef
from reward_lens.studies.plan import bind, check_closure, closure_report, plan_of
from reward_lens.studies.spec import Hypothesis, KillCriterion, Prediction, StudySpec

# ---------------------------------------------------------------------------
# The clause
# ---------------------------------------------------------------------------


def test_a_metric_no_arc_produces_raises_before_any_work_runs():
    """The clause, in the smallest plan that can carry it.

    Two hypotheses and a kill criterion. One hypothesis is answerable and the other two
    predictions are not, so the failure has to be specific about which.
    """
    spec = StudySpec(
        id="w2.5-acceptance",
        title="a study that registers more than the plan can answer",
        science="S03-thermo",
        hypotheses=(
            Hypothesis(
                id="H-reachable",
                statement="the drift the battery measures exceeds the registered threshold",
                prediction=Prediction(metric="exploit_drift", comparator=">", threshold=0.3),
            ),
            Hypothesis(
                id="H-unreachable",
                statement="the bias battery correlates with the benchmark",
                prediction=Prediction(
                    metric="spearman_battery_vs_benchmark", comparator=">", threshold=0.6
                ),
            ),
        ),
        analysis="campaign.style.analyze",
        kill_criteria=(
            KillCriterion(
                id="K-collapse",
                metric="entropy_collapse",
                comparator=">",
                threshold=0.1,
                description="the arm collapsed, so nothing measured on it is a reading",
            ),
        ),
    )
    battery = on("campaign.bias.battery", roster_key="armorm", slice="diagnostic-v3-degradation")
    plan = plan_of(
        studies=[spec],
        arcs=[
            ArcSpec(
                id="arc:analysis:STYLE",
                produces=frozenset(
                    {on("campaign.result.STYLE", card="STYLE", metric="exploit_drift")}
                ),
            )
        ],
        bindings=bind(spec, "campaign.result.STYLE", card="STYLE"),
    )

    with pytest.raises(ClosureError) as caught:
        check_closure(plan)

    report = caught.value.report
    message = str(caught.value)

    # It names the prediction, the metric and the gap, in that order, for each unreachable one.
    assert "H-unreachable" in message
    assert "spearman_battery_vs_benchmark > 0.6" in message
    assert "K-collapse" in message
    assert "entropy_collapse > 0.1" in message
    assert "no arc in this plan produces" in message

    # And it does not complain about the one the plan can answer.
    assert "H-reachable" not in message
    assert {g.demand.owner for g in report.gaps} == {"H-unreachable", "K-collapse"}
    assert {g.kind for g in report.gaps} == {GapKind.NO_PRODUCER}

    # Every gap carries an instruction rather than a diagnosis.
    for gap in report.gaps:
        assert gap.remedy.startswith("add an arc")
        assert "drop the prediction before the run starts" in gap.remedy

    # All three predictions were considered, so the two that pass did so by being answerable
    # rather than by being skipped.
    assert [d.owner for d in report.demands] == ["H-reachable", "H-unreachable", "K-collapse"]

    # The same plan with the missing producer added closes, which is what makes this a gate.
    fixed = plan_of(
        studies=[spec],
        arcs=[
            ArcSpec(id="arc:model:armorm", produces=frozenset({battery})),
            ArcSpec(
                id="arc:analysis:STYLE",
                produces=frozenset(
                    {
                        on("campaign.result.STYLE", card="STYLE", metric=m)
                        for m in (
                            "exploit_drift",
                            "spearman_battery_vs_benchmark",
                            "entropy_collapse",
                        )
                    }
                ),
                requires=frozenset({battery}),
            ),
        ],
        bindings=bind(spec, "campaign.result.STYLE", card="STYLE"),
    )
    assert check_closure(fixed).closed


# ---------------------------------------------------------------------------
# The campaign, rebuilt from its own source
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]
CAMPAIGN_ROOT = _REPO.parent / "reward-lens-campaign"
CAMPAIGN_STORE = _REPO.parent.parent / "campaign-results" / "runs" / "campaign" / "evidence.jsonl"

#: The eight cards the campaign's scoreboard labels inconclusive at card level. Counted with
#: ``awk -F'|' '/inconclusive/ {gsub(/ /,"",$3); if ($3=="") c++; else h++} END {print c, h}'``
#: over the campaign's scoreboard, which gives 8 and 16.
INCONCLUSIVE_CARDS = frozenset(
    {
        "STYLE-RMB",
        "PPE-BON",
        "HUMP",
        "GAUGE-E19",
        "GAUGE-XFAM",
        "HACK-FORE",
        "VALUES-CONTEST",
        "T3-FIELD",
    }
)

#: The three closure cannot reach, and why. Asserted so that the limit is a checked fact rather
#: than a caveat somebody remembers.
NOT_REACHABLE_BY_CLOSURE = {
    "GAUGE-E19": "resolve_subjects failed with PermissionError, not a missing intermediate",
    "GAUGE-XFAM": "resolve_subjects failed with PermissionError, not a missing intermediate",
    "HUMP": "campaign.hump does not import against this branch, so its reads cannot be read",
}

_UNRESOLVED = object()


def _skip_without_campaign() -> None:
    if not (CAMPAIGN_ROOT / "campaign" / "arcs.py").exists():
        pytest.skip(f"campaign worktree not present at {CAMPAIGN_ROOT}")
    if not CAMPAIGN_STORE.exists():
        pytest.skip(f"campaign evidence store not present at {CAMPAIGN_STORE}")


@pytest.fixture(scope="module")
def campaign() -> Any:
    """Import the campaign's plan objects without writing anything into its worktree."""
    _skip_without_campaign()
    dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # the worktree is read-only reference; leave no __pycache__
    root = str(CAMPAIGN_ROOT)
    added = root not in sys.path
    if added:
        sys.path.insert(0, root)
    try:
        registry = importlib.import_module("campaign.registry")
        yield registry
    finally:
        sys.dont_write_bytecode = dont_write
        if added and root in sys.path:
            sys.path.remove(root)


def _const(node: ast.AST | None, env: dict[str, Any]) -> Any:
    """Evaluate an expression node against a module's globals, or report that it cannot be."""
    if node is None:
        return None
    try:
        return eval(compile(ast.Expression(node), "<plan>", "eval"), env)  # noqa: S307
    except Exception:  # noqa: BLE001 - anything unevaluable is an unreadable declaration
        return _UNRESOLVED


def _loop_bindings(fn_node: ast.AST) -> dict[ast.Call, list[tuple[ast.AST, ast.AST]]]:
    """Every call in a function, with the `for` targets and iterables enclosing it."""
    found: dict[ast.Call, list[tuple[ast.AST, ast.AST]]] = {}

    def walk(node: ast.AST, binders: list[tuple[ast.AST, ast.AST]]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.For):
                walk(child.iter, binders)
                inner = binders + [(child.target, child.iter)]
                for statement in child.body:
                    walk(statement, inner)
                for statement in child.orelse:
                    walk(statement, binders)
                continue
            if isinstance(child, ast.Call):
                found[child] = binders
            walk(child, binders)

    walk(fn_node, [])
    return found


def resolver_reads(module: Any) -> tuple[set[tuple[str, str, str]], list[str]]:
    """Every intermediate a card's ``resolve_subjects`` requires, as (observable, roster, slice).

    Only ``find_one`` counts. That is the campaign's own distinction and it is exactly the one
    closure needs: ``find_one`` raises when the intermediate is absent, so the read is required,
    while ``find_intermediates`` returns a possibly-empty list and every caller of it guards on
    emptiness, so the read is optional. Treating both as required manufactures gaps on
    `CHI-DRIFT`'s optional concept sub-arm and on `FORENSIC-RECEIPT`'s deliberately unrostered
    scan, neither of which is a defect.

    Reads whose arguments cannot be evaluated from the module's globals and its enclosing loops
    are returned as problems rather than dropped, because an under-declared requirement is how a
    closure check quietly stops checking.
    """
    fn = getattr(module, "resolve_subjects", None)
    if fn is None:
        return set(), ["module exports no resolve_subjects"]
    node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    env = dict(vars(module))
    triples: set[tuple[str, str, str]] = set()
    problems: list[str] = []

    for call, binders in _loop_bindings(node).items():
        func = call.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "find_one":
            continue
        environments = [dict(env)]
        for target, iterable in binders:
            values = _const(iterable, env)
            if values is _UNRESOLVED or isinstance(values, (str, bytes)):
                values = [_UNRESOLVED]
            elif not hasattr(values, "__iter__"):
                values = [_UNRESOLVED]
            expanded = []
            for environment in environments:
                for value in values:
                    child = dict(environment)
                    if isinstance(target, ast.Name):
                        child[target.id] = value
                    expanded.append(child)
            environments = expanded
        keywords = {k.arg: k.value for k in call.keywords}
        observable_node = call.args[1] if len(call.args) > 1 else keywords.get("observable")
        for environment in environments:
            observable = _const(observable_node, environment)
            roster = _const(keywords.get("roster_key"), environment)
            slice_name = _const(keywords.get("slice_name"), environment)
            if _UNRESOLVED in (observable, roster, slice_name) or roster is None:
                problems.append(ast.unparse(call)[:90])
                continue
            triples.add((observable, roster, slice_name))
    return triples, problems


def unreadable_record_sites(arcs_module: Any) -> list[str]:
    """Record sites in ``campaign/arcs.py`` whose observable or subject no checker can resolve.

    This is the measured reason the campaign had no closable plan. An arc that writes
    ``ctx.record(ek.OBS_SUBSPACE, f'{FRAME_SLICE}-reward', ...)`` has declared its output to a
    human reader and to nobody else.
    """
    tree = ast.parse(Path(arcs_module.__file__).read_text(encoding="utf-8"))
    env = dict(vars(arcs_module))
    field = importlib.import_module("campaign.field")
    env.update({k: v for k, v in vars(field).items() if k.startswith("OBS_")})
    unreadable: list[str] = []
    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
            continue
        if call.func.attr != "record":
            continue
        observable = _const(call.args[0] if call.args else None, env)
        slice_name = _const(call.args[1] if len(call.args) > 1 else None, env)
        if _UNRESOLVED in (observable, slice_name):
            unreadable.append(ast.unparse(call)[:80])
    return unreadable


def delivered_by_arc(path: Path) -> dict[tuple[str, str, str], set[str]]:
    """What each arc actually produced, from the arc label the campaign stamps on every row.

    ``evidence_keys.record_intermediate`` puts the arc label in ``provenance.extra['arc']`` and
    the join key in ``subject.extra``, so the run itself carries the producer relation the plan
    never wrote down. Reading it here is what lets closure be run against the campaign at all.
    """
    delivered: dict[tuple[str, str, str], set[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            extra = (row.get("subject") or {}).get("extra") or {}
            arc = ((row.get("provenance") or {}).get("extra") or {}).get("arc")
            if not arc:
                continue
            key = (row["observable"], extra.get("roster_key"), extra.get("slice"))
            delivered.setdefault(key, set()).add(arc)
    return delivered


def _output(observable: str, roster: str, slice_name: str) -> Output:
    return Output(observable, SubjectRef(extra={"roster_key": roster, "slice": slice_name}))


@pytest.fixture(scope="module")
def campaign_plan(campaign) -> Any:
    """The campaign as a `Plan`: its 27 studies, its arcs, and the bindings it never wrote.

    Studies come from `campaign/specs.py`. The arcs are one per label `campaign/arcs.py` stamps,
    each declaring what that label delivered. Each card gets an analysis arc that produces the
    card's metrics and requires the intermediates its own resolver reads.
    """
    specs_module = importlib.import_module("campaign.specs")
    arcs_module = importlib.import_module("campaign.arcs")
    specs = specs_module.build_all_specs()

    delivered = delivered_by_arc(CAMPAIGN_STORE)
    work_arcs: dict[str, set[Output]] = {}
    for (observable, roster, slice_name), labels in delivered.items():
        for label in labels:
            work_arcs.setdefault(label, set()).add(_output(observable, roster, slice_name))

    arcs = [
        ArcSpec(id=label, produces=frozenset(outputs))
        for label, outputs in sorted(work_arcs.items())
    ]
    bindings: list[Any] = []
    unreadable_cards: dict[str, str] = {}

    for card in campaign.CARDS:
        spec = specs[card.spec_id]
        module_path = card.analysis.rsplit(".", 1)[0]
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001 - an unimportable analysis is data, not a failure
            unreadable_cards[card.card] = f"{type(exc).__name__}: {exc}"
            requires: set[Output] = set()
        else:
            triples, _problems = resolver_reads(module)
            requires = {_output(*t) for t in triples}
        arcs.append(
            ArcSpec(
                id=f"analysis:{card.card}",
                produces=frozenset(
                    on(f"campaign.result.{card.card}", card=card.card, metric=metric)
                    for metric in _metrics_of(spec)
                ),
                requires=frozenset(requires),
            )
        )
        bindings.extend(bind(spec, f"campaign.result.{card.card}", card=card.card))

    plan = plan_of(
        studies=[specs[c.spec_id] for c in campaign.CARDS],
        arcs=arcs,
        bindings=bindings,
        name="campaign 2.0",
    )
    return plan, unreadable_cards, arcs_module


def _metrics_of(spec: StudySpec) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for hypothesis in spec.hypotheses:
        seen.setdefault(hypothesis.prediction.metric)
    for kill in spec.kill_criteria:
        seen.setdefault(kill.metric)
    return tuple(seen)


def test_the_campaign_plan_does_not_close(campaign_plan):
    """The regression. Run the gate over the campaign's own plan and it refuses."""
    plan, _unreadable, _arcs = campaign_plan

    with pytest.raises(ClosureError) as caught:
        check_closure(plan)

    report = caught.value.report
    assert not report.closed
    assert len(plan.studies) == 27  # every card, not a subset
    assert all(gap.kind is GapKind.UNSATISFIED_INPUT for gap in report.gaps)


def test_closure_names_the_cards_that_came_back_inconclusive(campaign_plan):
    """Five of the eight, with the intermediate and the arc for each, before any adjudication."""
    plan, _unreadable, _arcs = campaign_plan
    report = closure_report(plan)

    blocked = {gap.demand.study for gap in report.gaps if gap.demand is not None}
    assert len(blocked) == 5, sorted(blocked)

    named = {gap.arc for gap in report.gaps}
    assert named == {f"analysis:{card}" for card in _cards_for(plan, blocked)}

    # Every card closure names is one the campaign labelled inconclusive. No false positives.
    assert _cards_for(plan, blocked) <= INCONCLUSIVE_CARDS
    assert _cards_for(plan, blocked) == {
        "STYLE-RMB",
        "PPE-BON",
        "HACK-FORE",
        "VALUES-CONTEST",
        "T3-FIELD",
    }

    # And each gap names the exact intermediate the campaign's own failure message named.
    missing = {str(gap.missing) for gap in report.gaps}
    assert any(
        "campaign.bias.battery" in m and "armorm" in m and "diagnostic-v3-degradation" in m
        for m in missing
    )
    assert any("campaign.subspace.flat" in m and "skywork-v2-llama31-8b" in m for m in missing)
    assert any("campaign.index.table" in m and "armorm" in m for m in missing)


def _cards_for(plan: Any, study_ids: set[str]) -> set[str]:
    """Card names for a set of spec ids, via the plan's own bindings."""
    cards: set[str] = set()
    for binding in plan.bindings:
        if binding.study in study_ids:
            cards.add(str(binding.subject.extra["card"]))
    return cards


def test_closure_blocks_thirteen_predictions_of_which_four_are_kill_criteria(campaign_plan):
    """The four kills are the sharp end, and the scoreboard shows a blank for every one of them.

    A kill criterion whose metric no arc produces was, under the old adjudication, rendered
    exactly like a criterion that was evaluated and did not fire. Closure sees them at plan time
    and on the same footing as a hypothesis.
    """
    plan, _unreadable, _arcs = campaign_plan
    report = closure_report(plan)

    hypotheses = [g for g in report.gaps if g.demand is not None and g.demand.kind == "hypothesis"]
    kills = [g for g in report.gaps if g.demand is not None and g.demand.kind == "kill"]

    assert len(report.gaps) == 13
    assert len(hypotheses) == 9
    assert len(kills) == 4
    assert {g.demand.owner for g in kills} == {"K-style", "K-ppe", "K-hackfore", "K-contest"}
    for gap in kills:
        assert "kill criterion" in gap.render()

    # The nine hypotheses are, name for name, the rows the campaign's scoreboard marks inconclusive
    # for these five cards. Closure reaches the same verdict off the plan that adjudication reached off the
    # run, which is the whole claim.
    assert {g.demand.owner for g in hypotheses} == {
        "H-style-transfer",
        "H-style-perm",
        "H-style-baseline",
        "H-tail-plateau",
        "H-lowerquantile",
        "H-flag",
        "H-contest",
        "H-contest-vs-ensemble",
        "H-flat",
    }


def test_the_three_closure_cannot_reach_are_named_and_the_reason_is_recorded(campaign_plan):
    """A gate that overclaims is worse than one with a stated limit, so the limit is asserted."""
    plan, unreadable, _arcs = campaign_plan
    report = closure_report(plan)

    blocked_cards = _cards_for(plan, {g.demand.study for g in report.gaps if g.demand})
    unreached = INCONCLUSIVE_CARDS - blocked_cards

    assert unreached == set(NOT_REACHABLE_BY_CLOSURE)
    # HUMP is unreached for a reason this run can see directly: its module does not import.
    assert "HUMP" in unreadable
    assert "FeatureBank" in unreadable["HUMP"]
    # The GAUGE pair is unreached because their failure was not a missing metric at all.
    assert "GAUGE-E19" not in unreadable
    assert "GAUGE-XFAM" not in unreadable


def test_the_campaign_could_not_have_declared_a_closable_plan(campaign_plan):
    """Why the gate is worth building rather than the old code being one annotation away.

    A static read of ``campaign/arcs.py`` finds record sites whose observable or subject is an
    f-string or a local, so the arc's output is legible to a person and to no checker. Until an
    arc says what it produces in a form something can read, there is no plan to close.
    """
    _plan, _unreadable, arcs_module = campaign_plan

    unreadable_sites = unreadable_record_sites(arcs_module)

    # 24 of the 61 record sites in the frozen campaign tree; the other 37 do resolve.
    assert len(unreadable_sites) == 24, len(unreadable_sites)
    assert any("f'" in site or 'f"' in site for site in unreadable_sites)


def test_the_report_hands_the_runner_an_arc_for_every_metric_it_can_answer(campaign_plan):
    """Closure rule 3. The predictions that do close get their producing arc recorded."""
    plan, _unreadable, _arcs = campaign_plan
    report = closure_report(plan)

    # 78 registered predictions across the 27 cards, 65 of them answerable; 49 distinct metric
    # names, because the campaign reuses a metric between a hypothesis and its own kill criterion.
    assert len(report.demands) == 78
    assert len(report.metric_arcs) == 49
    assert all(arc.startswith("analysis:") for arc in report.metric_arcs.values())

    # 40 of the plan's 52 arcs are reached by some prediction. The other 12 are work nothing
    # registered a prediction against, so the cost of the plan does not include them.
    assert len(plan.arcs) == 52
    assert len(report.required_arcs) == 40
    assert len(report.unused_arcs) == 12
    assert report.required_arcs < set(plan.arcs_by_id)

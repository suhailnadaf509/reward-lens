"""Unit tests for `reward_lens.access`: the resolvers under the capability report.

The clauses live in `tests/acceptance/test_w1_6_access.py`. These are the smaller checks
that keep the three resolvers honest one at a time, and most of them exist because the resolver
could plausibly have been written the other way: a probe that assumed rather than called, a walk
that counted a disabled branch, a phase that read the access matrix.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from reward_lens.access import (
    DRY,
    SEED_PROBE_CALLS,
    ProbeBudget,
    RecordFacts,
    classify_leaf,
    classify_substrate,
    is_score_node,
    probe_endpoint,
    record_facts,
    render_access,
    resolution_from_matrix,
    resolve_access,
    resolve_phase,
)
from reward_lens.access.report import (
    CatalogueInstrument,
    _as_names,
    _condition_status,
    _leaves_matching,
    _natural,
    _status_word,
    _substrate_remedy,
    load_instrument_catalogue,
)
from reward_lens.core.envelope import ConditionReading, RegimeCondition, RegimeReading
from reward_lens.core.types import Access, Capability, Component, Phase, Substrate

# ---------------------------------------------------------------------------
# Small fakes
# ---------------------------------------------------------------------------


class Node:
    """A minimal `ScoreNode`: children, a combining rule, and the optional fields."""

    def __init__(self, name, *, children=(), combine=None, substrate=None, enabled=True):
        self.name = name
        self.children = list(children)
        self.combine = combine
        self.substrate = substrate
        self.enabled = enabled


class Counted:
    """A callable that records how many times it was called and with which seeds."""

    def __init__(self, fn):
        self._fn = fn
        self.calls: list[int | None] = []

    def __call__(self, item, *, seed=None):
        self.calls.append(seed)
        return self._fn(item, seed)


class SyntheticRecord:
    n_steps = 401
    n_rollouts = 25_664
    sampling = 1.0
    complete = True
    scores = True
    task_refs = True


# ---------------------------------------------------------------------------
# render_access
# ---------------------------------------------------------------------------


def test_access_renders_flag_names_not_an_integer():
    assert render_access(Access.QUERY | Access.REPLICATE) == "QUERY | REPLICATE"
    assert render_access(Access.NONE) == "NONE"
    assert render_access(Access.RECORD) == "RECORD"


# ---------------------------------------------------------------------------
# record_facts
# ---------------------------------------------------------------------------


def test_record_facts_reads_an_object_and_names_what_it_could_not_read():
    facts = record_facts(SyntheticRecord())
    assert facts.n_steps == 401
    assert facts.n_rollouts == 25_664
    assert facts.sampling == 1.0
    assert facts.taps == frozenset({"scores", "task_refs"})
    assert facts.unread == ()
    assert facts.render() == "401 steps, 25,664 rollouts, sampling 1.0"


def test_a_record_path_is_not_opened_and_says_so(tmp_path: Path):
    facts = record_facts(tmp_path)
    assert facts.is_present
    assert "steps" in facts.unread
    assert facts.n_steps is None, "reading a path must not invent counts"
    assert "not opened" in facts.render()


def test_a_missing_field_is_reported_rather_than_defaulted():
    class Thin:
        n_steps = 10

    facts = record_facts(Thin())
    assert facts.sampling is None
    assert "sampling" in facts.unread
    assert "not read" in facts.render()


def test_no_record_is_not_a_record():
    assert not record_facts(None).is_present
    assert record_facts(RecordFacts(n_steps=3)).n_steps == 3


# ---------------------------------------------------------------------------
# The probe. REPLICATE does not follow from QUERY.
# ---------------------------------------------------------------------------


def test_the_default_budget_makes_no_calls():
    endpoint = Counted(lambda item, seed: 1.0)
    result = probe_endpoint(endpoint, budget=DRY)
    assert endpoint.calls == []
    assert result.calls_made == 0
    assert result.access is Access.NONE
    assert result.verdict == "not_probed"
    assert "REPLICATE" in result.unresolved


def test_an_endpoint_that_ignores_its_seed_is_query_only():
    endpoint = Counted(lambda item, seed: 0.5)
    result = probe_endpoint(endpoint, budget=ProbeBudget(calls=SEED_PROBE_CALLS))
    assert result.access is Access.QUERY
    assert not (result.access & Access.REPLICATE)
    assert result.verdict == "seed_ignored"
    assert result.calls_made == SEED_PROBE_CALLS
    assert endpoint.calls == [0, 1, 0], "the third call is what separates ignored from uncontrolled"


def test_an_endpoint_that_honours_its_seed_is_query_and_replicate():
    endpoint = Counted(lambda item, seed: 0.5 + 0.1 * (seed or 0))
    result = probe_endpoint(endpoint, budget=ProbeBudget(calls=SEED_PROBE_CALLS))
    assert result.access == (Access.QUERY | Access.REPLICATE)
    assert result.verdict == "seed_honoured"
    assert result.note == "endpoint responds; seed is settable"


def test_an_endpoint_with_no_seed_parameter_costs_one_call():
    def no_seed(item):
        return 0.5

    result = probe_endpoint(no_seed, budget=ProbeBudget(calls=SEED_PROBE_CALLS))
    assert result.access is Access.QUERY
    assert result.verdict == "no_seed"
    assert result.calls_made == 1, "a signature with no seed settles it without a second call"


def test_an_endpoint_that_rejects_a_seed_at_runtime_is_query_only():
    class Opaque:
        """No introspectable signature, so the probe has to find out by calling."""

        def score(self, item, **kwargs):
            if "seed" in kwargs:
                raise TypeError("score() got an unexpected keyword argument 'seed'")
            return 0.5

    result = probe_endpoint(Opaque(), budget=ProbeBudget(calls=SEED_PROBE_CALLS))
    assert result.access is Access.QUERY
    assert result.verdict == "no_seed"


def test_variation_the_seed_does_not_control_is_not_replicate():
    """Test-retest is available and a crossed design is not, so this is QUERY."""
    ticker = iter(range(100))

    endpoint = Counted(lambda item, seed: float(next(ticker)))
    result = probe_endpoint(endpoint, budget=ProbeBudget(calls=SEED_PROBE_CALLS))
    assert result.access is Access.QUERY
    assert result.verdict == "seed_uncontrolled"
    assert "does not control" in result.note


def test_an_endpoint_that_raises_is_no_access_and_carries_the_exception():
    def dead(item, *, seed=None):
        raise ConnectionRefusedError("connection refused")

    result = probe_endpoint(dead, budget=ProbeBudget(calls=SEED_PROBE_CALLS))
    assert result.access is Access.NONE
    assert result.verdict == "unreachable"
    assert "ConnectionRefusedError" in result.error


def test_a_budget_too_small_for_the_seed_probe_leaves_replicate_unresolved():
    endpoint = Counted(lambda item, seed: 0.5 + 0.1 * (seed or 0))
    result = probe_endpoint(endpoint, budget=ProbeBudget(calls=1))
    assert result.access is Access.QUERY
    assert result.unresolved == ("REPLICATE",)
    assert result.calls_made == 1
    assert "unresolved rather than absent" in result.note


# ---------------------------------------------------------------------------
# resolve_access
# ---------------------------------------------------------------------------


def test_a_supplied_endpoint_is_not_query_until_it_is_called():
    resolution = resolve_access(record=SyntheticRecord(), grader=lambda item, *, seed=None: 1.0)
    assert not (resolution.of(Component.GRADER) & Access.QUERY)
    assert "not called" in resolution.note_for(Component.GRADER)
    assert resolution.calls_made == 0


def test_probing_from_resolve_access_reaches_replicate_and_reports_the_calls():
    resolution = resolve_access(
        record=SyntheticRecord(),
        grader=lambda item, *, seed=None: 0.5 + 0.1 * (seed or 0),
        probe=ProbeBudget(calls=SEED_PROBE_CALLS),
    )
    assert resolution.of(Component.GRADER) & Access.REPLICATE
    assert resolution.calls_made == SEED_PROBE_CALLS


def test_a_policy_checkpoint_adds_forward_and_leaves_backward_unresolved(tmp_path: Path):
    resolution = resolve_access(record=SyntheticRecord(), policy=tmp_path)
    access = resolution.of(Component.POLICY)
    assert access & Access.FORWARD
    assert not (access & Access.BACKWARD), "BACKWARD needs the weights loaded, which nothing did"
    assert "BACKWARD" in resolution.note_for(Component.POLICY)


def test_no_policy_names_the_flag_that_would_supply_it():
    resolution = resolve_access(record=SyntheticRecord())
    assert (
        resolution.note_for(Component.POLICY) == "no weights supplied; --policy would add FORWARD"
    )


def test_a_declared_flag_is_labelled_as_declared():
    resolution = resolve_access(
        record=SyntheticRecord(), declared={Component.GRADER: Access.REPLICATE}
    )
    assert resolution.of(Component.GRADER) & Access.REPLICATE
    assert "declared by the caller, not probed" in resolution.note_for(Component.GRADER)


def test_task_refs_give_record_not_query():
    resolution = resolve_access(record=SyntheticRecord())
    assert resolution.of(Component.TASK) is Access.RECORD
    assert "QUERY" in [f for c in resolution.components for f in c.unresolved]


def test_a_bare_matrix_wraps_without_pretending_it_was_resolved():
    wrapped = resolution_from_matrix({Component.GRADER: Access.QUERY})
    assert wrapped.of(Component.GRADER) is Access.QUERY
    assert wrapped.note_for(Component.GRADER) == "supplied by the caller"


# ---------------------------------------------------------------------------
# Substrate
# ---------------------------------------------------------------------------


def test_a_python_function_is_a_program_because_it_has_source():
    def verifier(item):
        return 1.0

    substrate, why = classify_leaf(verifier)
    assert substrate is Substrate.PROGRAM
    assert "source" in why


def test_capabilities_classify_the_two_neural_kinds():
    class Scalar:
        caps = Capability.SCORES | Capability.LINEAR_READOUT

    class Judge:
        caps = Capability.SCORES | Capability.GENERATIVE

    assert classify_leaf(Scalar())[0] is Substrate.NEURAL_SCALAR
    assert classify_leaf(Judge())[0] is Substrate.NEURAL_GEN


def test_rater_identity_is_human_and_an_aggregation_rule_is_procedural():
    assert classify_leaf({"raters": ["a", "b"]})[0] is Substrate.HUMAN
    assert classify_leaf({"rubric": {"clarity": 0.5}})[0] is Substrate.PROCEDURAL


def test_an_unidentifiable_grader_refuses_with_a_remedy_rather_than_guessing():
    reading = classify_substrate(object())
    assert reading.substrate is None
    assert reading.render() == "UNRESOLVED"
    assert reading.refusal is not None
    assert "declare it" in reading.refusal.remedy


def test_one_leaf_is_not_a_composition():
    tree = Node("root", combine="identity", children=[Node("v", substrate=Substrate.PROGRAM)])
    reading = classify_substrate(tree)
    assert reading.substrate is Substrate.PROGRAM
    assert "not a composition" in reading.note


def test_a_disabled_branch_does_not_count_as_a_live_leaf():
    tree = Node(
        "root",
        combine="weighted_sum",
        children=[
            Node("verifier", substrate=Substrate.PROGRAM),
            Node(
                "off",
                combine="mean",
                enabled=False,
                children=[
                    Node("j1", substrate=Substrate.NEURAL_GEN),
                    Node("j2", substrate=Substrate.NEURAL_GEN),
                ],
            ),
            Node("rubric", substrate=Substrate.PROCEDURAL),
        ],
    )
    reading = classify_substrate(tree)
    assert len(reading.leaves) == 4
    assert len(reading.live_leaves) == 2
    assert reading.counts() == {Substrate.PROGRAM: 1, Substrate.PROCEDURAL: 1}
    assert reading.render() == "COMPOSITE (2 leaves: 1 PROGRAM, 1 PROCEDURAL)"


def test_an_unclassified_leaf_is_named_and_the_tree_is_still_composite():
    tree = Node(
        "root",
        combine="min",
        children=[Node("known", substrate=Substrate.PROGRAM), Node("mystery")],
    )
    reading = classify_substrate(tree)
    assert reading.substrate is Substrate.COMPOSITE
    assert reading.unclassified == ("mystery",)
    assert reading.refusal is not None
    assert "leaf_classifier" in reading.refusal.remedy


def test_a_cycle_stops_the_walk_and_is_reported():
    a = Node("a", combine="sum")
    b = Node("b", combine="sum", children=[a])
    a.children.append(b)
    reading = classify_substrate(a)
    assert reading.truncated


def test_is_score_node_rejects_things_that_merely_have_the_attribute_names():
    assert not is_score_node("children")
    assert not is_score_node(object())
    assert is_score_node(Node("n"))


def test_a_supplied_leaf_classifier_is_used():
    tree = Node("root", combine="sum", children=[Node("x"), Node("y")])
    reading = classify_substrate(
        tree, leaf_classifier=lambda node: (Substrate.HUMAN, "the caller says so")
    )
    assert reading.counts() == {Substrate.HUMAN: 2}


def test_a_leaf_instrument_is_pointed_at_the_leaf_rather_than_told_it_will_never_apply():
    """COMPOSITE admits the other substrates on its leaves, so this is not a dead end."""
    tree = Node(
        "root",
        combine="weighted_sum",
        children=[
            Node("unit_tests", substrate=Substrate.PROGRAM),
            Node("style_judge", substrate=Substrate.NEURAL_GEN),
        ],
    )
    reading = classify_substrate(tree)
    assert _leaves_matching(frozenset({Substrate.PROGRAM}), reading) == ("unit_tests",)

    matched = _substrate_remedy(frozenset({Substrate.PROGRAM}), reading, "an instrument")
    assert "point it at a leaf" in matched
    assert "unit_tests" in matched

    unmatched = _substrate_remedy(frozenset({Substrate.HUMAN}), reading, "an instrument")
    assert "different kind of object" in unmatched
    assert "point it at a leaf" not in unmatched


def test_a_non_composite_grader_has_no_leaves_to_be_pointed_at():
    reading = classify_substrate({"rubric": {"clarity": 1.0}})
    assert reading.substrate is Substrate.PROCEDURAL
    assert _leaves_matching(frozenset({Substrate.PROGRAM}), reading) == ()


# ---------------------------------------------------------------------------
# Phase
# ---------------------------------------------------------------------------


def test_phase_does_not_take_an_access_matrix():
    """Phase is not derivable from access, and the signature is the enforcement."""
    params = set(inspect.signature(resolve_phase).parameters)
    assert "access" not in params
    assert params == {"record", "live", "artifact_only", "declared"}


def test_the_four_phases_resolve_from_what_exists():
    class Open(SyntheticRecord):
        complete = False

    assert resolve_phase().phase is Phase.PRE_RUN
    assert resolve_phase(record=SyntheticRecord()).phase is Phase.POST_RUN
    assert resolve_phase(record=Open()).phase is Phase.IN_RUN
    assert resolve_phase(live=True).phase is Phase.IN_RUN
    assert resolve_phase(artifact_only=True).phase is Phase.DEPLOYED


def test_an_open_record_beats_a_caller_who_did_not_say_it_was_live():
    class Open(SyntheticRecord):
        complete = False

    resolution = resolve_phase(record=Open(), live=False)
    assert resolution.phase is Phase.IN_RUN
    assert "still being written" in resolution.note


def test_a_declared_phase_wins_and_is_marked_declared():
    resolution = resolve_phase(record=SyntheticRecord(), declared="PRE_RUN")
    assert resolution.phase is Phase.PRE_RUN
    assert resolution.declared
    assert resolution.render() == "PRE_RUN"


def test_a_phase_nobody_supplied_renders_as_unresolved():
    from reward_lens.access.report import _as_phase_resolution

    placeholder = _as_phase_resolution(None)
    assert placeholder.render() == "UNRESOLVED"
    assert not placeholder.enforced


# ---------------------------------------------------------------------------
# The catalogue reader
# ---------------------------------------------------------------------------


def test_the_open_string_normalises_to_nothing_rather_than_to_four_characters():
    """E14: iterating the bare string OPEN yields 'O', 'P', 'E', 'N'."""
    assert _as_names("OPEN") == ()
    assert _as_names(None) == ()
    assert _as_names(["a", "b"]) == ("a", "b")


def test_the_catalogue_loads_every_instrument():
    rows = load_instrument_catalogue()
    # 85 catalogue rows, plus the four of series N that the registry carries no quantity for.
    # See E23.
    assert len(rows) == 95
    assert all(isinstance(r, CatalogueInstrument) for r in rows)
    a1 = next(r for r in rows if r.id == "A1")
    # A1 claims two quantities. The correction that stopped multiplying Kish's shape
    # factor into the effective size did not delete the shape factor, it separated it: A1 still
    # measures it and now reports it beside the reading under its own id, because it is a property
    # of the reward distribution rather than of the grader.
    assert a1.quantities == ("grader.effective_group_size", "run.group_shape_factor")
    assert RegimeCondition.GROUP_NONDEGENERATE in a1.envelope_requires


def test_a_row_with_no_status_reads_as_open_rather_than_as_blank():
    """`status` is the schedule signal, and it has to carry the absent case too."""
    assert _status_word("built") == "built"
    assert _status_word(" planned ") == "planned"
    assert _status_word("") == "OPEN"
    assert _status_word("OPEN") == "OPEN"


def test_instrument_ids_sort_numerically():
    assert sorted(["D10", "D2", "D1"], key=_natural) == ["D1", "D2", "D10"]


# ---------------------------------------------------------------------------
# Conditions: measured-and-failed is not the same as nobody-measured-it
# ---------------------------------------------------------------------------


def test_an_unmeasured_condition_is_unmeasured_not_failed():
    regime = RegimeReading(
        conditions={
            RegimeCondition.NEAR_POLICY: ConditionReading(
                RegimeCondition.NEAR_POLICY, True, 2.0, 8.0
            )
        }
    )
    failed, unmeasured = _condition_status(
        frozenset({RegimeCondition.NEAR_POLICY, RegimeCondition.QUASI_STATIC}), regime
    )
    assert failed == []
    assert unmeasured == [RegimeCondition.QUASI_STATIC]


def test_a_condition_measured_as_indeterminate_counts_as_failed():
    regime = RegimeReading(
        conditions={
            RegimeCondition.QUASI_STATIC: ConditionReading(
                RegimeCondition.QUASI_STATIC, None, float("nan"), float("nan")
            )
        }
    )
    failed, unmeasured = _condition_status(frozenset({RegimeCondition.QUASI_STATIC}), regime)
    assert failed == [RegimeCondition.QUASI_STATIC]
    assert unmeasured == []


@pytest.mark.parametrize("regime", [None])
def test_with_no_reading_at_all_every_condition_is_unmeasured(regime):
    failed, unmeasured = _condition_status(frozenset({RegimeCondition.ABOVE_LOD}), regime)
    assert failed == []
    assert unmeasured == [RegimeCondition.ABOVE_LOD]

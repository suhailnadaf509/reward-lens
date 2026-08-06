"""Unit tests for D3, `verifier.metamorphic_violations`.

Three things are worth testing separately from the clause. The transformations have to
actually preserve what they say they preserve, because a violation is only as good as that claim.
The applicability accounting has to distinguish a relation that fired and passed from one that
never fired, because those look the same on a count and are opposite findings. And the shrink has
to come from `hypothesis.find` rather than from the census, because a reproducer nobody can
minimise is a reproducer nobody debugs.
"""

from __future__ import annotations

from random import Random

import pytest

from reward_lens.core.reading import Refusal
from reward_lens.measure.base import lint_instrument
from reward_lens.verifier import Rollout
from reward_lens.verifier.metamorphic import (
    KIND_ASSERTION,
    MetamorphicRelation,
    MetamorphicViolations,
    QuerySubject,
    Transformation,
    answer_text_relations,
    identity_relation,
    looks_like_source,
    python_source_relations,
    resolve_grader,
)

SOLUTION = '''\
def solve(n):
    """Double it."""
    doubled = n * 2
    spare = n + 1
    return doubled
'''


def source_corpus(k: int = 4) -> list[Rollout]:
    return [Rollout(id=f"s{i}", inputs={"solution": SOLUTION}) for i in range(k)]


# ---------------------------------------------------------------------------
# The dependency, pinned by file rather than by name
# ---------------------------------------------------------------------------


def test_hypothesis_is_the_package_we_think_it_is():
    """Several plausible names on PyPI install different software.

    Also re-verifies the one claim D3 leans on. `find(strategy, predicate)` returns the shrunk
    example as a **Python object**: `101`, an `int`, for `x > 100`. The `@given` route returns the
    counterexample only as a note string, which is why `find` is the one used here.
    """
    import hypothesis
    from hypothesis import find
    from hypothesis import strategies as st

    assert hypothesis.__file__.endswith("hypothesis/__init__.py")
    assert hypothesis.__version__ >= "6.100"

    found = find(st.integers(), lambda x: x > 100)
    assert found == 101
    assert type(found) is int


# ---------------------------------------------------------------------------
# The transformations preserve what they claim
# ---------------------------------------------------------------------------


def test_the_rename_preserves_behaviour_and_every_other_character():
    """A rename done through `ast.unparse` would also strip the docstring and reflow the file."""
    from reward_lens.verifier.metamorphic import _rename_local

    renamed = _rename_local(SOLUTION, Random(3))
    assert renamed != SOLUTION
    assert '"""Double it."""' in renamed, "the docstring survives a token-level rewrite"
    assert renamed.count("\n") == SOLUTION.count("\n")

    namespace_before: dict = {}
    namespace_after: dict = {}
    exec(compile(SOLUTION, "<before>", "exec"), namespace_before)
    exec(compile(renamed, "<after>", "exec"), namespace_after)
    for n in (-3, 0, 7, 41):
        assert namespace_before["solve"](n) == namespace_after["solve"](n)


def test_the_rename_leaves_attributes_alone():
    source = "def f(xs):\n    count = 0\n    count = count + xs.count(1)\n    return count\n"
    from reward_lens.verifier.metamorphic import _rename_local

    renamed = _rename_local(source, Random(0))
    assert ".count(" in renamed, "an attribute named like the local must not be renamed"
    assert "count = 0" not in renamed


def test_the_reorder_only_swaps_statements_with_disjoint_read_and_write_sets():
    from reward_lens.verifier.metamorphic import _reorder_statements

    dependent = "def f(n):\n    a = n + 1\n    b = a * 2\n    return b\n"
    assert _reorder_statements(dependent, Random(0)) == dependent, (
        "b reads a, so the pair is not independent and must not be swapped"
    )

    effectful = "def f(n):\n    a = g(n)\n    b = h(n)\n    return a + b\n"
    assert _reorder_statements(effectful, Random(0)) == effectful, (
        "both right-hand sides are calls, which may have effects"
    )

    independent = "def f(n):\n    a = n + 1\n    b = n * 2\n    return a + b\n"
    swapped = _reorder_statements(independent, Random(0))
    assert swapped != independent
    assert swapped.index("b = n * 2") < swapped.index("a = n + 1")


def test_a_source_relation_declines_to_touch_something_that_is_not_source():
    """`42` is a syntactically valid Python module, and appending a comment to it is not harmless."""
    assert looks_like_source(SOLUTION)
    assert not looks_like_source("42")
    assert not looks_like_source("-7")
    assert not looks_like_source("The answer is 3/4")
    assert not looks_like_source("x = 8"), "a single assignment is an answer, not a module"


# ---------------------------------------------------------------------------
# Applicability accounting
# ---------------------------------------------------------------------------


def test_a_relation_that_fires_and_passes_is_not_the_same_as_one_that_never_fired():
    """The distinction the whole rate depends on."""

    def blind(solution: str) -> float:
        return 1.0

    reading = MetamorphicViolations(blind, source_corpus(), python_source_relations()).estimate()
    report = reading.value
    assert report.pairs_applicable == 16
    assert report.violations == 0
    assert report.violation_rate == 0.0
    assert report.inapplicable_relations == ()

    reading = MetamorphicViolations(
        blind, [Rollout(id="a", inputs={"solution": "42"})], python_source_relations()
    ).estimate()
    report = reading.value
    assert report.pairs_applicable == 0
    assert len(report.inapplicable_relations) == 4
    assert report.violation_rate != report.violation_rate, "NaN, not zero"


def test_the_rate_divides_by_applicable_pairs_so_a_dead_relation_cannot_dilute_it():
    """Two relations, one live and always violating, one that never fires. The rate is 1.0."""

    def picky(solution: str) -> float:
        return 1.0 if "# " not in solution else 0.0

    live, *_ = python_source_relations()[1:2]
    dead = MetamorphicRelation(
        name="never_fires",
        transformation=Transformation(
            name="noop",
            on="absent_key",
            rewrite=lambda text, rng: text + "!",
            preserves="nothing, because the key it names is not in the inputs",
        ),
    )
    report = MetamorphicViolations(picky, source_corpus(2), [live, dead]).estimate().value
    assert report.violation_rate == 1.0
    assert report.inapplicable_relations == ("never_fires",)


# ---------------------------------------------------------------------------
# Reproducers
# ---------------------------------------------------------------------------


def test_the_shrunk_reproducer_comes_from_find_and_reruns_against_a_bare_callable():
    def picky(solution: str) -> float:
        return 0.0 if "# note" in solution else 1.0

    report = (
        MetamorphicViolations(picky, source_corpus(6), python_source_relations()[1:2])
        .estimate()
        .value
    )
    assert report.violations == 6
    shrunk = report.shrunk_reproducers
    assert len(shrunk) == 1
    reproducer = shrunk[0]
    assert reproducer.source == "hypothesis.find"
    assert reproducer.shrunk is True
    assert reproducer.delta == -1.0
    assert reproducer.rerun(picky) == (1.0, 0.0)
    assert reproducer.still_violates(picky)
    assert "comment" in reproducer.preserves
    assert reproducer.assertion == KIND_ASSERTION["invariance"]


def test_a_reproducer_holds_its_inputs_rather_than_a_pointer_into_the_corpus():
    def picky(solution: str) -> float:
        return 0.0 if "# note" in solution else 1.0

    corpus = source_corpus(2)
    report = MetamorphicViolations(picky, corpus, python_source_relations()[1:2]).estimate().value
    corpus.clear()
    reproducer = report.reproducers[0]
    assert reproducer.inputs_before["solution"] == SOLUTION
    assert reproducer.still_violates(picky), "the reproducer outlives the corpus it came from"


# ---------------------------------------------------------------------------
# Refusals and declarations
# ---------------------------------------------------------------------------


def test_an_empty_corpus_is_refused_with_a_remedy():
    reading = MetamorphicViolations(lambda solution: 1.0, [], python_source_relations()).estimate()
    assert isinstance(reading, Refusal)
    assert reading.reason.name == "ACCESS_INSUFFICIENT"
    assert "denominator" in reading.remedy


def test_a_corpus_the_grader_accepts_nothing_from_is_refused_rather_than_scored():
    reading = MetamorphicViolations(
        lambda solution: 0.0, source_corpus(), python_source_relations()
    ).estimate()
    assert isinstance(reading, Refusal)
    assert reading.reason.name == "ACCESS_INSUFFICIENT"
    assert reading.statistics["max_score"] == 0.0


def test_the_instrument_declares_the_twelve_and_passes_lint():
    inst = MetamorphicViolations(lambda solution: 1.0, source_corpus(), answer_text_relations())
    assert lint_instrument(inst) == []
    assert inst.quantity == "verifier.metamorphic_violations"
    assert inst.invariance == "trivial"
    assert inst.baselines == ("identity_transformation",)
    assert inst.envelope is not None and inst.envelope.requires
    assert inst.deviations, "an instrument that departs from its theory anchor has to say how"


def test_the_rung_is_taken_from_the_relations_rather_than_declared():
    source = MetamorphicViolations(lambda solution: 1.0, source_corpus(), python_source_relations())
    answers = MetamorphicViolations(lambda solution: 1.0, source_corpus(), answer_text_relations())
    assert source.rung == 0
    assert answers.rung == 1


def test_the_baseline_relation_is_a_real_object_and_not_only_a_string():
    baseline = identity_relation()
    rollout = Rollout(id="a", inputs={"response": "42"})
    assert baseline.transformation.apply(rollout, Random(0)).inputs == rollout.inputs


def test_a_query_only_grader_is_fingerprinted_by_name_and_says_so():
    """A card that cannot say which grader it measured is not evidence about anything."""

    def grade(response: str) -> float:
        return 1.0

    fn, subject, name = resolve_grader(grade)
    assert fn is grade
    assert name == "grade"
    assert isinstance(subject, QuerySubject)
    assert subject.fingerprint_basis == "qualname"
    assert subject.meta.lineage["fingerprint_basis"] == "qualname"


def test_preflight_costs_the_run_without_calling_the_grader():
    calls = {"n": 0}

    def counted(solution: str) -> float:
        calls["n"] += 1
        return 1.0

    from reward_lens.measure.base import Context

    inst = MetamorphicViolations(counted, source_corpus(10), python_source_relations())
    plan = inst.preflight(Context(signal=inst.subject))
    assert plan.ok
    assert calls["n"] == 0, "preflight does NO compute"
    assert plan.cost.calls == 10 * (2 + 2 * 4)
    assert "access" in plan.unchecked


@pytest.mark.parametrize("kind", sorted(KIND_ASSERTION))
def test_every_relation_kind_has_a_written_assertion(kind):
    assert KIND_ASSERTION[kind].startswith("score(")

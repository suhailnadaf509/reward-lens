"""D9: the control-flow graph, the clause map, and the trusted-input analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

from reward_lens.core.invariance import check_invariance
from reward_lens.measure.base import lint_instrument
from reward_lens.verifier import VerifierUnderTest
from reward_lens.verifier.static import (
    StaticStructure,
    TrustedInputs,
    analyse_source,
    analyse_trusted_inputs,
    arc_key,
    control_flow,
    dominators,
    extract_clauses,
    guard_gated_arcs,
    map_arcs_to_clauses,
)

GRADER = '''"""A grader with a timeout-only branch and a marked rubric."""


def grade(payload, reference, elapsed=0.0, timeout=5.0):
    if elapsed > timeout:  # rubric: timed_out
        if payload is None:
            return 0.0
        return 0.25
    answer = payload["answer"]
    if answer == reference:  # rubric: exact
        return 1.0
    if len(answer) > 3:  # rubric: partial
        return 0.5
    return 0.0
'''


@pytest.fixture
def grader(tmp_path: Path) -> VerifierUnderTest:
    path = tmp_path / "grader.py"
    path.write_text(GRADER, encoding="utf-8")
    return VerifierUnderTest(source_path=path, entrypoint="grade")


# ---------------------------------------------------------------------------
# The package-name traps. Several plausible names on PyPI install different software.
# ---------------------------------------------------------------------------


def test_third_party_imports_are_the_packages_we_think_they_are() -> None:
    import cosmic_ray
    import coverage
    import libcst

    assert "site-packages" in coverage.__file__
    assert coverage.__version__.startswith("7.")
    assert Path(cosmic_ray.__file__).parent.name == "cosmic_ray"
    assert "site-packages" in libcst.__file__

    # `reward_lens.verifier.coverage` must not shadow the third-party `coverage` inside itself.
    from reward_lens.verifier import coverage as our_coverage

    assert our_coverage.__name__ == "reward_lens.verifier.coverage"
    assert our_coverage is not coverage


def test_libcst_needs_the_wrapper_not_the_module() -> None:
    """E9: `wrapper.visit()` is required; `module.visit()` raises."""
    import libcst as cst
    from libcst.metadata import PositionProvider

    class _V(cst.CSTVisitor):
        METADATA_DEPENDENCIES = (PositionProvider,)

        def visit_If(self, node: object) -> None:
            self.get_metadata(PositionProvider, node)

    module = cst.parse_module(GRADER)
    with pytest.raises((AttributeError, KeyError)):
        module.visit(_V())


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


def test_control_flow_matches_coverages_own_analysis(grader: VerifierUnderTest) -> None:
    import coverage
    from coverage.python import PythonFileReporter

    arcs, exits, statements = control_flow(grader.source_path)
    cov = coverage.Coverage(branch=True, data_file=None)
    reporter = PythonFileReporter(str(grader.source_path), cov)
    assert arcs == frozenset(reporter.arcs())
    assert exits == dict(reporter.exit_counts())
    assert statements == frozenset(reporter.lines())
    assert len([line for line, n in exits.items() if n > 1]) == 4


def test_dominators_on_a_diamond() -> None:
    arcs = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]
    dom = dominators(arcs, roots={1})
    assert dom[1] == frozenset({1})
    assert dom[4] == frozenset({1, 4})
    assert dom[5] == frozenset({1, 4, 5})
    assert 2 not in dom[4]


def test_dominators_handle_an_unreachable_node() -> None:
    dom = dominators([(1, 2), (9, 10)], roots={1})
    assert dom[10] == frozenset({9, 10})


def test_the_timeout_branch_is_reachable_only_past_the_guard(grader: VerifierUnderTest) -> None:
    structure, _ = analyse_source(grader)
    assert structure.guard_lines, "the `if elapsed > timeout` line should be recognised as a guard"
    assert structure.guard_gated_arcs, "arcs inside the timeout branch should be gated"
    gated_heads = {arc[0] for arc in structure.guard_gated_arcs}
    inner_if = GRADER.splitlines().index("        if payload is None:") + 1
    assert inner_if in gated_heads


def test_a_grader_with_no_guard_reports_no_gated_arcs(tmp_path: Path) -> None:
    path = tmp_path / "plain.py"
    path.write_text("def grade(x):\n    if x > 1:\n        return 1.0\n    return 0.0\n", "utf-8")
    structure, _ = analyse_source(VerifierUnderTest(source_path=path))
    assert structure.guard_lines == ()
    assert structure.guard_gated_arcs == ()


# ---------------------------------------------------------------------------
# The clause map, which is what turns an arc into a sentence
# ---------------------------------------------------------------------------


def test_marker_comments_name_the_clauses() -> None:
    clauses = extract_clauses(GRADER, "grade")
    names = [c.name for c in clauses]
    # Four decision points, three of them marked. The unmarked one is the `if payload is None`
    # nested inside the timeout guard: a trailing marker names its own line only, so the guard's
    # marker does not leak down into it.
    assert names == ["timed_out", "grade.clause2", "exact", "partial"]
    assert [c.named_by_marker for c in clauses] == [True, False, True, True]
    assert [c.ordinal for c in clauses] == [1, 2, 3, 4]


def test_a_marker_on_its_own_line_names_the_statement_below_it() -> None:
    src = "def grade(x):\n    # rubric: gate\n    if x:\n        return 1.0\n    return 0.0\n"
    (clause,) = extract_clauses(src, "grade")
    assert clause.name == "gate"
    assert clause.named_by_marker


def test_unmarked_ifs_get_an_ordinal_naming_their_function() -> None:
    src = "def grade(x):\n    if x:\n        return 1.0\n    return 0.0\n"
    (clause,) = extract_clauses(src, "grade")
    assert clause.name == "grade.clause1"
    assert clause.named_by_marker is False


def test_clauses_carry_the_arc_that_means_they_fired(grader: VerifierUnderTest) -> None:
    structure, _ = analyse_source(grader)
    for clause in structure.clauses:
        assert clause.enter_arc is not None, f"{clause.name} has no enter arc"
        assert clause.enter_arc[0] == clause.test_line


def test_every_branch_arc_maps_to_a_clause(grader: VerifierUnderTest) -> None:
    structure, _ = analyse_source(grader)
    branch_lines = set(structure.branch_lines)
    mapped = 0
    for arc in structure.arcs:
        if arc[0] in branch_lines:
            assert arc_key(arc) in structure.arc_clause, f"{arc} is unnamed"
            mapped += 1
    assert mapped >= 8


def test_the_innermost_clause_wins() -> None:
    src = (
        "def grade(x, y):\n"
        "    if x:  # rubric: outer\n"
        "        if y:  # rubric: inner\n"
        "            return 1.0\n"
        "    return 0.0\n"
    )
    clauses = extract_clauses(src, "grade")
    mapping = map_arcs_to_clauses([(2, 3), (3, 4)], clauses)
    assert mapping[arc_key((2, 3))] == "outer"
    assert mapping[arc_key((3, 4))] == "inner"


def test_guard_gated_arcs_is_empty_without_guards() -> None:
    assert guard_gated_arcs([(1, 2), (2, 3)], guard_lines=[], roots=[1]) == ()


# ---------------------------------------------------------------------------
# Trusted inputs
# ---------------------------------------------------------------------------


def _trusted(source: str, tmp_path: Path, rung: int = 2) -> object:
    path = tmp_path / "t.py"
    path.write_text(source, encoding="utf-8")
    _, report = analyse_source(VerifierUnderTest(source_path=path), rung=rung)
    return report


def test_an_unguarded_field_read_is_trusted(tmp_path: Path) -> None:
    report = _trusted("def grade(payload):\n    return float(payload['answer'])\n", tmp_path)
    assert [a.field_name for a in report.trusted] == ["answer"]
    assert report.n_trusted == 1


def test_a_membership_test_validates_the_field(tmp_path: Path) -> None:
    report = _trusted(
        "def grade(payload):\n"
        "    if 'answer' not in payload:\n"
        "        return 0.0\n"
        "    return float(payload['answer'])\n",
        tmp_path,
    )
    assert report.n_trusted == 0
    assert any("membership" in a.validated_by for a in report.accesses)


def test_a_try_except_validates_the_reads_inside_it(tmp_path: Path) -> None:
    report = _trusted(
        "def grade(payload):\n"
        "    try:\n"
        "        return float(payload['answer'])\n"
        "    except (KeyError, ValueError):\n"
        "        return 0.0\n",
        tmp_path,
    )
    assert report.n_trusted == 0


def test_get_with_a_default_validates_and_get_without_one_does_not(tmp_path: Path) -> None:
    with_default = _trusted("def grade(p):\n    return p.get('a', 0.0)\n", tmp_path)
    assert with_default.n_trusted == 0
    bare = _trusted("def grade(p):\n    return p.get('a')\n", tmp_path)
    assert bare.n_trusted == 1


def test_rung_2_sees_a_field_read_through_an_alias(tmp_path: Path) -> None:
    source = "def grade(payload):\n    data = payload\n    return float(data['answer'])\n"
    at_rung_2 = _trusted(source, tmp_path, rung=2)
    at_rung_1 = _trusted(source, tmp_path, rung=1)
    assert at_rung_2.n_trusted == 1
    assert at_rung_1.n_trusted == 0, "rung 1 cannot resolve the alias, and says so in its notes"
    assert any("alias" in note for note in at_rung_1.notes)


def test_a_missing_entrypoint_reports_it_rather_than_returning_zero(tmp_path: Path) -> None:
    path = tmp_path / "e.py"
    path.write_text("def scorer(x):\n    return x['a']\n", encoding="utf-8")
    report = analyse_trusted_inputs(
        path.read_text("utf-8"), "grade", rung=1, arcs=(), roots=(), source_path=str(path)
    )
    assert report.accesses == ()
    assert report.parameters == ()
    assert "no function named 'grade'" in report.notes[0]


# ---------------------------------------------------------------------------
# Rungs
# ---------------------------------------------------------------------------


def test_rung_0_reports_no_graph_rather_than_an_empty_one(grader: VerifierUnderTest) -> None:
    structure, _ = analyse_source(grader, rung=0)
    assert structure.arcs == ()
    assert structure.clauses == ()
    assert structure.branches > 0
    assert any("No control-flow graph" in note for note in structure.notes)


# ---------------------------------------------------------------------------
# The instrument contract
# ---------------------------------------------------------------------------


def test_both_d9_instruments_lint_clean(grader: VerifierUnderTest) -> None:
    assert lint_instrument(StaticStructure(grader)) == []
    assert lint_instrument(TrustedInputs(grader)) == []


def test_the_generated_invariance_test_passes(grader: VerifierUnderTest) -> None:
    instrument = StaticStructure(grader)
    report = check_invariance(
        instrument,
        instrument.invariance if instrument.invariance != "none" else "trivial",
        ctx=None,
        run=lambda inst, _payload: float(analyse_source(grader)[0].branches),
    )
    assert report.passed
    assert "trivial group has no generators" in (report.skipped or "")

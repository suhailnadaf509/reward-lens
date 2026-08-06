"""Acceptance: coverage with per-rollout attribution, survivors with spans and diffs, and
the control-flow graph that maps arcs back onto rubric clauses.

Two subjects, and they are different tests. The fixture below is a math-answer grader written here:
a parse layer, a normalisation layer, a threshold, and one branch reachable only when a timeout
fires. It proves the code runs and it is the only place the acceptance conditions can be asserted
exactly, because it is the only verifier whose answer key is known.

The second subject is `is_equiv` from `hendrycks/math`, the answer-equivalence checker the MATH
benchmark ships and most open RLVR math pipelines still call. A synthetic test proves the code
runs; a real verifier proves the instrument measures. The real run is fetched over the network and
skipped without it, which is a documented gap rather than a pass.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request

import pytest

from reward_lens.core.reading import Refusal
from reward_lens.measure.base import lint_instrument
from reward_lens.verifier import (
    ListCorpus,
    Rollout,
    VerifierUnderTest,
    is_sensitive,
    program_context,
    redact,
)
from reward_lens.verifier.coverage import CoverageReading, DecisionCoverage, measure_coverage
from reward_lens.verifier.mutate import (
    MutationReading,
    MutationScore,
    SurvivingMutants,
    run_mutation,
)
from reward_lens.verifier.static import StaticStructure, TrustedInputs, analyse_source, arc_key

# ---------------------------------------------------------------------------
# The fixture verifier: parse, normalise, threshold, and a timeout-only arm.
# ---------------------------------------------------------------------------

MATH_GRADER = '''"""A math-answer grader. Parse the number out, normalise it, threshold the gap."""

import re

_NUMBER = re.compile(r"-?\\d+(?:\\.\\d+)?")


def parse(text):
    """The parse layer: pull a number out of free text, or fail."""
    if text is None:
        return None
    match = _NUMBER.search(text)
    if match is None:
        return None
    return float(match.group(0))


def normalise(value, scale):
    """The normalisation layer: divide through by the problem's scale."""
    if scale == 0:
        return value
    return value / scale


def grade(response, reference, scale=1.0, elapsed=0.0, timeout=5.0):
    if elapsed > timeout:  # rubric: timed_out
        if response is None:
            return 0.0
        return 0.25
    parsed = parse(response)
    if parsed is None:  # rubric: unparseable
        return 0.0
    got = normalise(parsed, scale)
    want = normalise(float(reference), scale)
    gap = abs(got - want)
    if gap < 1e-6:  # rubric: exact_match
        return 1.0
    if gap < 0.5:  # rubric: near_match
        return 0.5
    return 0.0
'''


@pytest.fixture(scope="module")
def grader(tmp_path_factory: pytest.TempPathFactory) -> VerifierUnderTest:
    path = tmp_path_factory.mktemp("w34a") / "math_grader.py"
    path.write_text(MATH_GRADER, encoding="utf-8")
    return VerifierUnderTest(source_path=path, entrypoint="grade")


@pytest.fixture(scope="module")
def corpus() -> ListCorpus:
    """Sixteen rollouts with real score spread. None of them times out."""
    cases = [
        ("42", 42.0),
        ("the answer is 42", 42.0),
        ("42.0", 42.0),
        ("42.1", 42.0),
        ("-7", -7.0),
        ("-7.2", -7.0),
        ("100", 3.0),
        ("0", 0.0),
        ("no number here", 5.0),
        ("", 5.0),
        ("3.14159", 3.14159),
        ("3.2", 3.14159),
        ("1000000", 1.0),
        ("x = 12", 12.0),
        ("12 apples", 12.0),
        ("-0.5", 0.5),
    ]
    return ListCorpus.of(
        [
            Rollout(id=f"r{i:02d}", inputs={"response": text, "reference": ref})
            for i, (text, ref) in enumerate(cases)
        ]
    )


@pytest.fixture(scope="module")
def coverage_reading(grader, corpus) -> CoverageReading:
    reading = measure_coverage(grader, corpus, rung=2, baseline_resamples=64, seed=0)
    assert isinstance(reading, CoverageReading), reading
    return reading


@pytest.fixture(scope="module")
def mutation_reading(grader, corpus, coverage_reading) -> MutationReading:
    reading = run_mutation(grader, corpus, rung=2, coverage=coverage_reading, timeout=30.0)
    assert isinstance(reading, MutationReading), reading
    return reading


# ---------------------------------------------------------------------------
# Acceptance 1: coverage with per-rollout attribution
# ---------------------------------------------------------------------------


def test_coverage_attributes_arcs_to_the_rollout_that_took_them(coverage_reading, corpus) -> None:
    per_rollout = coverage_reading.per_rollout_arcs
    assert set(per_rollout) == {r.id for r in corpus}
    assert all(arcs for arcs in per_rollout.values())
    distinct = {frozenset(map(tuple, arcs)) for arcs in per_rollout.values()}
    assert len(distinct) > 1, "different rollouts must take different paths through the grader"

    union: set[tuple[int, int]] = set()
    for arcs in per_rollout.values():
        union |= {tuple(a) for a in arcs}
    covered = {tuple(a) for a in coverage_reading.uncovered_branch_arcs}
    assert not (union & covered), "an arc some rollout took cannot also be uncovered"


def test_the_mandatory_baseline_is_computed_and_reported(coverage_reading) -> None:
    """A random rollout sample of the same size, which is what tests corpus redundancy."""
    baselines = coverage_reading.baselines
    assert "random_rollout_sample_same_size" in baselines
    base = baselines["random_rollout_sample_same_size"]
    assert 0.0 < base <= coverage_reading.branch_fraction + 1e-12
    assert baselines["saturation_size"] >= 1.0
    assert 0.0 <= baselines["redundant_fraction"] < 1.0
    assert "a random sample of the same size" in coverage_reading.render()


def test_the_instrument_declares_that_baseline_and_lints_clean(grader, corpus) -> None:
    instrument = DecisionCoverage(grader, corpus)
    assert lint_instrument(instrument) == []
    assert instrument.baselines == ("coverage of a random rollout sample of the same size",)


def test_all_three_rungs_are_reported(coverage_reading) -> None:
    assert 0.0 < coverage_reading.statement_fraction <= 1.0
    assert 0.0 < coverage_reading.branch_fraction < 1.0
    assert coverage_reading.path_fraction is not None


# ---------------------------------------------------------------------------
# Acceptance 2: an uncovered arc maps to a named rubric clause
# ---------------------------------------------------------------------------


def test_the_timeout_arm_is_uncovered_and_maps_to_its_named_clause(
    grader, coverage_reading
) -> None:
    structure, _ = analyse_source(grader)
    assert "timed_out" in coverage_reading.uncovered_clauses

    clause = next(c for c in structure.clauses if c.name == "timed_out")
    assert clause.named_by_marker
    assert clause.enter_arc is not None
    enter = tuple(clause.enter_arc)
    assert enter in {tuple(a) for a in coverage_reading.uncovered_branch_arcs}
    assert structure.arc_clause[arc_key(enter)] == "timed_out"
    assert structure.clause_for(enter).name == "timed_out"


def test_the_timeout_branch_is_reachable_only_past_the_guard(grader) -> None:
    structure, _ = analyse_source(grader)
    assert structure.guard_lines, "`if elapsed > timeout` must register as a guard"
    assert structure.guard_gated_arcs, "the arm inside it is reachable only when the guard fires"
    guard_line = _line_of(MATH_GRADER, "if elapsed > timeout")
    assert guard_line in structure.guard_lines
    gated_sources = {arc[0] for arc in structure.guard_gated_arcs}
    assert _line_of(MATH_GRADER, "if response is None") in gated_sources


def test_every_uncovered_branch_arc_gets_a_name_not_a_coordinate(grader, coverage_reading) -> None:
    """ "branch 14 uncovered" is a coordinate. This is the mapping that makes it a sentence."""
    structure, _ = analyse_source(grader)
    named = 0
    for arc in coverage_reading.uncovered_branch_arcs:
        clause = structure.arc_clause.get(arc_key(tuple(arc)))
        if clause is not None:
            named += 1
            assert clause in {c.name for c in structure.clauses}
    assert named == len(coverage_reading.uncovered_branch_arcs)


def test_covering_the_timeout_arm_removes_it_from_the_never_fired_list(grader, corpus) -> None:
    """The negative control on the mapping: name it uncovered only while it is uncovered."""
    extended = ListCorpus.of(
        list(corpus.rollouts)
        + [
            Rollout(
                id="slow",
                inputs={"response": "42", "reference": 42.0, "elapsed": 99.0, "timeout": 1.0},
            )
        ]
    )
    reading = measure_coverage(grader, extended, baseline_resamples=16)
    assert isinstance(reading, CoverageReading)
    assert "timed_out" not in reading.uncovered_clauses


def test_d9_reports_the_shape_the_spec_sentence_quotes(grader) -> None:
    structure, trusted = analyse_source(grader, rung=2)
    assert structure.branches > 0
    assert len(structure.guard_gated_arcs) >= 1
    assert structure.functions == ("parse", "normalise", "grade")
    assert structure.parameters == ("response", "reference", "scale", "elapsed", "timeout")
    assert trusted.parameters
    assert lint_instrument(StaticStructure(grader)) == []
    assert lint_instrument(TrustedInputs(grader)) == []


# ---------------------------------------------------------------------------
# Acceptance 3: mutants killed and surviving, with spans and diffs
# ---------------------------------------------------------------------------


def test_a_mutant_that_changes_a_score_on_the_corpus_is_killed(mutation_reading) -> None:
    assert mutation_reading.killed > 0
    for mutant in mutation_reading.killed_mutants:
        if mutant.outcome == "killed":
            assert mutant.rollouts_changed >= 1

    threshold_line = _line_of(MATH_GRADER, "if gap < 1e-6:")
    on_the_threshold = [
        m for m in mutation_reading.killed_mutants if m.start_line == threshold_line
    ]
    assert on_the_threshold, "mutating the exact-match threshold must change scores"


def test_a_mutant_that_changes_nothing_survives_with_a_span_and_a_diff(mutation_reading) -> None:
    assert mutation_reading.survived > 0
    for mutant in mutation_reading.survivors:
        assert mutant.rollouts_changed == 0

        # A span: two positions, in order, inside the file.
        (start_line, start_col), (end_line, end_col) = mutant.span
        assert 1 <= start_line <= len(MATH_GRADER.splitlines())
        assert (end_line, end_col) >= (start_line, start_col)

        # A diff: a real unified diff, and one that changes the line the span names.
        assert mutant.diff.startswith("--- a/")
        assert "+++ b/" in mutant.diff
        hunks = re.findall(r"@@ -(\d+),(\d+)", mutant.diff)
        assert hunks, mutant.diff
        assert any(
            int(begin) <= start_line < int(begin) + int(length) for begin, length in hunks
        ), f"the diff does not touch the line the span names:\n{mutant.diff}"


def test_the_two_survivor_classes_answer_different_questions(mutation_reading) -> None:
    """A survivor on unreached code is D1's finding restated. A reached one is D2's."""
    assert mutation_reading.unreached_survivors, "the timeout arm is unreached by this corpus"
    assert mutation_reading.blind_spots, "and some reached code is still undistinguished"
    assert all(m.rollouts_reached == 0 for m in mutation_reading.unreached_survivors)
    assert all(m.rollouts_reached > 0 for m in mutation_reading.blind_spots)
    reached = [m.rollouts_reached for m in mutation_reading.survivors]
    assert reached == sorted(reached, reverse=True), "rung 2 ranks by rollouts reached"


def test_survivors_in_the_timeout_arm_name_that_clause(mutation_reading) -> None:
    timeout_line = _line_of(MATH_GRADER, "if elapsed > timeout")
    in_arm = [
        m for m in mutation_reading.survivors if timeout_line <= m.start_line <= timeout_line + 3
    ]
    assert in_arm
    assert any(m.clause == "timed_out" for m in in_arm)


def test_the_positive_control_was_killed_and_is_reported(mutation_reading, corpus) -> None:
    assert mutation_reading.control_killed
    assert mutation_reading.control_rollouts_changed > 0
    assert mutation_reading.baselines["null_grader_rollouts_changed"] > 0
    assert 0.0 < mutation_reading.mutation_score < 1.0


def test_both_d2_instruments_lint_clean(grader, corpus) -> None:
    assert lint_instrument(SurvivingMutants(grader, corpus)) == []
    assert lint_instrument(MutationScore(grader, corpus)) == []


def test_the_survivor_list_is_sensitive_by_default(mutation_reading) -> None:
    assert is_sensitive(mutation_reading)
    published = redact(mutation_reading)
    assert published.survivors == () and published.killed_mutants == ()
    assert published.survived == mutation_reading.survived
    assert not is_sensitive(published)


# ---------------------------------------------------------------------------
# Every instrument returns Evidence or a Refusal, never an exception
# ---------------------------------------------------------------------------


def test_every_instrument_returns_a_reading_through_the_context(grader, corpus) -> None:
    from reward_lens.core.types import Phase

    ctx = program_context(grader, corpus, phase=Phase.POST_RUN)
    for instrument in (
        StaticStructure(grader),
        TrustedInputs(grader),
        DecisionCoverage(grader, corpus),
        MutationScore(grader, corpus, limit=40),
    ):
        reading = instrument.estimate(ctx)
        assert not isinstance(reading, Refusal), reading
        assert reading.value is not None
        assert reading.observable == instrument.name


def test_preflight_names_what_it_could_not_check(grader, corpus) -> None:
    ctx = program_context(grader, corpus)
    result = DecisionCoverage(grader, corpus).preflight(ctx)
    assert result.ok
    assert "access" in result.unchecked
    assert any("envelope" in u for u in result.unchecked)


# ---------------------------------------------------------------------------
# The real verifier. A synthetic test proves the code runs; this proves it measures.
# ---------------------------------------------------------------------------

#: `is_equiv` from `hendrycks/math`, the answer-equivalence checker the MATH benchmark ships. It is
#: still the verifier most open RLVR math pipelines call, directly or through a fork.
REAL_VERIFIER_URL = (
    "https://raw.githubusercontent.com/hendrycks/math/main/modeling/math_equivalence.py"
)

REAL_PAIRS = [
    ("\\frac{1}{2}", "\\frac{1}{2}"),
    ("1/2", "\\frac{1}{2}"),
    ("0.5", "\\frac{1}{2}"),
    ("2", "2"),
    ("-3", "-3"),
    ("x=5", "5"),
    ("\\sqrt2", "\\sqrt{2}"),
    ("50\\%", "50"),
    ("\\$5", "5"),
    ("5 \\text{ cm}", "5"),
    ("\\dfrac{3}{4}", "\\frac{3}{4}"),
    ("1 2", "12"),
    ("\\left(3\\right)", "(3)"),
    (".5", "0.5"),
    ("10", "11"),
    ("\\frac{a}{b}", "\\frac{b}{a}"),
    ("", ""),
    ("\\frac{1}{3}", "0.333"),
    ("\\pi", "\\pi"),
    ("3.14", "\\pi"),
    ("[0,1]", "[0, 1]"),
    ("y = 2x", "2x"),
    ("0", "0"),
    ("100", "10^2"),
    ("6", "six"),
    ("1.0", "1"),
    ("abc", "abc"),
    ("12345", "12345"),
]


@pytest.fixture(scope="module")
def real_verifier(tmp_path_factory: pytest.TempPathFactory) -> VerifierUnderTest:
    try:
        with urllib.request.urlopen(REAL_VERIFIER_URL, timeout=20) as response:
            source = response.read().decode("utf-8")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        pytest.skip(
            f"no network for the real-verifier run ({type(exc).__name__}). The fixture tests "
            f"above prove the code runs; this one proves it measures, and skipping it leaves that "
            f"unproven rather than proven."
        )
    path = tmp_path_factory.mktemp("real") / "math_equivalence.py"
    path.write_text(source, encoding="utf-8")
    return VerifierUnderTest(source_path=path, entrypoint="is_equiv")


@pytest.fixture(scope="module")
def real_corpus() -> ListCorpus:
    return ListCorpus.of(
        [
            Rollout(id=f"m{i:02d}", inputs={"str1": a, "str2": b})
            for i, (a, b) in enumerate(REAL_PAIRS)
        ]
    )


def test_d1_and_d9_measure_a_real_public_verifier(real_verifier, real_corpus) -> None:
    structure, trusted = analyse_source(real_verifier, rung=2)
    assert structure.branches >= 20, "is_equiv's module is a real parser with real branching"
    assert structure.clauses, "its three decision points are extracted as clauses"

    reading = measure_coverage(real_verifier, real_corpus, rung=2, baseline_resamples=32, seed=0)
    assert isinstance(reading, CoverageReading)
    assert 0.0 < reading.branch_fraction < 1.0, (
        "a real verifier on a real corpus is neither fully covered nor uncovered; a 1.0 here "
        "would be this instrument's kill condition arriving"
    )
    assert reading.uncovered_branch_arcs
    assert reading.uncovered_clauses, (
        "at least one of is_equiv's own guards is never exercised by 28 non-None answer pairs"
    )
    assert reading.baselines["random_rollout_sample_same_size"] <= reading.branch_fraction + 1e-12
    assert len(reading.per_rollout_arcs) == len(real_corpus)


def test_d2_finds_reachable_blind_spots_in_a_real_public_verifier(
    real_verifier, real_corpus
) -> None:
    reading = run_mutation(real_verifier, real_corpus, rung=2, timeout=60.0)
    assert isinstance(reading, MutationReading), reading
    assert reading.control_killed, "the corpus separates a constant grader from the real one"
    assert reading.mutants_generated >= 100
    assert 0.0 < reading.mutation_score < 0.95, (
        "a score above 0.95 on a real verifier is D2's kill condition, not a passing test"
    )
    assert reading.blind_spots, (
        "some mutation of code every rollout runs must be invisible to the whole corpus"
    )
    for mutant in reading.blind_spots[:20]:
        assert mutant.diff.startswith("--- a/")
        assert mutant.start_line >= 1
        assert mutant.rollouts_reached > 0
        assert mutant.rollouts_changed == 0


def _line_of(source: str, needle: str) -> int:
    for i, line in enumerate(source.splitlines(), start=1):
        if needle in line:
            return i
    raise AssertionError(f"{needle!r} is not in the fixture source")

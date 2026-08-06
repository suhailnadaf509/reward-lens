"""D1: per-rollout branch attribution, the same-size baseline, and the two refusals."""

from __future__ import annotations

from pathlib import Path

import pytest

from reward_lens.core.invariance import check_invariance
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import lint_instrument
from reward_lens.verifier import ListCorpus, Rollout, VerifierUnderTest
from reward_lens.verifier.coverage import (
    CoverageReading,
    DecisionCoverage,
    branch_universe,
    measure_coverage,
    trace_corpus,
)

GRADER = '''"""Parse, normalise, threshold. The timeout arm fires only when it is asked to."""


def grade(response, reference, scale=1.0, elapsed=0.0, timeout=5.0):
    if elapsed > timeout:  # rubric: timed_out
        return 0.25
    if response is None:  # rubric: unparseable
        return 0.0
    got = float(response) / scale
    want = float(reference) / scale
    if abs(got - want) < 1e-6:  # rubric: exact
        return 1.0
    if abs(got - want) < 0.5:  # rubric: near
        return 0.5
    return 0.0
'''


@pytest.fixture
def grader(tmp_path: Path) -> VerifierUnderTest:
    path = tmp_path / "grader.py"
    path.write_text(GRADER, encoding="utf-8")
    return VerifierUnderTest(source_path=path, entrypoint="grade")


def _corpus(n: int = 12) -> ListCorpus:
    """Twelve rollouts that never time out, so the timeout arm is provably never taken."""
    rollouts = []
    for i in range(n):
        rollouts.append(
            Rollout(id=f"r{i}", inputs={"response": str(float(i)), "reference": 0.0}, score=None)
        )
    return ListCorpus.of(rollouts)


# ---------------------------------------------------------------------------
# The package trap, and the E9 call this instrument stands on
# ---------------------------------------------------------------------------


def test_the_coverage_we_import_is_the_third_party_one_not_our_module() -> None:
    import coverage

    from reward_lens.verifier import coverage as ours

    assert "site-packages" in coverage.__file__
    assert ours.__file__.endswith("reward_lens/verifier/coverage.py")
    assert ours is not coverage
    # The absolute import inside our module resolves to the package, not to itself.
    assert ours.trace_corpus.__module__ == "reward_lens.verifier.coverage"


def test_the_context_api_is_switch_context_not_set_context() -> None:
    """E9. The write-up names `set_context()`; `Coverage` has no such method."""
    import coverage

    assert not hasattr(coverage.Coverage, "set_context")
    assert hasattr(coverage.Coverage, "switch_context")
    assert hasattr(coverage.CoverageData, "set_query_context")


def test_json_report_wants_a_path_not_a_file_object(tmp_path: Path, grader) -> None:
    """E9, the second half of the coverage entry. Held, re-checked here."""
    import io

    import coverage

    cov = coverage.Coverage(branch=True, data_file=None, include=[str(grader.source_path)])
    cov.start()
    grader.load()(response="1.0", reference=1.0)
    cov.stop()
    out = tmp_path / "cov.json"
    assert cov.json_report(outfile=str(out)) >= 0.0
    assert out.exists()
    with pytest.raises(TypeError):
        cov.json_report(outfile=io.StringIO())


# ---------------------------------------------------------------------------
# Per-rollout attribution: the thing that makes this an instrument
# ---------------------------------------------------------------------------


def test_two_rollouts_taking_different_branches_get_different_arcs(grader) -> None:
    corpus = ListCorpus.of(
        [
            Rollout(id="exact", inputs={"response": "1.0", "reference": 1.0}),
            Rollout(id="miss", inputs={"response": "9.0", "reference": 1.0}),
        ]
    )
    traced = trace_corpus(grader, corpus)
    assert traced.per_rollout_arcs["exact"] != traced.per_rollout_arcs["miss"]
    assert traced.per_rollout_arcs["exact"] <= traced.aggregate_arcs
    assert traced.per_rollout_arcs["miss"] <= traced.aggregate_arcs


def test_the_branch_denominator_agrees_with_coverages_own_arithmetic(grader) -> None:
    """A denominator that quietly disagrees with the tool it came from is the whole failure mode."""
    import coverage
    from coverage.python import PythonFileReporter
    from coverage.results import analysis_from_file_reporter

    corpus = _corpus(4)
    traced = trace_corpus(grader, corpus)
    ours = branch_universe(traced.arc_universe, traced.exit_counts)

    cov = coverage.Coverage(branch=True, data_file=None, include=[str(grader.source_path)])
    cov.start()
    fn = grader.load()
    for rollout in corpus:
        fn(**rollout.inputs)
    cov.stop()
    reporter = PythonFileReporter(str(grader.source_path), cov)
    analysis = analysis_from_file_reporter(cov.get_data(), 2, reporter, str(grader.source_path))
    stats = analysis.branch_stats()
    assert len(ours) == sum(total for total, _ in stats.values())
    assert len(ours & traced.aggregate_arcs) == sum(taken for _, taken in stats.values())


def test_an_uncovered_arc_names_the_clause_it_belongs_to(grader) -> None:
    reading = measure_coverage(grader, _corpus(), baseline_resamples=8)
    assert isinstance(reading, CoverageReading)
    assert "timed_out" in reading.uncovered_clauses
    assert reading.branch_fraction < 1.0
    assert reading.uncovered_branch_arcs


def test_covering_the_timeout_arm_removes_it_from_the_never_fired_list(grader) -> None:
    corpus = ListCorpus.of(
        list(_corpus(4).rollouts)
        + [
            Rollout(
                id="slow",
                inputs={"response": "1.0", "reference": 1.0, "elapsed": 99.0, "timeout": 1.0},
            )
        ]
    )
    reading = measure_coverage(grader, corpus, baseline_resamples=8)
    assert isinstance(reading, CoverageReading)
    assert "timed_out" not in reading.uncovered_clauses


def test_statement_and_branch_fractions_are_both_reported(grader) -> None:
    reading = measure_coverage(grader, _corpus(), rung=1, baseline_resamples=8)
    assert isinstance(reading, CoverageReading)
    assert 0.0 < reading.statement_fraction <= 1.0
    assert 0.0 < reading.branch_fraction < 1.0
    assert reading.headline == reading.branch_fraction
    assert measure_coverage(grader, _corpus(), rung=0, baseline_resamples=4).headline == (
        reading.statement_fraction
    )


# ---------------------------------------------------------------------------
# The mandatory baseline
# ---------------------------------------------------------------------------


def test_the_same_size_random_sample_baseline_is_computed(grader) -> None:
    reading = measure_coverage(grader, _corpus(20), baseline_resamples=32, seed=7)
    assert isinstance(reading, CoverageReading)
    base = reading.baselines["random_rollout_sample_same_size"]
    assert 0.0 < base <= reading.branch_fraction + 1e-12
    assert "saturation_size" in reading.baselines
    assert 0.0 <= reading.baselines["redundant_fraction"] < 1.0


def test_a_redundant_corpus_saturates_early(grader) -> None:
    """Twenty copies of one behaviour reach full coverage from a small fraction of themselves."""
    corpus = ListCorpus.of(
        [Rollout(id=f"c{i}", inputs={"response": "1.0", "reference": 1.0}) for i in range(20)]
    )
    reading = measure_coverage(grader, corpus, baseline_resamples=32, seed=1)
    assert isinstance(reading, CoverageReading)
    assert reading.baselines["saturation_size"] <= 2.0
    assert reading.baselines["redundant_fraction"] >= 0.9


# ---------------------------------------------------------------------------
# Rung 2
# ---------------------------------------------------------------------------


def test_rung_2_counts_observed_paths_against_a_stated_bound(grader) -> None:
    reading = measure_coverage(grader, _corpus(10), rung=2, baseline_resamples=8)
    assert isinstance(reading, CoverageReading)
    assert reading.paths_observed is not None and reading.paths_observed >= 1
    assert reading.paths_bound is not None and reading.paths_bound >= reading.paths_observed
    assert reading.path_fraction is not None and 0.0 < reading.path_fraction <= 1.0
    assert reading.hot_lines
    assert any("back edge" in note for note in reading.notes)


# ---------------------------------------------------------------------------
# Refusal is a value
# ---------------------------------------------------------------------------


def test_an_empty_corpus_refuses_with_a_remedy(grader) -> None:
    reading = measure_coverage(grader, ListCorpus.of([]))
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "ListCorpus.of" in reading.remedy
    assert reading.statistics["n_rollouts"] == 0


def test_a_source_path_the_tracer_never_enters_refuses_and_names_what_it_saw(
    tmp_path: Path, grader
) -> None:
    other = tmp_path / "decoy.py"
    other.write_text("def grade(response, reference):\n    return 1.0\n", encoding="utf-8")
    real = grader.load()

    class _Elsewhere(VerifierUnderTest):
        def load(self):  # type: ignore[override]
            return real

    decoy = _Elsewhere(source_path=other, entrypoint="grade")
    reading = measure_coverage(decoy, _corpus(3))
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "--source" in reading.remedy


def test_a_verifier_that_raises_is_recorded_not_swallowed(tmp_path: Path) -> None:
    path = tmp_path / "boom.py"
    path.write_text(
        "def grade(x):\n    if x < 0:\n        raise ValueError('negative')\n    return 1.0\n",
        encoding="utf-8",
    )
    verifier = VerifierUnderTest(source_path=path)
    corpus = ListCorpus.of([Rollout(id="ok", inputs={"x": 1}), Rollout(id="bad", inputs={"x": -1})])
    reading = measure_coverage(verifier, corpus, baseline_resamples=4)
    assert isinstance(reading, CoverageReading)
    assert reading.errors["bad"].startswith("ValueError")
    assert "ok" not in reading.errors
    assert reading.per_rollout_arcs["bad"], "the raising rollout's arcs are kept"


# ---------------------------------------------------------------------------
# The instrument contract
# ---------------------------------------------------------------------------


def test_d1_lints_clean_and_declares_its_baseline(grader) -> None:
    instrument = DecisionCoverage(grader, _corpus(3))
    assert lint_instrument(instrument) == []
    assert instrument.baselines == ("coverage of a random rollout sample of the same size",)
    assert instrument.quantity == "verifier.decision_coverage"


def test_the_ladder_is_registered_for_all_three_rungs() -> None:
    from reward_lens.core.quantity import ladder

    rungs = [e.rung for e in ladder("verifier.decision_coverage")]
    assert rungs == [0, 1, 2]


def test_the_generated_invariance_test_passes(grader) -> None:
    instrument = DecisionCoverage(grader, _corpus(4))
    report = check_invariance(
        instrument,
        "trivial",
        ctx=None,
        run=lambda inst, _p: measure_coverage(grader, _corpus(4), baseline_resamples=4).headline,
    )
    assert report.passed


def test_an_instrument_with_no_corpus_refuses_rather_than_guessing(grader) -> None:
    from reward_lens.verifier import program_context

    ctx = program_context(grader)
    reading = DecisionCoverage(grader).estimate(ctx)
    assert isinstance(reading, Refusal)
    assert "ListCorpus.of" in reading.remedy

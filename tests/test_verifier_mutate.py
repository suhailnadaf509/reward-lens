"""D2: the corpus as the test suite, the positive control, and the survivor's span and diff."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from reward_lens.core.invariance import check_invariance
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import lint_instrument
from reward_lens.verifier import ListCorpus, Rollout, VerifierUnderTest, is_sensitive, redact
from reward_lens.verifier.mutate import (
    MutationReading,
    MutationScore,
    SurvivingMutants,
    generate_mutants,
    null_grader_source,
    run_mutation,
    wall_limit,
)

GRADER = '''"""A grader with a reachable-but-undistinguished branch and an unreachable one."""


def grade(response, reference, elapsed=0.0, timeout=5.0):
    if elapsed > timeout:  # rubric: timed_out
        return 0.25
    if response is None:  # rubric: unparseable
        return 0.0
    gap = abs(float(response) - float(reference))
    if gap < 1e-6:  # rubric: exact
        return 1.0
    if gap < 0.5:  # rubric: near
        return 0.5
    return 0.0
'''


@pytest.fixture
def grader(tmp_path: Path) -> VerifierUnderTest:
    path = tmp_path / "grader.py"
    path.write_text(GRADER, encoding="utf-8")
    return VerifierUnderTest(source_path=path, entrypoint="grade")


def _corpus() -> ListCorpus:
    """A corpus with real score spread and no rollout anywhere near the `near` boundary."""
    return ListCorpus.of(
        [
            Rollout(id="exact", inputs={"response": "1.0", "reference": 1.0}, score=1.0),
            Rollout(id="near", inputs={"response": "1.1", "reference": 1.0}, score=0.5),
            Rollout(id="far", inputs={"response": "9.0", "reference": 1.0}, score=0.0),
            Rollout(id="far2", inputs={"response": "17.0", "reference": 1.0}, score=0.0),
            Rollout(id="exact2", inputs={"response": "3.0", "reference": 3.0}, score=1.0),
        ]
    )


# ---------------------------------------------------------------------------
# The engine, re-verified. E9 rewrote both of these access paths.
# ---------------------------------------------------------------------------


def test_cosmic_ray_module_layout_is_work_db_not_db() -> None:
    """E9: `cosmic_ray.db` does not exist; the module is `cosmic_ray.work_db`."""
    import cosmic_ray.work_db  # noqa: F401
    from cosmic_ray.work_item import MutationSpec, TestOutcome, WorkItem, WorkResult

    with pytest.raises(ModuleNotFoundError):
        __import__("cosmic_ray.db")

    assert TestOutcome.SURVIVED.value == "survived"
    # The span is on MutationSpec, not on WorkItem; the diff is on WorkResult.
    fields = {f.name for f in MutationSpec.__attrs_attrs__}
    assert {"start_pos", "end_pos", "module_path"} <= fields
    assert not hasattr(WorkItem, "start_pos")
    assert "diff" in {f.name for f in WorkResult.__attrs_attrs__}


def test_generate_mutants_returns_spans_and_a_real_diff(grader: VerifierUnderTest) -> None:
    specs = generate_mutants(grader.source())
    assert len(specs) >= 10
    for spec in specs:
        assert spec.source != grader.source()
        assert len(spec.start) == 2 and len(spec.end) == 2
        assert spec.start[0] >= 1
    assert len({s.source for s in specs}) == len(specs), "duplicate mutants are deduplicated"


def test_rung_0_mutates_only_the_scoring_function(tmp_path: Path) -> None:
    source = "HELPER = 1 + 2\n\n\ndef grade(x):\n    return x + HELPER\n"
    path = tmp_path / "g.py"
    path.write_text(source, encoding="utf-8")
    whole = generate_mutants(source)
    entry_only = generate_mutants(source, line_range=(4, 5))
    assert len(entry_only) < len(whole)
    assert all(m.start[0] >= 4 for m in entry_only)


def test_covered_lines_restricts_mutation(grader: VerifierUnderTest) -> None:
    everywhere = generate_mutants(grader.source())
    restricted = generate_mutants(grader.source(), covered_lines={9})
    assert 0 < len(restricted) < len(everywhere)
    assert {m.start[0] for m in restricted} == {9}


def test_mutmut_import_raises_outside_a_prepared_directory() -> None:
    """E9: importing `mutmut.__main__` loads a config and dies without one. Still true."""
    import subprocess
    import sys

    with tempfile.TemporaryDirectory() as empty:
        result = subprocess.run(
            [sys.executable, "-c", "from mutmut.__main__ import mutate_file_contents"],
            cwd=empty,
            capture_output=True,
            text=True,
        )
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr

    # And it works from a directory that has one, which is what the fallback prepares.
    with tempfile.TemporaryDirectory() as prepared:
        os.makedirs(os.path.join(prepared, "src"), exist_ok=True)
        Path(prepared, "setup.cfg").write_text("[mutmut]\nsource_paths=src\n", encoding="utf-8")
        ok = subprocess.run(
            [
                sys.executable,
                "-c",
                "from mutmut.__main__ import mutate_file_contents as f; import mutmut;"
                " print(mutmut.__file__)",
            ],
            cwd=prepared,
            capture_output=True,
            text=True,
        )
    assert ok.returncode == 0, ok.stderr
    assert "site-packages" in ok.stdout


def test_mutmut_return_shape_is_pinned_because_3_7_changed_it(tmp_path: Path) -> None:
    """E9 verified mutmut 3.6.0, which returns `(code, names)`.

    `mutmut>=3.6.0,<4.0` resolves to 3.7.0 today, and 3.7.0 returns a `MutatedFile` object with
    `.code`, `.mutant_names` and `.line_span_by_function_name` instead. The fallback handles both;
    this pins which one is installed, so an upgrade that changes it again fails here rather than in
    a survivor list.
    """
    import mutmut

    from reward_lens.verifier.mutate import generate_mutants_mutmut

    assert "site-packages" in mutmut.__file__
    major, minor = (int(p) for p in mutmut.__version__.split(".")[:2])
    assert (major, minor) >= (3, 6)

    path = tmp_path / "g.py"
    source = "def grade(x):\n    return x > 10\n"
    path.write_text(source, encoding="utf-8")
    specs = generate_mutants_mutmut(source, path)
    assert specs, "the fallback engine produces mutants for a two-line grader"
    assert all(spec.start == (0, 0) for spec in specs), (
        "mutmut's trampoline has no per-mutant span, and the zero span is how that is declared"
    )


# ---------------------------------------------------------------------------
# The positive control
# ---------------------------------------------------------------------------


def test_the_null_grader_replaces_only_the_entrypoints_body() -> None:
    null = null_grader_source(GRADER, "grade")
    assert "return 0.0" in null
    assert "if elapsed > timeout" not in null
    assert '"""A grader with' in null


def test_a_corpus_that_cannot_kill_the_null_grader_refuses(grader: VerifierUnderTest) -> None:
    """Every rollout scores 0.0, so a constant grader is indistinguishable from the real one."""
    flat = ListCorpus.of(
        [Rollout(id=f"z{i}", inputs={"response": "99.0", "reference": 1.0}) for i in range(4)]
    )
    reading = run_mutation(grader, flat, timeout=10.0)
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.NO_MATCHED_CONTROL
    assert reading.statistics["control_rollouts_changed"] == 0
    assert "cannot kill any mutant" in reading.remedy


def test_an_empty_corpus_refuses(grader: VerifierUnderTest) -> None:
    reading = run_mutation(grader, ListCorpus.of([]))
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ACCESS_INSUFFICIENT


# ---------------------------------------------------------------------------
# The inversion: the corpus is the test suite
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> MutationReading:
    path = tmp_path_factory.mktemp("mut") / "grader.py"
    path.write_text(GRADER, encoding="utf-8")
    verifier = VerifierUnderTest(source_path=path, entrypoint="grade")
    reading = run_mutation(verifier, _corpus(), rung=2, timeout=10.0)
    assert isinstance(reading, MutationReading)
    return reading


def test_the_run_produces_both_kills_and_survivors(run: MutationReading) -> None:
    assert run.mutants_generated > 20
    assert run.killed > 0
    assert run.survived > 0
    assert 0.0 < run.mutation_score < 1.0
    assert run.control_killed and run.control_rollouts_changed > 0


def test_every_killed_mutant_changed_at_least_one_score(run: MutationReading) -> None:
    for mutant in run.killed_mutants:
        assert mutant.outcome in ("killed", "timeout")
        if mutant.outcome == "killed":
            assert mutant.rollouts_changed >= 1


def test_every_survivor_changed_no_score_and_carries_a_span_and_a_diff(
    run: MutationReading,
) -> None:
    assert run.survivors
    for mutant in run.survivors:
        assert mutant.outcome == "survived"
        assert mutant.rollouts_changed == 0
        assert mutant.start_line >= 1 and mutant.end_line >= mutant.start_line
        assert mutant.span[0] <= mutant.span[1]
        assert mutant.diff.startswith("---")
        assert "+" in mutant.diff and "-" in mutant.diff
        assert mutant.source_line


def test_a_survivor_on_the_unreached_timeout_arm_is_separated_from_a_real_blind_spot(
    run: MutationReading,
) -> None:
    """The two survivor classes answer different questions and are reported apart."""
    timeout_line = GRADER.splitlines().index("    if elapsed > timeout:  # rubric: timed_out") + 1
    unreached = {m.start_line for m in run.unreached_survivors}
    assert timeout_line + 1 in unreached or timeout_line in unreached
    assert run.blind_spots, "some survivor should sit on code the corpus actually runs"
    assert all(m.rollouts_reached > 0 for m in run.blind_spots)
    assert all(m.rollouts_reached == 0 for m in run.unreached_survivors)


def test_survivors_are_ranked_by_how_many_rollouts_reached_them(run: MutationReading) -> None:
    reached = [m.rollouts_reached for m in run.survivors]
    assert reached == sorted(reached, reverse=True)


def test_survivors_name_the_rubric_clause_they_sit_in(run: MutationReading) -> None:
    named = [m for m in run.survivors + run.killed_mutants if m.clause]
    assert named
    assert {"timed_out", "unparseable", "exact", "near"} & {m.clause for m in named}


def test_a_mutant_that_makes_the_grader_raise_is_killed(tmp_path: Path) -> None:
    path = tmp_path / "d.py"
    path.write_text("def grade(x):\n    return 1.0 / (x - 10)\n", encoding="utf-8")
    verifier = VerifierUnderTest(source_path=path)
    corpus = ListCorpus.of([Rollout(id="a", inputs={"x": 1}), Rollout(id="b", inputs={"x": 2})])
    reading = run_mutation(verifier, corpus, timeout=10.0)
    assert isinstance(reading, MutationReading)
    raising = [m for m in reading.killed_mutants if "10" in m.diff]
    assert raising, "the mutant that turns the divisor into zero should be killed"


# ---------------------------------------------------------------------------
# The timeout guard
# ---------------------------------------------------------------------------


def test_wall_limit_reports_whether_it_could_guard() -> None:
    import threading

    with wall_limit(1.0) as guarded:
        assert guarded is True
    with wall_limit(None) as guarded:
        assert guarded is False

    seen = []

    def _off_thread() -> None:
        with wall_limit(1.0) as g:
            seen.append(g)

    thread = threading.Thread(target=_off_thread)
    thread.start()
    thread.join()
    assert seen == [False], "SIGALRM only fires on the main thread, and the reading must say so"


def test_a_hanging_mutant_is_a_timeout_kill_not_a_hang(tmp_path: Path) -> None:
    path = tmp_path / "loop.py"
    path.write_text(
        "def grade(n):\n"
        "    total = 0\n"
        "    i = 0\n"
        "    while i < n:\n"
        "        total += i\n"
        "        i += 1\n"
        "    return float(total)\n",
        encoding="utf-8",
    )
    verifier = VerifierUnderTest(source_path=path)
    corpus = ListCorpus.of([Rollout(id="a", inputs={"n": 3}), Rollout(id="b", inputs={"n": 5})])
    reading = run_mutation(verifier, corpus, timeout=2.0)
    assert isinstance(reading, MutationReading)
    assert reading.timeout_guard is True
    assert reading.timed_out >= 1, "mutating `i += 1` to `i -= 1` must not hang the run"


# ---------------------------------------------------------------------------
# Dual use
# ---------------------------------------------------------------------------


def test_the_survivor_list_is_sensitive_and_redaction_keeps_the_counts(
    run: MutationReading,
) -> None:
    assert is_sensitive(run)
    published = redact(run)
    assert published.survivors == ()
    assert published.killed_mutants == ()
    assert published.survived == run.survived
    assert published.mutation_score == run.mutation_score
    assert is_sensitive(published) is False
    assert any("redacted" in note for note in published.notes)


def test_redact_refuses_a_sensitive_payload_with_no_redaction() -> None:
    class _Leaky:
        sensitive = True

    with pytest.raises(TypeError, match="implements no redacted"):
        redact(_Leaky())


def test_a_non_sensitive_payload_passes_through_redact_unchanged() -> None:
    from reward_lens.verifier.coverage import CoverageReading

    reading = CoverageReading(
        source_path="x",
        entrypoint="grade",
        fingerprint="f",
        rung=1,
        n_rollouts=1,
        statements_total=1,
        statements_covered=1,
        branch_arcs_total=2,
        branch_arcs_covered=1,
        uncovered_branch_arcs=(),
        uncovered_clauses=(),
    )
    assert is_sensitive(reading) is False
    assert redact(reading) is reading


# ---------------------------------------------------------------------------
# The instrument contract
# ---------------------------------------------------------------------------


def test_both_d2_instruments_lint_clean(grader: VerifierUnderTest) -> None:
    assert lint_instrument(SurvivingMutants(grader, _corpus())) == []
    assert lint_instrument(MutationScore(grader, _corpus())) == []


def test_the_ladder_is_registered_for_both_quantities() -> None:
    from reward_lens.core.quantity import ladder

    assert [e.rung for e in ladder("verifier.surviving_mutants")] == [0, 1, 2]
    assert [e.rung for e in ladder("verifier.mutation_score")] == [0, 1, 2]


def test_the_generated_invariance_test_passes(grader: VerifierUnderTest) -> None:
    instrument = MutationScore(grader, _corpus())
    report = check_invariance(
        instrument,
        "trivial",
        ctx=None,
        run=lambda inst, _p: 1.0,
    )
    assert report.passed


def test_the_instrument_refuses_without_a_corpus(grader: VerifierUnderTest) -> None:
    from reward_lens.verifier import program_context

    reading = SurvivingMutants(grader).estimate(program_context(grader))
    assert isinstance(reading, Refusal)
    assert "test suite" in reading.remedy

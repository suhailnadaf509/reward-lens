"""Acceptance: D6 exploit-family coverage, D8 attack surface, D10 replay fidelity.

**On the clause.** The overall clause covers all ten D instruments and reads:

    *a card renders for one marketplace environment with coverage, surviving mutants (with source
    spans and diffs), at least one metamorphic violation with a reproducer, Sobol total-effect
    indices, a false-positive catalogue, the silent-zero rate and a flakiness spread.*

Seven items. Five are D1 to D5 and were discharged elsewhere. The remaining two belong to
other series: the silent-zero rate is B4's `grader.silent_zero_rate` and the flakiness spread is
A7's `env.flakiness`, which the catalogue registers at line 417 with a unit of percentage points.
**Nothing in the clause is produced by D6, D8 or D10.** That is a gap in the clause rather than in this
package, and it is reported rather than papered over, so the clause this file discharges is stated
here in full and is the one the three catalogue records imply:

    *the unseen exploit mass and the reliability-growth exponent with its interval, computed from
    an exploit log; a harness attack surface that names an unchecked route from untrusted input to
    a scoring decision and is sensitive by default; and a replay fidelity measured by re-grading a
    record.*

**On pointing at something real.** A synthetic subject proves the code runs and a real one proves
it measures, and this file runs both. The real subjects are `is_equiv` from `hendrycks/math`, the
answer checker most open RLVR math pipelines call, and `swebench/harness/grading.py`, the shipped
grading module of the benchmark that defines the agentic-SWE field. Both are fetched over the
network and both skip with a message rather than passing quietly when there is none, because a
skipped real-subject test leaves the claim unproven rather than proven.
"""

from __future__ import annotations

import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from random import Random

import pytest

from reward_lens.core.invariance import InvariancePayload, check_invariance
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import lint_instrument
from reward_lens.verifier import ListCorpus, Rollout, VerifierUnderTest, is_sensitive, redact
from reward_lens.verifier.attack import (
    AttackCheapness,
    AttackCheapnessRatio,
    AttackSurfaceInventory,
    CostSample,
    TaskBoundary,
    analyse_environment,
    attack_cheapness,
)
from reward_lens.verifier.fuzz import DisclosureRequired
from reward_lens.verifier.growth import (
    CrowFit,
    ExploitCoverage,
    ExploitFamilyCoverage,
    ExploitFind,
    ExploitLog,
    ReliabilityGrowth,
    exploit_coverage,
)
from reward_lens.verifier.metamorphic import answer_text_relations
from reward_lens.verifier.replay import ReplayFidelity, ReplayReport, replay_corpus

# ---------------------------------------------------------------------------
# The real subjects
# ---------------------------------------------------------------------------

#: `is_equiv` from `hendrycks/math`. The same subject the other D packages used, reused so
#: the D-series numbers on this grader accumulate against one program rather than four.
MATH_URL = "https://raw.githubusercontent.com/hendrycks/math/main/modeling/math_equivalence.py"

#: SWE-bench's shipped grading module. `get_logs_eval(test_spec, log_fp)` opens a log file written
#: by running the agent's own tests inside the container and derives the resolution verdict from
#: its text, which is the specification's example sentence in the most-used harness in the field.
SWEBENCH_URL = (
    "https://raw.githubusercontent.com/SWE-bench/SWE-bench/main/swebench/harness/grading.py"
)

#: Answer pairs the MATH checker accepts or rejects, chosen to span its normalisation paths.
MATH_PAIRS = [
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
    ("-2", "- 2"),
    ("2.50", "2.5"),
    ("\\text{x}", "x"),
]


def _fetch(url: str, into: Path, name: str) -> Path:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            source = response.read().decode("utf-8")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        pytest.skip(
            f"no network for the real-subject run ({type(exc).__name__}). The fixture tests in "
            f"this file prove the code runs; this one proves it measures, and skipping it leaves "
            f"that unproven rather than proven."
        )
    path = into / name
    path.write_text(source, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def math_verifier(tmp_path_factory: pytest.TempPathFactory) -> VerifierUnderTest:
    path = _fetch(MATH_URL, tmp_path_factory.mktemp("math"), "math_equivalence.py")
    return VerifierUnderTest(source_path=path, entrypoint="is_equiv")


@pytest.fixture(scope="module")
def swebench_harness(tmp_path_factory: pytest.TempPathFactory) -> VerifierUnderTest:
    path = _fetch(SWEBENCH_URL, tmp_path_factory.mktemp("swebench"), "grading.py")
    return VerifierUnderTest(source_path=path, entrypoint="get_logs_eval")


def _math_exploit_log(verifier: VerifierUnderTest) -> ExploitLog:
    """A real exploit log: every metamorphic violation `is_equiv` commits, in discovery order.

    An exploit here is a concrete pair of inputs on which the MATH answer checker returns a verdict
    that is wrong by construction, because the transformation applied to the accepted answer
    preserves its meaning and the verdict changed anyway. The family is the relation that broke it,
    which is a published failure mode rather than somebody's guess, and the effort axis is the
    running count of rollout-relation trials, which is the cumulative search this log bought.

    This is a real log about a real verifier and it is not the thing D6 was written for, which is a
    lab's own blacklist of hacking strategies observed during training. The families here are ways
    a checker's verdict moves under a semantics-preserving rewrite, not strategies a policy found.
    """
    fn = verifier.load()
    relations = answer_text_relations(on="str1")
    rng = Random(0)
    finds: list[ExploitFind] = []
    trial = 0
    for i, (answer, gold) in enumerate(MATH_PAIRS):
        seed = Rollout(id=f"m{i:02d}", inputs={"str1": answer, "str2": gold})
        base = 1.0 if fn(str1=answer, str2=gold) else 0.0
        if base <= 0.0:
            # D3's convention: a relation about an accepted solution says nothing about a rejected
            # one, so a rejected pair contributes no trials and no denominator.
            continue
        for relation in relations:
            moved = relation.transformation.apply(seed, rng)
            if moved is seed:
                continue
            trial += 1
            after = 1.0 if fn(**dict(moved.inputs)) else 0.0
            if abs(after - base) > 1e-12:
                finds.append(ExploitFind(family=relation.name, effort=float(trial), id=moved.id))
    return ExploitLog.of(
        finds,
        total_effort=float(trial),
        effort_unit="rollout-relation trials",
        source="hendrycks/math is_equiv",
    )


# ---------------------------------------------------------------------------
# Clause 1 — D6 on a real exploit log
# ---------------------------------------------------------------------------


def test_d6_reads_a_real_exploit_log_from_a_real_public_verifier(math_verifier) -> None:
    """*The unseen exploit mass and the growth exponent, computed from an exploit log.*

    The measured result on `is_equiv`, and it is worth reading before the assertions: four families
    over forty-five finds, every one of them found more than once, so `f1 = 0` and Good-Turing puts
    the unseen mass at zero. That is the arithmetic working, and it is also **the kill condition
    originally named for this instrument arriving on the first real log it was pointed at**: with no
    singleton families the estimator has nothing to extrapolate from and the reading collapses onto
    the baseline count it was supposed to improve on. That kill condition was later dropped and D6
    declared unkillable; the first real log says keeping it was right.
    """
    log = _math_exploit_log(math_verifier)
    assert log.n >= 40, "the MATH checker breaks under a semantics-preserving rewrite in bulk"
    assert log.s_obs == 4, log.counts()
    assert log.has_effort

    reading = exploit_coverage(log, rung=2)
    assert isinstance(reading, ExploitCoverage), getattr(reading, "render", lambda: reading)()

    # Good-Turing on a log where nothing was seen exactly once.
    assert reading.f1 == 0
    assert reading.f2 == 2
    assert reading.novelty_probability == 0.0
    assert reading.unseen_fraction == 0.0
    assert reading.chao1 == pytest.approx(float(log.s_obs))
    assert any("stopped turning up new things" in n for n in reading.notes)

    # The baseline the catalogue makes mandatory, printed beside the claim.
    assert f"baseline (the raw count of families found): {log.s_obs}" in reading.render()


def test_d6_reports_an_interval_on_beta_and_the_interval_is_what_decides(math_verifier) -> None:
    """*Report the interval on beta.* On this log it spans 1, and that is the finding.

    beta = 0.51 alone reads as "the blacklist is converging". The interval on four points runs from
    below zero to above one, which means the log does not settle the question, and the difference
    between those two sentences is the whole reason the interval is not optional.
    """
    log = _math_exploit_log(math_verifier)
    reading = exploit_coverage(log, rung=1)
    assert isinstance(reading, ExploitCoverage)
    fit = reading.fit
    assert isinstance(fit, CrowFit)

    assert fit.points == 4, "one point per family first-discovery, which is what a blacklist holds"
    assert fit.beta < 1.0
    assert fit.ci_low < 1.0 < fit.ci_high
    assert fit.converging_at_interval is None
    assert "does not yet settle" in fit.render()

    # The maximum-likelihood cross-check, which is what reliability practice actually uses.
    assert fit.beta_mle is not None and 0.0 < fit.beta_mle < 1.0


def test_d6_refuses_on_a_real_short_log_and_the_remedy_names_the_n_needed(math_verifier) -> None:
    """*f2 = 0 is a refusal with a remedy naming the n needed, not an infinity.*

    Run on the first two finds of the real log, which is what the same search would have produced
    had somebody looked after two hits. Both are singletons, so the Chao1 bound is undefined.
    """
    log = _math_exploit_log(math_verifier)
    prefix = ExploitLog.of(
        list(log.finds[:2]),
        total_effort=log.finds[1].effort,
        effort_unit=log.effort_unit,
        source=f"{log.source} (first two finds)",
    )
    assert prefix.spectrum() == {1: 2}

    refusal = exploit_coverage(prefix, rung=2)
    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.ABOVE_LOD_BELOW_LOQ
    assert refusal.statistics["f2"] == 0

    needed = refusal.statistics["additional_finds_needed"]
    assert needed >= 1
    assert f"about {needed} more find" in refusal.remedy
    assert refusal.is_bounded and refusal.partial is not None

    # And the number that survives f2 = 0 is offered rather than withheld.
    assert "f1/n" in refusal.remedy


# ---------------------------------------------------------------------------
# Clause 2 — D8 on a real harness
# ---------------------------------------------------------------------------


def test_d8_finds_the_specification_sentence_in_a_real_shipped_harness(swebench_harness) -> None:
    """*The scoring script reads a file the agent can write.*

    Measured on SWE-bench's own `grading.py`. `get_logs_eval(test_spec, log_fp)` opens `log_fp`,
    reads it, splits the text on marker constants and hands the result to a per-repository parser
    whose output is the returned status map. Every step of that chain is inside the sandbox the
    agent was working in, and nothing between the read and the return checks anything.
    """
    surface = analyse_environment(swebench_harness, rung=0)

    reads = [a for a in surface.accesses if a.kind == "read"]
    assert reads, surface.render(include_targets=True)
    assert any(a.call == "open" and not a.target_is_literal for a in reads), (
        "the file the verdict is read from is named by a parameter, not by a fixed path"
    )
    assert any(a.tainted_target for a in reads), (
        "the agent's own container decides which file the grader opens"
    )

    routes = surface.unchecked_taints
    assert routes, "a scoring decision computed from unchecked file contents is the D8 pattern"
    assert any(r.sink_kind == "return" for r in routes), (
        "the resolution verdict itself, not only a branch on the way to one"
    )
    assert all(r.source_kind == "resource_read" for r in routes)
    assert surface.headline == len(routes)


def test_d8s_output_is_sensitive_and_does_not_leave_the_building_by_default(
    swebench_harness,
) -> None:
    """*D8's output is an exploit list.* The mechanism is D5's, not a second one.

    An inventory of a real, widely deployed harness is exactly the payload the dual-use rule is
    about, so this asserts the three properties on the real reading rather than on a fixture.
    """
    surface = analyse_environment(swebench_harness, rung=0)
    assert is_sensitive(surface)

    with pytest.raises(DisclosureRequired, match="no recorded decision"):
        surface.for_publication()

    reduced = redact(surface)
    assert not is_sensitive(reduced)
    assert reduced.by_kind == surface.by_kind
    assert reduced.headline == surface.headline
    rendered = reduced.render(include_targets=True)
    assert "get_logs_eval" in rendered, "which function, so the count stays attributable"
    assert not any(
        a.target_is_literal and a.target in rendered
        for a in surface.accesses
        if a.target_is_literal
    )

    # And the flag is on the evidence row, not only on the payload.
    reading = AttackSurfaceInventory(swebench_harness, rung=0).estimate()
    assert not isinstance(reading, Refusal), getattr(reading, "render", lambda: reading)()
    assert reading.subject.extra["sensitive"] == "true"


def test_d8_rung_two_refuses_to_infer_a_cost_ratio_from_source(swebench_harness) -> None:
    """*Every number came from code that ran.* Reading cheapness off an inventory is a guess."""
    reading = AttackCheapnessRatio(verifier=swebench_harness).estimate()
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "This rung is a measurement" in reading.remedy

    # Given measurements it computes the ratio and puts an interval on it.
    measured = attack_cheapness(
        CostSample(what="write a passing log file", unit="tokens", values=(210.0, 190.0, 205.0)),
        CostSample(what="fix the bug", unit="tokens", values=(2300.0, 2100.0, 2500.0)),
        seed=0,
    )
    assert isinstance(measured, AttackCheapness)
    assert measured.ratio > 1.0
    assert measured.cheaper_to_attack is True
    assert measured.ci_low > 1.0


def test_d8_refuses_across_a_unit_boundary_rather_than_converting() -> None:
    """`env.attack_cheapness` declares the `units` group, whose assertion is a refusal."""
    refusal = attack_cheapness(
        CostSample(what="attack", unit="tokens", values=(200.0,)),
        CostSample(what="solve", unit="seconds", values=(45.0,)),
    )
    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.UNIT_MISMATCH


def test_d8_names_the_boundary_crossings_when_a_boundary_is_declared() -> None:
    """Rung 1 on a fixture, because a boundary is a declaration about an environment.

    SWE-bench's real boundary would have to be written by somebody who runs it; making one up and
    reporting crossings against it would be a measurement of the invention.
    """
    source = textwrap.dedent('''
        """A harness with the SWE-bench shape and a boundary somebody wrote down."""

        import json
        import os

        GOLD = "/opt/gold/answers.json"


        def grade(submission_path):
            token = os.environ["EVAL_API_TOKEN"]
            with open("/workspace/run.log") as fh:
                produced = fh.read()
            with open(GOLD) as g:
                gold = json.load(g)
            if "ALL TESTS PASSED" in produced:
                return 1.0
            return 0.0 if gold else 0.0
    ''')
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "harness.py"
        path.write_text(source, encoding="utf-8")
        verifier = VerifierUnderTest(path, entrypoint="grade")
        boundary = TaskBoundary(
            agent_writable=("/workspace/*",),
            harness_private=("/opt/gold/*",),
            secret=("*TOKEN*", "*API_KEY*"),
            name="swe-shaped",
        )
        surface = analyse_environment(verifier, rung=1, boundary=boundary)

    rules = {c.rule for c in surface.crossings}
    assert "reads inside the agent-writable region" in rules, surface.render(include_targets=True)
    assert "reaches a declared secret" in rules
    assert any("produced" in t.chain for t in surface.unchecked_taints)
    assert not any("gold" in t.chain for t in surface.taints), (
        "the harness reading its own answer key is not a finding once the boundary says so"
    )


# ---------------------------------------------------------------------------
# Clause 3 — D10 on a real record
# ---------------------------------------------------------------------------


def test_d10_measures_replay_fidelity_on_a_real_public_verifier(math_verifier) -> None:
    """*Replaying the recorded trajectory reproduces the score.*

    The measured result on `is_equiv` is 1.00 over thirty tasks, deterministic across three
    repeats, and **that is this instrument's kill condition arriving on this grader.** It is also
    the expected answer: `is_equiv` is a pure function of two strings with no clock, no filesystem
    and no subprocess, so there is nothing for a replay to disagree with. The number is cheap and
    true, and reporting it is what lets the three instruments that require `STATIONARY_GRADER`
    admit on this grader instead of declining for want of a measurement.
    """
    fn = math_verifier.load()

    def grade(str1: str, str2: str) -> float:
        return 1.0 if fn(str1=str1, str2=str2) else 0.0

    recorded = ListCorpus.of(
        [
            Rollout(id=f"m{i:02d}", inputs={"str1": a, "str2": b}, score=grade(a, b))
            for i, (a, b) in enumerate(MATH_PAIRS)
        ]
    )
    report = replay_corpus(grade, recorded, repeats=3)
    assert isinstance(report, ReplayReport), getattr(report, "render", lambda: report)()

    assert report.n_attempted == len(MATH_PAIRS)
    assert report.replay_fidelity == 1.0
    assert report.n_nondeterministic == 0
    assert report.deterministic_fraction == 1.0
    assert report.unauditable == ()

    condition = report.condition_reading()
    assert condition.holds is True
    assert condition.statistic == 1.0


def test_d10_finds_the_tasks_a_changed_grader_makes_unauditable(math_verifier) -> None:
    """The instrument has range: the same record against a grader that normalises differently.

    The change is one line, folding U+2212 MINUS SIGN onto ASCII hyphen, which is the normalisation
    gap D4 measured in this same checker. A record produced before that fix and read after it
    is a record whose scores cannot all be reproduced, and the tasks where it fails are named.
    """
    fn = math_verifier.load()

    def original(str1: str, str2: str) -> float:
        return 1.0 if fn(str1=str1, str2=str2) else 0.0

    def patched(str1: str, str2: str) -> float:
        fold = str.maketrans({"−": "-"})
        return original(str1.translate(fold), str2.translate(fold))

    pairs = [*MATH_PAIRS, ("−3", "-3"), ("−2", "- 2")]
    recorded = ListCorpus.of(
        [
            Rollout(id=f"p{i:02d}", inputs={"str1": a, "str2": b}, score=original(a, b))
            for i, (a, b) in enumerate(pairs)
        ]
    )
    report = replay_corpus(patched, recorded, repeats=2)
    assert isinstance(report, ReplayReport)

    assert 0.0 < report.replay_fidelity < 1.0
    assert report.n_mismatched >= 1
    unauditable = {t.id for t in report.unauditable}
    assert unauditable, report.render()

    # The record's own vocabulary carries the failures out, with remedies attached.
    refs = report.absent_refs()
    assert refs and all(r.remedy.strip() for r in refs)

    # And the envelope condition three other instruments declare now reads FAIL rather than unknown.
    condition = report.condition_reading()
    assert condition.holds is False
    assert condition.statistic == pytest.approx(report.replay_fidelity)


# ---------------------------------------------------------------------------
# The two gates every instrument in this library passes
# ---------------------------------------------------------------------------


def test_lint_is_empty_for_all_four_instruments(math_verifier, swebench_harness) -> None:
    """*Lint is the gate.* An instrument that cannot pass it does not merge."""
    log = _math_exploit_log(math_verifier)
    instruments = [
        ExploitFamilyCoverage(log),
        ReliabilityGrowth(log),
        AttackSurfaceInventory(swebench_harness),
        AttackCheapnessRatio(),
        ReplayFidelity(math_verifier.load(), ListCorpus.of([])),
    ]
    for inst in instruments:
        assert lint_instrument(inst) == [], inst.name


def test_the_generated_invariance_test_passes_for_all_four(math_verifier, swebench_harness) -> None:
    """*No instrument merges without its generated invariance test passing.*

    Four of the five declare `none`, which resolves to the trivial group: no affine rescaling of
    the reward acts on a count of exploit families, on the exponent of their arrival process, on an
    inventory of file reads, or on the fraction of records that reproduce. That is an answer rather
    than an omission, per E11.

    `env.attack_cheapness` is the exception and its group is `units`, whose assertion is a refusal
    rather than a numeric relation, so `check_invariance` routes it away from a value comparison
    and `check_unit_refusal` is where the real assertion lives. It is exercised directly in
    `test_d8_refuses_across_a_unit_boundary_rather_than_converting` above.
    """
    log = _math_exploit_log(math_verifier)
    fn = math_verifier.load()

    trivial = [
        (ExploitFamilyCoverage(log), lambda i, _p: float(exploit_coverage(log).unseen_fraction)),
        (ReliabilityGrowth(log), lambda i, _p: float(exploit_coverage(log, rung=1).fit.beta)),
        (
            AttackSurfaceInventory(swebench_harness),
            lambda i, _p: float(analyse_environment(swebench_harness, rung=0).headline),
        ),
        (
            ReplayFidelity(fn, ListCorpus.of([])),
            lambda i, _p: 0.0,
        ),
    ]
    for inst, run in trivial:
        group = inst.invariance if inst.invariance != "none" else "trivial"
        report = check_invariance(inst, group, InvariancePayload(), n=4, run=run)
        assert report.passed, report.render()
        assert "trivial group" in report.skipped

    units = check_invariance(AttackCheapnessRatio(), "units", InvariancePayload(), n=4)
    assert units.passed
    assert "refusal" in units.skipped


# ---------------------------------------------------------------------------
# Debt series L: the two D6 defects the statistical review found, on this same log
# ---------------------------------------------------------------------------


def test_the_unseen_mass_is_labelled_chao1_and_good_turing_carries_esty(math_verifier) -> None:
    """Two numbers that were one, and an interval that did not exist.

    The reading used to print "Good-Turing bounds the unseen mass at X%" where X was the **Chao1
    richness fraction**, a count of families, while Good-Turing's unseen mass is the probability
    `f1/n`. They are not close: on a synthetic log of 100 finds with `f1 = 7` and `f2 = 3` they
    read 0.262 and 0.070, a factor of 3.7.

    And Good-Turing had no interval at all. On this real log it matters more than usual, because
    the point estimate is exactly zero: `f1 = 0` says no family was seen exactly once, so the
    estimated novelty probability is 0.000, and a reader takes that as "the search is exhausted".
    Esty's variance says the same forty-five finds are consistent with a novelty probability up to
    0.0871, which is a different conclusion about whether to keep fuzzing.
    """
    log = _math_exploit_log(math_verifier)
    reading = exploit_coverage(log, rung=2)
    assert isinstance(reading, ExploitCoverage)
    rendered = reading.render()

    # The label is on the right statistic now.
    assert "Chao1 puts 0.0% of the families still unseen" in rendered
    assert "Good-Turing puts the probability that the next find is novel at 0.000" in rendered
    assert "Good-Turing bounds the unseen mass" not in rendered

    # And the interval exists, and it is not the point estimate.
    assert reading.novelty_probability == 0.0
    lo, hi = reading.novelty_ci
    assert lo == 0.0
    assert hi == pytest.approx(2.0 * 1.959963984540054 / log.n, rel=1e-9)
    assert hi > 0.08, "a zero point estimate with an interval reaching 8.7% is the whole finding"
    assert f"[{lo:.3f}, {hi:.3f}]" in rendered


def test_the_least_squares_intervals_coverage_is_on_the_growth_reading_itself(
    math_verifier,
) -> None:
    """E42 recorded this as handled at the reading. It was handled on the wrong reading.

    The caveat about the least-squares interval lived in `ExploitCoverage.render`, so the D6 half
    that reports the unseen mass carried it and `ReliabilityGrowth`, whose entire quantity is beta,
    did not. It is now on `CrowFit.render`, which both go through.

    The number in it is measured rather than asserted: simulated on a homogeneous Poisson process,
    where beta is exactly 1, this interval contains the truth in 0.401, 0.308 and 0.219 of runs at
    8, 12 and 25 finds against a nominal 0.95, and calls the blacklist converging in 0.462, 0.533
    and 0.574 of them where a one-sided 2.5% claim should.
    """
    log = _math_exploit_log(math_verifier)
    growth = ReliabilityGrowth(log).estimate()
    rendered = growth.value.render()
    assert "narrower than its own level" in rendered
    assert "against a nominal 0.95" in rendered
    # On this log the interval spans 1, so there is no over-claim to correct and the caveat says
    # the honest interval is wider still, which is the direction that leaves the verdict standing.
    assert growth.value.converging_at_interval is None
    assert "spans it under any correction" in rendered

    # And the same text reaches the coverage half, through one implementation rather than two.
    coverage = exploit_coverage(log, rung=1)
    assert isinstance(coverage, ExploitCoverage)
    assert "narrower than its own level" in coverage.render()

    # A decisive fit gets the sentence that names the over-claim.
    decisive = CrowFit(
        beta=0.6,
        lam=1.0,
        ci_low=0.4,
        ci_high=0.8,
        ci_level=0.95,
        points=12,
        r_squared=0.99,
        fit_on="families",
        beta_mle=0.7,
        beta_unbiased=0.65,
    )
    assert decisive.converging_at_interval is True
    assert "rather than as the evidence" in decisive.render()


def test_crow_beta_is_none_checked_rather_than_truthiness_checked() -> None:
    """`beta_unbiased or beta_mle` falls through on a bias-corrected exponent of exactly 0.0."""
    fit = CrowFit(
        beta=0.5,
        lam=1.0,
        ci_low=0.1,
        ci_high=0.9,
        ci_level=0.95,
        points=4,
        r_squared=0.9,
        fit_on="families",
        beta_mle=0.7,
        beta_unbiased=0.0,
    )
    assert fit.crow_beta == 0.0
    assert (fit.beta_unbiased or fit.beta_mle) == 0.7, "the expression this replaced"

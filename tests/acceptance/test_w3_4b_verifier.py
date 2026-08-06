"""Acceptance: D3 metamorphic violations, D4 Sobol sensitivity, D5 false-positive fuzzing.

The overall clause covers all ten D instruments. The third of it is this package's:

    *at least one metamorphic violation with a reproducer, Sobol total-effect indices, and a
    false-positive catalogue.*

Discharged here against a small real verifier rather than against a mock. `math_grader.py` below is
written to disk by a fixture and is a working math-answer grader with the three layers a real one
has: a parse that pulls the answer out of a response, a normalisation that strips delimiters and
punctuation, and a threshold that turns a similarity into a score. It also has two real bugs,
planted because they are the two the literature reports most often: the comparison treats substring
containment as a match, and the normalisation misses U+2212 MINUS SIGN. Everything asserted below
is measured on it.

Four clauses, one test each, plus the two gates every instrument in this library passes:

1. a semantics-preserving transformation that changes the score produces a violation carrying a
   shrunk, rerunnable reproducer;
2. `S_Ti` returns bit-exact `0.0` for a deliberately non-influential rubric criterion;
3. the false-positive catalogue is produced and is flagged sensitive;
4. `crosshair` is routed by layer rather than pointed at the whole grader.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from reward_lens.core.invariance import INVARIANT, InvariancePayload, check_invariance
from reward_lens.core.reading import Refusal
from reward_lens.measure.base import lint_instrument
from reward_lens.verifier import Rollout, VerifierUnderTest, is_sensitive, redact
from reward_lens.verifier.fuzz import (
    DisclosureDecision,
    DisclosureRequired,
    FalsePositiveFuzzing,
    GraderLayer,
    LayerKind,
    StrictReference,
    crosshair_available,
    mutation_space,
    route_symbolic,
    run_crosshair,
)
from reward_lens.verifier.metamorphic import (
    MetamorphicViolations,
    answer_text_relations,
)
from reward_lens.verifier.sensitivity import (
    RubricInput,
    SobolSensitivity,
    sobol_sample,
    total_effect,
)

# ---------------------------------------------------------------------------
# The verifier under test
# ---------------------------------------------------------------------------

#: A working math-answer grader with a parse layer, a normalisation layer and a threshold, plus two
#: layers a symbolic checker cannot read. The two bugs are deliberate and are the two the
#: literature reports: containment counts as a match (the shape behind the published 0.832 math
#: false-positive rate), and the normalisation does not fold U+2212 onto U+002D, because NFKC does
#: not either. Both are checked below rather than asserted here.
GRADER_SOURCE = textwrap.dedent(
    r'''
    """A small math-answer grader: parse, normalise, compare, threshold."""

    import re
    import unicodedata

    BOXED = re.compile(r"\\boxed\{([^}]*)\}")
    CUTOFF = 0.5


    def parse_answer(response: str) -> str:
        """Pull the answer out of a response: the last boxed group, else the last line.

        post: isinstance(__return__, str)
        """
        found = BOXED.findall(response)
        if found:
            return found[-1]
        return response.strip().split("\n")[-1]


    def normalise(answer: str) -> str:
        """Strip whitespace, inline-math delimiters and a trailing full stop.

        post: __return__ == __return__.strip()
        """
        text = unicodedata.normalize("NFKC", answer).strip()
        if len(text) > 1 and text.startswith("$") and text.endswith("$"):
            text = text[1:-1]
        if text.startswith("\\(") and text.endswith("\\)"):
            text = text[2:-2]
        if text.endswith("."):
            text = text[:-1]
        return text.strip()


    def similarity(answer: str, gold: str) -> float:
        """How close the answer is to the gold. Containment counts as a full match."""
        a = normalise(answer)
        g = normalise(gold)
        if not g:
            return 1.0
        if a == g:
            return 1.0
        if g in a:
            return 1.0
        return 0.0


    def threshold(similarity_score: float, cutoff: float) -> float:
        """Turn a similarity into a score.

        pre: 0.0 <= similarity_score <= 1.0
        pre: 0.0 <= cutoff <= 1.0
        post: __return__ in (0.0, 1.0)
        """
        if similarity_score > cutoff:
            return 1.0
        return similarity_score


    def symbolic_equal(answer: str, gold: str) -> bool:
        """A computer-algebra equivalence check. Not symbolically executable."""
        import sympy

        try:
            return bool(sympy.simplify(sympy.sympify(answer) - sympy.sympify(gold)) == 0)
        except (sympy.SympifyError, TypeError, ValueError):
            return False


    def run_unit_tests(solution: str, tests: str) -> float:
        """Stands in for a harness that shells out to a container."""
        raise NotImplementedError("this layer executes code in a sandbox")


    def grade(response: str, gold: str) -> float:
        return threshold(similarity(parse_answer(response), gold), CUTOFF)
    '''
).strip()


def _def_line(source: str, name: str) -> int:
    for number, line in enumerate(source.split("\n"), start=1):
        if line.startswith(f"def {name}("):
            return number
    raise AssertionError(f"no top-level def {name} in the fixture grader")


@pytest.fixture(scope="module")
def grader_path(tmp_path_factory) -> object:
    path = tmp_path_factory.mktemp("w3_4b") / "math_grader.py"
    path.write_text(GRADER_SOURCE, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def verifier(grader_path) -> VerifierUnderTest:
    return VerifierUnderTest(source_path=grader_path, entrypoint="grade")


@pytest.fixture(scope="module")
def grade(verifier) -> object:
    return verifier.load()


@pytest.fixture(scope="module")
def corpus() -> list[Rollout]:
    """Accepted answers, three of them negative so the minus-sign relation has something to act on."""
    pairs = [
        ("42", "42"),
        ("-7", "-7"),
        ("The answer is 3/4", "3/4"),
        ("-1/2", "-1/2"),
        (r"\boxed{17}", "17"),
        ("-13", "-13"),
        ("2.5", "2.5"),
        ("x = 8", "8"),
    ]
    return [
        Rollout(id=f"m{i}", inputs={"response": response, "gold": gold})
        for i, (response, gold) in enumerate(pairs)
    ]


@pytest.fixture(scope="module")
def layers(grader_path) -> list[GraderLayer]:
    """The grader's five layers, each with what a symbolic checker would be pointed at."""
    source = GRADER_SOURCE
    return [
        GraderLayer(
            "parse",
            LayerKind.PARSE,
            target=f"{grader_path}:{_def_line(source, 'parse_answer')}",
        ),
        GraderLayer(
            "normalise",
            LayerKind.NORMALISE,
            target=f"{grader_path}:{_def_line(source, 'normalise')}",
        ),
        GraderLayer(
            "threshold",
            LayerKind.THRESHOLD,
            target=f"{grader_path}:{_def_line(source, 'threshold')}",
        ),
        GraderLayer("symbolic_equal", LayerKind.EQUIVALENCE, target=None),
        GraderLayer("run_unit_tests", LayerKind.EXECUTION, target=None),
    ]


@pytest.fixture(scope="module")
def strict_reference() -> StrictReference:
    """Exact equality after NFKC and stripping. No containment, no tolerance, no delimiters."""

    def decide(response: str, gold: str, **_: object) -> bool:
        import unicodedata

        return (
            unicodedata.normalize("NFKC", response).strip()
            == unicodedata.normalize("NFKC", gold).strip()
        )

    return StrictReference(
        name="exact_nfkc",
        decide=decide,
        basis=(
            "exact string equality after Unicode NFKC normalisation and whitespace stripping, "
            "with no substring matching, no numeric tolerance and no delimiter handling"
        ),
    )


# ---------------------------------------------------------------------------
# Clause 1 — D3
# ---------------------------------------------------------------------------


def test_a_semantics_preserving_transformation_that_changes_the_score_has_a_shrunk_reproducer(
    verifier, grade, corpus
):
    """*At least one metamorphic violation with a reproducer.*

    The relation that fires is the minus sign. U+2212 MINUS SIGN is what LaTeX renders and what
    models emit; U+002D HYPHEN-MINUS is what a keyboard types. NFKC does not fold one onto the
    other, which is checked in `test_the_relation_that_fires_is_a_real_property_of_unicode` below
    rather than assumed, so a grader that normalises with NFKC and compares strings treats the same
    number written two ways as two different answers.

    The reproducer has to survive three things to count: it must be shrunk by `hypothesis.find`
    rather than being the first census hit, it must rerun against a freshly loaded grader, and it
    must still violate when it does.
    """
    inst = MetamorphicViolations(verifier, corpus, answer_text_relations(on="response"))
    reading = inst.estimate()
    assert not isinstance(reading, Refusal), getattr(reading, "render", lambda: reading)()

    report = reading.value
    assert report.violations >= 1, report.render()
    assert report.baseline_identity_violations == 0

    firing = [r for r in report.relations if r.violations]
    assert [r.relation for r in firing] == ["unicode_minus_invariance"], report.render()

    shrunk = report.shrunk_reproducers
    assert shrunk, "a violated relation must carry a shrunk reproducer"
    reproducer = shrunk[0]
    assert reproducer.shrunk is True
    assert reproducer.source == "hypothesis.find"

    # Re-runnable: a fresh load of the grader from disk reproduces both scores and the violation.
    before, after = reproducer.rerun(verifier)
    assert (before, after) == (reproducer.score_before, reproducer.score_after)
    assert reproducer.still_violates(grade)
    assert before == 1.0 and after == 0.0

    # And the reproducer says what the transformation claimed to preserve, so the violation can be
    # argued with rather than only counted.
    assert "U+2212" in reproducer.preserves


def test_the_relation_that_fires_is_a_real_property_of_unicode():
    """The premise behind clause 1, checked rather than assumed."""
    import unicodedata

    assert unicodedata.normalize("NFKC", "\u221242") != "-42"


def test_an_inapplicable_relation_is_reported_as_inapplicable_and_not_as_a_pass(verifier, corpus):
    """Rung 0 is written for source. Pointed at answers it transforms nothing, and says so.

    This is the failure mode that would make D3 useless: four relations that cannot fire report
    zero violations, and zero violations reads as a clean grader.
    """
    from reward_lens.verifier.metamorphic import python_source_relations

    inst = MetamorphicViolations(verifier, corpus, python_source_relations(on="response"))
    reading = inst.estimate()
    report = reading.value
    assert report.pairs_applicable == 0
    assert set(report.inapplicable_relations) == {
        "whitespace_invariance",
        "comment_invariance",
        "rename_invariance",
        "reorder_invariance",
    }
    assert np.isnan(report.violation_rate), "no applicable pairs means no rate, not a rate of zero"


def test_a_grader_that_disagrees_with_itself_is_refused_below_lod(corpus):
    """Refusal is a value. A violation under the identity transformation is flakiness, not a finding.

    The flaky grader here draws from a seeded generator, so the test is reproducible while the
    grader genuinely disagrees with itself between two calls on the same input. A grader whose
    disagreement is *periodic* with the corpus length would pass the probe, which is a stated
    limitation of a single re-call rather than a hidden one: `env.flakiness` measures it properly
    over twenty replays and this is a one-shot screen.
    """
    from random import Random as _Random

    rng = _Random(11)

    def flaky(response: str, gold: str) -> float:
        return rng.random()

    reading = MetamorphicViolations(flaky, corpus, answer_text_relations()).estimate()
    assert isinstance(reading, Refusal)
    assert reading.reason.name == "BELOW_LOD"
    assert reading.statistics["identity_violations"] > 0
    assert "deterministic" in reading.remedy


# ---------------------------------------------------------------------------
# Clause 2 — D4
# ---------------------------------------------------------------------------

#: A three-criterion rubric built to separate the three rungs. `formatting` cannot move the score
#: at all. `rigour` moves it only *in company with* `correctness`, through a term that is centred
#: on the midpoint of both, which is precisely where one-at-a-time takes its base point.
RUBRIC_INPUTS = [
    RubricInput("correctness", 0.0, 1.0, "does the answer match"),
    RubricInput("rigour", 0.0, 1.0, "is the derivation sound"),
    RubricInput("formatting", 0.0, 1.0, "deliberately weighted zero"),
]


def rubric(criteria) -> float:
    """`formatting` is multiplied by exactly 0.0 and reaches no other arithmetic.

    The second term is `3·(c − ½)·(r − ½)`. It vanishes when either input sits at its midpoint,
    which is where rung 0 fixes everything it is not sweeping, so sweeping `rigour` moves the score
    by exactly zero. Its conditional mean over `correctness` is also zero, so `S1` for `rigour` is
    zero too. Only the total effect sees it.
    """
    correctness = criteria["correctness"]
    rigour = criteria["rigour"]
    return (
        2.0 * correctness
        + 3.0 * (correctness - 0.5) * (rigour - 0.5)
        + 0.0 * criteria["formatting"]
    )


def test_sobol_total_effect_is_bit_exact_zero_for_a_non_influential_criterion():
    """*Sobol total-effect indices.* And the one that must be exact.

    `S_Ti == 0.0` bit-exact, not merely small. It holds here because the coefficient is literally
    `0.0`, so the A and AB sample matrices produce bit-identical output columns for that input and
    the total-order estimator's difference term is exactly zero rather than a float residue.

    **This does not generalise and the test is not written as though it does.** An input that is
    merely weakly influential, or one that is zero-influence but reached through arithmetic that
    perturbs the output, gives a small nonzero and possibly negative `S_Ti`. A test asserting
    `ST == 0.0` for irrelevance in general would be wrong; this one asserts it for a construction
    that makes it exact and says which construction.
    """
    inst = SobolSensitivity(rubric, RUBRIC_INPUTS, n_base=512, seed=0, name="fixture_rubric")
    reading = inst.estimate()
    assert not isinstance(reading, Refusal), getattr(reading, "render", lambda: reading)()
    profile = reading.value

    by_name = {i.name: i for i in profile.indices}
    assert by_name["formatting"].st == 0.0
    assert by_name["formatting"].s1 == 0.0
    assert by_name["formatting"].st_conf == 0.0
    assert profile.inert == ("formatting",)

    # The two that do matter are ordered and the dominant one carries most of the variance.
    assert by_name["correctness"].st > by_name["rigour"].st > 0.0
    assert profile.dominant.name == "correctness"

    # Rung 2 has bootstrap intervals and they compose as a GUM budget.
    assert by_name["correctness"].st_conf > 0.0
    budget = profile.budget()
    assert budget.combined > 0.0
    assert budget.dominant.name == "sobol_bootstrap"
    assert budget.effective_dof() == pytest.approx(99.0)


def test_rung_zero_is_computed_and_is_shown_to_be_biased():
    """One-at-a-time is *computed only to show it is biased*, so the bias has to be a number.

    On this rubric `rigour` carries a real and substantial share of the output variance and rung 0
    scores it at **exactly zero**, because its whole contribution is an interaction centred on the
    base point rung 0 sweeps through. Rung 1 misses it too: `S1` is the variance removed by
    learning one input on its own, and learning `rigour` alone removes none of it. Only the
    total-effect index sees it, which is why rung 2 is the one to quote.

    The `mu_prime` assertion is the caveat that travels to the contract layer with the number: a
    least-squares slope is a linear summary, so a component whose entire influence is an
    interaction has `μ' ≈ 0` while its total effect is a sixth of the variance. The
    equal-compensation check has to read both.
    """
    profile = (
        SobolSensitivity(rubric, RUBRIC_INPUTS, n_base=512, seed=0, name="fixture_rubric")
        .estimate()
        .value
    )
    by_name = {i.name: i for i in profile.indices}

    assert by_name["rigour"].oat_effect == 0.0, profile.render()
    assert by_name["rigour"].s1 == pytest.approx(0.0, abs=0.02), profile.render()
    assert by_name["rigour"].st > 0.1, profile.render()
    assert profile.interaction_mass > 0.2, profile.render()
    for index in profile.indices:
        assert index.st >= index.s1 - 1e-9
    assert by_name["correctness"].oat_effect > 0.0

    contract = {c.name: c for c in profile.contract_inputs()}
    assert contract["correctness"].mu_prime == pytest.approx(2.0, abs=0.05)
    assert contract["rigour"].mu_prime == pytest.approx(0.0, abs=0.05)
    assert contract["rigour"].total_effect > 0.1


def test_a_non_power_of_two_base_sample_is_refused_at_construction_and_names_the_next_one():
    """SALib warns and returns numbers anyway, which is easy to lose under a warning filter."""
    with pytest.raises(ValueError, match="1024"):
        SobolSensitivity(rubric, RUBRIC_INPUTS, n_base=1000)


def test_a_grader_with_no_output_variance_is_refused_rather_than_dividing_by_zero():
    reading = SobolSensitivity(lambda c: 1.0, RUBRIC_INPUTS, n_base=64).estimate()
    assert isinstance(reading, Refusal)
    assert reading.reason.name == "BELOW_LOD"
    assert reading.statistics["constant_score"] == 1.0


# ---------------------------------------------------------------------------
# Clause 3 — D5
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def catalogue_reading(verifier, strict_reference, corpus, layers):
    inst = FalsePositiveFuzzing(
        verifier,
        strict_reference,
        corpus,
        space=mutation_space(corpus, on="response"),
        layers=layers,
        max_examples=200,
        seed=0,
    )
    return inst.estimate()


def test_a_false_positive_catalogue_is_produced_and_is_flagged_sensitive(
    catalogue_reading, verifier, strict_reference
):
    """*A false-positive catalogue.* Produced, populated, rerunnable, and sensitive by default."""
    assert not isinstance(catalogue_reading, Refusal), catalogue_reading
    catalogue = catalogue_reading.value

    assert catalogue.false_positives > 0, catalogue.render()
    assert 0.0 < catalogue.false_positive_rate <= 1.0
    assert catalogue.reference_rejects >= catalogue.false_positives
    assert "substring_containment" in catalogue.by_family

    # Every entry reruns and still disagrees the same way.
    for entry in catalogue.entries[:20]:
        grader_score, reference_accepts = entry.rerun(verifier, strict_reference)
        assert grader_score == entry.grader_score
        assert reference_accepts is False

    # The flag is on the payload and on the evidence row's subject, not in a docs page.
    assert catalogue.sensitive is True
    assert is_sensitive(catalogue) and is_sensitive(catalogue_reading)
    assert catalogue_reading.subject.extra.get("sensitive") == "true"


def test_a_sensitive_catalogue_does_not_render_its_reproducers_by_default(catalogue_reading):
    catalogue = catalogue_reading.value
    rendered = catalogue.render()
    assert "reproducers withheld" in rendered
    for entry in catalogue.entries[:5]:
        assert repr(dict(entry.inputs)) not in rendered


def test_redaction_keeps_the_counts_and_drops_the_exploits(catalogue_reading):
    reduced = redact(catalogue_reading)
    assert reduced.value.entries == ()
    assert reduced.value.withheld == len(catalogue_reading.value.entries)
    assert reduced.value.false_positives == catalogue_reading.value.false_positives
    assert reduced.value.false_positive_rate == catalogue_reading.value.false_positive_rate
    assert is_sensitive(reduced) is False


def test_publishing_without_a_recorded_decision_raises(catalogue_reading):
    """*Never included in a published artifact without a decision recorded.*"""
    catalogue = catalogue_reading.value
    with pytest.raises(DisclosureRequired, match="no recorded decision"):
        catalogue.for_publication()

    decided = catalogue.with_disclosure(
        DisclosureDecision(
            decided_by="the grader's owner",
            reason="the fixes have shipped and the families are already in the literature",
            scope="published",
        )
    )
    assert decided.for_publication() is decided
    assert decided.disclosure.decided_at


def test_the_search_beats_its_declared_random_mutation_baseline(catalogue_reading):
    """The kill condition, operationalised: a search that random edits match has not earned its calls."""
    catalogue = catalogue_reading.value
    assert catalogue.baseline_random_mutation_hits >= 0
    assert catalogue.beats_baseline, catalogue.render()


def test_rung_one_produces_a_shrunk_entry_alongside_the_replayed_families(catalogue_reading):
    catalogue = catalogue_reading.value
    rungs = {entry.rung for entry in catalogue.entries}
    assert 0 in rungs, "rung 0 replays the known exploit families"
    assert 1 in rungs, "rung 1 is the hypothesis-driven search"
    shrunk = [entry for entry in catalogue.entries if entry.shrunk]
    assert shrunk and shrunk[0].source == "hypothesis.find"


def test_an_uncertified_reference_is_refused(verifier, corpus):
    reading = FalsePositiveFuzzing(
        verifier, StrictReference(name="unstated", decide=lambda **_: False), corpus
    ).estimate()
    assert isinstance(reading, Refusal)
    assert reading.reason.name == "REFERENCE_UNCERTIFIED"


def test_rung_two_is_declared_unreachable_rather_than_silently_skipped(catalogue_reading):
    """atheris is not in the verifier extra, so rung 2 did not run and the reading says so."""
    from reward_lens.verifier.fuzz import coverage_guided_search

    catalogue = catalogue_reading.value
    assert catalogue.coverage_guided_available is False
    assert "atheris" in catalogue.coverage_guided_gap
    with pytest.raises(ImportError, match="atheris"):
        coverage_guided_search()


# ---------------------------------------------------------------------------
# Clause 4 — D5 rung 3
# ---------------------------------------------------------------------------


def test_crosshair_is_routed_by_layer_rather_than_pointed_at_the_whole_grader(layers):
    """*`crosshair` is routed by layer.*

    Three of the five layers are pure Python over values a solver can model and are routed to
    crosshair. The sympy layer and the sandbox layer are routed away *with the reason recorded*,
    which is the part that matters: pointing crosshair at the whole file would spend the timeout
    inside sympy and come back `CANNOT_CONFIRM` for the grader, and a whole-grader
    `CANNOT_CONFIRM` is indistinguishable from a clean result to anyone reading an exit code.
    """
    routes = route_symbolic(layers)
    by_layer = {r.layer: r for r in routes}

    assert [r.layer for r in routes if r.applicable] == ["parse", "normalise", "threshold"]
    assert all(r.tool == "crosshair" for r in routes if r.applicable)

    assert by_layer["symbolic_equal"].applicable is False
    assert "sympy" in by_layer["symbolic_equal"].reason
    assert by_layer["run_unit_tests"].applicable is False
    assert "container" in by_layer["run_unit_tests"].reason


@pytest.mark.skipif(not crosshair_available(), reason="crosshair-tool is not installed")
def test_the_routed_layer_produces_a_real_counterexample_and_a_real_cannot_confirm(layers):
    """Two of the three outcomes, measured, and kept apart.

    The threshold layer's contract says the score is binary and it is not: a similarity exactly
    equal to the cutoff falls through and is returned unchanged. crosshair refutes it.

    The parse layer is unbounded in its input length, so crosshair cannot exhaust the paths and
    returns `CANNOT_CONFIRM`, which it renders as `info: Not confirmed.` and **only under
    `--report_all`**. Exit code 0 for both that and a real proof is why the status is parsed from
    the message rather than from the exit code.
    """
    by_name = {layer.name: layer for layer in layers}

    refuted = run_crosshair(by_name["threshold"], per_condition_timeout=3.0, timeout=90.0)
    assert refuted.status == "refuted", refuted.messages
    assert refuted.exit_code == 1
    assert any("threshold(" in message for message in refuted.messages)
    assert refuted.is_proof is False

    unconfirmed = run_crosshair(by_name["parse"], per_condition_timeout=2.0, timeout=90.0)
    assert unconfirmed.status == "cannot_confirm", unconfirmed.messages
    assert unconfirmed.exit_code == 0, "exit 0 does not mean verified"
    assert unconfirmed.is_proof is False


@pytest.mark.skipif(not crosshair_available(), reason="crosshair-tool is not installed")
def test_the_symbolic_layers_that_are_routed_away_are_never_invoked(
    verifier, strict_reference, corpus, layers
):
    """The instrument runs crosshair on the applicable layers only, which is the routing working."""
    inst = FalsePositiveFuzzing(
        verifier,
        strict_reference,
        corpus,
        layers=layers,
        max_examples=1,
        run_symbolic=True,
        per_condition_timeout=1.0,
    )
    catalogue = inst.estimate().value
    checked = {finding.layer for finding in catalogue.symbolic_findings}
    assert checked == {"parse", "normalise", "threshold"}
    assert "symbolic_equal" not in checked
    assert "run_unit_tests" not in checked
    assert catalogue.rung == 3


# ---------------------------------------------------------------------------
# The two gates every instrument passes
# ---------------------------------------------------------------------------


def test_lint_is_empty_for_all_three_instruments(verifier, strict_reference, corpus):
    """*Lint is the gate.* An instrument that cannot pass it does not merge."""
    instruments = [
        MetamorphicViolations(verifier, corpus, answer_text_relations()),
        SobolSensitivity(rubric, RUBRIC_INPUTS, n_base=64),
        FalsePositiveFuzzing(verifier, strict_reference, corpus),
    ]
    for inst in instruments:
        assert lint_instrument(inst) == [], inst.name


def test_the_generated_invariance_test_passes_for_all_three(verifier, strict_reference, corpus):
    """*No instrument merges without its generated invariance test passing.*

    D4's group is `reward.affine` and its check is not vacuous: the Sobol' design is evaluated
    once, the outputs are rescaled by the drawn `a` and shifted by the drawn `b`, and the indices
    are recomputed from the transformed outputs. They must not move, and they do not, because SALib
    centres and scales the output internally before estimating.

    D3 and D5 declare `none`, which resolves to the trivial group. Their generated test passes
    because nothing acts on a violation count or on a threshold-crossing rate, which is an answer
    rather than an omission: see E11.
    """
    inputs = RUBRIC_INPUTS
    x = sobol_sample(inputs, 256, seed=0)
    y = np.array([rubric(dict(zip([i.name for i in inputs], row))) for row in x])

    report = check_invariance(
        SobolSensitivity(rubric, inputs, n_base=256),
        "reward.affine",
        InvariancePayload(scores=y),
        n=16,
        relation=INVARIANT,
        run=lambda inst, payload: float(total_effect(inputs, np.asarray(payload.scores))[0]),
    )
    assert report.passed, report.render()
    assert report.n == 16 and report.max_deviation < report.tol

    for inst in (
        MetamorphicViolations(verifier, corpus, answer_text_relations()),
        FalsePositiveFuzzing(verifier, strict_reference, corpus),
    ):
        trivial = check_invariance(inst, inst.invariance, InvariancePayload(), n=4)
        assert trivial.passed
        assert "trivial group" in trivial.skipped

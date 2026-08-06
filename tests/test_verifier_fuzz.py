"""Unit tests for D5, `verifier.false_positive_rate` and `verifier.fp_catalogue`.

The parts worth testing away from the clause are the ones where being wrong is quiet.
The false-positive rate has a denominator, and a search that reports only its hits loses every true
negative and drives the rate towards one. The sensitive flag has to survive a store round trip,
because a flag that lives only in memory is a convention. `crosshair` exits 0 for both a proof and
a failure to find a counterexample, so the status has to come from the message rather than the exit
code. And the declared baseline has to be capable of beating the search, or it is decoration.

The last test in this file points D3 and D5 at a **real published verifier** rather than at a
fixture. It is skipped unless the source is on disk, so the suite stays offline; the recorded
numbers from running it are in the report that accompanied this package.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import unicodedata

import pytest

from reward_lens.core.reading import Refusal
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.verifier import Rollout, is_sensitive, redact
from reward_lens.verifier.fuzz import (
    ATHERIS_GAP,
    SYMBOLIC_TRACTABLE,
    DisclosureDecision,
    DisclosureRequired,
    FalsePositiveFuzzing,
    GraderLayer,
    LayerKind,
    StrictReference,
    answer_exploit_families,
    atheris_available,
    crosshair_available,
    mutation_space,
    route_symbolic,
    run_crosshair,
)

# ---------------------------------------------------------------------------
# A leaky grader and a strict reference
# ---------------------------------------------------------------------------


def leaky(response: str, gold: str) -> float:
    """Containment counts as a match, which is the shape behind the published math FPR of 0.832."""
    a = response.strip()
    g = gold.strip()
    if not g:
        return 1.0
    return 1.0 if a == g or g in a else 0.0


def exact(response: str, gold: str, **_: object) -> bool:
    return (
        unicodedata.normalize("NFKC", response).strip()
        == unicodedata.normalize("NFKC", gold).strip()
    )


REFERENCE = StrictReference(
    name="exact_nfkc",
    decide=exact,
    basis="exact equality after Unicode NFKC normalisation and whitespace stripping",
)

SEEDS = [
    Rollout(id="s0", inputs={"response": "42", "gold": "42"}),
    Rollout(id="s1", inputs={"response": "-7", "gold": "-7"}),
    Rollout(id="s2", inputs={"response": "3/4", "gold": "3/4"}),
]


def catalogue(**kwargs: object):
    inst = FalsePositiveFuzzing(leaky, REFERENCE, SEEDS, **kwargs)
    reading = inst.estimate()
    assert not isinstance(reading, Refusal), reading
    return reading


# ---------------------------------------------------------------------------
# The dependencies, pinned by file rather than by name
# ---------------------------------------------------------------------------


def test_crosshair_is_the_package_we_think_it_is():
    """`crosshair-tool` on PyPI installs a package importable as `crosshair`."""
    import importlib.metadata

    crosshair = pytest.importorskip("crosshair")
    assert crosshair.__file__.endswith("crosshair/__init__.py")
    assert importlib.metadata.version("crosshair-tool") >= "0.0.100"

    from crosshair.statespace import MessageType

    assert MessageType.CANNOT_CONFIRM.value == "cannot_confirm"
    assert MessageType.CONFIRMED.value == "confirmed"


def test_atheris_is_absent_and_the_module_says_so_rather_than_guessing():
    assert atheris_available() is False
    assert "atheris" in ATHERIS_GAP
    assert "rung-1 reading" in ATHERIS_GAP


# ---------------------------------------------------------------------------
# The rate has a denominator
# ---------------------------------------------------------------------------


def test_the_rate_is_over_what_the_reference_rejects_and_not_over_every_trial():
    """`FP / (FP + TN)`, which is the definition the published 0.832 and 0.869 figures use."""
    cat = catalogue().value
    assert cat.false_positives > 0
    assert cat.reference_rejects >= cat.false_positives
    assert cat.false_positive_rate == pytest.approx(cat.false_positives / cat.reference_rejects)
    assert cat.trials >= cat.reference_rejects


def test_the_search_pass_contributes_its_true_negatives_to_the_denominator():
    """The bug this test exists to catch: reporting only the hits inflates the rate towards one."""
    without = catalogue().value
    with_search = catalogue(space=mutation_space(SEEDS), max_examples=200).value

    assert with_search.trials > without.trials + 100
    assert with_search.reference_rejects > without.reference_rejects
    assert with_search.false_positive_rate < 1.0


def test_a_reference_that_accepts_what_the_grader_rejects_is_reported_not_swallowed():
    """The reference is assumed *stricter*, not correct, and that assumption is checkable."""

    def picky(response: str, gold: str) -> float:
        return 0.0

    reading = FalsePositiveFuzzing(picky, REFERENCE, SEEDS).estimate()
    cat = reading.value
    assert cat.reference_disagreements > 0
    assert "not ordered" in cat.render()


# ---------------------------------------------------------------------------
# Rungs
# ---------------------------------------------------------------------------


def test_rung_zero_replays_named_families_that_carry_their_citation():
    families = answer_exploit_families()
    assert {f.name for f in families} == {
        "partial_answer",
        "substring_containment",
        "empty_answer",
        "prompt_echo",
        "numeric_tolerance",
    }
    for family in families:
        assert family.citation, f"{family.name} has no citation"

    cat = catalogue().value
    assert set(cat.by_family) <= {f.name for f in families}
    assert "substring_containment" in cat.by_family
    assert all(entry.rung == 0 for entry in cat.entries)


def test_rung_one_adds_a_shrunk_entry_and_marks_it_as_the_minimal_one():
    cat = catalogue(space=mutation_space(SEEDS), max_examples=200).value
    shrunk = [entry for entry in cat.entries if entry.shrunk]
    assert len(shrunk) == 1, "one shrunk example per search space, not one per hit"
    assert shrunk[0].source == "hypothesis.find"
    assert shrunk[0].rung == 1


def test_the_declared_baseline_is_computed_and_can_win():
    """A baseline that cannot beat the search is decoration, so it has to be a real comparison."""
    cat = catalogue(space=mutation_space(SEEDS), max_examples=200).value
    assert cat.baseline_random_mutation_hits >= 0
    assert 0.0 <= cat.baseline_random_mutation_fpr <= 1.0
    assert cat.beats_baseline == (cat.false_positive_rate > cat.baseline_random_mutation_fpr)


def test_rung_two_raises_a_typed_error_naming_the_extra_rather_than_degrading():
    from reward_lens.core.extras import ExtraRequiredError
    from reward_lens.verifier.fuzz import coverage_guided_search

    with pytest.raises(ExtraRequiredError) as caught:
        coverage_guided_search()
    assert "atheris" in str(caught.value)
    assert "verifier" in str(caught.value)


# ---------------------------------------------------------------------------
# Routing, and the three symbolic outcomes
# ---------------------------------------------------------------------------


def test_only_three_layer_kinds_are_symbolically_tractable():
    assert SYMBOLIC_TRACTABLE == {LayerKind.PARSE, LayerKind.NORMALISE, LayerKind.THRESHOLD}
    assert LayerKind.EQUIVALENCE not in SYMBOLIC_TRACTABLE
    assert LayerKind.EXECUTION not in SYMBOLIC_TRACTABLE


def test_a_tractable_layer_with_no_entry_point_is_routed_away_with_an_actionable_reason():
    """A grader whose parse and threshold are one inlined function cannot be checked a layer apart."""
    routes = route_symbolic([GraderLayer("inlined", LayerKind.THRESHOLD, target=None)])
    assert routes[0].applicable is False
    assert "Extract it into its own function" in routes[0].reason


@pytest.mark.skipif(not crosshair_available(), reason="crosshair-tool is not installed")
def test_a_layer_with_no_contract_is_not_reported_as_confirmed(tmp_path):
    """Exit 0 with nothing to say means no contract was found, which is not a proof of anything."""
    module = tmp_path / "no_contract.py"
    module.write_text("def scale(x: float) -> float:\n    return 2.0 * x\n", encoding="utf-8")
    finding = run_crosshair(
        GraderLayer("scale", LayerKind.THRESHOLD, target=str(module)),
        per_condition_timeout=1.0,
        timeout=90.0,
    )
    assert finding.status == "no_contract"
    assert finding.is_proof is False


@pytest.mark.skipif(not crosshair_available(), reason="crosshair-tool is not installed")
def test_a_confirmed_layer_is_the_only_status_that_counts_as_a_proof(tmp_path):
    module = tmp_path / "bounded.py"
    module.write_text(
        "def clamp(x: int) -> int:\n"
        '    """\n'
        "    pre: 0 <= x <= 10\n"
        "    post: 0 <= __return__ <= 10\n"
        '    """\n'
        "    return x\n",
        encoding="utf-8",
    )
    finding = run_crosshair(
        GraderLayer("clamp", LayerKind.THRESHOLD, target=str(module)),
        per_condition_timeout=5.0,
        timeout=90.0,
    )
    assert finding.status == "confirmed"
    assert finding.exit_code == 0
    assert finding.is_proof is True


# ---------------------------------------------------------------------------
# Dual use
# ---------------------------------------------------------------------------


def test_the_sensitive_flag_survives_a_store_round_trip(tmp_path):
    """A flag that lives only in memory is a convention. This one is on the row.

    The subject extra is hashed into the evidence id, so a redacted row and its unredacted original
    are two rows rather than one row with two histories.
    """
    from reward_lens.core.store import EvidenceStore

    reading = catalogue()
    store = EvidenceStore(tmp_path)
    store.append(reading)
    restored = store.get(reading.id)

    assert is_sensitive(restored)
    assert restored.value.sensitive is True
    assert restored.subject.extra.get("sensitive") == "true"
    assert restored.value.false_positives == reading.value.false_positives
    assert len(restored.value.entries) == len(reading.value.entries)
    assert dict(restored.value.entries[0].inputs) == dict(reading.value.entries[0].inputs)


def test_redaction_is_idempotent_and_the_redacted_row_is_no_longer_sensitive():
    reading = catalogue()
    once = redact(reading)
    twice = redact(once)
    assert once.value.entries == ()
    assert twice.value.entries == ()
    assert is_sensitive(once) is False
    assert once.value.withheld == len(reading.value.entries)


def test_a_disclosure_decision_needs_a_person_and_a_reason():
    with pytest.raises(ValueError, match="a person and a reason"):
        DisclosureDecision(decided_by="", reason="because")
    with pytest.raises(ValueError, match="a person and a reason"):
        DisclosureDecision(decided_by="someone", reason="   ")


def test_publication_is_refused_until_a_decision_exists():
    cat = catalogue().value
    with pytest.raises(DisclosureRequired):
        cat.for_publication()
    signed = cat.with_disclosure(
        DisclosureDecision(decided_by="grader owner", reason="fixed upstream", scope="published")
    )
    assert signed.for_publication().entries == cat.entries


def test_a_refuted_symbolic_finding_is_withheld_from_the_redacted_view():
    """A crosshair counterexample is a working input that breaks the layer. Same policy."""
    from reward_lens.verifier.fuzz import SymbolicFinding

    cat = catalogue().value
    with_finding = type(cat)(
        **{
            **{f.name: getattr(cat, f.name) for f in cat.__dataclass_fields__.values()},
            "symbolic_findings": (
                SymbolicFinding(
                    layer="threshold",
                    target="grader.py:1",
                    status="refuted",
                    messages=("error: false when calling threshold(0.25, 0.25)",),
                    exit_code=1,
                    seconds=0.4,
                ),
            ),
        }
    )
    reduced = with_finding.redacted()
    assert reduced.symbolic_findings[0].status == "refuted"
    assert reduced.symbolic_findings[0].messages == ("<withheld: contains a counterexample>",)


# ---------------------------------------------------------------------------
# Refusals and declarations
# ---------------------------------------------------------------------------


def test_an_unstated_reference_basis_is_refused_with_the_sentence_it_wants():
    reading = FalsePositiveFuzzing(
        leaky, StrictReference(name="mystery", decide=exact), SEEDS
    ).estimate()
    assert isinstance(reading, Refusal)
    assert reading.reason.name == "REFERENCE_UNCERTIFIED"
    assert "NFKC" in reading.remedy


def test_no_seeds_is_refused_because_searching_from_nothing_finds_nothing():
    reading = FalsePositiveFuzzing(leaky, REFERENCE, []).estimate()
    assert isinstance(reading, Refusal)
    assert reading.reason.name == "ACCESS_INSUFFICIENT"
    assert "kill condition" in reading.remedy


def test_the_instrument_declares_the_twelve_and_passes_lint():
    inst = FalsePositiveFuzzing(leaky, REFERENCE, SEEDS)
    assert lint_instrument(inst) == []
    assert inst.quantity == "verifier.false_positive_rate"
    assert inst.invariance == "trivial"
    assert inst.baselines == ("random_mutation",)
    assert inst.envelope is not None and inst.envelope.requires
    plan = inst.preflight(Context(signal=inst.subject))
    assert plan.ok and plan.cost.calls > 0


# ---------------------------------------------------------------------------
# A real verifier, not a fixture
# ---------------------------------------------------------------------------

#: Point this at a checkout of a published grader to run the probe below. It is deliberately not a
#: network fetch: a test that reaches the internet fails in the wrong way when it fails.
REAL_VERIFIER = os.environ.get("REWARD_LENS_REAL_VERIFIER", "")


@pytest.mark.skipif(
    not REAL_VERIFIER or not pathlib.Path(REAL_VERIFIER).is_file(),
    reason="set REWARD_LENS_REAL_VERIFIER to a module exposing is_equiv(str, str) -> bool",
)
def test_d3_and_d5_run_against_a_real_published_verifier():
    """A synthetic fixture proves the code runs; a real verifier proves the instrument measures.

    Written against `hendrycks/math`'s `modeling/math_equivalence.py`, whose `is_equiv` normalises
    LaTeX and compares strings and is used as the grader in a long line of RLVR work. What it found
    when this was run is recorded in the package report: 24 metamorphic violations over 34
    applicable pairs, and one false positive against exact equality, shrunk to
    `is_equiv('.5', '0.5')`.
    """
    from reward_lens.verifier.metamorphic import MetamorphicViolations, answer_text_relations

    spec = importlib.util.spec_from_file_location("_real_verifier", REAL_VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def grade(response: str, gold: str) -> float:
        return 1.0 if module.is_equiv(response, gold) else 0.0

    pairs = [
        ("42", "42"),
        ("-7", "-7"),
        ("0.5", "0.5"),
        ("-1/2", "-1/2"),
        ("17", "17"),
        (r"\frac{3}{4}", r"\frac{3}{4}"),
    ]
    corpus = [
        Rollout(id=f"r{i}", inputs={"response": r, "gold": g}) for i, (r, g) in enumerate(pairs)
    ]
    assert all(grade(**dict(c.inputs)) == 1.0 for c in corpus), "the seeds must be accepted"

    report = MetamorphicViolations(grade, corpus, answer_text_relations()).estimate().value
    assert report.baseline_identity_violations == 0
    assert report.violations > 0, report.render()
    assert report.shrunk_reproducers
    assert report.shrunk_reproducers[0].still_violates(grade)

    cat = (
        FalsePositiveFuzzing(
            grade, REFERENCE, corpus, space=mutation_space(corpus), max_examples=800, seed=0
        )
        .estimate()
        .value
    )
    assert cat.trials > 100
    assert cat.reference_rejects > 0
    assert cat.sensitive is True

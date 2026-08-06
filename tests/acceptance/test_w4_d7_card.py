"""W4 D7 acceptance: the grader card, on a real public verifier.

The clause, in full:

    *On one real grader reachable from this repository, `GraderCard` renders every field the
    catalogue names, each as a reading or a refusal with its remedy, with the trust level computed
    and the dual-use fields withheld by default. The same card preflights with zero grader calls
    and reports its cost. A second card on a grader with strictly less access renders strictly
    more refusals, and a test asserts that ordering rather than asserting a fixed count.*

The subject is `is_equiv` from `hendrycks/math`, the answer-equivalence checker the MATH benchmark
ships and the one most open RLVR math pipelines still call, directly or through a fork. It is the
same subject the D packages point at, reused deliberately so the D-series numbers on this
grader accumulate against one program rather than against four.

It is fetched over the network and the tests skip with a message when there is none, because a
skipped real-subject test leaves the claim unproven rather than proven. `tests/test_measure_card.py`
proves the composition runs against a written-to-disk fixture and needs no network; this file
proves it measures.

**What this card is expected to refuse, and why that is the result rather than a gap.** `is_equiv`
takes two strings and has no numeric input space, so D4 has no sensitivity profile to decompose.
Nobody has built a reference for it that is both stricter and correct, so D5 has nothing to measure
false positives against. It is a program, so B1's curl mass and B2's Afriat index do not apply to it
at all. And nothing recorded its per-leaf scores, so B4 has no abstention channel to count. Four of
those five are things a reader could go and fix, and the card says which.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from random import Random
from typing import Any

import numpy as np
import pytest

from reward_lens.core.gates import compute_trust
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Access, Component, Phase, Substrate, TrustLevel
from reward_lens.measure.base import lint_instrument
from reward_lens.measure.card import (
    CARD_FIELDS,
    CardInputs,
    CardReading,
    GraderCard,
    card_context,
    render_card,
)
from reward_lens.measure.metrology.flakiness import ReplaySet
from reward_lens.measure.metrology.gstudy import GroupScores, ReplicationDesign
from reward_lens.verifier import (
    ExploitFind,
    ExploitLog,
    ListCorpus,
    Rollout,
    VerifierUnderTest,
    answer_text_relations,
    is_sensitive,
)

#: `is_equiv` from `hendrycks/math`. The same URL the D packages fetch.
REAL_VERIFIER_URL = (
    "https://raw.githubusercontent.com/hendrycks/math/main/modeling/math_equivalence.py"
)

#: Answer pairs in the shapes the MATH benchmark actually produces: fraction spellings, LaTeX
#: wrappers, trailing whitespace, unit text, and a handful the checker gets wrong.
REAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("1/2", "\\frac{1}{2}"),
    ("0.5", "1/2"),
    ("x", "\\text{x}"),
    ("2", "2.0"),
    ("\\frac{1}{2}", "0.5"),
    ("1", "1"),
    ("2.50", "2.5"),
    ("$5$", "5"),
    ("10", "10 "),
    ("\\dfrac{3}{4}", "\\frac{3}{4}"),
    ("1,000", "1000"),
    ("0", "0.0"),
    ("\\frac{3}{4}", "\\frac{3}{4}"),
    ("7", "7"),
    ("abc", "abc"),
    ("\\sqrt{2}", "sqrt(2)"),
    ("50\\%", "50"),
    ("\\text{5 cm}", "5"),
    ("-2", "- 2"),
    ("3/4", "0.75"),
)

#: What a reader with the grader's source, the right to call it, and a record of a run can reach.
FULL_ACCESS = {
    Component.GRADER: Access.SOURCE | Access.QUERY | Access.REPLICATE | Access.RECORD,
    Component.TASK: Access.SOURCE | Access.QUERY | Access.REPLICATE,
    Component.RECORD: Access.RECORD,
    Component.GOLD: Access.QUERY,
}

#: Strictly less, on every component. An auditor holding a log and nothing else.
LESS_ACCESS = {Component.GRADER: Access.RECORD, Component.RECORD: Access.RECORD}


@pytest.fixture(scope="module")
def real_verifier(tmp_path_factory: pytest.TempPathFactory) -> VerifierUnderTest:
    try:
        with urllib.request.urlopen(REAL_VERIFIER_URL, timeout=20) as response:
            source = response.read().decode("utf-8")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        pytest.skip(
            f"no network for the real-verifier card ({type(exc).__name__}). "
            f"tests/test_measure_card.py proves the composition runs; this file proves it "
            f"measures, and skipping it leaves that unproven rather than proven."
        )
    path: Path = tmp_path_factory.mktemp("d7_real") / "math_equivalence.py"
    path.write_text(source, encoding="utf-8")
    return VerifierUnderTest(source_path=path, entrypoint="is_equiv")


@pytest.fixture(scope="module")
def real_inputs(real_verifier: VerifierUnderTest) -> CardInputs:
    """Everything a reader of `is_equiv` can honestly supply. Absent slots stay absent.

    Nothing here is manufactured to make a field read. The exploit log is derived by running the
    verifier's own metamorphic relations against it, which is what a team hunting exploits would
    have written down; the replay set is three identical calls per task, which is what measuring a
    determinism claim looks like; and the replication design's facet is which way round the two
    strings are passed, which is a real condition of measurement for a checker whose arguments are
    not symmetric.
    """
    fn = real_verifier.load()

    def grade(str1: str, str2: str) -> float:
        return 1.0 if fn(str1=str1, str2=str2) else 0.0

    corpus = ListCorpus.of(
        [
            Rollout(id=f"m{i:02d}", inputs={"str1": a, "str2": b})
            for i, (a, b) in enumerate(REAL_PAIRS)
        ]
    )
    relations = answer_text_relations(on="str1")

    rng = Random(0)
    finds: list[ExploitFind] = []
    trial = 0
    for i, (answer, gold) in enumerate(REAL_PAIRS):
        seed = Rollout(id=f"m{i:02d}", inputs={"str1": answer, "str2": gold})
        base = grade(answer, gold)
        if base <= 0.0:
            continue
        for relation in relations:
            moved = relation.transformation.apply(seed, rng)
            if moved is seed:
                continue
            trial += 1
            if abs(grade(**dict(moved.inputs)) - base) > 1e-12:
                finds.append(ExploitFind(family=relation.name, effort=float(trial), id=moved.id))
    log = ExploitLog.of(
        finds,
        total_effort=float(trial),
        effort_unit="rollout-relation trials",
        source="hendrycks/math is_equiv",
    )

    scores = np.array([grade(a, b) for a, b in REAL_PAIRS], dtype=float)
    replays = ReplaySet(
        scores=np.array([[grade(a, b) for _ in range(3)] for a, b in REAL_PAIRS], dtype=float),
        task_ids=tuple(f"m{i:02d}" for i in range(len(REAL_PAIRS))),
        environment="hendrycks/math is_equiv",
    )

    values, objects, forms = [], [], []
    for i, (a, b) in enumerate(REAL_PAIRS):
        values.extend([grade(a, b), grade(b, a)])
        objects.extend([f"m{i:02d}"] * 2)
        forms.extend(["answer_first", "gold_first"])
    design = ReplicationDesign.from_long(
        np.asarray(values, dtype=float),
        objects,
        forms,
        object_label="answer pair",
        facet_labels=("argument order", "occasion"),
    )

    return CardInputs(
        verifier=real_verifier,
        grader=grade,
        corpus=corpus,
        relations=relations,
        exploit_log=log,
        replays=replays,
        design=design,
        group_scores=GroupScores.of(
            [scores[i : i + 5] for i in range(0, len(scores), 5)], grader="is_equiv"
        ),
        mutation_rung=1,
        mutation_limit=60,
        grader_name="hendrycks/math is_equiv",
    )


@pytest.fixture(scope="module")
def real_card(real_inputs: CardInputs) -> Any:
    ctx = card_context(real_inputs, access=FULL_ACCESS, phase=Phase.PRE_RUN)
    return GraderCard(real_inputs).estimate(ctx)


# ---------------------------------------------------------------------------
# Clause 1: every field renders, as a reading or as a refusal with its remedy
# ---------------------------------------------------------------------------


def test_the_card_renders_every_field_the_catalogue_names(real_card: Any) -> None:
    assert not isinstance(real_card, Refusal), render_card(real_card)
    card = real_card.value
    assert isinstance(card, CardReading)
    assert [f.name for f in card.fields] == [spec.name for spec in CARD_FIELDS]
    assert len(card.fields) == 13
    text = render_card(real_card)
    for spec in CARD_FIELDS:
        assert spec.name in text, f"{spec.name} is not on the rendered page"


def test_every_field_is_a_reading_or_a_refusal_and_never_a_blank(real_card: Any) -> None:
    """The rule the card exists for. A field is present in both directions or the test fails."""
    card = real_card.value
    for f in card.fields:
        if f.read:
            assert f.value is not None, f"{f.name} claims a reading with no value"
            assert f.headline.strip(), f"{f.name} claims a reading with no summary"
            assert f.trust is not None, f"{f.name} claims a reading with no trust level"
            assert f.evidence.startswith("ev:"), f"{f.name} carries no evidence id"
        else:
            assert f.reason in RefusalReason.__members__, f"{f.name}: {f.reason} is not a reason"
            assert f.detail.strip(), f"{f.name} refuses with no detail"
            assert f.remedy.strip(), f"{f.name} refuses with no remedy"
    assert card.read_fields, "the real card is supposed to have readings on it"
    assert card.refused_fields, "and refusals, which is the common case and not a failure"


def test_every_refusal_names_a_reason_whose_remedy_points_the_right_way(real_card: Any) -> None:
    """The E30 test, applied to the card's own refusals.

    A `RECORD_INCOMPLETE` remedy sends the reader upstream to record something; an
    `ACCESS_INSUFFICIENT` remedy is answerable where they are standing; a `SUBSTRATE_MISMATCH`
    remedy sends them to a different instrument. Asserting the shape of the sentence rather than
    its text is what keeps this a test of the distinction rather than of the wording.
    """
    upstream = ("record", "log ", "instrument the grader", "replay", "score each item", "collect")
    here = ("supply", "install", "pass ", "ask for", "expose", "build")
    for f in real_card.value.refused_fields:
        remedy = f.remedy.lower()
        if f.reason == RefusalReason.RECORD_INCOMPLETE.name:
            assert any(w in remedy for w in upstream + here), f"{f.name}: {f.remedy}"
        elif f.reason == RefusalReason.ACCESS_INSUFFICIENT.name:
            assert any(w in remedy for w in here), f"{f.name}: {f.remedy}"
        elif f.reason == RefusalReason.SUBSTRATE_MISMATCH.name:
            assert "instrument declared for" in remedy, f"{f.name}: {f.remedy}"


def test_the_trust_level_is_computed_by_the_gates(real_card: Any) -> None:
    """No caller set this and no caller could. Nothing calibrates the card and no study froze it,
    so `compute_trust` returns the bottom rung and the page says so."""
    assert real_card.trust == compute_trust(calibration=None, registered=False)
    assert real_card.trust is TrustLevel.EXPLORATORY
    assert real_card.quantity == "grader.card"
    assert real_card.observable == "GraderCard"
    text = render_card(real_card)
    assert "computed by the gates" in text
    floor = real_card.value.constituent_trust
    assert floor is not None
    assert floor == min(TrustLevel(f.trust) for f in real_card.value.read_fields)


def test_the_dual_use_fields_are_withheld_from_the_rendered_card_by_default(
    real_card: Any,
) -> None:
    """D2's surviving-mutant list is a reproducible set of ways to make `is_equiv` wrong.

    The flag is read off the payload, so the card cannot forget to set it, and the redaction is
    what the page shows. The store row carries the same flag, which is what keeps a redacted card
    and its original two rows rather than one row with two histories.
    """
    card = real_card.value
    sensitive = card.sensitive_fields
    assert sensitive, "D2 flags itself and it reads on this grader"
    assert {f.name for f in sensitive} <= {"surviving mutants", "false-positive catalogue"}
    for f in sensitive:
        assert is_sensitive(f.value), f"{f.name}: the flag is the payload's, not the card's"

    default = render_card(real_card)
    assert "reproducers withheld" in default
    redacted = card.redacted()
    assert redacted.sensitive is False
    for f in redacted.fields:
        if f.redacted_here:
            assert is_sensitive(f.value) is False
    assert real_card.subject.extra["sensitive"] == "true"


def test_the_card_lints_clean_on_the_real_subject(real_inputs: CardInputs) -> None:
    assert lint_instrument(GraderCard(real_inputs)) == []


# ---------------------------------------------------------------------------
# Clause 2: preflight with zero grader calls, and a cost
# ---------------------------------------------------------------------------


def test_the_same_card_preflights_with_zero_grader_calls_and_reports_its_cost(
    real_inputs: CardInputs,
) -> None:
    """A buyer asks what the card would contain before paying for it.

    The counter wraps the callable the card would score with, so a preflight that made even one
    call fails here. Loading the verifier's module is not a call: it executes the file's function
    definitions and scores nothing, which is why the count is taken on the scoring function.
    """
    calls = {"n": 0}
    real = real_inputs.grader

    def counted(**kw: Any) -> float:
        calls["n"] += 1
        return real(**kw)  # type: ignore[misc]

    from dataclasses import replace

    watched = replace(real_inputs, grader=counted)
    ctx = card_context(watched, access=FULL_ACCESS, phase=Phase.PRE_RUN)
    card = GraderCard(watched)

    plan = card.capability_report(ctx)
    assert calls["n"] == 0, f"the capability report made {calls['n']} grader calls"

    pre = card.preflight(ctx)
    assert calls["n"] == 0, f"preflight made {calls['n']} grader calls"
    assert pre.ok
    assert pre.cost.calls and pre.cost.calls > 0
    assert "floor" in pre.cost.note

    assert len(plan.fields) == 13
    assert plan.available, "several fields are reachable at this access"
    assert plan.refused, "and several are not, each saying what it would take"
    text = plan.render()
    for spec in CARD_FIELDS:
        assert spec.name in text
    assert "grader calls" in text


def test_the_plan_and_the_card_agree_on_which_fields_were_out_of_reach(
    real_inputs: CardInputs, real_card: Any
) -> None:
    """A field the plan calls unreachable must not appear as a reading on the card.

    The converse does not hold and the plan says so: a field that would read can still refuse once
    it runs, when the data turns out not to support the estimator. On this grader D6 does exactly
    that, because no exploit family in the log appears exactly twice and Chao1 has nothing to
    divide by.
    """
    ctx = card_context(real_inputs, access=FULL_ACCESS, phase=Phase.PRE_RUN)
    plan = GraderCard(real_inputs).capability_report(ctx)
    unreachable = {f.name for f in plan.refused}
    read = {f.name for f in real_card.value.read_fields}
    assert unreachable.isdisjoint(read), f"{sorted(unreachable & read)} read despite being refused"
    assert read <= {f.name for f in plan.available}


# ---------------------------------------------------------------------------
# Clause 3: less access, strictly more refusals, as an ordering
# ---------------------------------------------------------------------------


def test_less_access_renders_strictly_more_refusals(
    real_inputs: CardInputs, real_card: Any
) -> None:
    """The ordering, asserted as an ordering.

    A fixed count would pin this test to today's field list and to today's corpus, and would then
    have to be edited every time either changed, which is how a pinned count stops being a check.
    What is actually being claimed is monotone: taking access away can only ever turn a reading
    into a refusal, never the other way round.
    """
    poor_ctx = card_context(real_inputs, access=LESS_ACCESS, phase=Phase.PRE_RUN)
    poor = GraderCard(real_inputs).estimate(poor_ctx).value
    rich = real_card.value

    rich_refused = {f.name for f in rich.refused_fields}
    poor_refused = {f.name for f in poor.refused_fields}
    assert rich_refused < poor_refused, (
        f"strictly less access must refuse strictly more. With source and query: "
        f"{sorted(rich_refused)}. With a record only: {sorted(poor_refused)}."
    )
    assert {f.name for f in poor.read_fields} < {f.name for f in rich.read_fields}

    assert poor.access_shortfall, "a record-only reader is short of D7's stated minimum"
    assert rich.access_shortfall == {}
    assert "short of D7's stated minimum" in render_card(GraderCard(real_inputs).estimate(poor_ctx))


def test_the_poorer_card_is_still_a_card(real_inputs: CardInputs) -> None:
    """Below the card's access minimum the artifact does not disappear. That is the point."""
    poor_ctx = card_context(real_inputs, access=LESS_ACCESS, phase=Phase.PRE_RUN)
    reading = GraderCard(real_inputs).estimate(poor_ctx)
    assert not isinstance(reading, Refusal)
    card = reading.value
    assert len(card.fields) == 13
    for f in card.refused_fields:
        assert f.remedy.strip()
    assert "13" in card.says or "of 13" in card.says


def test_a_reader_with_no_access_at_all_gets_thirteen_remedies(
    real_inputs: CardInputs,
) -> None:
    """The floor of the ladder, and the artifact a marketplace listing should have carried."""
    ctx = card_context(real_inputs, access={}, phase=Phase.PRE_RUN, substrate=Substrate.PROGRAM)
    card = GraderCard(real_inputs).estimate(ctx).value
    assert len(card.refused_fields) == 13
    assert card.constituent_trust is None
    reasons = set(card.by_reason)
    assert reasons <= set(RefusalReason.__members__)
    assert RefusalReason.ACCESS_INSUFFICIENT.name in reasons

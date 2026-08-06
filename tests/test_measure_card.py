"""Unit tests for D7, the grader card.

Nothing here reaches the network. The subject is a small verifier written to a temporary file: it
proves the composition runs, that every field is present in both directions, and that the three
rules the card exists to enforce are enforced by the code rather than by the docstrings.

The clause, on a real public verifier, is in `tests/acceptance/test_w4_d7_card.py`.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reward_lens.core.evidence import ValueCodec
from reward_lens.core.extras import ExtraRequiredError
from reward_lens.core.gates import compute_trust
from reward_lens.core.invariance import InvariancePayload, check_invariance
from reward_lens.core.quantity import QUANTITIES, catalogue_path, load_quantities
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Access, Component, Phase, Substrate, TrustLevel
from reward_lens.measure.base import Instrument, Observable, lint_instrument
from reward_lens.measure.card import (
    CARD_FIELDS,
    D7_ACCESS_MIN,
    CardField,
    CardInputs,
    CardReading,
    FieldSpec,
    GraderCard,
    card_context,
    card_plan,
    grader_card,
    render_card,
)
from reward_lens.measure.metrology.flakiness import ReplaySet
from reward_lens.measure.metrology.gstudy import GroupScores, ReplicationDesign
from reward_lens.verifier import (
    ExploitFind,
    ExploitLog,
    ListCorpus,
    Rollout,
    RubricInput,
    StrictReference,
    VerifierUnderTest,
    answer_text_relations,
)

# ---------------------------------------------------------------------------
# The fixture grader: parse, normalise, threshold, and a timeout-only arm
# ---------------------------------------------------------------------------

GRADER_SOURCE = '''"""A math-answer grader. Parse the number out, normalise it, threshold the gap."""

import re

_NUMBER = re.compile(r"-?\\d+(?:\\.\\d+)?")


def parse(text):
    if text is None:
        return None
    match = _NUMBER.search(text)
    if match is None:
        return None
    return float(match.group(0))


def grade(response, reference, scale=1.0, elapsed=0.0, timeout=5.0):
    if elapsed > timeout:  # rubric: timed_out
        return 0.0
    parsed = parse(response)
    if parsed is None:  # rubric: unparseable
        return 0.0
    gap = abs(parsed / scale - float(reference) / scale)
    if gap < 1e-6:  # rubric: exact_match
        return 1.0
    if gap < 0.5:  # rubric: near_match
        return 0.5
    return 0.0
'''

PAIRS: tuple[tuple[str, str], ...] = (
    ("3", "3"),
    ("3.2", "3"),
    ("4", "3"),
    ("banana", "3"),
    ("10", "10"),
    ("10.4", "10"),
    ("-2", "-2"),
    ("0", "0"),
    ("7.25", "7"),
    ("12", "5"),
)

FULL_ACCESS = {
    Component.GRADER: Access.SOURCE | Access.QUERY | Access.REPLICATE | Access.RECORD,
    Component.TASK: Access.SOURCE | Access.QUERY | Access.REPLICATE,
    Component.RECORD: Access.RECORD,
    Component.GOLD: Access.QUERY,
}

RECORD_ONLY = {Component.GRADER: Access.RECORD, Component.RECORD: Access.RECORD}


@pytest.fixture(scope="module")
def verifier(tmp_path_factory: pytest.TempPathFactory) -> VerifierUnderTest:
    path = tmp_path_factory.mktemp("d7") / "math_grader.py"
    path.write_text(GRADER_SOURCE, encoding="utf-8")
    return VerifierUnderTest(source_path=path, entrypoint="grade")


@pytest.fixture(scope="module")
def inputs(verifier: VerifierUnderTest) -> CardInputs:
    """Everything the fixture grader can honestly supply. Absent slots stay absent."""
    fn = verifier.load()

    def grade(response: str, reference: str, **kw: Any) -> float:
        return float(fn(response=response, reference=reference, **kw))

    corpus = ListCorpus.of(
        [
            Rollout(id=f"r{i:02d}", inputs={"response": a, "reference": b})
            for i, (a, b) in enumerate(PAIRS)
        ]
    )

    # -- D4 reads a rubric, which is a scorer over named numeric inputs with declared ranges
    def rubric(values: dict[str, float]) -> float:
        return float(
            fn(
                response=f"{values['answer']:.4f}",
                reference="3",
                scale=values["scale"],
                elapsed=values["elapsed"],
                timeout=5.0,
            )
        )

    rubric_inputs = (
        RubricInput("answer", 2.0, 4.0, "the parsed answer"),
        RubricInput("scale", 0.5, 2.0, "the problem's scale"),
        RubricInput("elapsed", 0.0, 10.0, "seconds the harness took"),
    )

    # -- D5 measures against a stricter oracle: exact equality, where this grader gives half a
    # -- mark for anything within 0.5. Every disagreement is credit for a wrong answer.
    def exactly_equal(response: str, reference: str, **kw: Any) -> bool:
        try:
            return abs(float(response) - float(reference)) < 1e-9
        except (TypeError, ValueError):
            return False

    reference = StrictReference(
        name="exact numeric equality",
        decide=exactly_equal,
        basis="the two answers parse to the same float. Stricter than a 0.5 tolerance band.",
    )

    scores = np.array([grade(a, b) for a, b in PAIRS], dtype=float)
    replays = ReplaySet(
        scores=np.tile(scores.reshape(-1, 1), (1, 3)),
        task_ids=tuple(f"r{i:02d}" for i in range(len(PAIRS))),
        environment="fixture math grader",
    )

    values, objects, forms = [], [], []
    for i, (a, b) in enumerate(PAIRS):
        values.extend([grade(a, b, scale=1.0), grade(a, b, scale=2.0)])
        objects.extend([f"r{i:02d}"] * 2)
        forms.extend(["scale_1", "scale_2"])
    design = ReplicationDesign.from_long(
        np.asarray(values, dtype=float),
        objects,
        forms,
        object_label="answer pair",
        facet_labels=("problem scale", "occasion"),
    )

    log = ExploitLog.of(
        [
            ExploitFind(family="off_by_less_than_half", effort=3.0, id="r01"),
            ExploitFind(family="off_by_less_than_half", effort=9.0, id="r08"),
            ExploitFind(family="unparseable_scores_zero", effort=14.0, id="r03"),
            ExploitFind(family="unparseable_scores_zero", effort=21.0, id="r03b"),
            ExploitFind(family="timeout_arm", effort=25.0, id="t01"),
        ],
        total_effort=30.0,
        effort_unit="mutation trials",
        source="fixture math grader",
    )

    return CardInputs(
        verifier=verifier,
        grader=grade,
        corpus=corpus,
        # Named rather than defaulted. D3's default set transforms a `solution` argument and this
        # grader has none, and a relation that transformed nothing is not a relation that passed.
        relations=answer_text_relations(on="response"),
        scorer=rubric,
        rubric_inputs=rubric_inputs,
        reference=reference,
        fp_seeds=tuple(corpus),
        exploit_log=log,
        replays=replays,
        design=design,
        group_scores=GroupScores.of([scores[:5], scores[5:]], grader="fixture"),
        mutation_rung=1,
        mutation_limit=30,
        sobol_n_base=64,
        fuzz_max_examples=40,
        grader_name="fixture math grader",
    )


@pytest.fixture(scope="module")
def full_card(inputs: CardInputs) -> Any:
    ctx = card_context(inputs, access=FULL_ACCESS, phase=Phase.PRE_RUN)
    return GraderCard(inputs).estimate(ctx)


# ---------------------------------------------------------------------------
# The field list is the catalogue's, and nothing else
# ---------------------------------------------------------------------------


def _d7_record() -> dict[str, Any]:
    path = catalogue_path("CATALOGUE.json")
    assert path is not None, "spec/CATALOGUE.json is what pins this test to the specification"
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = doc["instruments"] if isinstance(doc, dict) else doc
    for row in rows:
        if row.get("id") == "D7":
            return dict(row)
    raise AssertionError("no D7 record in spec/CATALOGUE.json")


def test_the_card_has_exactly_the_thirteen_fields_the_catalogue_names() -> None:
    """The `says` line is the field list, and drift in either direction is a failure.

    A fourteenth field is a card claiming something the catalogue did not ask for; a twelfth is a
    quantity that silently stopped being reported, which is the failure mode this whole artifact
    exists to remove.
    """
    says = _d7_record()["says"].lower()
    for spec in CARD_FIELDS:
        head = spec.name.lower().split()[0]
        assert head in says, f"{spec.name!r} is not named in D7's `says`: {says}"
    assert len(CARD_FIELDS) == 13
    assert len({spec.name for spec in CARD_FIELDS}) == 13


def test_every_field_names_a_registered_quantity_and_its_instrument_agrees() -> None:
    """The spec's quantity and the instrument's declaration are two places to say one thing."""
    if "grader.card" not in QUANTITIES:
        load_quantities()
    for spec in CARD_FIELDS:
        assert spec.quantity in QUANTITIES, f"{spec.name}: {spec.quantity} is not registered"
        cls = spec.resolve()
        assert cls.quantity == spec.quantity, f"{spec.name}: {cls.quantity} != {spec.quantity}"
        assert cls.name, f"{spec.name}: the instrument declares no name"


def test_the_card_is_a_registered_quantity_and_lints_clean() -> None:
    if "grader.card" not in QUANTITIES:
        load_quantities()
    assert "grader.card" in QUANTITIES
    assert lint_instrument(GraderCard()) == []


def test_the_card_satisfies_both_protocols() -> None:
    card = GraderCard()
    assert isinstance(card, Observable)
    assert isinstance(card, Instrument)


# ---------------------------------------------------------------------------
# A card with nothing in it is still a card
# ---------------------------------------------------------------------------


def test_a_card_over_no_inputs_renders_thirteen_refusals_each_with_a_remedy() -> None:
    """The common case, and the one a blank-field card would get wrong.

    A grader nobody instrumented has none of the thirteen inputs. The card that comes back is
    thirteen refusals, and it is a useful artifact precisely because each of them says what to go
    and record.
    """
    reading = grader_card(CardInputs(grader_name="an uninstrumented grader"), access=FULL_ACCESS)
    card = reading.value
    assert len(card.fields) == 13
    assert card.read_fields == ()
    assert len(card.refused_fields) == 13
    for f in card.fields:
        assert f.remedy.strip(), f"{f.name} refuses with no remedy"
        assert f.detail.strip(), f"{f.name} refuses with no detail"
        assert RefusalReason[f.reason] in RefusalReason
    assert card.constituent_trust is None
    text = render_card(reading)
    for f in card.fields:
        assert f.name in text


def test_a_field_that_refuses_cannot_be_built_without_a_remedy() -> None:
    with pytest.raises(ValueError, match="carries no remedy"):
        CardField(name="x", quantity="grader.card", instrument="i", read=False, detail="d")


# ---------------------------------------------------------------------------
# The RECORD_INCOMPLETE / ACCESS_INSUFFICIENT split (E30)
# ---------------------------------------------------------------------------


def test_a_missing_input_at_sufficient_access_is_record_incomplete() -> None:
    """The remedy is upstream, so the reason is the one whose remedy is upstream.

    The reader holds every access D6 asks for and the exploit log was never written. Nothing they
    do to what they are holding recovers it, and `refuse_incomplete`'s remedy has the shape E30
    names: record this and re-run.
    """
    reading = grader_card(CardInputs(), access=FULL_ACCESS, substrate=Substrate.PROGRAM)
    by_name = {f.name: f for f in reading.value.fields}
    exploit = by_name["exploit-family accounting"]
    assert exploit.reason == RefusalReason.RECORD_INCOMPLETE.name
    assert "log every exploit" in exploit.remedy


def test_the_same_missing_input_at_insufficient_access_is_access_insufficient() -> None:
    """The remedy is answerable where the reader is standing, so the reason changes with it."""
    reading = grader_card(CardInputs(), access={}, substrate=Substrate.PROGRAM)
    by_name = {f.name: f for f in reading.value.fields}
    exploit = by_name["exploit-family accounting"]
    assert exploit.reason == RefusalReason.ACCESS_INSUFFICIENT.name
    assert "RECORD" in exploit.detail


def test_a_category_error_outranks_a_missing_input() -> None:
    """Collecting a comparison flow does not make B1 apply to a program, so it is not the remedy."""
    reading = grader_card(CardInputs(), access=FULL_ACCESS, substrate=Substrate.PROGRAM)
    by_name = {f.name: f for f in reading.value.fields}
    assert by_name["curl mass"].reason == RefusalReason.SUBSTRATE_MISMATCH.name
    assert by_name["Afriat index"].reason == RefusalReason.SUBSTRATE_MISMATCH.name


def test_a_missing_optional_extra_becomes_a_refusal_naming_the_install() -> None:
    """An uninstallable field is an anticipated condition with a one-line remedy, not a crash.

    Simulated by a spec whose `resolve` raises what `require_extra` raises, because the extra the
    verifier series needs is installed in this environment and uninstalling it to test the branch
    would be a worse test.
    """

    from dataclasses import fields as dc_fields

    class Unresolvable(FieldSpec):
        def resolve(self) -> Any:
            raise ExtraRequiredError(
                "the verifier series needs the optional 'verifier' extra, which is not installed"
            )

    template = CARD_FIELDS[0]
    unresolvable = Unresolvable(**{f.name: getattr(template, f.name) for f in dc_fields(FieldSpec)})
    reading = grader_card(CardInputs(), fields=[unresolvable], access=FULL_ACCESS)
    row = reading.value.fields[0]
    assert row.read is False
    assert row.reason == RefusalReason.ACCESS_INSUFFICIENT.name
    assert row.statistics["missing_extra"] == "verifier"
    assert "pip install 'reward-lens[verifier]'" in row.remedy


# ---------------------------------------------------------------------------
# Trust is computed, never supplied
# ---------------------------------------------------------------------------


def test_the_card_states_its_trust_and_offers_no_way_to_set_it() -> None:
    """There is no seam. Not on the instrument, not on the payload, not on the one-call form."""
    from dataclasses import fields as dc_fields

    assert "trust" not in {f.name for f in dc_fields(CardReading)}
    assert not hasattr(GraderCard, "trust")
    reading = grader_card(CardInputs(), access=FULL_ACCESS)
    assert reading.trust == compute_trust(calibration=None, registered=False)
    assert reading.trust is TrustLevel.EXPLORATORY


def test_the_card_reports_the_floor_under_its_constituent_readings(full_card: Any) -> None:
    card = full_card.value
    floor = card.constituent_trust
    assert floor is not None, "the fixture card is supposed to have readings on it"
    assert floor == min(TrustLevel(f.trust) for f in card.read_fields)
    assert floor.name.lower() in render_card(full_card)


# ---------------------------------------------------------------------------
# Dual use
# ---------------------------------------------------------------------------


def test_the_sensitive_flag_is_read_off_the_payload_and_not_declared_by_the_card(
    full_card: Any,
) -> None:
    from reward_lens.verifier import is_sensitive

    card = full_card.value
    for f in card.read_fields:
        assert f.sensitive == is_sensitive(f.value), f.name
    assert card.sensitive, "D2 and D5 both flag themselves; one of them should have read"


def test_a_sensitive_field_is_withheld_from_the_rendered_card_by_default(full_card: Any) -> None:
    card = full_card.value
    sensitive = card.sensitive_fields
    assert sensitive, "the fixture card is supposed to carry at least one sensitive field"
    default = render_card(full_card)
    assert "reproducers withheld" in default
    for f in sensitive:
        assert f.redacted().sensitive is False
    assert card.redacted().sensitive is False


def test_the_store_row_carries_the_flag_so_a_redacted_card_is_a_different_row(
    full_card: Any,
) -> None:
    """The subject is hashed into the evidence id, so the flag is on the row and not on a note."""
    assert full_card.subject.extra["sensitive"] == "true"
    assert full_card.subject.extra["grader"] == "fixture math grader"


def test_asking_for_the_exploit_content_still_goes_through_the_payloads_own_gate(
    full_card: Any,
) -> None:
    """`FPCatalogue.for_publication` raises without a signed decision, and the card does not
    catch it. An explicit request says the reader wants the content; a `DisclosureDecision` says
    somebody accountable released it, and the card requires both."""
    from reward_lens.verifier import DisclosureDecision, DisclosureRequired

    card = full_card.value
    catalogue = {f.name: f for f in card.read_fields}.get("false-positive catalogue")
    assert catalogue is not None, "the fixture is supposed to produce a false-positive catalogue"
    with pytest.raises(DisclosureRequired):
        render_card(full_card, include_sensitive=True)

    decided = replace(
        catalogue,
        value=catalogue.value.with_disclosure(
            DisclosureDecision(decided_by="the test", reason="asserting the gate opens")
        ),
    )
    released = replace(
        card, fields=tuple(decided if f.name == catalogue.name else f for f in card.fields)
    )
    text = released.render(include_sensitive=True)
    assert "reproducers withheld" not in text
    assert len(text) > len(card.render())


# ---------------------------------------------------------------------------
# The capability report
# ---------------------------------------------------------------------------


def test_the_capability_report_calls_no_grader_and_prices_what_it_can(
    inputs: CardInputs,
) -> None:
    calls = {"n": 0}
    real = inputs.grader
    scorer = inputs.scorer

    def counted_grader(*a: Any, **kw: Any) -> float:
        calls["n"] += 1
        return real(*a, **kw)  # type: ignore[misc]

    def counted_scorer(values: Any) -> float:
        calls["n"] += 1
        return scorer(values)  # type: ignore[misc]

    watched = replace(inputs, grader=counted_grader, scorer=counted_scorer)
    plan = card_plan(watched, access=FULL_ACCESS, phase=Phase.PRE_RUN)
    assert calls["n"] == 0, "a capability report that calls the grader is not a capability report"
    assert len(plan.fields) == 13
    assert plan.available, "the fixture is supposed to reach several fields"
    assert plan.cost.calls and plan.cost.calls > 0
    assert "floor" in plan.cost.note
    text = plan.render()
    for f in plan.fields:
        assert f.name in text


def test_the_cost_report_distinguishes_unpriced_from_free(inputs: CardInputs) -> None:
    """`PreflightResult.cost` defaults to `FREE`, so an instrument that never costed itself is
    indistinguishable from a free one unless the card keeps the two apart. D2 re-grades the whole
    corpus once per mutant and prices itself at nothing."""
    plan = card_plan(inputs, access=FULL_ACCESS, phase=Phase.PRE_RUN)
    priced = {f.name for f in plan.available if f.priced}
    unpriced = {f.name for f in plan.available if not f.priced}
    assert "metamorphic violations" in priced
    assert "surviving mutants" in unpriced
    assert "cost not modelled by this instrument" in plan.render()


# ---------------------------------------------------------------------------
# The access ordering
# ---------------------------------------------------------------------------


def test_less_access_refuses_a_superset_of_what_more_access_refuses(inputs: CardInputs) -> None:
    """The ordering, not a count. A fixed count would break every time a field changes."""
    rich = grader_card(inputs, access=FULL_ACCESS, phase=Phase.PRE_RUN).value
    poor = grader_card(inputs, access=RECORD_ONLY, phase=Phase.PRE_RUN).value
    rich_refused = {f.name for f in rich.refused_fields}
    poor_refused = {f.name for f in poor.refused_fields}
    assert rich_refused < poor_refused, (
        f"strictly less access should refuse strictly more: {sorted(rich_refused)} then "
        f"{sorted(poor_refused)}"
    )
    assert poor.access_shortfall, "RECORD-only access is short of D7's stated minimum"
    assert rich.access_shortfall == {}


def test_the_stated_access_minimum_is_reported_and_not_gated_on() -> None:
    """A reader below the card's access minimum still gets a card. That is the whole instrument."""
    assert GraderCard.requires == {}
    assert D7_ACCESS_MIN == {Component.GRADER: Access.QUERY | Access.REPLICATE}
    reading = grader_card(CardInputs(), access={})
    assert not isinstance(reading, Refusal)
    assert len(reading.value.fields) == 13


# ---------------------------------------------------------------------------
# The payload survives a store
# ---------------------------------------------------------------------------


def test_the_card_payload_round_trips_through_the_value_codec(full_card: Any) -> None:
    """A card that cannot be written and read back is a report, not evidence."""
    codec = ValueCodec()
    decoded = codec.decode(codec.encode(full_card.value, None), None)
    assert isinstance(decoded, CardReading)
    assert [f.name for f in decoded.fields] == [f.name for f in full_card.value.fields]
    assert [f.read for f in decoded.fields] == [f.read for f in full_card.value.fields]
    assert [f.trust for f in decoded.fields] == [f.trust for f in full_card.value.fields]
    assert [f.reason for f in decoded.fields] == [f.reason for f in full_card.value.fields]
    assert decoded.by_reason == full_card.value.by_reason


# ---------------------------------------------------------------------------
# The generated invariance property test
# ---------------------------------------------------------------------------


def test_the_declared_invariance_group_is_the_one_that_resolves() -> None:
    """`grader.card` prints `none` in the registry and `trivial` is what that resolves to.

    The generated test is a formality here and says so: the trivial group has no generators, so
    the report comes back uninformative rather than passing on the strength of a check that never
    ran. Declaring `none` is an answer and a blank is not, which is the rule this pins.
    """
    payload = InvariancePayload(
        scores=np.array([1.0, 2.0, 3.0, 4.0]), group_ids=np.array([0, 0, 1, 1])
    )
    report = check_invariance(
        GraderCard(),
        GraderCard.invariance,
        payload,
        n=8,
        run=lambda inst, pl: float(len(inst.fields)),
    )
    assert report.passed
    assert report.informative is False
    assert "no generators" in report.skipped


# ---------------------------------------------------------------------------
# The two fields a program refuses read on the substrate they are declared for
# ---------------------------------------------------------------------------


def test_curl_mass_and_the_afriat_index_read_on_a_procedural_grader() -> None:
    """B1 and B2 declare NEURAL_GEN and PROCEDURAL, so a program refuses them by category.

    That refusal is only honest if the fields are reachable somewhere, and this is where. A
    tournament over four items with every pair compared is the smallest subject either instrument
    can say anything about.
    """
    from reward_lens.measure.base import Context
    from reward_lens.measure.composition.hodge import PairCount, edge_flow
    from reward_lens.measure.composition.revealed import bank_from_scores

    pairs = [
        PairCount(0, 1, 7.0, 3.0),
        PairCount(0, 2, 6.0, 4.0),
        PairCount(0, 3, 8.0, 2.0),
        PairCount(1, 2, 5.0, 5.0),
        PairCount(1, 3, 6.0, 4.0),
        PairCount(2, 3, 4.0, 6.0),
    ]
    inputs = CardInputs(
        flow=edge_flow(pairs, 4),
        comparisons=bank_from_scores(np.arange(24, dtype=float).reshape(6, 4)),
        curl_nulls=("E",),
        curl_draws=8,
        afriat_baseline_draws=4,
        grader_name="a four-item tournament",
    )
    ctx = Context(
        signal=None,
        readout="score",
        access={Component.GRADER: Access.QUERY | Access.RECORD},
        substrate=Substrate.PROCEDURAL,
        phase=Phase.PRE_RUN,
    )
    card = GraderCard(inputs).estimate(ctx).value
    by_name = {f.name: f for f in card.fields}
    assert by_name["curl mass"].read, by_name["curl mass"].detail
    assert by_name["Afriat index"].read, by_name["Afriat index"].detail


def test_a_mapping_payload_with_no_sentence_is_printed_as_its_own_entries() -> None:
    """B1 and B2 emit a plain dict, and `dict` is not a reading.

    A row that prints the payload's type name looks filled and says nothing, which is the failure
    mode a blank field has in a different costume.
    """
    from reward_lens.measure.base import Context
    from reward_lens.measure.composition.hodge import PairCount, edge_flow

    inputs = CardInputs(
        flow=edge_flow([PairCount(0, 1, 3.0, 1.0), PairCount(1, 2, 2.0, 2.0)], 3),
        curl_nulls=("E",),
        curl_draws=8,
    )
    ctx = Context(
        signal=None,
        readout="score",
        access={Component.GRADER: Access.QUERY},
        substrate=Substrate.PROCEDURAL,
        phase=Phase.PRE_RUN,
    )
    card = GraderCard(inputs).estimate(ctx).value
    row = {f.name: f for f in card.fields}["curl mass"]
    assert row.read
    assert row.headline != "dict"
    assert "curl_mass=" in row.headline
    assert any("structured entries not summarised" in line for line in row.continues)

"""Acceptance: the kernel everything else is checked against.

Clauses discharged here:

- *the ladder for `grader.effective_group_size` resolves to three different rungs under three
  access matrices, asserted.*
- *an instrument declaring a condition absent from `measured_by` fails lint.*
- *a refusal carries both numbers*, and the parts of the reading contract that do not need
  `budget.py` or `reference.py`, which are not built yet.

The kernel is small and subtle and everything is checked against it, which is why it is built and
tested before any fan-out. A subtle divergence here is the one bug that costs a rebuild rather
than a patch.
"""

from __future__ import annotations

import pytest

from reward_lens.core.envelope import (
    UNCONDITIONAL,
    ConditionReading,
    EnvelopeLintError,
    EnvelopeSpec,
    RegimeCondition,
    RegimeReading,
)
from reward_lens.core.quantity import (
    ESTIMATORS,
    FREE,
    QUANTITIES,
    BiasStatement,
    CostModel,
    EstimatorEntry,
    Quantity,
    Unit,
    best_estimator,
    ladder,
    load_quantities,
    open_quantities,
    register_estimator,
    what_would_it_take,
)
from reward_lens.core.reading import (
    REASON_MEANING,
    Refusal,
    RefusalReason,
    bounded_refusal,
    is_refusal,
)
from reward_lens.core.types import (
    PROFILE_AUDITOR,
    PROFILE_MARKETPLACE_BUYER,
    Access,
    Component,
    Phase,
    Substrate,
    missing_access,
    satisfies,
)

_ESS = "grader.effective_group_size"


@pytest.fixture(scope="module", autouse=True)
def _registry():
    """Load the quantities, then take this module's synthetic estimators back out.

    The estimator registry is process-global and this module registers stand-in rungs for
    `grader.effective_group_size` to exercise the ladder machinery. When this file was written
    those were the only rungs for that quantity, so leaving them behind cost nothing. The real A1
    ladder landed later, and from then on every later test in the same process saw eight rungs where
    there are four: `ladder()` returned `[0, 0, 1, 1, 2, 2, 3, 3]` and the metrology test that
    checks A1's registered bias directions failed under the full suite while passing alone.

    Snapshotting the keys and removing what appeared is deliberate rather than listing the impls:
    a list would go stale the next time a test in this module registers one more.
    """
    if _ESS not in QUANTITIES:
        load_quantities()
    before = set(ESTIMATORS._items)
    yield
    for impl in set(ESTIMATORS._items) - before:
        ESTIMATORS._items.pop(impl, None)


# ---------------------------------------------------------------------------
# The four typing dimensions
# ---------------------------------------------------------------------------


def test_replicate_does_not_follow_from_query():
    """The distinction that decides whether half of series A exists for a given user.

    A hosted judge with a fixed internal seed is callable and not facet-varyable, and without
    facet variation there is no variance decomposition, no effective group size and no
    attenuation factor.
    """
    assert not (Access.QUERY & Access.REPLICATE)
    query_only = {Component.GRADER: Access.QUERY}
    assert not satisfies(query_only, {Component.GRADER: Access.REPLICATE})


def test_an_absent_component_is_no_access_not_an_error():
    assert missing_access(PROFILE_AUDITOR, {Component.POLICY: Access.BACKWARD}) == {
        Component.POLICY: Access.BACKWARD
    }


def test_phase_is_not_derivable_from_access():
    """Two people with different access are both POST_RUN; two with the same access differ."""
    assert Phase.POST_RUN is not Phase.IN_RUN
    assert len(list(Phase)) == 4
    assert len(list(Substrate)) == 6


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def _ess_ladder():
    """A1's four rungs, transcribed from the worked example."""
    rungs = [
        (
            0,
            "kish_ess_on_observed_spread",
            {Component.GRADER: Access.RECORD},
            BiasStatement("upward", "ignores correlated grader error entirely"),
            FREE,
        ),
        (
            1,
            "test_retest_same_input",
            {Component.GRADER: Access.REPLICATE},
            BiasStatement("upward", "captures occasion but not rater or rubric"),
            CostModel(calls=110, note="R calls per item"),
        ),
        (
            2,
            "crossed_g_study",
            {Component.GRADER: Access.REPLICATE},
            BiasStatement("approximately_unbiased", "unbiased if the design is fully crossed"),
            CostModel(calls=420, dollars=1.10),
        ),
        (
            3,
            "crossed_g_study_with_person_by_rater",
            {Component.GRADER: Access.REPLICATE, Component.RECORD: Access.RECORD},
            BiasStatement(
                "approximately_unbiased",
                "adds the person-by-rater interaction, which governs co-directional gradients",
            ),
            CostModel(calls=840, dollars=2.20),
        ),
    ]
    env = EnvelopeSpec(
        requires=frozenset({RegimeCondition.GROUP_NONDEGENERATE}),
        measured_by={RegimeCondition.GROUP_NONDEGENERATE: "estimator.degenerate_fraction"},
    )
    for rung, impl, requires, bias, cost in rungs:
        if impl in ESTIMATORS:
            continue
        register_estimator(
            EstimatorEntry(
                quantity=_ESS,
                impl=impl,
                requires=requires,
                envelope=env,
                rung=rung,
                bias=bias,
                cost=cost,
            )
        )


def test_the_ladder_resolves_to_three_different_rungs_under_three_access_matrices():
    """The ladder clause."""
    _ess_ladder()
    assert [e.rung for e in ladder(_ESS)] == [0, 1, 2, 3]

    auditor = best_estimator(_ESS, PROFILE_AUDITOR)
    buyer = best_estimator(_ESS, PROFILE_MARKETPLACE_BUYER)
    lab = best_estimator(
        _ESS,
        {
            Component.GRADER: Access.REPLICATE | Access.RECORD | Access.QUERY,
            Component.RECORD: Access.RECORD,
        },
    )

    assert auditor is not None and buyer is not None and lab is not None
    assert auditor.rung == 0
    assert buyer.rung == 2
    assert lab.rung == 3
    assert len({auditor.rung, buyer.rung, lab.rung}) == 3

    # And the cheap rung says which way it is wrong, in words a reader can act on.
    assert auditor.bias.direction == "upward"
    assert "correlated grader error" in auditor.bias.why


def test_no_access_at_all_yields_no_estimator_and_a_remedy():
    _ess_ladder()
    assert best_estimator(_ESS, {}) is None
    plan = what_would_it_take(_ESS, {})
    assert plan["reachable"] is False
    assert plan["needs"] == {"GRADER": "RECORD"}
    assert plan["rung"] == 0


def test_the_registry_refuses_to_redefine_a_name():
    with pytest.raises(ValueError, match="already registered with a different definition"):
        QUANTITIES.register(
            _ESS,
            Quantity(
                id=_ESS,
                definition="something else entirely",
                unit=Unit("count"),
                invariance="units",
            ),
        )


def test_an_estimator_for_an_unregistered_quantity_fails_at_import():
    """An Instrument whose quantity is not registered fails at import."""
    with pytest.raises(ValueError, match="not a registered quantity"):
        register_estimator(
            EstimatorEntry(
                quantity="grader.effective_group_sizee",  # a typo, which is the realistic case
                impl="typo_estimator",
                requires={},
                envelope=UNCONDITIONAL,
                rung=0,
                bias=BiasStatement("unknown", "n/a"),
            )
        )


def test_a_quantity_with_no_estimator_is_an_open_research_target_not_a_bug():
    _ess_ladder()
    open_ids = open_quantities()
    assert _ESS not in open_ids
    assert "credit.successor_representation" in open_ids


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def test_per_token_and_per_sequence_are_not_the_same_unit():
    """The most common silent error in this literature, made a type error."""
    per_token = Unit("nats", per="token", scale="nats")
    per_sequence = Unit("nats", per="sequence", scale="nats")
    assert not per_token.compatible_with(per_sequence)
    assert per_token.compatible_with(Unit("nats", per="token", scale="nats"))
    assert not Unit("nats", per="token", scale="nats").compatible_with(
        Unit("nats", per="token", scale="bits")
    )


def test_the_registry_carries_the_printed_unit_token_so_the_decomposition_is_checkable():
    q = QUANTITIES.get("update.kl_spent")
    assert q.unit.as_printed == "nats/seq"
    assert (q.unit.dimension, q.unit.per) == ("nats", "sequence")


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


def test_an_envelope_naming_an_unmeasurable_condition_fails_lint():
    """The envelope clause. A declared precondition nobody can check enforces nothing."""
    with pytest.raises(EnvelopeLintError, match="declares no way to measure"):
        EnvelopeSpec(requires=frozenset({RegimeCondition.QUASI_STATIC}), measured_by={})


def test_an_envelope_naming_a_quantity_nobody_registered_is_rejected():
    """The rejection half of E37, which the wave-3 tests never asserted.

    E37 checked the accepting half: every id the kernel's own `MEASURED_BY` names resolves. That is
    a test of the mapping, not of the guard, and it would still have passed with the guard deleted.
    E45 is the standing lesson about a guard in this area, so the rejecting case gets its own
    assertion: an id that resolves to no row must not construct, because appearing in `measured_by`
    was the whole check that a precondition is measurable.

    Three spellings resolve to nothing and all three are rejected. The unregistered id is the one
    E37 names. The empty string and `None` are the same defect written the easier way: a key whose
    value never got filled in still appears in `measured_by`, so before this it satisfied the first
    rule and was skipped by the second.
    """
    with pytest.raises(EnvelopeLintError, match="carries no such row"):
        EnvelopeSpec(
            requires=frozenset({RegimeCondition.QUASI_STATIC}),
            measured_by={RegimeCondition.QUASI_STATIC: "grader.no_such_quantity"},
        )
    for blank in ("", None):
        with pytest.raises(EnvelopeLintError, match="no quantity id"):
            EnvelopeSpec(
                requires=frozenset({RegimeCondition.QUASI_STATIC}),
                measured_by={RegimeCondition.QUASI_STATIC: blank},  # type: ignore[dict-item]
            )
    # An id on a condition the envelope does not require is still checked, because a mapping that
    # names a quantity nobody registered is wrong whether or not anything reads it today.
    with pytest.raises(EnvelopeLintError, match="carries no such row"):
        EnvelopeSpec(
            requires=frozenset({RegimeCondition.QUASI_STATIC}),
            measured_by={
                RegimeCondition.QUASI_STATIC: "run.adiabaticity",
                RegimeCondition.LIGHT_TAILED: "grader.no_such_quantity",
            },
        )
    # And the registered case still constructs, so the guard is not simply refusing everything.
    EnvelopeSpec(
        requires=frozenset({RegimeCondition.QUASI_STATIC}),
        measured_by={RegimeCondition.QUASI_STATIC: "run.adiabaticity"},
    )


def test_every_catalogue_envelope_measurer_resolves_or_is_openly_unmeasured():
    """The same guard, asked of the catalogue rather than of a constructor.

    `envelope_measured_by` is `OPEN` on most records and a mapping on four. A mapping entry is
    either a registered quantity id or the literal `OPEN`, which is the record saying nobody has
    said how the condition is measured. Anything else is an id that would fail at construction the
    moment an instrument lifted the record into an `EnvelopeSpec`, and finding that out in the
    package that lifts it is how E37 was found the first time.
    """
    import json
    import pathlib

    from reward_lens.core.quantity import QUANTITIES

    root = pathlib.Path(__file__).resolve().parents[2]
    doc = json.loads((root / "src/reward_lens/spec/CATALOGUE.json").read_text(encoding="utf-8"))
    bad: list[tuple[str, str, str]] = []
    for row in doc["instruments"]:
        mapping = row.get("envelope_measured_by")
        if not isinstance(mapping, dict):
            continue
        for condition, qid in mapping.items():
            if qid == "OPEN" or qid in QUANTITIES:
                continue
            bad.append((row["id"], condition, str(qid)))
    assert bad == [], (
        f"{bad} name neither a registered quantity nor OPEN. N4 is the known exception and it "
        f"carries prose rather than an id, which is its own finding."
    )


def test_an_empty_envelope_needs_an_explicit_justification():
    with pytest.raises(EnvelopeLintError, match="must say so explicitly"):
        EnvelopeSpec()
    with pytest.raises(EnvelopeLintError, match="needs a justification"):
        EnvelopeSpec(unconditional=True)
    assert UNCONDITIONAL.justification


def test_bound_on_violation_must_name_the_bounding_estimator():
    """A promise with nothing behind it, made impossible by the type."""
    with pytest.raises(EnvelopeLintError, match="has to name one"):
        EnvelopeSpec(
            requires=frozenset({RegimeCondition.LINEAR_RESPONSE}),
            measured_by={RegimeCondition.LINEAR_RESPONSE: "selection.explained_fraction"},
            on_violation="bound",
        )


def test_an_instrument_declaring_quasi_static_is_refused_on_a_run_with_ad_of_three():
    """The other half of the envelope clause: the refusal carries the statistic and the threshold."""
    env = EnvelopeSpec(
        requires=frozenset({RegimeCondition.QUASI_STATIC}),
        measured_by={RegimeCondition.QUASI_STATIC: "run.adiabaticity"},
    )
    reading = RegimeReading(
        conditions={
            RegimeCondition.QUASI_STATIC: ConditionReading(
                condition=RegimeCondition.QUASI_STATIC,
                holds=False,
                statistic=3.0,
                threshold=1.0,
                detail="driven faster than it relaxes",
            )
        }
    )
    assert env.admits(reading) is False
    (violation,) = env.violations(reading)
    assert violation.statistic == 3.0
    assert violation.threshold == 1.0
    assert "3" in violation.render() and "FAIL" in violation.render()


def test_unknown_is_not_a_pass():
    """The failure this whole module addresses: a check that did not happen reading as one that did."""
    env = EnvelopeSpec(
        requires=frozenset({RegimeCondition.NEAR_POLICY}),
        measured_by={RegimeCondition.NEAR_POLICY: "run.regime"},
    )
    assert env.admits(RegimeReading.of(NEAR_POLICY=None)) is False
    assert env.admits(RegimeReading.of(NEAR_POLICY=True)) is True
    assert env.admits(None) is False
    assert "unknown" in RegimeReading.of(NEAR_POLICY=None).render()


def test_there_are_twelve_regime_conditions():
    assert len(list(RegimeCondition)) == 13


# ---------------------------------------------------------------------------
# Refusal is a value
# ---------------------------------------------------------------------------


def test_a_refusal_without_a_remedy_cannot_be_constructed():
    """A refusal without one is a tool that looks broken instead of one that looks careful."""
    with pytest.raises(ValueError, match="carries no remedy"):
        Refusal(
            instrument="A1",
            reason=RefusalReason.ENVELOPE_VIOLATED,
            detail="GROUP_NONDEGENERATE fails",
            remedy="   ",
        )


def test_a_refusal_can_still_carry_an_honest_bound(tmp_path):
    """ "I cannot give you the effective group size, but I can bound it above by 6.1"."""
    from reward_lens.core import ModelFP, Provenance, SubjectRef, make_evidence

    bound = make_evidence(
        observable="A1",
        observable_version="1",
        subject=SubjectRef(signals=(ModelFP("mfp:x"),)),
        value=6.1,
        provenance=Provenance(),
    )
    r = bounded_refusal(
        "A1",
        RefusalReason.ABOVE_LOD_BELOW_LOQ,
        detail="LOD 0.02, LOQ 0.07, measured 0.04",
        remedy="Raise the replication count to 11 per item to clear the LOQ.",
        bound=bound,
        lod=0.02,
        loq=0.07,
    )
    assert is_refusal(r)
    assert r.is_bounded
    assert r.partial.value == 6.1
    assert r.statistics == {"lod": 0.02, "loq": 0.07}
    # Both numbers appear in what the user reads.
    assert "0.02" in r.render() and "0.07" in r.render()
    assert "6.1" in r.render()


def test_every_refusal_reason_has_a_meaning_a_user_can_read():
    """The refusal reference is the page users actually open."""
    assert set(REASON_MEANING) == set(RefusalReason)
    for reason, text in REASON_MEANING.items():
        assert len(text) > 40, reason


def test_there_are_sixteen_refusal_reasons():
    """Fifteen documented reasons, plus RECORD_INCOMPLETE.

    E12 records the fifteen. E30 is the sixteenth, a ratified amendment rather than a drift: ACCESS_INSUFFICIENT is about what you can touch and
    RECORD_INCOMPLETE is about what is there, and their remedies point in opposite directions.
    """
    assert len(list(RefusalReason)) == 17
    assert RefusalReason.RECORD_INCOMPLETE in REASON_MEANING
    # E48, the seventeenth.
    # The three reasons are separated by one question and it is worth asserting rather than
    # describing: where is the remedy answerable? Access, where the reader stands. Record, upstream.
    # This one, nowhere, because the question does not apply to the object.
    assert RefusalReason.QUANTITY_UNDEFINED in REASON_MEANING


# ---------------------------------------------------------------------------
# The catalogue the registry loads from
# ---------------------------------------------------------------------------


def test_the_catalogue_loads_every_quantity_and_the_counts_are_what_was_recounted():
    """124 catalogue rows, plus four quantities named by instruments and never registered.

    The four are `credit.successor_representation` (E10), `verifier.fp_catalogue`,
    `verifier.trusted_inputs` and `selection.dimensionality` (E14). Each is named in an
    instrument's own catalogue entry and has no registry row. Registered here so the docs build
    names them as open research targets rather than losing them.

    Plus the 28 the retrofit added (E19): the shipped 2.0.1 corpus survives as instruments and
    gives none of them a row, so 28 of the 29 retrofitted observables had no quantity to declare. E19 marked all 28 out of the wedge on the claim that the corpus needs
    `GRADER: FORWARD`; thirteen of them declare `GRADER: QUERY` or `RECORD: RECORD` in the
    installed source and are in the wedge on the catalogue's own legend. Those thirteen were flipped,
    which is why the wedge count is 98 rather than 85. The fifteen that genuinely need FORWARD or
    FORWARD|MUTATE stay out.

    Plus the five of series N (E23): the Level 0 frontier is developed in full and the catalogue
    gives it no rows. All five are in the wedge, because the whole layer needs a callable grader
    and a gold channel on the same n samples and nothing else.

    Plus five more, from six ids that were computed and then declined rather than invented. Three
    register what B2 and B3 already compute (`grader.houtman_maks_index`,
    `grader.money_pump_index`, `grader.null_reach`) and two split `grader.score_distribution`, whose
    spread is covariant under `reward.affine` and whose flip rate is invariant, which one
    `invariance_group` field cannot say. The sixth, `grader.copeland_slater_disagreement`, is
    refused rather than deferred and has no row: it reads 0.1813 against an achievable range over
    tie-breaks of [0.0200, 0.4820].

    The trivial-group count falls by one because `grader.graph_lambda2` moved off `none`. Its
    eigenvalues are invariant under relabelling the comparison graph, which is a similarity
    transform of the Laplacian, so it is provably invariant under a group that really acts on it and
    `none` was discarding a free test of a property that holds.

    And four more when `forecast/` landed: `forecast.brier_score` and the two Murphy terms it
    decomposes into, plus `forecast.decision_value`. The last is the one worth registering rather
    than leaving on a payload, because it is the number that answers "so what",
    it is covariant under a rescaling of the loss where the other three are dimensionless, and on the
    campaign's own re-scored ledger it is exactly 0.000 while the Brier score looks informative.

    And one more when A1 stopped multiplying Kish's shape factor into
    `grader.effective_group_size`. The two factors answer different questions and their product is
    not a quantity: the shape factor asks how unequal the deviations within a group are, which is a
    fact about the policy and the reward, and the reliability asks how much of the observed variance
    is true score, which is a fact about the grader. So the shape factor gets its own id,
    `run.group_shape_factor`, and it is `run.` rather than `grader.` deliberately, because filing it
    under `grader.` would re-assert the exact conflation the correction removed. It is in the wedge:
    it needs a scored group and nothing else. Measured across eleven real reward models it runs
    0.7346 to 0.7562, which is where the per-grader variation in the old reading was coming from.

    And four more which were not an addition but a repair. The contract layer, N5 to N8,
    shipped and its four quantity ids were never registered, so `reward.optimal_weights`,
    `reward.equal_compensation_ratio`, `reward.sorting_cutoff` and
    `reward.component_congruity` had no rows and no `reward.*` prefix existed in the registry at all.
    Four shipped instruments were failing `lint_instrument` rule 1 while their package read `done`.
    Found by enumerating the registry for an unrelated reason, which is the third time an
    enumeration-as-a-test has caught a hole nobody was looking for.
    """
    report = load_quantities()
    assert report.skipped_open == []
    assert len(QUANTITIES) == 190
    assert sum(1 for q in QUANTITIES.values() if q.wedge) == 112
    assert len(report.trivial_group) == 41


def test_every_registered_quantity_has_a_unit_and_an_invariance_group():
    """The enumeration that stops a large build developing holes nobody notices."""
    for q in QUANTITIES.values():
        assert q.unit.dimension, q.id
        assert q.invariance, q.id


def test_the_packaged_json_catalogue_agrees_with_the_yaml_source():
    """One source of truth, two encodings.

    The YAML carries the comments saying where each row came from; the JSON is what the wheel
    ships, because PyYAML has a compiled extension and the core installs without one. A test is
    the only thing keeping them the same file.
    """
    import json
    import pathlib

    yaml = pytest.importorskip("yaml", reason="the YAML source is only present in a checkout")
    root = pathlib.Path(__file__).resolve().parents[2]
    for stem in ("QUANTITIES", "CATALOGUE"):
        src = root / "spec" / f"{stem}.yaml"
        if not src.exists():
            pytest.skip("not a source checkout")
        from_yaml = yaml.safe_load(src.read_text(encoding="utf-8"))
        for packaged in (
            root / "spec" / f"{stem}.json",
            root / "src" / "reward_lens" / "spec" / f"{stem}.json",
        ):
            assert packaged.exists(), f"{packaged} is missing; regenerate it from the YAML"
            assert json.loads(packaged.read_text(encoding="utf-8")) == from_yaml, (
                f"{packaged} has drifted from {src}"
            )

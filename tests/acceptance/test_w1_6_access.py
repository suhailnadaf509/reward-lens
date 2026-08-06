"""Acceptance: four-dimensional access resolution and the capability report.

The clause: *the capability report renders for a synthetic record with no policy and lists at least
one refusal with a remedy.* Four more clauses follow, and each of them is a way the module could
have been built wrong:

- `REPLICATE` is resolved from a probe rather than assumed. A fake endpoint that ignores its seed
  resolves to `QUERY` only; one that honours it resolves to `QUERY | REPLICATE`.
- A `COMPOSITE` grader is classified by walking its tree.
- Every refusal in the rendered report carries a non-empty remedy that is not merely the reason
  name.
- `unchecked` checks are rendered as unchecked rather than as passes.

The scenario is the worked one: 401 steps, 25,664 rollouts, a three-leaf composite
grader, no policy checkpoint, and a grader whose rubric weights changed at step 240. The estimators
are registered here rather than imported, because none of the catalogue's 89 instruments has a
registered estimator yet and the point of the report is to work correctly both before and after
that changes. Their costs are deliberately *not* the numbers the worked example prints, so that
`test_no_number_in_the_report_came_from_the_specification` can tell a rendered number from a
transcribed one.
"""

from __future__ import annotations

import pytest

from reward_lens.access import (
    SEED_PROBE_CALLS,
    ProbeBudget,
    capability_report,
    classify_substrate,
    resolve_access,
    resolve_phase,
)
from reward_lens.core.budget import LimitOfDetection, SubstrateKey
from reward_lens.core.envelope import (
    ConditionReading,
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
    load_quantities,
    register_estimator,
)
from reward_lens.core.reading import REASON_MEANING, RefusalReason
from reward_lens.core.reference import MatrixDescription, ReferenceMaterial
from reward_lens.core.types import Access, Component, Phase, Substrate
from reward_lens.measure.base import BaseObservable, PreflightResult

# ---------------------------------------------------------------------------
# The synthetic run
# ---------------------------------------------------------------------------


class SyntheticRecord:
    """A finished GRPO run. No policy checkpoint, which is the clause's condition."""

    n_steps = 401
    n_rollouts = 25_664
    sampling = 1.0
    complete = True
    scores = True
    task_refs = True
    logprobs = True


class Node:
    """A stand-in for the `ScoreTree`, carrying only the structural protocol's fields."""

    def __init__(self, name, *, children=(), combine=None, substrate=None, enabled=True):
        self.name = name
        self.children = list(children)
        self.combine = combine
        self.substrate = substrate
        self.enabled = enabled


def composite_grader() -> Node:
    """The frontier default: a verifier, a judge and a rubric under one weighted sum."""
    return Node(
        "grader",
        combine="weighted_sum",
        children=[
            Node("unit_tests", substrate=Substrate.PROGRAM),
            Node("style_judge", substrate=Substrate.NEURAL_GEN),
            Node("rubric_panel", substrate=Substrate.PROCEDURAL),
        ],
    )


def measured_regime() -> RegimeReading:
    """The five conditions, with one failure and one that could not be determined."""
    return RegimeReading(
        conditions={
            RegimeCondition.GROUP_NONDEGENERATE: ConditionReading(
                RegimeCondition.GROUP_NONDEGENERATE,
                True,
                0.04,
                0.20,
                detail="degenerate group fraction",
            ),
            RegimeCondition.NEAR_POLICY: ConditionReading(
                RegimeCondition.NEAR_POLICY, True, 2.0, 8.0, detail="max staleness, steps"
            ),
            RegimeCondition.STATIONARY_GRADER: ConditionReading(
                RegimeCondition.STATIONARY_GRADER,
                False,
                1.0,
                0.0,
                detail="rubric weights changed at step 240",
            ),
            RegimeCondition.NO_COMPACTION: ConditionReading(
                RegimeCondition.NO_COMPACTION, True, 0.0, 0.0, detail="compaction events"
            ),
            RegimeCondition.QUASI_STATIC: ConditionReading(
                RegimeCondition.QUASI_STATIC,
                None,
                float("nan"),
                float("nan"),
                detail="requires two anneal rates; see G3",
            ),
        }
    )


# ---------------------------------------------------------------------------
# Estimators, registered for this module and removed afterwards
# ---------------------------------------------------------------------------

_ESS = "grader.effective_group_size"
_BETA = "selection.gradient_beta"
_H2 = "selection.heritability_h2"
_RECOVERY = "instrument.recovery_auc"

_GROUP_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.GROUP_NONDEGENERATE}),
    measured_by={RegimeCondition.GROUP_NONDEGENERATE: "estimator.degenerate_fraction"},
)
_STATIONARY_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER, RegimeCondition.LINEAR_RESPONSE}),
    measured_by={
        RegimeCondition.STATIONARY_GRADER: "grader.composition",
        RegimeCondition.LINEAR_RESPONSE: "selection.explained_fraction",
    },
)
_UNCONDITIONAL_CENSUS = EnvelopeSpec(
    unconditional=True,
    justification="a census over the record: it counts what is there and asserts nothing about "
    "the process that produced it.",
)

_ENTRIES = (
    EstimatorEntry(
        quantity=_ESS,
        impl="w16.kish_ess",
        requires={Component.GRADER: Access.RECORD},
        envelope=_GROUP_ENVELOPE,
        rung=0,
        bias=BiasStatement("upward", "ignores correlated grader error entirely"),
        cost=FREE,
    ),
    EstimatorEntry(
        quantity=_ESS,
        impl="w16.crossed_g_study",
        requires={Component.GRADER: Access.REPLICATE},
        envelope=_GROUP_ENVELOPE,
        rung=2,
        bias=BiasStatement("approximately_unbiased", "unbiased if the design is fully crossed"),
        cost=CostModel(calls=384, dollars=0.96),
    ),
    EstimatorEntry(
        quantity=_BETA,
        impl="w16.gradient_beta",
        requires={Component.RECORD: Access.RECORD},
        envelope=_STATIONARY_ENVELOPE,
        rung=0,
        bias=BiasStatement("unknown", "first-order in the step size"),
        cost=FREE,
    ),
    EstimatorEntry(
        quantity=_H2,
        impl="w16.heritability",
        requires={Component.POLICY: Access.BACKWARD},
        envelope=_UNCONDITIONAL_CENSUS,
        rung=1,
        bias=BiasStatement("unknown", "assumes the breeder's equation applies to this parameter"),
        cost=CostModel(gpu_seconds=900.0),
    ),
    EstimatorEntry(
        quantity=_RECOVERY,
        impl="w16.recovery_auc",
        requires={Component.RECORD: Access.RECORD},
        envelope=_UNCONDITIONAL_CENSUS,
        rung=0,
        bias=BiasStatement("unknown", "inherits the reference's own uncertainty"),
        cost=FREE,
        calibrated_against="org:planted-family",
    ),
)


class CoverageCensus(BaseObservable):
    """A real `Instrument` that reports an exact reading, so the est. u column has both forms."""

    name = "w16.decision_coverage"
    version = "1.0"
    quantity = "verifier.decision_coverage"
    access = {Component.RECORD: Access.RECORD}
    substrates = frozenset({Substrate.PROGRAM, Substrate.COMPOSITE})
    phases = frozenset({Phase.POST_RUN})
    envelope = _UNCONDITIONAL_CENSUS
    invariance = "trivial"
    baselines = ("line count",)
    rung = 0

    def preflight(self, ctx) -> PreflightResult:
        pre = super().preflight(ctx)
        if not pre.ok:
            return pre
        return PreflightResult(
            instrument=pre.instrument,
            ok=True,
            rung=self.rung,
            cost=FREE,
            expected_uncertainty=0.0,
            regime=pre.regime,
            unchecked=pre.unchecked,
        )


@pytest.fixture(scope="module", autouse=True)
def _registry():
    """Register this module's estimators, then take them back out.

    The estimator registry is process-global. Leaving four synthetic rows in it would change what
    every later test in the same process sees from `ladder()`, so the fixture removes exactly what
    it added.
    """
    if _ESS not in QUANTITIES:
        load_quantities()
    added = []
    for entry in _ENTRIES:
        if entry.impl not in ESTIMATORS:
            register_estimator(entry)
            added.append(entry.impl)
    yield
    for impl in added:
        ESTIMATORS._items.pop(impl, None)


@pytest.fixture(scope="module")
def report(_registry):
    """The capability report for the synthetic run, resolved end to end."""
    record = SyntheticRecord()
    access = resolve_access(record=record, grader=composite_grader(), policy=None)
    return capability_report(
        access,
        classify_substrate(composite_grader()),
        resolve_phase(record=record),
        measured_regime(),
        reference=ReferenceMaterial(
            id="org:planted-family",
            kind="planted_organism",
            assigned_value=0.75,
            u_characterisation=0.02,
            matrix=MatrixDescription(system="0.6B organism", scale="LoRA plant"),
            u_homogeneity=None,
        ),
        instruments=(CoverageCensus(),),
    )


# ---------------------------------------------------------------------------
# The clause
# ---------------------------------------------------------------------------


def test_the_report_renders_for_a_synthetic_record_with_no_policy(report):
    """The clause, first half."""
    assert report.access.of(Component.POLICY) is Access.RECORD
    assert not (report.access.of(Component.POLICY) & Access.BACKWARD)
    text = report.render()
    assert text.splitlines()[0] == "ACCESS RESOLVED"
    for heading in ("ACCESS RESOLVED", "REGIME MEASURED", "AVAILABLE NOW", "REFUSED, WITH REMEDY"):
        assert heading in text
    positions = [text.index(h) for h in ("ACCESS RESOLVED", "REGIME MEASURED", "AVAILABLE NOW")]
    assert positions == sorted(positions), "the sections print in the documented order"


def test_the_report_lists_at_least_one_refusal_with_a_remedy(report):
    """The clause, second half."""
    assert report.refused, "a run with no policy and a non-stationary grader refuses something"
    for refusal in report.refused:
        assert refusal.remedy.strip()
    text = report.render()
    assert "Remedy:" in text


def test_the_missing_policy_is_refused_by_name_with_the_flag_that_would_supply_it(report):
    refusal = _refusal_for(report, "selection.heritability_h2")
    assert refusal.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "POLICY: BACKWARD" in refusal.detail
    assert "--policy" in refusal.remedy


def test_the_grader_that_changed_at_step_240_is_refused_with_the_window_remedy(report):
    refusal = _refusal_for(report, "selection.gradient_beta")
    assert refusal.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "rubric weights changed at step 240" in refusal.detail
    assert "restrict the window" in refusal.remedy
    assert "weight schedule" in refusal.remedy


def test_a_reference_with_no_homogeneity_refuses_rather_than_calibrating(report):
    refusal = _refusal_for(report, "instrument.recovery_auc")
    assert refusal.reason is RefusalReason.REFERENCE_UNCERTIFIED
    assert "u_homogeneity" in refusal.detail
    assert "three seeds" in refusal.remedy


# ---------------------------------------------------------------------------
# REPLICATE is probed, not assumed
# ---------------------------------------------------------------------------


def test_an_endpoint_that_ignores_its_seed_resolves_to_query_only():
    calls: list[int | None] = []

    def fixed_internal_seed(item, *, seed=None):
        calls.append(seed)
        return 0.5

    resolution = resolve_access(
        record=SyntheticRecord(),
        grader=fixed_internal_seed,
        probe=ProbeBudget(calls=SEED_PROBE_CALLS),
    )
    access = resolution.of(Component.GRADER)
    assert access & Access.QUERY
    assert not (access & Access.REPLICATE)
    assert len(calls) == SEED_PROBE_CALLS
    assert "look identical from outside" in resolution.note_for(Component.GRADER)


def test_an_endpoint_that_honours_its_seed_resolves_to_query_and_replicate():
    def settable_seed(item, *, seed=None):
        return 0.5 + 0.01 * (seed or 0)

    resolution = resolve_access(
        record=SyntheticRecord(),
        grader=settable_seed,
        probe=ProbeBudget(calls=SEED_PROBE_CALLS),
    )
    access = resolution.of(Component.GRADER)
    assert access & Access.QUERY
    assert access & Access.REPLICATE
    assert resolution.note_for(Component.GRADER).endswith("seed is settable")


def test_replicate_is_never_assumed_from_a_supplied_endpoint():
    """The default resolution calls nothing, so it cannot and does not claim REPLICATE."""
    calls: list[int | None] = []

    def settable_seed(item, *, seed=None):
        calls.append(seed)
        return 0.5 + 0.01 * (seed or 0)

    resolution = resolve_access(record=SyntheticRecord(), grader=settable_seed)
    assert calls == []
    assert not (resolution.of(Component.GRADER) & Access.REPLICATE)
    assert (
        "REPLICATE"
        in dict((c.component.name, c.unresolved) for c in resolution.components)["GRADER"]
    )


def test_the_rung_that_needs_replicate_is_not_offered_without_it(report):
    """Rung 2 of the ESS ladder needs REPLICATE, so this record resolves to rung 0."""
    available = {a.quantity: a for a in report.available}
    assert _ESS in available
    assert available[_ESS].preflight.rung == 0
    assert available[_ESS].preflight.instrument == "w16.kish_ess"


# ---------------------------------------------------------------------------
# COMPOSITE by walking
# ---------------------------------------------------------------------------


def test_a_composite_grader_is_classified_by_walking_its_tree(report):
    reading = report.substrate
    assert reading.substrate is Substrate.COMPOSITE
    assert len(reading.live_leaves) == 3
    assert [leaf.substrate for leaf in reading.live_leaves] == [
        Substrate.PROGRAM,
        Substrate.NEURAL_GEN,
        Substrate.PROCEDURAL,
    ]
    assert reading.render() == "COMPOSITE (3 leaves: 1 PROGRAM, 1 NEURAL_GEN, 1 PROCEDURAL)"
    assert reading.combine == ("weighted_sum",)
    assert reading.render() in report.render()


def test_an_instrument_for_a_leaf_substrate_is_pointed_at_the_leaf(report):
    """A7 declares PROGRAM and the tree has a PROGRAM leaf, so the remedy is a redirection."""
    refusal = _refusal_for(report, "env.flakiness")
    assert refusal.reason is RefusalReason.SUBSTRATE_MISMATCH
    assert "unit_tests" in refusal.detail
    assert "point it at a leaf" in refusal.remedy
    assert "will not help" not in refusal.remedy


def test_the_substrate_is_not_taken_from_the_root_node():
    """A root that declares nothing still classifies, because the leaves are what carry it."""
    reading = classify_substrate(composite_grader())
    assert reading.substrate is Substrate.COMPOSITE
    assert "walked 3 live leaves" in reading.note


# ---------------------------------------------------------------------------
# Every refusal carries an instruction
# ---------------------------------------------------------------------------

#: A remedy is an instruction, so it contains one. This list is deliberately generous; the
#: assertion it backs is that a remedy tells the reader to do something, not that it is well
#: written.
_IMPERATIVES = (
    "supply",
    "pass",
    "restrict",
    "raise",
    "measure",
    "ask",
    "re-run",
    "shorten",
    "trim",
    "hold",
    "slow",
    "give",
    "declare",
    "use",
    "characterise",
    "plant",
    "shrink",
    "build",
)


def test_every_refusal_carries_a_remedy_that_is_not_the_reason_name(report):
    assert report.refused
    for refusal in report.refused:
        remedy = refusal.remedy.strip()
        assert remedy, f"{refusal.instrument} refused with an empty remedy"
        assert remedy != refusal.reason.name
        assert refusal.reason.name.lower() not in remedy.lower().replace("_", " ")
        assert remedy != REASON_MEANING[refusal.reason], (
            f"{refusal.instrument} restates the reason's generic meaning instead of saying what "
            f"to do about this run"
        )
        assert len(remedy) > 40
        assert any(verb in remedy.lower() for verb in _IMPERATIVES), remedy


def test_the_rendered_report_carries_a_remedy_line_for_every_refusal(report):
    text = report.render()
    assert text.count("Remedy:") == len(report.refused)


# ---------------------------------------------------------------------------
# Unchecked is unchecked, never a pass
# ---------------------------------------------------------------------------


def test_a_check_that_could_not_run_is_rendered_as_unchecked(report):
    text = report.render()
    assert "NOT CHECKED" in text
    assert "limit of detection" in text
    unchecked = " ".join(report.unchecked)
    assert "limit of detection" in unchecked


def test_an_unmeasured_envelope_condition_is_named_on_the_row_it_affects():
    """The ESS envelope needs GROUP_NONDEGENERATE; with no regime at all, that is unchecked."""
    record = SyntheticRecord()
    report = capability_report(
        resolve_access(record=record),
        Substrate.COMPOSITE,
        resolve_phase(record=record),
        None,
    )
    row = next(a for a in report.available if a.quantity == _ESS)
    assert row.preflight.unchecked
    assert any("envelope" in item for item in row.preflight.unchecked)
    assert "not checked: envelope (regime not measured)" in report.render()
    assert "regime: not measured" in " ".join(report.unchecked)


def test_an_unresolved_substrate_is_unchecked_rather_than_matching():
    record = SyntheticRecord()
    report = capability_report(
        resolve_access(record=record), None, resolve_phase(record=record), measured_regime()
    )
    assert report.substrate.substrate is None
    assert any(item.startswith("substrate:") for item in report.unchecked)
    assert "UNRESOLVED" in report.render()


def test_a_supplied_limit_of_detection_removes_that_line_from_unchecked():
    record = SyntheticRecord()
    lod = LimitOfDetection(
        sigma_blank=0.01, sensitivity=1.0, key=SubstrateKey(model="synthetic", engine="test")
    )
    report = capability_report(
        resolve_access(record=record),
        Substrate.COMPOSITE,
        resolve_phase(record=record),
        measured_regime(),
        lod=lod,
    )
    assert not any(item.startswith("limit of detection") for item in report.unchecked)


# ---------------------------------------------------------------------------
# Honesty about what is not built, and about where numbers come from
# ---------------------------------------------------------------------------


def test_a_specified_but_unbuilt_instrument_is_reported_and_never_counted_as_available(report):
    assert report.catalogue_size == 95
    assert report.not_built
    available_ids = {a.quantity for a in report.available}
    for item in report.not_built:
        assert item.quantity not in available_ids
    text = report.render()
    assert "SPECIFIED, NOT YET BUILT" in text
    assert "not yet built." in text


def test_the_counts_in_the_footer_reconcile(report):
    text = report.render()
    tail = text.strip().splitlines()[-1].strip()
    assert tail.startswith(f"{report.catalogue_size} catalogued instruments")
    assert f"{len(report.available)} available" in tail
    assert f"{len(report.refused)} refused" in tail
    assert f"{len(report.not_built)} not yet built" in tail


def test_no_number_in_the_report_came_from_the_specification(report):
    """The worked example's block is illustrative. Every number here is rendered from a registered value."""
    text = report.render()
    for illustrative in ("420 calls", "$1.10", "±0.4", "±0.06", "18 min CPU", "20 replays"):
        assert illustrative not in text
    assert "384 calls, $0.96" not in text, "rung 2 is out of reach, so its price is not offered"
    assert "exact" in text, "the census instrument reports an exact reading"


def test_the_costed_row_prints_the_registered_cost_model(report):
    row = next(a for a in report.available if a.quantity == _ESS)
    assert row.preflight.cost.render() == "free"


def test_the_section_4_2_signature_takes_a_bare_matrix(report):
    """`capability_report(access, substrate, phase, regime=None)`, with a plain `AccessMatrix`."""
    plain = capability_report(
        {Component.GRADER: Access.RECORD, Component.RECORD: Access.RECORD},
        Substrate.COMPOSITE,
        Phase.POST_RUN,
        measured_regime(),
    )
    assert plain.access.note_for(Component.GRADER) == "supplied by the caller"
    # A superset rather than an exact set. The two registered in this module are what this test is
    # about, and the registry is process-global, so any instrument that registers a real estimator
    # at import and is reachable in the same session legitimately appears here too.
    # `run.regime` was the first, and pinning an exact set makes every later package's arrival a
    # failure in a test that is not about them.
    assert {_ESS, _RECOVERY} <= {a.quantity for a in plain.available}
    assert plain.render().startswith("ACCESS RESOLVED")

    recovery = next(a for a in plain.available if a.quantity == _RECOVERY)
    assert any("reference" in item for item in recovery.preflight.unchecked), (
        "an estimator that names a calibration reference, with none supplied, has not been "
        "calibrated and the row has to say so"
    )


def test_the_machine_form_carries_the_same_facts(report):
    doc = report.to_dict()
    assert doc["substrate"]["value"] == "COMPOSITE"
    assert len(doc["substrate"]["leaves"]) == 3
    assert doc["phase"]["value"] == "POST_RUN"
    assert doc["access"]["POLICY"]["access"] == "RECORD"
    assert doc["catalogue_size"] == 95
    assert {r["quantity"] for r in doc["refused"]} == {
        a.quantity for a in report.assessments if a.refusal
    }
    assert all(r["remedy"] for r in doc["refused"])


def test_the_report_makes_no_grader_calls(report):
    assert report.access.calls_made == 0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _refusal_for(report, quantity):
    for assessment in report.assessments:
        if assessment.quantity == quantity and assessment.refusal is not None:
            return assessment.refusal
    raise AssertionError(f"{quantity} was not refused; the report has {len(report.refused)}")

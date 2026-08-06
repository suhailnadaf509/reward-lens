"""Unit tests for the meta-instruments: the arithmetic, every refusal, and the generated groups.

The acceptance file measures these six on real readings. This one checks the parts that a real
dataset cannot: that the Hill fit recovers parameters it was given, that the Birge ratio is one when
it should be one, that an instrument which is an exact copy of another adds exactly nothing, and
that each of the fifteen refusal paths in this package fires on the condition it names rather than
on some other one.

Nothing here is synthetic for want of data. It is synthetic because a planted answer is the only way
to test an estimator's arithmetic, and every planted value below is stated at the point it is
planted so a reader can see what the test would have caught.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reward_lens.core.budget import (
    LIMITS,
    BudgetTerm,
    CalibrationCurve,
    SubstrateKey,
    UncertaintyBudget,
)
from reward_lens.core.envelope import RegimeReading
from reward_lens.core.evidence import ValueCodec
from reward_lens.core.invariance import (
    INVARIANT,
    InvariancePayload,
    check_invariance,
    check_unit_refusal,
)
from reward_lens.core.quantity import ESTIMATORS, QUANTITIES, ladder, load_quantities
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.reference import MatrixDescription, ReferenceMaterial, ladder_disagreement
from reward_lens.core.types import Access, Component, Phase, Substrate
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.meta import (
    CATALOGUE_IDS,
    DECIDED,
    DEFINITIONS,
    META,
    BlankReplicates,
    ControlPanel,
    Detector,
    DoseSweep,
    IncrementalValidityReading,
    InstrumentEffectReading,
    InterlaboratoryComparison,
    Lab,
    RungDisagreement,
    RungReading,
    StepBasis,
    SubstrateNoiseFloor,
    UncertaintyBudgetReading,
    as_yaml_rows,
    bootstrap_control,
    compare_rungs,
    compose,
    definition_of,
    fit_hill,
    instances,
    lint_budget,
    mean_margin,
    per_step,
    phi,
    register_ladders,
    standardised_margin,
    two_arm_blanks,
)


@pytest.fixture(scope="module", autouse=True)
def _registry():
    """Load the catalogue, and take back out whatever this module registers into `ESTIMATORS`.

    `ESTIMATORS` is a process-global that nothing cleans, and this module calls
    `register_ladders()`, which adds twelve rows for nine quantities. Left in, they would change
    what `ladder()` and `open_quantities()` return for every test module that runs after this one
    in the same process, and the failure would surface somewhere else and read as that module's
    bug. E40 records the last two times that happened.

    Snapshotting the keys and removing what appeared is deliberate rather than listing the twelve
    impls: an enumerated list goes stale the next time a test here registers one more.
    """
    if "substrate.noise_floor" not in QUANTITIES:
        load_quantities()
    before = set(ESTIMATORS._items)
    yield
    for impl in set(ESTIMATORS._items) - before:
        ESTIMATORS._items.pop(impl, None)


def _blanks(sigma: float = 0.1, n: int = 50, seed: int = 0) -> BlankReplicates:
    rng = np.random.default_rng(seed)
    return BlankReplicates(
        values=rng.normal(0.0, sigma, n), method="planted normal blank", scale=1.0
    )


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------


def test_the_package_is_six_instruments_and_each_names_its_catalogue_record():
    assert len(META) == 6
    assert {i.name for i in instances()} == set(CATALOGUE_IDS)
    assert set(CATALOGUE_IDS.values()) == {"M1", "M2", "M7", "M8", "M9", "M11"}


def test_lint_instrument_is_empty_for_every_meta_instrument():
    for inst in instances():
        assert lint_instrument(inst) == [], inst.name


def test_m7_with_no_subject_is_visibly_undeclared_rather_than_silently_defaulted():
    """The one instrument whose quantity is its subject's, so an M7 with no subject has none."""
    findings = lint_instrument(UncertaintyBudgetReading())
    assert [f.field for f in findings] == ["quantity"]
    assert lint_instrument(UncertaintyBudgetReading(quantity_id="study.tau2")) == []


def test_every_quantity_this_package_declares_is_registered():
    for inst in instances():
        assert inst.quantity in QUANTITIES, inst.name


def test_the_definitions_written_out_here_fill_registry_rows_that_are_open():
    for qid, quantity in DEFINITIONS.items():
        assert qid in QUANTITIES, qid
        assert QUANTITIES.get(qid).definition == "", (
            f"{qid} now carries a definition in the registry; drop this package's copy"
        )
        assert quantity.definition and quantity.interpretation
        assert definition_of(qid) == quantity.definition
    rows = as_yaml_rows()
    assert "substrate.noise_floor" in rows and "study.tau2" in rows
    assert "definition:" in rows and "instrument:" in rows


def test_the_two_open_catalogue_fields_are_decided_and_the_decision_is_recorded():
    assert set(DECIDED) == {"M7.quantities", "M8.quantities"}
    assert InterlaboratoryComparison.quantity == "study.tau2"
    assert QUANTITIES.get("study.tau2").definition, (
        "the row this decision leans on has a definition"
    )


def test_register_ladders_is_idempotent_and_registers_rungs_that_have_no_implementation():
    first = register_ladders()
    assert register_ladders() == first
    entries = ladder("substrate.noise_floor")
    # Rung 0 is M1's within-record blank; both rung-1 entries need a second engine, and one of
    # them is M8's, because an interlaboratory comparison of two configurations of one model *is*
    # the numerics noise floor measured directly.
    assert [e.rung for e in entries] == [0, 1, 1]
    assert [e.impl for e in entries] == [
        "m1.within_record_blank",
        "m1.two_engine_residual",
        "m8.between_engine_s_l",
    ]
    assert all(e.run is None for e in entries)
    assert entries[1].requires == {Component.GRADER: Access.REPLICATE}
    assert "second engine" in entries[1].bias.why


# ---------------------------------------------------------------------------
# M1
# ---------------------------------------------------------------------------


def test_the_hill_fit_recovers_the_curve_it_was_given():
    """Planted E_max 0.4, EC50 0.5, n 2.0, read back to within a percent from a clean sweep."""
    planted = CalibrationCurve(e_max=0.4, ec50=0.5, hill_n=2.0)
    doses = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0])
    sweep = DoseSweep(doses, np.array([planted.response(d) for d in doses]))
    fit = fit_hill(sweep)
    assert fit.converged
    assert fit.curve.e_max == pytest.approx(0.4, rel=0.01)
    assert fit.curve.ec50 == pytest.approx(0.5, rel=0.01)
    assert fit.curve.hill_n == pytest.approx(2.0, rel=0.01)
    assert fit.residual_rms < 1e-6
    assert fit.monotone


def test_a_sweep_that_does_not_respond_gives_no_curve_and_says_so():
    doses = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    flat = DoseSweep(doses, np.zeros_like(doses) - 0.01 * doses)
    fit = fit_hill(flat)
    assert not fit.converged
    assert "does not increase with dose" in fit.reason or "did not converge" in fit.reason
    assert not fit.monotone


def test_too_few_dose_points_is_reported_rather_than_fitted():
    fit = fit_hill(DoseSweep(np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.2, 0.3])))
    assert not fit.converged
    assert "at least 4" in fit.reason or "at least" in fit.reason


def test_the_lod_is_three_point_three_sigma_over_the_slope():
    planted = CalibrationCurve(e_max=0.4, ec50=0.5, hill_n=2.0, at_dose=0.5)
    doses = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0])
    sweep = DoseSweep(doses, np.array([planted.response(d) for d in doses]))
    reading = SubstrateNoiseFloor(_blanks(sigma=0.02, n=64), sweep).compute()
    assert not isinstance(reading, Refusal), reading
    assert reading.lod == pytest.approx(3.3 * reading.sigma_blank / reading.sensitivity)
    assert reading.loq == pytest.approx(10.0 * reading.sigma_blank / reading.sensitivity)
    assert reading.sigma_blank == pytest.approx(0.02, rel=0.25)


def test_a_floor_is_cached_per_configuration_when_a_key_is_given():
    key = SubstrateKey(model="unit-test", engine="numpy", dtype="float64", layer=3)
    assert key not in LIMITS
    SubstrateNoiseFloor(_blanks(), substrate_key=key, cache=True).compute()
    cached = LIMITS.get(key)
    assert cached is not None and cached.sigma_blank > 0
    assert not cached.is_determinate, "no sweep was supplied, so there is no slope to divide by"


def test_two_arm_blanks_are_disjoint_so_the_floor_is_not_zero():
    outcomes = (np.arange(400) % 3 == 0).astype(np.float64)
    blanks = two_arm_blanks(outcomes, n_replicates=100, seed=0)
    assert blanks.n == 100
    assert blanks.sigma > 0.0, "a paired blank would be identically zero and report a floor of zero"
    assert abs(blanks.mean) < 4.0 * blanks.sigma / math.sqrt(blanks.n)
    with pytest.raises(ValueError, match="four items"):
        two_arm_blanks([1.0, 0.0])


def test_a_blank_must_say_how_it_was_made():
    with pytest.raises(ValueError, match="how it was produced"):
        BlankReplicates(values=np.zeros(10), method="  ")


def test_m1_refuses_with_no_blanks_and_with_too_few():
    empty = SubstrateNoiseFloor().compute()
    assert isinstance(empty, Refusal) and empty.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "two_arm_blanks" in empty.remedy
    thin = SubstrateNoiseFloor(_blanks(n=2)).compute()
    assert isinstance(thin, Refusal) and thin.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert thin.statistics["blank_n"] == 2


def test_m1_refuses_outside_its_envelope_when_the_grader_is_drifting():
    ctx = Context(regime_reading=RegimeReading.of(STATIONARY_GRADER=False))
    out = SubstrateNoiseFloor(_blanks()).estimate(ctx)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "STATIONARY_GRADER" in out.detail


def test_m1_is_invariant_under_a_change_of_representation_basis():
    """The registry gives this quantity `repr.basis`, and a relative L2 residual really is invariant.

    The payload carries two engines' activations stacked. The group rotates both together, which is
    what a change of basis does, and the relative L2 between them is unchanged. A reading that moved
    here would be reporting a coordinate artefact as a noise floor.
    """
    rng = np.random.default_rng(5)
    a = rng.normal(size=(40, 16))
    b = a + rng.normal(scale=0.05, size=(40, 16))

    def run(inst, payload):
        h = np.asarray(payload.activations, dtype=np.float64)
        first, second = h[:40], h[40:]
        residual = np.linalg.norm(first - second, axis=1) / np.linalg.norm(first, axis=1)
        return float(np.std(residual, ddof=1))

    report = check_invariance(
        SubstrateNoiseFloor(),
        "repr.basis",
        InvariancePayload(activations=np.vstack([a, b])),
        n=16,
        relation=INVARIANT,
        run=run,
    )
    assert report.passed, report.render()
    assert report.n == 16


# ---------------------------------------------------------------------------
# M2
# ---------------------------------------------------------------------------


class _Snapshot:
    """The three fields M2 reads off a `tap.InstrumentEffect`, with planted values."""

    def __init__(self, calls: int, added_ns: int, inner_ns: int) -> None:
        self.tap_name = "planted"
        self.calls = calls
        self.added_ns_total = added_ns
        self.inner_ns_total = inner_ns
        self.added_ns_p50 = 1000.0
        self.added_ns_p99 = 4000.0
        self.fraction_p99 = 0.02
        self.resident_bytes = 4096
        self.added_alloc_bytes_this_step = None
        self.window_n = 512
        self.enabled = True
        self.recorder_exceptions = 0
        self.unchecked = ("added allocation",)


def test_the_per_step_number_is_a_difference_of_two_snapshots():
    """Planted: 10 steps between the snapshots, 2,000,000 ns added, so 0.2 ms per step."""
    basis = StepBasis(steps=10, calls=640, wall_seconds=1.0)
    window = per_step(
        _Snapshot(100, 1_000_000, 10_000_000), _Snapshot(740, 3_000_000, 40_000_000), basis
    )
    assert window.calls == 640
    assert window.added_ns == 2_000_000
    assert window.added_ms_per_step == pytest.approx(0.2)
    with pytest.raises(ValueError, match="fewer calls"):
        per_step(_Snapshot(740, 3_000_000, 0), _Snapshot(100, 1_000_000, 0), basis)


def test_m2_reports_the_window_rather_than_the_cumulative_total():
    """The two differ whenever the tap was on before the window, which is the whole point."""
    before, after = _Snapshot(1000, 9_000_000, 90_000_000), _Snapshot(1640, 9_500_000, 96_000_000)
    basis = StepBasis(steps=10, calls=640, wall_seconds=2.0)
    windowed = InstrumentEffectReading(before=before, after=after, basis=basis).compute()
    cumulative = InstrumentEffectReading(after, basis).compute()
    assert windowed.added_ms_per_step == pytest.approx(0.05)
    assert cumulative.added_ms_per_step == pytest.approx(0.95)
    assert windowed.calls == 640 and cumulative.calls == 1640
    assert windowed.fraction_of_step_time == pytest.approx(0.05 / 200.0)


def test_m2_refuses_to_convert_a_per_call_number_into_a_per_step_one():
    out = InstrumentEffectReading(_Snapshot(100, 1, 1)).compute()
    assert isinstance(out, Refusal) and out.reason is RefusalReason.UNIT_MISMATCH
    assert "per step" in out.detail and "per call" not in out.remedy.split(".")[0]


def test_m2_refuses_a_tap_with_no_calls_and_one_that_is_absent():
    absent = InstrumentEffectReading().compute()
    assert isinstance(absent, Refusal) and absent.reason is RefusalReason.ACCESS_INSUFFICIENT
    silent = InstrumentEffectReading(_Snapshot(0, 0, 0), StepBasis(steps=1, calls=0)).compute()
    assert isinstance(silent, Refusal) and silent.reason is RefusalReason.RECORD_INCOMPLETE


def test_the_step_basis_refuses_impossible_arithmetic():
    with pytest.raises(ValueError, match="at least one step"):
        StepBasis(steps=0, calls=10)
    with pytest.raises(ValueError, match="cannot be negative"):
        StepBasis(steps=1, calls=-1)
    with pytest.raises(ValueError, match="must be positive"):
        StepBasis(steps=1, calls=1, wall_seconds=0.0)


def test_the_overhead_term_divides_its_half_width_by_root_three():
    reading = InstrumentEffectReading(
        _Snapshot(640, 1_000_000, 50_000_000), StepBasis(steps=10, calls=640)
    ).compute()
    term = reading.as_term(sensitivity=0.5)
    assert term.value == pytest.approx(0.02 / math.sqrt(3.0))
    assert term.contribution == pytest.approx(term.value * 0.5)


# ---------------------------------------------------------------------------
# M7
# ---------------------------------------------------------------------------


def test_the_budget_composes_in_quadrature_and_names_the_largest_term():
    """Planted 3, 4: the combined uncertainty is 5 and the dominant term is the 4."""
    budget = UncertaintyBudget(
        terms=(BudgetTerm(name="small", value=3.0), BudgetTerm(name="large", value=4.0))
    )
    reading = UncertaintyBudgetReading(budget, quantity_id="study.tau2").compute()
    assert reading.combined == pytest.approx(5.0)
    assert reading.expanded == pytest.approx(10.0)
    assert reading.dominant_term == "large"
    assert reading.dominant_share == pytest.approx(16.0 / 25.0)


def test_compose_assembles_the_pieces_the_rest_of_the_package_produces():
    reference = ReferenceMaterial(
        id="ref",
        kind="planted_organism",
        assigned_value=1.0,
        u_characterisation=0.1,
        matrix=MatrixDescription(system="unit-test"),
        u_homogeneity=0.2,
        u_stability=0.05,
    )
    budget = compose(
        sampling=0.01,
        sampling_dof=99,
        transfers=[ladder_disagreement(3.0, 1.9, n=100)],
        overhead=BudgetTerm(name="instrument_overhead", value=0.001, kind="B"),
        reference=reference,
    )
    names = [t.name for t in budget.terms]
    assert names == ["sampling", "t21", "instrument_overhead", "u_char", "u_bb", "u_stab"]
    assert budget.dominant.name == "t21"


def test_compose_disambiguates_two_terms_that_would_share_a_name():
    """Two ladders both contributing a `t21` is a real case, and a duplicate name cannot compose."""
    budget = compose(
        transfers=[ladder_disagreement(3.0, 1.9), ladder_disagreement(2.0, 1.0)],
    )
    assert [t.name for t in budget.terms] == ["t21", "t21#2"]
    assert budget.combined == pytest.approx(math.hypot(1.1, 1.0))


def test_the_budget_lint_names_what_is_missing_without_refusing():
    one_term = UncertaintyBudget(terms=(BudgetTerm(name="sampling", value=0.1),))
    fields = {f.field for f in lint_budget(one_term, sampling_terms=("sampling",))}
    assert {"terms", "kind", "dominant", "dof"} <= fields
    complete = UncertaintyBudget(
        terms=(
            BudgetTerm(name="sampling", value=0.1, dof=99),
            BudgetTerm(name="floor", value=0.4, kind="B", dof=30),
        ),
        correlations={("sampling", "floor"): 0.2},
    )
    assert lint_budget(complete, sampling_terms=("sampling",)) == []


def test_an_empty_budget_is_a_finding_rather_than_a_combined_uncertainty_of_zero():
    findings = lint_budget(UncertaintyBudget())
    assert len(findings) == 1 and findings[0].field == "terms"
    assert "zero by construction" in findings[0].problem


def test_m7_refuses_without_a_subject_without_terms_and_with_an_unregistered_quantity():
    budget = UncertaintyBudget(terms=(BudgetTerm(name="a", value=1.0),))
    orphan = UncertaintyBudgetReading(budget).compute()
    assert isinstance(orphan, Refusal) and orphan.reason is RefusalReason.RECORD_INCOMPLETE
    empty = UncertaintyBudgetReading(None, quantity_id="study.tau2").compute()
    assert isinstance(empty, Refusal) and empty.reason is RefusalReason.RECORD_INCOMPLETE
    unknown = UncertaintyBudgetReading(budget, quantity_id="nope.not.a.quantity").compute()
    assert isinstance(unknown, Refusal) and unknown.reason is RefusalReason.UNIT_MISMATCH


def test_m7_takes_its_quantity_off_the_evidence_it_is_handed():
    class _Reading:
        quantity = "grader.effective_group_size"
        value = 1.9

    instrument = UncertaintyBudgetReading(
        UncertaintyBudget(terms=(BudgetTerm(name="a", value=0.5),)), subject=_Reading()
    )
    assert instrument.quantity == "grader.effective_group_size"
    reading = instrument.compute()
    assert reading.quantity == "grader.effective_group_size"
    assert reading.relative == pytest.approx(0.5 / 1.9)


# ---------------------------------------------------------------------------
# M8
# ---------------------------------------------------------------------------


def _labs(values, u=0.01, n=1000):
    return [Lab(id=f"lab{i}", value=float(v), u=u, n=n) for i, v in enumerate(values)]


def test_the_birge_ratio_is_one_when_the_labs_understand_their_own_errors():
    """Planted: five labs drawn at sigma 0.01 and stating u = 0.01, so Q/df is about 1."""
    rng = np.random.default_rng(4)
    values = 0.5 + rng.normal(0.0, 0.01, 40)
    labs = _labs(values)
    control = ControlPanel(labs=tuple(_labs(0.5 + rng.normal(0.0, 0.01, 40))), how="a second draw")
    reading = InterlaboratoryComparison(labs, control).compute()
    assert not isinstance(reading, Refusal), reading
    assert reading.birge == pytest.approx(1.0, abs=0.25)
    assert reading.s_l < 0.01
    assert reading.labs_understand_their_errors


def test_the_birge_ratio_is_large_when_they_do_not():
    """Planted: the same stated uncertainty with ten times the real spread."""
    rng = np.random.default_rng(4)
    labs = _labs(0.5 + rng.normal(0.0, 0.1, 40))
    control = ControlPanel(labs=tuple(_labs(0.5 + rng.normal(0.0, 0.01, 40))), how="a matched draw")
    reading = InterlaboratoryComparison(labs, control).compute()
    assert reading.birge > 5.0
    assert reading.s_l == pytest.approx(0.1, rel=0.35)
    assert not reading.labs_understand_their_errors
    assert reading.excess_dispersion > 3.0
    assert reading.outliers


def test_a_lab_can_be_built_from_a_published_interval():
    lab = Lab.from_interval("published", 0.8, 0.76, 0.84)
    assert lab.u == pytest.approx(0.04 / 1.959963984540054)
    assert lab.expanded == pytest.approx(2.0 * lab.u)
    with pytest.raises(ValueError, match="coverage factor"):
        Lab.from_interval("x", 0.8, 0.7, 0.9, level=0.9)


def test_the_bootstrap_control_shares_a_measurand_by_construction():
    rng = np.random.default_rng(2)
    outcomes = (rng.random(800) < 0.6).astype(np.float64)
    control = bootstrap_control(outcomes, k=7, seed=1)
    assert control.k == 7
    assert all(lab.n == 800 for lab in control.labs)
    assert np.std([lab.value for lab in control.labs]) < 0.05
    assert "bootstrap resamples" in control.how


def test_a_control_panel_has_to_say_what_makes_it_one():
    with pytest.raises(ValueError, match="share a measurand"):
        ControlPanel(labs=tuple(_labs([0.1, 0.2, 0.3])), how="   ")


def test_m8_refuses_every_way_a_comparison_can_be_uninterpretable():
    labs = _labs([0.4, 0.5, 0.6, 0.55])
    control = bootstrap_control(np.array([1.0, 0.0] * 500), k=4, seed=0)
    no_control = InterlaboratoryComparison(labs).compute()
    assert isinstance(no_control, Refusal)
    assert no_control.reason is RefusalReason.NO_MATCHED_CONTROL
    one_lab = InterlaboratoryComparison(labs[:1], control).compute()
    assert isinstance(one_lab, Refusal) and one_lab.reason is RefusalReason.ACCESS_INSUFFICIENT
    silent = InterlaboratoryComparison([Lab(id="mute", value=0.5, u=0.0), *labs[1:]], control)
    out = silent.compute()
    assert isinstance(out, Refusal) and out.reason is RefusalReason.RECORD_INCOMPLETE
    mismatched = InterlaboratoryComparison(
        labs, ControlPanel(labs=control.labs[:2], how=control.how)
    ).compute()
    assert isinstance(mismatched, Refusal)
    assert mismatched.reason is RefusalReason.NO_MATCHED_CONTROL


def test_m8_refuses_a_control_run_at_a_different_sample_size():
    labs = _labs([0.4, 0.5, 0.6, 0.55], n=1000)
    small = ControlPanel(labs=tuple(_labs([0.5, 0.5, 0.5, 0.5], n=100)), how="a tenth of the items")
    out = InterlaboratoryComparison(labs, small).compute()
    assert isinstance(out, Refusal) and out.reason is RefusalReason.NO_MATCHED_CONTROL
    assert "different sample size" in out.detail


def test_m8_forwards_the_meta_analysis_refusal_below_three_labs():
    labs = _labs([0.4, 0.6])
    control = ControlPanel(labs=tuple(_labs([0.5, 0.5])), how="two matched draws")
    out = InterlaboratoryComparison(labs, control).compute()
    assert isinstance(out, Refusal)
    assert out.reason in {RefusalReason.ABOVE_LOD_BELOW_LOQ, RefusalReason.ESS_BELOW_FLOOR}
    assert out.instrument == "InterlaboratoryComparison"


# ---------------------------------------------------------------------------
# M9
# ---------------------------------------------------------------------------


def _detector(id: str, correct: np.ndarray, magnitude: float = 1.0) -> Detector:
    return Detector(id=id, margin=np.where(correct, magnitude, -magnitude))


def test_an_exact_copy_of_a_baseline_adds_exactly_nothing():
    """The redundancy case, planted: identical errors, so an ensemble cannot use the second copy."""
    rng = np.random.default_rng(0)
    a = rng.random(600) < 0.7
    base = [_detector("a", a)]
    reading = IncrementalValidityReading(_detector("a-copy", a), base, n_resamples=500).compute()
    assert reading.error_correlation == pytest.approx(1.0)
    assert reading.increment == 0.0
    assert reading.ci_low == 0.0 and reading.ci_high == 0.0
    assert reading.adds_nothing and reading.record.is_redundant
    assert "adds nothing measurable" in reading.says()


def test_a_decorrelated_instrument_adds_something_a_better_correlated_one_does_not():
    """The whole argument, planted: worse and independent beats better and redundant.

    Two baselines that are right on the easy 65% and wrong on the same hard 35%. The redundant
    candidate has that same profile and scores 0.646; the complementary one is right on exactly the
    items the pair gets wrong and scores 0.354. The ensemble can use the second and not the first,
    and a comparison by score alone would pick the wrong one.
    """
    rng = np.random.default_rng(11)
    n = 4000
    hard = rng.random(n) < 0.35
    shared = np.where(hard, -0.5, 0.5)
    base = [Detector(id="a", margin=shared), Detector(id="b", margin=shared)]
    redundant = Detector(id="red", margin=shared + rng.normal(0.0, 0.01, n))
    complementary = Detector(id="comp", margin=np.where(hard, 2.0, -0.4))

    red = IncrementalValidityReading(redundant, base, n_resamples=500).compute()
    comp = IncrementalValidityReading(complementary, base, n_resamples=500).compute()
    assert red.own_score > comp.own_score, "the redundant one scores better on its own"
    assert red.increment == pytest.approx(0.0) and red.adds_nothing
    assert comp.increment > 0.3, "and the worse one is the one that adds"
    assert red.error_correlation == pytest.approx(1.0)
    assert comp.error_correlation == pytest.approx(-1.0)


def test_the_increment_carries_an_interval_that_can_cover_zero():
    rng = np.random.default_rng(3)
    n = 300
    base = [_detector(f"b{i}", rng.random(n) < 0.7) for i in range(3)]
    noise = _detector("noise", rng.random(n) < 0.5)
    reading = IncrementalValidityReading(noise, base, n_resamples=800, seed=1).compute()
    assert reading.ci_low <= reading.increment <= reading.ci_high
    assert reading.n == n and reading.n_resamples == 800
    assert reading.combiner == "mean_margin"


def test_the_combining_rule_travels_on_the_reading_because_it_changes_the_number():
    rng = np.random.default_rng(9)
    n = 500
    base = [
        Detector(id="loud", margin=100.0 * (rng.random(n) - 0.4)),
        Detector(id="quiet", margin=0.01 * (rng.random(n) - 0.4)),
    ]
    own = Detector(id="own", margin=rng.random(n) - 0.45)
    plain = IncrementalValidityReading(own, base, n_resamples=200).compute()
    scaled = IncrementalValidityReading(
        own, base, combiner=standardised_margin, n_resamples=200
    ).compute()
    assert plain.combiner == "mean_margin" and scaled.combiner == "standardised_margin"
    assert plain.increment != scaled.increment


def test_the_error_correlation_is_between_errors_and_is_nan_when_it_cannot_be_taken():
    a = np.array([1.0, 0.0, 1.0, 0.0])
    assert phi(a, a) == pytest.approx(1.0)
    assert phi(a, 1.0 - a) == pytest.approx(-1.0)
    assert math.isnan(phi(np.zeros(4), a)), "a detector with no errors has none to correlate"


def test_m9_refuses_an_empty_baseline_set_and_a_ragged_item_set():
    rng = np.random.default_rng(1)
    own = _detector("own", rng.random(50) < 0.6)
    alone = IncrementalValidityReading(own, []).compute()
    assert isinstance(alone, Refusal) and alone.reason is RefusalReason.ACCESS_INSUFFICIENT
    ragged = IncrementalValidityReading(own, [_detector("b", rng.random(40) < 0.6)]).compute()
    assert isinstance(ragged, Refusal) and ragged.reason is RefusalReason.RECORD_INCOMPLETE
    nothing = IncrementalValidityReading().compute()
    assert isinstance(nothing, Refusal) and nothing.reason is RefusalReason.ACCESS_INSUFFICIENT


def test_m9_refuses_below_the_limit_of_detection_when_the_regime_says_so():
    rng = np.random.default_rng(1)
    own = _detector("own", rng.random(50) < 0.6)
    base = [_detector("b", rng.random(50) < 0.6)]
    out = IncrementalValidityReading(own, base).estimate(
        Context(regime_reading=RegimeReading.of(ABOVE_LOD=False))
    )
    assert isinstance(out, Refusal) and out.reason is RefusalReason.ENVELOPE_VIOLATED


def test_a_detector_can_be_built_from_scores_and_labels():
    scores = np.array([0.9, 0.2, 0.8, 0.1])
    labels = np.array([1, 0, 0, 1])
    det = Detector.from_scores("d", scores, labels, threshold=0.5)
    assert list(det.correct) == [True, True, False, False]
    assert det.score == pytest.approx(0.5)
    with pytest.raises(ValueError, match="same item set"):
        Detector.from_scores("d", scores, labels[:2])


def test_mean_margin_and_standardised_margin_are_both_available_and_differ():
    """A panel whose members' scales differ by two orders of magnitude is one member under the
    plain mean and two members after standardising, and the two rules then disagree on an item."""
    loud = Detector(id="loud", margin=np.array([10.0, -10.0, 10.0]))
    quiet = Detector(id="quiet", margin=np.array([-0.1, 0.3, -0.1]))
    assert list(mean_margin([loud, quiet]) > 0) == [True, False, True]
    assert list(standardised_margin([loud, quiet]) > 0) == [True, True, True]


# ---------------------------------------------------------------------------
# M11
# ---------------------------------------------------------------------------


def _rung(rung: int, value: float, *, n: int = 100, quantity: str = "") -> RungReading:
    return RungReading(
        rung=rung,
        value=value,
        n=n,
        access="GRADER: RECORD" if rung == 0 else "GRADER: REPLICATE",
        ci_low=value - 0.1,
        ci_high=value + 0.1,
        quantity=quantity,
    )


def test_the_disagreement_is_the_absolute_difference_and_the_transfer_is_t21():
    reading = RungDisagreement(_rung(0, 3.0), _rung(3, 1.9)).compute()
    assert reading.delta == pytest.approx(1.1)
    assert reading.relative_delta == pytest.approx(1.1 / 3.0)
    assert reading.transfer().name == "t21"
    assert reading.transfer().value == pytest.approx(1.1)
    assert not reading.intervals_overlap


def test_a_ladder_of_more_than_two_rungs_keeps_the_middle_ones_rather_than_dropping_them():
    instrument = RungDisagreement(rungs=[_rung(0, 3.0), _rung(1, 2.6), _rung(3, 1.9)])
    reading = instrument.compute()
    assert (reading.cheap_rung, reading.expensive_rung) == (0, 3)
    payload = instrument.payload(reading)
    assert [r["rung"] for r in payload["intermediate_rungs"]] == [1]


def test_m11_never_produces_a_field_that_picks_a_winner():
    instrument = RungDisagreement(_rung(0, 3.0), _rung(3, 1.9))
    reading = instrument.compute()
    payload = instrument.payload(reading)
    assert "cheap_value" in payload and "expensive_value" in payload
    for word in ("preferred", "resolved", "winner", "best_", "corrected", "adjusted"):
        assert not [k for k in payload if word in k], word
    assert "both numbers stand" in reading.says()
    assert reading.render().count("rung") >= 3


def test_m11_refuses_two_readings_of_two_different_quantities():
    out = compare_rungs(
        _rung(0, 3.0, quantity="grader.effective_group_size"),
        _rung(3, 1.9, quantity="grader.grr_percent"),
    )
    assert isinstance(out, Refusal) and out.reason is RefusalReason.UNIT_MISMATCH
    assert "two facts rather than a disagreement" in out.remedy


def test_m11_carries_the_units_group_whose_assertion_is_a_refusal():
    report = check_invariance(
        RungDisagreement(), "units", InvariancePayload(), relation=INVARIANT, run=lambda i, p: 1.0
    )
    assert report.passed and "refusal" in report.skipped
    assert check_unit_refusal(
        compare_rungs,
        _rung(0, 3.0, quantity="grader.effective_group_size"),
        _rung(3, 1.9, quantity="grader.grr_percent"),
    )


def test_m11_refuses_rungs_on_different_data_and_rungs_with_no_access_statement():
    ragged = RungDisagreement(_rung(0, 3.0, n=100), _rung(3, 1.9, n=90)).compute()
    assert isinstance(ragged, Refusal) and ragged.reason is RefusalReason.RECORD_INCOMPLETE
    assert "identical data" in ragged.remedy

    silent = RungReading(rung=3, value=1.9, n=100, access="  ")
    out = RungDisagreement(_rung(0, 3.0), silent).compute()
    assert isinstance(out, Refusal) and out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "access statement" in out.detail


def test_m11_refuses_one_rung_and_two_readings_at_the_same_rung():
    single = RungDisagreement(_rung(0, 3.0), quantity_id="grader.effective_group_size").compute()
    assert isinstance(single, Refusal) and single.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert single.statistics["rungs_supplied"] == 1
    same = RungDisagreement(_rung(0, 3.0), _rung(0, 2.9)).compute()
    assert isinstance(same, Refusal) and same.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "no ladder step" in same.detail


def test_m11_refuses_an_unregistered_quantity():
    out = RungDisagreement(_rung(0, 3.0), _rung(3, 1.9), quantity_id="not.a.quantity").compute()
    assert isinstance(out, Refusal) and out.reason is RefusalReason.UNIT_MISMATCH


def test_overlapping_intervals_are_reported_as_a_reading_about_the_ladder():
    reading = RungDisagreement(_rung(0, 2.0), _rung(3, 1.95)).compute()
    assert reading.intervals_overlap
    assert "not established at this n" in reading.render()


# ---------------------------------------------------------------------------
# Shared contracts
# ---------------------------------------------------------------------------


def test_every_instrument_emits_a_payload_the_store_codec_can_encode():
    codec = ValueCodec()
    rng = np.random.default_rng(0)
    pairs = [
        (SubstrateNoiseFloor(_blanks()), None),
        (
            InstrumentEffectReading(_Snapshot(640, 1_000, 10_000), StepBasis(steps=10, calls=640)),
            None,
        ),
        (
            UncertaintyBudgetReading(
                UncertaintyBudget(terms=(BudgetTerm(name="a", value=1.0),)),
                quantity_id="study.tau2",
            ),
            None,
        ),
        (
            InterlaboratoryComparison(
                _labs([0.4, 0.5, 0.6, 0.55]),
                bootstrap_control(np.array([1.0, 0.0] * 500), k=4, seed=0),
            ),
            None,
        ),
        (
            IncrementalValidityReading(
                _detector("own", rng.random(80) < 0.6),
                [_detector("b", rng.random(80) < 0.6), _detector("c", rng.random(80) < 0.6)],
                n_resamples=100,
            ),
            None,
        ),
        (RungDisagreement(_rung(0, 3.0), _rung(3, 1.9)), None),
    ]
    for instrument, _ in pairs:
        reading = instrument.compute()
        assert not isinstance(reading, Refusal), (instrument.name, reading)
        payload = instrument.payload(reading)
        assert "baselines" in payload and payload["baselines"]
        assert "says" in payload
        encoded = codec.encode(payload, None)
        assert codec.decode(encoded, None)["says"] == payload["says"]


def test_every_instrument_returns_evidence_or_a_refusal_and_never_raises():
    """The contract's return type, exercised on the empty construction of each, which all refuse."""
    for instrument in instances():
        out = instrument.estimate(Context())
        assert isinstance(out, Refusal), instrument.name
        assert out.remedy.strip()
        assert out.reason in set(RefusalReason)


def test_calling_measure_on_a_refusing_instrument_raises_rather_than_softening_the_refusal():
    with pytest.raises(ValueError, match="declines to produce"):
        SubstrateNoiseFloor().measure(Context())


def test_the_preflight_names_the_checks_it_could_not_run():
    result = SubstrateNoiseFloor(_blanks()).preflight(Context())
    assert result.ok
    assert "access" in result.unchecked
    assert any("envelope" in u for u in result.unchecked)
    assert "limit of detection" in result.unchecked


def test_the_access_refusal_names_what_would_close_the_gap():
    ctx = Context(access={Component.GRADER: Access.RECORD}, substrate=Substrate.PROGRAM)
    out = SubstrateNoiseFloor(_blanks()).estimate(ctx)
    assert isinstance(out, Refusal) and out.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "record" in out.remedy.lower()


def test_a_phase_mismatch_is_reported_as_a_phase_mismatch():
    ctx = Context(phase=Phase.DEPLOYED)
    out = InstrumentEffectReading(_Snapshot(10, 1, 1), StepBasis(steps=1, calls=10)).estimate(ctx)
    assert isinstance(out, Refusal) and out.reason is RefusalReason.PHASE_MISMATCH

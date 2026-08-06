"""`core/budget.py` and `core/reference.py`: the GUM table, the limits, and the trust cap.

What this file has to show, in three parts: an instrument below LOD returns a `Refusal` carrying both
numbers; a budget whose terms do not compose fails a property test; a `ReferenceMaterial` with
`u_homogeneity is None` caps the trust level at CALIBRATED, and the cap is asserted.

The third is the one with teeth. It is the rule that would have changed how `CAL-TRANSFER` reads,
and the test that pins it is the test that stops someone quietly relaxing it later.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reward_lens.core.budget import (
    DIVISORS,
    BudgetLintError,
    BudgetTerm,
    CalibrationCurve,
    LimitOfDetection,
    LODCache,
    SubstrateKey,
    UncertaintyBudget,
    budget_of,
    refuse_below_lod,
)
from reward_lens.core.gates import CalibrationRef, compute_trust
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.reference import (
    CalibrationChain,
    MatrixDescription,
    ReferenceMaterial,
    Transfer,
    ladder_disagreement,
    uncertified_refusal,
)
from reward_lens.core.types import TrustLevel

# ---------------------------------------------------------------------------
# The budget composes, and the composition is a property
# ---------------------------------------------------------------------------


def test_quadrature_composition_against_a_hand_computed_case():
    """3-4-5, so the arithmetic is checkable by eye rather than against itself."""
    b = budget_of(a=3.0, b=4.0)
    assert b.combined == pytest.approx(5.0)
    assert b.expanded == pytest.approx(10.0)  # k=2
    assert b.shares() == pytest.approx({"a": 9 / 25, "b": 16 / 25})


def test_sensitivity_coefficients_multiply_before_squaring():
    """`c_i * u_i`, not `c_i * u_i^2`. Getting this backwards is silent and quadratic."""
    b = UncertaintyBudget(
        terms=(
            BudgetTerm(name="grader", value=2.0, sensitivity=1.5),
            BudgetTerm(name="sampling", value=4.0, sensitivity=0.5),
        )
    )
    assert b.combined == pytest.approx(math.hypot(3.0, 2.0))
    assert b.dominant.name == "grader"


@pytest.mark.parametrize("n_terms", [1, 2, 3, 5, 9])
def test_composition_is_the_quadrature_sum_for_any_number_of_terms(n_terms):
    """The composition property, over random budgets."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        values = rng.uniform(0.0, 10.0, size=n_terms)
        sens = rng.uniform(-3.0, 3.0, size=n_terms)
        b = UncertaintyBudget(
            terms=tuple(
                BudgetTerm(name=f"t{i}", value=float(v), sensitivity=float(c))
                for i, (v, c) in enumerate(zip(values, sens))
            )
        )
        expected = math.sqrt(float(np.sum((values * sens) ** 2)))
        assert b.combined == pytest.approx(expected, rel=1e-12, abs=1e-12)
        assert sum(b.shares().values()) == pytest.approx(1.0)


def test_a_budget_that_does_not_compose_is_rejected_at_construction():
    """Two terms with one name compose into a number nobody can attribute."""
    with pytest.raises(BudgetLintError, match="duplicate term names"):
        UncertaintyBudget(terms=(BudgetTerm(name="u", value=1.0), BudgetTerm(name="u", value=2.0)))

    with pytest.raises(BudgetLintError, match="no term named"):
        UncertaintyBudget(
            terms=(BudgetTerm(name="a", value=1.0),), correlations={("a", "ghost"): 0.5}
        )


def test_a_negative_standard_uncertainty_is_rejected():
    with pytest.raises(ValueError, match="magnitude"):
        BudgetTerm(name="u", value=-1.0)


def test_correlated_terms_compose_larger_than_independent_ones():
    """Assuming independence is the commonest way a budget understates itself."""
    indep = UncertaintyBudget(
        terms=(BudgetTerm(name="a", value=3.0), BudgetTerm(name="b", value=4.0))
    )
    corr = UncertaintyBudget(
        terms=(BudgetTerm(name="a", value=3.0), BudgetTerm(name="b", value=4.0)),
        correlations={("a", "b"): 1.0},
    )
    assert indep.combined == pytest.approx(5.0)
    assert corr.combined == pytest.approx(7.0)  # fully correlated: they add linearly


def test_type_b_half_widths_use_the_right_divisor():
    """The whole of a Type B evaluation is the divisor, and a wrong one scales the term by 1.7x."""
    assert BudgetTerm.from_half_width("r", 3.0, "rectangular").value == pytest.approx(
        3.0 / math.sqrt(3)
    )
    assert BudgetTerm.from_half_width("t", 3.0, "triangular").value == pytest.approx(
        3.0 / math.sqrt(6)
    )
    assert BudgetTerm.from_half_width("n", 3.0, "normal").value == pytest.approx(3.0)
    assert set(DIVISORS) == {"normal", "rectangular", "triangular", "u_shaped"}
    with pytest.raises(ValueError, match="unknown distribution"):
        BudgetTerm.from_half_width("x", 1.0, "lognormal")


def test_the_dominant_term_is_named_and_it_need_not_be_sampling_noise():
    """The spec's observation, made mechanical: a budget that cannot name its dominant term is
    not doing its job."""
    b = UncertaintyBudget(
        terms=(
            BudgetTerm(name="sampling", value=0.01, kind="A", dof=63),
            BudgetTerm(name="grader_replication", value=0.14, kind="A", dof=9),
            BudgetTerm(name="engine_disagreement", value=0.03, kind="B"),
        )
    )
    assert b.dominant.name == "grader_replication"
    assert b.shares()["grader_replication"] > 0.9
    assert "dominant term: grader_replication" in b.render()


def test_effective_dof_is_none_rather_than_infinite_when_a_term_lacks_it():
    """Substituting infinity is the common shortcut and it narrows the interval, which is the
    wrong direction for a shortcut to err in."""
    with_all = UncertaintyBudget(
        terms=(BudgetTerm(name="a", value=1.0, dof=9), BudgetTerm(name="b", value=1.0, dof=9))
    )
    assert with_all.effective_dof() == pytest.approx(18.0)
    missing = UncertaintyBudget(
        terms=(BudgetTerm(name="a", value=1.0, dof=9), BudgetTerm(name="b", value=1.0))
    )
    assert missing.effective_dof() is None


def test_the_coverage_factor_follows_the_degrees_of_freedom():
    """`k = 2` is a large-sample approximation and the table knows when it does not apply.

    The full set of cases, with the numbers each of them moved, is in
    `tests/acceptance/test_debt_k_fixes.py`. What belongs in the acceptance file for this package
    is the clause itself: an expanded uncertainty quoted at a level has to carry that level.
    """
    floor = UncertaintyBudget(terms=(BudgetTerm(name="u_char", value=0.06, kind="A", dof=1),))
    assert floor.coverage_factor == pytest.approx(12.70620, rel=1e-5)
    assert floor.expanded == pytest.approx(0.7623720, rel=1e-6)
    assert floor.coverage_achieved == pytest.approx(0.95)

    ample = UncertaintyBudget(
        terms=(
            BudgetTerm(name="a", value=1.0, kind="A", dof=400),
            BudgetTerm(name="b", value=1.0, kind="A", dof=400),
        )
    )
    assert ample.coverage_factor == pytest.approx(2.0)
    assert ample.coverage_achieved >= ample.coverage_level

    bare = UncertaintyBudget(terms=(BudgetTerm(name="a", value=1.0),))
    assert bare.coverage_factor == 2.0
    assert bare.coverage_achieved == pytest.approx(0.9545, abs=1e-4)


def test_welch_satterthwaite_runs_on_the_independent_combination():
    """It has no correlated form. Feeding the correlated `u_c` in returned `nu_eff = 0.08` on two
    four-dof terms at r = -0.9, whose t quantile is 1.2e15."""
    for r in (-0.9, 0.0, 0.9):
        b = UncertaintyBudget(
            terms=(
                BudgetTerm(name="a", value=1.0, kind="A", dof=4),
                BudgetTerm(name="b", value=1.0, kind="A", dof=4),
            ),
            correlations={("a", "b"): r},
        )
        assert b.effective_dof() == pytest.approx(8.0)
        assert b.combined == pytest.approx(math.sqrt(2.0 + 2.0 * r))


def test_a_correlation_must_be_a_correlation():
    """`r = -3.0` composed to a variance of -4 and the square root clamped it to zero."""
    with pytest.raises(BudgetLintError, match=r"lies in \[-1, 1\]"):
        UncertaintyBudget(
            terms=(BudgetTerm(name="a", value=1.0), BudgetTerm(name="b", value=1.0)),
            correlations={("a", "b"): -3.0},
        )


# ---------------------------------------------------------------------------
# The limits of detection, and the three-outcome rule
# ---------------------------------------------------------------------------


def test_lod_and_loq_follow_the_stated_formulas():
    lod = LimitOfDetection(sigma_blank=0.02, sensitivity=0.5)
    assert lod.lod == pytest.approx(3.3 * 0.02 / 0.5)
    assert lod.loq == pytest.approx(10.0 * 0.02 / 0.5)


def test_the_decision_rule_has_three_outcomes():
    lod = LimitOfDetection(sigma_blank=0.02, sensitivity=0.5)  # LOD 0.132, LOQ 0.4
    assert lod.verdict(0.05) == "below_lod"
    assert lod.verdict(0.25) == "above_lod_below_loq"
    assert lod.verdict(0.90) == "quantifiable"
    # Magnitude, so a negative effect of the same size decides the same way.
    assert lod.verdict(-0.90) == "quantifiable"
    # And it stays three. An override that puts LOQ below LOD deletes the middle outcome for
    # every reading on the real line without saying so, so the ordering is enforced rather than
    # assumed.
    with pytest.raises(ValueError, match="three outcomes"):
        LimitOfDetection(sigma_blank=0.02, sensitivity=0.5, lod_k=10.0, loq_k=3.3)


def test_a_reading_below_lod_returns_a_refusal_carrying_both_numbers():
    """Never an exception, never a zero, and both limits are in the detail."""
    lod = LimitOfDetection(
        sigma_blank=0.02,
        sensitivity=0.5,
        key=SubstrateKey(
            model="gemma-3-1b-it", engine="vllm", dtype="bf16", attention_impl="eager"
        ),
        blank_n=30,
    )
    r = refuse_below_lod("chi.susceptibility", 0.05, lod)
    assert isinstance(r, Refusal)
    assert r.reason is RefusalReason.BELOW_LOD
    assert r.statistics["reading"] == pytest.approx(0.05)
    assert r.statistics["lod"] == pytest.approx(0.132)
    assert r.statistics["loq"] == pytest.approx(0.4)
    assert "0.132" in r.detail and "0.05" in r.detail
    assert r.remedy  # a Refusal cannot be constructed without one, but assert it says something
    assert "averaging more of them will not help" in r.remedy


def test_a_non_positive_slope_makes_the_limits_infinite_rather_than_zero():
    """An instrument whose reading does not respond to dose has no detection limit; it has no
    calibration. Returning zero would make everything quantifiable."""
    flat = LimitOfDetection(sigma_blank=0.02, sensitivity=0.0)
    assert not flat.is_determinate
    assert flat.lod == math.inf and flat.loq == math.inf
    assert flat.verdict(1e9) == "below_lod"
    assert "does not respond to dose" in flat.render()


def test_the_noise_floor_composes_as_a_budget_term():
    """The blank sigma is an uncertainty on the reading, so it belongs in the table, not beside it."""
    lod = LimitOfDetection(sigma_blank=0.02, sensitivity=0.5, blank_n=30)
    term = lod.as_term()
    assert term.kind == "B"
    assert term.value == pytest.approx(0.02)
    assert term.dof == 29
    assert UncertaintyBudget(terms=(term,)).combined == pytest.approx(0.02)


def test_limits_are_cached_per_configuration_not_per_model():
    """Two configurations of the same weights are two measurement instruments."""
    cache = LODCache()
    eager = SubstrateKey(model="m", engine="vllm", attention_impl="eager")
    compiled = SubstrateKey(model="m", engine="vllm", attention_impl="compiled")
    cache.put(LimitOfDetection(sigma_blank=0.0018, sensitivity=1.0, key=eager))
    cache.put(LimitOfDetection(sigma_blank=0.0035, sensitivity=1.0, key=compiled))
    assert len(cache) == 2
    assert cache.get(eager).sigma_blank != cache.get(compiled).sigma_blank
    assert cache.get(SubstrateKey(model="m", engine="hf")) is None


def test_an_lod_without_a_key_cannot_be_cached():
    with pytest.raises(ValueError, match="not a property of anything"):
        LODCache().put(LimitOfDetection(sigma_blank=0.1, sensitivity=1.0))


# ---------------------------------------------------------------------------
# The calibration curve, whose slope is S
# ---------------------------------------------------------------------------


def test_hill_response_and_analytic_slope_agree_with_finite_differences():
    curve = CalibrationCurve(e_max=1.0, ec50=0.5, hill_n=2.0)
    assert curve.response(0.5) == pytest.approx(0.5)  # by definition of EC50
    for dose in (0.1, 0.5, 1.0, 3.0):
        h = 1e-6
        numeric = (curve.response(dose + h) - curve.response(dose - h)) / (2 * h)
        assert curve.slope(dose) == pytest.approx(numeric, rel=1e-4)


def test_a_slope_cannot_be_quoted_without_a_dose():
    """The Hill slope varies with dose, so a single S without one is meaningless, and defaulting
    to EC50 would silently quote the maximum."""
    with pytest.raises(ValueError, match="quoted at one"):
        CalibrationCurve(e_max=1.0, ec50=0.5).slope()
    assert CalibrationCurve(e_max=1.0, ec50=0.5, at_dose=0.5).slope() > 0


def test_therapeutic_index_is_the_ratio_the_surgery_result_measured_at_one_dose():
    curve = CalibrationCurve(e_max=1.0, ec50=0.2)
    assert curve.therapeutic_index(td50=0.8) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# The reference material, and the trust cap
# ---------------------------------------------------------------------------


ORGANISM = MatrixDescription(system="planted 0.6B organism", scale="0.6B")
PRODUCTION = MatrixDescription(system="production reward model", scale="8B")


def _reference(**kw) -> ReferenceMaterial:
    base = dict(
        id="ref:tiny-plant",
        kind="planted_organism",
        assigned_value=0.75,
        u_characterisation=0.06,
        matrix=ORGANISM,
    )
    base.update(kw)
    return ReferenceMaterial(**base)  # type: ignore[arg-type]


def test_an_uncharacterised_reference_caps_trust_at_calibrated():
    """The rule that would have changed how CAL-TRANSFER reads."""
    ref = _reference(u_homogeneity=None, u_stability=None)
    assert not ref.is_certified
    assert ref.uncharacterised == ("u_homogeneity", "u_stability")
    assert ref.trust_cap() is TrustLevel.CALIBRATED

    cal = CalibrationRef(scorecard_entry="ev:abc", organism_family="tiny")
    # Without the reference this is ADJUDICATED. With it, the cap bites.
    assert (
        compute_trust(calibration=cal, registered=True, adjudicated=True) is TrustLevel.ADJUDICATED
    )
    assert (
        compute_trust(calibration=cal, registered=True, adjudicated=True, reference=ref)
        is TrustLevel.CALIBRATED
    )
    # And a registered-but-uncalibrated reading is capped too, not merely held level.
    assert compute_trust(calibration=None, registered=True, reference=ref) is TrustLevel.CALIBRATED


def test_a_certified_reference_does_not_cap():
    ref = _reference(u_homogeneity=0.14, u_stability=0.03)
    assert ref.is_certified
    assert ref.trust_cap() is TrustLevel.ADJUDICATED
    cal = CalibrationRef(scorecard_entry="ev:abc", organism_family="tiny")
    assert (
        compute_trust(calibration=cal, registered=True, adjudicated=True, reference=ref)
        is TrustLevel.ADJUDICATED
    )


def test_the_cap_never_promotes():
    """A cap is a maximum. An exploratory reading against a perfect reference stays exploratory."""
    ref = _reference(u_homogeneity=0.14, u_stability=0.03)
    assert (
        compute_trust(calibration=None, registered=False, reference=ref) is TrustLevel.EXPLORATORY
    )


def test_u_crm_is_none_rather_than_a_partial_sum():
    """Summing the terms that happen to exist makes an uncharacterised reference look better than
    a characterised one with a large homogeneity term."""
    ref = _reference(u_homogeneity=None, u_stability=0.03)
    assert ref.u_crm is None
    assert ref.u_crm_lower_bound == pytest.approx(math.hypot(0.06, 0.03))
    certified = _reference(u_homogeneity=0.14, u_stability=0.03)
    assert certified.u_crm == pytest.approx(math.sqrt(0.06**2 + 0.14**2 + 0.03**2))


def test_the_status_line_is_the_sentence_that_reaches_the_reader():
    ref = _reference(u_homogeneity=None)
    line = ref.status_line()
    assert "reference uncertainty not characterised" in line
    assert "trust capped at CALIBRATED" in line


def test_uncertified_refusal_names_what_to_do():
    r = uncertified_refusal("cal.transfer", _reference(u_homogeneity=None))
    assert r.reason is RefusalReason.REFERENCE_UNCERTIFIED
    assert "at least three seeds" in r.remedy
    assert r.statistics["u_homogeneity"] is None


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------


def test_the_chain_composes_the_five_named_terms():
    """`u_total² = u₁² + t₂₁² + t₃₂² + u_CRM² + u_instrument²`, against a hand sum."""
    ref = _reference(u_homogeneity=0.14, u_stability=0.03)
    chain = CalibrationChain(
        reference=ref,
        transfers=(
            Transfer(from_level="working_method", to_level="reference_method", value=0.10),
            Transfer(from_level="reference_method", to_level="primary", value=0.419),
        ),
        u_working=0.05,
        u_instrument=0.02,
    )
    expected = math.sqrt(0.05**2 + 0.10**2 + 0.419**2 + 0.06**2 + 0.14**2 + 0.03**2 + 0.02**2)
    assert chain.u_total == pytest.approx(expected)
    assert chain.as_budget().dominant.name == "t32"


def test_the_transfer_names_follow_the_chain_diagram():
    assert Transfer(from_level="reference_method", to_level="primary", value=0.4).name == "t32"
    assert (
        Transfer(from_level="working_method", to_level="reference_method", value=0.1).name == "t21"
    )


def test_a_transfer_is_a_magnitude():
    with pytest.raises(ValueError, match="cannot be negative"):
        Transfer(from_level="working_method", to_level="reference_method", value=-0.1)


def test_an_uncertified_chain_reports_a_lower_bound_rather_than_a_smaller_total():
    """Dropping the missing term and presenting a smaller total is the failure mode."""
    chain = CalibrationChain(
        reference=_reference(u_homogeneity=None),
        transfers=(Transfer(from_level="reference_method", to_level="primary", value=0.419),),
        u_working=0.05,
    )
    assert chain.u_total is None
    assert chain.u_total_lower_bound == pytest.approx(math.sqrt(0.05**2 + 0.419**2 + 0.06**2))
    assert "not computable" in chain.render()
    assert chain.trust_cap() is TrustLevel.CALIBRATED


def test_a_matrix_mismatch_is_reported_by_name():
    """The metrological name for the CAL-TRANSFER result: calibrated in one matrix, applied in
    another."""
    chain = CalibrationChain(
        reference=_reference(u_homogeneity=0.1, u_stability=0.0),
        working_matrix=PRODUCTION,
    )
    assert "calibrated in planted 0.6B organism" in chain.matrix_mismatch()
    assert "applied to production reward model" in chain.matrix_mismatch()
    assert "matrix effect" in chain.render()


def test_matching_matrices_report_no_mismatch():
    chain = CalibrationChain(
        reference=_reference(u_homogeneity=0.1, u_stability=0.0), working_matrix=ORGANISM
    )
    assert chain.matrix_mismatch() == ""


def test_ladder_disagreement_turns_two_rungs_into_a_transfer_row():
    """M11 in one call, because the only work is remembering to record it."""
    t = ladder_disagreement(cheap=4.2, expensive=2.1, n=64)
    assert t.value == pytest.approx(2.1)
    assert t.name == "t21"
    assert t.as_term().dof == 63
    assert "4.2" in t.note and "2.1" in t.note

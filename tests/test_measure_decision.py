"""Unit tests for the contract layer, N5 to N8.

The arithmetic here is closed form, so most of these are exact rather than approximate: the
zero-weight theorem is asserted with `== 0.0` and not with a tolerance, the matrix ordering is
checked against fractions computed by hand, and the sorting theorem's identity for two components is
checked against `(1 + x)^2 / (1 + 2x)` at three values of `x`.

The clause runs on real recorded scores and lives in
`tests/acceptance/test_w4_decision.py`. This file is the arithmetic, the refusals and the
declarations.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reward_lens.core.envelope import RegimeReading
from reward_lens.core.evidence import Evidence
from reward_lens.core.invariance import InvariancePayload, check_invariance
from reward_lens.core.quantity import QUANTITIES, load_quantities
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Phase
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.decision import (
    ASSUMPTION_KEYS,
    ASSUMPTIONS,
    CATALOGUE_RECORDS,
    COVARIANT_INVERSE,
    DECISION,
    PROPOSED,
    PROPOSED_INSTRUMENTS,
    PROPOSED_MIN_ACCESS,
    ContractParameters,
    EqualCompensation,
    NoiseAndAngle,
    OptimalWeights,
    ParameterSource,
    SortingCutoff,
    Sweep,
    as_catalogue_rows,
    as_yaml_rows,
    assumption,
    assumptions_payload,
    clean_measure_congruity,
    component_congruity,
    congruity,
    contract_value,
    equal_compensation,
    information_coefficient,
    noise_and_angle,
    noise_correlation_from_residuals,
    noise_to_signal,
    optimal_weights,
    optimal_weights_diagonal,
    principal_surplus,
    recommend_weights,
    register_proposed,
    render_assumptions,
    shrinkage,
    sorting_cutoff,
    sweep_weights,
    two_task_surplus,
    unmeasurable_correction,
)

THREE = ("unit_tests", "format_ok", "judge")


@pytest.fixture(autouse=True)
def _registry():
    """Register this package's six proposals, then take them back out.

    The quantity registry is process-global. Leaving six rows in it changes what every later test in
    the same process sees, and `tests/acceptance/test_w1_kernel.py` asserts an exact count, so a
    module that registers and does not clean up breaks a test in a package it has nothing to do
    with. That is E40's pattern for the third time, and the fix is the one E40 settled
    on: snapshot what was there and remove whatever appeared, rather than listing the six ids, which
    goes stale the moment a seventh is proposed.
    """
    load_quantities()
    before = set(QUANTITIES)
    yield
    for added in set(QUANTITIES) - before:
        QUANTITIES._items.pop(added, None)


def three_components(**overrides) -> ContractParameters:
    """A three-component composite with usefully different noise. Synthetic, and used for shape."""
    kwargs = dict(
        benefit=[1.0, 1.0, 1.0],
        cost_curvature=np.eye(3),
        noise=np.diag([0.05, 0.8, 0.5]),
        risk_aversion=1.0,
        sensitivity=np.eye(3),
    )
    kwargs.update(overrides)
    return ContractParameters.supplied(THREE, **kwargs).assume_equal_effort()


# ===========================================================================
# The five assumptions
# ===========================================================================


def test_there_are_exactly_five_assumptions_and_two_of_them_are_checkable():
    assert len(ASSUMPTIONS) == 5
    assert ASSUMPTION_KEYS == (
        "LINEAR_CONTRACT",
        "GAUSSIAN_NOISE",
        "CARA_AND_COST_CURVATURE",
        "COMMITMENT_ONE_PERIOD",
        "AGENT_KNOWS_THE_CONTRACT",
    )
    checkable = [a.key for a in ASSUMPTIONS if a.checkable]
    assert checkable == ["GAUSSIAN_NOISE", "COMMITMENT_ONE_PERIOD"]
    for a in ASSUMPTIONS:
        assert a.statement and a.holds_when and a.breaks and a.check


def test_every_assumption_names_what_would_test_it_or_says_nothing_can():
    """`checkable` is a claim in both directions and each one has to be backed by the `check` text."""
    for a in ASSUMPTIONS:
        if a.checkable:
            assert "testable" in a.check
        else:
            assert (
                "not testable" in a.check or "not measurable" in a.check or "only after" in a.check
            )


def test_render_assumptions_prints_all_five_and_assumption_raises_on_a_typo():
    text = render_assumptions()
    for key in ASSUMPTION_KEYS:
        assert key in text
    assert assumption("GAUSSIAN_NOISE").checkable
    with pytest.raises(KeyError):
        assumption("GAUSIAN_NOISE")
    assert [a["key"] for a in assumptions_payload()] == list(ASSUMPTION_KEYS)


def test_every_rendered_reading_in_the_package_carries_all_five():
    """The property the package is built around, checked on all four readings at once."""
    p = three_components()
    renders = [
        recommend_weights(p).render(),
        equal_compensation([0.5, 0.2, 0.3], p).render(),
        sorting_cutoff(p, n_contracts=2).render(),
        noise_and_angle(p).render(),
    ]
    for text in renders:
        for key in ASSUMPTION_KEYS:
            assert key in text, key


# ===========================================================================
# The parameter set and its provenance
# ===========================================================================


def test_nothing_has_a_plausible_default_and_an_unstated_parameter_refuses():
    p = ContractParameters.supplied(
        THREE,
        benefit=[1.0, 1.0, 1.0],
        cost_curvature=np.eye(3),
        noise=np.eye(3),
        risk_aversion=1.0,
    )
    assert p.unknown() == ("sensitivity", "effort")
    bad = p.refuse_unstated("X", ["sensitivity"])
    assert isinstance(bad, Refusal)
    assert bad.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "sensitivity" in bad.detail
    assert "weight sweep" in bad.remedy
    assert p.refuse_unstated("X", ["noise", "risk_aversion"]) is None


def test_assuming_unit_sensitivity_is_a_named_act_that_records_itself():
    p = three_components(sensitivity=None).assume_unit_sensitivity()
    assert p.source["sensitivity"] is ParameterSource.ASSUMED
    assert "M = I" in p.note["sensitivity"]
    assert "Gauge-fix" in p.note["sensitivity"]
    assert np.array_equal(p.sensitivity, np.eye(3))
    rows = {r["parameter"]: r for r in p.provenance_rows()}
    assert rows["sensitivity"]["source"] == "ASSUMED"
    assert "ASSUMED" in p.render_provenance()


def test_the_parameter_set_rejects_a_matrix_that_is_not_a_hessian_or_a_covariance():
    with pytest.raises(ValueError, match="not symmetric"):
        ContractParameters.supplied(
            ("a", "b"),
            benefit=[1, 1],
            cost_curvature=[[1.0, 2.0], [0.0, 1.0]],
            noise=np.eye(2),
            risk_aversion=1.0,
        )
    with pytest.raises(ValueError, match="not convex"):
        ContractParameters.supplied(
            ("a", "b"),
            benefit=[1, 1],
            cost_curvature=[[1.0, 3.0], [3.0, 1.0]],
            noise=np.eye(2),
            risk_aversion=1.0,
        )
    with pytest.raises(ValueError, match="not a covariance"):
        ContractParameters.supplied(
            ("a", "b"),
            benefit=[1, 1],
            cost_curvature=np.eye(2),
            noise=[[1.0, 3.0], [3.0, 1.0]],
            risk_aversion=1.0,
        )
    with pytest.raises(ValueError, match="component name"):
        ContractParameters.supplied(
            ("a",), benefit=[1, 1], cost_curvature=np.eye(2), noise=np.eye(2), risk_aversion=1.0
        )


def test_the_gauge_action_scales_sigma_by_a_squared_and_m_by_a():
    p = three_components()
    q = p.rescale(3.0)
    assert np.allclose(q.noise, 9.0 * p.noise)
    assert np.allclose(q.sensitivity, 3.0 * p.sensitivity)
    assert np.allclose(q.benefit, p.benefit)
    assert np.allclose(q.cost_curvature, p.cost_curvature)
    with pytest.raises(ValueError, match="projection"):
        p.rescale(0.0)


def test_the_noise_correlation_has_no_default_and_a_covariance_passed_as_one_is_caught():
    class FakeGauge:
        def __init__(self, sd):
            self.sigma_grr = sd

    class FakeStudy:
        def __init__(self, sd, rung=1):
            self.gauge = FakeGauge(sd)
            self.rung = rung

    from reward_lens.measure.decision import noise_from_gauge_studies

    studies = [("a", FakeStudy(0.5)), ("b", FakeStudy(2.0))]
    sigma, note = noise_from_gauge_studies(studies, np.array([[1.0, 0.4], [0.4, 1.0]]))
    assert sigma[0, 0] == pytest.approx(0.25)
    assert sigma[1, 1] == pytest.approx(4.0)
    assert sigma[0, 1] == pytest.approx(0.4 * 0.5 * 2.0)
    assert "grader.variance_components" in note
    with pytest.raises(ValueError, match="unit diagonal"):
        noise_from_gauge_studies(studies, np.array([[0.25, 0.4], [0.4, 4.0]]))


def test_the_noise_correlation_from_residuals_needs_paired_cells():
    rng = np.random.default_rng(0)
    shared = rng.normal(size=(400, 1))
    resid = np.hstack(
        [shared + rng.normal(0, 0.5, (400, 1)), shared + rng.normal(0, 0.5, (400, 1))]
    )
    r = noise_correlation_from_residuals(resid)
    assert r.shape == (2, 2)
    assert 0.5 < r[0, 1] < 0.95
    with pytest.raises(ValueError, match="zero residual spread"):
        noise_correlation_from_residuals(np.hstack([resid[:, :1], np.zeros((400, 1))]))


# ===========================================================================
# N5: the formula, the ordering, and the two theorems
# ===========================================================================


def test_the_matrix_ordering_is_c_before_sigma_and_transposing_it_triples_one_weight():
    """The same worked example the clause carries, asserted here as arithmetic.

    `C'' = [[2, 1], [1, 4]]`, `Sigma = diag(1, 3)`, `r = 0.5`, `B' = (1, 1)`. The two products
    differ off the diagonal, `C'' Sigma = [[2, 3], [1, 12]]` against `Sigma C'' = [[2, 1], [3, 12]]`,
    and the recommendations they produce are `(22/53, 6/53)` and `(26/53, 2/53)`. Transposing the
    product divides the weight on the second component by exactly three. The multiplication is
    written out entry by entry in `tests/acceptance/test_w4_decision.py`, where the clause lives.
    """
    c = np.array([[2.0, 1.0], [1.0, 4.0]])
    sigma = np.diag([1.0, 3.0])
    r, b = 0.5, np.array([1.0, 1.0])

    assert not np.allclose(c @ sigma, sigma @ c)
    assert np.array_equal(c @ sigma, np.array([[2.0, 3.0], [1.0, 12.0]]))
    assert np.array_equal(sigma @ c, np.array([[2.0, 1.0], [3.0, 12.0]]))

    p = ContractParameters.supplied(
        ("a", "b"), benefit=b, cost_curvature=c, noise=sigma, risk_aversion=r
    ).assume_unit_sensitivity()
    alpha = optimal_weights(p)
    assert alpha[0] == pytest.approx(22.0 / 53.0, rel=1e-12)
    assert alpha[1] == pytest.approx(6.0 / 53.0, rel=1e-12)
    assert np.allclose((np.eye(2) + r * c @ sigma) @ alpha, b, atol=1e-14)

    wrong = np.linalg.solve(np.eye(2) + r * sigma @ c, b)
    assert wrong[0] == pytest.approx(26.0 / 53.0, rel=1e-12)
    assert wrong[1] == pytest.approx(2.0 / 53.0, rel=1e-12)
    assert alpha[1] / wrong[1] == pytest.approx(3.0, rel=1e-12)


def test_the_general_form_reduces_to_the_diagonal_formula_the_specification_states():
    b, cd, sd, r = [1.0, 2.0], [2.0, 4.0], [1.0, 3.0], 0.5
    p = ContractParameters.supplied(
        ("a", "b"),
        benefit=b,
        cost_curvature=np.diag(cd),
        noise=np.diag(sd),
        risk_aversion=r,
    ).assume_unit_sensitivity()
    assert np.allclose(optimal_weights(p), optimal_weights_diagonal(b, cd, sd, r))
    assert optimal_weights_diagonal(b, cd, sd, r)[0] == pytest.approx(1.0 / (1 + 0.5 * 2 * 1))


def test_the_shrinkage_factor_is_the_regressional_goodhart_factor():
    """`1/(1 + r C sigma^2)` and `Var(X)/(Var(X) + Var(Z))` are one object, checked rather than said.

    Put `Var(X) = 1/(r C)` and `Var(Z) = sigma^2` and the two expressions coincide identically.
    Contract theory derives it as what the principal should do; the Goodhart scaling literature
    derives it as what happens if the principal does not.
    """
    for r, c, s2 in [(0.5, 2.0, 1.0), (3.0, 0.25, 4.0), (1.0, 1.0, 0.1)]:
        var_x = 1.0 / (r * c)
        assert shrinkage(r, c, s2) == pytest.approx(var_x / (var_x + s2), rel=1e-14)
    with pytest.raises(ValueError, match="not positive"):
        shrinkage(-1.0, 1.0, 10.0)


def test_the_recommendation_maximises_the_surplus_it_reports():
    """Optimality is computed rather than asserted: every perturbation lowers the objective."""
    p = three_components()
    alpha = optimal_weights(p)
    best = principal_surplus(alpha, p)
    rng = np.random.default_rng(0)
    for _ in range(200):
        step = rng.normal(0.0, 0.05, size=3)
        assert principal_surplus(alpha + step, p) < best + 1e-15


def test_the_unmeasurable_task_correction_turns_the_gain_down_by_a_fifth():
    """`B' = (1, 1)`, `C'' = [[2, 1], [1, 4]]`, `sigma_1^2 = 1`, `r = 0.5`, worked by hand.

    `C_12/C_22 = 1/4 = 0.25`, so the numerator is `1 - 1*0.25 = 0.75`. The Schur complement is
    `C_11 - C_12^2/C_22 = 2 - 1/4 = 1.75`, so the denominator is `1 + 0.5*1*1.75 = 1.875`, and
    `0.75/1.875 = 0.4` exactly. Ignoring the unmeasured task entirely gives the single-task answer
    `1/(1 + 0.5*2*1) = 0.5`. The presence of the thing you cannot measure takes the weight on the
    thing you can from 0.5 to 0.4, a fifth off, and nothing in current practice does this.
    """
    corr = unmeasurable_correction(
        benefit=[1.0, 1.0],
        cost_curvature=[[2.0, 1.0], [1.0, 4.0]],
        noise_variance=1.0,
        risk_aversion=0.5,
        names=("judge", "the unmeasured half of quality"),
    )
    assert corr.substitution == pytest.approx(0.25)
    assert corr.schur == pytest.approx(1.75)
    assert corr.numerator == pytest.approx(0.75)
    assert corr.denominator == pytest.approx(1.875)
    assert corr.weight == pytest.approx(0.4, rel=1e-14)
    assert corr.weight_ignoring == pytest.approx(0.5, rel=1e-14)
    assert corr.discount == pytest.approx(0.2, rel=1e-12)
    assert not corr.zero_weight and not corr.degenerate
    assert "turn" not in corr.says()
    assert "0.4" in corr.says() and "0.5" in corr.says()


def test_the_zero_weight_theorem_gives_exactly_zero_and_not_something_small():
    """Perfect substitutes plus one unmeasurable task of equal value. Bit-exact, deliberately.

    `C_11 = C_12 = C_22 = 1` makes `C_12/C_22 = 1` and the Schur complement `1 - 1/1 = 0`, both
    exactly in floating point. The numerator is `1 - 1*1 = 0` and the denominator is `1 + r*s*0 = 1`,
    so the optimum is `0.0/1.0`, which is `0.0` and not `1e-17`. The assertion is `== 0.0` because a
    tolerance here would hide the whole content of the theorem, which is that the answer is zero
    rather than small.
    """
    corr = unmeasurable_correction(
        benefit=[1.0, 1.0],
        cost_curvature=[[1.0, 1.0], [1.0, 1.0]],
        noise_variance=2.0,
        risk_aversion=3.0,
    )
    assert corr.substitution == 1.0
    assert corr.schur == 0.0
    assert corr.numerator == 0.0
    assert corr.denominator == 1.0
    assert corr.weight == 0.0
    assert corr.zero_weight is True
    assert corr.degenerate is False
    assert "exactly zero, not small" in corr.says()
    assert "worse than none" in corr.says()


def test_any_positive_power_is_worse_than_none_and_unboundedly_so_at_the_boundary():
    """The other half of the theorem, evaluated rather than asserted.

    At exactly perfect substitutability the agent's effort response is unbounded and the surplus has
    no finite value, so `two_task_surplus` raises there rather than returning one. The statement is
    made along a family approaching the boundary: the optimum goes to zero, the surplus at a fixed
    positive weight goes to minus infinity, and `S(0) = 0` throughout.
    """
    surpluses = []
    optima = []
    for rho in (0.9, 0.99, 0.999, 0.9999):
        corr = unmeasurable_correction(
            benefit=[1.0, 1.0],
            cost_curvature=[[1.0, rho], [rho, 1.0]],
            noise_variance=2.0,
            risk_aversion=3.0,
        )
        optima.append(corr.weight)
        surpluses.append(two_task_surplus(0.1, corr))
        assert two_task_surplus(0.0, corr) == 0.0
        assert two_task_surplus(corr.weight, corr) > 0.0
        assert two_task_surplus(0.1, corr) < 0.0
    assert optima == sorted(optima, reverse=True)
    assert optima[-1] < 1e-4
    assert surpluses == sorted(surpluses, reverse=True)
    assert surpluses[-1] < -20.0

    at_boundary = unmeasurable_correction(
        benefit=[1.0, 1.0],
        cost_curvature=[[1.0, 1.0], [1.0, 1.0]],
        noise_variance=2.0,
        risk_aversion=3.0,
    )
    with pytest.raises(ValueError, match="unbounded"):
        two_task_surplus(0.1, at_boundary)


def test_unequal_values_at_perfect_substitutability_are_a_knife_edge_and_say_so():
    """The case the specification's one sentence collapses. Zero weight needs equal value too."""
    corr = unmeasurable_correction(
        benefit=[2.0, 1.0],
        cost_curvature=[[1.0, 1.0], [1.0, 1.0]],
        noise_variance=2.0,
        risk_aversion=3.0,
    )
    assert corr.weight == pytest.approx(1.0)
    assert not corr.zero_weight
    assert corr.degenerate is True
    assert "no interior optimum" in corr.says()


def test_the_recommendation_reports_the_condition_number_of_the_cost_matrix():
    p = ContractParameters.supplied(
        ("a", "b"),
        benefit=[1.0, 1.0],
        cost_curvature=[[1.0, 1.0 - 1e-14], [1.0 - 1e-14, 1.0]],
        noise=np.eye(2),
        risk_aversion=1.0,
    ).assume_unit_sensitivity()
    reading = recommend_weights(p)
    assert reading.cost_condition_number > 1e12
    assert any("perfect substitutes" in n for n in reading.notes)
    assert "condition number" in reading.render()


# ===========================================================================
# N5: the recommendation as a function of what nobody measured
# ===========================================================================


def test_the_ordering_survives_not_knowing_the_risk_aversion():
    """The claim that survives a sweep over six decades of `r`, which is the reading's real output."""
    p = three_components(benefit=[1.0, 1.0, 1.0], noise=np.diag([0.05, 0.8, 3.0]))
    curve = sweep_weights(p, Sweep.for_risk_aversion(n=41))
    assert curve.weights.shape == (41, 3)
    assert curve.ordering_is_stable
    assert curve.distinct_orderings == 1
    assert curve.dominant_span == 1.0
    assert curve.named_ordering(curve.dominant_ordering) == "unit_tests > format_ok > judge"
    assert "ordering does not" in curve.says()
    # The weights themselves are anything but stable across the same range.
    assert curve.weights[0, 2] / curve.weights[-1, 2] > 100.0


def test_a_crossing_is_found_and_reported_where_the_ordering_is_not_stable():
    """A low-value crisp component against a high-value noisy one: they swap, and the sweep says where.

    With `alpha_i = B_i/(1 + x sigma_i^2)` the two are equal at
    `x = (B_2 - B_1)/(B_1 sigma_2^2 - B_2 sigma_1^2)`, which for `B = (1, 4)` and
    `sigma^2 = (0.1, 4)` is `3/(4 - 0.4) = 0.8333...` exactly.
    """
    p = ContractParameters.supplied(
        ("crisp", "valuable"),
        benefit=[1.0, 4.0],
        cost_curvature=np.eye(2),
        noise=np.diag([0.1, 4.0]),
        risk_aversion=1.0,
    ).assume_unit_sensitivity()
    curve = sweep_weights(p, Sweep.for_risk_aversion(low=1e-2, high=1e2, n=81))
    assert not curve.ordering_is_stable
    assert len(curve.crossings) == 1
    assert curve.crossings[0].at == pytest.approx(3.0 / 3.6, rel=1e-6)
    assert "swap order" in curve.crossings[0].says()
    assert "not stable" in curve.says()


def test_a_sweep_needs_at_least_two_values_and_states_its_own_reason():
    with pytest.raises(ValueError, match="at least two values"):
        Sweep(parameter="risk_aversion", values=[1.0], reason="one value")
    s = Sweep.for_risk_aversion()
    assert s.values.size == 61
    assert "not measurable" in s.reason
    assert "1e-03" in s.render() or "0.001" in s.render()


# ===========================================================================
# N6: the equal-compensation table
# ===========================================================================


def test_the_starved_component_is_the_one_with_the_lowest_commission():
    """The derivation, not the specification's inverted restatement of it.

    `kappa_i = alpha_i mu'_i`. With weights `(0.5, 0.2, 0.3)` and slopes `(1, 0.5, 2)` the
    commissions are `(0.5, 0.1, 0.6)`, so `format_ok` is starved at `0.1/0.6 = 1/6`. The
    specification's rule, lowest `alpha_i / mu'_i`, would name `judge` at `0.15` against
    `unit_tests` at `0.5` and `format_ok` at `0.4`, which is the wrong component.
    """
    p = three_components(sensitivity=np.diag([1.0, 0.5, 2.0]))
    table = equal_compensation([0.5, 0.2, 0.3], p)
    assert [r.commission for r in table.rows] == pytest.approx([0.5, 0.1, 0.6])
    assert table.starved.component == "format_ok"
    assert table.spread == pytest.approx(1.0 / 6.0)
    assert not table.holds
    # The rule the specification prints picks a different component, which is why it is not used.
    ratios_as_printed = np.array([0.5 / 1.0, 0.2 / 0.5, 0.3 / 2.0])
    assert THREE[int(np.argmin(ratios_as_printed))] == "judge"
    assert table.starved.component != "judge"


def test_the_equalising_weights_are_proportional_to_the_inverse_sensitivity():
    p = three_components(sensitivity=np.diag([1.0, 0.5, 2.0]))
    table = equal_compensation([0.5, 0.2, 0.3], p)
    equalising = np.array([r.equalising_weight for r in table.rows])
    inverse = 1.0 / np.array([1.0, 0.5, 2.0])
    assert np.allclose(equalising / equalising.sum(), inverse / inverse.sum())
    assert equalising.sum() == pytest.approx(1.0)
    # And they do equalise: the commissions under them are all the same.
    levelled = equal_compensation(equalising, p)
    assert np.allclose(levelled.commissions, levelled.commissions[0])
    assert levelled.holds
    assert levelled.spread == pytest.approx(1.0)


def test_the_table_is_invariant_under_an_affine_rescaling_of_the_reward():
    """`alpha -> alpha/a` and `mu' -> a mu'`, so every commission and every ratio is unchanged."""
    p = three_components(sensitivity=np.diag([1.0, 0.5, 2.0]))
    alpha = np.array([0.5, 0.2, 0.3])
    base = equal_compensation(alpha, p)
    scaled = equal_compensation(alpha / 4.0, p.rescale(4.0))
    assert np.allclose(base.ratios, scaled.ratios)
    assert np.allclose(base.commissions, scaled.commissions)
    assert base.starved.component == scaled.starved.component


def test_a_component_the_policy_cannot_move_is_a_finding_rather_than_a_row():
    p = three_components(sensitivity=np.diag([1.0, 0.0, 2.0]))
    with pytest.raises(ValueError, match="dose-response slope of exactly zero"):
        equal_compensation([0.5, 0.2, 0.3], p)


def test_a_non_diagonal_sensitivity_makes_the_table_a_summary_and_the_task_returns_the_reading():
    m = np.array([[1.0, 0.3, 0.0], [0.0, 0.5, 0.0], [0.4, 0.0, 2.0]])
    p = three_components(sensitivity=m)
    alpha = np.array([0.5, 0.2, 0.3])
    table = equal_compensation(alpha, p)
    assert not table.diagonal_sensitivity
    assert np.allclose(table.task_returns, m.T @ alpha)
    assert "task returns" in table.render()


# ===========================================================================
# N7: the sorting cutoff
# ===========================================================================


def test_the_effort_equivalent_noise_is_what_makes_two_scales_comparable():
    """A component reported on a scale ten times larger has a hundred times the variance and the
    same `n`, which is the whole reason the sort is on `n` rather than on `sigma^2`."""
    assert noise_to_signal([1.0, 100.0], [1.0, 10.0]) == pytest.approx([1.0, 1.0])
    with pytest.raises(ValueError, match="slope of exactly zero"):
        noise_to_signal([1.0], [0.0])
    assert information_coefficient([2.0, 4.0], [1.0, 2.0]) == pytest.approx([2.0, 2.0])
    with pytest.raises(ValueError, match="non-positive effort"):
        information_coefficient([1.0], [0.0])


def test_the_contract_value_is_the_optimal_weight_surplus_specialised_to_one_component():
    """N7's value function is not ad hoc: it is N5's `g' H^-1 g / 2` with one component in the group."""
    for b, c, mu, s2, r in [(1.0, 2.0, 1.0, 0.5, 1.0), (3.0, 0.5, 2.0, 4.0, 0.25)]:
        p = ContractParameters.supplied(
            ("only",),
            benefit=[b],
            cost_curvature=[[c]],
            noise=[[s2]],
            risk_aversion=r,
            sensitivity=[[mu]],
        )
        n = s2 / (mu * mu)
        assert principal_surplus(optimal_weights(p), p) == pytest.approx(
            contract_value(b, n, r, c), rel=1e-12
        )


def test_splitting_is_neutral_for_identical_components_and_pays_as_soon_as_the_noise_differs():
    """The specification's claim, as a theorem at fixed capacity rather than as a slogan.

    Two components in one contract against two contracts, with the capacity correction `C -> k C`
    that makes a given total effort cost the same either way. For identical components the two are
    exactly equal. For one noiseless component and one with `x = r C n`, the split beats the sum by
    `(1 + x)^2 / (1 + 2x)`, which is `1 + x^2/(1 + 2x)` and is strictly above 1 for any `x > 0`.
    """
    same = ContractParameters.supplied(
        ("a", "b"),
        benefit=[1.0, 1.0],
        cost_curvature=np.eye(2),
        noise=np.diag([0.3, 0.3]),
        risk_aversion=2.0,
        sensitivity=np.eye(2),
    ).assume_equal_effort()
    reading = sorting_cutoff(same, n_contracts=2)
    assert reading.value == pytest.approx(reading.single_contract_value, rel=1e-14)
    assert reading.split_gain == pytest.approx(0.0, abs=1e-14)

    for x in (0.5, 1.0, 4.0):
        p = ContractParameters.supplied(
            ("crisp", "noisy"),
            benefit=[1.0, 1.0],
            cost_curvature=np.eye(2),
            noise=np.diag([0.0, x]),
            risk_aversion=1.0,
            sensitivity=np.eye(2),
        ).assume_equal_effort()
        r2 = sorting_cutoff(p, n_contracts=2)
        assert r2.value / r2.single_contract_value == pytest.approx(
            (1 + x) ** 2 / (1 + 2 * x), rel=1e-12
        )
        assert r2.split_gain > 0.0


def test_the_crisp_component_gets_its_own_contract_and_the_cutoff_falls_between():
    p = three_components(noise=np.diag([0.05, 0.8, 0.5]))
    reading = sorting_cutoff(p, n_contracts=2)
    groups = [set(c.members) for c in reading.contracts]
    assert {"unit_tests"} in groups
    assert {"format_ok", "judge"} in groups
    assert reading.cutoff_bracket == pytest.approx((0.05, 0.5))
    assert reading.cutoff == pytest.approx(math.sqrt(0.05 * 0.5))
    assert reading.best_contract_count == 2
    assert reading.split_gain > 0.3
    assert "Split at rho" in reading.says()


def _random_composite(rng, m: int, *, equal_benefit: bool) -> ContractParameters:
    return ContractParameters.supplied(
        tuple(f"c{i}" for i in range(m)),
        benefit=np.ones(m) if equal_benefit else rng.uniform(0.5, 3.0, m),
        cost_curvature=np.eye(m),
        noise=np.diag(rng.uniform(0.01, 5.0, m)),
        risk_aversion=float(rng.uniform(0.1, 5.0)),
        sensitivity=np.eye(m),
    ).assume_equal_effort()


def test_the_sorting_theorem_holds_on_every_equally_valued_composite_tested():
    """Checked against an exhaustive enumeration rather than assumed.

    With equal benefits the total noise is the same whatever the assignment, so the objective
    depends only on how the total splits, the per-contract value is convex in the group's noise, and
    the optimum is at an extreme of the achievable range. That extreme is reached by an interval
    partition in `rho`, which is why the property holds here exactly.
    """
    rng = np.random.default_rng(3)
    for k in (2, 3):
        for _ in range(40):
            reading = sorting_cutoff(
                _random_composite(rng, int(rng.integers(3, 7)), equal_benefit=True), n_contracts=k
            )
            assert reading.exhaustive_ran
            assert reading.interval_optimal is True, (
                reading.interval_value,
                reading.exhaustive_value,
            )


def test_the_sorting_theorem_fails_on_unequally_valued_composites_and_the_instrument_says_so():
    """The finding this instrument exists to be able to make, rather than an assumption it holds.

    A contract's value depends on the largest `B` among its members as well as on its summed noise,
    and those two do not order together, so a partition that is not an interval in `rho` can win. Of
    200 random instances with benefits drawn from [0.5, 3.0], 79 have a non-interval optimum at two
    contracts and 78 at three, and the reported assignment on those is the better of the two rather
    than the one the sort would have produced.
    """
    rng = np.random.default_rng(3)
    failures = [
        sorting_cutoff(
            _random_composite(rng, int(rng.integers(3, 7)), equal_benefit=False), n_contracts=2
        )
        for _ in range(200)
    ]
    broken = [r for r in failures if r.interval_optimal is False]
    assert 40 <= len(broken) <= 120, len(broken)
    worst = max(broken, key=lambda r: r.exhaustive_value - r.interval_value)
    assert worst.value == pytest.approx(worst.exhaustive_value)
    assert worst.exhaustive_value > worst.interval_value
    assert "does not hold on this instance" in worst.render()
    assert any("not equally valuable" in n for n in worst.notes)


def test_the_enumeration_is_declined_above_its_limit_and_the_reading_records_that():
    p = ContractParameters.supplied(
        tuple(f"c{i}" for i in range(8)),
        benefit=np.ones(8),
        cost_curvature=np.eye(8),
        noise=np.diag(np.linspace(0.1, 4.0, 8)),
        risk_aversion=1.0,
        sensitivity=np.eye(8),
    ).assume_equal_effort()
    reading = sorting_cutoff(p, n_contracts=3, enumeration_limit=10)
    assert not reading.exhaustive_ran
    assert reading.interval_optimal is None
    assert math.isnan(reading.exhaustive_value)
    assert any("enumeration limit" in n for n in reading.notes)
    assert "declined" in reading.render()


def test_the_dilution_factor_is_the_tax_a_noisy_component_levies_on_the_rest():
    p = three_components(noise=np.diag([0.05, 0.8, 0.5]))
    reading = sorting_cutoff(p, n_contracts=2)
    total = 0.05 + 0.8 + 0.5
    expected = [(1 + (total - n)) / (1 + total) for n in (0.05, 0.8, 0.5)]
    assert reading.dilution == pytest.approx(expected)
    assert reading.worst_dilution[0] == "format_ok"


def test_the_sort_is_unchanged_by_a_common_scaling_of_the_effort():
    p = three_components(noise=np.diag([0.05, 0.8, 0.5]))
    a = sorting_cutoff(p.assume_equal_effort(1.0), n_contracts=2)
    b = sorting_cutoff(p.assume_equal_effort(7.0), n_contracts=2)
    assert [set(c.members) for c in a.contracts] == [set(c.members) for c in b.contracts]
    assert b.cutoff == pytest.approx(a.cutoff / 7.0)
    assert a.effort_assumed_equal and b.effort_assumed_equal


# ===========================================================================
# N8: noise and angle
# ===========================================================================


def test_a_contract_pointed_exactly_at_the_value_direction_has_congruity_one():
    m = np.eye(3)
    assert congruity(m, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert congruity(m, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert congruity(m, [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]) == pytest.approx(-1.0)
    assert math.isnan(congruity(m, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))


def test_a_per_component_cosine_has_to_be_read_against_the_clean_measure_reference():
    """`B_i / ||B'||` and not 1, or every component of every three-part composite is distorted.

    With three equally valued tasks a component that responds only to its own task reaches
    `1/sqrt(3) = 0.577`, however good it is. Judged against 1 it looks 42% distorted; judged against
    the reference it is at exactly zero, which is the truth.
    """
    b = np.array([1.0, 1.0, 1.0])
    assert clean_measure_congruity(b) == pytest.approx([1 / math.sqrt(3)] * 3)
    p = three_components(benefit=b, sensitivity=np.diag([1.0, 0.5, 2.0]))
    reading = noise_and_angle(p)
    for row in reading.rows:
        assert row.congruity == pytest.approx(1 / math.sqrt(3))
        assert row.reference == pytest.approx(1 / math.sqrt(3))
        assert row.distortion == pytest.approx(0.0, abs=1e-15)
    assert all(math.isnan(x) for x in clean_measure_congruity([0.0, 0.0, 0.0]))


def test_a_diagonal_sensitivity_has_a_distortion_of_exactly_zero_and_the_reading_says_so():
    b = np.array([1.0, 2.0, 4.0])
    p = three_components(benefit=b, sensitivity=np.diag([1.0, 0.5, 2.0]))
    assert component_congruity(p.sensitivity, b) == pytest.approx(b / np.linalg.norm(b))
    reading = noise_and_angle(p)
    assert reading.diagonal_sensitivity
    assert all(not r.angle_is_informative for r in reading.rows)
    assert all(r.distortion == pytest.approx(0.0, abs=1e-15) for r in reading.rows)
    assert reading.wants_a_different_measure == ()
    assert "exactly zero by construction" in reading.says()
    assert all("noise half only" in r.verdict for r in reading.rows)


def test_distortion_is_a_measurement_only_when_the_sensitivity_has_off_diagonal_structure():
    """One component that moves a task nobody values: precise, distorted, and no weight fixes it.

    `m_judge = (3, 0, 1)` against `B' = (0, 1, 1)` gives `cos = 1/(sqrt(10) sqrt(2)) = 0.2236` while
    a clean own-task measure would reach `B_3/||B'|| = 1/sqrt(2) = 0.7071`, so the distortion is
    0.4835. The other two rows are coordinate axes and sit at exactly their references.
    """
    m = np.eye(3)
    m[2, 0] = 3.0  # the third signal responds mostly to effort on a task valued at zero
    b = np.array([0.0, 1.0, 1.0])
    p = ContractParameters.supplied(
        THREE,
        benefit=b,
        cost_curvature=np.eye(3),
        noise=np.diag([0.05, 0.05, 0.05]),
        risk_aversion=1.0,
        sensitivity=m,
    )
    reading = noise_and_angle(p)
    assert not reading.diagonal_sensitivity
    assert all(r.angle_is_informative for r in reading.rows)
    rows = {r.component: r for r in reading.rows}
    assert rows["judge"].congruity == pytest.approx(1.0 / math.sqrt(20.0))
    assert rows["judge"].reference == pytest.approx(1.0 / math.sqrt(2.0))
    assert rows["judge"].distortion == pytest.approx(0.48345, abs=1e-4)
    assert rows["judge"].precise
    assert reading.wants_a_different_measure == ("judge",)
    assert "replace the measure" in rows["judge"].verdict
    assert rows["format_ok"].verdict.startswith("keep")
    assert rows["unit_tests"].verdict.startswith("keep")


def test_the_noisy_but_congruent_component_wants_a_lower_weight_rather_than_a_new_measure():
    m = np.eye(3)
    m[0, 1] = 0.05  # enough to make M non-diagonal without moving any component off its own task
    p = ContractParameters.supplied(
        THREE,
        benefit=[1.0, 1.0, 1.0],
        cost_curvature=np.eye(3),
        noise=np.diag([0.02, 0.02, 20.0]),
        risk_aversion=1.0,
        sensitivity=m,
    )
    reading = noise_and_angle(p)
    rows = {r.component: r for r in reading.rows}
    assert not rows["judge"].precise
    assert rows["judge"].congruent
    assert rows["judge"].distortion == pytest.approx(0.0, abs=1e-15)
    assert reading.wants_lower_weight == ("judge",)
    assert reading.wants_a_different_measure == ()
    assert "lower the weight" in rows["judge"].verdict
    assert "Lower the weight on judge" in reading.says()


def test_the_pair_is_invariant_under_an_affine_rescaling_of_the_reward():
    m = np.eye(3)
    m[2, 0] = 0.4
    p = ContractParameters.supplied(
        THREE,
        benefit=[1.0, 1.0, 1.0],
        cost_curvature=np.eye(3),
        noise=np.diag([0.05, 0.5, 2.0]),
        risk_aversion=1.0,
        sensitivity=m,
    )
    base = noise_and_angle(p)
    scaled = noise_and_angle(p.rescale(5.0))
    assert [r.noise for r in base.rows] == pytest.approx([r.noise for r in scaled.rows])
    assert [r.congruity for r in base.rows] == pytest.approx([r.congruity for r in scaled.rows])
    assert base.contract_congruity == pytest.approx(scaled.contract_congruity)


# ===========================================================================
# The instruments: gates, refusals, declarations
# ===========================================================================


def test_lint_instrument_is_empty_for_all_four_once_the_proposals_are_registered():
    load_quantities()
    register_proposed()
    for cls in DECISION:
        assert lint_instrument(cls()) == [], cls.__name__


def test_the_only_lint_finding_before_registration_is_the_unregistered_quantity():
    """Which is the lint doing its job: the catalogue carries no N5 to N8 row. E23."""
    load_quantities()
    for cls in DECISION:
        findings = [f for f in lint_instrument(cls()) if f.field != "quantity"]
        assert findings == [], cls.__name__


def test_the_four_instruments_return_evidence_carrying_the_five_assumptions():
    load_quantities()
    register_proposed()
    p = three_components(sensitivity=np.diag([1.0, 0.5, 2.0]))
    ctx = Context(phase=Phase.PRE_RUN)
    readings = [
        OptimalWeights(p).estimate(ctx),
        EqualCompensation([0.5, 0.2, 0.3], p).estimate(ctx),
        SortingCutoff(p).estimate(ctx),
        NoiseAndAngle(p).estimate(ctx),
    ]
    for reading in readings:
        assert isinstance(reading, Evidence), reading
        assert [a["key"] for a in reading.value["assumptions"]] == list(ASSUMPTION_KEYS)
        for key in ASSUMPTION_KEYS:
            assert key in reading.value["assumptions_rendered"]
        assert reading.value["says"]
        assert reading.value["baselines"]
        assert reading.quantity.startswith("reward.")


def test_every_instrument_refuses_as_a_value_rather_than_raising_when_it_has_nothing():
    ctx = Context(phase=Phase.PRE_RUN)
    for cls in DECISION:
        out = cls().estimate(ctx)
        assert isinstance(out, Refusal), cls.__name__
        assert out.reason is RefusalReason.ACCESS_INSUFFICIENT
        assert out.remedy.strip()


def test_measure_raises_when_asked_for_evidence_from_a_measurement_that_declines():
    ctx = Context(phase=Phase.PRE_RUN)
    with pytest.raises(ValueError, match="declines to produce"):
        OptimalWeights().measure(ctx)


def test_the_envelope_refuses_outside_a_stationary_grader():
    """The measurable half of `COMMITMENT_ONE_PERIOD`, enforced rather than only printed."""
    p = three_components()
    drifting = Context(
        phase=Phase.PRE_RUN, regime_reading=RegimeReading.of(STATIONARY_GRADER=False)
    )
    for cls, args in (
        (OptimalWeights, (p,)),
        (SortingCutoff, (p,)),
        (NoiseAndAngle, (p,)),
    ):
        out = cls(*args).estimate(drifting)
        assert isinstance(out, Refusal), cls.__name__
        assert out.reason is RefusalReason.ENVELOPE_VIOLATED
        assert "STATIONARY_GRADER" in out.detail


def test_an_in_run_reading_is_a_phase_mismatch_because_the_model_has_one_period():
    p = three_components()
    out = OptimalWeights(p).estimate(Context(phase=Phase.IN_RUN))
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.PHASE_MISMATCH


def test_a_missing_weight_sweep_on_a_record_is_record_incomplete_and_not_access_insufficient():
    """E30's test: whether the remedy is answerable where the reader is standing.

    With a record named, it is not. The sweep was never run, nothing in the record recovers a
    response to a change nobody made, and the fix is upstream in whatever produced the run. With no
    record named the caller is being asked for a parameter it can simply state, and that is
    `ACCESS_INSUFFICIENT`.
    """
    bare = ContractParameters.supplied(
        THREE,
        benefit=[1.0, 1.0, 1.0],
        cost_curvature=np.eye(3),
        noise=np.eye(3),
        risk_aversion=1.0,
    )
    ctx = Context(phase=Phase.PRE_RUN)
    from_record = EqualCompensation(
        [0.5, 0.2, 0.3], bare, record="the 24-step GRPO run, 512 rollouts"
    ).estimate(ctx)
    assert isinstance(from_record, Refusal)
    assert from_record.reason is RefusalReason.RECORD_INCOMPLETE
    assert "the 24-step GRPO run" in from_record.detail
    assert "re-run with a weight sweep" in from_record.remedy
    assert "more access" not in from_record.remedy

    from_nowhere = EqualCompensation([0.5, 0.2, 0.3], bare).estimate(ctx)
    assert isinstance(from_nowhere, Refusal)
    assert from_nowhere.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "sensitivity" in from_nowhere.detail


def test_the_sort_refuses_on_a_composite_with_one_component():
    p = ContractParameters.supplied(
        ("only",),
        benefit=[1.0],
        cost_curvature=[[1.0]],
        noise=[[1.0]],
        risk_aversion=1.0,
        sensitivity=[[1.0]],
    ).assume_equal_effort()
    out = SortingCutoff(p).estimate(Context(phase=Phase.PRE_RUN))
    assert isinstance(out, Refusal)
    assert "nothing to sort" in out.detail
    assert "N5 answers" in out.remedy


def test_a_sweep_over_the_risk_aversion_removes_the_need_to_state_it():
    """The third option in the rule: supplied, refused, or reported as a function of the unknown."""
    p = ContractParameters.supplied(
        THREE,
        benefit=[1.0, 1.0, 1.0],
        cost_curvature=np.eye(3),
        noise=np.diag([0.05, 0.8, 3.0]),
        risk_aversion=1.0,
        sensitivity=np.eye(3),
        source={"risk_aversion": ParameterSource.UNKNOWN},
    )
    ctx = Context(phase=Phase.PRE_RUN)
    assert isinstance(OptimalWeights(p).estimate(ctx), Refusal)
    with_sweep = OptimalWeights(p, sweep=Sweep.for_risk_aversion(n=21)).estimate(ctx)
    assert isinstance(with_sweep, Evidence)
    assert with_sweep.value["sensitivity"]["ordering_is_stable"] is True
    assert with_sweep.value["sensitivity"]["parameter"] == "risk_aversion"


def test_the_weight_vector_is_covariant_with_weight_minus_one_under_reward_affine():
    """The generated property test, on the group action this instrument lives under.

    A weight is a price per unit of signal, so rescaling every component's score by `a` must divide
    every weight by `a`. The payload is the component score matrix; `Sigma` is its covariance and
    `M` its per-component standard deviation, so the affine action carries `(Sigma, M)` to
    `(a^2 Sigma, a M)` exactly, which is the action `ContractParameters.rescale` implements.
    """
    rng = np.random.default_rng(0)
    scores = rng.normal(0.0, [1.0, 2.0, 0.5], size=(500, 3))

    def run(_inst, payload: InvariancePayload) -> float:
        x = np.asarray(payload.scores, dtype=np.float64)
        sd = x.std(axis=0, ddof=1)
        p = ContractParameters.supplied(
            THREE,
            benefit=[1.0, 1.0, 1.0],
            cost_curvature=np.eye(3),
            noise=np.cov(x, rowvar=False),
            risk_aversion=1.0,
            sensitivity=np.diag(sd),
        )
        return float(optimal_weights(p)[0])

    report = check_invariance(
        OptimalWeights(),
        "reward.affine",
        InvariancePayload(scores=scores),
        n=24,
        relation=COVARIANT_INVERSE,
        run=run,
    )
    assert report.passed, report.render()
    assert COVARIANT_INVERSE.weight == -1.0


def test_the_other_three_readings_are_exactly_invariant_under_the_same_action():
    """Five of the six quantities in this layer are gauge free, and that is a structural claim."""
    m = np.eye(3)
    m[2, 0] = 0.4
    p = ContractParameters.supplied(
        THREE,
        benefit=[1.0, 1.0, 1.0],
        cost_curvature=np.eye(3),
        noise=np.diag([0.05, 0.5, 2.0]),
        risk_aversion=1.0,
        sensitivity=m,
    ).assume_equal_effort()
    scaled = p.rescale(6.0)
    alpha = np.array([0.5, 0.2, 0.3])

    assert np.allclose(
        equal_compensation(alpha, p).ratios, equal_compensation(alpha / 6.0, scaled).ratios
    )
    a, b = sorting_cutoff(p, n_contracts=2), sorting_cutoff(scaled, n_contracts=2)
    assert np.allclose(a.noise, b.noise)
    assert np.allclose(a.rho, b.rho)
    assert a.cutoff == pytest.approx(b.cutoff)
    ga, gb = noise_and_angle(p), noise_and_angle(scaled)
    assert [r.congruity for r in ga.rows] == pytest.approx([r.congruity for r in gb.rows])


# ===========================================================================
# The proposed catalogue rows
# ===========================================================================


def test_six_quantities_are_proposed_and_none_of_them_is_registered():
    load_quantities()
    ids = {q.id for q in PROPOSED}
    assert ids == {
        "reward.optimal_weights",
        "reward.equal_compensation_ratio",
        "reward.information_coefficient",
        "reward.sorting_cutoff",
        "reward.component_congruity",
        "reward.component_noise",
    }
    assert set(PROPOSED_INSTRUMENTS) == ids
    assert set(PROPOSED_MIN_ACCESS) == ids
    for q in PROPOSED:
        assert q.definition and q.interpretation
        assert q.invariance == "reward.affine"
        assert q.wedge
        assert q.unit.is_decided


def test_every_unit_token_the_proposals_use_already_appears_in_the_registry():
    """A new quantity may need a new row; it should not need a new dimension."""
    load_quantities()
    known_dimensions = {q.unit.dimension for q in QUANTITIES.values()}
    known_scales = {q.unit.scale for q in QUANTITIES.values()}
    known_pers = {q.unit.per for q in QUANTITIES.values()}
    for q in PROPOSED:
        assert q.unit.dimension in known_dimensions, q.id
        assert q.unit.scale in known_scales, q.id
        assert q.unit.per in known_pers, q.id


def test_the_proposed_rows_and_records_parse_as_yaml_and_name_only_proposed_quantities():
    yaml = pytest.importorskip("yaml")
    rows = yaml.safe_load(as_yaml_rows())
    assert [r["id"] for r in rows] == [q.id for q in PROPOSED]
    for row, q in zip(rows, PROPOSED):
        assert row["invariance_group"] == q.invariance
        assert row["min_access"] == PROPOSED_MIN_ACCESS[q.id]
        assert row["instrument"] == list(PROPOSED_INSTRUMENTS[q.id])
    records = yaml.safe_load(as_catalogue_rows())
    assert [r["id"] for r in records] == ["N5", "N6", "N7", "N8"]
    proposed = {q.id for q in PROPOSED}
    for record in records:
        assert set(record["quantities"]) <= proposed
        assert record["status"] == "planned"
        assert record["envelope_requires"] == ["STATIONARY_GRADER"]
        assert record["kill_condition"]
        assert len(record["baselines"]) == 2


def test_each_instrument_declares_the_quantities_its_catalogue_record_claims():
    by_id = {r["id"]: r for r in CATALOGUE_RECORDS}
    for cls in DECISION:
        record = by_id[cls.faithful_to]
        declared = {cls.quantity, *getattr(cls, "also_reports", ())}
        assert declared == set(record["quantities"]), cls.__name__
        assert tuple(cls.baselines) and len(cls.baselines) == len(record["baselines"])
        assert cls.deviations

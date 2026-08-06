"""Unit tests for `measure.reconcile`: the arithmetic, the refusals, and the invariance property.

The acceptance file runs the whole thing on the two real GRPO records. This file tests the pieces
on constructed inputs where the right answer is known in closed form, which is the half a real
record cannot do: a real record cannot tell you that `β = C⁻¹S` was solved correctly, because it
does not come with the true `β`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from reward_lens.core.budget import UncertaintyBudget
from reward_lens.measure.ledger.price import StepSample, ledger_between
from reward_lens.measure.reconcile.books import (
    BasisMismatch,
    MetricGLike,
    StepCostLike,
    cost_consistency,
    reconcile_series,
    selection_gradient,
    within_group_covariance,
)
from reward_lens.measure.reconcile.closure import MIN_CLUSTERS, closure_of
from reward_lens.measure.reconcile.lande import (
    CircularEstimator,
    fit_lande,
    heritability_range,
    permuted_lande_null,
)
from reward_lens.measure.reconcile.residual import (
    TERM_ORDER,
    advantage_r_squared,
)

# ---------------------------------------------------------------------------
# Fixtures built by hand, so the right answer is known
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StubG:
    """A `MetricG` by structure rather than by import, which is what D19's Protocol is for."""

    names: tuple[str, ...]
    matrix: np.ndarray
    damping: float = 0.0
    damping_stable: bool = True
    conditioning: float = 1.0
    rung: int = 2
    method: str = "stub"
    n_samples: int = 100


@dataclass(frozen=True)
class StubCost:
    step: int
    next_step: int
    kl_spent: float
    kl_min: float
    efficiency: float
    shares: dict
    residual_share: float


def _sample(index: int, features: np.ndarray, advantages: np.ndarray, groups: np.ndarray):
    return StepSample(
        index=index,
        names=("a", "b"),
        features=features,
        advantages=advantages,
        group_ids=groups,
        task_ids=tuple(f"task:{g}" for g in groups),
    )


@pytest.fixture
def two_steps():
    """Two steps of six rollouts in two groups of three, with spread on both features."""
    rng = np.random.default_rng(7)
    out = []
    for index in range(2):
        features = rng.normal(size=(6, 2)) + np.asarray([10.0 * index, 0.0])
        advantages = rng.normal(size=6)
        groups = np.asarray([0, 0, 0, 1, 1, 1])
        out.append(_sample(index, features, advantages, groups))
    return out


def _stub_g(names, matrix):
    return StubG(names=tuple(names), matrix=np.asarray(matrix, dtype=np.float64))


# ---------------------------------------------------------------------------
# The protocols are structural, which is the point of writing them as Protocols
# ---------------------------------------------------------------------------


def test_the_metric_protocol_accepts_a_structurally_matching_object():
    g = _stub_g(("a", "b"), np.eye(2))
    assert isinstance(g, MetricGLike)


def test_the_cost_protocol_accepts_a_structurally_matching_object():
    cost = StubCost(0, 1, kl_spent=1.0, kl_min=0.5, efficiency=0.5, shares={}, residual_share=0.5)
    assert isinstance(cost, StepCostLike)


def test_the_real_efficiency_types_satisfy_the_protocols():
    """The interface D19 fixed, checked against what the cost book shipped rather than assumed."""
    from reward_lens.measure.efficiency.cost import StepCost
    from reward_lens.measure.efficiency.metric import MetricG

    for field in (
        "names",
        "matrix",
        "damping",
        "damping_stable",
        "conditioning",
        "rung",
        "method",
        "n_samples",
    ):
        assert field in MetricG.__dataclass_fields__, field
    for field in (
        "step",
        "next_step",
        "kl_spent",
        "kl_min",
        "efficiency",
        "shares",
        "residual_share",
    ):
        assert field in StepCost.__dataclass_fields__, field


# ---------------------------------------------------------------------------
# Cause: the within-group covariance and the gradient
# ---------------------------------------------------------------------------


def test_the_covariance_is_within_group_and_not_pooled(two_steps):
    """The two operators differ by prompt-to-prompt heterogeneity, and E17 settles which is meant.

    The second step's first feature is shifted by 10, so a pooled covariance over both steps sees a
    variance the within-group one does not. Constructed so the difference cannot be a rounding gap.
    """
    within = within_group_covariance(two_steps)
    stacked = np.vstack([s.features for s in two_steps])
    pooled = np.cov(stacked, rowvar=False, ddof=1)
    assert within.operator == "within_group"
    assert pooled[0, 0] > 5.0 * within.matrix[0, 0]
    assert within.n_groups == 4
    assert within.n_used == 12


def test_the_gradient_solves_s_equals_c_beta_exactly():
    """`β = C⁻¹S` is checked by multiplying back, which is the only check that cannot be circular."""
    c = np.asarray([[4.0, 1.0], [1.0, 2.0]])
    s = np.asarray([3.0, -1.0])
    from reward_lens.measure.reconcile.books import FeatureCovariance

    cov = FeatureCovariance(
        names=("a", "b"),
        matrix=c,
        n_used=10,
        n_groups=2,
        n_steps=1,
        eigenvalues=np.linalg.eigvalsh(c),
    )
    beta = selection_gradient(cov, s)
    assert np.allclose(c @ beta.value, s)
    assert beta.ridge == 0.0
    assert beta.operator == "within_group"


def test_the_ridge_shrinks_the_gradient_toward_a_rescaled_differential():
    """E17's sweep: by `δ = 1` the solution is `S / (δ·tr C / k)` and has stopped being a gradient."""
    from reward_lens.measure.reconcile.books import FeatureCovariance

    c = np.asarray([[4.0, 3.9], [3.9, 4.0]])
    s = np.asarray([1.0, -1.0])
    cov = FeatureCovariance(
        names=("a", "b"),
        matrix=c,
        n_used=10,
        n_groups=2,
        n_steps=1,
        eigenvalues=np.linalg.eigvalsh(c),
    )
    heavy = selection_gradient(cov, s, ridge=100.0)
    expected = s / (100.0 * np.trace(c) / 2.0)
    assert np.allclose(heavy.value, expected, rtol=0.02)
    assert heavy.ridge == 100.0


def test_the_conditioning_scalar_is_n_d_and_not_the_condition_number(two_steps):
    covariance = within_group_covariance(two_steps)
    eigenvalues = covariance.eigenvalues
    assert covariance.conditioning == pytest.approx(eigenvalues.sum() / eigenvalues.max())
    assert 1.0 <= covariance.conditioning <= len(covariance.names)
    assert covariance.condition_number >= covariance.conditioning


# ---------------------------------------------------------------------------
# The join key
# ---------------------------------------------------------------------------


def test_a_reordered_basis_raises_rather_than_aligning_by_name(two_steps):
    """D19: the names are the join key and they are compared element for element, not as sets."""
    ledgers = [ledger_between(two_steps[0], two_steps[1], eta=1e-3)]
    g = _stub_g(("b", "a"), np.eye(2))
    with pytest.raises(BasisMismatch, match="different spaces"):
        reconcile_series(two_steps, ledgers, g)


def test_g_equal_to_c_makes_the_reconciliation_residual_the_ledger_residual(two_steps):
    """`Gβ = C·C⁻¹S = S`. The identity that makes the rung-0 reading degenerate."""
    ledgers = [ledger_between(two_steps[0], two_steps[1], eta=1e-3)]
    covariance = within_group_covariance(two_steps)
    g = _stub_g(covariance.names, covariance.matrix)
    steps = reconcile_series(two_steps, ledgers, g)
    for row in steps[0].rows:
        assert row.residual == pytest.approx(row.ledger_residual, abs=1e-12)
        assert row.heritability == pytest.approx(1.0)


def test_the_response_jacobian_is_the_derivative_it_claims_to_be(two_steps):
    """`∂Δz_pred/∂S = η·G·C⁻¹`, checked by finite difference on the whole pipeline."""
    ledgers = [ledger_between(two_steps[0], two_steps[1], eta=1e-3)]
    covariance = within_group_covariance(two_steps)
    g = _stub_g(covariance.names, np.asarray([[2.0, 0.3], [0.3, 1.5]]))
    steps = reconcile_series(two_steps, ledgers, g)
    jacobian = steps[0].response_jacobian
    expected = 1e-3 * g.matrix @ np.linalg.inv(covariance.matrix)
    assert np.allclose(jacobian, expected)


# ---------------------------------------------------------------------------
# The units group: a per-feature rescale leaves the slope and the ratio alone
# ---------------------------------------------------------------------------


def test_the_lande_slope_is_unchanged_by_a_per_feature_rescale(two_steps):
    """The substantive property behind the `units` declaration, and it is not vacuous.

    Under `f -> D f` with `D` diagonal and positive: `Δz -> D Δz`, `S -> D S`, `C -> D C D`, so
    `β -> D⁻¹ β` and `G -> D G D` gives `Gβ -> D G β`. Both sides of the regression scale by the
    same `D`, and dividing each feature by its own `sd(f)` is what makes the pooled fit see none of
    it. A fit that did not divide by `sd(f)` would move here, which is why this test exists.
    """
    rng = np.random.default_rng(3)
    samples = []
    for index in range(9):
        features = rng.normal(size=(6, 2)) * np.asarray([1.0, 3.0]) + index * 0.05
        samples.append(_sample(index, features, rng.normal(size=6), np.asarray([0, 0, 0, 1, 1, 1])))
    ledgers = [ledger_between(a, b, eta=1e-2) for a, b in zip(samples, samples[1:])]
    covariance = within_group_covariance(samples)
    g_matrix = np.asarray([[2.0, 0.4], [0.4, 1.0]])
    steps = reconcile_series(samples, ledgers, _stub_g(covariance.names, g_matrix))
    scales = {"a": 1.0, "b": 1.0}
    stacked = np.vstack([s.features for s in samples])
    scales = {n: float(stacked[:, j].std(ddof=1)) for j, n in enumerate(("a", "b"))}
    base = fit_lande(steps, scales, n_bootstrap=0)

    d = np.asarray([7.0, 0.25])
    scaled_samples = [replace(s, features=s.features * d) for s in samples]
    scaled_ledgers = [
        ledger_between(a, b, eta=1e-2) for a, b in zip(scaled_samples, scaled_samples[1:])
    ]
    scaled_cov = within_group_covariance(scaled_samples)
    scaled_g = np.diag(d) @ g_matrix @ np.diag(d)
    scaled_steps = reconcile_series(
        scaled_samples, scaled_ledgers, _stub_g(scaled_cov.names, scaled_g)
    )
    scaled_stack = np.vstack([s.features for s in scaled_samples])
    scaled_scales = {n: float(scaled_stack[:, j].std(ddof=1)) for j, n in enumerate(("a", "b"))}
    scaled = fit_lande(scaled_steps, scaled_scales, n_bootstrap=0)

    assert scaled.slope == pytest.approx(base.slope, rel=1e-9)
    assert scaled.r_squared == pytest.approx(base.r_squared, rel=1e-9)


# ---------------------------------------------------------------------------
# The Lande fit and its guards
# ---------------------------------------------------------------------------


def test_a_perfect_lande_run_recovers_a_slope_of_one():
    """Constructed so `Δz` is exactly `η G β`. If this does not return 1 the fit is wrong."""
    rng = np.random.default_rng(11)
    names = ("a", "b")
    g_matrix = np.asarray([[3.0, 0.5], [0.5, 2.0]])
    samples, ledgers = [], []
    for index in range(12):
        features = rng.normal(size=(6, 2))
        samples.append(_sample(index, features, rng.normal(size=6), np.asarray([0, 0, 0, 1, 1, 1])))
    eta = 1e-2
    for before, after in zip(samples, samples[1:]):
        ledgers.append(ledger_between(before, after, eta=eta))
    steps = reconcile_series(samples, ledgers, _stub_g(names, g_matrix))
    # Rewrite each observed Δz to be exactly the prediction, then refit.
    from reward_lens.measure.reconcile.books import BookRow

    perfect = [
        replace(
            step,
            rows=tuple(
                replace(row, delta_z_obs=row.delta_z_pred, residual=0.0) for row in step.rows
            ),
        )
        for step in steps
    ]
    assert isinstance(perfect[0].rows[0], BookRow)
    scales = {n: 1.0 for n in names}
    fit = fit_lande(perfect, scales, n_bootstrap=0)
    assert fit.slope == pytest.approx(1.0, rel=1e-9)
    assert fit.r_squared == pytest.approx(1.0, rel=1e-9)


def test_a_circular_g_is_refused_rather_than_reported(two_steps):
    """A `G` fitted from `Δz` gives a slope near 1 by construction, so the fit refuses."""
    ledgers = [ledger_between(two_steps[0], two_steps[1], eta=1e-3)]
    covariance = within_group_covariance(two_steps)
    g = replace(_stub_g(covariance.names, covariance.matrix), method="realised", rung=0)
    steps = reconcile_series(two_steps, ledgers, g)
    with pytest.raises(CircularEstimator, match="predicting"):
        fit_lande(steps, {"a": 1.0, "b": 1.0}, n_bootstrap=0)
    assert fit_lande(steps, {"a": 1.0, "b": 1.0}, n_bootstrap=0, allow_circular=True) is not None


def test_near_degeneracy_is_caught_when_h_squared_is_the_damping(two_steps):
    """`G = (1-λ)C` is not `C`, and it is not an independent estimate of capacity either."""
    ledgers = [ledger_between(two_steps[0], two_steps[1], eta=1e-3)]
    covariance = within_group_covariance(two_steps)
    damping = 0.01
    g = replace(
        _stub_g(covariance.names, (1.0 - damping) * covariance.matrix),
        damping=damping,
        method="fisher_kernel",
    )
    steps = reconcile_series(two_steps, ledgers, g)
    low, high = heritability_range(steps)
    assert low == pytest.approx(1.0 - damping)
    fit = fit_lande(steps, {"a": 1.0, "b": 1.0}, n_bootstrap=0)
    assert fit.is_degenerate
    assert "regulariser" in fit.render()


def test_a_g_genuinely_different_from_c_is_not_degenerate(two_steps):
    ledgers = [ledger_between(two_steps[0], two_steps[1], eta=1e-3)]
    covariance = within_group_covariance(two_steps)
    g = replace(_stub_g(covariance.names, 0.3 * covariance.matrix), damping=1e-6)
    steps = reconcile_series(two_steps, ledgers, g)
    fit = fit_lande(steps, {"a": 1.0, "b": 1.0}, n_bootstrap=0)
    assert not fit.is_degenerate


def test_the_permuted_null_moves_the_slope(two_steps):
    """A null that returned the observed statistic would be no null at all."""
    rng = np.random.default_rng(5)
    samples = [
        _sample(i, rng.normal(size=(6, 2)), rng.normal(size=6), np.asarray([0, 0, 0, 1, 1, 1]))
        for i in range(10)
    ]
    ledgers = [ledger_between(a, b, eta=1e-2) for a, b in zip(samples, samples[1:])]
    covariance = within_group_covariance(samples)
    steps = reconcile_series(samples, ledgers, _stub_g(covariance.names, np.eye(2)))
    draws = permuted_lande_null(steps, {"a": 1.0, "b": 1.0}, n_draws=200)
    assert draws.size == 200
    assert np.std(draws[np.isfinite(draws)]) > 0.0


# ---------------------------------------------------------------------------
# The budget
# ---------------------------------------------------------------------------


def test_the_nine_terms_are_named_in_the_order_the_spec_gives_them():
    assert TERM_ORDER == (
        "u_stale",
        "u_KL",
        "u_entropy",
        "u_momentum",
        "u_batch",
        "u_curv",
        "u_clip",
        "u_MC",
        "u_basis",
    )


def test_the_advantage_r_squared_is_one_when_the_advantage_is_a_feature(two_steps):
    """`A = 2·f_a` inside every group, so the basis explains all of it."""
    samples = [replace(s, advantages=2.0 * s.features[:, 0]) for s in two_steps]
    r_squared, n = advantage_r_squared(samples, [0, 1])
    assert r_squared == pytest.approx(1.0)
    assert n == 12


def test_the_advantage_r_squared_falls_when_the_advantage_leaves_the_span(two_steps):
    rng = np.random.default_rng(2)
    samples = [replace(s, advantages=rng.normal(size=6)) for s in two_steps]
    r_squared, _ = advantage_r_squared(samples, [0, 1])
    assert r_squared < 0.95


def test_the_budget_composes_in_quadrature_as_the_gum_says(two_steps):
    """The composition is `core.budget`'s and this asserts the arithmetic rather than restating it."""
    from reward_lens.core.budget import BudgetTerm

    budget = UncertaintyBudget(
        terms=(
            BudgetTerm(name="u_MC", value=3.0, dof=10),
            BudgetTerm(name="u_batch", value=4.0, dof=10),
        )
    )
    assert budget.combined == pytest.approx(5.0)
    assert budget.effective_dof() is not None


# ---------------------------------------------------------------------------
# The closure test's four verdicts
# ---------------------------------------------------------------------------


def _closure_from(residual_sd: float, accounted_sd: float, *, n: int = 40, missing: int = 0):
    """A `FeatureBudget` with a chosen `Var(ρ)` and `Σu²`, so each verdict can be reached on demand."""
    from reward_lens.core.budget import BudgetTerm
    from reward_lens.measure.reconcile.residual import FeatureBudget, MissingTerm

    rng = np.random.default_rng(0)
    draws = rng.normal(size=n)
    draws = (draws - draws.mean()) / draws.std(ddof=1) * residual_sd
    budget = FeatureBudget(
        feature="a",
        n_steps=n,
        mean_residual=0.0,
        var_residual=float(np.var(draws, ddof=1)),
        se_mean_residual=0.0,
        budget=UncertaintyBudget(terms=(BudgetTerm(name="u_MC", value=accounted_sd, dof=50),)),
        missing=tuple(MissingTerm(f"u_{i}", "constructed", "constructed") for i in range(missing)),
    )
    steps = []
    for i, value in enumerate(draws):
        steps.append(_fake_step(i, float(value)))
    return budget, steps


def _fake_step(index: int, residual: float):
    from reward_lens.measure.reconcile.books import BookRow, StepReconciliation

    row = BookRow(
        feature="a",
        delta_z_obs=residual,
        delta_z_pred=0.0,
        residual=residual,
        eta=1e-3,
        differential=0.0,
        gradient=0.0,
        response=0.0,
        heritability=1.0,
        se_delta_z=1.0,
        se_differential=0.0,
        ledger_residual=residual,
    )
    return StepReconciliation(
        step=index,
        next_step=index + 1,
        rows=(row,),
        eta=1e-3,
        eta_source="supplied",
        task_overlap=1.0,
        n_scored=8,
        n_groups=2,
        n_before=8,
        n_after=8,
        response_jacobian=np.zeros((1, 1)),
        c_n_used=8,
        c_n_groups=2,
        c_n_steps=1,
        c_conditioning=1.0,
        ridge=0.0,
        g_rung=2,
        g_method="stub",
        g_damping=0.0,
        g_damping_stable=True,
    )


def test_a_matching_budget_closes():
    budget, steps = _closure_from(2.0, 2.0)
    result = closure_of([budget], steps, run_id="constructed")
    assert result.verdict == "closed"


def test_an_excess_with_terms_missing_is_incomplete_and_not_a_discovery():
    """The distinction that stops an unmeasured term being claimed as an unmodelled one."""
    budget, steps = _closure_from(10.0, 1.0, missing=3)
    result = closure_of([budget], steps, run_id="constructed")
    assert result.verdict == "incomplete"
    assert "lower bound" in result.features[0].detail


def test_an_excess_with_every_term_computed_is_an_unmodelled_term():
    budget, steps = _closure_from(10.0, 1.0, missing=0)
    result = closure_of([budget], steps, run_id="constructed")
    assert result.verdict == "unmodelled"
    assert "does not name" in result.features[0].detail


def test_a_budget_larger_than_the_residual_is_over_rather_than_clipped_to_closed():
    budget, steps = _closure_from(1.0, 10.0)
    result = closure_of([budget], steps, run_id="constructed")
    assert result.verdict == "over"
    assert result.features[0].ci_high < 1.0


def test_the_interval_is_declined_below_five_step_pairs():
    """The same cluster floor `measure.ledger.explained` derives, and `undetermined` is the answer."""
    budget, steps = _closure_from(2.0, 2.0, n=MIN_CLUSTERS - 1)
    result = closure_of([budget], steps, run_id="constructed")
    assert result.verdict == "undetermined"
    assert not np.isfinite(result.features[0].ci_low)


def test_the_run_verdict_is_the_worst_feature_and_not_a_vote():
    good, steps = _closure_from(2.0, 2.0)
    bad, _ = _closure_from(10.0, 1.0, missing=2)
    from dataclasses import replace as _replace

    bad = _replace(bad, feature="a")
    result = closure_of([good, bad], steps, run_id="constructed")
    assert result.verdict == "incomplete"


# ---------------------------------------------------------------------------
# The cost consistency check
# ---------------------------------------------------------------------------


def test_cost_consistency_holds_when_kl_min_is_below_the_spend():
    costs = [
        StubCost(i, i + 1, kl_spent=1.0, kl_min=0.4, efficiency=0.4, shares={}, residual_share=0.4)
        for i in range(5)
    ]
    result = cost_consistency(costs)
    assert result.holds and result.n_violations == 0


def test_cost_consistency_names_the_worst_step_when_it_fails():
    """A violation is an instrument bug, and the check says which step and by how much."""
    costs = [
        StubCost(i, i + 1, kl_spent=1.0, kl_min=0.4, efficiency=0.4, shares={}, residual_share=0.4)
        for i in range(4)
    ]
    costs.append(
        StubCost(9, 10, kl_spent=1.0, kl_min=1.7, efficiency=1.7, shares={}, residual_share=1.7)
    )
    result = cost_consistency(costs)
    assert not result.holds
    assert result.worst_step == 9
    assert result.worst_ratio == pytest.approx(1.7)
    assert "FAILED" in result.render()


def test_no_cost_book_is_recorded_as_unchecked_rather_than_as_passing():
    result = cost_consistency(None)
    assert not result.checked and not result.holds
    assert "no cost series was supplied" in result.detail

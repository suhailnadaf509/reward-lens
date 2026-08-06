"""Unit tests for `measure.efficiency`: the metric, the allocation, and the refusals.

The clause is in `tests/acceptance/test_w4_4_efficiency.py`. This file covers the parts
that clause does not reach: what each of the three `G` estimators does and declines to do, the
properties of the Shapley game, the pseudo-inverse path, and the two invariance groups F3 declares.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.invariance import (
    INVARIANT,
    GroupAction,
    InvarianceGroup,
    InvariancePayload,
    check_invariance,
    check_unit_refusal,
)
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.efficiency import (
    MAX_EXACT_FEATURES,
    MetricG,
    StepCost,
    cost_series,
    kl_min_series,
    kl_spent_from_record,
    metric_g,
    noise_floor,
    pooled_rollouts,
    shapley_shares,
)
from reward_lens.measure.efficiency.metric import _kernel_g, _within_group_centre
from reward_lens.measure.estimator.units import EstimatorQuantity, difference
from reward_lens.measure.ledger.price import StepSample, ledger_between

NAMES = ("length", "words", "ratio")


def _sample(index: int, features: np.ndarray, *, groups: np.ndarray | None = None) -> StepSample:
    n = features.shape[0]
    labels = groups if groups is not None else np.repeat([0, 1], n // 2)[:n]
    return StepSample(
        index=index,
        names=NAMES[: features.shape[1]],
        features=np.asarray(features, dtype=np.float64),
        advantages=np.linspace(-1.0, 1.0, n),
        group_ids=np.asarray(labels, dtype=np.int64),
        task_ids=tuple(f"task{int(g)}" for g in labels),
    )


def _random_window(seed: int = 0, steps: int = 4, per_step: int = 8) -> list[StepSample]:
    rng = np.random.default_rng(seed)
    return [
        _sample(
            i,
            rng.normal(size=(per_step, 3))
            @ np.array([[2.0, 0.5, 0.0], [0.0, 1.0, 0.3], [0.0, 0.0, 1.0]]),
        )
        for i in range(steps)
    ]


def _metric(matrix: np.ndarray, names: tuple[str, ...] = NAMES) -> MetricG:
    return MetricG(
        names=names,
        matrix=np.asarray(matrix, dtype=np.float64),
        damping=0.0,
        damping_stable=True,
        conditioning=1.0,
        rung=0,
        method="test",
        n_samples=0,
    )


# ---------------------------------------------------------------------------
# MetricG validates what it is
# ---------------------------------------------------------------------------


def test_a_metric_whose_shape_disagrees_with_its_basis_does_not_construct():
    with pytest.raises(ValueError, match="cannot be joined"):
        _metric(np.eye(2))


def test_an_asymmetric_matrix_does_not_construct():
    with pytest.raises(ValueError, match="not symmetric"):
        _metric(np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))


def test_a_negative_eigenvalue_does_not_construct():
    """`G` is PSD by construction, so an indefinite one would make `KL_min` negative for some `Δz`."""
    with pytest.raises(ValueError, match="negative eigenvalue"):
        _metric(np.diag([1.0, 1.0, -0.5]))


def test_the_pseudo_inverse_reports_movement_it_cannot_explain():
    """A `Δz` with mass in the null space of `G` is movement no parameter step could produce."""
    g = _metric(np.diag([1.0, 1.0, 0.0]))
    value, out_of_range = g.kl_min(np.array([1.0, 0.0, 1.0]))
    # Half the squared norm of Δz is in the null direction.
    assert out_of_range == pytest.approx(0.5, rel=1e-12)
    # And KL_min is computed on the reachable half alone: 1/2 * 1^2 / 1 = 0.5.
    assert value == pytest.approx(0.5, rel=1e-12)


def test_rank_and_conditioning_read_the_spectrum():
    g = _metric(np.diag([4.0, 1.0, 0.0]))
    assert g.rank == 2
    values, vectors = g.eigen()
    assert np.allclose(vectors @ np.diag(values) @ vectors.T, g.matrix, atol=1e-12)


# ---------------------------------------------------------------------------
# The three estimators
# ---------------------------------------------------------------------------


def test_the_covariance_bound_is_exactly_the_within_group_covariance():
    """`G = C` at rung 0, which is the bound taken at equality."""
    window = _random_window()
    g = metric_g(window)
    assert not isinstance(g, Refusal)
    assert g.rung == 0
    assert g.method == "covariance_bound"
    assert g.covariance is not None
    assert np.allclose(g.matrix, g.covariance, atol=1e-12)
    assert all(h == pytest.approx(1.0, rel=1e-12) for h in g.heritability().values())


def test_the_fisher_kernel_stays_under_the_covariance_by_construction():
    """`K(K + mλI)⁻¹` has eigenvalues in [0,1), so `Ĝ ⪯ Ĉ` holds in finite samples."""
    window = _random_window(seed=3, steps=1)
    rng = np.random.default_rng(7)
    scores = rng.normal(size=(window[0].n, 200))
    g = metric_g(window, method="fisher_kernel", scores=scores, damping=1e-1)
    assert not isinstance(g, Refusal), g
    assert g.rung == 2
    assert g.damping > 0.0
    gap = np.linalg.eigvalsh(g.covariance - g.matrix)
    assert gap.min() >= -1e-9
    for value in g.heritability().values():
        assert 0.0 <= value <= 1.0


def test_the_undamped_fisher_kernel_collapses_onto_the_covariance():
    """The degeneracy the module docstring names: at `n` below `p`, `G = C` exactly at λ→0.

    This is the reason `damping` is a reported field rather than a hidden parameter. With fewer
    rollouts than parameters the feature matrix lies inside the row space of the score matrix, the
    shrinkage operator is the identity on it, and `h² = 1` for every feature no matter what is true.
    """
    window = _random_window(seed=5, steps=1)
    rng = np.random.default_rng(11)
    scores = rng.normal(size=(window[0].n, 5000))
    centred_features = _within_group_centre(window[0].features, window[0].group_ids)
    centred_scores = _within_group_centre(scores, window[0].group_ids)
    dof = window[0].n - 2
    covariance = centred_features.T @ centred_features / dof

    # Stated as the limit it is. Driving lambda to zero outright makes `(K + m lambda I)` singular
    # to working precision and the inverse stops meaning anything, so the claim is checked as a
    # trend across four decades: the gap to `C` falls by roughly a decade per decade of lambda.
    live = np.linalg.eigvalsh(centred_scores @ centred_scores.T)
    scale = float(live[live > 1e-6 * live.max()].min()) / dof
    gaps = [
        float(
            np.linalg.norm(
                _kernel_g(centred_features, centred_scores, dof, scale * f) - covariance, "fro"
            )
        )
        for f in (1e-2, 1e-4, 1e-6, 1e-8)
    ]
    assert gaps == sorted(gaps, reverse=True)
    assert gaps[-1] / float(np.linalg.norm(covariance, "fro")) < 1e-6


def test_the_fisher_kernel_refuses_without_scores():
    window = _random_window(steps=1)
    out = metric_g(window, method="fisher_kernel")
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "covariance_bound" in out.remedy


def test_a_misaligned_score_matrix_raises_rather_than_pairing_the_wrong_rows():
    window = _random_window(steps=1)
    with pytest.raises(ValueError, match="row order"):
        metric_g(window, method="fisher_kernel", scores=np.zeros((3, 10)))


def test_the_realised_estimator_refuses_when_the_fit_is_not_a_metric():
    """A least-squares `G` fitted to noise comes back indefinite, and an indefinite `G` is refused."""
    window = _random_window(seed=2, steps=12)
    ledgers = [ledger_between(before, after, eta=1e-6) for before, after in zip(window, window[1:])]
    out = metric_g(window, method="realised", ledgers=ledgers)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "not positive semi-definite" in out.detail
    assert "covariance_bound" in out.remedy


def test_the_realised_estimator_refuses_without_ledgers():
    out = metric_g(_random_window(), method="realised")
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE


def test_an_unknown_method_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="unknown method"):
        metric_g(_random_window(), method="whatever")


def test_an_empty_window_refuses():
    out = metric_g([])
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE


def test_groups_of_one_leave_no_within_group_spread_and_refuse():
    sample = _sample(0, np.arange(6.0).reshape(3, 2), groups=np.array([0, 1, 2]))
    out = metric_g([sample])
    assert isinstance(out, Refusal)
    assert out.statistics["n_groups"] == 3


def test_a_window_that_changes_basis_raises():
    a = _sample(0, np.zeros((4, 3)))
    b = StepSample(
        index=1,
        names=("length", "words"),
        features=np.zeros((4, 2)),
        advantages=np.zeros(4),
        group_ids=np.zeros(4, dtype=np.int64),
        task_ids=("t",) * 4,
    )
    with pytest.raises(ValueError, match="changes basis"):
        metric_g([a, b])


def test_pooling_relabels_groups_so_two_steps_are_not_centred_against_each_other():
    """Step 5's first prompt group and step 6's first are different prompts."""
    window = _random_window(steps=3, per_step=8)
    _features, labels, n_groups = pooled_rollouts(window)
    assert n_groups == 6
    assert sorted(set(labels.tolist())) == list(range(6))


# ---------------------------------------------------------------------------
# The allocation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_shapley_shares_are_non_negative_and_sum_to_the_attributed_total(seed):
    """The game is monotone, so every marginal contribution is non-negative and so is every share."""
    rng = np.random.default_rng(seed)
    root = rng.normal(size=(3, 3))
    g = _metric(root @ root.T + np.eye(3) * 0.5)
    dz = rng.normal(size=3)
    shares, attributed, kl_min = shapley_shares(g, dz)
    assert all(v >= -1e-12 for v in shares.values())
    assert sum(shares.values()) == pytest.approx(attributed, rel=1e-9)
    assert attributed <= kl_min + 1e-9
    assert kl_min == pytest.approx(0.5 * dz @ np.linalg.inv(g.matrix) @ dz, rel=1e-9)


def test_the_eigen_decomposition_is_exactly_additive():
    """The decomposition, which needs no allocation rule because its directions have no names."""
    rng = np.random.default_rng(1)
    root = rng.normal(size=(3, 3))
    g = _metric(root @ root.T + np.eye(3))
    dz = rng.normal(size=3)
    value, _ = g.kl_min(dz)
    assert g.eigen_shares(dz).sum() == pytest.approx(value, rel=1e-12)


def test_an_inert_feature_keeps_its_key_at_zero_rather_than_being_dropped():
    """The mapping's keys are the join key, so two cost books never disagree about their key set."""
    g = _metric(np.diag([1.0, 1.0, 0.0]))
    shares, _attributed, _kl = shapley_shares(g, np.array([1.0, 1.0, 0.0]))
    assert set(shares) == set(NAMES)
    assert shares["ratio"] == 0.0


def test_attributing_to_a_name_outside_the_basis_raises():
    g = _metric(np.eye(3))
    with pytest.raises(ValueError, match="join key"):
        shapley_shares(g, np.ones(3), attribute_to=["nonexistent"])


def test_too_many_named_features_refuses_rather_than_silently_sampling():
    """An exact share for eleven features and a Monte Carlo one for thirteen are two quantities."""
    k = MAX_EXACT_FEATURES + 1
    names = tuple(f"f{i}" for i in range(k))
    g = MetricG(
        names=names,
        matrix=np.eye(k),
        damping=0.0,
        damping_stable=True,
        conditioning=1.0,
        rung=0,
        method="test",
        n_samples=0,
    )
    before = StepSample(
        index=0,
        names=names,
        features=np.zeros((2, k)),
        advantages=np.zeros(2),
        group_ids=np.zeros(2, dtype=np.int64),
        task_ids=("t", "t"),
    )
    after = StepSample(
        index=1,
        names=names,
        features=np.ones((2, k)),
        advantages=np.zeros(2),
        group_ids=np.zeros(2, dtype=np.int64),
        task_ids=("t", "t"),
    )
    out = kl_min_series([ledger_between(before, after, eta=1.0)], g)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.BUDGET_EXCEEDED
    assert "attribute_to" in out.remedy


def test_the_noise_floor_is_the_expected_kl_min_of_a_step_that_did_nothing():
    """`½ tr(G⁻¹ Σ)`, which for a diagonal pair is one term per feature and checkable by hand."""
    g = _metric(np.diag([2.0, 4.0, 1.0]))
    se = np.array([1.0, 2.0, 0.0])
    # 1/2 * (1^2/2 + 2^2/4 + 0) = 1/2 * (0.5 + 1.0) = 0.75
    assert noise_floor(g, se) == pytest.approx(0.75, rel=1e-12)


def test_a_step_whose_movement_is_inside_its_noise_says_so():
    """The reading names it rather than leaving a reader to compare two columns."""
    g = _metric(np.diag([1.0, 1.0, 1.0]))
    before = StepSample(
        index=0,
        names=NAMES,
        features=np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
        advantages=np.zeros(2),
        group_ids=np.zeros(2, dtype=np.int64),
        task_ids=("t", "t"),
    )
    after = StepSample(
        index=1,
        names=NAMES,
        features=np.array([[0.0, 0.0, 0.0], [2.1, 2.0, 2.0]]),
        advantages=np.zeros(2),
        group_ids=np.zeros(2, dtype=np.int64),
        task_ids=("t", "t"),
    )
    rows = kl_min_series([ledger_between(before, after, eta=1.0)], g)
    assert not isinstance(rows, Refusal)
    assert not rows[0].above_noise
    assert any("noise floor" in n for n in rows[0].notes)


# ---------------------------------------------------------------------------
# Where the denominator comes from
# ---------------------------------------------------------------------------


def test_kl_spent_from_a_record_that_has_none_is_an_empty_mapping(tmp_path):
    """Empty is a normal return here, and it is what `cost_series` turns into a refusal."""
    from pathlib import Path

    from reward_lens.record.reader import open_run

    root = Path(__file__).resolve().parent / "fixtures" / "grpo_run" / "long"
    if not (root / "runs").exists():
        pytest.skip("no GRPO record on disk")
    run_id = next(p.name for p in (root / "runs").iterdir()).replace("run_", "run:")
    assert kl_spent_from_record(open_run(root, run_id)) == {}


def test_cost_series_without_any_source_of_a_denominator_refuses():
    g = _metric(np.eye(3))
    before = StepSample(
        index=0,
        names=NAMES,
        features=np.zeros((2, 3)),
        advantages=np.zeros(2),
        group_ids=np.zeros(2, dtype=np.int64),
        task_ids=("t", "t"),
    )
    after = StepSample(
        index=1,
        names=NAMES,
        features=np.ones((2, 3)),
        advantages=np.zeros(2),
        group_ids=np.zeros(2, dtype=np.int64),
        task_ids=("t", "t"),
    )
    out = cost_series([ledger_between(before, after, eta=1.0)], g)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert out.statistics["source"] == "nothing"


def test_from_kl_min_is_the_only_way_a_step_cost_gets_a_denominator():
    g = _metric(np.diag([1.0, 1.0, 1.0]))
    before = StepSample(
        index=0,
        names=NAMES,
        features=np.zeros((2, 3)),
        advantages=np.zeros(2),
        group_ids=np.zeros(2, dtype=np.int64),
        task_ids=("t", "t"),
    )
    after = StepSample(
        index=1,
        names=NAMES,
        features=np.tile([1.0, 0.0, 0.0], (2, 1)),
        advantages=np.zeros(2),
        group_ids=np.zeros(2, dtype=np.int64),
        task_ids=("t", "t"),
    )
    rows = kl_min_series([ledger_between(before, after, eta=1.0)], g)
    assert not isinstance(rows, Refusal)
    cost = StepCost.from_kl_min(rows[0], kl_spent=1.0, note="from a test")
    assert cost.kl_min == pytest.approx(0.5, rel=1e-12)
    assert cost.efficiency == pytest.approx(0.5, rel=1e-12)
    assert "from a test" in cost.notes


# ---------------------------------------------------------------------------
# The two invariance groups F3 declares
# ---------------------------------------------------------------------------


def _reparam_on_scores(seed: int, scale: float) -> GroupAction:
    """`policy.reparam` in the coordinates this instrument actually reads.

    The group's generator acts on a flat parameter vector as `θ → Jθ`. `G` is not a function of
    `θ`: it is a function of the score matrix `S` and the feature matrix, and what a smooth
    invertible reparameterisation `θ' = Aθ` does to a score is `s → A⁻ᵀs`, which on the `(n, |θ|)`
    matrix of row covectors is `S → S A⁻¹`. Applying `θ → Jθ` to a flattened score matrix instead
    would mix one rollout's gradient into another's, which is not a change of coordinates on the
    parameter space at all. So the generator here carries the flattened score matrix in
    `parameters` and applies the transform the mathematics asks for. The group id is unchanged;
    only the concrete action is supplied, which `check_invariance` takes as an argument.
    """

    def apply(payload: InvariancePayload) -> InvariancePayload:
        payload.require("parameters")
        flat = np.asarray(payload.parameters, dtype=np.float64)
        n = int(payload.extra["n"])
        scores = flat.reshape(n, -1)
        width = scores.shape[1]
        rng = np.random.default_rng(seed)
        a = np.eye(width) + scale * rng.standard_normal((width, width)) / np.sqrt(width)
        return payload.replace(parameters=(scores @ np.linalg.inv(a)).ravel())

    return GroupAction(
        name=f"θ → Aθ, ‖A−I‖~{scale:.3g} (acting on the score matrix as S → S A⁻¹)",
        apply=apply,
        params={"seed": float(seed), "scale": scale},
        sample=lambda rng: _reparam_on_scores(rng.randrange(2**31), rng.uniform(1e-3, 3e-2)),
    )


POLICY_REPARAM = InvarianceGroup(
    id="policy.reparam",
    generators=(_reparam_on_scores(0, 1e-2),),
    acts_on="parameters",
    admits=frozenset({"invariant", "raw_only"}),
    assertion="Fisher-metric quantities (KL, G, h², efficiency) unchanged",
)


def test_kl_min_is_invariant_under_a_reparameterisation_of_the_policy():
    """`J → J A⁻¹` and `F → A⁻ᵀ F A⁻¹`, so `G = J F⁻¹ Jᵀ` and everything from it is unchanged.

    Checked at a damping small enough that the shrinkage is effectively a projection, because the
    projection onto the row space of `S` is what is exactly invariant: `range(S A⁻¹) = range(S)`
    for any invertible `A`. A working damping breaks it, and by how much is the next test.
    """
    window = _random_window(seed=4, steps=1, per_step=8)
    rng = np.random.default_rng(21)
    scores = rng.normal(size=(window[0].n, 40))
    dz = np.array([0.4, -0.2, 0.1])

    def read(payload: InvariancePayload) -> float:
        n = int(payload.extra["n"])
        g = metric_g(
            window,
            method="fisher_kernel",
            scores=np.asarray(payload.parameters).reshape(n, -1),
            damping=1e-9,
        )
        assert not isinstance(g, Refusal), g
        return g.kl_min(dz)[0]

    payload = InvariancePayload(parameters=scores.ravel(), extra={"n": window[0].n})
    report = check_invariance(
        read,
        POLICY_REPARAM,
        payload,
        n=16,
        relation=INVARIANT,
        run=lambda f, p: f(p),
        tol=1e-6,
    )
    assert report.passed, report.render()


def test_the_damping_is_what_breaks_that_invariance_and_the_reading_says_so():
    """`(F + λI)⁻¹` is not reparameterisation-covariant, which is why `damping` is a reported field."""
    window = _random_window(seed=4, steps=1, per_step=8)
    rng = np.random.default_rng(21)
    scores = rng.normal(size=(window[0].n, 40))
    dz = np.array([0.4, -0.2, 0.1])

    def read(payload: InvariancePayload) -> float:
        n = int(payload.extra["n"])
        g = metric_g(
            window,
            method="fisher_kernel",
            scores=np.asarray(payload.parameters).reshape(n, -1),
            damping=1e-1,
        )
        assert not isinstance(g, Refusal), g
        return g.kl_min(dz)[0]

    payload = InvariancePayload(parameters=scores.ravel(), extra={"n": window[0].n})
    report = check_invariance(
        read,
        POLICY_REPARAM,
        payload,
        n=16,
        relation=INVARIANT,
        run=lambda f, p: f(p),
        tol=1e-12,
    )
    assert not report.passed
    assert report.max_deviation > 1e-12


def test_the_units_group_assertion_is_a_refusal_and_it_fires():
    """`update.kl_min` is nats per sequence and `update.efficiency` is dimensionless."""
    assert check_unit_refusal(
        difference,
        EstimatorQuantity("update.kl_min", 0.5),
        EstimatorQuantity("update.efficiency", 0.5),
    )

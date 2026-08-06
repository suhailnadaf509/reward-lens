"""Unit and property tests for the Price ledger and the selection-explained fraction.

Four kinds of test are required before an instrument merges and all four are here: hand-computed
values, the generated invariance test, hypothesis property tests, and refusal tests that assert the
reason **and** the remedy string. The clause lives in
`tests/acceptance/test_w4_3_ledger.py`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from reward_lens.core.invariance import (
    INVARIANT,
    InvariancePayload,
    check_invariance,
    check_unit_refusal,
)
from reward_lens.core.quantity import QUANTITIES
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Access, Component, Phase, Substrate
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.estimator.units import EstimatorQuantity, difference
from reward_lens.measure.ledger import (
    EffectiveStepSize,
    SelectionExplainedFraction,
    SelectionResidual,
    SelectionTerm,
    StepSample,
    SurfaceFeatures,
    advantages_from_rewards,
    feature_scales,
    fit_lambda,
    ledger_between,
    ledger_series,
    selection_differential,
)
from reward_lens.measure.ledger.features import surface_features
from reward_lens.measure.ledger.nulls import (
    permuted_advantage_null,
    permuted_step_null,
    random_feature_null,
    summarise,
)
from reward_lens.measure.ledger.prediction import (
    LAMBDA_LEAD_TIME_SPEC,
    freeze_prediction,
    onset_of,
    score_lead_time,
    transition_window,
)
from reward_lens.record.schema import Group, GroupStats, Step, make_trajectory
from reward_lens.record.turns import Turn

ACCESS = {Component.RECORD: Access.RECORD}


# ---------------------------------------------------------------------------
# Fixtures: small samples whose answers can be worked out by hand
# ---------------------------------------------------------------------------


def sample(
    index: int,
    features: list[list[float]],
    advantages: list[float],
    groups: list[int],
    tasks: list[str] | None = None,
    names: tuple[str, ...] = ("f",),
) -> StepSample:
    return StepSample(
        index=index,
        names=names,
        features=np.asarray(features, dtype=np.float64),
        advantages=np.asarray(advantages, dtype=np.float64),
        group_ids=np.asarray(groups, dtype=np.int64),
        task_ids=tuple(tasks or [f"t{g}" for g in groups]),
    )


# ---------------------------------------------------------------------------
# 1. Hand-computed values
# ---------------------------------------------------------------------------


def test_the_within_group_covariance_matches_the_number_worked_out_by_hand():
    """One group of four. `S = sum (A - Abar)(f - fbar) / (n - G)` with n = 4 and G = 1.

    A = (-1.5, -0.5, 0.5, 1.5) is already centred; f = (10, 20, 30, 40) centres to
    (-15, -5, 5, 15). The cross-products are 22.5, 2.5, 2.5, 22.5, summing to 50, over 4 - 1 = 3.
    """
    s = sample(0, [[10.0], [20.0], [30.0], [40.0]], [-1.5, -0.5, 0.5, 1.5], [0, 0, 0, 0])
    d = selection_differential(s.features, s.advantages, s.group_ids, s.names)
    assert d.value[0] == pytest.approx(50.0 / 3.0)
    assert d.n_scored == 4 and d.n_groups == 1
    assert d.operator == "within_group"
    # One group gives no between-group spread, so the clustered standard error is declined.
    assert math.isnan(d.standard_error[0])


def test_the_pooled_denominator_is_n_minus_g_not_n_minus_one():
    """Two identical groups of four. Each contributes 50; the denominator is 8 - 2 = 6.

    This is the operator E17 settles on: within-group, pooled. Centring inside each
    group costs one degree of freedom per group, and using `n - 1` here would understate the
    covariance by 6/7 on this data.
    """
    s = sample(
        0,
        [[10.0], [20.0], [30.0], [40.0]] * 2,
        [-1.5, -0.5, 0.5, 1.5] * 2,
        [0, 0, 0, 0, 1, 1, 1, 1],
    )
    d = selection_differential(s.features, s.advantages, s.group_ids, s.names)
    assert d.value[0] == pytest.approx(100.0 / 6.0)
    assert d.n_groups == 2
    # Two identical groups disagree by nothing, so the clustered standard error is exactly zero.
    assert d.standard_error[0] == pytest.approx(0.0)


def test_the_two_operators_agree_on_the_advantage_and_disagree_on_the_raw_reward():
    """Why relocating `χ` from `r` to `A` matters more than the operator choice does.

    E17 settles `C` as the within-group covariance and measures the two operators reordering five
    of seven features on a real bank. That measurement was taken against the **raw reward**, and
    this test shows why the size of the gap depends on which multiplier is used.

    Two groups, feature levels 100 apart, rewards 100 apart, and the same within-group pattern in
    both. Against the **raw reward** the pooled operator picks up the whole between-group level
    term and returns 3336.7 where the within-group operator returns 5.0, a factor of 667. Against
    the **advantage** the two differ only by the degrees of freedom, 10.0 against 6.67, because a
    group-centred advantage has a group mean of exactly zero and the between-group term
    `Cov(mean_g f, mean_g A)` vanishes by construction. Relocating the multiplier from `r` to `A`
    is what removes the confound; fixing the operator is what keeps the identity consistent once it
    is removed.
    """
    features = np.asarray([[10.0], [20.0], [110.0], [120.0]])
    groups = np.asarray([0, 0, 1, 1])
    rewards = np.asarray([1.0, 2.0, 101.0, 102.0])
    advantages = advantages_from_rewards(rewards, groups, std_epsilon=0.0)

    within_a = selection_differential(features, advantages, groups, ("f",)).value[0]
    pooled_a = float(np.cov(features[:, 0], advantages, ddof=1)[0, 1])
    assert within_a == pytest.approx(10.0)
    assert pooled_a == pytest.approx(20.0 / 3.0)
    assert pooled_a > 0.6 * within_a

    within_r = selection_differential(features, rewards, groups, ("f",)).value[0]
    pooled_r = float(np.cov(features[:, 0], rewards, ddof=1)[0, 1])
    assert within_r == pytest.approx(5.0)
    assert pooled_r == pytest.approx(10010.0 / 3.0)
    assert pooled_r > 600 * within_r


def test_a_group_of_one_contributes_nothing_and_is_counted():
    s = sample(0, [[1.0], [2.0], [3.0]], [-0.5, 0.5, 0.0], [0, 0, 1])
    d = selection_differential(s.features, s.advantages, s.group_ids, s.names)
    assert d.n_groups == 1 and d.n_degenerate == 1
    assert d.value[0] == pytest.approx(0.5)


def test_an_abstained_rollout_carries_nan_and_leaves_the_covariance_alone():
    """A NaN advantage drops out of the covariance and the surviving pair is re-centred.

    This is the case that makes re-centring the advantage non-vacuous: with the third rollout
    masked, the surviving advantages are (-1.5, -0.5) and no longer sum to zero, so an uncentred
    product would pick up the group's residual mean advantage times the feature mean.
    """
    s = sample(0, [[10.0], [20.0], [30.0]], [-1.5, -0.5, float("nan")], [0, 0, 0])
    d = selection_differential(s.features, s.advantages, s.group_ids, s.names)
    # (A - Abar) = (-0.5, +0.5); (f - fbar) = (-5, +5); sum 5.0 over (2 - 1).
    assert d.value[0] == pytest.approx(5.0)
    assert d.n_scored == 2


def test_the_ledger_row_is_the_identity_and_the_residual_closes_it():
    before = sample(0, [[10.0], [20.0], [30.0], [40.0]], [-1.5, -0.5, 0.5, 1.5], [0, 0, 0, 0])
    after = sample(1, [[12.0], [22.0], [32.0], [42.0]], [-1.5, -0.5, 0.5, 1.5], [0, 0, 0, 0])
    led = ledger_between(before, after, eta=0.06)
    row = led.row("f")
    assert row.delta_z == pytest.approx(2.0)
    assert row.covariance == pytest.approx(50.0 / 3.0)
    assert row.selection == pytest.approx(1.0)
    assert row.residual == pytest.approx(1.0)
    assert row.delta_z == pytest.approx(row.selection + row.residual)
    assert row.selection_share == pytest.approx(0.5)
    assert row.residual_dominates is False


def test_lambda_is_exactly_one_when_the_identity_holds_exactly():
    """A planted series where `Δz = η·Cov` at every step. The uncentred R-squared is 1 and the
    slope is the planted `η`."""
    rng = np.random.default_rng(0)
    eta = 0.25
    samples, ledgers = [], []
    for t in range(12):
        f = rng.normal(0.0, 1.0, (8, 1)) + t
        a = np.asarray([-1.5, -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5])
        g = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        samples.append(StepSample(t, ("f",), f, a, g, tuple("t%d" % x for x in g)))
    for before, after in zip(samples, samples[1:]):
        cov = selection_differential(
            before.features, before.advantages, before.group_ids, before.names
        ).value[0]
        moved = np.full_like(after.features, eta * cov) + before.features
        after = StepSample(
            after.index, after.names, moved, after.advantages, after.group_ids, after.task_ids
        )
        ledgers.append(ledger_between(before, after, eta=eta))
    scales = feature_scales(samples)
    fit = fit_lambda(ledgers, scales, n_bootstrap=0)
    assert fit is not None
    assert fit.lambda_ == pytest.approx(1.0)
    assert fit.eta_eff == pytest.approx(eta)


def test_lambda_is_bounded_in_zero_one_by_construction():
    """Cauchy-Schwarz, not clipping. The through-origin uncentred R-squared cannot leave [0, 1]."""
    rng = np.random.default_rng(3)
    ledgers, samples = _random_series(rng, n_steps=15, k=4)
    fit = fit_lambda(ledgers, feature_scales(samples), n_bootstrap=0)
    assert fit is not None
    assert 0.0 <= fit.lambda_ <= 1.0
    assert all(0.0 <= v <= 1.0 for v in fit.by_feature.values())


def _random_series(rng, n_steps=15, k=3, n=8):
    samples = []
    for t in range(n_steps):
        f = rng.normal(0.0, 1.0, (n, k))
        a = np.tile(np.asarray([-1.5, -0.5, 0.5, 1.5]), n // 4)
        g = np.repeat(np.arange(n // 4), 4)[:n]
        samples.append(
            StepSample(t, tuple(f"f{i}" for i in range(k)), f, a, g, tuple(f"t{x}" for x in g))
        )
    return ledger_series(samples, eta=1.0), samples


def test_the_advantage_reconstruction_reproduces_the_z_score_the_trainer_computed():
    rewards = [1.0, 2.0, 3.0, 4.0]
    got = advantages_from_rewards(rewards, [0, 0, 0, 0], std_epsilon=0.0)
    want = (np.asarray(rewards) - 2.5) / np.asarray(rewards).std()
    assert got == pytest.approx(want)


def test_a_null_reward_reconstructs_to_nan_and_never_to_zero():
    got = advantages_from_rewards([1.0, np.nan, 3.0], [0, 0, 0], std_epsilon=0.0)
    assert math.isnan(got[1])
    assert np.isfinite(got[0]) and np.isfinite(got[2])


def test_an_all_fail_group_reconstructs_to_zero_advantages_which_is_what_the_trainer_did():
    got = advantages_from_rewards([0.0, 0.0, 0.0, 0.0], [0] * 4, std_epsilon=1e-4)
    assert got == pytest.approx(np.zeros(4))


def test_the_surface_features_are_computed_the_same_way_from_text_and_from_a_record():
    text = "alpha beta beta gamma"
    direct = surface_features(text, 2)
    turns = (
        Turn(index=0, role="user", text="prompt"),
        Turn(index=1, role="assistant", text=text),
    )
    trajectory = make_trajectory(id="traj:1", task_ref="task:1", turns=turns, advantage=0.0)
    through_record = SurfaceFeatures().featurise(trajectory)
    assert direct == through_record
    assert direct["response_words"] == 4.0
    assert direct["type_token_ratio"] == pytest.approx(0.75)
    assert direct["mean_word_length"] == pytest.approx(18 / 4)


def test_a_rollout_with_no_assistant_text_is_dropped_rather_than_scored_zero():
    turns = (Turn(index=0, role="user", text="prompt"),)
    trajectory = make_trajectory(id="traj:1", task_ref="task:1", turns=turns, advantage=0.0)
    assert SurfaceFeatures().featurise(trajectory) is None
    assert SurfaceFeatures(on_empty="zero").featurise(trajectory)["response_chars"] == 0.0


# ---------------------------------------------------------------------------
# 2. The generated invariance test
# ---------------------------------------------------------------------------


def _differential_under(payload: InvariancePayload) -> float:
    """`Cov_group(A, f)` recomputed from the payload's scores, which is what the group acts on.

    The advantage has to be recomputed rather than read, or the transform would not reach it: an
    advantage already written into a record does not move when the reward is rescaled, so a test
    that transformed the reward would assert nothing about the estimator.
    """
    features = np.asarray(payload.extra["features"], dtype=np.float64)
    groups = np.asarray(payload.group_ids)
    advantages = advantages_from_rewards(
        payload.scores, groups, std_epsilon=payload.extra["std_epsilon"]
    )
    return float(selection_differential(features, advantages, groups, ("f",)).value[0])


def _affine_payload(std_epsilon: float) -> InvariancePayload:
    rng = np.random.default_rng(11)
    return InvariancePayload(
        scores=rng.normal(3.0, 2.0, 16),
        group_ids=np.repeat(np.arange(4), 4),
        extra={"features": rng.normal(0.0, 1.0, (16, 1)), "std_epsilon": std_epsilon},
    )


def test_the_selection_differential_is_invariant_under_an_affine_rescaling_of_the_reward():
    """`Cov(f, A)` does not move under `r -> a·r + b`, which `Cov(f, r)` does. That is the
    substantive difference between the ledger and the shipped `chi`.

    The tolerance is set from the estimator's own epsilon rather than from a constant. At
    `std_epsilon = 0` the invariance is exact; at a positive epsilon the numerator scales by `a`
    while the denominator goes to `a·std + eps` rather than `a·(std + eps)`, which is the caveat
    E13 measured at 1e-7 for `eps = 1e-8`.
    """
    report = check_invariance(
        _differential_under,
        "reward.affine",
        _affine_payload(0.0),
        n=64,
        relation=INVARIANT,
        run=lambda f, p: f(p),
        tol=1e-9,
    )
    assert report.passed, report.render()


def test_the_epsilon_in_the_denominator_is_what_makes_that_invariance_approximate():
    """At the epsilon these records carry, the violation is measurable and it is named."""
    report = check_invariance(
        _differential_under,
        "reward.affine",
        _affine_payload(1e-4),
        n=64,
        relation=INVARIANT,
        run=lambda f, p: f(p),
        tol=1e-12,
    )
    assert not report.passed
    assert report.max_deviation > 1e-12
    assert report.max_deviation < 1e-2


def test_lambda_declares_the_units_group_and_its_assertion_is_a_refusal():
    """The `units` group's generated assertion is not a numeric relation, so `check_invariance`
    routes it to `check_unit_refusal` and the real test is that a comparison across a unit boundary
    refuses. `selection.explained_fraction` is dimensionless and a logprob gap is per token."""
    for instrument in (_lambda_instrument(), _eta_instrument()):
        report = check_invariance(
            instrument, "units", InvariancePayload(), relation=INVARIANT, run=lambda i, p: 1.0
        )
        assert report.passed and "refusal" in report.skipped

    assert check_unit_refusal(
        difference,
        EstimatorQuantity("selection.explained_fraction", 0.62),
        EstimatorQuantity("policy.train_infer_logprob_mismatch", 0.4),
    )
    out = difference(
        EstimatorQuantity("selection.explained_fraction", 0.62),
        EstimatorQuantity("policy.train_infer_logprob_mismatch", 0.4),
    )
    assert isinstance(out, Refusal) and out.reason is RefusalReason.UNIT_MISMATCH


def test_lambda_and_eta_eff_are_unchanged_by_a_per_feature_rescale_of_the_features():
    """The non-vacuous check the `units` group cannot generate, written out.

    Every feature entering the fit divided by its own pooled standard deviation is what makes the
    pooled `Λ` a statement about behaviour rather than about the units the converter recorded in.
    Rescaling one feature by 1000 must therefore change nothing.
    """
    rng = np.random.default_rng(5)
    ledgers, samples = _random_series(rng, n_steps=20, k=3)
    base = fit_lambda(ledgers, feature_scales(samples), n_bootstrap=0)

    factors = np.asarray([1000.0, 0.001, 7.0])
    scaled = [
        StepSample(s.index, s.names, s.features * factors, s.advantages, s.group_ids, s.task_ids)
        for s in samples
    ]
    rescaled = fit_lambda(ledger_series(scaled, eta=1.0), feature_scales(scaled), n_bootstrap=0)
    assert rescaled.lambda_ == pytest.approx(base.lambda_)
    assert rescaled.eta_eff == pytest.approx(base.eta_eff)


# ---------------------------------------------------------------------------
# 3. Property tests
# ---------------------------------------------------------------------------


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    values=st.lists(
        st.floats(min_value=-100.0, max_value=100.0, allow_nan=False), min_size=8, max_size=8
    ),
    scale=st.floats(min_value=0.01, max_value=100.0, allow_nan=False),
    shift=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False),
)
def test_the_differential_scales_linearly_in_the_feature_and_ignores_a_feature_shift(
    values, scale, shift
):
    """`Cov(A, c·f + d) = c·Cov(A, f)`. Bilinearity in one argument, which is what makes the
    per-feature rescaling in the Lambda fit legitimate."""
    features = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    advantages = np.tile(np.asarray([-1.5, -0.5, 0.5, 1.5]), 2)
    groups = np.repeat(np.arange(2), 4)
    base = selection_differential(features, advantages, groups, ("f",)).value[0]
    moved = selection_differential(features * scale + shift, advantages, groups, ("f",)).value[0]
    assert moved == pytest.approx(scale * base, abs=1e-8, rel=1e-8)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    rewards=st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False), min_size=8, max_size=8
    )
)
def test_a_reconstructed_advantage_sums_to_zero_within_every_group(rewards):
    groups = np.repeat(np.arange(2), 4)
    a = advantages_from_rewards(rewards, groups, std_epsilon=1e-8)
    for label in (0, 1):
        block = a[groups == label]
        assert float(np.nansum(block)) == pytest.approx(0.0, abs=1e-9)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=500))
def test_lambda_stays_in_the_unit_interval_on_arbitrary_series(seed):
    rng = np.random.default_rng(seed)
    ledgers, samples = _random_series(rng, n_steps=8, k=3)
    fit = fit_lambda(ledgers, feature_scales(samples), n_bootstrap=0)
    if fit is None:
        return
    assert 0.0 <= fit.lambda_ <= 1.0 or math.isnan(fit.lambda_)


# ---------------------------------------------------------------------------
# 4. Refusals: the reason and the remedy, not just that something came back
# ---------------------------------------------------------------------------


def _run_of(steps):
    from reward_lens.record.schema import InMemoryStepStream, RegimeDeclaration, Run

    return Run(
        id="run:test",
        kind="training",
        components={},
        access=dict(ACCESS),
        regime=RegimeDeclaration(),
        steps=InMemoryStepStream(steps),
    )


def _step(index: int, *, text: str | None = "alpha beta gamma", lr: float | None = 1e-6) -> Step:
    """One step of four rollouts. ``text=None`` gives assistant turns with nothing to featurise."""
    trajectories = [
        make_trajectory(
            id=f"traj:{index}:{i}",
            task_ref=f"task:{index}",
            turns=(
                Turn(index=0, role="user", text="q"),
                Turn(index=1, role="assistant", text="" if text is None else f"{text} {'x' * i}"),
            ),
            advantage=float(i) - 1.5,
        )
        for i in range(4)
    ]
    from reward_lens.record.schema import EstimatorSpec

    group = Group(
        id=f"grp:{index}",
        task_ref=f"task:{index}",
        trajectories=tuple(trajectories),
        estimator=EstimatorSpec(family="grpo", std_epsilon=1e-4),
        group_stats=GroupStats(k=4, mean=0.0, std=1.0),
    )
    schedule = {} if lr is None else {"learning_rate": lr}
    return Step(index=index, groups=(group,), schedule=schedule, optimizer=_telemetry())


def _telemetry():
    from reward_lens.record.schema import OptimizerTelemetry

    return OptimizerTelemetry(grad_norm_clipped=0.5)


def _ctx(**kwargs):
    return Context(access=dict(ACCESS), substrate=Substrate.PROGRAM, phase=Phase.POST_RUN, **kwargs)


def test_an_empty_window_is_void_and_the_remedy_says_where_the_steps_are():
    run = _run_of([_step(i) for i in range(4)])
    out = SelectionTerm(run, SurfaceFeatures(), window=(50, 60)).estimate(_ctx())
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.VOID
    assert "steps 0 to 3" in out.detail
    assert "Ask for a window inside the recorded range" in out.remedy


def test_a_one_step_window_refuses_and_names_the_step_that_would_close_it():
    run = _run_of([_step(i) for i in range(4)])
    out = SelectionTerm(run, SurfaceFeatures(), window=(1, 2)).estimate(_ctx())
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "successor step" in out.detail
    assert "Widen the window to include step 2" in out.remedy


def test_a_record_with_no_learning_rate_refuses_and_points_at_the_fitted_step_size():
    run = _run_of([_step(i, lr=None) for i in range(4)])
    out = SelectionTerm(run, SurfaceFeatures(), eta="schedule").estimate(_ctx())
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "schedule['learning_rate']" in out.detail
    assert "eta_eff" in out.remedy and "needs no schedule" in out.remedy


def test_a_featuriser_that_reads_nothing_refuses_and_names_both_banks():
    run = _run_of([_step(i, text=None) for i in range(4)])
    out = SelectionTerm(run, SurfaceFeatures()).estimate(_ctx())
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "SurfaceFeatures" in out.remedy and "RecordedFeatures" in out.remedy


def test_lambda_refuses_on_a_window_too_short_to_be_a_fraction_of_variance():
    run = _run_of([_step(i) for i in range(4)])
    out = SelectionExplainedFraction(run, SurfaceFeatures(), window=(0, 2)).estimate(_ctx())
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "second step pair" in out.detail
    assert "uncentred R-squared is 1 whatever the data says" in out.remedy


def test_the_envelope_remedy_is_specific_to_the_condition_that_failed():
    """The base preflight writes one remedy for all twelve conditions and "restrict the window" is
    not advice for `LINEAR_RESPONSE`: no window makes a large step small."""
    from reward_lens.core.envelope import RegimeReading

    run = _run_of([_step(i) for i in range(4)])
    reading = RegimeReading.of(LINEAR_RESPONSE=False, GROUP_NONDEGENERATE=True, NEAR_POLICY=True)
    out = SelectionTerm(run, SurfaceFeatures()).estimate(_ctx(regime_reading=reading))
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "smaller learning rate" in out.remedy
    assert "restrict the window to a span" not in out.remedy


def test_every_refusal_in_this_package_carries_a_remedy_that_is_an_instruction():
    """A `Refusal` cannot be constructed without one, so this asserts the stronger property: the
    remedy names an action rather than restating the failure."""
    run = _run_of([_step(i) for i in range(4)])
    refusals = [
        SelectionTerm(run, SurfaceFeatures(), window=(50, 60)).estimate(_ctx()),
        SelectionTerm(run, SurfaceFeatures(), window=(1, 2)).estimate(_ctx()),
        SelectionExplainedFraction(run, SurfaceFeatures(), window=(0, 2)).estimate(_ctx()),
    ]
    verbs = ("Ask", "Widen", "Supply", "Pass", "Read", "Restrict", "Lambda")
    for r in refusals:
        assert isinstance(r, Refusal)
        assert any(r.remedy.lstrip().startswith(v) for v in verbs), r.remedy


# ---------------------------------------------------------------------------
# 5. The nulls
# ---------------------------------------------------------------------------


def test_a_random_feature_has_a_covariance_indistinguishable_from_nothing():
    rng = np.random.default_rng(9)
    s = StepSample(
        0,
        ("f",),
        rng.normal(0.0, 1.0, (32, 1)),
        np.tile(np.asarray([-1.5, -0.5, 0.5, 1.5]), 8),
        np.repeat(np.arange(8), 4),
        tuple(f"t{g}" for g in np.repeat(np.arange(8), 4)),
    )
    draws = random_feature_null(s, n_draws=200, seed=1)
    observed = abs(
        float(selection_differential(s.features, s.advantages, s.group_ids, ("f",)).value[0])
    )
    result = summarise("baseline.random_feature", "|Cov(A, f)|", observed, draws)
    assert result.n_draws == 200
    assert result.p_value > 0.05


def test_a_planted_feature_beats_both_f1_nulls():
    """`f = A` exactly. The covariance is the advantage's own within-group variance and no
    permutation of the advantage and no random feature reaches it."""
    a = np.tile(np.asarray([-1.5, -0.5, 0.5, 1.5]), 8)
    g = np.repeat(np.arange(8), 4)
    s = StepSample(0, ("f",), a.reshape(-1, 1), a, g, tuple(f"t{x}" for x in g))
    observed = abs(
        float(selection_differential(s.features, s.advantages, s.group_ids, ("f",)).value[0])
    )
    random_result = summarise(
        "baseline.random_feature", "|Cov(A, f)|", observed, random_feature_null(s, n_draws=200)
    )
    permuted_result = summarise(
        "baseline.permuted_advantage",
        "|Cov(A, f)|",
        observed,
        permuted_advantage_null(s, n_draws=200)[:, 0],
    )
    assert random_result.p_value < 0.02
    assert permuted_result.p_value < 0.02


def test_the_permuted_step_null_destroys_a_planted_lambda():
    rng = np.random.default_rng(2)
    eta = 0.3
    samples, ledgers = [], []
    for t in range(20):
        f = rng.normal(0.0, 1.0, (8, 1))
        a = np.tile(np.asarray([-1.5, -0.5, 0.5, 1.5]), 2)
        g = np.repeat(np.arange(2), 4)
        samples.append(StepSample(t, ("f",), f, a, g, tuple(f"t{x}" for x in g)))
    for before, after in zip(samples, samples[1:]):
        cov = selection_differential(
            before.features, before.advantages, before.group_ids, before.names
        ).value[0]
        moved = before.features + eta * cov
        ledgers.append(
            ledger_between(
                before,
                StepSample(
                    after.index,
                    after.names,
                    moved,
                    after.advantages,
                    after.group_ids,
                    after.task_ids,
                ),
                eta=eta,
            )
        )
    scales = feature_scales(samples)
    fit = fit_lambda(ledgers, scales, n_bootstrap=0)
    draws = permuted_step_null(ledgers, scales, n_draws=500, seed=0)
    result = summarise("baseline.permuted_step", "Lambda", fit.lambda_, draws)
    assert fit.lambda_ == pytest.approx(1.0)
    assert result.null_median < 0.5
    assert result.p_value < 0.01


# ---------------------------------------------------------------------------
# 6. The frozen prediction
# ---------------------------------------------------------------------------


def test_the_transition_fit_recovers_a_planted_transition():
    steps = np.arange(0, 400)
    t50, scale = 240.0, 12.0
    truth = 0.02 + 0.75 / (1 + np.exp(-(steps - t50) / scale))
    rng = np.random.default_rng(7)
    noisy = np.clip(truth + rng.normal(0, 0.02, steps.size), 0, 1)
    fitted = transition_window(noisy, steps)
    assert fitted.fitted
    assert fitted.t50 == pytest.approx(240.0, abs=1.0)
    assert fitted.width == pytest.approx(scale * math.log(81.0), rel=0.05)


def test_the_transition_fit_declines_on_a_series_with_no_transition():
    rng = np.random.default_rng(7)
    flat = rng.normal(0.3, 0.02, 400)
    fitted = transition_window(flat, np.arange(400))
    assert not fitted.fitted
    assert "declined" in fitted.detail


def test_the_frozen_spec_hash_is_stable():
    """The spec is hashed into the StudyID, so an edit to any registered field is visible as a new
    version rather than as a silent revision. This asserts the hash of version 1."""
    frozen = freeze_prediction(frozen_at="2026-08-05T00:00:00+00:00")
    assert frozen.study_id.startswith("study:f2-lambda-lead-time@v1#")
    assert frozen.spec_hash == "spec:9551a981f5cac07f78f7890c562a24a3"
    assert LAMBDA_LEAD_TIME_SPEC.version == 1
    assert len(LAMBDA_LEAD_TIME_SPEC.hypotheses) == 2


def test_the_registered_cusum_metric_can_be_fired_by_the_future_and_the_test_says_so():
    """The defect found by running the frozen analysis on a real series, reproduced in the small.

    With `baseline=None` the CUSUM standardises against the whole series, so on a series that rises
    late every early point sits far below the mean, the negative accumulator crosses within a few
    points, and the reported onset is the detector using the future to define normal.
    """
    rng = np.random.default_rng(1)
    steps = np.arange(200)
    series = np.concatenate([np.full(120, 0.01), np.full(80, 0.6)]) + rng.normal(0, 0.005, 200)
    _, whole = onset_of(series, steps)
    _, early = onset_of(series, steps, baseline=60)
    assert whole < 20  # fires almost immediately, on the low side
    assert early >= 118  # fires where the series actually rises
    assert early - whole > 90


def test_the_prediction_is_unresolved_rather_than_scored_when_the_series_does_not_transition():
    rng = np.random.default_rng(4)
    steps = np.arange(200)
    result = score_lead_time(rng.normal(0.3, 0.05, 200), steps, rng.normal(0.2, 0.02, 200), steps)
    assert not result.resolved
    assert "no fitted transition" in result.detail
    assert math.isnan(result.metrics()["lambda_lead_fraction"])


def test_h2_is_unresolved_when_no_gradient_norm_series_exists():
    steps = np.arange(300)
    hack = 0.02 + 0.9 / (1 + np.exp(-(steps - 150) / 10.0))
    lam = 0.05 + 0.5 / (1 + np.exp(-(steps - 100) / 8.0))
    result = score_lead_time(lam, steps, hack, steps)
    assert math.isnan(result.metrics()["lambda_minus_gradnorm_lead_fraction"])
    assert "no optimiser telemetry" in result.detail


def test_the_detector_free_comparison_recovers_a_planted_lead_and_a_planted_lag():
    steps = np.arange(400)
    hack = 0.02 + 0.9 / (1 + np.exp(-(steps - 200) / 12.0))
    width = 12.0 * math.log(81.0)
    leading = 0.05 + 0.5 / (1 + np.exp(-(steps - 160) / 8.0))
    lagging = 0.05 + 0.5 / (1 + np.exp(-(steps - 240) / 8.0))
    assert score_lead_time(leading, steps, hack, steps).midpoint_lead == pytest.approx(
        40.0 / width, abs=0.05
    )
    assert score_lead_time(lagging, steps, hack, steps).midpoint_lead == pytest.approx(
        -40.0 / width, abs=0.05
    )


# ---------------------------------------------------------------------------
# 7. The declarations
# ---------------------------------------------------------------------------


def _lambda_instrument():
    return SelectionExplainedFraction(_run_of([_step(i) for i in range(4)]), SurfaceFeatures())


def _eta_instrument():
    return EffectiveStepSize(_run_of([_step(i) for i in range(4)]), SurfaceFeatures())


@pytest.mark.parametrize("cls", [SelectionTerm, SelectionResidual])
def test_the_f1_instruments_pass_lint(cls):
    run = _run_of([_step(i) for i in range(4)])
    assert lint_instrument(cls(run, SurfaceFeatures())) == []


@pytest.mark.parametrize("cls", [SelectionExplainedFraction, EffectiveStepSize])
def test_the_f2_instruments_pass_lint(cls):
    run = _run_of([_step(i) for i in range(4)])
    assert lint_instrument(cls(run, SurfaceFeatures())) == []


def test_all_four_quantities_are_registered_with_the_units_the_catalogue_gives_them():
    for qid, dimension in (
        ("selection.term", "feature"),
        ("selection.residual", "feature"),
        ("selection.explained_fraction", "1"),
        ("selection.eta_eff", "1"),
    ):
        assert qid in QUANTITIES
        assert QUANTITIES.get(qid).unit.dimension == dimension


def test_the_estimator_ladder_carries_one_rung_for_each_and_a_bias_statement():
    from reward_lens.core.quantity import ladder

    for qid in (
        "selection.term",
        "selection.residual",
        "selection.explained_fraction",
        "selection.eta_eff",
    ):
        rungs = ladder(qid)
        assert len(rungs) == 1, qid
        assert rungs[0].bias.why
        assert rungs[0].requires == {Component.RECORD: Access.RECORD}

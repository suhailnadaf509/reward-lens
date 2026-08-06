"""Unit tests: A3, A4, A6 and A7.

Three kinds of test, and the middle one is the one that catches real errors.

Hand-computed values, where the answer is known before the code runs: Spearman's factor on
components chosen so the square root is exact, a garbling LP built by multiplying a channel by a
matrix chosen in advance, a flip rate on four numbers.

**Constructions whose answer is known by how they were built.** ``P_B = P_A M`` for a chosen ``M``
is feasible because it was assembled that way. Walking ``P_B`` away from that point and recording
where the LP stops returning zero tests something the arithmetic alone does not: the set of
garblings of a channel is a convex body with interior, so a small perturbation stays feasible and
only overtaking ``P_A``'s own information content breaks it. That measured transition point is in
the test rather than in a comment.

The generated invariance test per instrument, and a refusal test per instrument asserting both the
reason and the remedy. The remedy is asserted because a refusal without an actionable remedy is a
tool that looks broken, and a string nobody checks is a string that rots.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import binom

from reward_lens.core.invariance import (
    COVARIANT_LINEAR,
    InvariancePayload,
    check_invariance,
)
from reward_lens.core.quantity import QUANTITIES, load_quantities
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import lint_instrument
from reward_lens.measure.metrology.attenuation import (
    AttenuationFactor,
    ErrorsInVariablesAttenuation,
    RewardVariance,
    eiv_gradient,
    factor_from_scores,
    spearman_factor,
    within_prompt_covariance,
)
from reward_lens.measure.metrology.blackwell import (
    AgreementTable,
    BlackwellOrder,
    Verdict,
    bayes_accuracy,
    deficiency_from_scores,
    garbling_feasible,
    group_ranking_regret,
    le_cam_deficiency,
    physical_gap,
    quantile_levels,
)
from reward_lens.measure.metrology.distribution import (
    GraderStochasticity,
    RepeatedScores,
    facet_effects,
    facet_shares,
    flip_rate_from_scores,
    flip_rates,
    repeat_variance,
    repeats_for_majority,
    sigma_from_scores,
)
from reward_lens.measure.metrology.flakiness import (
    EnvironmentFlakiness,
    ReplaySet,
    attribute,
    range_from_scores,
    replay,
    spread,
)


@pytest.fixture(scope="module", autouse=True)
def _registry() -> None:
    """The quantity registry, loaded once, so `lint_instrument` can check the ids resolve.

    Loading is idempotent: `Registry.register` is a no-op on an identical entry, so a second load
    from another test module costs nothing and cannot conflict.
    """
    if "grader.attenuation" not in QUANTITIES:
        load_quantities()


# ===========================================================================
# A3, the attenuation factor
# ===========================================================================


def test_spearman_factor_is_the_square_root_of_the_reliability() -> None:
    """Components chosen so the answer is exact: 0.75 over 1.0 is 0.75, whose root is 0.8660254."""
    rv = RewardVariance(sigma2_true=0.75, sigma2_err=0.25)
    assert rv.reliability == pytest.approx(0.75)
    assert spearman_factor(rv) == pytest.approx(math.sqrt(0.75))


def test_a_grader_with_no_error_attenuates_nothing() -> None:
    assert spearman_factor(RewardVariance(sigma2_true=2.0, sigma2_err=0.0)) == pytest.approx(1.0)


def test_rung_one_reduces_to_rung_zero_when_the_features_are_measured_exactly() -> None:
    """The claim that makes them two rungs of one quantity rather than two quantities.

    With ``C_err = 0`` the errors-in-variables step is the identity and the only correction left is
    the reward-side factor, so rung 1 has to return rung 0's number to floating point.
    """
    rv = RewardVariance(sigma2_true=0.6, sigma2_err=0.4)
    rung1 = ErrorsInVariablesAttenuation(
        np.array([[2.0]]), np.zeros((1, 1)), np.array([0.4]), reward_components=rv
    ).compute()
    assert rung1.factor == pytest.approx(spearman_factor(rv))
    assert rung1.feature_factor == pytest.approx(1.0)


def test_the_errors_in_variables_correction_reorders_features() -> None:
    """The kill condition's test case, built so the reordering is arithmetic rather than luck.

    Feature 0 has almost no rollout noise and feature 1 is almost all rollout noise. Observed, they
    look nearly equally selected. Corrected, feature 0 is an order of magnitude ahead.
    """
    c_obs = np.array([[1.0, 0.0], [0.0, 1.0]])
    c_err = np.array([[0.9, 0.0], [0.0, 0.1]])
    s = np.array([0.30, 0.35])
    out = ErrorsInVariablesAttenuation(c_obs, c_err, s).compute()
    assert out.rank_observed == (1, 0)
    assert out.rank_corrected == (0, 1)
    assert out.rank_changed is True
    assert out.max_rank_move == 1
    # beta_corr = (C_obs - C_err)^-1 S = [0.30/0.1, 0.35/0.9]
    assert out.beta_corrected == pytest.approx([3.0, 0.35 / 0.9])
    assert "rank 1 to rank 2" in out.says


def test_rung_zero_cannot_reorder_and_says_so() -> None:
    """A scalar multiple preserves order, so a rung 0 reading has not tested the kill condition."""
    out = AttenuationFactor(RewardVariance(sigma2_true=1.0, sigma2_err=1.0)).compute()
    assert out.rank_changed is False
    assert "cannot reorder" in out.says


def test_within_prompt_covariance_recovers_a_planted_within_group_covariance() -> None:
    """Two groups, two features, both built from the same within-group deviations."""
    dev = np.array([[1.0, 2.0], [-1.0, -2.0]])
    features = np.vstack([dev + 10.0, dev - 5.0])
    gids = np.array([0, 0, 1, 1])
    got = within_prompt_covariance(features, gids)
    # Each group contributes centred rows (+-1, +-2), so the pooled df is 2 and the sum of outer
    # products is 2 * [[2, 4], [4, 8]].
    assert got == pytest.approx(np.array([[2.0, 4.0], [4.0, 8.0]]))


def test_within_prompt_covariance_refuses_singleton_groups_rather_than_returning_zeros() -> None:
    with pytest.raises(ValueError, match="one member"):
        within_prompt_covariance(np.eye(3), np.array([0, 1, 2]))


def test_a_negative_error_variance_is_a_caller_bug_and_raises() -> None:
    """A within-cell mean square cannot be negative, so this is not an anticipated condition."""
    with pytest.raises(ValueError, match="cannot be negative"):
        RewardVariance(sigma2_true=1.0, sigma2_err=-0.1)


def test_a_negative_true_variance_is_anticipated_and_refuses_with_the_numbers() -> None:
    """ANOVA component estimates go negative near zero, so this returns a Refusal, not an exception."""
    out = AttenuationFactor(RewardVariance(sigma2_true=-0.01, sigma2_err=1.0)).compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.BELOW_LOD
    assert "disagrees with itself" in out.detail
    assert "replications" in out.remedy
    assert out.statistics["sigma2_true"] == pytest.approx(-0.01)


def test_attenuation_refuses_without_components_and_names_what_would_supply_them() -> None:
    out = AttenuationFactor().compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "A2" in out.remedy
    assert "REPLICATE" in out.remedy


def test_attenuation_refuses_on_an_unstandardised_advantage() -> None:
    """The square root is derived from dividing by the observed spread. No division, no factor."""
    rv = RewardVariance(sigma2_true=1.0, sigma2_err=1.0, standardised=False)
    out = AttenuationFactor(rv).compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "standardised=True" in out.remedy


def test_from_replicates_recovers_planted_components() -> None:
    """A one-way design with a known item spread and a known residual."""
    rng = np.random.default_rng(7)
    truth = rng.normal(0.0, 2.0, size=400)
    scores = truth[:, None] + rng.normal(0.0, 1.0, size=(400, 8))
    rv = RewardVariance.from_replicates(scores)
    assert rv.sigma2_true == pytest.approx(4.0, rel=0.25)
    assert rv.sigma2_err == pytest.approx(1.0, rel=0.1)


def test_a3_invariance_under_an_affine_rescaling_of_the_reward() -> None:
    """The generated test. The whole path runs on the transformed scores, not on cached components."""
    rng = np.random.default_rng(3)
    scores = rng.normal(size=(20, 5)) + 2.0 * rng.normal(size=(20, 1))

    def attenuation_of(payload: InvariancePayload) -> float:
        return factor_from_scores(np.asarray(payload.scores).reshape(20, 5))

    report = check_invariance(
        attenuation_of,
        "reward.affine",
        InvariancePayload(scores=scores.ravel(), group_ids=np.repeat(np.arange(20), 5)),
        n=48,
    )
    assert report.passed, report.render()


@settings(max_examples=60, deadline=None)
@given(
    true=st.floats(min_value=1e-6, max_value=1e6),
    err=st.floats(min_value=0.0, max_value=1e6),
)
def test_the_factor_lies_in_zero_to_one_and_falls_as_error_rises(true: float, err: float) -> None:
    """A factor above one would say grader error sharpens the signal."""
    a = spearman_factor(RewardVariance(sigma2_true=true, sigma2_err=err))
    assert 0.0 < a <= 1.0
    worse = spearman_factor(RewardVariance(sigma2_true=true, sigma2_err=err * 2 + 1.0))
    assert worse <= a + 1e-12


def test_eiv_gradient_reports_conditioning_rather_than_hiding_it() -> None:
    """A feature whose observed variance is nearly all noise leaves an ill-conditioned matrix."""
    c_obs = np.array([[1.0, 0.0], [0.0, 1.0]])
    c_err = np.array([[0.0, 0.0], [0.0, 0.9999]])
    _, _, cond = eiv_gradient(c_obs, c_err, np.array([1.0, 1.0]))
    assert cond > 1e3


# ===========================================================================
# A4, the Blackwell order
# ===========================================================================


def _channel(rows: list[list[float]]) -> np.ndarray:
    p = np.asarray(rows, dtype=np.float64)
    return p / p.sum(axis=1, keepdims=True)


def test_a_channel_built_as_a_garbling_is_found_to_be_one() -> None:
    """``P_B = P_A M`` for a chosen ``M``, so feasibility is a property of the construction."""
    p_a = _channel([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
    m = _channel([[0.6, 0.4], [0.5, 0.5], [0.2, 0.8]])
    p_b = p_a @ m
    assert le_cam_deficiency(p_a, p_b).delta == pytest.approx(0.0, abs=1e-9)
    assert garbling_feasible(p_a, p_b)


def test_perturbing_the_garbling_stays_feasible_until_b_overtakes_a() -> None:
    """Walk away from the constructed point and record where the LP stops returning zero.

    The interesting part is that small perturbations do **not** break feasibility, and that is
    correct rather than a tolerance artifact: the set of garblings of ``P_A`` is convex with a
    non-empty interior, and the constructed ``P_B`` sits inside it rather than on its boundary. What
    does break feasibility is pushing ``P_B`` past ``P_A``'s own information content, and the
    measured transition here is at a perturbation of 0.2, where ``P_B`` becomes sharper than the
    channel it was supposed to be a degradation of.

    The deficiency is monotone along that walk, which is the property the verdict relies on, and the
    fact that the transition is gradual rather than a cliff is exactly why the instrument tests
    against a simulated null instead of picking a tolerance.
    """
    p_a = _channel([[0.9, 0.1], [0.1, 0.9]])
    m = _channel([[0.8, 0.2], [0.3, 0.7]])
    p_b = p_a @ m
    deltas = []
    for eps in (0.0, 0.01, 0.05, 0.2, 0.4):
        moved = np.clip(p_b + np.array([[eps, -eps], [-eps, eps]]), 1e-9, None)
        moved = moved / moved.sum(axis=1, keepdims=True)
        deltas.append(le_cam_deficiency(p_a, moved).delta)
    assert deltas[:3] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert deltas[3] > 0.0
    assert deltas == sorted(deltas)
    # The extreme: a noisy channel cannot be garbled into a perfect one.
    assert le_cam_deficiency(p_a, np.eye(2)).delta == pytest.approx(0.1, abs=1e-6)


def test_every_channel_garbles_into_an_uninformative_one() -> None:
    """A constant channel carries no information, so it is a garbling of anything."""
    p_a = _channel([[0.9, 0.1], [0.2, 0.8]])
    uninformative = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert garbling_feasible(p_a, uninformative)
    assert not garbling_feasible(uninformative, p_a)


def test_identical_channels_are_equivalent_in_both_directions() -> None:
    p = _channel([[0.8, 0.2], [0.3, 0.7]])
    assert le_cam_deficiency(p, p).delta == pytest.approx(0.0, abs=1e-9)


def test_quantile_levels_depend_on_ranks_alone() -> None:
    """Which is what makes the whole instrument affine-invariant rather than approximately so."""
    s = np.array([-3.0, 0.5, 0.5, 2.0, 11.0, 4.0])
    assert np.array_equal(quantile_levels(s, 3), quantile_levels(7.5 * s + 100.0, 3))


def test_a4_invariance_under_an_affine_rescaling_of_either_grader() -> None:
    """The generated test, and the catalogue's group means exactly this.

    A garbling order that moved when somebody rescaled a reward would be an artifact of the scale.
    The transformation is applied to grader A's scores; grader B and the states are held, so the
    check is on the estimator rather than on a symmetry of the data.
    """
    rng = np.random.default_rng(11)
    n = 600
    states = rng.integers(0, 2, size=n)
    b = states + rng.normal(0.0, 1.5, n)
    a = states + rng.normal(0.0, 0.6, n)

    def deficiency_of(payload: InvariancePayload) -> float:
        return deficiency_from_scores(np.asarray(payload.scores), b, states)

    report = check_invariance(
        deficiency_of,
        "reward.affine",
        InvariancePayload(scores=a, group_ids=np.zeros(n, dtype=int)),
        n=32,
    )
    assert report.passed, report.render()


def test_an_unordered_pair_is_a_finding_and_not_a_refusal() -> None:
    """Two graders good at different states garble each other in neither direction."""
    rng = np.random.default_rng(5)
    n = 3000
    states = rng.integers(0, 2, size=n)
    # A separates state 0 from the rest; B separates state 1. Neither dominates.
    a = np.where(states == 0, rng.normal(3.0, 1.0, n), rng.normal(0.0, 1.0, n))
    b = np.where(states == 1, rng.normal(3.0, 1.0, n), rng.normal(0.0, 1.0, n))
    table = AgreementTable.from_scores(a, b, states, n_levels=4)
    out = BlackwellOrder(table, simulations=40, seed=2).compute()
    assert not isinstance(out, Refusal)
    assert out.verdict in {Verdict.NOT_ORDERED.value, Verdict.EQUIVALENT.value}


def test_blackwell_dominance_implies_no_worse_group_ranking_regret() -> None:
    """The theorem, instantiated at the decision the loop actually makes.

    If B is a garbling of A then A is at least as good for *every* decision problem, so its regret
    at the real K cannot be higher. A violation here would mean the LP and the loss disagree, which
    is a defect in one of them rather than a finding about the graders.
    """
    p_a = _channel([[0.85, 0.15], [0.15, 0.85]])
    m = _channel([[0.7, 0.3], [0.4, 0.6]])
    p_b = p_a @ m
    prior = np.array([0.5, 0.5])
    regret_a, se_a = group_ranking_regret(p_a, prior, k=8, draws=40_000, seed=0)
    regret_b, se_b = group_ranking_regret(p_b, prior, k=8, draws=40_000, seed=0)
    assert regret_a <= regret_b + 2.0 * math.hypot(se_a, se_b)
    assert regret_a < regret_b


def test_the_physical_gap_is_zero_when_b_was_generated_from_a_alone() -> None:
    """The question the joint answers and the marginals cannot."""
    rng = np.random.default_rng(13)
    n = 8000
    states = rng.integers(0, 2, size=n)
    a = states + rng.normal(0.0, 1.0, n)
    la = quantile_levels(a, 3)
    m = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    lb = np.array([rng.choice(3, p=m[level]) for level in la])
    table = AgreementTable.from_scores(a, lb + rng.uniform(0.0, 0.9, n), states, n_levels=3)
    assert physical_gap(table) < 0.06


def test_bayes_accuracy_is_the_best_the_channel_can_do() -> None:
    """Hand-checked: a perfect channel on a balanced prior gets 1.0 and a constant one gets 0.5."""
    assert bayes_accuracy(np.eye(2), np.array([0.5, 0.5])) == pytest.approx(1.0)
    assert bayes_accuracy(np.full((2, 2), 0.5), np.array([0.5, 0.5])) == pytest.approx(0.5)


def test_blackwell_refuses_without_a_shared_labelled_slice() -> None:
    out = BlackwellOrder().compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "AgreementTable.from_scores" in out.remedy


def test_blackwell_refuses_on_a_slice_with_one_state() -> None:
    """Every one-row channel garbles every other, so the order would be vacuously `equivalent`."""
    rng = np.random.default_rng(1)
    n = 200
    table = AgreementTable.from_scores(rng.normal(size=n), rng.normal(size=n), np.ones(n, int))
    out = BlackwellOrder(table).compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "chosen-first" in out.remedy
    assert out.statistics["observed_states"] == 1


def test_the_agreement_table_stores_the_joint_and_reduces_only_on_the_way_out() -> None:
    """The storage claim: a marginal is recoverable from the table and the table is not from margins."""
    rng = np.random.default_rng(17)
    n = 500
    states = rng.integers(0, 2, size=n)
    table = AgreementTable.from_scores(rng.normal(size=n), rng.normal(size=n), states, n_levels=3)
    assert table.counts.shape == (2, 3, 3)
    assert table.counts.sum() == n
    assert np.allclose(table.channel("a"), table.counts.sum(axis=2) / table.state_totals()[:, None])


# ===========================================================================
# A6, the stochasticity profile
# ===========================================================================


def test_repeat_variance_on_four_numbers_by_hand() -> None:
    """Two items, two repeats. Item 0 spreads by 2, item 1 by 4, pooled variance is (2 + 8) / 2."""
    data = RepeatedScores(scores=np.array([[1.0, 3.0], [10.0, 14.0]]))
    rv = repeat_variance(data)
    assert rv.df == 2
    assert rv.variance == pytest.approx((2.0 + 8.0) / 2.0)
    assert rv.sigma == pytest.approx(math.sqrt(5.0))


def test_a_deterministic_grader_reads_zero_rather_than_refusing() -> None:
    """The kill condition, as a reading. This is the honest scope limit and not a failure."""
    out = GraderStochasticity(
        RepeatedScores(scores=np.tile(np.arange(6.0)[:, None], (1, 4)))
    ).compute()
    assert not isinstance(out, Refusal)
    assert out.deterministic is True
    assert out.sigma == 0.0
    assert out.flip_rate == 0.0
    assert "deterministic on this design" in out.says


def test_the_flip_rate_on_a_pair_built_to_flip_a_known_fraction() -> None:
    """Item 0 beats item 1 on three of four repeat combinations, so the modal probability is 0.75."""
    data = RepeatedScores(scores=np.array([[1.0, -1.0], [0.0, 0.0]]))
    profile = flip_rates(data)
    assert profile.n_pairs == 1
    assert profile.modal_probability[0] == pytest.approx(0.5)
    data2 = RepeatedScores(scores=np.array([[1.0, 1.0, 1.0, -5.0], [0.0, 0.0, 0.0, 0.0]]))
    assert flip_rates(data2).modal_probability[0] == pytest.approx(0.75)
    assert flip_rates(data2).flip_rate == pytest.approx(0.25)


def test_ties_are_reported_and_never_split() -> None:
    """A grader returning the same number for both has not expressed a preference."""
    data = RepeatedScores(scores=np.array([[2.0, 2.0], [2.0, 2.0]]))
    profile = flip_rates(data)
    assert profile.tie_rate == pytest.approx(1.0)
    assert profile.flip_rate == pytest.approx(0.0)


def test_repeats_for_majority_against_the_binomial_it_is_solving() -> None:
    """A single pair at p = 0.8, checked against `scipy.stats.binom` directly."""
    data = RepeatedScores(scores=np.array([[1.0] * 8 + [-9.0] * 2, [0.0] * 10]))
    profile = flip_rates(data)
    assert profile.modal_probability[0] == pytest.approx(0.8)
    m, achieved = repeats_for_majority(profile, reference_repeats=51, target=0.95)
    assert m % 2 == 1
    a_m = binom.sf((m - 1) // 2, m, 0.8)
    a_ref = binom.sf(25, 51, 0.8)
    assert achieved == pytest.approx(a_m * a_ref + (1 - a_m) * (1 - a_ref))
    smaller = binom.sf((m - 3) // 2, m - 2, 0.8)
    assert smaller * a_ref + (1 - smaller) * (1 - a_ref) < 0.95


def test_the_facet_share_null_is_what_the_facet_would_explain_by_doing_nothing() -> None:
    """Two levels over six repeats is one parameter out of five, so the null share is 0.20."""
    rng = np.random.default_rng(2)
    scores = rng.normal(size=(200, 6))
    data = RepeatedScores(scores=scores, facets={"order": np.array([0, 1, 0, 1, 0, 1])})
    shares, null = facet_shares(data)
    assert null["order"] == pytest.approx(0.2)
    assert shares["order"] == pytest.approx(0.2, abs=0.05)


def test_facet_effects_recover_a_planted_systematic_shift() -> None:
    """Half a point added to every odd repeat, and the estimator has to find half a point."""
    rng = np.random.default_rng(4)
    scores = rng.normal(size=(300, 4)) + np.arange(300)[:, None]
    scores[:, 1::2] += 0.5
    data = RepeatedScores(scores=scores, facets={"order": np.array([0, 1, 0, 1])})
    fx = facet_effects(data)["order"]
    assert fx.effect == pytest.approx(0.5, abs=0.08)
    assert fx.significant is True


def test_distribution_refuses_on_one_draw_per_item_and_names_the_access_it_needs() -> None:
    """A variance over one observation is undefined, not zero, and the two must stay apart."""
    out = GraderStochasticity(RepeatedScores(scores=np.arange(9.0)[:, None])).compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "GRADER:REPLICATE" in out.detail
    assert "presentation order" in out.remedy
    assert out.statistics["items_with_two_or_more"] == 0


def test_a6_invariance_under_permuting_repeats_within_an_item() -> None:
    """The generated test for the catalogue's group, on both the sigma and the flip rate.

    The flip rate is the one that bites: a diagonal pairing would fail this, which is why the
    U-statistic over all repeat combinations is the default rather than a refinement.
    """
    rng = np.random.default_rng(21)
    scores = rng.normal(size=(15, 4)) + rng.normal(size=(15, 1)) * 3.0
    payload = InvariancePayload(scores=scores.ravel(), group_ids=np.repeat(np.arange(15), 4))
    for fn in (
        lambda p: sigma_from_scores(np.asarray(p.scores), 4),
        lambda p: flip_rate_from_scores(np.asarray(p.scores), 4),
    ):
        report = check_invariance(fn, "group.permutation", payload, n=40)
        assert report.passed, report.render()


def test_sigma_is_covariant_under_an_affine_rescaling_with_weight_one() -> None:
    """Not the catalogue's group, and worth asserting anyway: a standard deviation scales by ``a``.

    The catalogue assigns `group.permutation`, which is the group whose failure would be a finding
    about the grader. This second check is about the estimator: a sigma that did not scale linearly
    with the reward would mean the pooling had gone wrong somewhere.
    """
    rng = np.random.default_rng(31)
    scores = rng.normal(size=(12, 5))
    report = check_invariance(
        lambda p: sigma_from_scores(np.asarray(p.scores), 5),
        "reward.affine",
        InvariancePayload(scores=scores.ravel(), group_ids=np.repeat(np.arange(12), 5)),
        n=32,
        relation=COVARIANT_LINEAR,
    )
    assert report.passed, report.render()


@settings(max_examples=40, deadline=None)
@given(
    st.lists(
        st.lists(st.floats(min_value=-50, max_value=50, allow_nan=False), min_size=3, max_size=3),
        min_size=2,
        max_size=6,
    )
)
def test_the_flip_rate_never_exceeds_one_half(rows: list[list[float]]) -> None:
    """It is one minus a modal probability, and a modal probability is at least a half."""
    rate = flip_rates(RepeatedScores(scores=np.array(rows))).flip_rate
    assert 0.0 <= rate <= 0.5 + 1e-12


# ===========================================================================
# A7, environment flakiness
# ===========================================================================


def test_the_range_is_the_range() -> None:
    data = ReplaySet(scores=np.array([[0.579, 0.768, 0.6, 0.7]]))
    sp = spread(data)
    assert sp.ranges[0] == pytest.approx(0.768 - 0.579)
    assert sp.modal_agreement[0] == pytest.approx(0.25)


def test_a_deterministic_environment_reads_zero_rather_than_refusing() -> None:
    """The kill condition again: under one point on ten environments retires the instrument."""
    out = EnvironmentFlakiness(ReplaySet(scores=np.full((3, 20), 0.62))).compute()
    assert not isinstance(out, Refusal)
    assert out.deterministic is True
    assert out.range_pp == pytest.approx(0.0)
    assert "deterministic here" in out.says


def test_flakiness_refuses_on_a_single_replay() -> None:
    out = EnvironmentFlakiness(ReplaySet(scores=np.array([[0.6]]))).compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "replay each task at least twice" in out.remedy.lower()


def test_flakiness_refuses_scores_that_are_not_fractions() -> None:
    """`env.flakiness` is registered in pp, so a 0-to-100 score would report a spread times a hundred."""
    out = EnvironmentFlakiness(ReplaySet(scores=np.array([[57.9, 76.8, 60.0]]))).compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.UNIT_MISMATCH
    assert "score_scale" in out.remedy
    assert out.statistics["max"] == pytest.approx(76.8)


def test_a_declared_scale_makes_the_same_scores_readable() -> None:
    out = EnvironmentFlakiness(
        ReplaySet(scores=np.array([[57.9, 76.8, 60.0]]), score_scale=1.0)
    ).compute()
    assert not isinstance(out, Refusal)
    assert out.max_range_pp == pytest.approx(76.8 - 57.9)
    assert "Any improvement under 18.9 points" in out.says


def test_replay_records_a_crash_as_a_non_completion_and_never_as_a_score() -> None:
    """Scoring an exception as zero is the defect B4 exists to count, and this is where it would go."""
    calls = {"n": 0}

    def flaky() -> float:
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("sandbox died")
        return 0.7

    data = replay(flaky, n=5, environment="test")
    assert np.isnan(data.scores[0, 2])
    assert data.completion_rate == pytest.approx(0.8)
    out = EnvironmentFlakiness(data).compute()
    assert out.deterministic is True
    assert "did not complete" in out.says


def test_the_occasion_facet_hands_a3_its_components() -> None:
    """Rung 1's whole content: an environment's spread is a term in the grader's error variance."""
    rng = np.random.default_rng(9)
    truth = rng.uniform(0.2, 0.8, size=50)
    scores = np.clip(truth[:, None] + rng.normal(0.0, 0.05, size=(50, 10)), 0.0, 1.0)
    components = ReplaySet(scores=scores, environment="synthetic").as_variance_components()
    assert components.sigma2_err == pytest.approx(0.05**2, rel=0.25)
    assert spearman_factor(components) < 1.0
    assert AttenuationFactor(components).compute().factor == pytest.approx(
        spearman_factor(components)
    )


def test_cause_attribution_finds_a_planted_cause_above_its_null() -> None:
    """Half the replays run in a reused sandbox and score a fifth of a point lower."""
    rng = np.random.default_rng(15)
    scores = np.full((4, 20), 0.6) + rng.normal(0.0, 0.01, size=(4, 20))
    reused = np.tile(np.arange(20) % 2, (4, 1))
    scores += reused * -0.2
    data = ReplaySet(scores=scores, causes={"sandbox": reused}, environment="synthetic")
    shares, null = attribute(data)
    assert shares["sandbox"] > 0.95
    assert null["sandbox"] == pytest.approx(1.0 / 19.0)
    assert "sandbox" in EnvironmentFlakiness(data).compute().says


def test_a7_invariance_under_permuting_replay_order() -> None:
    """The generated test. A reading that moved here would be picking up on replay order itself."""
    rng = np.random.default_rng(23)
    scores = rng.uniform(0.0, 1.0, size=(8, 6))
    report = check_invariance(
        lambda p: range_from_scores(np.asarray(p.scores), 6),
        "group.permutation",
        InvariancePayload(scores=scores.ravel(), group_ids=np.repeat(np.arange(8), 6)),
        n=40,
    )
    assert report.passed, report.render()


# ===========================================================================
# What every instrument in this package has to declare
# ===========================================================================


@pytest.mark.parametrize(
    "instrument",
    [
        AttenuationFactor(),
        ErrorsInVariablesAttenuation(),
        BlackwellOrder(),
        GraderStochasticity(),
        EnvironmentFlakiness(),
    ],
    ids=lambda i: type(i).__name__,
)
def test_every_instrument_passes_the_section_four_two_lint(instrument: object) -> None:
    """Quantity registered, baselines named, envelope declared, invariance group declared."""
    assert lint_instrument(instrument) == []


@pytest.mark.parametrize(
    "instrument",
    [
        AttenuationFactor(),
        ErrorsInVariablesAttenuation(),
        BlackwellOrder(),
        GraderStochasticity(),
        EnvironmentFlakiness(),
    ],
    ids=lambda i: type(i).__name__,
)
def test_an_instrument_with_no_data_refuses_rather_than_raising(instrument: object) -> None:
    """The standing rule, checked at the one place every instrument shares: never an exception."""
    out = instrument.compute()  # type: ignore[attr-defined]
    assert isinstance(out, Refusal)
    assert out.remedy.strip()


def test_relative_error_excludes_a_facet_main_effect_and_absolute_includes_it():
    """E43. The two are different numbers and the module argues for the relative one.

    A facet main effect shifts every rollout in a group by the same amount, so it cancels out of a
    group-centred contrast. `from_facets` used to sum every component outside the universe, which is
    verbatim Brennan's *absolute* error variance, while the module docstring argued for the relative
    one. The correction was overstated whenever a facet carried a main effect.
    """
    from reward_lens.measure.metrology.attenuation import RewardVariance

    facets = {"p": 1.0, "r": 0.6, "pr,e": 0.4}
    relative = RewardVariance.from_facets(facets, universe="p")
    absolute = RewardVariance.from_facets(facets, universe="p", kind="absolute")

    assert relative.sigma2_err == pytest.approx(0.4), "only the interaction with p"
    assert absolute.sigma2_err == pytest.approx(1.0), "every component except p's own"
    assert relative.reliability**0.5 == pytest.approx(0.845154, abs=1e-6)
    assert absolute.reliability**0.5 == pytest.approx(0.707107, abs=1e-6)


def test_word_separated_facet_names_resolve_the_same_way_as_single_letter_ones():
    from reward_lens.measure.metrology.attenuation import RewardVariance

    words = RewardVariance.from_facets(
        {"item": 1.0, "rater": 0.6, "item x rater": 0.4}, universe="item"
    )
    assert words.sigma2_err == pytest.approx(0.4)


def test_a_naming_mismatch_refuses_rather_than_reporting_a_perfect_grader():
    """Relative error of exactly zero is an attenuation factor of exactly 1, which is a claim."""
    from reward_lens.measure.metrology.attenuation import RewardVariance

    with pytest.raises(ValueError, match="naming-convention mismatch"):
        RewardVariance.from_facets({"item": 1.0, "rater": 0.6}, universe="item")

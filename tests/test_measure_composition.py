"""B3 and B4: the counterfactual composition, and the silent-zero census.

Every expected number here was computed by hand from the composition and is written into the test
that asserts it, so a change in the implementation cannot quietly move the target. The worked group
is the score tree's, kept deliberately: four rollouts, a task reward of `correct + 0.25 * style`,
and a
Kimi-K3-shaped override pinning the score to -1 when a rollout's token count exceeds 1.5 times the
task's budget.

    rollout   correct  style  tokens  b0    over budget   score
    0         1.0      0.8    100     100   no             1.20
    1         1.0      0.4    200     100   yes           -1.00   (would be 1.10)
    2         0.0      0.6     90     100   no             0.15
    3         0.0      0.2    400     100   yes           -1.00   (would be 0.05)

The group mean is -0.1625 with the override and 0.625 without it, and the advantages follow.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from reward_lens.core.invariance import TRIVIAL_GROUP, InvariancePayload, check_invariance
from reward_lens.core.quantity import QUANTITIES, load_quantities
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.composition.abstention import (
    AbstentionRate,
    SilentZeroRate,
    as_the_framework_used_it,
    baseline_shifts,
    read_census,
)
from reward_lens.measure.composition.composition import (
    CompositionTree,
    CounterfactualComposition,
    default_offset_node,
    null_leak,
    null_probe,
    offset_node,
    summarise,
)
from reward_lens.record.schema import EstimatorSpec
from reward_lens.record.scores import (
    GraderCallRef,
    GroupContext,
    GroupScores,
    Leaf,
    Mask,
    Override,
    PredicateRef,
    Product,
    ScoreContext,
    WeightedSum,
    evaluate,
)

load_quantities()

# ---------------------------------------------------------------------------
# The worked group
# ---------------------------------------------------------------------------

ROWS = [
    # correct, style, tokens
    (1.0, 0.8, 100.0),
    (1.0, 0.4, 200.0),
    (0.0, 0.6, 90.0),
    (0.0, 0.2, 400.0),
]

OVER_BUDGET = PredicateRef(
    name="over_budget", feature="tokens", op=">", threshold_feature="b0", scale=1.5
)

CENTRED = EstimatorSpec(
    family="grpo", group_centred=True, std_normalised=False, degenerate_policy="skip"
)


def call(grader: str = "rubric", outcome: str = "returned") -> GraderCallRef:
    return GraderCallRef(grader=grader, outcome=outcome, latency_s=0.012)


def task_tree(correct: float | None, style: float | None, *, failed: bool = False) -> WeightedSum:
    return WeightedSum(
        name="task",
        children=(
            Leaf("correct", correct, call("verifier", "raised" if failed else "returned"), failed),
            Leaf("style", style, call("judge", "raised" if failed else "returned"), failed),
        ),
        weights=(1.0, 0.25),
    )


def k3_tree(correct: float, style: float, *, failed: bool = False) -> Override:
    return Override(
        name="length_override",
        condition=OVER_BUDGET,
        constant=-1.0,
        otherwise=task_tree(correct, style, failed=failed),
    )


def k3_ctx(correct: float, tokens: float) -> ScoreContext:
    return ScoreContext(
        features={"tokens": tokens, "b0": 100.0, "correct": correct},
        group=GroupContext(accuracy=0.5, k=4, id="g0"),
    )


def worked_group(estimator: EstimatorSpec = CENTRED) -> GroupScores:
    return GroupScores(
        trees=tuple(k3_tree(c, s) for c, s, _ in ROWS),
        contexts=tuple(k3_ctx(c, t) for c, _, t in ROWS),
        estimator=estimator,
        id="g0",
    )


def plain_group(estimator: EstimatorSpec = CENTRED) -> GroupScores:
    """The same rollouts with the override taken out of the composition entirely."""
    return GroupScores(
        trees=tuple(task_tree(c, s) for c, s, _ in ROWS),
        contexts=tuple(k3_ctx(c, t) for c, _, t in ROWS),
        estimator=estimator,
        id="g0",
    )


# ---------------------------------------------------------------------------
# B3, part one: what the reward is made of
# ---------------------------------------------------------------------------


def test_the_summary_counts_the_primitives_a_weights_dict_cannot_hold() -> None:
    got = summarise([t for t in worked_group().trees])
    assert got.n_trees == 4
    # Four rollouts, each an Override over a WeightedSum over two Leaves.
    assert got.n_nodes == 16
    assert got.n_leaves == 8
    assert got.node_types == {"Override": 4, "WeightedSum": 4, "Leaf": 8}
    assert got.inexpressible == ("Override",)
    assert got.n_inexpressible == 1
    assert got.is_additive is False
    # A `Mapping[str, float]` would have carried two entries out of this reward: `correct` and
    # `style`. It could not have carried the override at all, which is the whole argument.
    assert got.weights_dict_components == 2


def test_a_plain_weighted_sum_is_a_weights_dict_and_the_summary_says_so() -> None:
    got = summarise([t for t in plain_group().trees])
    assert got.n_inexpressible == 0
    assert got.is_additive is True
    assert got.weights_dict_components == 2
    assert "would have lost nothing" in got.says()


def test_the_ablation_deltas_are_hand_computable_and_do_not_sum_to_the_total() -> None:
    """Inside the override region every term below it is worth exactly nothing.

    `correct` is worth 1.0 on rollout 0 and nothing on the other three: rollout 2 has correct = 0,
    and rollouts 1 and 3 are pinned to the constant, so removing the term changes nothing. The
    average is 1.0/4.
    """
    g = worked_group()
    got = summarise(list(g.trees), list(g.contexts))
    assert got.ablation["correct"] == pytest.approx(0.25)
    # `style` is worth 0.25*0.8 = 0.2 on rollout 0 and 0.25*0.6 = 0.15 on rollout 2, both over four.
    assert got.ablation["style"] == pytest.approx(0.35 / 4)
    # And the task reward itself is worth exactly nothing where the override binds, which is the
    # only place removing it leaves a score at all.
    assert got.ablation["task"] == pytest.approx(0.0)
    assert sum(got.ablation.values()) != pytest.approx(evaluate(g.trees[0], g.contexts[0]))


def test_every_primitive_a_weights_dict_cannot_express_is_counted_and_named() -> None:
    tree = WeightedSum(
        "total",
        (
            Override("ov", OVER_BUDGET, -1.0, Leaf("a", 1.0, call())),
            Mask("masked", Leaf("b", 0.5, call())),
            Product("gate", (Leaf("c", 1.0, call()),)),
        ),
        (1.0, 1.0, 1.0),
    )
    got = summarise([tree])
    assert got.inexpressible == ("Mask", "Override", "Product")
    assert got.n_inexpressible == 3
    assert got.is_additive is False


# ---------------------------------------------------------------------------
# B3, part two: the counterfactual
# ---------------------------------------------------------------------------


def test_the_counterfactual_reports_the_advantage_scale_numbers_and_the_score_scale_baseline() -> (
    None
):
    got = CounterfactualComposition([worked_group()], {"length_override"}).compute()
    assert not isinstance(got, Refusal), got
    assert got.result.scores_before == (1.2, -1.0, 0.15, -1.0)
    assert got.result.scores_after == pytest.approx((1.2, 1.1, 0.15, 0.05))
    assert got.result.advantages_before == pytest.approx((1.3625, -0.8375, 0.3125, -0.8375))
    assert got.result.advantages_after == pytest.approx((0.575, 0.475, -0.475, -0.575))
    # Advantage scale: the shifts are 0.7875, 1.3125, 0.7875, 0.2625 against a standard deviation
    # of 0.91609, so exactly one clears it.
    assert got.result.n_moved == 1
    assert got.result.fraction_moved == pytest.approx(0.25)
    # Score scale: the shifts are 0, 2.1, 0, 1.05 against the same standard deviation, because
    # centring subtracts a constant and does not change the spread. Two clear it.
    assert got.score_scale_fraction_moved == pytest.approx(0.5)
    assert got.score_scale_sign_reversed == 2
    assert got.score_sd == pytest.approx(0.9160888330287625)
    # So half of what removing the override did to the scores was absorbed by group centring and
    # never reached the update, and reporting only the headline would have hidden that.
    assert got.score_scale_fraction_moved == 2 * got.result.fraction_moved


def test_the_interpretation_names_both_scales_and_the_leak() -> None:
    got = CounterfactualComposition([worked_group()], {"length_override"}).compute()
    text = got.interpretation()
    assert "removing length_override changes 25% of advantages" in text
    assert "before the estimator" in text
    assert "50% of scores move" in text
    assert "reaches only 2 of 4 rollouts" in text
    assert "reading the level of the reward and not only the contrast" in text


def test_the_null_leak_is_zero_on_a_composition_with_no_override() -> None:
    got = null_leak([plain_group()], constant=1.0)
    assert not isinstance(got, Refusal), got
    assert got.node == "task"
    assert got.n == 4 and got.n_reached == 4
    assert got.reach == 1.0
    assert got.cancels is True
    assert got.max_advantage_shift < 1e-12
    assert "cancels in group centring" in got.says()


def test_the_null_leak_measures_exactly_what_the_override_blocks() -> None:
    """Two of four rollouts are pinned, so the constant reaches half the group and half of it
    survives centring.

    By hand: adding 1.0 at the task reward takes the scores from (1.2, -1, 0.15, -1) to
    (2.2, -1, 1.15, -1). The mean moves from -0.1625 to 0.3375, so it moves by 0.5, and every
    advantage moves by either +0.5 (the two the constant reached) or -0.5 (the two it did not).
    """
    got = null_leak([worked_group()], constant=1.0)
    assert not isinstance(got, Refusal), got
    assert got.n_reached == 2 and got.n == 4
    assert got.reach == 0.5
    assert got.cancels is False
    assert got.max_advantage_shift == pytest.approx(0.5)
    assert got.per_group_reach == (0.5,)


def test_the_leak_is_linear_in_the_constant_for_a_mean_centred_estimator() -> None:
    """Which is why the constant is reported beside the number instead of divided out of it."""
    one = null_leak([worked_group()], constant=1.0)
    ten = null_leak([worked_group()], constant=10.0)
    assert ten.max_advantage_shift == pytest.approx(10.0 * one.max_advantage_shift)
    assert ten.reach == one.reach


def test_the_default_offset_node_is_the_task_reward_under_the_override() -> None:
    assert default_offset_node(k3_tree(1.0, 0.8)) == "task"
    assert default_offset_node(Leaf("only", 1.0, call())) == "only"


def test_offsetting_a_leaf_moves_the_score_by_the_leaf_s_weight_times_the_constant() -> None:
    tree = task_tree(1.0, 0.8)
    ctx = k3_ctx(1.0, 100.0)
    assert evaluate(tree, ctx) == pytest.approx(1.2)
    assert evaluate(offset_node(tree, "style", 1.0), ctx) == pytest.approx(1.2 + 0.25)
    assert evaluate(offset_node(tree, "task", 1.0), ctx) == pytest.approx(2.2)


def test_offsetting_an_abstaining_leaf_does_not_give_it_a_value() -> None:
    tree = task_tree(None, None, failed=True)
    shifted = offset_node(tree, "correct", 5.0)
    assert shifted.children[0].value is None
    assert shifted.children[0].abstained is True


def test_offsetting_a_node_that_has_no_place_for_a_constant_raises() -> None:
    with pytest.raises(TypeError, match="no unambiguous place"):
        offset_node(k3_tree(1.0, 0.8), "length_override", 1.0)


# ---------------------------------------------------------------------------
# B4: the census, and what the substituted numbers cost
# ---------------------------------------------------------------------------


def failing_group(estimator: EstimatorSpec = CENTRED) -> GroupScores:
    """The worked rollouts with rollout 2's grader raising and 0.0 written in its place.

    That is `verifiers`' shape exactly: `except Exception: ans = 0.0`, with both facts recorded.
    """
    trees = []
    for i, (c, s, _) in enumerate(ROWS):
        if i == 2:
            trees.append(task_tree(0.0, 0.0, failed=True))
        else:
            trees.append(task_tree(c, s))
    return GroupScores(
        trees=tuple(trees),
        contexts=tuple(k3_ctx(c, t) for c, _, t in ROWS),
        estimator=estimator,
        id="g0",
    )


def test_the_census_separates_the_abstention_from_the_silent_zero() -> None:
    g = failing_group()
    got = read_census(list(g.trees), list(g.contexts), [g])
    assert not isinstance(got, Refusal), got
    assert got.n_leaves == 8
    assert got.n_abstained == 2  # one failed call, two leaves
    assert got.n_silent_zero == 2  # both carry the substituted 0.0
    assert got.abstention_rate == pytest.approx(0.25)
    assert got.silent_zero_rate == pytest.approx(0.25)
    assert got.n_unattributable == 0
    assert got.substituted_total == 0.0
    assert got.n_boundary_failures == 2
    assert got.outcomes == {"returned": 6, "raised": 2}
    assert got.channel_observed is True


def test_a_working_abstention_channel_has_a_silent_zero_rate_of_zero_and_an_abstention_rate() -> (
    None
):
    """TRL's shape: `None` maps to NaN and is excluded, so nothing was substituted."""
    trees = [task_tree(1.0, 0.8), task_tree(None, None, failed=True)]
    got = read_census(trees, None, ())
    assert not isinstance(got, Refusal), got
    assert got.n_abstained == 2
    assert got.n_silent_zero == 0
    assert got.abstention_rate == pytest.approx(0.5)
    assert got.silent_zero_rate == 0.0
    # And the two are not the same number, which is the whole reason E29 registered the second one.
    assert got.abstention_rate != got.silent_zero_rate


def test_the_baseline_shift_is_the_difference_between_the_two_group_means() -> None:
    """Hand-computed. The task reward is `correct + 0.25 * style`, so the scores the framework used
    are 1.20, 1.10, 0.00 and 0.05, and its group mean is 2.35/4 = 0.5875. The mean over the three
    rollouts that were actually scored is 2.35/3 = 0.783333. The framework's baseline is therefore
    0.195833 too low, and every rollout whose own grader worked had its advantage inflated by
    exactly that.
    """
    (shift,) = baseline_shifts([failing_group()])
    assert shift.k == 4
    assert shift.n_scored == 3
    assert shift.n_silent_zero == 2
    assert shift.mean_as_used == pytest.approx(2.35 / 4)
    assert shift.mean_honest == pytest.approx(2.35 / 3)
    assert shift.shift == pytest.approx(2.35 / 4 - 2.35 / 3)
    assert shift.shift == pytest.approx(-0.1958333333)
    assert shift.contaminated is True
    assert shift.additive_only is True  # this estimator does not divide by the group std


def test_the_consequence_names_the_mechanism_and_the_number() -> None:
    g = failing_group()
    got = read_census(list(g.trees), list(g.contexts), [g])
    text = got.consequence()
    assert "advantage = reward_i - mean(rewards)" in text
    assert "1 of 1 groups had their mean moved" in text
    assert "purely additive" in text
    assert "rubric.py:406-409" in text
    assert f"{got.max_abs_shift:.4g}" in text


def test_the_consequence_says_what_a_z_scoring_estimator_changes() -> None:
    spec = EstimatorSpec(
        family="grpo",
        group_centred=True,
        std_normalised=True,
        std_epsilon=1e-8,
        degenerate_policy="zero",
    )
    g = failing_group(spec)
    got = read_census(list(g.trees), list(g.contexts), [g])
    assert "divides by the group standard deviation" in got.consequence()
    assert "rescales" in got.consequence()


def test_a_record_with_no_failures_reports_zero_as_a_lower_bound_and_says_so() -> None:
    """The `is_equiv` case: a grader that catches its own exception never reaches a wrapper."""
    got = read_census([task_tree(1.0, 0.8), task_tree(0.0, 0.6)], None, ())
    assert got.silent_zero_rate == 0.0
    assert got.channel_observed is False
    assert "lower bound rather than a measurement" in got.says()
    assert "never reaches a wrapper" in got.says()
    assert "bare `except: return str1 == str2`" in got.limitation()


def test_unattributable_leaves_are_reported_as_a_bound_and_not_folded_into_the_successes() -> None:
    trees = [
        task_tree(1.0, 0.8),
        WeightedSum("task", (Leaf("correct", 0.0, None), Leaf("style", 0.0, None)), (1.0, 0.25)),
    ]
    got = read_census(trees, None, ())
    assert got.n_leaves == 4
    assert got.n_unattributable == 2
    assert got.n_known == 2
    assert got.silent_zero_rate == 0.0
    # Assuming every unknown outcome was a failure is the other end of the interval, and reporting
    # only the point estimate would be biased downward by exactly the reassuring amount.
    assert got.silent_zero_rate_upper == pytest.approx(0.5)
    assert "carry no call record at all" in got.limitation()

    inst = SilentZeroRate(trees)
    unc = inst.uncertainty(got)
    assert unc is not None
    assert unc.ci_low == pytest.approx(0.0) and unc.ci_high == pytest.approx(0.5)
    assert unc.method == "unattributable-bounds"


def test_reconstructing_the_arm_the_framework_ran_leaves_a_clean_tree_alone() -> None:
    clean = task_tree(1.0, 0.8)
    assert as_the_framework_used_it(clean) == clean
    dirty = task_tree(0.0, 0.0, failed=True)
    assert math.isnan(evaluate(dirty, k3_ctx(0.0, 100.0)))
    assert evaluate(as_the_framework_used_it(dirty), k3_ctx(0.0, 100.0)) == pytest.approx(0.0)


def test_the_census_reports_leaves_a_binding_override_shadowed() -> None:
    """A grader failure under a binding override was never used, so counting it either way is
    wrong."""
    g = GroupScores(
        trees=(k3_tree(0.0, 0.0, failed=True), k3_tree(1.0, 0.8)),
        contexts=(k3_ctx(0.0, 400.0), k3_ctx(1.0, 100.0)),
        estimator=CENTRED,
        id="g0",
    )
    got = read_census(list(g.trees), list(g.contexts), [g])
    assert got.n_shadowed == 2
    assert "never used the score" in got.limitation()


# ---------------------------------------------------------------------------
# Refusals: the reason and the remedy, per instrument
# ---------------------------------------------------------------------------


def test_the_counterfactual_passes_the_substrate_s_refusal_through_unchanged() -> None:
    """Reused rather than restated: the reason, the detail and the remedy are the score tree's own.

    Asserted against a direct call rather than against a hard-coded reason, because the reason
    belongs to `record.scores` and an instrument that quietly reworded it would be inventing a
    second answer to a question the substrate already answers.
    """
    from reward_lens.record.scores import counterfactual as substrate

    g = GroupScores(
        trees=(k3_tree(1.0, 0.8), None),
        contexts=(k3_ctx(1.0, 100.0), k3_ctx(1.0, 200.0)),
        estimator=CENTRED,
    )
    direct = substrate([g], {"length_override"})
    got = CounterfactualComposition([g], {"length_override"}).compute()
    assert isinstance(got, Refusal) and isinstance(direct, Refusal)
    assert got.reason is direct.reason
    assert got.detail == direct.detail and got.remedy == direct.remedy
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert "1 of 2 trajectories carry no score tree" in got.detail
    assert "instrument the grader with reward_lens.tap" in got.remedy
    assert "cannot be recovered from a total" in got.remedy


def test_the_counterfactual_refuses_an_estimator_it_cannot_replay() -> None:
    from reward_lens.record.scores import counterfactual as substrate

    spec = EstimatorSpec(family="ppo", group_centred=False)
    direct = substrate([worked_group(spec)], {"length_override"})
    got = CounterfactualComposition([worked_group(spec)], {"length_override"}).compute()
    assert isinstance(got, Refusal) and isinstance(direct, Refusal)
    assert got.reason is direct.reason
    assert "not group-relative" in got.detail
    assert "record the per-rollout value-function baseline" in got.remedy


def test_the_counterfactual_refuses_when_nothing_was_named_for_removal() -> None:
    got = CounterfactualComposition([worked_group()], set()).compute()
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "is the baseline arm reported against itself" in got.detail
    assert "disabled={'length_override'}" in got.remedy


def test_the_counterfactual_refuses_an_empty_record_rather_than_returning_nothing() -> None:
    got = CounterfactualComposition([], {"length_override"}).compute()
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "no recorded composition to re-evaluate" in got.detail
    assert "record.scores.group_scores" in got.remedy


def test_naming_a_node_no_tree_carries_raises_rather_than_refusing() -> None:
    """A mis-specified measurement is a call error, not a limit of the data.

    The score tree raises here on purpose and the exception carries the inventory, so the reply to "removing
    kl_penalty changed nothing" is a list of the nodes this record actually has. A refusal would
    read as a fact about the run.
    """
    from reward_lens.record.scores import UnknownNode

    with pytest.raises(UnknownNode, match="false negative"):
        CounterfactualComposition([worked_group()], {"kl_penalty"}).compute()


def test_the_composition_tree_refuses_a_record_that_kept_only_totals() -> None:
    got = CompositionTree([None, None, None]).compute()
    assert isinstance(got, Refusal)
    # The access was sufficient and the field is not there, which is what the sixteenth reason is
    # for (E30). "Instrument your grader" and "ask for more access" are different
    # remedies and this is the first of the two.
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert "kept a total and not a composition" in got.detail
    assert "B3's kill condition" in got.remedy
    assert got.statistics == {"n_trees": 3, "n_with_composition": 0}


def test_the_composition_tree_refuses_an_empty_record() -> None:
    got = CompositionTree([]).compute()
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "no composition to read" in got.detail
    assert "CompositionTree.over(groups)" in got.remedy


def test_the_census_refuses_when_no_leaf_carries_an_outcome() -> None:
    """A rate over an empty denominator would be 0.0, which is the reassuring answer."""
    trees = [WeightedSum("task", (Leaf("correct", 1.0, None),), (1.0,))]
    for inst in (SilentZeroRate(trees), AbstentionRate(trees)):
        got = inst.compute()
        assert isinstance(got, Refusal)
        assert got.reason is RefusalReason.RECORD_INCOMPLETE
        assert "no outcome is knowable" in got.detail
        assert "would be the reassuring answer and it would be made up" in got.detail
        assert "instrument_grader" in got.remedy
        assert got.instrument == inst.name


def test_the_census_refuses_an_empty_record() -> None:
    got = SilentZeroRate([]).compute()
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "nothing to count" in got.detail
    assert "record.scores.group_scores" in got.remedy


def test_the_census_refuses_a_record_with_no_trees_at_all() -> None:
    got = SilentZeroRate([None, None]).compute()
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert "no grader calls to count" in got.detail
    assert "cannot say whether the grader that produced it succeeded" in got.remedy


# ---------------------------------------------------------------------------
# The generated invariance test, and the one place it fails on purpose
# ---------------------------------------------------------------------------

PAYLOAD = InvariancePayload(scores=np.array([1.0, 0.4, 0.9, 0.1]), group_ids=np.array([0, 0, 0, 0]))
TWO_GROUPS = InvariancePayload(
    scores=np.array([1.0, 0.4, 0.9, 0.1, 0.5, 0.7]), group_ids=np.array([0, 0, 0, 1, 1, 1])
)


def _plain(payload: InvariancePayload) -> float:
    return null_probe(payload, with_override=False)


def _with_override(payload: InvariancePayload) -> float:
    return null_probe(payload, with_override=True)


_plain.__name__ = "measure.composition.CounterfactualComposition"
_with_override.__name__ = "measure.composition.CounterfactualComposition[override]"


def test_the_counterfactual_advantage_is_invariant_under_a_per_prompt_constant() -> None:
    """`reward.null`, the group B3 declares. Generated, 32 draws, one group and then two."""
    report = check_invariance(_plain, "reward.null", PAYLOAD, n=32)
    assert report.passed, report.render()
    assert report.max_deviation < 1e-9
    two = check_invariance(_plain, "reward.null", TWO_GROUPS, n=32)
    assert two.passed, two.render()


def test_an_override_breaks_that_invariance_and_the_instrument_measures_by_how_much() -> None:
    """The failure is a measurement of the composition, not a defect in the instrument.

    A constant added to the task reward does not reach the rollouts the override pinned, so the
    group is no longer shifted by a constant, the shift does not cancel, and every advantage in the
    group moves. `null_leak` is what turns that from a failing test into a reported number, and the
    two agree: the reach is a half, so half of any constant survives centring.
    """
    report = check_invariance(_with_override, "reward.null", PAYLOAD, n=32)
    assert not report.passed
    assert report.max_deviation > 1.0
    worst = report.worst
    assert worst is not None
    # The kernel's `reward.null` draws sigma from LogUniform(0.1, 10) and the shift is exactly half
    # the drawn constant, so the deviation is bounded by the draw and never zero for a real one.
    assert abs(worst.transformed - worst.expected) == pytest.approx(report.max_deviation)

    groups = _payload_groups_for_test()
    leak = null_leak(groups, constant=1.0)
    assert leak.reach == 0.5
    assert leak.cancels is False
    assert leak.max_advantage_shift == pytest.approx(0.5)


def _payload_groups_for_test():
    from reward_lens.measure.composition.composition import (
        NULL_TEST_ESTIMATOR,
        _payload_groups,
    )

    return _payload_groups(PAYLOAD, with_override=True, estimator=NULL_TEST_ESTIMATOR)


def test_the_composition_summary_is_invariant_because_it_is_structural() -> None:
    """A weak test, and it is stated as one rather than presented as coverage.

    `reward.null` acts on values and this reading counts node types, so it could not have moved.
    The strong test of the same declaration is the counterfactual above, where the reading goes
    through the estimator.
    """

    def structural(payload: InvariancePayload) -> float:
        trees = [
            WeightedSum("task", (Leaf("component", float(s), None),), (1.0,))
            for s in np.asarray(payload.scores, dtype=float)
        ]
        return float(summarise(trees).n_inexpressible)

    structural.__name__ = "measure.composition.CompositionTree"
    report = check_invariance(structural, "reward.null", PAYLOAD, n=32)
    assert report.passed
    assert report.max_deviation == 0.0


def test_the_census_declares_the_trivial_group_and_its_generated_test_says_so() -> None:
    """`none` is an answer. Nothing acts on a count of grader failures, and the report says that
    rather than manufacturing a pass."""

    def rate(payload: InvariancePayload) -> float:
        trees = [task_tree(float(s), 0.0) for s in np.asarray(payload.scores, dtype=float)]
        out = read_census(trees, None, ())
        return float(out.silent_zero_rate)

    rate.__name__ = "measure.composition.SilentZeroRate"
    assert SilentZeroRate.invariance == TRIVIAL_GROUP
    assert AbstentionRate.invariance == TRIVIAL_GROUP
    report = check_invariance(rate, TRIVIAL_GROUP, PAYLOAD, n=32)
    assert report.passed
    assert "nothing acts on this quantity" in report.skipped


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

_finite = st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)


def _additive_group(values: list[float]) -> GroupScores:
    return GroupScores(
        trees=tuple(WeightedSum("task", (Leaf("component", v, None),), (1.0,)) for v in values),
        contexts=tuple(ScoreContext() for _ in values),
        estimator=CENTRED,
        id="g",
    )


@given(values=st.lists(_finite, min_size=2, max_size=8), c=_finite)
@settings(max_examples=200, deadline=None)
def test_a_constant_added_to_an_additive_composition_leaves_every_advantage_alone(
    values: list[float], c: float
) -> None:
    """`reward.null` as a property over random data, beside the 32 generated draws.

    The reach is asserted as 0 or n rather than as n, and the difference is not pedantry: a
    constant of zero reaches nothing, and a constant small enough to vanish into the last bit of a
    score it is added to reaches nothing either. Both are cases where the advantages are unchanged
    for a reason, and `cancels` is the property that covers all three.
    """
    got = null_leak([_additive_group(values)], constant=c)
    assert not isinstance(got, Refusal)
    assert got.cancels or not got.resolved
    if got.resolved:
        assert got.n_reached == len(values)
    assert got.max_advantage_shift < 1e-8 * max(1.0, abs(c))


def test_a_constant_of_zero_reaches_nothing_and_that_is_not_a_leak() -> None:
    got = null_leak([_additive_group([1.0, 2.0, 3.0])], constant=0.0)
    assert got.n_reached == 0
    assert got.cancels is True
    assert got.resolved is False  # a constant of zero resolves nothing, and the reading says so
    assert got.max_advantage_shift == 0.0
    assert "below the resolution of the scores" in got.says()


def test_a_constant_too_small_to_register_is_not_a_leak_either() -> None:
    """Found by the property test above, and it is a real defect in the first implementation.

    With exact inequality, a constant of 1e-205 added to scores of 0.0 and 1.0 moves the first and
    vanishes into the last bit of the second, so a perfectly additive composition reported a reach
    of one half and a leak that does not exist. The reach is measured against the kernel's own
    `default_tol`, which is the same tolerance the generated invariance test uses.
    """
    got = null_leak([_additive_group([0.0, 1.0])], constant=1e-205)
    assert got.n_reached == 0
    assert got.cancels is True
    assert got.resolved is False
    assert got.max_advantage_shift == 0.0


@given(
    values=st.lists(_finite, min_size=2, max_size=8),
    over=st.lists(st.booleans(), min_size=2, max_size=8),
    c=st.floats(min_value=0.1, max_value=20.0),
)
@settings(max_examples=200, deadline=None)
def test_the_leak_matches_the_closed_form_the_mechanism_implies(
    values: list[float], over: list[bool], c: float
) -> None:
    """A constant reaching m of n rollouts moves an advantage by `c * max(m, n-m) / n`.

    Derived once and then checked against the measurement rather than substituted for it, because
    the closed form only holds for an estimator that does not divide by the group standard
    deviation and the measurement holds either way.

    It also only holds where there are advantages on both sides of the offset to difference. The
    closed form is arithmetic on the centring, and `CENTRED` records `degenerate_policy="skip"`,
    so a group whose scores are all equal has no advantages at all. When the constant moves the
    group across that boundary the difference is not small, it does not exist, and the reading says
    so in `n_advantage_state_changed` rather than in the shift. The draw pinned in
    `test_a_constant_that_breaks_a_tie_is_a_state_change_and_not_a_shift` is that case.
    """
    n = min(len(values), len(over))
    assume(n >= 2)
    values, over = values[:n], over[:n]
    trees = tuple(k3_tree(v, 0.0) for v in values)
    ctxs = tuple(k3_ctx(v, 400.0 if o else 100.0) for v, o in zip(values, over))
    group = GroupScores(trees=trees, contexts=ctxs, estimator=CENTRED, id="g")
    got = null_leak([group], constant=c)
    assert not isinstance(got, Refusal)
    m = sum(1 for o in over if o)  # pinned by the override, so the constant never reaches them
    reached = n - m
    assert got.n_reached == reached

    # Every rollout is accounted for exactly once: comparable on both sides, or crossing the
    # degeneracy boundary, or carrying no advantage on either side. That partition is what stops a
    # crossing being absorbed into the shift as a zero.
    assert got.n_comparable + got.n_advantage_state_changed <= n

    if got.n_advantage_state_changed:
        # The constant changed whether the trainer would use this group. There is no per-rollout
        # difference to compare the closed form against, and `cancels` must not claim exactness.
        assert got.cancels is False
        assert m not in (0, n)
    elif got.n_comparable == n:
        expected = 0.0 if m in (0, n) else c * max(m, reached) / n
        assert got.max_advantage_shift == pytest.approx(expected, abs=1e-9, rel=1e-9)
        assert got.cancels is (m in (0, n))
    else:
        # The remaining case is a group with no advantages on either side: every score identical
        # before and identical after, so `degenerate_policy="skip"` drops it in both arms. There is
        # nothing to compare the closed form against and nothing crossed the boundary either.
        assert got.n_comparable == 0
        assert got.max_advantage_shift == 0.0


def test_a_constant_that_breaks_a_tie_is_a_state_change_and_not_a_shift() -> None:
    """Found by the property test above on `values=[0.0, -1.0], over=[True, False], c=1.0`.

    The draw is pinned here rather than left in `.hypothesis/`, which is gitignored, so the case
    survives a fresh checkout.

    Both rollouts score exactly -1.0 before the offset: the first is pinned at the override's
    constant because it is over budget, and the second's task tree happens to evaluate to the same
    number. `CENTRED` records `degenerate_policy="skip"`, so that group is dropped and neither
    rollout has an advantage. The constant reaches the second rollout only, the scores become
    [-1.0, 0.0], and the advantages become [-0.5, +0.5].

    That is the largest thing a per-prompt constant can do to a group: it took one the trainer would
    have skipped and gave it a full contrast. The first implementation required an advantage on both
    sides before it would difference them, so this contributed nothing and the reading came back
    `max_advantage_shift = 0.0`, which is what a perfectly cancelling composition also reports. The
    fix does not invent a shift against advantages the trainer never computed; it counts the
    crossing separately and refuses to call the composition cancelling.
    """
    values, over, c = [0.0, -1.0], [True, False], 1.0
    trees = tuple(k3_tree(v, 0.0) for v in values)
    ctxs = tuple(k3_ctx(v, 400.0 if o else 100.0) for v, o in zip(values, over))
    group = GroupScores(trees=trees, contexts=ctxs, estimator=CENTRED, id="g")

    # The scores that produce it, asserted so the case cannot drift out from under the test.
    node = default_offset_node(trees[0])
    before = [evaluate(t, cx) for t, cx in zip(trees, ctxs)]
    after = [evaluate(offset_node(t, node, c), cx) for t, cx in zip(trees, ctxs)]
    assert before == [-1.0, -1.0]
    assert after == [-1.0, 0.0]

    got = null_leak([group], constant=c)
    assert not isinstance(got, Refusal)
    assert got.n_reached == 1
    assert got.n_comparable == 0  # no rollout has an advantage on both sides
    assert got.n_advantage_state_changed == 2
    assert got.max_advantage_shift == 0.0
    assert got.cancels is False
    assert got.resolved is True  # not the tolerance path: the constant is 1.0
    assert "degeneracy boundary" in got.says()
    assert "not that nothing happened" in got.says()


def test_a_constant_that_creates_a_tie_deletes_the_signal_and_says_so() -> None:
    """The mirror of the draw above, and the one that matters more in practice.

    Here the advantages exist before the offset and not after: the constant makes both scores -1.0,
    the group becomes degenerate, and `degenerate_policy="skip"` drops it. A constant that deletes a
    group's entire learning signal reported `max_advantage_shift = 0.0` under the first
    implementation, for the same reason and with the sign of the mistake pointing the same way.
    """
    values, over, c = [0.0, -2.0], [True, False], 1.0
    trees = tuple(k3_tree(v, 0.0) for v in values)
    ctxs = tuple(k3_ctx(v, 400.0 if o else 100.0) for v, o in zip(values, over))
    node = default_offset_node(trees[0])
    before = [evaluate(t, cx) for t, cx in zip(trees, ctxs)]
    after = [evaluate(offset_node(t, node, c), cx) for t, cx in zip(trees, ctxs)]
    assert before == [-1.0, -2.0]
    assert after == [-1.0, -1.0]

    got = null_leak(
        [GroupScores(trees=trees, contexts=ctxs, estimator=CENTRED, id="g")], constant=c
    )
    assert not isinstance(got, Refusal)
    assert got.n_comparable == 0
    assert got.n_advantage_state_changed == 2
    assert got.cancels is False
    assert "degeneracy boundary" in got.says()


def test_a_degeneracy_verdict_that_flips_on_one_ulp_is_not_a_state_change() -> None:
    """The third draw the property test found, and the reason the crossing carries a tolerance.

    `values=[7.665180039911412] * 3, c=14.0`. Every score is identical before the offset and
    identical after it, so the group is degenerate on both sides by any reading a person would give.
    `_is_degenerate` compares `np.std` against the estimator's epsilon, which is zero here because
    the record sets none, so the verdict is decided by whether the standard deviation lands on
    exactly 0.0. Before the offset the mean of three copies is exact and it does. After, the mean is
    one ulp high, the centred values are -3.552713678800501e-15 each, the standard deviation is that
    same number, and the group is judged live.

    Counting that as a crossing would report a constant of 14 as having handed a skipped group a
    contrast of 3.55e-15. The guard is the kernel's `default_tol`, which at a score of 21.67 is
    2.17e-6, and the same tolerance the reach is measured against. `cancels` stays True: the
    constant reached all three rollouts and it does cancel.
    """
    v, c = 7.665180039911412, 14.0
    group = _additive_group([v, v, v])
    assert float(np.std(np.asarray([v, v, v]))) == 0.0
    assert float(np.std(np.asarray([v + c, v + c, v + c]))) == pytest.approx(3.552713678800501e-15)

    got = null_leak([group], constant=c)
    assert not isinstance(got, Refusal)
    assert got.n_reached == 3
    assert got.n_advantage_state_changed == 0
    assert got.cancels is True
    assert got.max_advantage_shift == 0.0


@given(
    n=st.integers(min_value=1, max_value=12),
    failures=st.lists(st.booleans(), min_size=1, max_size=12),
    substitute=st.booleans(),
)
@settings(max_examples=200, deadline=None)
def test_the_silent_zeros_are_a_subset_of_the_abstentions_always(
    n: int, failures: list[bool], substitute: bool
) -> None:
    failures = (failures * n)[:n]
    trees = [
        task_tree(0.0 if substitute else None, 0.0 if substitute else None, failed=True)
        if f
        else task_tree(1.0, 0.5)
        for f in failures
    ]
    got = read_census(trees, None, ())
    assert not isinstance(got, Refusal)
    assert 0 <= got.n_silent_zero <= got.n_abstained <= got.n_leaves
    assert got.n_abstained == 2 * sum(failures)
    assert got.n_silent_zero == (2 * sum(failures) if substitute else 0)
    assert 0.0 <= got.silent_zero_rate <= got.abstention_rate <= 1.0


@given(values=st.lists(_finite, min_size=1, max_size=8))
@settings(max_examples=100, deadline=None)
def test_the_summary_and_is_additive_agree_on_every_tree(values: list[float]) -> None:
    trees = [
        WeightedSum("task", (Leaf("a", v, None), Leaf("b", 0.5, None)), (1.0, 0.25)) for v in values
    ]
    plain = summarise(trees)
    assert plain.is_additive is (plain.n_inexpressible == 0)
    gated = summarise([k3_tree(v, 0.0) for v in values])
    assert gated.is_additive is False
    assert gated.n_inexpressible == 1


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------


def test_all_four_instruments_are_lint_clean_and_estimate_a_registered_quantity() -> None:
    instruments = [
        CompositionTree.over([worked_group()]),
        CounterfactualComposition([worked_group()], {"length_override"}),
        SilentZeroRate.over([failing_group()]),
        AbstentionRate.over([failing_group()]),
    ]
    for inst in instruments:
        assert lint_instrument(inst) == [], inst.name
        assert inst.quantity in QUANTITIES
        assert inst.baselines
        assert inst.envelope is not None and inst.envelope.unconditional
    assert {i.quantity for i in instruments} == {
        "grader.composition",
        "grader.counterfactual_score",
        "grader.silent_zero_rate",
        "grader.abstention_rate",
    }


def test_estimate_returns_evidence_carrying_the_baseline_and_the_interpretation() -> None:
    ev = CounterfactualComposition([worked_group()], {"length_override"}).estimate(Context())
    assert not isinstance(ev, Refusal), ev
    assert ev.value["fraction_moved"] == pytest.approx(0.25)
    assert ev.value["baselines"]["baseline.deployed_score_scale"] == pytest.approx(0.5)
    assert "before the estimator" in ev.value["interpretation"]

    g = failing_group()
    ev2 = SilentZeroRate.over([g]).estimate(Context())
    assert ev2.value["value"] == pytest.approx(0.25)
    assert ev2.value["reports"] == "silent_zero"
    assert "advantage = reward_i - mean(rewards)" in ev2.value["interpretation"]
    ev3 = AbstentionRate.over([g]).estimate(Context())
    assert ev3.value["reports"] == "abstention"
    assert ev3.value["value"] == pytest.approx(0.25)


def test_estimate_returns_the_refusal_as_a_value_rather_than_raising() -> None:
    out = CounterfactualComposition([], {"x"}).estimate(Context())
    assert isinstance(out, Refusal)
    assert "ACCESS_INSUFFICIENT" in out.render()
    with pytest.raises(ValueError, match="declines to produce"):
        CounterfactualComposition([], {"x"}).measure(Context())

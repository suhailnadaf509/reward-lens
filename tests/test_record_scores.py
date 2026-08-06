"""The score tree, its eight nodes, and the counterfactual that reads them.

Every expected number in this file was computed by hand from the composition and is written out in
the test that asserts it, so a change in the implementation cannot quietly move the target. The
worked group is the same one the acceptance test uses, and it is small enough to check on paper:
four rollouts, a task reward of `correct + 0.25 * style`, and a Kimi-K3-shaped override that pins
the score to -1 when a rollout's token count exceeds 1.5 times the task's budget.

    rollout   correct  style  tokens  b0    over budget   score
    0         1.0      0.8    100     100   no            1.20
    1         1.0      0.4    200     100   yes          -1.00   (would be 1.10)
    2         0.0      0.6     90     100   no            0.15
    3         0.0      0.2    400     100   yes          -1.00   (would be 0.05)
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from reward_lens.core.invariance import InvariancePayload, check_invariance
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.record.schema import EstimatorSpec, decode_foreign, encode_foreign
from reward_lens.record.scores import (
    AllAbstainedWarning,
    DuplicateNodeName,
    GraderCallRef,
    GroupContext,
    GroupGate,
    GroupScores,
    Leaf,
    Lexicographic,
    Mask,
    MissingFeature,
    MissingGroupStatistic,
    Override,
    Piecewise,
    PredicateRef,
    Product,
    ScoreContext,
    UnknownNode,
    WeightedSum,
    ablation_deltas,
    census,
    counterfactual,
    deadzone,
    evaluate,
    find,
    is_additive,
    leaves,
    node_names,
    replay_advantages,
    walk,
)

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
    return GraderCallRef(grader=grader, outcome=outcome, latency_s=0.012, cost_usd=0.0004)


def task_tree(correct: float, style: float) -> WeightedSum:
    return WeightedSum(
        name="task",
        children=(Leaf("correct", correct, call("verifier")), Leaf("style", style, call("judge"))),
        weights=(1.0, 0.25),
    )


def k3_tree(correct: float, style: float) -> Override:
    return Override(
        name="length_override",
        condition=OVER_BUDGET,
        constant=-1.0,
        otherwise=task_tree(correct, style),
    )


def k3_ctx(correct: float, tokens: float, *, entropy: float = 1.0) -> ScoreContext:
    return ScoreContext(
        features={"tokens": tokens, "b0": 100.0, "correct": correct, "entropy": entropy},
        group=GroupContext(accuracy=0.5, k=4, id="g0"),
    )


def worked_group(estimator: EstimatorSpec = CENTRED) -> GroupScores:
    return GroupScores(
        trees=tuple(k3_tree(c, s) for c, s, _ in ROWS),
        contexts=tuple(k3_ctx(c, t) for c, _, t in ROWS),
        estimator=estimator,
        id="g0",
    )


# ---------------------------------------------------------------------------
# Construction: what the types refuse to let you build
# ---------------------------------------------------------------------------


def test_a_leaf_with_no_value_must_say_it_abstained() -> None:
    with pytest.raises(ValueError, match="abstained=True"):
        Leaf("correct", None, call(), abstained=False)


def test_a_leaf_keeps_both_the_abstention_and_the_number_the_framework_substituted() -> None:
    """The `verifiers` shape: the call raised and 0.0 went into the group anyway."""
    leaf = Leaf("correct", 0.0, call("verifier", outcome="raised"), abstained=True)
    assert leaf.silent_zero is True
    assert leaf.substituted == 0.0
    # And it still contributes nothing, whatever number the framework wrote next to it.
    assert math.isnan(evaluate(leaf, k3_ctx(1.0, 10.0)))


def test_a_non_finite_value_must_also_declare_the_abstention() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        Leaf("correct", float("nan"), call())


def test_two_nodes_with_one_name_are_refused_at_construction() -> None:
    with pytest.raises(DuplicateNodeName, match="rubric_a"):
        WeightedSum(
            "total",
            children=(
                WeightedSum("a", (Leaf("format", 1.0, None),)),
                WeightedSum("b", (Leaf("format", 0.0, None),)),
            ),
        )


def test_weights_must_match_children() -> None:
    with pytest.raises(ValueError, match="2 weights for 3 children"):
        WeightedSum(
            "t", (Leaf("a", 1.0, None), Leaf("b", 1.0, None), Leaf("c", 1.0, None)), (1.0, 1.0)
        )


def test_an_empty_product_is_refused() -> None:
    with pytest.raises(ValueError, match="empty product"):
        Product("gates", ())


def test_a_group_gate_with_no_condition_is_refused() -> None:
    with pytest.raises(ValueError, match="never fires"):
        GroupGate("wpsm", Leaf("a", 1.0, None))


def test_piecewise_needs_two_strictly_increasing_knots() -> None:
    with pytest.raises(ValueError, match="at least two knots"):
        Piecewise("anchor", Leaf("a", 1.0, None), ((0.0, 0.0),))
    with pytest.raises(ValueError, match="strictly increase"):
        Piecewise("anchor", Leaf("a", 1.0, None), ((0.0, 0.0), (0.0, 1.0)))


def test_a_predicate_needs_exactly_one_right_hand_side() -> None:
    with pytest.raises(ValueError, match="against nothing"):
        PredicateRef("p", feature="tokens", op=">")
    with pytest.raises(ValueError, match="both a constant threshold"):
        PredicateRef("p", feature="tokens", op=">", threshold=1.0, threshold_feature="b0")


def test_a_predicate_renders_its_own_expression() -> None:
    assert OVER_BUDGET.expr == "tokens > 1.5 * b0"
    assert PredicateRef("fabricated").expr == "recorded(fabricated)"


# ---------------------------------------------------------------------------
# Evaluation, against values computed on paper
# ---------------------------------------------------------------------------


def test_the_worked_group_evaluates_to_the_four_scores_in_the_table() -> None:
    got = [evaluate(k3_tree(c, s), k3_ctx(c, t)) for c, s, t in ROWS]
    assert got == [1.2, -1.0, 0.15, -1.0]


def test_disabling_the_override_recovers_the_task_reward() -> None:
    got = [evaluate(k3_tree(c, s), k3_ctx(c, t), {"length_override"}) for c, s, t in ROWS]
    assert got == pytest.approx([1.2, 1.1, 0.15, 0.05])


def test_a_product_gates_multiplicatively() -> None:
    tree = Product("total", (Leaf("task", 0.8, None), Leaf("format_gate", 0.0, None)))
    assert evaluate(tree, k3_ctx(1.0, 10.0)) == 0.0
    # Removing the gate is the counterfactual "what if format had not been mandatory".
    assert evaluate(tree, k3_ctx(1.0, 10.0), {"format_gate"}) == 0.8


def test_lexicographic_reproduces_the_ling_rule() -> None:
    """`R_length = p(l)` if correct, `min(p(l), 0)` if incorrect."""

    def tree(correct: float, p: float) -> Lexicographic:
        return Lexicographic(
            name="length_tier",
            primary=Leaf("correct", correct, None),
            secondary=Leaf("p_length", p, None),
            threshold=1.0,
            cap=0.0,
            include_primary=False,
        )

    ctx = k3_ctx(1.0, 10.0)
    assert evaluate(tree(1.0, 0.3), ctx) == 0.3  # correct: the bonus lands
    assert evaluate(tree(0.0, 0.3), ctx) == 0.0  # incorrect: it cannot help
    assert evaluate(tree(0.0, -0.4), ctx) == -0.4  # incorrect: it can still hurt
    # Ungated, the bonus lands whether or not the answer was right.
    assert evaluate(tree(0.0, 0.3), ctx, {"length_tier"}) == 0.3


def test_piecewise_anchoring_interpolates_and_clamps() -> None:
    knots = ((0.0, -1.0), (0.5, 0.0), (1.0, 1.0))
    tree = Piecewise("anchor", Leaf("raw", 0.75, None), knots)
    # 0.75 is halfway along the segment from (0.5, 0.0) to (1.0, 1.0).
    assert evaluate(tree, k3_ctx(1.0, 10.0)) == 0.5
    assert evaluate(Piecewise("a", Leaf("raw", 2.0, None), knots), k3_ctx(1.0, 10.0)) == 1.0
    assert (
        evaluate(Piecewise("a", Leaf("raw", 2.0, None), knots, "linear"), k3_ctx(1.0, 10.0)) == 3.0
    )
    # Disabling the anchoring passes the raw score through.
    assert evaluate(tree, k3_ctx(1.0, 10.0), {"anchor"}) == 0.75


def test_a_group_gate_masks_on_group_accuracy_and_sample_entropy() -> None:
    """ERNIE's WPSM: masked when group accuracy exceeds tau *and* sample entropy falls below eta."""
    gate = GroupGate("wpsm", Leaf("task", 0.7, None), accuracy_above=0.8, entropy_below=0.2)
    hot = ScoreContext(features={"entropy": 0.5}, group=GroupContext(accuracy=0.9, k=8))
    cold = ScoreContext(features={"entropy": 0.1}, group=GroupContext(accuracy=0.9, k=8))
    easy = ScoreContext(features={"entropy": 0.1}, group=GroupContext(accuracy=0.5, k=8))
    assert evaluate(gate, hot) == 0.7  # entropy too high: not masked
    assert evaluate(gate, easy) == 0.7  # group not accurate enough: not masked
    assert math.isnan(evaluate(gate, cold))  # both conditions: masked
    assert evaluate(gate, cold, {"wpsm"}) == 0.7


def test_a_gate_in_zero_mode_writes_a_real_zero_into_the_group() -> None:
    gate = GroupGate("wpsm", Leaf("task", 0.7, None), accuracy_above=0.8, mode="zero")
    ctx = ScoreContext(features={}, group=GroupContext(accuracy=0.9, k=8))
    assert evaluate(gate, ctx) == 0.0


def test_a_gate_that_cannot_read_its_condition_raises_rather_than_never_firing() -> None:
    gate = GroupGate("wpsm", Leaf("task", 0.7, None), accuracy_above=0.8)
    with pytest.raises(MissingGroupStatistic, match="no accuracy"):
        evaluate(gate, ScoreContext(group=GroupContext(accuracy=None)))


def test_a_mask_removes_the_rollout_rather_than_zeroing_it() -> None:
    tree = Mask("overlong", Leaf("task", 0.9, None), when=PredicateRef("truncated"))
    on = ScoreContext(predicates={"truncated": True})
    off = ScoreContext(predicates={"truncated": False})
    assert math.isnan(evaluate(tree, on))
    assert evaluate(tree, off) == 0.9
    assert evaluate(tree, on, {"overlong"}) == 0.9


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------


def test_one_abstaining_term_makes_the_whole_sum_unknown() -> None:
    """Deliberately not TRL's `nansum`, which returns the rest and calls it the total."""
    tree = WeightedSum(
        "total",
        (Leaf("correct", 1.0, call()), Leaf("style", None, call("judge", "raised"), True)),
        (1.0, 0.25),
    )
    assert math.isnan(evaluate(tree, k3_ctx(1.0, 10.0)))
    # Removing the failing term is a different claim, and it is available.
    assert evaluate(tree, k3_ctx(1.0, 10.0), {"style"}) == 1.0


def test_a_zero_gate_does_not_rescue_an_abstention() -> None:
    tree = Product("total", (Leaf("gate", 0.0, call()), Leaf("judge", None, call(), True)))
    assert math.isnan(evaluate(tree, k3_ctx(1.0, 10.0)))


def test_an_override_hides_an_abstention_underneath_it() -> None:
    """The constant does not consult the child, so a failed grader in the region is invisible."""
    tree = Override(
        "length_override",
        OVER_BUDGET,
        -1.0,
        WeightedSum("task", (Leaf("correct", None, call("verifier", "raised"), True),)),
    )
    assert evaluate(tree, k3_ctx(1.0, 400.0)) == -1.0
    assert math.isnan(evaluate(tree, k3_ctx(1.0, 10.0)))


def test_an_abstaining_primary_makes_the_tier_unknown() -> None:
    tree = Lexicographic(
        "tier",
        primary=Leaf("correct", None, call(), True),
        secondary=Leaf("p_length", 0.3, call()),
        include_primary=False,
    )
    assert math.isnan(evaluate(tree, k3_ctx(1.0, 10.0)))


# ---------------------------------------------------------------------------
# Disabling
# ---------------------------------------------------------------------------


def test_disabling_a_summed_leaf_drops_its_weight_with_it() -> None:
    tree = task_tree(1.0, 0.8)
    assert evaluate(tree, k3_ctx(1.0, 10.0)) == 1.2
    assert evaluate(tree, k3_ctx(1.0, 10.0), {"style"}) == 1.0


def test_disabling_an_unknown_node_raises_rather_than_reporting_no_change() -> None:
    with pytest.raises(UnknownNode, match="false negative|no such node"):
        evaluate(k3_tree(1.0, 0.8), k3_ctx(1.0, 10.0), {"lenght_override"})


def test_disabling_the_whole_composition_raises() -> None:
    with pytest.raises(ValueError, match="removes the whole composition"):
        evaluate(task_tree(1.0, 0.8), k3_ctx(1.0, 10.0), {"correct", "style"})


def test_ablation_deltas_show_the_task_reward_is_worth_nothing_inside_the_dead_zone() -> None:
    outside = ablation_deltas(k3_tree(1.0, 0.8), k3_ctx(1.0, 100.0))
    # "task" is absent from the table outside the region: removing it empties the composition, so
    # there is no score to difference against and the entry is dropped rather than reported as 0.
    assert outside == pytest.approx({"correct": 1.0, "style": 0.2})
    inside = ablation_deltas(k3_tree(1.0, 0.4), k3_ctx(1.0, 200.0))
    # Inside the region every term below the override is worth exactly nothing, "task" included,
    # and this time "task" is in the table because the override still supplies a score without it.
    assert inside == pytest.approx({"task": 0.0, "correct": 0.0, "style": 0.0})


def test_walking_a_tree() -> None:
    tree = k3_tree(1.0, 0.8)
    assert node_names(tree) == ("length_override", "task", "correct", "style")
    assert [leaf.name for leaf in leaves(tree)] == ["correct", "style"]
    assert find(tree, "task") is tree.otherwise
    assert find(tree, "nope") is None
    assert len(list(walk(tree))) == 4
    assert is_additive(tree) is False
    assert is_additive(task_tree(1.0, 0.8)) is True


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


def test_a_missing_feature_is_not_false() -> None:
    ctx = ScoreContext(features={"b0": 100.0})
    with pytest.raises(MissingFeature, match="not carry"):
        evaluate(k3_tree(1.0, 0.8), ctx)


def test_an_opaque_predicate_reads_its_recorded_outcome_and_refuses_to_guess() -> None:
    pred = PredicateRef("citation_fabricated")
    tree = Override("penalty", pred, -1.0, Leaf("task", 0.9, call()))
    assert evaluate(tree, ScoreContext(predicates={"citation_fabricated": True})) == -1.0
    assert evaluate(tree, ScoreContext(predicates={"citation_fabricated": False})) == 0.9
    with pytest.raises(MissingFeature, match="has to have been recorded"):
        evaluate(tree, ScoreContext())


def test_moving_the_threshold_constant_is_one_field_and_one_more_pass() -> None:
    """The sensitivity analysis nobody publishes for any of these constants."""
    ctx = k3_ctx(1.0, 140.0)
    assert evaluate(k3_tree(1.0, 0.4), ctx) == pytest.approx(1.1)  # 140 is under 1.5 * 100
    tighter = Override(
        "length_override",
        replace(OVER_BUDGET, scale=1.0, expr=""),
        -1.0,
        task_tree(1.0, 0.4),
    )
    assert tighter.condition.expr == "tokens > b0"
    assert evaluate(tighter, ctx) == -1.0  # 140 is over 1.0 * 100


# ---------------------------------------------------------------------------
# The abstention census (B4's substrate)
# ---------------------------------------------------------------------------


def test_the_census_separates_a_silent_zero_from_a_real_zero_and_from_an_unknown() -> None:
    trees = [
        task_tree(1.0, 0.8),  # two clean calls
        WeightedSum(  # the grader raised and 0.0 went in anyway
            "task",
            (
                Leaf("correct", 0.0, call("verifier", "raised"), abstained=True),
                Leaf("style", 0.0, call("judge")),  # a real zero
            ),
            (1.0, 0.25),
        ),
        WeightedSum("task", (Leaf("correct", 1.0, None),)),  # no call record at all
    ]
    got = census(trees)
    assert got.n_leaves == 5
    assert got.n_abstained == 1
    assert got.n_silent_zero == 1
    assert got.n_unattributable == 1
    assert got.substituted_total == 0.0
    assert got.silent_zero_rate == pytest.approx(0.25)  # 1 of the 4 leaves with a known outcome
    assert got.by_grader == {"verifier": 1}
    assert "never used" in got.render()


def test_the_census_reports_leaves_the_override_shadowed() -> None:
    trees = [k3_tree(c, s) for c, s, _ in ROWS]
    ctxs = [k3_ctx(c, t) for c, _, t in ROWS]
    got = census(trees, ctxs)
    assert got.n_leaves == 8
    # Rollouts 1 and 3 are overridden, so their four leaves were never used.
    assert got.n_shadowed == 4


def test_the_census_needs_a_context_per_tree() -> None:
    with pytest.raises(ValueError, match="One context per tree"):
        census([task_tree(1.0, 0.8)], [])


# ---------------------------------------------------------------------------
# The dead zone (I3's substrate)
# ---------------------------------------------------------------------------


def test_the_dead_zone_is_half_this_group_and_it_erased_a_spread_of_0_525() -> None:
    trees = [k3_tree(c, s) for c, s, _ in ROWS]
    ctxs = [k3_ctx(c, t) for c, _, t in ROWS]
    (zone,) = deadzone(trees, ctxs)
    assert zone.node == "length_override"
    assert zone.n == 4
    assert zone.n_region == 2
    assert zone.fraction == 0.5
    assert zone.n_binding == 2
    # The underlying scores in the region are 1.10 and 0.05; their spread is 0.525.
    assert zone.erased_spread == pytest.approx(0.525)
    assert zone.distinct_underlying == 2
    assert "exactly zero" in zone.render()


def test_an_override_that_never_fires_has_an_empty_dead_zone() -> None:
    trees = [k3_tree(c, s) for c, s, _ in ROWS]
    ctxs = [k3_ctx(c, 10.0) for c, _, _ in ROWS]
    (zone,) = deadzone(trees, ctxs)
    assert zone.n_region == 0
    assert zone.fraction == 0.0
    assert zone.erased_spread is None


# ---------------------------------------------------------------------------
# Replaying the estimator
# ---------------------------------------------------------------------------


def test_group_centring_reproduces_the_four_advantages() -> None:
    """mean of (1.2, -1.0, 0.15, -1.0) is -0.1625."""
    got = replay_advantages([1.2, -1.0, 0.15, -1.0], CENTRED)
    assert got == pytest.approx([1.3625, -0.8375, 0.3125, -0.8375])


def test_z_scoring_divides_by_the_group_standard_deviation_plus_epsilon() -> None:
    spec = EstimatorSpec(
        family="grpo",
        group_centred=True,
        std_normalised=True,
        std_epsilon=0.0,
        std_ddof=0,
        degenerate_policy="zero",
    )
    got = replay_advantages([1.0, 3.0], spec)
    # mean 2.0, population std 1.0, so the advantages are -1 and +1 exactly.
    assert got == pytest.approx([-1.0, 1.0])


def test_the_divisor_convention_changes_the_answer_and_the_record_has_to_say_which() -> None:
    """`std_ddof` is a field rather than a convention because the two answers differ by a lot.

    On a two-element group the gap is its widest: the population form divides by `|a-b|/2` and
    Bessel's by `|a-b|/sqrt(2)`, a factor of `sqrt(2)`. At the `num_generations = 4` GRPO usually
    runs it is still 15.5%, against a replay tolerance of 1e-4. E50.
    """
    base = dict(
        family="grpo",
        group_centred=True,
        std_normalised=True,
        std_epsilon=0.0,
        degenerate_policy="zero",
    )
    population = replay_advantages([1.0, 3.0], EstimatorSpec(**base, std_ddof=0))
    bessel = replay_advantages([1.0, 3.0], EstimatorSpec(**base, std_ddof=1))
    assert population == pytest.approx([-1.0, 1.0])
    assert bessel == pytest.approx([-(2**-0.5), 2**-0.5])


def test_an_unstated_divisor_is_refused_rather_than_assumed() -> None:
    """Every framework in scope uses Bessel's correction, and near-certain is not certain.

    Assuming it would be a plausible default in a denominator, which is the shape of confident
    wrong number this module exists to prevent. It refuses exactly as it does for a missing
    `std_epsilon`, and the remedy names the value to record and where to read it from.
    """
    spec = EstimatorSpec(
        family="grpo",
        group_centred=True,
        std_normalised=True,
        std_epsilon=1e-4,
        degenerate_policy="zero",
    )
    got = replay_advantages([1.0, 3.0], spec)
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert "15.5%" in got.detail
    assert "std_ddof" in got.remedy
    assert got.statistics["std_ddof"] is None


def test_a_group_that_is_not_std_normalised_needs_no_divisor() -> None:
    """The refusal is conditioned on the divisor being used, not on the field being absent."""
    spec = EstimatorSpec(family="grpo", group_centred=True, degenerate_policy="keep")
    assert spec.std_ddof is None
    got = replay_advantages([1.0, 3.0], spec)
    assert got == pytest.approx([-1.0, 1.0])


def test_rloo_carries_its_leave_one_out_factor() -> None:
    spec = EstimatorSpec(family="rloo", group_centred=True, degenerate_policy="keep")
    got = replay_advantages([1.0, 0.0, 0.0, 0.0], spec)
    # mean 0.25, and k/(k-1) = 4/3: (1 - 0.25) * 4/3 = 1.0, (0 - 0.25) * 4/3 = -1/3.
    assert got == pytest.approx([1.0, -1 / 3, -1 / 3, -1 / 3])


def test_an_abstention_is_excluded_from_the_mean_and_gets_no_advantage() -> None:
    got = replay_advantages([1.0, 3.0, float("nan")], CENTRED)
    assert got[:2] == pytest.approx([-1.0, 1.0])  # mean of the two present scores is 2.0
    assert math.isnan(got[2])


def test_a_group_where_everything_abstained_warns_and_returns_no_advantages() -> None:
    with pytest.warns(AllAbstainedWarning, match="not a group of zeros"):
        got = replay_advantages([float("nan"), float("nan")], CENTRED)
    assert all(math.isnan(v) for v in got)


def test_clipping_is_applied_to_the_advantage() -> None:
    spec = EstimatorSpec(
        family="grpo", group_centred=True, clip_low=-0.5, clip_high=0.5, degenerate_policy="keep"
    )
    assert replay_advantages([1.0, 0.0], spec) == pytest.approx([0.5, -0.5])


def test_a_non_group_relative_estimator_is_refused_with_a_remedy() -> None:
    spec = EstimatorSpec(family="ppo", group_centred=False)
    got = replay_advantages([1.0, 2.0], spec, where="group 7")
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert "group 7" in got.detail and "'ppo'" in got.detail
    assert "value-function baseline" in got.remedy
    assert got.statistics["family"] == "ppo"


def test_a_missing_epsilon_is_refused_because_it_sits_in_every_denominator() -> None:
    spec = EstimatorSpec(family="grpo", group_centred=True, std_normalised=True, std_epsilon=None)
    got = replay_advantages([1.0, 2.0], spec)
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert "std_epsilon" in got.remedy


def test_a_degenerate_group_with_no_recorded_policy_is_refused_with_the_numbers() -> None:
    """Skip, zero and keep give three different answers, so the record has to say which."""
    spec = EstimatorSpec(
        family="grpo",
        group_centred=True,
        std_normalised=True,
        std_epsilon=1e-6,
        std_ddof=0,
        degenerate_policy="unknown",
    )
    got = replay_advantages([0.5, 0.5, 0.5], spec, where="group 3")
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert "group 3" in got.detail
    assert got.statistics["std"] == 0.0
    assert got.statistics["std_epsilon"] == 1e-6
    assert "degenerate_policy" in got.remedy
    assert "'skip', 'zero' or 'keep'" in got.remedy
    # And the same group is answerable once the record says what the trainer did.
    answered = replay_advantages([0.5, 0.5, 0.5], replace(spec, degenerate_policy="zero"))
    assert answered == pytest.approx([0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# The counterfactual
# ---------------------------------------------------------------------------


def test_the_counterfactual_reports_the_sentence_with_this_run_s_numbers() -> None:
    got = counterfactual([worked_group()], {"length_override"})
    assert not isinstance(got, Refusal)
    assert got.scores_before == (1.2, -1.0, 0.15, -1.0)
    assert got.scores_after == pytest.approx((1.2, 1.1, 0.15, 0.05))
    assert got.advantages_before == pytest.approx((1.3625, -0.8375, 0.3125, -0.8375))
    assert got.advantages_after == pytest.approx((0.575, 0.475, -0.475, -0.575))
    # Population standard deviation of the baseline advantages.
    assert got.sd_reference == pytest.approx(0.9160888330287625)
    # Rollouts 1 and 2 change sign; only rollout 1 moves by more than one standard deviation.
    assert got.n_sign_reversed == 2
    assert got.n_moved == 1
    assert got.n_comparable == 4
    assert got.n_trees_with_node == 4
    assert got.says() == (
        "removing length_override changes 25% of advantages by more than one standard deviation, "
        "and reverses the sign for 2 of 4 rollouts."
    )


def test_the_counterfactual_checks_itself_against_the_recorded_advantages() -> None:
    recorded = [[1.3625, -0.8375, 0.3125, -0.8375]]
    got = counterfactual([worked_group()], {"length_override"}, recorded_advantages=recorded)
    assert not isinstance(got, Refusal)
    assert got.reproduction_error == pytest.approx(0.0, abs=1e-12)
    assert got.notes == ()


def test_a_baseline_that_does_not_reproduce_says_so_in_the_result() -> None:
    recorded = [[1.0, -1.0, 0.0, -1.0]]
    got = counterfactual([worked_group()], {"length_override"}, recorded_advantages=recorded)
    assert not isinstance(got, Refusal)
    assert got.reproduction_error == pytest.approx(0.3625)
    assert any("does not fully describe what the trainer did" in n for n in got.notes)


def test_a_group_missing_its_composition_is_refused_rather_than_dropped() -> None:
    group = GroupScores(
        trees=(k3_tree(1.0, 0.8), None),
        contexts=(k3_ctx(1.0, 100.0), k3_ctx(1.0, 200.0)),
        estimator=CENTRED,
    )
    got = counterfactual([group], {"length_override"})
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert "1 of 2 trajectories carry no score tree" in got.detail
    assert "reward_lens.tap" in got.remedy
    assert got.statistics == {"n_missing": 1, "n_total": 2}


def test_a_counterfactual_that_collapses_the_group_says_which_policy_decided_it() -> None:
    """Removing the node can make the group degenerate, and that is the finding."""
    spec = EstimatorSpec(
        family="grpo",
        group_centred=True,
        std_normalised=True,
        std_epsilon=1e-6,
        std_ddof=0,
        degenerate_policy="zero",
    )
    # Both rollouts are over budget, so with the override in place both score -1 and the group is
    # already degenerate; removing it separates them. Run it the other way round.
    trees = (k3_tree(1.0, 0.8), k3_tree(1.0, 0.8))
    ctxs = (k3_ctx(1.0, 100.0), k3_ctx(1.0, 400.0))
    got = counterfactual(
        [GroupScores(trees=trees, contexts=ctxs, estimator=spec, id="g1")], {"length_override"}
    )
    assert not isinstance(got, Refusal)
    assert got.degenerate_groups_before == 0
    assert got.degenerate_groups_after == 1
    assert any("become degenerate" in n for n in got.notes)


def test_disabling_nothing_is_not_a_counterfactual() -> None:
    with pytest.raises(ValueError, match="is the baseline"):
        counterfactual([worked_group()], set())


def test_disabling_a_node_no_tree_carries_raises() -> None:
    with pytest.raises(UnknownNode, match="false negative"):
        counterfactual([worked_group()], {"kl_penalty"})


def test_the_counterfactual_refuses_when_the_estimator_cannot_be_replayed() -> None:
    got = counterfactual(
        [worked_group(EstimatorSpec(family="ppo", group_centred=False))], {"length_override"}
    )
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert "score scale" in got.remedy


# ---------------------------------------------------------------------------
# The record seam
# ---------------------------------------------------------------------------


def test_a_tree_round_trips_through_the_kernel_codec_unchanged() -> None:
    tree = WeightedSum(
        "total",
        (
            k3_tree(1.0, 0.8),
            Mask("overlong", Leaf("aux", 0.2, call()), when=PredicateRef("truncated")),
            Piecewise("anchor", Leaf("raw", 0.4, call()), ((0.0, 0.0), (1.0, 2.0))),
            GroupGate("wpsm", Leaf("g", 0.1, call()), accuracy_above=0.8, entropy_below=0.2),
            Lexicographic("tier", Leaf("p", 1.0, call()), Leaf("q", 0.5, call())),
            Product("gates", (Leaf("fmt", 1.0, call()),)),
        ),
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    back = decode_foreign(encode_foreign(tree))
    assert back == tree
    assert isinstance(back.children[0], Override)
    assert back.children[0].otherwise.weights == (1.0, 0.25)


def test_a_grader_call_projects_out_of_the_tap_s_record() -> None:
    from reward_lens.tap import CallOutcome, GraderCall

    live = GraderCall(
        seq=3,
        run_id="r",
        grader="mathverify",
        outcome=CallOutcome.RAISED,
        inner_ns=12_000_000,
        added_ns=800,
        value=None,
        error_type="ValueError",
        error_message="could not parse",
        facets={"rater": "gpt", "occasion": 1},
        step=7,
    )
    ref = GraderCallRef.from_call(live, cost_usd=0.002)
    assert ref.grader == "mathverify"
    assert ref.outcome == "raised"
    assert ref.raised is True
    assert ref.latency_s == pytest.approx(0.012)
    assert ref.cost_usd == 0.002
    assert ref.facets == {"rater": "gpt", "occasion": 1}
    assert ref.error_type == "ValueError"
    assert ref.seq == 3 and ref.step == 7
    assert decode_foreign(encode_foreign(ref)) == ref


def test_a_raw_output_the_codec_cannot_hold_is_stored_as_a_repr_and_says_so() -> None:
    from reward_lens.tap import CallOutcome, GraderCall

    class Weird:
        def __repr__(self) -> str:
            return "<Weird verdict>"

    live = GraderCall(
        seq=0, run_id="r", grader="j", outcome=CallOutcome.RETURNED, inner_ns=1, added_ns=1
    )
    live.value = {"score": 0.5, "detail": Weird()}
    ref = GraderCallRef.from_call(live)
    assert ref.raw == {"score": 0.5, "detail": "<Weird verdict>"}
    assert ref.raw_is_repr is True
    assert decode_foreign(encode_foreign(ref)) == ref


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

_finite = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)


@given(
    values=st.lists(_finite, min_size=2, max_size=8),
    weights=st.lists(_finite, min_size=8, max_size=8),
)
@settings(max_examples=200, deadline=None)
def test_on_an_additive_tree_the_ablation_deltas_sum_to_the_total(
    values: list[float], weights: list[float]
) -> None:
    tree = WeightedSum(
        "total",
        tuple(Leaf(f"n{i}", v, None) for i, v in enumerate(values)),
        tuple(weights[: len(values)]),
    )
    ctx = ScoreContext()
    assert is_additive(tree)
    total = evaluate(tree, ctx)
    deltas = ablation_deltas(tree, ctx)
    assert sum(deltas.values()) == pytest.approx(total, rel=1e-9, abs=1e-9)


@given(values=st.lists(_finite, min_size=2, max_size=6), drop=st.integers(min_value=0, max_value=5))
@settings(max_examples=200, deadline=None)
def test_disabling_a_term_equals_never_having_built_it(values: list[float], drop: int) -> None:
    assume(drop < len(values))
    kept = [(i, v) for i, v in enumerate(values) if i != drop]
    full = WeightedSum("total", tuple(Leaf(f"n{i}", v, None) for i, v in enumerate(values)))
    without = WeightedSum("total", tuple(Leaf(f"n{i}", v, None) for i, v in kept))
    ctx = ScoreContext()
    assert evaluate(full, ctx, {f"n{drop}"}) == evaluate(without, ctx)


@given(
    values=st.lists(_finite, min_size=1, max_size=6),
    where=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=200, deadline=None)
def test_one_abstention_anywhere_in_a_sum_absorbs_the_total(
    values: list[float], where: int
) -> None:
    assume(where < len(values))
    children = [Leaf(f"n{i}", v, None) for i, v in enumerate(values)]
    children[where] = Leaf(f"n{where}", None, None, abstained=True)
    assert math.isnan(evaluate(WeightedSum("total", tuple(children)), ScoreContext()))


# ---------------------------------------------------------------------------
# The generated invariance test, and the one place it fails on purpose
# ---------------------------------------------------------------------------


def _advantage_of_first(payload: InvariancePayload, tree_kind: str) -> float:
    """Build a group from the payload's scores and return the first rollout's advantage."""
    scores = np.asarray(payload.scores, dtype=float)
    trees = []
    ctxs = []
    for i, s in enumerate(scores):
        tokens = 400.0 if (tree_kind == "override" and i % 2 == 1) else 100.0
        if tree_kind == "override":
            trees.append(k3_tree(float(s), 0.0))
        else:
            trees.append(task_tree(float(s), 0.0))
        ctxs.append(k3_ctx(float(s), tokens))
    group = GroupScores(trees=tuple(trees), contexts=tuple(ctxs), estimator=CENTRED, id="g")
    out = replay_advantages([evaluate(t, c) for t, c in zip(trees, ctxs)], group.estimator)
    assert not isinstance(out, Refusal)
    return float(out[0])


def test_the_advantage_is_invariant_under_adding_a_per_prompt_constant() -> None:
    """`reward.null`, the group the counterfactual score declares. Generated, 32 draws."""

    def instrument(payload: InvariancePayload) -> float:
        return _advantage_of_first(payload, "plain")

    instrument.__name__ = "record.scores.replay_advantages"
    payload = InvariancePayload(
        scores=np.array([1.0, 0.4, 0.9, 0.1]), group_ids=np.array([0, 0, 0, 0])
    )
    report = check_invariance(instrument, "reward.null", payload, n=32)
    assert report.passed, report.render()
    assert report.max_deviation < 1e-9


def test_an_override_breaks_that_invariance_and_the_test_measures_by_how_much() -> None:
    """A constant added to the task reward does not reach the rollouts the override pinned.

    This is a property of the composition rather than a defect in the estimator, and it is the
    concrete cost of an override: the group is no longer shifted by a constant, so the shift does
    not cancel and every advantage in the group moves.
    """

    def instrument(payload: InvariancePayload) -> float:
        return _advantage_of_first(payload, "override")

    instrument.__name__ = "record.scores.evaluate[override]"
    payload = InvariancePayload(
        scores=np.array([1.0, 0.4, 0.9, 0.1]), group_ids=np.array([0, 0, 0, 0])
    )
    report = check_invariance(instrument, "reward.null", payload, n=32)
    assert not report.passed
    assert report.max_deviation > 0.1

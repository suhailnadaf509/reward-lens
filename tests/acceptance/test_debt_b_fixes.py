"""Regressions for the series B defects the statistical review found, one test per defect.

Every test here carries the number that was wrong and the number that replaced it, because a
regression that only asserts the new behaviour cannot tell a reader what it is protecting. The
findings are R5's on B2 and B6 and R4's on B1 and its nulls.

One of them did not survive reproduction and is recorded here as a correction rather than a fix:
the annotation-budget bound was reported as matching no published result, and it matches Shah et
al. exactly. `test_the_annotation_budget_bound_is_shah_et_al_and_the_15x_is_the_factor_that_cancels`
is that finding.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reward_lens.core.invariance import InvariancePayload, check_invariance
from reward_lens.measure.base import Context
from reward_lens.measure.composition.hodge import PairCount, edge_flow
from reward_lens.measure.composition.nulls import (
    _signed,
    curl_anisotropy,
    random_profile_baseline,
    split_half_reliability,
    transitive_baseline,
)
from reward_lens.measure.composition.revealed import (
    AfriatIndex,
    ComparisonSet,
    afriat_index,
    afriat_index_of,
    bank_from_scores,
    garp,
    garp_violations,
    houtman_maks,
    money_pump,
    random_tournament_index,
)
from reward_lens.measure.composition.tournament import (
    CondorcetRate,
    algebraic_connectivity,
    condorcet,
    copeland_slater_split,
    copeland_winners,
    slater_winners,
    smith_set,
    summarise,
)


def _tournament(n: int, rng: np.random.Generator, p_missing: float = 0.0) -> ComparisonSet:
    w, lo, m = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p_missing:
                continue
            a, b = (i, j) if rng.random() < 0.5 else (j, i)
            w.append(a)
            lo.append(b)
            m.append(float(rng.uniform(0.05, 1.0)))
    return ComparisonSet(
        n_items=n,
        winners=np.asarray(w, dtype=np.intp),
        losers=np.asarray(lo, dtype=np.intp),
        margins=np.asarray(m, dtype=np.float64),
        weights=np.ones(len(w), dtype=np.float64),
    )


def _complete_flow(n: int, wins_b: np.ndarray, reps: np.ndarray, **kw):
    pairs = []
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(PairCount(i, j, reps[k] - wins_b[k], wins_b[k]))
            k += 1
    return edge_flow(pairs, n, **kw)


# ---------------------------------------------------------------------------
# E42 item 14. B2's declared invariance test was vacuous
# ---------------------------------------------------------------------------


def test_the_score_path_cannot_produce_a_violation_and_the_reading_says_so():
    """Before: 5,000 of 5,000 score banks returned 1.0 and this was read as the instrument working.

    A single score vector is a strict weak order, so the dominance graph is acyclic, GARP holds at
    every efficiency and the money pump finds nothing. The numbers are unchanged because they were
    never wrong; what was wrong was treating them as evidence about the margin rule.
    """
    rng = np.random.default_rng(0)
    ones = 0
    violations = 0
    cycles = 0
    for _ in range(500):
        cs = ComparisonSet.from_scores(rng.normal(size=int(rng.integers(3, 9))))
        ones += int(afriat_index_of(cs)[0] == 1.0)
        violations += garp_violations([cs]).n_violations
        cycles += money_pump([cs]).n_cycles
    assert ones == 500
    assert violations == 0
    assert cycles == 0
    assert summarise(bank_from_scores(rng.normal(size=(300, 6)))).cycle_rate == 0.0


def test_a_panel_of_scores_is_the_path_on_which_the_index_has_content():
    """After: the constructor Afriat's index is actually defined for, and it does cycle.

    `from_score_panel` pools several score vectors over one item set, which is several observations
    with the same menu. On 5 observations of 4 items the index sits at 0.4914 with real GARP
    violations behind it, where `from_scores` on the same numbers is pinned at 1.0.
    """
    rng = np.random.default_rng(0)
    rows = rng.normal(size=20).reshape(5, 4)
    panel = ComparisonSet.from_score_panel(rows)
    assert panel.n_comparisons == 30
    assert panel.has_repeated_pairs
    assert afriat_index([panel]).pooled == pytest.approx(0.491409, abs=1e-5)
    assert garp(panel, 1.0).n_violations > 0

    flat = ComparisonSet.from_scores(rows.reshape(-1)[:4])
    assert afriat_index([flat]).pooled == 1.0


def test_the_within_group_normalisation_is_what_buys_the_declared_invariance():
    """The demonstration the module's load-bearing claim never had.

    Before: the generated check ran on `from_scores`, where the index is constant, so a raw
    score-difference margin passed it at a maximum deviation of 0.0. After: on the panel path the
    shipped rule moves the index by 1.9e-15 over 64 sampled affine maps and the raw rule by 0.8477,
    and the raw rule's report comes back failed.
    """
    rng = np.random.default_rng(0)
    n_obs, n_items = 5, 4
    payload = InvariancePayload(
        scores=rng.normal(size=n_obs * n_items),
        group_ids=np.repeat(np.arange(n_obs), n_items),
    )

    def shipped(_inst, p: InvariancePayload) -> float:
        rows = np.asarray(p.scores, dtype=np.float64).reshape(n_obs, n_items)
        return afriat_index([ComparisonSet.from_score_panel(rows)]).pooled

    def raw(_inst, p: InvariancePayload) -> float:
        rows = np.asarray(p.scores, dtype=np.float64).reshape(n_obs, n_items)
        w, lo, m = [], [], []
        for r in range(n_obs):
            s = rows[r]
            for i in range(n_items):
                for j in range(i + 1, n_items):
                    d = float(s[i] - s[j])
                    if d == 0.0:
                        continue
                    (w if d > 0 else lo).append(i)
                    (lo if d > 0 else w).append(j)
                    m.append(min(1.0, abs(d)))
        cs = ComparisonSet(
            n_items=n_items,
            winners=np.asarray(w, dtype=np.intp),
            losers=np.asarray(lo, dtype=np.intp),
            margins=np.asarray(m, dtype=np.float64),
            weights=np.ones(len(w), dtype=np.float64),
        )
        return afriat_index([cs]).pooled

    good = check_invariance(AfriatIndex(), "reward.affine", payload, n=64, run=shipped)
    bad = check_invariance(AfriatIndex(), "reward.affine", payload, n=64, run=raw)
    assert good.passed and good.max_deviation < 1e-12
    assert not bad.passed and bad.max_deviation > 0.1


def test_a_panel_collapses_to_the_majority_tournament_b6_is_defined_on():
    """Before: `dominance` on a panel means "beat at least once", so everything is a Condorcet winner.

    A three-observation panel that cycles has every alternative beating every other at least once,
    which reads as a Condorcet winner on every prompt. The tournament concepts collapse to the
    majority first, and then the cycle is visible where it belongs.
    """
    rows = np.array([[1.0, 0.4, 0.0], [0.0, 1.0, 0.6], [0.3, 0.0, 1.0]])
    panel = ComparisonSet.from_score_panel(rows)
    assert panel.dominance().sum() == 6, "every ordered pair appears, which is not a tournament"
    collapsed = panel.majority_tournament()
    assert collapsed.n_comparisons == 3
    assert not collapsed.has_repeated_pairs
    assert condorcet(panel).winner is None, "collapsing first is what makes this answer possible"
    assert len(smith_set(panel)) == 3


# ---------------------------------------------------------------------------
# E42 item 15. The annotation-budget bound. R5's finding does not survive reproduction
# ---------------------------------------------------------------------------


def test_the_annotation_budget_bound_is_shah_et_al_and_the_15x_is_the_factor_that_cancels():
    """A correction to the record, not a fix: the shipped formula matches a published result exactly.

    E42 item 15 says the bound "matches no published result" and is "15x out on six items and 190x
    on twenty" because it evaluates λ₂ on the raw unnormalised Laplacian while the source normalises
    by |E|. The 15 and the 190 reproduce exactly, and they are |E| at six and twenty items. That is
    the point: Shah et al.'s Laplacian is normalised by the **sample count**, and on a design of L
    comparisons per pair the sample count is L·|E|, so the |E| cancels and the raw-Laplacian form is
    the same statement. R5 divided λ₂ by |E| without multiplying the budget by it.

    Checked against the paper's own arithmetic rather than against the rearrangement: Shah et al.
    state 1/λ₂(L) = (d−1)/2 for the complete graph, and λ₂(L_raw)/|E| = d/(d(d−1)/2) gives exactly
    that at every d tested.
    """
    for d in (4, 6, 10, 20):
        cs = ComparisonSet.from_scores(np.arange(float(d)))
        conn = algebraic_connectivity(cs)
        n_edges = d * (d - 1) // 2
        assert conn.n_edges == n_edges
        assert conn.lambda2 == pytest.approx(float(d))
        assert 1.0 / (conn.lambda2 / n_edges) == pytest.approx((d - 1) / 2.0)

        reps = 3
        raw = d * math.log(d) / (reps * conn.lambda2)
        normalised = d * math.log(d) / ((conn.lambda2 / n_edges) * reps * n_edges)
        assert raw == pytest.approx(normalised, rel=1e-12)

    # the reviewer's factor, reproduced and identified
    for d, factor in ((6, 15), (20, 190)):
        cs = ComparisonSet.from_scores(np.arange(float(d)))
        conn = algebraic_connectivity(cs)
        shipped = d * math.log(d) / (conn.lambda2 * 0.01)
        reviewers = d * math.log(d) / ((conn.lambda2 / conn.n_edges) * 0.01)
        assert reviewers / shipped == pytest.approx(float(factor))
        assert conn.n_edges == factor


def test_no_absolute_comparison_count_is_printed_because_the_constant_is_unknown():
    """Before: "reaching 0.1 needs about 155.4 comparisons per pair" on the twelve-model panel.

    155.4 is n·log n / (λ₂·0.01) with the bound's constant silently set to 1. The constant is
    c·ζ²σ²/γ² in the source and is not known here, so the sentence stated a budget the bound does
    not license. What is reported instead divides the constant out.
    """
    bank = bank_from_scores(np.arange(24.0).reshape(6, 4))
    from reward_lens.measure.composition.tournament import ComparisonConnectivity

    value = ComparisonConnectivity(bank, baseline_draws=2).estimate(Context()).value
    assert "replications_for_error_0p1" not in value
    assert "155" not in value["annotation_budget_reading"]
    assert value["replications_to_halve_the_error"] == pytest.approx(4.0)
    assert value["budget_ratio_against_complete_design"] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# R5's queue on B2 and B6
# ---------------------------------------------------------------------------


def test_copeland_versus_slater_is_reported_as_a_range_and_not_as_a_scalar():
    """Before: 0.2150 on this bank, against an achievable range over tie-breaks of [0.0000, 0.5250].

    1,233 of the 3,000 groups have a tied Copeland maximum and 892 have more than one Slater head,
    so `argmax` and the first enumerated ordering were reading the item indices. After: the three
    forced rates, which have no tie-break in them, and the interval they bracket.
    """
    rng = np.random.default_rng(0)
    bank = [_tournament(5, rng) for _ in range(3000)]
    result = summarise(bank)

    assert result.copeland_slater_forced_disagreements == 0
    assert result.copeland_slater_forced_agreements == 1425
    assert result.copeland_slater_undetermined == 1575
    assert result.copeland_slater_range == (pytest.approx(0.0), pytest.approx(0.525))

    tied = sum(1 for cs in bank if len(copeland_winners(cs)) > 1)
    multi = sum(1 for cs in bank if len(slater_winners(cs)) > 1)
    assert tied == 1233
    assert multi == 892

    reading = CondorcetRate(bank[:200], baseline_draws=2).estimate(Context()).value
    assert "copeland_slater_disagreement_rate" not in reading
    assert (
        reading["copeland_slater_disagreement_range"][0]
        <= (reading["copeland_slater_disagreement_range"][1])
    )


def test_a_forced_disagreement_is_reported_when_one_exists():
    """The lower end of the range is not always zero, and this is a case where it is not.

    Constructed so the Copeland maximum and the Slater head cannot be reconciled: without a case
    like this the forced-disagreement rate would be untested at anything but 0.
    """
    rng = np.random.default_rng(0)
    found = None
    for _ in range(20000):
        cs = _tournament(int(rng.integers(4, 7)), rng, p_missing=float(rng.uniform(0.0, 0.4)))
        if copeland_slater_split(cs) == "disagree":
            found = cs
            break
    assert found is not None, "no forced disagreement in 20,000 draws; the branch would be untested"
    assert not (set(copeland_winners(found)) & set(slater_winners(found)))
    assert summarise([found]).copeland_slater_forced_disagreements == 1


def test_the_error_bound_is_reported_per_group_because_a_mean_bounds_nothing():
    """Before: 2.4761 at the mean group size and mean λ₂, on a bank whose worst group needs 5.0355.

    185 of the 400 groups sat above the figure that was being read as a bound. After: the worst
    group's bound is the headline and the count above the mean figure is reported beside it.
    """
    rng = np.random.default_rng(0)
    bank = []
    while len(bank) < 400:
        cs = _tournament(int(rng.integers(4, 10)), rng, p_missing=0.35)
        if algebraic_connectivity(cs).connected:
            bank.append(cs)
    result = summarise(bank)
    per_group = [algebraic_connectivity(cs).error_bound(1) for cs in bank]
    assert result.max_error_bound == pytest.approx(max(per_group))
    assert result.max_error_bound > 1.5 * float(np.median(per_group))

    from reward_lens.measure.composition.tournament import Connectivity

    at_mean = Connectivity(
        lambda2=result.mean_lambda2,
        n_items=int(round(float(result.n_items.mean()))),
        n_edges=0,
        n_components=1,
        method="mean over groups",
    ).error_bound(1)
    assert sum(1 for b in per_group if b > at_mean) > 100, (
        "the point of the fix: a bound at the mean fails to bound a large share of the bank"
    )


def test_the_cycle_rate_is_conditioned_on_the_condorcet_failure_it_explains():
    """Before: cycle_rate 1.0 with no_condorcet_rate 0.0 on this group, so the two could not be added.

    A three-cycle among 0, 1 and 2 with a fourth alternative beating all of them has a Condorcet
    winner and a cycle at the same time. `cycle_rate` now counts prompts whose winner is missing
    *because of* a cycle, and `any_cycle_rate` carries the raw count.
    """
    cs = ComparisonSet(
        n_items=4,
        winners=np.asarray([0, 1, 2, 3, 3, 3], dtype=np.intp),
        losers=np.asarray([1, 2, 0, 0, 1, 2], dtype=np.intp),
        margins=np.ones(6),
        weights=np.ones(6),
    )
    result = summarise([cs])
    assert result.any_cycle_rate == 1.0
    assert result.cycle_rate == 0.0
    assert result.no_condorcet_rate == 0.0
    assert result.cycle_rate <= result.no_condorcet_rate

    rng = np.random.default_rng(0)
    bank = [_tournament(5, rng) for _ in range(500)]
    result = summarise(bank)
    assert result.cycle_rate <= result.no_condorcet_rate
    assert result.any_cycle_rate >= result.cycle_rate


def test_the_pooled_index_is_compared_against_the_pooled_baseline():
    """Before: a min over groups reported against a mean over groups.

    On banks drawn from the baseline's own process the pooled index falls with bank size, 0.1193 at
    50 groups to 0.0390 at 3,000, while the mean holds near 0.64. Comparing 0.039 with the mean
    baseline of 0.633 makes chance look like a catastrophe.
    """
    for size, expected in ((50, 0.1193), (500, 0.0592), (3000, 0.0390)):
        rng = np.random.default_rng(11)
        bank = [_tournament(5, rng) for _ in range(size)]
        result = afriat_index(bank)
        assert result.pooled == pytest.approx(expected, abs=5e-4)
        assert result.mean == pytest.approx(0.64, abs=0.02)

    rng = np.random.default_rng(11)
    bank = [_tournament(5, rng) for _ in range(200)]
    base = random_tournament_index(bank, n_draws=8, seed=0)
    assert "afriat_pooled" in base and "afriat_mean" in base
    assert base["afriat_pooled"] < base["afriat_mean"]

    value = AfriatIndex(bank, baseline_draws=8).estimate(Context()).value
    assert value["baselines"]["baseline.random_tournament"] == pytest.approx(base["afriat_pooled"])
    assert value["baseline_random_tournament_mean"] == pytest.approx(base["afriat_mean"])


def test_repeated_pair_rows_are_summed_rather_than_overwritten():
    """Before: weight_matrix()[0,1] was 2.0 on three rows totalling 9.0, and the margin was the last.

    Fancy indexing is last-write-wins, so a pair with three observations kept one of them, and the
    Houtman-Maks denominator disagreed with the arc weights it was dividing.
    """
    dup = ComparisonSet(
        n_items=2,
        winners=np.asarray([0, 0, 0], dtype=np.intp),
        losers=np.asarray([1, 1, 1], dtype=np.intp),
        margins=np.asarray([0.9, 0.5, 0.1]),
        weights=np.asarray([4.0, 3.0, 2.0]),
    )
    assert dup.weight_matrix()[0, 1] == 9.0
    assert dup.weight_matrix().sum() == dup.weights.sum()
    assert dup.margin_matrix()[0, 1] == pytest.approx((0.9 * 4 + 0.5 * 3 + 0.1 * 2) / 9.0)
    assert dup.margin_matrix(aggregate="sum")[0, 1] == pytest.approx(0.9 * 4 + 0.5 * 3 + 0.1 * 2)

    single = ComparisonSet(
        n_items=2,
        winners=np.asarray([0], dtype=np.intp),
        losers=np.asarray([1], dtype=np.intp),
        margins=np.asarray([0.4]),
        weights=np.asarray([7.0]),
    )
    assert single.margin_matrix()[0, 1] == pytest.approx(0.4), "unchanged where there is one row"
    assert single.weight_matrix()[0, 1] == 7.0


def test_a_margin_outside_the_unit_interval_is_refused_at_construction():
    """Before: margins [3.0, -2.0, 0.5] around a three-cycle returned an index of exactly 1.0.

    A margin of -2.0 has a critical efficiency of 3.0, which no efficiency reaches, so the verdict
    never entered the relation, the cycle never closed, and a maximally irrational record was
    reported as perfectly rationalizable.
    """
    with pytest.raises(ValueError, match=r"margins must be finite and lie in \[0, 1\]"):
        ComparisonSet(
            n_items=3,
            winners=np.asarray([0, 1, 2], dtype=np.intp),
            losers=np.asarray([1, 2, 0], dtype=np.intp),
            margins=np.asarray([3.0, -2.0, 0.5]),
            weights=np.ones(3),
        )
    with pytest.raises(ValueError, match="weights count the observations"):
        ComparisonSet(
            n_items=2,
            winners=np.asarray([0], dtype=np.intp),
            losers=np.asarray([1], dtype=np.intp),
            margins=np.asarray([0.5]),
            weights=np.asarray([0.0]),
        )


def test_the_houtman_maks_greedy_route_is_a_floor_and_the_registry_now_says_so():
    """Before: the rung declared an upward bias while its own text called the value a floor.

    The greedy ordering over-counts deletions, so the kept fraction sits below the truth. Measured
    on one forty-group bank of eight-item tournaments: 0.7536 greedy against 0.8098 exact.
    """
    rng = np.random.default_rng(5)
    bank = [_tournament(8, rng) for _ in range(40)]
    greedy = houtman_maks(bank, exact_max=4)
    exact = houtman_maks(bank, exact_max=8)
    assert not greedy.exact and exact.exact
    assert greedy.fraction < exact.fraction
    assert greedy.lower_fraction <= exact.fraction <= greedy.upper_fraction

    from reward_lens.core.quantity import ESTIMATORS

    entry = ESTIMATORS.get("grader.houtman_maks.feedback_arcs")
    assert entry is not None, "the estimator is registered against its own quantity id"
    assert entry.quantity == "grader.houtman_maks_index"
    assert entry.bias.direction == "downward"
    assert "floor" in entry.bias.why
    money = ESTIMATORS.get("grader.money_pump.cycle_mean")
    assert money is not None and money.quantity == "grader.money_pump_index"


def test_the_index_is_a_supremum_that_the_axiom_fails_at():
    """The module docstring said "the largest e at which GARP holds"; on a cyclic record it is not.

    Measured over random comparison sets: of the cyclic ones, `garp(cs, afriat_index(cs))` is False
    on every single one, because the verdict that closes the cycle enters the weak relation exactly
    at the critical value.
    """
    rng = np.random.default_rng(0)
    cyclic = 0
    attained = 0
    for _ in range(800):
        cs = _tournament(int(rng.integers(3, 7)), rng)
        e = afriat_index([cs]).pooled
        if e < 1.0:
            cyclic += 1
            attained += int(garp(cs, e).holds)
            assert garp(cs, e * (1.0 - 1e-9)).holds, "and it holds strictly below"
    assert cyclic > 400
    assert attained == 0


def test_the_smith_condorcet_identity_is_a_theorem_and_is_not_reported_as_evidence():
    """A correction to the record: the two numbers agreeing is not a measurement.

    The Smith set is a singleton exactly when its member beats every other alternative, which is the
    definition of a Condorcet winner. It cannot fail for n >= 2 and it holds on 4,000 random designs
    including incomplete ones. It is checked here as a code-correctness invariant, and B2's
    report of it as a corroborating measurement is withdrawn.
    """
    rng = np.random.default_rng(0)
    for _ in range(4000):
        cs = _tournament(int(rng.integers(2, 8)), rng, p_missing=float(rng.uniform(0.0, 0.5)))
        assert (len(smith_set(cs)) == 1) == (condorcet(cs).winner is not None)


# ---------------------------------------------------------------------------
# R4's queue on B1 and the nulls
# ---------------------------------------------------------------------------


def test_the_continuity_correction_is_inverted_at_the_value_that_was_applied():
    """Before: `_signed` hard-coded 0.5, so h = 0.1 came back wrong by 0.0455 and h = 2.0 by 0.2000.

    The recovered win fractions are what the Bradley-Terry fit consumes, so a non-default correction
    moved the null that Null A compares the observation against.
    """
    n = 4
    reps = np.full(6, 11.0)
    wins = np.array([8.0, 11.0, 2.0, 5.0, 0.0, 7.0])
    truth = (2.0 * wins - reps) / reps
    for h in (0.5, 0.1, 2.0, 1e-3):
        flow = _complete_flow(n, wins, reps, scale="logit", correction=h)
        assert flow.correction_value == pytest.approx(h)
        assert float(np.max(np.abs(_signed(flow) - truth))) < 1e-9

    margin = _complete_flow(n, wins, reps)
    assert margin.correction_value == 0.0


def test_the_split_half_interval_for_the_mean_shrinks_and_the_split_spread_does_not():
    """Before: one interval, the dispersion of the per-split coefficient, labelled as a CI.

    Its width went 1.899 at 20 splits to 1.775 at 1,600 and stopped, because it describes how much
    the answer moves with the split rather than how well the mean is pinned. Both are now reported
    and each says which it is.
    """
    rng = np.random.default_rng(1)
    wins = rng.binomial(11, 0.6, size=6).astype(float)
    flow = _complete_flow(4, wins, np.full(6, 11.0))
    widths_ci = []
    widths_spread = []
    for n_splits in (25, 100, 400):
        r = split_half_reliability(flow, None, n_splits=n_splits, seed=3)
        widths_ci.append(r.curl_ci[1] - r.curl_ci[0])
        widths_spread.append(r.curl_split_spread[1] - r.curl_split_spread[0])
    assert widths_ci[-1] < widths_ci[0] / 2.0, "the interval for the mean shrinks with n_splits"
    assert widths_spread[-1] > 0.5 * widths_spread[0], "the split spread does not"
    assert widths_spread[-1] > widths_ci[-1]


def test_the_curl_control_is_matched_in_dimension_rather_than_borrowed_from_the_gradient():
    """Before: the `does_not_replicate` note asserted measurement error on the gradient's say-so.

    `im(grad)` has dimension n-1 where `im(C)` has C(n,2)-(n-1): 9 against 36 at ten items. At equal
    planted signal energy the gradient replicates at 0.6212 and the curl at 0.3882, so the gradient
    passing says nothing about whether a curl of that size would.
    """
    from reward_lens.measure.composition.hodge import curl_operator, incidence, triangles

    for n, expect_grad, expect_curl in ((6, 5, 10), (10, 9, 36)):
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        grad_dim = int(np.linalg.matrix_rank(incidence(n, edges).T.toarray()))
        curl_dim = int(np.linalg.matrix_rank(curl_operator(edges, triangles(edges)).toarray()))
        assert grad_dim == expect_grad == n - 1
        assert curl_dim == expect_curl == n * (n - 1) // 2 - (n - 1)

    rng = np.random.default_rng(0)
    wins = rng.binomial(21, 0.55, size=10).astype(float)
    flow = _complete_flow(5, wins, np.full(10, 21.0))
    result = split_half_reliability(flow, None, n_splits=30, seed=0)
    assert result.curl_subspace_dim == 6
    assert result.gradient_subspace_dim == 4
    assert np.isfinite(result.planted_curl_reliability)
    if result.verdict == "does_not_replicate":
        assert result.planted_curl_reliability >= result.floor


def test_the_participation_ratio_states_which_of_the_two_conventions_it_uses():
    """Before: exactly twice its sibling's convention, with nothing saying so.

    A real skew matrix's singular values come in equal pairs and the code sums over all 2m of them,
    so an isotropic curl over m planes reads 2m. Measured 2.223646 against 1.111823 on a six-item
    design, a ratio of exactly 2.
    """
    rng = np.random.default_rng(0)
    wins = rng.binomial(21, 0.5, size=15).astype(float)
    flow = _complete_flow(6, wins, np.full(15, 21.0))
    result = curl_anisotropy(flow, n_draws=30, seed=0)
    sigma = np.asarray(result.singular_values)
    assert np.allclose(sigma[0::2], sigma[1::2], atol=1e-9), "equal pairs, which is the mechanism"
    assert result.participation_ratio == pytest.approx(2.0 * result.participation_ratio_planes)
    assert result.n_planes == sigma.size // 2
    assert "factor of 2" in result.to_dict()["participation_ratio_convention"]


def test_the_verdict_split_path_uses_every_replication_the_counts_path_uses():
    """Before: at R = 11 the verdict path split 5 and 5 and discarded one while the counts path used all.

    The two paths are documented as equivalent, and eleven is the smallest replication count the
    nulls accept, so the discarded observation was 9% of the data at the module's own floor.
    """
    rng = np.random.default_rng(1)
    reps = np.full(6, 11.0)
    wins = rng.binomial(11, 0.6, size=6).astype(float)
    flow = _complete_flow(4, wins, reps)
    replicates = {
        edge: [1] * int(wins[k]) + [-1] * int(reps[k] - wins[k])
        for k, edge in enumerate(flow.edges)
    }
    from reward_lens.measure.composition.nulls import _split_verdicts

    a1, b1, a2, b2 = _split_verdicts(flow, replicates, np.random.default_rng(0))
    used = a1 + b1 + a2 + b2
    assert np.all(used == reps), (
        f"every replication has to land in one half or the other; got {used}"
    )
    assert np.all(np.abs((a1 + b1) - (a2 + b2)) == 1.0), "5 and 6 at eleven, not 5 and 5"


def test_the_two_null_e_members_do_not_share_a_centre_and_the_module_says_which():
    """Before: "above one replication they still share a centre". The gap grows instead.

    The curl mass is a ratio of quadratic forms and the mean of a ratio is not the ratio of the
    means. Measured on a complete five-item design: 0.0000 at R = 1, 0.0358 at 11, 0.0669 at 101.
    The downstream conclusion survives and by a wider margin: the transitive baseline is a point
    mass at 0.2000 and the random profile spans [0.0386, 0.6739].
    """
    n = 5
    gaps = {}
    for R in (1, 11, 101):
        rng = np.random.default_rng(2)
        wins = rng.binomial(R, 0.55, size=10).astype(float)
        flow = _complete_flow(n, wins, np.full(10, float(R)))
        profile = random_profile_baseline(flow, n_draws=120, seed=0)
        transitive = transitive_baseline(flow, n_draws=120, seed=0)
        gaps[R] = profile.curl_null_mean - transitive.curl_null_mean
        assert transitive.curl_null_mean == pytest.approx((n - 2) / (3.0 * n), abs=1e-12)
        width_t = transitive.curl_null_ci[1] - transitive.curl_null_ci[0]
        width_p = profile.curl_null_ci[1] - profile.curl_null_ci[0]
        assert width_p >= width_t, "the random profile is the weaker null, which is what survives"

    assert gaps[1] == pytest.approx(0.0, abs=1e-12)
    assert gaps[11] > 0.02
    assert gaps[101] > gaps[11], "the gap opens with replication rather than closing"


def test_e32_still_holds_after_all_of_it():
    """The transitive encoding floor is untouched: (n-2)/(3n) exactly, at every size."""
    for n in (3, 4, 5, 6, 7, 8):
        edges = n * (n - 1) // 2
        wins = np.zeros(edges)
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                wins[k] = 1.0  # the higher-indexed item always wins: a total order
                k += 1
        flow = _complete_flow(n, wins, np.ones(edges))
        from reward_lens.measure.composition.hodge import split_flow

        assert split_flow(flow, with_betti=False).curl_mass == pytest.approx(
            (n - 2) / (3.0 * n), abs=1e-12
        )

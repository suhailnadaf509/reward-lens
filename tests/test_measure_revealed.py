"""B2, the Afriat efficiency index: hand-computed cases, the properties, and the two refusals.

Every case here is one somebody can check on paper. A strict total order is rationalizable and its
index is 1. A three-cycle whose weakest link was decided by 13% of the group's spread has an index
of 0.87, and that number is derived below rather than asserted. The Houtman-Maks index of a
three-cycle is 2/3 because exactly one of the three verdicts has to go. Those are the honest tests;
the property tests bracket the estimator's behaviour where a hand computation stops being possible.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from reward_lens.core.invariance import InvariancePayload, check_invariance
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.composition.revealed import (
    CURL_MASS,
    RANDOM_TOURNAMENT,
    AfriatIndex,
    ComparisonSet,
    afriat_index,
    afriat_index_of,
    bank_from_payload,
    bank_from_scores,
    direct_preference,
    garp,
    garp_violations,
    houtman_maks,
    min_feedback_arcs,
    money_pump,
    transitive_closure,
)


def _cycle(margins=(1.0, 1.0, 1.0), source="") -> ComparisonSet:
    """The canonical three-cycle: 0 beats 1 beats 2 beats 0, at the given margins."""
    return ComparisonSet(
        n_items=3,
        winners=np.asarray([0, 1, 2], dtype=np.intp),
        losers=np.asarray([1, 2, 0], dtype=np.intp),
        margins=np.asarray(margins, dtype=np.float64),
        weights=np.ones(3),
        margin_source=source,
    )


# ---------------------------------------------------------------------------
# Hand-computed cases
# ---------------------------------------------------------------------------


def test_a_strict_total_order_is_exactly_rationalizable():
    """Four responses scored 4, 3, 2, 1. Six verdicts, no cycle, so the index is exactly 1."""
    cs = ComparisonSet.from_scores([4.0, 3.0, 2.0, 1.0])
    assert cs.n_comparisons == 6, "four alternatives with distinct scores give six ordered pairs"
    result = afriat_index([cs])
    assert result.pooled == 1.0
    assert result.n_violations == 0
    assert result.rationalizable
    assert result.method == "exact"


def test_equal_scores_record_no_verdict_rather_than_a_tie():
    """A grader that scored two responses identically expressed no preference between them.

    Recording a coin-flip winner would manufacture a comparison the grader never made, and B6's
    whole Condorcet story turns on the difference between an unobserved pair and a tie.
    """
    cs = ComparisonSet.from_scores([1.0, 1.0, 0.0])
    assert cs.n_comparisons == 2
    assert not cs.observed()[0, 1]


def test_a_three_cycle_at_stated_margins_has_the_index_the_margins_imply():
    """0 beats 1 by 0.90, 1 beats 2 by 0.80, 2 beats 0 by 0.13. The index is 0.87.

    Worked: a verdict enters the revealed-preference relation at ``e = 1 - margin``, so the three
    critical efficiencies are 0.10, 0.20 and 0.87. Below 0.87 the weakest verdict is outside the
    relation, nothing is cyclic and the axiom holds. At 0.87 it enters, the closure becomes a cycle
    and the axiom fails. So the largest efficiency the data supports is 0.87, and thirteen percent of
    the decisive margin is what has to be given up.
    """
    cs = _cycle((0.9, 0.8, 0.13), source="win fraction")
    assert not cs.is_ordinal
    index, method = afriat_index_of(cs)
    assert method == "exact"
    assert index == pytest.approx(0.87, abs=1e-12)


def test_bisection_and_exact_enumeration_agree():
    """Two routes to the same supremum, so neither is trusted on its own."""
    cs = _cycle((0.9, 0.8, 0.13), source="win fraction")
    exact, _ = afriat_index_of(cs)
    bisected, how = afriat_index_of(cs, max_candidates=0, tolerance=1e-9)
    assert how == "bisect"
    assert bisected == pytest.approx(exact, abs=1e-8)


def test_the_garp_violation_count_of_a_three_cycle_is_three():
    """Every ordered pair in a three-cycle is indirectly preferred to the one that beats it."""
    report = garp(_cycle())
    assert not report.holds
    assert report.n_violations == 3
    assert set(report.violating_pairs) == {(0, 2), (1, 0), (2, 1)}


def test_houtman_maks_of_a_three_cycle_keeps_two_of_three_verdicts():
    """One deletion breaks the cycle and no deletion is smaller, so the index is 2/3 exactly."""
    hm = houtman_maks([_cycle()])
    assert hm.exact
    assert hm.dropped == 1.0
    assert hm.fraction == pytest.approx(2.0 / 3.0)


def test_the_money_pump_of_a_cycle_is_the_mean_margin_around_it():
    """0.90, 0.80 and 0.13 average to 0.61, which is what walking the cycle once nets."""
    mp = money_pump([_cycle((0.9, 0.8, 0.13), source="win fraction")])
    assert mp.n_cycles == 1
    assert mp.mean == pytest.approx(0.61)
    assert not mp.truncated


def test_direct_preference_drops_exactly_the_verdicts_below_the_budget():
    """At e = 0.5 a verdict decided by 0.4 of the spread is outside the relation and one at 0.6 is in."""
    cs = ComparisonSet(
        n_items=3,
        winners=np.asarray([0, 1], dtype=np.intp),
        losers=np.asarray([1, 2], dtype=np.intp),
        margins=np.asarray([0.6, 0.4]),
        weights=np.ones(2),
        margin_source="win fraction",
    )
    weak, strict = direct_preference(cs, 0.5)
    assert weak[0, 1] and strict[0, 1]
    assert not weak[1, 2] and not strict[1, 2]


def test_warshall_closure_reaches_through_a_chain():
    r = np.zeros((4, 4), dtype=bool)
    r[0, 1] = r[1, 2] = r[2, 3] = True
    closed = transitive_closure(r)
    assert closed[0, 3]
    assert not closed[3, 0]
    assert not r[0, 3], "the input is not mutated"


def test_minimum_feedback_arcs_is_exact_inside_a_small_component_and_bracketed_above_it():
    """A twelve-item cycle is one strongly connected component too large to enumerate."""
    n = 12
    w = np.zeros((n, n))
    for i in range(n):
        w[i, (i + 1) % n] = 1.0
    fa = min_feedback_arcs(w, exact_max=8)
    assert not fa.exact
    assert fa.lower <= fa.cost <= fa.upper
    assert fa.upper == 1.0, "a single long cycle needs one arc broken and greedy finds it"
    assert "greedy" in fa.method
    small = min_feedback_arcs(w[:6, :6] + np.roll(np.eye(6), 1, axis=1) * 0, exact_max=8)
    assert small.exact


# ---------------------------------------------------------------------------
# The generated invariance test
# ---------------------------------------------------------------------------


def _index_of_payload(_instrument, payload: InvariancePayload) -> float:
    return afriat_index(bank_from_payload(payload)).mean


def test_the_index_survives_an_affine_rescaling_of_the_scores():
    """B2's declared group, generated, on the one-score-vector-per-group path.

    **This check passes and it is vacuous, which is worth saying rather than hiding.** A group built
    by `from_scores` is a strict weak order, so its dominance graph is acyclic and the index is 1.0
    before any margin is consulted; the deviation below is zero because the quantity is constant.
    Its docstring used to claim that "taking the margin as a raw score difference would fail it",
    and the check in
    `test_the_margin_rule_is_what_makes_the_index_affine_invariant_and_here_is_the_proof` shows the
    raw rule passing this very payload at a maximum deviation of 0.0. The test that discriminates is
    that one, on the panel path, and this one is kept because a regression that made the index move
    on ordinary score data would still be a bug.
    """
    rng = np.random.default_rng(0)
    payload = InvariancePayload(scores=rng.normal(size=64), group_ids=np.repeat(np.arange(16), 4))
    report = check_invariance(AfriatIndex(), "reward.affine", payload, n=64, run=_index_of_payload)
    assert report.passed, report.render()
    assert report.max_deviation == 0.0, report.render()
    bank = bank_from_payload(payload)
    assert afriat_index(bank).pooled == 1.0
    assert garp_violations(bank).n_violations == 0, "the path cannot produce a violation at all"


def _raw_margin_panel(rows: np.ndarray) -> ComparisonSet:
    """`from_score_panel` with the margin as the raw score difference, for the comparison below.

    Not shipped and not importable from the package: it exists to be the rule the declared
    invariance is supposed to reject, so that rejecting it can be demonstrated rather than asserted.
    Margins are clipped into [0, 1] only because `ComparisonSet` now validates them; the clipping is
    applied *after* the rescaling, so it does not restore the invariance the raw rule lacks.
    """
    rows = np.asarray(rows, dtype=np.float64)
    n_obs, n = rows.shape
    w, lo, m = [], [], []
    for r in range(n_obs):
        s = rows[r]
        for i in range(n):
            for j in range(i + 1, n):
                d = float(s[i] - s[j])
                if d == 0.0:
                    continue
                (w if d > 0 else lo).append(i)
                (lo if d > 0 else w).append(j)
                m.append(min(1.0, abs(d)))
    return ComparisonSet(
        n_items=n,
        winners=np.asarray(w, dtype=np.intp),
        losers=np.asarray(lo, dtype=np.intp),
        margins=np.asarray(m, dtype=np.float64),
        weights=np.ones(len(w), dtype=np.float64),
    )


def test_the_margin_rule_is_what_makes_the_index_affine_invariant_and_here_is_the_proof():
    """The invariance check with content in it: it passes on the shipped rule and fails on the other.

    A panel is several score vectors over one item set, which is Afriat's own setting and the only
    shipped constructor whose index is not pinned at 1.0. On this payload it produces genuine GARP
    violations, so the index sits strictly inside (0, 1) and the margins decide where.

    Both rules are then pushed through the identical generated check. The shipped within-observation
    normalisation moves the index by 2.2e-16 over 64 sampled affine maps; the raw score difference
    moves it by 0.48. That is the demonstration the module's claim was missing.
    """
    rng = np.random.default_rng(0)
    n_obs, n_items = 5, 4
    payload = InvariancePayload(
        scores=rng.normal(size=n_obs * n_items),
        group_ids=np.repeat(np.arange(n_obs), n_items),
    )

    def shipped(_inst, p: InvariancePayload) -> float:
        p.require("scores")
        rows = np.asarray(p.scores, dtype=np.float64).reshape(n_obs, n_items)
        return afriat_index([ComparisonSet.from_score_panel(rows)]).pooled

    def raw(_inst, p: InvariancePayload) -> float:
        p.require("scores")
        rows = np.asarray(p.scores, dtype=np.float64).reshape(n_obs, n_items)
        return afriat_index([_raw_margin_panel(rows)]).pooled

    base = shipped(None, payload)
    assert 0.0 < base < 1.0, (
        f"the payload has to have an interior index for this to bite; got {base}"
    )

    good = check_invariance(AfriatIndex(), "reward.affine", payload, n=64, run=shipped)
    bad = check_invariance(AfriatIndex(), "reward.affine", payload, n=64, run=raw)

    assert good.passed, good.render()
    assert good.max_deviation < 1e-12, good.render()
    assert not bad.passed, (
        "the raw score difference has to fail the declared invariance, or the within-group "
        "normalisation is not what is buying it: " + bad.render()
    )
    assert bad.max_deviation > 0.1, bad.render()


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_ordinal_data_with_a_cycle_refuses_with_the_margin_remedy_and_a_bound():
    """On a bare win/loss record the index can only be 1 or 0, and 0 would be a slander.

    The loser's value is 0 when there is no margin, so the budget constraint binds at every
    efficiency and the axiom's status never changes. Reporting 0 would say a grader with three
    cyclic verdicts among thousands is maximally irrational. The refusal names the two ways to get
    margins and carries the Houtman-Maks index, which stays informative on ordinal data.
    """
    bank = [ComparisonSet.from_scores([3.0, 2.0, 1.0]) for _ in range(4)] + [_cycle()]
    reading = AfriatIndex(bank, baseline_draws=4).estimate(Context())
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ABOVE_LOD_BELOW_LOQ
    assert "margins" in reading.remedy
    assert "from_edges" in reading.remedy
    assert reading.is_bounded
    assert reading.partial.value["afriat_index_upper"] == 1.0
    assert reading.partial.value["houtman_maks_fraction"] == pytest.approx(14.0 / 15.0)
    assert reading.statistics["n_violations"] == 3


def test_a_record_with_no_comparison_refuses_rather_than_returning_one():
    """A bank of all-equal scores records no preference, which is not an index of 1."""
    bank = bank_from_scores(np.zeros((4, 3)))
    reading = AfriatIndex(bank).estimate(Context())
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "bank_from_scores" in reading.remedy


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


def test_the_instrument_lints_and_emits_the_index_beside_its_violation_count():
    inst = AfriatIndex(bank_from_scores(np.arange(24.0).reshape(6, 4)), curl_mass=0.214)
    assert lint_instrument(inst) == []
    reading = inst.estimate(Context())
    assert not isinstance(reading, Refusal)
    assert reading.value["afriat_index"] == 1.0
    assert reading.value["garp_violations"] == 0
    assert reading.value["curl_mass"] == 0.214
    assert set(reading.value["baselines"]) == {CURL_MASS, RANDOM_TOURNAMENT}


def test_the_curl_mass_baseline_says_it_is_absent_rather_than_reporting_a_number():
    """B1 owns the curl mass. Not having it is a fact about the reading, not a zero."""
    inst = AfriatIndex(bank_from_scores(np.arange(24.0).reshape(6, 4)))
    reading = inst.estimate(Context())
    assert CURL_MASS not in reading.value["baselines"]
    assert "not supplied" in reading.value["curl_mass"]


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

_margins = st.lists(st.floats(min_value=0.01, max_value=1.0), min_size=1, max_size=8)


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=2, max_value=5),
    seed=st.integers(min_value=0, max_value=2**16),
    margins=_margins,
)
def test_the_index_is_always_in_the_unit_interval(n, seed, margins):
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if not pairs:
        return
    keep = rng.integers(0, 2, size=len(pairs)).astype(bool)
    keep[0] = True
    chosen = [p for p, k in zip(pairs, keep) if k]
    flip = rng.integers(0, 2, size=len(chosen)).astype(bool)
    cs = ComparisonSet(
        n_items=n,
        winners=np.asarray([j if f else i for (i, j), f in zip(chosen, flip)], dtype=np.intp),
        losers=np.asarray([i if f else j for (i, j), f in zip(chosen, flip)], dtype=np.intp),
        margins=np.asarray([margins[k % len(margins)] for k in range(len(chosen))]),
        weights=np.ones(len(chosen)),
        margin_source="win fraction",
    )
    index, _ = afriat_index_of(cs)
    assert 0.0 <= index <= 1.0


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=2**16))
def test_the_index_is_one_exactly_when_there_are_no_violations(seed):
    """The two rungs have to agree at the boundary, or rung 0 is estimating something else."""
    rng = np.random.default_rng(seed)
    n = 5
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    flip = rng.integers(0, 2, size=len(pairs)).astype(bool)
    cs = ComparisonSet(
        n_items=n,
        winners=np.asarray([j if f else i for (i, j), f in zip(pairs, flip)], dtype=np.intp),
        losers=np.asarray([i if f else j for (i, j), f in zip(pairs, flip)], dtype=np.intp),
        margins=rng.uniform(0.05, 1.0, size=len(pairs)),
        weights=np.ones(len(pairs)),
        margin_source="win fraction",
    )
    index, _ = afriat_index_of(cs)
    violations = garp(cs).n_violations
    assert (index == 1.0) == (violations == 0)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=2**16))
def test_adding_a_verdict_can_only_lower_the_index(seed):
    """Monotone in the violations it can create: a larger record is a tighter constraint.

    Every verdict added enters the revealed-preference relation at its own efficiency and stays in
    it above that, so the set of efficiencies where the axiom holds can only shrink. An estimator
    that let the index rise when a comparison arrived would be reading something other than
    rationalizability.
    """
    rng = np.random.default_rng(seed)
    n = 5
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    order = rng.permutation(len(pairs))
    flip = rng.integers(0, 2, size=len(pairs)).astype(bool)
    margins = rng.uniform(0.05, 1.0, size=len(pairs))
    previous = 1.0
    for k in range(1, len(pairs) + 1):
        idx = order[:k]
        chosen = [pairs[int(p)] for p in idx]
        f = flip[idx]
        cs = ComparisonSet(
            n_items=n,
            winners=np.asarray([j if a else i for (i, j), a in zip(chosen, f)], dtype=np.intp),
            losers=np.asarray([i if a else j for (i, j), a in zip(chosen, f)], dtype=np.intp),
            margins=margins[idx],
            weights=np.ones(k),
            margin_source="win fraction",
        )
        index, _ = afriat_index_of(cs)
        assert index <= previous + 1e-12
        previous = index

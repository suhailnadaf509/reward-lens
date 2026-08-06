"""B6, tournament solutions and comparison-graph connectivity: hand cases and the refusal.

The λ₂ cases are the ones worth checking by hand, because the number is easy to get almost right. A
complete graph on ``n`` vertices has algebraic connectivity exactly ``n``. A path on four has
``2 - √2``. A graph with an isolated vertex has 0, and 0 is where the annotation-budget bound stops
meaning anything, which is what the instrument refuses on.

The tournament cases are equally checkable. A three-cycle has no Condorcet winner and a Smith set of
three. A strict order has a winner and a Smith set of one. The two facts are the same fact, and the
property test asserts the equivalence rather than the two numbers separately.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from reward_lens.core.invariance import InvariancePayload, check_invariance
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.composition.revealed import ComparisonSet, bank_from_scores
from reward_lens.measure.composition.tournament import (
    RANDOM_DESIGN,
    RANDOM_TOURNAMENT,
    TOTAL_ORDER,
    ComparisonConnectivity,
    CondorcetRate,
    SmithSetSize,
    algebraic_connectivity,
    bank_from_invariance_payload,
    condorcet,
    copeland_scores,
    kemeny,
    pair_to_add,
    slater,
    smith_set,
    summarise,
)


def _graph(n: int, edges) -> ComparisonSet:
    """A comparison graph with the given edges, oriented so the lower index wins."""
    edges = list(edges)
    return ComparisonSet(
        n_items=n,
        winners=np.asarray([i for i, _ in edges], dtype=np.intp),
        losers=np.asarray([j for _, j in edges], dtype=np.intp),
        margins=np.ones(len(edges)),
        weights=np.ones(len(edges)),
    )


def _complete(n: int) -> ComparisonSet:
    return _graph(n, [(i, j) for i in range(n) for j in range(i + 1, n)])


def _cycle3() -> ComparisonSet:
    return ComparisonSet(
        n_items=3,
        winners=np.asarray([0, 1, 2], dtype=np.intp),
        losers=np.asarray([1, 2, 0], dtype=np.intp),
        margins=np.ones(3),
        weights=np.ones(3),
    )


# ---------------------------------------------------------------------------
# Condorcet and Smith, by hand
# ---------------------------------------------------------------------------


def test_a_three_cycle_has_no_condorcet_winner_and_a_smith_set_of_three():
    c = condorcet(_cycle3())
    assert c.winner is None
    assert c.has_cycle
    assert c.complete
    assert c.reason == "cycle"
    assert smith_set(_cycle3()) == (0, 1, 2)


def test_a_strict_order_has_a_condorcet_winner_and_a_singleton_smith_set():
    cs = ComparisonSet.from_scores([4.0, 3.0, 2.0, 1.0])
    c = condorcet(cs)
    assert c.winner == 0
    assert not c.has_cycle
    assert smith_set(cs) == (0,)
    assert list(copeland_scores(cs)) == [3.0, 1.0, -1.0, -3.0]


def test_a_missing_comparison_is_reported_as_a_missing_comparison_and_not_as_a_cycle():
    """0 beat 1 and 2 beat 3, and nobody compared them across. There is no winner and no cycle.

    This is the split that decides what to do next. A cycle is the grader disagreeing with itself
    and no further annotation removes it. A missing pair is a design gap and one comparison may
    close it. A rate that folds the two together sends a comparison-design problem to the wrong desk.
    """
    cs = _graph(4, [(0, 1), (2, 3)])
    c = condorcet(cs)
    assert c.winner is None
    assert not c.has_cycle
    assert not c.complete
    assert c.reason == "incomplete comparison graph"
    assert c.possible == (0, 2), "each of them beat everything it met"
    assert (0, 2) in c.missing_pairs


def test_an_unobserved_pair_enlarges_the_smith_set_rather_than_shrinking_it():
    """Nothing can be excluded from the set the reward picks from until it has been compared."""
    assert smith_set(_graph(4, [(0, 1), (2, 3)])) == (0, 1, 2, 3)


def test_slater_and_kemeny_are_the_same_computation_at_different_weights():
    """Both are exact here because the component has three alternatives, well under the cap of eight."""
    cyc = ComparisonSet(
        n_items=3,
        winners=np.asarray([0, 1, 2], dtype=np.intp),
        losers=np.asarray([1, 2, 0], dtype=np.intp),
        margins=np.asarray([0.9, 0.8, 0.1]),
        weights=np.ones(3),
        margin_source="win fraction",
    )
    s = slater(cyc)
    k = kemeny(cyc)
    assert s.exact and k.exact
    assert s.cost == 1.0, "one verdict reversed puts a three-cycle in order"
    assert k.cost == pytest.approx(0.1), "Kemeny reverses the cheapest verdict, not just any one"


# ---------------------------------------------------------------------------
# Connectivity, by hand
# ---------------------------------------------------------------------------


def test_the_complete_graph_has_algebraic_connectivity_equal_to_its_size():
    for n in (3, 4, 7):
        conn = algebraic_connectivity(_complete(n))
        assert conn.lambda2 == pytest.approx(float(n))
        assert conn.connected
        assert conn.missing_pairs == ()
        assert conn.method == "dense eigvalsh"


def test_the_path_on_four_vertices_has_algebraic_connectivity_two_minus_root_two():
    conn = algebraic_connectivity(_graph(4, [(0, 1), (1, 2), (2, 3)]))
    assert conn.lambda2 == pytest.approx(2.0 - math.sqrt(2.0))


def test_a_disconnected_graph_has_zero_connectivity_and_names_the_pairs_that_would_fix_it():
    cs = _graph(4, [(0, 1), (1, 2), (0, 2)])
    conn = algebraic_connectivity(cs)
    assert conn.lambda2 == 0.0
    assert conn.n_components == 2
    assert not conn.connected
    assert set(conn.missing_pairs) == {(0, 3), (1, 3), (2, 3)}
    pair, gained = pair_to_add(cs)
    assert pair in conn.missing_pairs
    assert gained == pytest.approx(1.0), "joining the isolate to a triangle gives λ₂ = 1"


def test_the_error_bound_reproduces_its_sources_own_arithmetic():
    """The bound anchored to Shah et al. rather than to itself.

    This replaces a test that asserted ``replications_for(error_bound(1)) == 1``, which is the
    statement that one algebraic rearrangement inverts another and could not have failed for any
    formula whatsoever. The bound is Shah, Balakrishnan, Bradley, Parekh, Ramchandran and
    Wainwright, JMLR 17(58), 2016, Theorem 2 equation (8b), whose Laplacian is normalised by the
    **sample count**. What is checked here is that using the raw Laplacian is the same statement:
    on a design of L comparisons per pair the sample count is ``L·|E|``, ``λ₂`` scales by ``1/|E|``,
    and the ``|E|`` cancels. The paper's own figure for the complete graph, ``1/λ₂(L) = (d-1)/2``,
    is the fixed point that makes the check independent of the rearrangement.
    """
    for d in (4, 6, 10):
        conn = algebraic_connectivity(_complete(d))
        n_edges = d * (d - 1) // 2
        assert conn.lambda2 == pytest.approx(float(d))
        # the paper's stated value for the sample-normalised Laplacian on the complete graph
        assert 1.0 / (conn.lambda2 / n_edges) == pytest.approx((d - 1) / 2.0)
        # and the two coordinate systems give the same squared error at L comparisons per pair
        reps = 3
        raw = d * math.log(d) / (reps * conn.lambda2)
        normalised = d * math.log(d) / ((conn.lambda2 / n_edges) * reps * n_edges)
        assert raw == pytest.approx(normalised, rel=1e-12)
        assert conn.error_bound(reps) == pytest.approx(math.sqrt(raw))


def test_the_budget_reading_is_a_ratio_because_the_constant_is_unknown():
    """No absolute comparison count, because the bound's constant is not 1 and is not known.

    The reading used to print "reaching 0.1 needs about 155.4 comparisons per pair", which is
    ``n log n / (λ₂ · 0.01)`` with ``c·ζ²σ²/γ²`` silently set to 1. What survives division by that
    constant is a ratio between two designs and the cost of a stated improvement, and those are
    what the class offers now.
    """
    conn = algebraic_connectivity(_complete(4))
    assert not hasattr(conn, "replications_for"), (
        "an absolute comparison count cannot be recovered from a bound stated up to a constant"
    )
    assert conn.error_bound(4) == pytest.approx(conn.error_bound(1) / 2.0)
    assert conn.relative_budget(2.0) == pytest.approx(4.0)
    assert conn.relative_budget(3.0) == pytest.approx(9.0)
    assert conn.error_bound(0) == math.inf

    path = _graph(4, [(0, 1), (1, 2), (2, 3)])
    ratio = algebraic_connectivity(path).replications_ratio_for(conn)
    assert ratio == pytest.approx(conn.lambda2 / algebraic_connectivity(path).lambda2, rel=1e-9)
    assert ratio > 1.0, "a path is worse connected than the complete comparison, so it costs more"


def test_the_sparse_eigensolver_agrees_with_the_dense_one():
    """`scipy.sparse.linalg.eigsh` is the route above the cap; it has to give the dense answer."""
    cs = _complete(40)
    dense = algebraic_connectivity(cs, sparse_above=64)
    sparse = algebraic_connectivity(cs, sparse_above=8)
    assert "eigsh" in sparse.method
    assert sparse.lambda2 == pytest.approx(dense.lambda2, rel=1e-6)


# ---------------------------------------------------------------------------
# The generated invariance tests
# ---------------------------------------------------------------------------


def _payload() -> InvariancePayload:
    rng = np.random.default_rng(0)
    return InvariancePayload(scores=rng.normal(size=64), group_ids=np.repeat(np.arange(16), 4))


def test_the_condorcet_rate_is_invariant_under_permuting_rollout_order():
    """B6's declared group, generated. Permuting a group relabels the alternatives, nothing more."""
    report = check_invariance(
        CondorcetRate(),
        "group.permutation",
        _payload(),
        n=64,
        run=lambda _i, p: summarise(bank_from_invariance_payload(p)).condorcet_rate,
    )
    assert report.passed, report.render()
    assert report.max_deviation == 0.0


def test_the_smith_set_size_is_invariant_under_permuting_rollout_order():
    report = check_invariance(
        SmithSetSize(),
        "group.permutation",
        _payload(),
        n=64,
        run=lambda _i, p: summarise(bank_from_invariance_payload(p)).mean_smith_size,
    )
    assert report.passed, report.render()
    assert report.max_deviation == 0.0


def test_lambda2_declares_the_trivial_group_the_registry_gives_it_and_is_permutation_invariant():
    """The generated test for `none` passes vacuously and says so, which is honest and is not a check.

    λ₂ is in fact invariant under relabelling the alternatives, because permuting them is a
    similarity transform of the Laplacian and a similarity transform does not move eigenvalues. That
    is asserted here directly rather than by changing the declaration, because an invariance group
    assignment is the registry's to make and not this instrument's.
    """
    report = check_invariance(
        ComparisonConnectivity(),
        "trivial",
        _payload(),
        run=lambda _i, p: summarise(bank_from_invariance_payload(p)).mean_lambda2,
    )
    assert report.passed
    assert "no generators" in report.skipped

    cs = _graph(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    perm = [3, 0, 4, 1, 2]
    relabelled = ComparisonSet(
        n_items=5,
        winners=np.asarray([perm[int(w)] for w in cs.winners], dtype=np.intp),
        losers=np.asarray([perm[int(x)] for x in cs.losers], dtype=np.intp),
        margins=cs.margins,
        weights=cs.weights,
    )
    assert algebraic_connectivity(relabelled).lambda2 == pytest.approx(
        algebraic_connectivity(cs).lambda2
    )


# ---------------------------------------------------------------------------
# Refusals and the instruments
# ---------------------------------------------------------------------------


def test_connectivity_refuses_on_a_disconnected_design_and_names_the_pair_to_compare():
    """λ₂ = 0 makes the error bound infinite, so a zero here would stand in for a missing quantity."""
    bank = [_complete(4), _complete(4), _graph(4, [(0, 1), (1, 2), (0, 2)])]
    reading = ComparisonConnectivity(bank, baseline_draws=2).estimate(Context())
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ABOVE_LOD_BELOW_LOQ
    assert "compare alternatives" in reading.remedy
    assert reading.is_bounded
    assert reading.partial.value["graph_lambda2"] == 0.0
    assert reading.partial.value["disconnected_groups"] == 1
    assert reading.statistics["disconnected_groups"] == 1


def test_a_bank_with_no_group_of_two_refuses_with_an_instruction():
    for cls in (CondorcetRate, SmithSetSize, ComparisonConnectivity):
        reading = cls([ComparisonSet.from_scores([1.0])]).estimate(Context())
        assert isinstance(reading, Refusal), cls.__name__
        assert reading.reason is RefusalReason.ACCESS_INSUFFICIENT
        assert "from_edges" in reading.remedy


def test_the_three_instruments_lint_and_carry_their_declared_baselines():
    bank = bank_from_scores(np.arange(24.0).reshape(6, 4))
    for cls, expected in (
        (CondorcetRate, {TOTAL_ORDER, RANDOM_TOURNAMENT}),
        (SmithSetSize, {TOTAL_ORDER, RANDOM_TOURNAMENT}),
        (ComparisonConnectivity, {TOTAL_ORDER, RANDOM_DESIGN}),
    ):
        inst = cls(bank, baseline_draws=2)
        assert lint_instrument(inst) == [], cls.__name__
        reading = inst.estimate(Context())
        assert not isinstance(reading, Refusal), cls.__name__
        assert set(reading.value["baselines"]) == expected


def test_a_bank_of_strict_orders_reads_as_a_total_order_on_every_prompt():
    """The reading carries three Copeland-versus-Slater rates and no scalar, which is the fix.

    The old assertion was ``reading.value["copeland_slater_disagreement_rate"] == 0.0``. That key is
    gone: the scalar it named was `argmax` against the first enumerated Slater ordering, and on a
    tied tournament both are choices of item index. On a bank of strict orders every group has one
    Copeland winner and one Slater head and they agree, so the forced-agreement rate is 1.0 and the
    achievable range collapses to a point, which is the case where a scalar would have been safe and
    is the only case where it would have been.
    """
    bank = bank_from_scores(np.arange(24.0).reshape(6, 4))
    reading = CondorcetRate(bank, baseline_draws=2).estimate(Context())
    assert reading.value["condorcet_rate"] == 1.0
    assert reading.value["no_condorcet_rate"] == 0.0
    assert reading.value["cycle_rate"] == 0.0
    assert reading.value["any_cycle_rate"] == 0.0
    assert "copeland_slater_disagreement_rate" not in reading.value
    assert reading.value["copeland_slater_forced_agreement_rate"] == 1.0
    assert reading.value["copeland_slater_forced_disagreement_rate"] == 0.0
    assert reading.value["copeland_slater_undetermined_rate"] == 0.0
    assert reading.value["copeland_slater_disagreement_range"] == [0.0, 0.0]
    assert reading.value["baselines"][TOTAL_ORDER] == 1.0


def test_the_lambda2_reading_prices_designs_against_each_other_and_not_against_a_target():
    """The budget sentence, with the absolute comparison count taken out of it.

    The old assertions were ``reading.value["error_bound_L1"] > 0`` and
    ``reading.value["replications_for_error_0p1"] > 1``. Both keys are gone. The first was evaluated
    at the mean group size and the mean λ₂, which bounds no group; the worst group's bound replaces
    it, with the count of groups the mean figure failed to bound beside it. The second printed an
    absolute comparison count from a bound whose constant is unknown.
    """
    bank = bank_from_scores(np.arange(24.0).reshape(6, 4))
    reading = ComparisonConnectivity(bank, baseline_draws=2).estimate(Context())
    assert reading.value["graph_lambda2"] == pytest.approx(4.0)
    assert "error_bound_L1" not in reading.value
    assert "replications_for_error_0p1" not in reading.value
    assert reading.value["error_bound_worst_group_L1"] > 0
    assert (
        reading.value["error_bound_worst_group_L1"] >= reading.value["error_bound_at_the_mean_L1"]
    )
    assert reading.value["groups_above_the_mean_bound"] == 0, "a uniform bank has nothing above it"
    assert reading.value["replications_to_halve_the_error"] == pytest.approx(4.0)
    assert "prices one design against another" in reading.value["annotation_budget_reading"]


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def _random_set(rng, n: int, density: float) -> ComparisonSet:
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    keep = rng.random(len(pairs)) < density
    chosen = [p for p, k in zip(pairs, keep) if k]
    flip = rng.integers(0, 2, size=len(chosen)).astype(bool)
    return ComparisonSet(
        n_items=n,
        winners=np.asarray([j if f else i for (i, j), f in zip(chosen, flip)], dtype=np.intp),
        losers=np.asarray([i if f else j for (i, j), f in zip(chosen, flip)], dtype=np.intp),
        margins=np.ones(len(chosen)),
        weights=np.ones(len(chosen)),
    )


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=2, max_value=6),
    density=st.floats(min_value=0.0, max_value=1.0),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_the_smith_set_is_never_empty(n, density, seed):
    cs = _random_set(np.random.default_rng(seed), n, density)
    assert len(smith_set(cs)) >= 1
    assert len(smith_set(cs)) <= n


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=2, max_value=6),
    density=st.floats(min_value=0.0, max_value=1.0),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_the_smith_set_is_a_singleton_exactly_when_a_condorcet_winner_exists(n, density, seed):
    """One fact wearing two names, and asserting the equivalence is what keeps them one fact."""
    cs = _random_set(np.random.default_rng(seed), n, density)
    s = smith_set(cs)
    winner = condorcet(cs).winner
    assert (len(s) == 1) == (winner is not None)
    if winner is not None:
        assert s == (winner,)


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=2, max_value=8),
    density=st.floats(min_value=0.0, max_value=1.0),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_connectivity_is_zero_exactly_when_the_graph_is_disconnected(n, density, seed):
    cs = _random_set(np.random.default_rng(seed), n, density)
    conn = algebraic_connectivity(cs)
    assert (conn.lambda2 == 0.0) == (conn.n_components > 1)
    assert conn.lambda2 <= n + 1e-9, "λ₂ of a simple graph never exceeds its number of vertices"

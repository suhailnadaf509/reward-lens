"""B6, tournament solutions and comparison-graph connectivity: which rule the reward implements.

A group reward built from binary comparisons **is** a tournament solution concept. Picking the
response with the most wins is Copeland. Picking the top of the ranking that breaks fewest verdicts
is Slater. Picking one at random from the set nothing outside it beats is the Smith set. Those are
different rules, they disagree on exactly the prompts where the comparisons are cyclic, and a
tournament reward chooses one of them whether or not anybody decided to. Which one it implements
determines what the policy is optimising, so the first job of this module is to say which one, and
the second is to count how often it matters.

Three quantities, all cheap.

``grader.condorcet_rate`` is the fraction of prompts with a Condorcet winner, an alternative that
beat every other one. Reported with its complement and with the reason for each failure split out,
because "there is no Condorcet winner" has two causes that call for opposite responses. A **cycle**
means the grader's own verdicts disagree and no extra comparison will fix it. A **missing
comparison** means nobody asked, and one more comparison might settle it. Collapsing the two is how
a comparison-design problem gets reported as a grader problem.

``grader.smith_set_size`` is the size of the smallest set whose members beat everything outside it.
It is 1 exactly when a Condorcet winner exists, and when it is larger the reward is choosing
arbitrarily inside it. Computed from the strongly connected components of the dominance digraph and
then closed upward, which is what makes the answer minimal rather than merely dominant.

``grader.graph_lambda2`` is the algebraic connectivity of the comparison graph, the second-smallest
eigenvalue of its Laplacian. It is the quantity in the preference-learning error bound
``||v̂ - v||₂ ≲ √(n log n / (L λ₂))`` of Shah et al., JMLR 17(58), 2016, Theorem 2 equation (8b),
which `Connectivity.error_bound` derives and cites in full. So it says what accuracy the *design*
allows at a fixed budget, and it prices one design against another at a fixed accuracy. It does
**not** convert an accuracy into a comparison count, because the bound carries an unknown constant,
and the reading no longer prints one. A disconnected graph has ``λ₂ = 0`` and the bound is vacuous,
which is a refusal with a remedy that names the pairs to compare rather than a zero passed
downstream.

On Slater and Kemeny, stated rather than skipped. Both are NP-hard, and both are the same minimum
feedback arc set that ``revealed.min_feedback_arcs`` already computes: Slater on the unweighted
majority digraph, Kemeny on the margin-weighted one. Inside a strongly connected component of at
most eight alternatives that computation is solved exactly by enumeration, which covers every real
group size this instrument has been pointed at. Above eight it returns a bracket, from a greedy
ordering above and an arc-disjoint cycle packing below, and the reading says which of the two
happened. Neither is registered as a quantity because the catalogue carries no row for either.

The catalogue's rung 2 for B6 is the Copeland-versus-Slater disagreement, and **no scalar
disagreement rate is reported**, here or in the registry. Both rules tie routinely: over 3,000
random five-item tournaments the Copeland maximum is tied on 1,233 and the Slater optimum has more
than one head on 892, so any single rate is a choice of tie-break. The achievable range over
tie-breaks on that bank is [0.0000, 0.5250] against the 0.2150 that taking `argmax` and the first
enumerated ordering produces, and relabelling the alternatives alone moves that convention between
0.2010 and 0.2157. What travels on the reading instead is the three-way split into forced agreement,
forced disagreement and undetermined, computed on the winner *sets*, which carry no tie-break.

Kill condition, verbatim: *if Condorcet cycles are under 1% at real group sizes.*
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Sequence, cast

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import Evidence, make_evidence
from reward_lens.core.invariance import INVARIANT, InvariancePayload
from reward_lens.core.quantity import (
    QUANTITIES,
    BaselineID,
    BiasDirection,
    BiasStatement,
    CostModel,
    EstimatorEntry,
    load_quantities,
    register_estimator,
)
from reward_lens.core.reading import Reading, Refusal, RefusalReason, bounded_refusal
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    SubjectRef,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.composition.revealed import (
    ComparisonSet,
    bank_from_payload,
    bank_from_scores,
    min_feedback_arcs,
    transitive_closure,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _ensure_quantities() -> None:
    if "grader.condorcet_rate" not in QUANTITIES:
        load_quantities()


_ensure_quantities()


#: The catalogue's baseline for B6: "assume a total order exists". Under that assumption a Condorcet
#: winner is present on every prompt and the Smith set is a singleton on every prompt, so the
#: baseline values are 1.0 and 1.0 and the instrument's job is to report the gap.
TOTAL_ORDER: BaselineID = "baseline.assume_total_order"

#: A matched random tournament on the same comparison graph. Needed because the cycle rate at K = 4
#: is small under *any* generating process, so a small measured rate is not evidence of a coherent
#: grader until chance has been quoted on the same design.
RANDOM_TOURNAMENT: BaselineID = "baseline.random_tournament"

#: The same annotation budget spent on randomly chosen pairs. The connectivity instrument uses this
#: one instead, because λ₂ depends on which pairs were compared and not on who won, so redrawing the
#: verdicts leaves it exactly unchanged and would be a baseline that cannot fail.
RANDOM_DESIGN: BaselineID = "baseline.random_comparison_design"

TOURNAMENT_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "a census over a comparison record: it counts which alternatives beat which, and asserts "
        "nothing about the process that produced the verdicts, so no regime can make the count "
        "wrong. B6's source entry prints no Env column and the catalogue records it as OPEN; this "
        "is the reading of that gap, and it is a positive claim rather than a default."
    ),
)


# ---------------------------------------------------------------------------
# Solution concepts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CondorcetReading:
    """Whether one prompt has a Condorcet winner, and if not, why not."""

    winner: int | None
    n_items: int
    complete: bool
    has_cycle: bool
    #: Alternatives that beat everything they were compared with but did not meet every rival. On a
    #: complete graph this is empty; on an incomplete one each of these is a Condorcet winner that
    #: one more comparison would confirm or refute.
    possible: tuple[int, ...] = ()
    missing_pairs: tuple[tuple[int, int], ...] = ()

    @property
    def reason(self) -> str:
        if self.winner is not None:
            return "condorcet winner"
        if self.has_cycle:
            return "cycle"
        if not self.complete:
            return "incomplete comparison graph"
        return "no dominant alternative"


def condorcet(cs: ComparisonSet) -> CondorcetReading:
    """The Condorcet winner of one group, with the reason there is none when there is none.

    A record that holds more than one row for a pair is collapsed to its majority first. Every
    concept in this module is defined on one verdict per pair, and reading `dominance` directly on a
    panel means "beat at least once", under which a twelve-member panel puts almost every pair in
    both directions and almost every alternative comes out a Condorcet winner. Measured on the
    1,763-prompt twelve-model panel read as scores: 100.0% Condorcet rate against a 0.2836% cycle
    rate on the same data once collapsed.
    """
    cs = cs.majority_tournament()
    n = cs.n_items
    dom = cs.dominance()
    obs = cs.observed()
    off = ~np.eye(n, dtype=bool)
    missing = np.argwhere(off & ~obs)
    complete = missing.shape[0] == 0
    pairs = tuple(sorted({(int(min(a, b)), int(max(a, b))) for a, b in missing}))
    reach = transitive_closure(dom)
    has_cycle = bool(np.any(np.diag(reach)))
    winner: int | None = None
    possible: list[int] = []
    for x in range(n):
        others = [y for y in range(n) if y != x]
        if not others:
            continue
        beat = [y for y in others if dom[x, y]]
        met = [y for y in others if obs[x, y]]
        if len(beat) == len(others):
            winner = x
        elif len(beat) == len(met) and met:
            possible.append(x)
    return CondorcetReading(
        winner=winner,
        n_items=n,
        complete=complete,
        has_cycle=has_cycle,
        possible=tuple(possible),
        missing_pairs=pairs,
    )


def copeland_scores(cs: ComparisonSet) -> np.ndarray:
    """Wins minus losses, per alternative. Polynomial, and the rule most tournament rewards use.

    Repeated rows for a pair are collapsed to their majority first, so a pair counts once however
    many times it was observed. That is what Copeland's rule is: a count over pairs, not over
    observations.
    """
    dom = cs.majority_tournament().dominance()
    return dom.sum(axis=1).astype(np.float64) - dom.sum(axis=0).astype(np.float64)


def copeland_winners(cs: ComparisonSet) -> tuple[int, ...]:
    """Every alternative attaining the maximum Copeland score. Usually more than one.

    Reported as a set rather than a winner because the maximum is tied on 1,233 of 3,000 random
    five-item tournaments, and picking one of them with `argmax` is picking by item index.
    """
    scores = copeland_scores(cs)
    if not scores.size:
        return ()
    return tuple(int(v) for v in np.flatnonzero(scores >= float(scores.max()) - 1e-12))


def slater_winners(cs: ComparisonSet, *, exact_max: int = 8) -> tuple[int, ...]:
    """Every alternative heading at least one Slater-optimal ranking.

    Slater's rule is a minimum feedback arc set, and a minimum is routinely attained by several
    orderings with different heads: 892 of the same 3,000 five-item tournaments have more than one.
    Enumerated exactly inside strongly connected components of at most ``exact_max`` alternatives.
    Above that the optimum is not available, the heads of the greedy ordering are returned, and
    `FeedbackArcs.exact` on the same tournament is what says so.
    """
    dom = cs.majority_tournament().dominance()
    n = cs.n_items
    if n == 0:
        return ()
    if n > exact_max:
        fa = min_feedback_arcs(dom.astype(np.float64), exact_max=exact_max)
        return (int(fa.order[0]),) if fa.order else ()
    best: float | None = None
    heads: set[int] = set()
    for perm in itertools.permutations(range(n)):
        position = np.empty(n, dtype=np.int64)
        position[list(perm)] = np.arange(n)
        cost = float(np.sum(dom & (position[:, None] > position[None, :])))
        if best is None or cost < best - 1e-12:
            best, heads = cost, {int(perm[0])}
        elif abs(cost - best) <= 1e-12:
            heads.add(int(perm[0]))
    return tuple(sorted(heads))


def smith_set(cs: ComparisonSet) -> tuple[int, ...]:
    """The smallest set whose every member beats every alternative outside it.

    Built by closing upward from each possible seed and keeping the smallest closed set, which is
    exact and cheap at the group sizes a reward loop uses. Closing from the Copeland winner alone is
    the usual shortcut and it is only correct on a complete tournament; a comparison graph with a
    missing pair can put the minimal dominant set somewhere the Copeland winner is not.

    An unobserved pair is not a win, so a missing comparison **enlarges** the Smith set rather than
    shrinking it. That is the conservative direction and it is the honest one: an alternative nobody
    compared cannot be excluded from the set the reward might pick from.
    """
    n = cs.n_items
    if n == 0:
        return ()
    dom = cs.majority_tournament().dominance()
    best: tuple[int, ...] | None = None
    for seed in range(n):
        members = {seed}
        while True:
            outside = [x for x in range(n) if x not in members]
            add = [x for x in outside if not all(dom[y, x] for y in members)]
            if not add:
                break
            members.update(add)
        candidate = tuple(sorted(members))
        if best is None or len(candidate) < len(best):
            best = candidate
    return best or ()


def slater(cs: ComparisonSet, *, exact_max: int = 8) -> Any:
    """Slater's rule: the ranking that reverses the fewest verdicts. NP-hard, so bounded above 8.

    This is `min_feedback_arcs` on the unweighted majority digraph. Inside a strongly connected
    component of at most ``exact_max`` alternatives it enumerates the orderings and the answer is
    the optimum; above that it returns the greedy ordering and brackets the cost, and
    ``FeedbackArcs.exact`` says which happened.
    """
    return min_feedback_arcs(
        cs.majority_tournament().dominance().astype(np.float64), exact_max=exact_max
    )


def kemeny(cs: ComparisonSet, *, exact_max: int = 8) -> Any:
    """Kemeny's rule on the margin-weighted digraph, which is the same computation as Slater.

    Named and provided rather than skipped, because the only difference from `slater` is the weight
    on each arc: Slater counts a reversed verdict once, Kemeny counts it by the margin it reverses.
    Reported nowhere by default, because the catalogue registers no quantity for it.

    The arc weight is the **total** margin over the rows recording that pair, not their mean:
    reversing a pair in the consensus ranking reverses every observation of it, so a pair observed
    ten times costs ten times as much to reverse as one observed once.
    """
    return min_feedback_arcs(cs.margin_matrix(aggregate="sum"), exact_max=exact_max)


# ---------------------------------------------------------------------------
# Connectivity, and the annotation budget it licenses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Connectivity:
    """The comparison graph's algebraic connectivity, and what it says about the design."""

    lambda2: float
    n_items: int
    n_edges: int
    n_components: int
    method: str
    #: The unobserved pairs, so a remedy can name them rather than gesture at them.
    missing_pairs: tuple[tuple[int, int], ...] = ()

    @property
    def connected(self) -> bool:
        return self.n_components <= 1 and self.lambda2 > 0.0

    def error_bound(self, replications: int = 1) -> float:
        """``√(n log n / (L λ₂))``: the order of the ranking error this design admits.

        **Source.** Shah, Balakrishnan, Bradley, Parekh, Ramchandran and Wainwright, "Estimation
        from Pairwise Comparisons: Sharp Minimax Bounds with Topology Dependence", JMLR 17(58),
        2016, Theorem 2 equation (8b): for the constrained maximum-likelihood estimator under an
        ordinal comparison model,
        ``P[‖ŵ − w*‖₂² > t·c(ζ²/γ²)σ²·d/(λ₂(L)·N)] ≤ e^{−t}``, where ``d`` is the number of items,
        ``N`` the number of samples, and ``L`` **the sample-normalised Laplacian**, whose entry
        ``L_jk`` is the *fraction* of samples comparing items ``j`` and ``k``. Setting ``t = log d``
        recovers the high-probability form.

        **Why λ₂ here is the raw Laplacian's and that is not an error.** Under the design this
        instrument reads, each of the ``|E|`` observed pairs is compared ``L`` times, so
        ``N = L·|E|`` and ``λ₂(L_normalised) = λ₂(L_raw)/|E|``. The ``|E|`` cancels exactly:

            d·log d / (λ₂(L_norm)·N) = d·log d / ((λ₂(L_raw)/|E|)·L·|E|) = d·log d / (L·λ₂(L_raw))

        Checked two ways. Numerically, ``λ₂(L_norm) = λ₂(L_raw)/|E|`` holds to float precision at
        four, six, ten and twenty items, and the cancellation is exact at all four. Against the
        paper's own arithmetic, Shah et al. state ``1/λ₂(L) = (d−1)/2`` for the complete graph,
        which is what ``λ₂(L_raw)/|E| = d/(d(d−1)/2)`` gives. The raw-Laplacian restatement is also
        in print: Li, Shrotriya and Rinaldo, "ℓ∞-Bounds of the MLE in the BTL Model under General
        Comparison Graphs", UAI 2022, Table 1, gives the Shah et al. row as
        ``n log n / (λ₂(L_A)·L)`` with ``L_A := D − A`` the unnormalised Laplacian.

        **A notation collision worth naming**, because it is what makes this look wrong: ``n`` here
        is the number of **items**, and ``n`` in Shah et al. is the number of **samples**, where the
        items are ``d``. The logarithm is ``log(items)``.

        **What the number is not.** The constant absorbs ``ζ²/γ²`` and ``σ²``, which depend on the
        link function's curvature and the noise scale, so this is a scaling and not a guarantee. It
        compares designs; it does not price an absolute annotation budget, and
        `replications_ratio_for` rather than an absolute count is what this class will report.

        **Scope.** The derivation above assumes every observed pair was compared the same number of
        times. On an unequally replicated design the substitution ``N = L·|E|`` does not hold and
        the result is an order-of-magnitude reading only. It also assumes a connected comparison
        graph; a disconnected one has ``λ₂ = 0``, the bound is infinite, and `ComparisonConnectivity`
        refuses rather than reporting it.
        """
        n = max(self.n_items, 2)
        if self.lambda2 <= 0 or replications <= 0:
            return float("inf")
        return math.sqrt(n * math.log(n) / (replications * self.lambda2))

    def replications_ratio_for(self, other: "Connectivity") -> float:
        """How many times as many comparisons per pair this design needs to match ``other``.

        This is the form of the budget question the bound can actually answer. Both designs carry
        the same unknown constant, so it divides out and the ratio is a real number rather than a
        number-shaped scaling. A ratio of 3.2 means this design needs 3.2 times the annotation of
        the other one to reach the same ranking accuracy.

        An **absolute** count does not survive the same division, which is why this class does not
        offer one. The reading used to print "reaching 0.1 needs about 155.4 comparisons per pair",
        and 155.4 is ``n log n / (λ₂ · 0.01)`` with the constant silently set to 1. The constant is
        ``c·ζ²σ²/γ²`` in the source and is not 1, is not known here, and is not close to 1 for any
        particular reason, so the sentence stated a budget the bound does not license.
        """
        mine = self.error_bound(1)
        theirs = other.error_bound(1)
        if not math.isfinite(mine) or theirs <= 0.0:
            return float("inf")
        return (mine / theirs) ** 2

    def relative_budget(self, target_ratio: float) -> float:
        """The replications needed to cut this design's order-of-error by ``target_ratio``.

        ``relative_budget(2.0)`` is "how many comparisons per pair to halve the error", which is 4
        because the bound falls as ``1/√L``. Exact, constant-free and useful; the absolute count
        that used to be printed beside it was neither.
        """
        if target_ratio <= 0:
            return float("inf")
        return float(target_ratio**2)


def algebraic_connectivity(cs: ComparisonSet, *, sparse_above: int = 32) -> Connectivity:
    """λ₂ of the comparison graph's Laplacian.

    Dense `numpy.linalg.eigvalsh` at or below ``sparse_above`` items and
    `scipy.sparse.linalg.eigsh` above it. The dense route is exact and it is the one that fires at
    every group size a reward loop uses; the sparse route exists for the aggregate design graph over
    a whole annotation campaign, where the item set is large and the graph is sparse.
    """
    n = cs.n_items
    obs = cs.observed()
    off = ~np.eye(n, dtype=bool) if n else np.zeros((0, 0), dtype=bool)
    missing = np.argwhere(off & ~obs)
    pairs = tuple(sorted({(int(min(a, b)), int(max(a, b))) for a, b in missing}))
    n_edges = int(obs.sum() // 2)
    if n < 2:
        return Connectivity(0.0, n, n_edges, max(n, 0), "degenerate", pairs)
    degree = obs.sum(axis=1).astype(np.float64)
    laplacian = np.diag(degree) - obs.astype(np.float64)
    n_components = int(n - np.linalg.matrix_rank(laplacian))
    if n <= sparse_above:
        vals = np.linalg.eigvalsh(laplacian)
        lam2 = float(vals[1])
        method = "dense eigvalsh"
    else:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh

        vals = eigsh(csr_matrix(laplacian), k=2, sigma=-1e-6, which="LM", return_eigenvectors=False)
        lam2 = float(np.sort(vals)[1])
        method = "scipy.sparse.linalg.eigsh, shift-invert at 0"
    if lam2 < 1e-12:
        lam2 = 0.0
    return Connectivity(lam2, n, n_edges, max(n_components, 1), method, pairs)


def pair_to_add(cs: ComparisonSet) -> tuple[tuple[int, int], float] | None:
    """The unobserved pair that raises λ₂ most, and the λ₂ it would give.

    This is the remedy a connectivity refusal carries. "Compare responses 1 and 3" is an instruction
    an annotation pipeline can act on; "the comparison graph is too sparse" is not.
    """
    base = algebraic_connectivity(cs)
    if not base.missing_pairs:
        return None
    best: tuple[tuple[int, int], float] | None = None
    for i, j in base.missing_pairs:
        probe = ComparisonSet(
            n_items=cs.n_items,
            winners=np.concatenate([cs.winners, [i]]).astype(np.intp),
            losers=np.concatenate([cs.losers, [j]]).astype(np.intp),
            margins=np.concatenate([cs.margins, [1.0]]),
            weights=np.concatenate([cs.weights, [1.0]]),
            margin_source=cs.margin_source,
        )
        lam = algebraic_connectivity(probe).lambda2
        if best is None or lam > best[1]:
            best = ((i, j), lam)
    return best


# ---------------------------------------------------------------------------
# Bank-level summaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TournamentSummary:
    """Every B6 number over a bank of prompts, computed in one pass.

    Two rates about cycles, because they answer different questions and one of them used to be
    reported under the other's name. ``cycle_rate`` is the share of prompts whose Condorcet winner
    is **missing because of a cycle**, which is the decomposition of ``no_condorcet_rate`` the
    catalogue's kill condition is written against. ``any_cycle_rate`` is the share of prompts that
    contain a cycle anywhere, which can be larger and carries no implication about the winner: a
    three-cycle among the also-rans with a fourth alternative beating all of them gives an
    ``any_cycle_rate`` of 1.0 with a Condorcet winner on every prompt. As shipped, the second number
    was returned under the first name, so ``cycle_rate`` could exceed ``no_condorcet_rate`` and the
    two could not be added up.

    The Copeland-versus-Slater comparison is three numbers rather than one, and the reason is in
    `copeland_slater_split`.
    """

    n_groups: int
    n_scored: int
    condorcet_rate: float
    no_condorcet_rate: float
    cycle_rate: float
    any_cycle_rate: float
    incomplete_rate: float
    smith_sizes: np.ndarray
    mean_smith_size: float
    modal_smith_size: int
    max_smith_size: int
    lambda2: np.ndarray
    mean_lambda2: float
    min_lambda2: float
    max_error_bound: float
    disconnected_groups: int
    #: Groups on which no tie-break can make Copeland and Slater pick the same alternative.
    copeland_slater_forced_disagreements: int
    #: Groups on which both rules have a unique winner and it is the same one.
    copeland_slater_forced_agreements: int
    #: Groups on which at least one rule is tied, so the answer is decided by the tie-break.
    copeland_slater_undetermined: int
    slater_exact: bool
    n_items: np.ndarray

    @property
    def copeland_slater_range(self) -> tuple[float, float]:
        """The achievable disagreement rate over all tie-breaks, as an interval.

        The lower end is the forced-disagreement rate and the upper end adds every undetermined
        group. A scalar between them is a choice of tie-break and not a measurement.
        """
        denom = self.n_scored if self.n_scored else 1
        lo = self.copeland_slater_forced_disagreements / denom
        hi = lo + self.copeland_slater_undetermined / denom
        return (lo, hi)


def copeland_slater_split(cs: ComparisonSet, *, exact_max: int = 8) -> str:
    """Whether the rule choice is forced, forced the other way, or left to the tie-break.

    Returns ``"agree"``, ``"disagree"`` or ``"undetermined"``.

    The question B6's rung 2 asks is whether it matters which tournament solution concept a group
    reward implements. Answering it with one number requires picking one Copeland winner and one
    Slater winner, and both are routinely tied: over 3,000 random five-item tournaments the Copeland
    maximum is tied on 1,233 and the Slater optimum has more than one head on 892. The shipped
    scalar took `argmax` and the first enumerated ordering, so it was reading the item indices: the
    same 3,000 tournaments relabelled give 0.2010 to 0.2157, and the range achievable by choosing
    tie-breaks freely is [0.0000, 0.5250] against a reported 0.2150.

    So the comparison is made on the winner **sets**, which have no tie-break in them. Disjoint sets
    mean no tie-break can reconcile the two rules; two singletons that agree mean every tie-break
    reconciles them; anything else is undetermined, and saying so is the honest answer.
    """
    cope = set(copeland_winners(cs))
    slat = set(slater_winners(cs, exact_max=exact_max))
    if not cope or not slat:
        return "undetermined"
    if not (cope & slat):
        return "disagree"
    if len(cope) == 1 and len(slat) == 1:
        return "agree"
    return "undetermined"


def summarise(bank: Sequence[ComparisonSet], *, exact_max: int = 8) -> TournamentSummary:
    """Condorcet, Smith, λ₂ and the Copeland-versus-Slater split, over a bank."""
    scored = [cs for cs in bank if cs.n_items >= 2]
    condorcet_hits = 0
    cycles_without_winner = 0
    any_cycle = 0
    incomplete = 0
    smith: list[int] = []
    lam: list[float] = []
    bounds: list[float] = []
    disconnected = 0
    forced_disagree = 0
    forced_agree = 0
    undetermined = 0
    slater_exact = True
    sizes: list[int] = []
    for cs in scored:
        c = condorcet(cs)
        condorcet_hits += int(c.winner is not None)
        any_cycle += int(c.has_cycle)
        cycles_without_winner += int(c.winner is None and c.has_cycle)
        incomplete += int(not c.complete)
        smith.append(len(smith_set(cs)))
        conn = algebraic_connectivity(cs)
        lam.append(conn.lambda2)
        bounds.append(conn.error_bound(1))
        disconnected += int(not conn.connected)
        sizes.append(cs.n_items)
        fa = slater(cs, exact_max=exact_max)
        slater_exact = slater_exact and fa.exact
        verdict = copeland_slater_split(cs, exact_max=exact_max)
        forced_disagree += int(verdict == "disagree")
        forced_agree += int(verdict == "agree")
        undetermined += int(verdict == "undetermined")
    n = len(scored)
    denom = n if n else 1
    smith_arr = np.asarray(smith or [0], dtype=np.int64)
    lam_arr = np.asarray(lam or [0.0], dtype=np.float64)
    counts = np.bincount(smith_arr)
    return TournamentSummary(
        n_groups=len(bank),
        n_scored=n,
        condorcet_rate=condorcet_hits / denom,
        no_condorcet_rate=1.0 - condorcet_hits / denom,
        cycle_rate=cycles_without_winner / denom,
        any_cycle_rate=any_cycle / denom,
        incomplete_rate=incomplete / denom,
        smith_sizes=smith_arr,
        mean_smith_size=float(smith_arr.mean()),
        modal_smith_size=int(np.argmax(counts)),
        max_smith_size=int(smith_arr.max()),
        lambda2=lam_arr,
        mean_lambda2=float(lam_arr.mean()),
        min_lambda2=float(lam_arr.min()),
        max_error_bound=float(max(bounds)) if bounds else float("inf"),
        disconnected_groups=disconnected,
        copeland_slater_forced_disagreements=forced_disagree,
        copeland_slater_forced_agreements=forced_agree,
        copeland_slater_undetermined=undetermined,
        slater_exact=slater_exact,
        n_items=np.asarray(sizes or [0], dtype=np.int64),
    )


def random_design_lambda2(
    bank: Sequence[ComparisonSet], *, n_draws: int = 16, seed: int = 0
) -> float:
    """Mean λ₂ of a design that spends the same number of comparisons on randomly chosen pairs.

    λ₂ is a property of *which pairs were compared* and not of who won, so redrawing the verdicts
    leaves it exactly unchanged and a verdict-shuffling baseline here would be tautological. The
    baseline that answers the annotation-budget question is the one that holds the budget fixed and
    redraws the design: at the same cost, is this set of pairs better connected than an arbitrary
    one?
    """
    rng = np.random.default_rng(seed)
    means: list[float] = []
    for _ in range(n_draws):
        vals: list[float] = []
        for cs in bank:
            n = cs.n_items
            if n < 2:
                continue
            all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            k = min(len(all_pairs), cs.n_comparisons)
            picked: list[int] = (
                [int(p) for p in rng.choice(len(all_pairs), size=k, replace=False)] if k else []
            )
            chosen = [all_pairs[p] for p in picked]
            probe = ComparisonSet(
                n_items=n,
                winners=np.asarray([i for i, _ in chosen], dtype=np.intp),
                losers=np.asarray([j for _, j in chosen], dtype=np.intp),
                margins=np.ones(len(chosen)),
                weights=np.ones(len(chosen)),
            )
            vals.append(algebraic_connectivity(probe).lambda2)
        means.append(float(np.mean(vals)) if vals else 0.0)
    return float(np.mean(means)) if means else 0.0


def random_tournament_summary(
    bank: Sequence[ComparisonSet], *, n_draws: int = 32, seed: int = 0
) -> dict[str, float]:
    """The same numbers for a grader choosing at random on the same comparison graph.

    The graph is held fixed and only the directions are redrawn, so the baseline prices chance on
    *this* design. It is the number that decides whether a measured 0.3% cycle rate at K = 4 is a
    coherent grader or a design that can barely express a cycle.
    """
    rng = np.random.default_rng(seed)
    cond: list[float] = []
    cyc: list[float] = []
    smith: list[float] = []
    for _ in range(n_draws):
        drawn: list[ComparisonSet] = []
        for cs in bank:
            if not cs.n_comparisons:
                drawn.append(cs)
                continue
            flip = rng.random(cs.n_comparisons) < 0.5
            drawn.append(
                ComparisonSet(
                    n_items=cs.n_items,
                    winners=np.where(flip, cs.losers, cs.winners).astype(np.intp),
                    losers=np.where(flip, cs.winners, cs.losers).astype(np.intp),
                    margins=cs.margins,
                    weights=cs.weights,
                    margin_source=cs.margin_source,
                )
            )
        s = summarise(drawn)
        cond.append(s.condorcet_rate)
        cyc.append(s.cycle_rate)
        smith.append(s.mean_smith_size)
    return {
        "condorcet_rate": float(np.mean(cond)),
        "condorcet_rate_sd": float(np.std(cond)),
        "cycle_rate": float(np.mean(cyc)),
        "cycle_rate_sd": float(np.std(cyc)),
        "mean_smith_size": float(np.mean(smith)),
        "n_draws": float(n_draws),
    }


# ---------------------------------------------------------------------------
# The instruments
# ---------------------------------------------------------------------------


class _B6Instrument(BaseObservable):
    """Shared declarations for B6's three quantities, which read one record in one pass."""

    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to: str | None = "B6"
    requires = {Component.GRADER: Access.RECORD}
    substrates = frozenset({Substrate.PROCEDURAL})
    #: The catalogue's phase column is OPEN for B6. Empty means no restriction.
    phases = frozenset()
    envelope = TOURNAMENT_ENVELOPE
    baselines = (TOTAL_ORDER, RANDOM_TOURNAMENT)
    rung = 0

    def __init__(
        self,
        bank: Sequence[ComparisonSet] | None = None,
        *,
        exact_max: int = 8,
        baseline_draws: int = 16,
        seed: int = 0,
    ) -> None:
        self.bank: tuple[ComparisonSet, ...] = tuple(bank or ())
        self.exact_max = exact_max
        self.baseline_draws = baseline_draws
        self.seed = seed
        self._computed: Any = None

    def _empty_refusal(self) -> Refusal:
        return Refusal(
            instrument=self.name,
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=(
                f"{len(self.bank)} groups, none of them holding two or more alternatives. A "
                f"tournament over one alternative has no solution concept to implement."
            ),
            remedy=(
                "supply groups of at least two alternatives, through `bank_from_scores` on a "
                "recorded score bank or `ComparisonSet.from_edges` on a tournament log."
            ),
            statistics={"n_groups": len(self.bank)},
        )

    def _summary(self) -> Any:
        if not any(cs.n_items >= 2 for cs in self.bank):
            return self._empty_refusal()
        return summarise(self.bank, exact_max=self.exact_max)

    def measure(self, ctx: Context) -> Evidence:
        return ctx.emit(self.payload(self._computed))

    def payload(self, computed: Any) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def compute(self) -> Any:
        return self._summary()

    def estimate(self, ctx: Context) -> Reading:
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        out = self.compute()
        if isinstance(out, Refusal):
            return out
        self._computed = out
        ctx._observable = self
        try:
            return self.measure(ctx)
        finally:
            ctx._observable = None


class CondorcetRate(_B6Instrument):
    """B6. How often a Condorcet winner exists, and what the reward does when it does not.

    ``grader.condorcet_rate`` is reported here as the fraction of prompts on which a Condorcet
    winner **exists**, with ``no_condorcet_rate`` carried beside it so the sign can never be read
    backwards. The failures are split by cause, because a cycle and a missing comparison are
    different problems: the first is the grader disagreeing with itself and no further annotation
    will remove it; the second is a design gap and one more comparison may close it.

    Kill condition, verbatim: *if Condorcet cycles are under 1% at real group sizes.* The cycle rate
    is reported separately from the Condorcet failure rate for exactly that reason, since a bank can
    have a large Condorcet failure rate and almost no cycles.
    """

    name = "CondorcetRate"
    quantity = "grader.condorcet_rate"
    invariance = "group.permutation"
    invariance_relation = INVARIANT
    deviations = (
        "The catalogue's B6 record names `grader.condorcet` alongside `grader.condorcet_rate`; the "
        "first is an alternate spelling, was never registered, and has since been removed from the "
        "record. Only the registered id is estimated here.",
        "The quantity's `definition` field is OPEN in the registry. The convention used is that "
        "the rate counts prompts where a Condorcet winner is present; the complement travels on "
        "the same reading under `no_condorcet_rate`.",
    )

    def payload(self, computed: TournamentSummary) -> dict[str, Any]:
        base = random_tournament_summary(self.bank, n_draws=self.baseline_draws, seed=self.seed)
        denom = max(computed.n_scored, 1)
        lo, hi = computed.copeland_slater_range
        return {
            "condorcet_rate": computed.condorcet_rate,
            "no_condorcet_rate": computed.no_condorcet_rate,
            "cycle_rate": computed.cycle_rate,
            "any_cycle_rate": computed.any_cycle_rate,
            "incomplete_rate": computed.incomplete_rate,
            "n_groups": computed.n_scored,
            "mean_group_size": float(computed.n_items.mean()),
            "copeland_slater_forced_disagreement_rate": (
                computed.copeland_slater_forced_disagreements / denom
            ),
            "copeland_slater_forced_agreement_rate": (
                computed.copeland_slater_forced_agreements / denom
            ),
            "copeland_slater_undetermined_rate": computed.copeland_slater_undetermined / denom,
            "copeland_slater_disagreement_range": [lo, hi],
            "copeland_slater_note": (
                "no single disagreement rate is reported, because the Copeland maximum and the "
                "Slater optimum are both routinely tied and any scalar is a choice of tie-break: "
                "over 3,000 random five-item tournaments the achievable range is [0.0000, 0.5250] "
                "and relabelling the alternatives alone moves the argmax convention from 0.2010 to "
                "0.2157. The forced rates carry no tie-break and are the numbers to read."
            ),
            "slater_exact": computed.slater_exact,
            "baseline_random_condorcet_sd": base["condorcet_rate_sd"],
            "baselines": {
                TOTAL_ORDER: 1.0,
                RANDOM_TOURNAMENT: base["condorcet_rate"],
            },
        }


class SmithSetSize(_B6Instrument):
    """B6. How many alternatives the reward is choosing between when it cannot choose one.

    The Smith set is a singleton exactly when a Condorcet winner exists. That is a theorem and not a
    measurement: ``{x}`` is dominant precisely when ``x`` beats every other alternative, which is the
    definition of a Condorcet winner, so the identity cannot fail for two or more alternatives and it
    holds on 4,000 random designs including incomplete ones. `singleton_rate` and B6's Condorcet rate
    agreeing is a check that the code preserves the identity, and reporting the two as separate
    agreeing numbers is not evidence about a grader.

    When the set is larger, every tournament rule picks somewhere inside it and they disagree about
    where, so its size is the number of responses a group reward is picking among arbitrarily.
    """

    name = "SmithSetSize"
    quantity = "grader.smith_set_size"
    invariance = "group.permutation"
    invariance_relation = INVARIANT

    def payload(self, computed: TournamentSummary) -> dict[str, Any]:
        base = random_tournament_summary(self.bank, n_draws=self.baseline_draws, seed=self.seed)
        sizes = computed.smith_sizes
        return {
            "smith_set_size": computed.mean_smith_size,
            "modal_smith_set_size": computed.modal_smith_size,
            "max_smith_set_size": computed.max_smith_size,
            "singleton_rate": float((sizes == 1).mean()),
            "n_groups": computed.n_scored,
            "mean_group_size": float(computed.n_items.mean()),
            "incomplete_rate": computed.incomplete_rate,
            "baselines": {
                TOTAL_ORDER: 1.0,
                RANDOM_TOURNAMENT: base["mean_smith_size"],
            },
        }


class ComparisonConnectivity(_B6Instrument):
    """B6. What the comparison design allows, before anybody looks at the verdicts.

    λ₂ of the comparison graph's Laplacian governs the statistical efficiency of the whole
    preference dataset through ``||v̂ - v||₂ ≲ √(n log n / (L λ₂(Δ₀)))``, so it is the quantity that
    turns an annotation budget into an accuracy. Two designs with the same number of comparisons can
    differ by an order of magnitude in λ₂, and the cheaper one is the one nobody computes.

    Refuses when the comparison graph is disconnected. λ₂ is then exactly 0, the error bound is
    infinite, and every ranking across the components is unidentified, so a number here would be a
    zero passed downstream as if it were a measurement. The refusal carries the λ₂ it found and a
    remedy naming the pair whose comparison raises it most.
    """

    name = "ComparisonConnectivity"
    quantity = "grader.graph_lambda2"
    baselines = (TOTAL_ORDER, RANDOM_DESIGN)
    #: The registry declares `none` for this quantity, which `load_quantities` resolves to the
    #: trivial group. Declared as printed rather than upgraded: a rescaling of the reward does not
    #: touch which pairs were compared, and the generated test for the trivial group passes
    #: vacuously and says so.
    invariance = "trivial"
    invariance_relation = INVARIANT
    deviations = (
        "The registry gives this quantity the group `none`. It is in fact invariant under "
        "`group.permutation`, since relabelling the alternatives is a similarity transform of the "
        "Laplacian, and the unit tests check that. The declaration follows the registry rather "
        "than the observation, because changing an invariance group assignment is not this "
        "instrument's decision to make.",
    )

    def compute(self) -> Any:
        out = self._summary()
        if isinstance(out, Refusal):
            return out
        if out.disconnected_groups:
            worst = min(
                (cs for cs in self.bank if cs.n_items >= 2),
                key=lambda cs: algebraic_connectivity(cs).lambda2,
            )
            suggestion = pair_to_add(worst)
            named = (
                f"compare alternatives {suggestion[0][0]} and {suggestion[0][1]} in the worst "
                f"group, which raises its λ₂ from 0 to {suggestion[1]:.4g}"
                if suggestion
                else "add comparisons joining the disconnected halves of the worst group"
            )
            return bounded_refusal(
                self.name,
                RefusalReason.ABOVE_LOD_BELOW_LOQ,
                detail=(
                    f"{out.disconnected_groups} of {out.n_scored} comparison graphs are "
                    f"disconnected, so λ₂ = 0 on them and the error bound √(n log n / (L λ₂)) is "
                    f"infinite. Across a disconnected graph the ranking is not identified at any "
                    f"annotation budget, so the reading would be a zero standing in for a "
                    f"quantity that does not exist yet."
                ),
                remedy=(
                    f"{named}. Then re-run: λ₂ over the connected groups is currently "
                    f"{out.mean_lambda2:.4g} and the bound is finite there."
                ),
                bound=make_evidence(
                    observable=self.name,
                    observable_version=self.version,
                    subject=SubjectRef(readout="comparisons"),
                    value={
                        "graph_lambda2": 0.0,
                        "graph_lambda2_connected_groups": out.mean_lambda2,
                        "disconnected_groups": out.disconnected_groups,
                        "n_groups": out.n_scored,
                    },
                    quantity=self.quantity,
                ),
                disconnected_groups=out.disconnected_groups,
                mean_lambda2=out.mean_lambda2,
            )
        return out

    def payload(self, computed: TournamentSummary) -> dict[str, Any]:
        n = float(computed.n_items.mean())
        # The bound is evaluated per group and the worst group is the one reported, because a bound
        # is only a bound if it holds. Evaluating √(n log n / (L λ₂)) at the mean group size and the
        # mean λ₂ bounds nothing: on a 400-group bank of mixed sizes it read 2.4761 while the worst
        # group needed 5.0355, and it failed to bound 185 of the 400.
        per_group = [algebraic_connectivity(cs) for cs in self.bank if cs.n_items >= 2]
        bounds = [c.error_bound(1) for c in per_group] or [float("inf")]
        worst = max(bounds)
        typical = Connectivity(
            lambda2=computed.mean_lambda2,
            n_items=int(round(n)),
            n_edges=0,
            n_components=1,
            method="mean over groups",
        )
        complete = Connectivity(
            lambda2=float(round(n)),
            n_items=int(round(n)),
            n_edges=0,
            n_components=1,
            method="the complete comparison on the mean group size",
        )
        return {
            "graph_lambda2": computed.mean_lambda2,
            "graph_lambda2_min": computed.min_lambda2,
            "n_groups": computed.n_scored,
            "mean_group_size": n,
            "error_bound_worst_group_L1": worst,
            "error_bound_worst_group_L11": worst / math.sqrt(11.0),
            "error_bound_median_group_L1": float(np.median(bounds)),
            "error_bound_at_the_mean_L1": typical.error_bound(1),
            "groups_above_the_mean_bound": int(
                sum(1 for b in bounds if b > typical.error_bound(1))
            ),
            "budget_ratio_against_complete_design": typical.replications_ratio_for(complete),
            "replications_to_halve_the_error": typical.relative_budget(2.0),
            "annotation_budget_reading": (
                f"at {int(round(n))} alternatives, the worst group's λ₂ of "
                f"{computed.min_lambda2:.4g} admits a ranking error of order {worst:.4g} at one "
                f"comparison per pair, against {typical.error_bound(1):.4g} at the bank's mean λ₂ "
                f"of {computed.mean_lambda2:.4g}; {int(sum(1 for b in bounds if b > typical.error_bound(1)))} "
                f"of {len(bounds)} groups sit above the mean figure, so read the worst. Cutting the "
                f"error in half costs 4 comparisons per pair, and this design costs "
                f"{typical.replications_ratio_for(complete):.4g} times what the complete comparison "
                f"on the same items would for the same accuracy. No absolute comparison count is "
                f"reported: the bound carries an unknown constant, so it prices one design against "
                f"another and not against a target error."
            ),
            "baselines": {
                TOTAL_ORDER: float(round(n)),
                RANDOM_DESIGN: random_design_lambda2(
                    self.bank, n_draws=self.baseline_draws, seed=self.seed
                ),
            },
        }


#: The three B6 instruments, in the order the catalogue's ladder names them.
B6_INSTRUMENTS: tuple[type[_B6Instrument], ...] = (
    CondorcetRate,
    SmithSetSize,
    ComparisonConnectivity,
)


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def _register() -> None:
    """B6's rungs, one ladder per quantity."""
    _ensure_quantities()
    rows = (
        (
            "grader.condorcet_rate",
            0,
            "grader.condorcet.direct",
            "approximately_unbiased",
            "a direct scan of the dominance matrix, plus Copeland. Exact for what it counts. An "
            "unobserved pair is not a win, so an incomplete comparison graph lowers the rate.",
        ),
        (
            "grader.condorcet_rate",
            2,
            "grader.condorcet.copeland_vs_slater",
            "unknown",
            "the disagreement rate between Copeland's winner and Slater's. Exact inside strongly "
            "connected components of at most eight alternatives, bracketed above that, and the "
            "reading says which fired.",
        ),
        (
            "grader.smith_set_size",
            1,
            "grader.smith.scc_closure",
            "upward",
            "upward closure from every seed, taking the smallest closed set. Exact on a complete "
            "comparison graph. An unobserved pair cannot exclude an alternative, so a sparse graph "
            "can only enlarge the set.",
        ),
        (
            "grader.graph_lambda2",
            1,
            "grader.lambda2.laplacian",
            "approximately_unbiased",
            "the second-smallest Laplacian eigenvalue, dense at the group sizes a loop uses and "
            "shift-invert eigsh on a large design graph. Exact up to floating point.",
        ),
    )
    for quantity, rung, impl, direction, why in rows:
        register_estimator(
            EstimatorEntry(
                quantity=quantity,
                impl=impl,
                requires={Component.GRADER: Access.RECORD},
                envelope=TOURNAMENT_ENVELOPE,
                rung=rung,
                bias=BiasStatement(direction=cast(BiasDirection, direction), why=why),
                cost=CostModel(note="arithmetic on a comparison record; no grader calls"),
                substrates=frozenset({Substrate.PROCEDURAL}),
                run=None,
            )
        )


_register()


def bank_from_invariance_payload(payload: InvariancePayload) -> tuple[ComparisonSet, ...]:
    """The bank an `InvariancePayload` describes, so the generated test is two lines here too."""
    return bank_from_payload(payload)


__all__ = [
    "B6_INSTRUMENTS",
    "RANDOM_DESIGN",
    "RANDOM_TOURNAMENT",
    "TOTAL_ORDER",
    "TOURNAMENT_ENVELOPE",
    "ComparisonConnectivity",
    "CondorcetReading",
    "CondorcetRate",
    "Connectivity",
    "SmithSetSize",
    "TournamentSummary",
    "algebraic_connectivity",
    "bank_from_invariance_payload",
    "bank_from_scores",
    "condorcet",
    "copeland_scores",
    "copeland_slater_split",
    "copeland_winners",
    "kemeny",
    "pair_to_add",
    "random_design_lambda2",
    "random_tournament_summary",
    "slater",
    "slater_winners",
    "smith_set",
    "summarise",
]

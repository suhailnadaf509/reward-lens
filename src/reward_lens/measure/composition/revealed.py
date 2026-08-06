"""B2, the Afriat efficiency index: how much of a grader's comparison record a scalar can carry.

Revealed-preference theory asks the question B1 asks, in a different vocabulary and with a bounded
answer. B1 decomposes the comparison flow and reports what share of its **energy** no scalar can
express. B2 asks whether the comparisons are *rationalizable* at all, and reports the **fraction of
the choices' own decisiveness** that has to be given up before a single utility explains them. An
energy share and a rationalizability fraction are different objects on the same data, and the two
can disagree completely: a strictly ranked seven-response tournament with no cycle anywhere has an
Afriat index of exactly 1.0 and a curl mass of 5/21, because equal unit margins around a total order
are not differences of any potential. Neither number is wrong. They answer different questions.

The construction, stated once so nothing here is a black box. Each recorded comparison is one
observation with a two-element menu. The winner's value in that observation is 1 and the loser's is
``1 - margin``, where the margin is the winner's lead normalised by the spread of the group it came
from. Afriat's direct revealed preference at efficiency ``e`` is then ``e * v(winner) >= v(loser)``,
which is ``e >= 1 - margin``: shrinking the budget by ``e`` drops the comparisons that were decided
by less than ``1 - e`` of the group's spread. The index is the **supremum** of the efficiencies at
which the surviving relation satisfies GARP.

Supremum, and not maximum, and the difference is not pedantry. A verdict enters the weak relation
exactly *at* its critical efficiency ``1 - margin``, so the axiom can hold on every ``e`` below a
critical value and fail at the value itself. Measured over 4,000 random comparison sets, ``garp(cs,
afriat_index(cs))`` is False on **all 2,763** of the ones that carry a cycle, and True on the rest
only because their index is exactly 1.0. So ``e = 0.87`` reads as "discard every verdict decided by
**at most** 13% of the group's spread and what is left is rationalizable by one objective". Reading
it as "keep everything above 13%" is off by the boundary and is wrong on every cyclic record.

Two consequences of that construction are load-bearing and are not hidden.

Normalising the margin **within the group** is what makes the index invariant under ``r -> a*r + b``
for ``a > 0``: the offset cancels in the lead and the scale cancels in the ratio. Where that claim
can actually be tested is worth stating, because for a long time it was not tested anywhere.
`ComparisonSet.from_scores` reads one score vector per group, which is a strict weak order, so its
dominance graph is acyclic by construction and the index is 1.0 whatever the margins are: over 5,000
random score banks it returned exactly 1.0 every time, 20,000 more produced zero GARP violations,
and a rule that took the margin as the **raw** score difference passed the same invariance check at
a maximum deviation of 0.0. An invariance that holds because the quantity is constant is not
evidence about the margin rule. `from_score_panel` is the constructor on which the claim has content
and where the generated test now runs: pooling several score vectors over one item set produces
genuine GARP violations, and there the raw-difference rule moves the index by 0.482 over the same
affine maps under which the shipped rule moves by 2.2e-16.

And on **purely ordinal** data, where every comparison is a bare win with no margin, the value of
the loser is 0, so ``e * 1 >= 0`` holds for every ``e`` and the index can only be 1 or 0. That is
not a defect in the estimator, it is the information content of the input: with no margins the only
fact available is the direction of each verdict. The instrument returns 1.0 when such data is
acyclic (a real measurement) and a bounded refusal when it is not, naming the Houtman-Maks index as
the ordinal quantity that stays informative and naming margins as the remedy.

Ladder, from the catalogue. Rung 0 is the GARP violation count, which is an estimator of the index
rather than a different quantity: zero violations means the index is exactly 1, and any violation
means it is below 1, so rung 0 returns a point value or a bound. Rung 1 is the index itself, exact
by enumerating the critical efficiencies where the relation changes, or by bisection when there are
too many of them. Rung 2 is Houtman-Maks, the largest rationalizable subset, which is a minimum
feedback arc set: exact by enumeration inside strongly connected components of at most eight items,
and otherwise an interval from a greedy ordering above and an arc-disjoint cycle packing below, with
which one fired stated on the reading. Rung 3 is the money-pump index over the cycles found.

Occupancy, from the catalogue: GARP and the Afriat index have been run on LLM agents making
budget-set choices (arXiv 2501.18190) and never on a grader's comparison data; Houtman-Maks and the
money-pump index return nothing in that literature at all.

Kill condition, verbatim: *if Afriat and curl mass are monotonically related across twenty graders,
in which case report one.*
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, cast

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import Evidence
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
from reward_lens.core.reading import (
    Reading,
    Refusal,
    RefusalReason,
    bounded_refusal,
)
from reward_lens.core.types import Access, Capability, Component, GaugeStatus, Substrate
from reward_lens.measure.base import BaseObservable, Context

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _ensure_quantities() -> None:
    """Load `spec/QUANTITIES.yaml` if nothing has yet.

    Nothing in the kernel populates the registry at import and `lint_instrument` fails an instrument
    whose quantity is unregistered. Loading is idempotent, so the guard is a cheap path rather than a
    correctness requirement.
    """
    if "grader.afriat_index" not in QUANTITIES:
        load_quantities()


_ensure_quantities()


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

#: The catalogue's baseline for B2: "the curl mass, as a different measure of the same failure".
#: Supplied by the caller, because B1 owns it and this instrument does not compute it. When it is
#: absent the reading says so rather than inventing a number.
CURL_MASS: BaselineID = "baseline.curl_mass"

#: A matched random tournament on the same comparison graph, always computed. Without it an index of
#: 1.0 is uninterpretable, because nobody knows what a grader choosing at random would score on the
#: same shape of data. It is the dumb baseline the claim needs to be a claim.
RANDOM_TOURNAMENT: BaselineID = "baseline.random_tournament"

AFRIAT_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "a census over a comparison record. The index counts which recorded verdicts can be "
        "ordered by one utility and asserts nothing about the process that produced them, so no "
        "regime can make the count wrong. No regime condition is recorded for B2."
    ),
)


# ---------------------------------------------------------------------------
# The comparison record both B2 and B6 read
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonSet:
    """The verdicts recorded over one menu of alternatives, which is one prompt's group.

    ``winners`` and ``losers`` index into ``0 .. n_items - 1``. ``margins`` is the winner's lead
    normalised to ``[0, 1]`` by the spread of the group, and ``margin_source`` names what produced
    it, so a reading can say whether its cardinal content is real. ``weights`` counts how many times
    each verdict was observed, which is what a repeated-comparison record carries and what the
    weighted feedback-arc computations use.

    A pair that was never compared is simply absent, and absent is not a tie. Distinguishing the two
    is the whole of B6's Condorcet story: an item that beats everything it met is not a Condorcet
    winner if it never met one of the alternatives.

    The same ordered pair may appear on more than one row, which is what a panel of graders or a
    repeated elicitation produces. Every method here handles that case explicitly: `dominance` and
    `observed` are unions, `weight_matrix` sums and `margin_matrix` takes the weighted mean, and
    `majority_tournament` collapses to the one-verdict-per-pair record the tournament solution
    concepts in B6 are defined on. Nothing silently keeps the last row.
    """

    n_items: int
    winners: np.ndarray
    losers: np.ndarray
    margins: np.ndarray
    weights: np.ndarray
    margin_source: str = ""
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        m = int(self.winners.shape[0])
        for name in ("losers", "margins", "weights"):
            got = int(getattr(self, name).shape[0])
            if got != m:
                raise ValueError(
                    f"ComparisonSet has {m} winners and {got} {name}; every array describes the "
                    f"same list of recorded verdicts, so they have to be the same length."
                )
        if self.n_items < 0:
            raise ValueError(f"n_items must be non-negative; got {self.n_items}")
        if m:
            # The whole Afriat correspondence rests on the margin being a fraction of the group's
            # own spread: the loser's value is `1 - margin`, so a margin outside [0, 1] is a
            # negative value or a value above the winner's, and neither means anything. It is worth
            # checking rather than documenting because the failure is silent and inverted. A record
            # of margins [3.0, -2.0, 0.5] around a three-item cycle used to return an index of
            # exactly 1.0: the -2.0 verdict has a critical efficiency of 3.0, which no efficiency
            # reaches, so the verdict never enters the relation, the cycle never closes, and a
            # maximally irrational record reported as perfectly rationalizable.
            lo, hi = float(self.margins.min()), float(self.margins.max())
            if not np.all(np.isfinite(self.margins)) or lo < 0.0 or hi > 1.0:
                raise ValueError(
                    f"margins must be finite and lie in [0, 1]; got [{lo:g}, {hi:g}]. A margin is "
                    f"the winner's lead as a fraction of its group's spread, so it is already "
                    f"normalised: use `from_scores` or `from_score_panel` for a score record, "
                    f"`from_edges` for a win-count record, and divide by the group's spread if you "
                    f"are building the arrays yourself."
                )
            if not np.all(np.isfinite(self.weights)) or float(self.weights.min()) <= 0.0:
                raise ValueError(
                    f"weights count the observations behind a verdict, so each must be finite and "
                    f"positive; the smallest here is {float(self.weights.min()):g}."
                )

    @property
    def n_comparisons(self) -> int:
        return int(self.winners.shape[0])

    @property
    def has_repeated_pairs(self) -> bool:
        """Whether any unordered pair carries more than one row.

        A record from a single scalar grader has one row per pair and this is False. A panel, a
        repeated elicitation, or any pooling of several occasions has more, and then the tournament
        solution concepts in B6 need `majority_tournament` first, because Copeland, Slater, Smith
        and Condorcet are all defined on one verdict per pair.
        """
        if not self.n_comparisons:
            return False
        lo = np.minimum(self.winners, self.losers)
        hi = np.maximum(self.winners, self.losers)
        keys = lo.astype(np.int64) * max(self.n_items, 1) + hi.astype(np.int64)
        return bool(np.unique(keys).size < keys.size)

    @property
    def is_ordinal(self) -> bool:
        """Whether the record carries no usable cardinal content.

        True when every recorded margin is the full spread of its group, which is what a bare
        win/loss record looks like after normalisation. On such data the Afriat index can only take
        the values 0 and 1, and the instrument says so rather than reporting a fabricated interior
        value.
        """
        return self.n_comparisons == 0 or bool(np.all(self.margins >= 1.0))

    def dominance(self) -> np.ndarray:
        """``(n, n)`` boolean: row beat column at least once, by a strict margin."""
        a = np.zeros((self.n_items, self.n_items), dtype=bool)
        if self.n_comparisons:
            a[self.winners, self.losers] = True
        return a

    def observed(self) -> np.ndarray:
        """``(n, n)`` boolean, symmetric: this pair was compared. The comparison graph."""
        u = np.zeros((self.n_items, self.n_items), dtype=bool)
        if self.n_comparisons:
            u[self.winners, self.losers] = True
            u[self.losers, self.winners] = True
        return u

    def margin_matrix(self, *, aggregate: str = "mean") -> np.ndarray:
        """``(n, n)`` float: the margin by which row beat column, else 0.

        ``aggregate="mean"`` is the weighted mean over the rows recording that ordered pair, which
        is what a per-step cost wants: walking a cycle once pays the average margin of that step,
        not the margin of whichever row happened to be written last. ``aggregate="sum"`` is the
        total margin, which is what Kemeny's arc weight is, because reversing a pair reverses every
        observation of it.

        Fancy indexing used to write these with ``m[winners, losers] = margins``, which is
        last-write-wins on a repeated pair: three rows for one pair with margins 0.9, 0.5 and 0.1
        returned 0.1, silently and with no way to see it from the result.
        """
        if aggregate not in ("mean", "sum"):
            raise ValueError(f"aggregate must be 'mean' or 'sum'; got {aggregate!r}")
        m = np.zeros((self.n_items, self.n_items), dtype=np.float64)
        if not self.n_comparisons:
            return m
        np.add.at(m, (self.winners, self.losers), self.margins * self.weights)
        if aggregate == "sum":
            return m
        w = self.weight_matrix()
        mean = np.divide(m, w, out=np.zeros_like(m), where=w > 0)
        return np.asarray(mean, dtype=np.float64)

    def weight_matrix(self) -> np.ndarray:
        """``(n, n)`` float: how many observations back the verdict that row beat column.

        Rows recording the same ordered pair add, so this matrix sums to ``weights.sum()``. It is
        the arc weight the feedback-arc computations use, and it has to agree with the total the
        Houtman-Maks fraction divides by or the fraction is a ratio of two different denominators.
        """
        w = np.zeros((self.n_items, self.n_items), dtype=np.float64)
        if self.n_comparisons:
            np.add.at(w, (self.winners, self.losers), self.weights)
        return w

    def majority_tournament(self) -> "ComparisonSet":
        """One verdict per unordered pair, decided by the observations, ties dropped.

        Every tournament solution concept in B6 (Condorcet, Copeland, Slater, Smith) is defined on a
        record with at most one verdict per pair. On a panel that is not what the record holds, and
        reading `dominance` directly then says "beat at least once", under which nearly every pair
        of a twelve-member panel beats nearly every other and every alternative comes out a
        Condorcet winner. Collapsing first is not a convenience, it is what makes the concept
        defined.

        The collapsed margin is the win fraction ``|w_a - w_b| / (w_a + w_b)``, weighted by the
        rows' own weights, which is the same quantity `from_edges` records. A pair whose weighted
        verdicts tie records nothing, for the same reason a tie in the scores records nothing.

        Checked against the shipped alternative: collapsing the 1,763-prompt twelve-model panel of
        scores reproduces the majority-vote `from_edges` bank verdict for verdict, and every B6
        number on the two agrees exactly.
        """
        if not self.has_repeated_pairs:
            return self
        w_for = self.weight_matrix()
        wins: list[int] = []
        loss: list[int] = []
        marg: list[float] = []
        wt: list[float] = []
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                a, b = float(w_for[i, j]), float(w_for[j, i])
                total = a + b
                if total <= 0.0 or a == b:
                    continue
                if a > b:
                    wins.append(i)
                    loss.append(j)
                else:
                    wins.append(j)
                    loss.append(i)
                marg.append(abs(a - b) / total)
                wt.append(total)
        source = f"majority of {self.n_comparisons} recorded verdicts, as a win fraction"
        return ComparisonSet(
            n_items=self.n_items,
            winners=np.asarray(wins, dtype=np.intp),
            losers=np.asarray(loss, dtype=np.intp),
            margins=np.asarray(marg, dtype=np.float64),
            weights=np.asarray(wt, dtype=np.float64),
            margin_source=source if wins else "",
            labels=self.labels,
        )

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_scores(
        cls, scores: Sequence[float] | np.ndarray, *, labels: Sequence[str] = ()
    ) -> "ComparisonSet":
        """The comparisons a scalar grader's scores over one group imply.

        Every unordered pair with distinct scores becomes one verdict; equal scores record nothing,
        because a grader that scored two responses identically expressed no preference between them
        and inventing one would be the silent-tie failure B6 exists to make visible.

        The margin is ``|s_i - s_j| / (max - min)`` over the group, which is exactly invariant under
        ``s -> a*s + b`` for ``a > 0``: the offset cancels in the difference and the scale cancels in
        the ratio.

        **A record built this way cannot violate GARP**, and that is a property of the input rather
        than a finding about the grader. One score vector is a strict weak order, so its dominance
        graph is acyclic, `afriat_index_of` short-circuits, and the index is exactly 1.0 on all
        5,000 of 5,000 random score banks measured. B6's cycle rate is identically 0 on the same
        records for the same reason, so B6's kill condition cannot fire on them either. Use
        `from_score_panel` when several graders, criteria or repeats scored the same items, which is
        the case in which the index has something to measure.
        """
        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        n = int(s.shape[0])
        spread = float(s.max() - s.min()) if n else 0.0
        wins: list[int] = []
        loss: list[int] = []
        marg: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                d = float(s[i] - s[j])
                if d == 0.0:
                    continue
                lead = abs(d) / spread if spread > 0 else 1.0
                if d > 0:
                    wins.append(i)
                    loss.append(j)
                else:
                    wins.append(j)
                    loss.append(i)
                marg.append(min(1.0, lead))
        m = len(wins)
        return cls(
            n_items=n,
            winners=np.asarray(wins, dtype=np.intp),
            losers=np.asarray(loss, dtype=np.intp),
            margins=np.asarray(marg, dtype=np.float64),
            weights=np.ones(m, dtype=np.float64),
            margin_source="score difference over the group's spread" if m else "",
            labels=tuple(labels),
        )

    @classmethod
    def from_score_panel(
        cls, rows: Sequence[Sequence[float]] | np.ndarray, *, labels: Sequence[str] = ()
    ) -> "ComparisonSet":
        """Several score vectors over **one** item set, pooled into one comparison record.

        ``rows`` is ``(n_observations, n_items)``. Each row is one occasion on which the whole item
        set was scored: one member of a panel of graders, one criterion of a rubric, one repeat of a
        stochastic judge, one prompt phrasing. Every within-row pair with distinct scores becomes one
        recorded verdict, and the verdicts of all the rows land in the same `ComparisonSet`.

        This is Afriat's setting and `from_scores` is not. Afriat's index is defined over *several
        observations*, each with its own menu; a single score vector is one observation, its
        dominance graph is a strict weak order, and its index is 1.0 before anything is measured. A
        panel is where the index has content: twelve reward models can put A over B, B over C and C
        over A, and no single utility orders all three.

        The margin is normalised **within each row** by that row's own spread, which is the choice
        that makes the pooled record meaningful at all. Two reward models' raw scores are not on a
        common scale, so a raw difference pools numbers that are not comparable; a within-row
        fraction of that row's own spread is. It also buys the declared invariance: the index is
        exactly unchanged under ``s -> a*s + b`` applied to the whole panel, and under a *separate*
        affine map per row, which is the honest group for a panel of graders with unrelated scales.

        A row whose scores are all equal contributes nothing, and a row with a single distinct pair
        contributes a margin of 1.0, which is the honest statement that one comparison with no
        spread to normalise by carries direction and nothing else.
        """
        arr = np.asarray(rows, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                f"from_score_panel takes a 2-D (n_observations, n_items) array, one row per "
                f"occasion on which the same items were scored; got shape {arr.shape}. For one "
                f"score vector per group use `bank_from_scores`."
            )
        n_obs, n = arr.shape
        wins: list[int] = []
        loss: list[int] = []
        marg: list[float] = []
        for r in range(n_obs):
            s = arr[r]
            spread = float(s.max() - s.min()) if n else 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    d = float(s[i] - s[j])
                    if d == 0.0:
                        continue
                    lead = abs(d) / spread if spread > 0 else 1.0
                    if d > 0:
                        wins.append(i)
                        loss.append(j)
                    else:
                        wins.append(j)
                        loss.append(i)
                    marg.append(min(1.0, lead))
        m = len(wins)
        return cls(
            n_items=int(n),
            winners=np.asarray(wins, dtype=np.intp),
            losers=np.asarray(loss, dtype=np.intp),
            margins=np.asarray(marg, dtype=np.float64),
            weights=np.ones(m, dtype=np.float64),
            margin_source=(
                f"score difference over each observation's own spread, {n_obs} observations"
                if m
                else ""
            ),
            labels=tuple(labels),
        )

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[Sequence[float]] | np.ndarray,
        n_items: int,
        *,
        labels: Sequence[str] = (),
    ) -> "ComparisonSet":
        """From an ``(i, j, wins_i, wins_j)`` edge record, which is what a tournament log holds.

        The margin is ``|wins_i - wins_j| / (wins_i + wins_j)``, the win-fraction lead. Repeated
        comparisons therefore carry real cardinal content: 7 wins to 4 is a margin of 0.27 and a
        single 1-0 verdict is a margin of 1.0, which is the honest statement that one observation
        says only who won. A tie in the wins records nothing, for the same reason a tie in the scores
        records nothing.
        """
        arr = np.asarray(list(edges) if not isinstance(edges, np.ndarray) else edges)
        wins: list[int] = []
        loss: list[int] = []
        marg: list[float] = []
        wt: list[float] = []
        if arr.size:
            arr = arr.reshape(-1, 4)
            for i, j, wi, wj in arr:
                i, j = int(i), int(j)
                total = float(wi) + float(wj)
                if wi == wj or total <= 0:
                    continue
                if wi > wj:
                    wins.append(i)
                    loss.append(j)
                else:
                    wins.append(j)
                    loss.append(i)
                marg.append(abs(float(wi) - float(wj)) / total)
                wt.append(total)
        m = len(wins)
        return cls(
            n_items=int(n_items),
            winners=np.asarray(wins, dtype=np.intp),
            losers=np.asarray(loss, dtype=np.intp),
            margins=np.asarray(marg, dtype=np.float64),
            weights=np.asarray(wt, dtype=np.float64),
            margin_source="win fraction" if m else "",
            labels=tuple(labels),
        )


def bank_from_scores(
    scores: Sequence[float] | np.ndarray, group_ids: Sequence[int] | np.ndarray | None = None
) -> tuple[ComparisonSet, ...]:
    """One `ComparisonSet` per prompt group, from a flat score vector and its group labels.

    This is the adapter the generated invariance test runs through: `InvariancePayload` carries
    ``scores`` and ``group_ids`` and nothing else, and every quantity in this module is a function of
    the comparisons those two imply. A 2-D ``(n_groups, k)`` array is accepted directly, which is the
    shape a recorded score bank is in.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim == 2 and group_ids is None:
        return tuple(ComparisonSet.from_scores(row) for row in arr)
    arr = arr.reshape(-1)
    if group_ids is None:
        return (ComparisonSet.from_scores(arr),)
    g = np.asarray(group_ids).reshape(-1)
    return tuple(ComparisonSet.from_scores(arr[g == k]) for k in np.unique(g))


def bank_from_score_panels(
    scores: Sequence[Sequence[Sequence[float]]] | np.ndarray,
) -> tuple[ComparisonSet, ...]:
    """One `ComparisonSet` per group, from an ``(n_groups, n_observations, n_items)`` score array.

    The shape a panel of graders produces: several graders, each scoring the same candidates of the
    same prompt. `bank_from_scores` reads the two-dimensional case, one score vector per group, and
    this reads the three-dimensional one.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(
            f"bank_from_score_panels takes an (n_groups, n_observations, n_items) array; got "
            f"shape {arr.shape}. Use `bank_from_scores` for one score vector per group."
        )
    return tuple(ComparisonSet.from_score_panel(arr[g]) for g in range(arr.shape[0]))


def bank_from_payload(payload: InvariancePayload) -> tuple[ComparisonSet, ...]:
    """The bank an `InvariancePayload` describes. Two lines, so the generated test stays two lines."""
    payload.require("scores")
    scores = payload.scores
    if scores is None:  # pragma: no cover - `require` raises first; this narrows for the checker
        raise ValueError("payload.require('scores') passed and payload.scores is None")
    return bank_from_scores(scores, payload.group_ids)


# ---------------------------------------------------------------------------
# The relations
# ---------------------------------------------------------------------------


def transitive_closure(relation: np.ndarray) -> np.ndarray:
    """Warshall's closure of a boolean adjacency matrix, in place on a copy."""
    r = np.array(relation, dtype=bool, copy=True)
    n = r.shape[0]
    for k in range(n):
        r |= r[:, k][:, None] & r[k, :][None, :]
    return r


def direct_preference(cs: ComparisonSet, efficiency: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Afriat's direct revealed preference at efficiency ``e``, weak and strict.

    ``R[w, l]`` holds when ``e * v(w) >= v(l)`` with ``v(w) = 1`` and ``v(l) = 1 - margin``, which
    reduces to ``e >= 1 - margin``. ``P`` is the strict version, with ``>``. Shrinking the budget by
    ``e`` therefore drops exactly the verdicts decided by less than ``1 - e`` of the group's spread,
    which is what scaling a budget set does in the setting this comes from.
    """
    n = cs.n_items
    weak = np.zeros((n, n), dtype=bool)
    strict = np.zeros((n, n), dtype=bool)
    if cs.n_comparisons:
        threshold = 1.0 - cs.margins
        keep = efficiency >= threshold
        strong = efficiency > threshold
        weak[cs.winners[keep], cs.losers[keep]] = True
        strict[cs.winners[strong], cs.losers[strong]] = True
    return weak, strict


@dataclass(frozen=True)
class GarpReport:
    """Whether the axiom holds, how badly it fails, and on which pairs."""

    efficiency: float
    holds: bool
    n_violations: int
    n_comparisons: int
    violating_pairs: tuple[tuple[int, int], ...] = ()

    @property
    def violation_rate(self) -> float:
        return self.n_violations / self.n_comparisons if self.n_comparisons else 0.0


def garp(cs: ComparisonSet, efficiency: float = 1.0, *, keep_pairs: int = 32) -> GarpReport:
    """The generalised axiom of revealed preference, checked at one efficiency.

    A violation is an ordered pair ``(x, y)`` with ``x`` indirectly revealed preferred to ``y`` and
    ``y`` strictly directly revealed preferred to ``x``. At ``e = 1`` that is exactly a directed
    cycle in the dominance graph, which is why rung 0 is a cycle count and why B1 reads the same
    structure through a different norm.
    """
    weak, strict = direct_preference(cs, efficiency)
    closed = transitive_closure(weak)
    bad = closed & strict.T
    idx = np.argwhere(bad)
    pairs = tuple((int(a), int(b)) for a, b in idx[:keep_pairs])
    return GarpReport(
        efficiency=float(efficiency),
        holds=not idx.size,
        n_violations=int(idx.shape[0]),
        n_comparisons=cs.n_comparisons,
        violating_pairs=pairs,
    )


def garp_violations(bank: Sequence[ComparisonSet], efficiency: float = 1.0) -> GarpReport:
    """GARP over a whole bank. Groups are independent, so the counts add."""
    total = 0
    comps = 0
    pairs: list[tuple[int, int]] = []
    for cs in bank:
        r = garp(cs, efficiency)
        total += r.n_violations
        comps += r.n_comparisons
        if len(pairs) < 32:
            pairs.extend(r.violating_pairs[: 32 - len(pairs)])
    return GarpReport(
        efficiency=float(efficiency),
        holds=total == 0,
        n_violations=total,
        n_comparisons=comps,
        violating_pairs=tuple(pairs),
    )


# ---------------------------------------------------------------------------
# Rung 1: the index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AfriatResult:
    """The index, the way it was found, and everything needed to read it.

    ``pooled`` is the index as Afriat defines it, applied to the whole bank: the largest efficiency
    at which *every* group satisfies GARP, which because groups are independent is the minimum of
    the per-group indices. It is a minimum statistic over however many prompts there are, so on a
    5,000-prompt bank one pathological group sets it. ``mean`` and ``median`` are the summaries a
    reader usually wants and they are carried beside it rather than instead of it.
    """

    pooled: float
    mean: float
    median: float
    per_group: np.ndarray
    n_groups: int
    n_comparisons: int
    n_violations: int
    ordinal: bool
    method: str
    tolerance: float = 0.0
    #: Groups that violate the axiom *and* carry no margin to bisect on. Each of them pins the
    #: pooled index at 0 for a reason about the record's information content rather than about the
    #: grader, which is what the instrument refuses on.
    ordinal_violations: int = 0

    @property
    def rationalizable(self) -> bool:
        return self.n_violations == 0


def _critical_efficiencies(cs: ComparisonSet) -> np.ndarray:
    """The efficiencies at which the relation changes, which is where the index can sit."""
    if not cs.n_comparisons:
        return np.asarray([1.0])
    t = np.unique(np.clip(1.0 - cs.margins, 0.0, 1.0))
    return np.concatenate([t, [1.0]]) if t[-1] < 1.0 else t


def afriat_index_of(
    cs: ComparisonSet, *, max_candidates: int = 256, tolerance: float = 1e-6
) -> tuple[float, str]:
    """One group's critical cost efficiency, and which method produced it.

    GARP at efficiency ``e`` is monotone: lowering the budget removes verdicts from the relation and
    can only remove violations. So the set where the axiom holds is an interval anchored at 0 and the
    index is its supremum.

    The exact route uses the fact that the relation changes only at the critical efficiencies
    ``1 - margin``. A verdict enters the weak relation exactly *at* its critical value and the strict
    relation just *above* it, so the axiom's status can differ between the point and the interval
    above it and both have to be probed. The supremum is then the first critical value at which
    either probe fails, whether or not the supremum is attained, which is the convention bisection
    converges to and is what makes the two routes agree.

    The bisection route is the fallback when there are more critical values than the cap, and it
    reports the tolerance it stopped at.
    """
    if not cs.n_comparisons or garp(cs, 1.0).holds:
        return 1.0, "exact"
    candidates = _critical_efficiencies(cs)
    if candidates.size <= max_candidates:
        thresholds = [float(t) for t in candidates if t < 1.0] or [0.0]
        for k, t in enumerate(thresholds):
            if not garp(cs, t).holds:
                return t, "exact"
            upper = thresholds[k + 1] if k + 1 < len(thresholds) else 1.0
            if not garp(cs, 0.5 * (t + upper)).holds:
                return t, "exact"
        return 1.0, "exact"
    lo, hi = 0.0, 1.0
    if not garp(cs, 0.0).holds:
        return 0.0, "bisect"
    while hi - lo > tolerance:
        mid = 0.5 * (lo + hi)
        if garp(cs, mid).holds:
            lo = mid
        else:
            hi = mid
    return lo, "bisect"


def afriat_index(
    bank: Sequence[ComparisonSet], *, max_candidates: int = 256, tolerance: float = 1e-6
) -> AfriatResult:
    """The Afriat efficiency index over a bank of comparison groups."""
    per: list[float] = []
    methods: set[str] = set()
    comps = 0
    violations = 0
    ordinal = True
    ordinal_violations = 0
    for cs in bank:
        e, how = afriat_index_of(cs, max_candidates=max_candidates, tolerance=tolerance)
        per.append(e)
        methods.add(how)
        comps += cs.n_comparisons
        group_violations = garp(cs, 1.0).n_violations
        violations += group_violations
        if not cs.is_ordinal:
            ordinal = False
        elif group_violations:
            ordinal_violations += 1
    arr = np.asarray(per, dtype=np.float64)
    if not arr.size:
        arr = np.asarray([1.0])
    weights = np.asarray([max(cs.n_comparisons, 0) for cs in bank], dtype=np.float64)
    if weights.size != arr.size or weights.sum() <= 0:
        weights = np.ones_like(arr)
    return AfriatResult(
        pooled=float(arr.min()),
        mean=float(np.average(arr, weights=weights)),
        median=float(np.median(arr)),
        per_group=arr,
        n_groups=len(bank),
        n_comparisons=comps,
        n_violations=violations,
        ordinal=ordinal,
        method="+".join(sorted(methods)) if methods else "exact",
        tolerance=tolerance if "bisect" in methods else 0.0,
        ordinal_violations=ordinal_violations,
    )


# ---------------------------------------------------------------------------
# Rung 2: Houtman-Maks, which is a minimum feedback arc set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeedbackArcs:
    """The minimum number of verdicts to delete to leave an acyclic record.

    ``exact`` says whether the number is the optimum or an interval. Minimum feedback arc set is
    NP-hard, so the exact route runs only inside strongly connected components small enough to
    enumerate, and everything else is bracketed: ``upper`` from a greedy ordering, ``lower`` from a
    packing of arc-disjoint cycles, each of which certifies its side.
    """

    cost: float
    lower: float
    upper: float
    exact: bool
    order: tuple[int, ...]
    method: str
    n_arcs: int


def _strong_components(adjacency: np.ndarray) -> list[list[int]]:
    """Strongly connected components, largest first. Iterative Tarjan, no recursion limit."""
    n = adjacency.shape[0]
    index = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack: list[int] = []
    out: list[list[int]] = []
    counter = 0
    succ = [np.flatnonzero(adjacency[v]).tolist() for v in range(n)]
    for root in range(n):
        if index[root] != -1:
            continue
        work: list[tuple[int, int]] = [(root, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on_stack[v] = True
            recursed = False
            for i in range(pi, len(succ[v])):
                w = succ[v][i]
                if index[w] == -1:
                    work[-1] = (v, i + 1)
                    work.append((w, 0))
                    recursed = True
                    break
                if on_stack[w]:
                    low[v] = min(low[v], index[w])
            if recursed:
                continue
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == v:
                        break
                out.append(sorted(comp))
            work.pop()
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
    out.sort(key=len, reverse=True)
    return out


def _greedy_order(weights: np.ndarray, nodes: Sequence[int]) -> list[int]:
    """Eades, Lin and Smyth's greedy sequencing heuristic, which bounds the deletion cost above."""
    remaining = list(nodes)
    left: list[int] = []
    right: list[int] = []

    def out_weight(v: int) -> float:
        return float(weights[v, [u for u in remaining if u != v]].sum())

    def in_weight(v: int) -> float:
        return float(weights[[u for u in remaining if u != v], v].sum())

    while remaining:
        moved = True
        while moved and remaining:
            moved = False
            for v in [v for v in remaining if out_weight(v) == 0.0]:
                right.insert(0, v)
                remaining.remove(v)
                moved = True
            for v in [v for v in remaining if in_weight(v) == 0.0]:
                left.append(v)
                remaining.remove(v)
                moved = True
        if remaining:
            _, best = max((out_weight(v) - in_weight(v), v) for v in remaining)
            left.append(best)
            remaining.remove(best)
    return left + right


def _backward_cost(weights: np.ndarray, order: Sequence[int]) -> float:
    """The weight of the arcs a linear order has to break."""
    pos = {v: p for p, v in enumerate(order)}
    total = 0.0
    for u in order:
        for v in order:
            if u != v and weights[u, v] and pos[u] > pos[v]:
                total += float(weights[u, v])
    return total


def _cycle_packing_bound(weights: np.ndarray, nodes: Sequence[int]) -> float:
    """A lower bound: greedily remove arc-disjoint cycles and count the cheapest arc of each.

    Every cycle needs at least one arc deleted, and arc-disjoint cycles need distinct arcs, so the
    sum of their cheapest arcs is a floor on the optimum. It is a weak floor and it is honest.
    """
    w = np.array(weights, dtype=np.float64, copy=True)
    keep = set(nodes)
    for i in range(w.shape[0]):
        if i not in keep:
            w[i, :] = 0.0
            w[:, i] = 0.0
    total = 0.0
    for _ in range(w.shape[0] * 4):
        cyc = _find_cycle(w > 0)
        if not cyc:
            break
        arcs = [(cyc[k], cyc[(k + 1) % len(cyc)]) for k in range(len(cyc))]
        cheapest = min(float(w[a, b]) for a, b in arcs)
        total += cheapest
        for a, b in arcs:
            w[a, b] = 0.0
    return total


def _find_cycle(adjacency: np.ndarray) -> list[int]:
    """One directed cycle, or an empty list. Iterative depth-first search with a colour map."""
    n = adjacency.shape[0]
    colour = [0] * n
    parent = [-1] * n
    for root in range(n):
        if colour[root]:
            continue
        stack = [(root, 0)]
        colour[root] = 1
        while stack:
            v, pi = stack[-1]
            succ = np.flatnonzero(adjacency[v])
            if pi < succ.size:
                stack[-1] = (v, pi + 1)
                w = int(succ[pi])
                if colour[w] == 0:
                    colour[w] = 1
                    parent[w] = v
                    stack.append((w, 0))
                elif colour[w] == 1:
                    cyc = [w]
                    u = v
                    while u != w and u != -1:
                        cyc.append(u)
                        u = parent[u]
                    cyc.reverse()
                    return cyc
            else:
                colour[v] = 2
                stack.pop()
    return []


def min_feedback_arcs(weights: np.ndarray, *, exact_max: int = 8) -> FeedbackArcs:
    """The cheapest set of arcs whose deletion leaves the digraph acyclic.

    Slater's distance, Kemeny's distance and the Houtman-Maks index are all this computation on
    different weights: unweighted majority arcs for Slater, margin-weighted arcs for Kemeny,
    observation-weighted arcs for Houtman-Maks. Writing it once is the reason all three are
    available and none of them is a separate approximation with its own quirks.

    The problem decomposes over strongly connected components, because no arc between two components
    lies on a cycle. Components of at most ``exact_max`` items are solved exactly by enumerating
    their orderings; larger ones return a bracket and say so.
    """
    adjacency = weights > 0
    total = 0.0
    lower = 0.0
    upper = 0.0
    exact = True
    order: list[int] = []
    methods: set[str] = set()
    comps = _strong_components(adjacency)
    for comp in sorted(comps, key=lambda c: min(c) if c else 0):
        if len(comp) == 1:
            order.extend(comp)
            continue
        if len(comp) <= exact_max:
            best = None
            best_order: tuple[int, ...] = tuple(comp)
            for perm in itertools.permutations(comp):
                cost = _backward_cost(weights, perm)
                if best is None or cost < best:
                    best, best_order = cost, perm
                    if cost == 0:
                        break
            cost = float(best or 0.0)
            total += cost
            lower += cost
            upper += cost
            order.extend(best_order)
            methods.add("exact")
        else:
            greedy = _greedy_order(weights, comp)
            hi = _backward_cost(weights, greedy)
            lo = _cycle_packing_bound(weights, comp)
            total += hi
            upper += hi
            lower += lo
            order.extend(greedy)
            exact = False
            methods.add("greedy order above, arc-disjoint cycle packing below")
    order = _topological_between_components(weights, comps, order)
    return FeedbackArcs(
        cost=total,
        lower=lower,
        upper=upper,
        exact=exact,
        order=tuple(order),
        method="+".join(sorted(methods)) if methods else "acyclic",
        n_arcs=int(adjacency.sum()),
    )


def _topological_between_components(
    weights: np.ndarray, comps: Sequence[Sequence[int]], within: Sequence[int]
) -> list[int]:
    """Order the components themselves, so the returned ranking is a real ranking."""
    comp_of: dict[int, int] = {}
    for ci, comp in enumerate(comps):
        for v in comp:
            comp_of[v] = ci
    k = len(comps)
    edges = np.zeros((k, k), dtype=bool)
    for u, v in np.argwhere(weights > 0):
        cu, cv = comp_of[int(u)], comp_of[int(v)]
        if cu != cv:
            edges[cu, cv] = True
    indeg = edges.sum(axis=0).astype(int)
    ready = [c for c in range(k) if indeg[c] == 0]
    seq: list[int] = []
    while ready:
        c = ready.pop(0)
        seq.append(c)
        for d in np.flatnonzero(edges[c]):
            indeg[d] -= 1
            if indeg[d] == 0:
                ready.append(int(d))
    seq.extend(c for c in range(k) if c not in seq)
    pos = {v: i for i, v in enumerate(within)}
    out: list[int] = []
    for c in seq:
        out.extend(sorted(comps[c], key=lambda v: pos.get(v, 0)))
    return out


@dataclass(frozen=True)
class HoutmanMaksResult:
    """The largest rationalizable subset of the record, as a count and as a fraction."""

    kept: float
    dropped: float
    fraction: float
    lower_fraction: float
    upper_fraction: float
    exact: bool
    method: str
    n_comparisons: int

    @property
    def is_bracketed(self) -> bool:
        return not self.exact


def houtman_maks(bank: Sequence[ComparisonSet], *, exact_max: int = 8) -> HoutmanMaksResult:
    """The largest subset of recorded verdicts that one utility can order.

    This is the number the catalogue's headline sentence is about: "thirteen percent of its choices
    cannot be reconciled with any single objective" is a statement about the fraction of the record
    that has to go, which is one minus this index. It stays informative on purely ordinal data,
    where the Afriat index cannot, which is why the refusal path points here.

    **When the exact route cannot fire, the returned `fraction` is a floor rather than a point.**
    Above ``exact_max`` items in one strongly connected component the deletion cost is the greedy
    ordering's, which is an upper bound on the true minimum, so the kept fraction is a lower bound
    on the true one. Measured over sixty eight-item tournaments the greedy route over-counts
    deletions by a mean of 26.1% and a worst of 150.0%, which on one forty-group bank moved the
    reported fraction from 0.8098 to 0.7536. `exact` says which route fired and
    ``lower_fraction``/``upper_fraction`` carry the bracket; read the bracket and not the point
    whenever `exact` is False.
    """
    total = 0.0
    dropped = 0.0
    lo = 0.0
    hi = 0.0
    exact = True
    methods: set[str] = set()
    for cs in bank:
        w = cs.weight_matrix()
        total += float(cs.weights.sum())
        fa = min_feedback_arcs(w, exact_max=exact_max)
        dropped += fa.cost
        lo += fa.lower
        hi += fa.upper
        exact = exact and fa.exact
        if fa.method != "acyclic":
            methods.add(fa.method)
    denom = total if total > 0 else 1.0
    return HoutmanMaksResult(
        kept=total - dropped,
        dropped=dropped,
        fraction=(total - dropped) / denom,
        lower_fraction=(total - hi) / denom,
        upper_fraction=(total - lo) / denom,
        exact=exact,
        method="+".join(sorted(methods)) if methods else "acyclic: nothing to drop",
        n_comparisons=int(sum(cs.n_comparisons for cs in bank)),
    )


# ---------------------------------------------------------------------------
# Rung 3: the money-pump index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoneyPumpResult:
    """What an arbitrageur could extract by walking the grader's own cycles.

    ``mean`` and ``maximum`` are over the cycles that were enumerated. Enumeration is capped, because
    the number of simple cycles is exponential in the group size; ``truncated`` says whether the cap
    bound, and ``n_cycles`` says how many were seen.
    """

    mean: float
    maximum: float
    n_cycles: int
    truncated: bool
    max_length: int


def _simple_cycles(adjacency: np.ndarray, *, cap: int, max_length: int) -> list[list[int]]:
    """Every simple directed cycle up to a length, up to a cap. Depth-first from each least vertex."""
    n = adjacency.shape[0]
    out: list[list[int]] = []
    succ = [np.flatnonzero(adjacency[v]).tolist() for v in range(n)]

    def walk(start: int, path: list[int], seen: set[int]) -> None:
        if len(out) >= cap:
            return
        v = path[-1]
        for w in succ[v]:
            if w == start and len(path) >= 2:
                out.append(list(path))
                if len(out) >= cap:
                    return
            elif w > start and w not in seen and len(path) < max_length:
                seen.add(w)
                path.append(w)
                walk(start, path, seen)
                path.pop()
                seen.discard(w)

    for s in range(n):
        walk(s, [s], {s})
        if len(out) >= cap:
            break
    return out


def money_pump(
    bank: Sequence[ComparisonSet], *, cap: int = 10_000, max_length: int | None = None
) -> MoneyPumpResult:
    """The money-pump index: the mean margin around each of the grader's own preference cycles.

    Walking a cycle ``x1 > x2 > ... > xk > x1`` and trading at each step nets the margin of that
    step, so the normalised cost of the cycle is the mean of its margins. A cycle whose steps were
    all decided by a hair costs nearly nothing; one decided by wide margins all the way round is a
    grader that will pay to go in a circle.
    """
    means: list[float] = []
    truncated = False
    longest = 0
    for cs in bank:
        adjacency = cs.dominance()
        limit = max_length or cs.n_items
        cycles = _simple_cycles(adjacency, cap=cap, max_length=limit)
        if len(cycles) >= cap:
            truncated = True
        m = cs.margin_matrix()
        for cyc in cycles:
            longest = max(longest, len(cyc))
            steps = [m[cyc[k], cyc[(k + 1) % len(cyc)]] for k in range(len(cyc))]
            means.append(float(np.mean(steps)))
    if not means:
        return MoneyPumpResult(0.0, 0.0, 0, False, longest)
    return MoneyPumpResult(
        mean=float(np.mean(means)),
        maximum=float(np.max(means)),
        n_cycles=len(means),
        truncated=truncated,
        max_length=longest,
    )


# ---------------------------------------------------------------------------
# The baseline
# ---------------------------------------------------------------------------


def random_tournament_index(
    bank: Sequence[ComparisonSet], *, n_draws: int = 64, seed: int = 0
) -> dict[str, float]:
    """The index a grader choosing at random would score on the same comparison graph.

    The graph is held fixed and only the directions and the margins are redrawn, so the baseline
    answers the question a reader actually has: on *this* comparison design, how far from 1.0 is
    chance? Without it an index of 1.0 says nothing, because a design with three comparisons over
    four items cannot cycle no matter who chose.

    Both summaries come back, and the reason is that they cannot be mixed. The pooled index is a
    minimum over groups and it falls with bank size for a reason that has nothing to do with the
    grader: over one bank drawn from the same process it reads 0.1193 at 50 groups, 0.0592 at 500
    and 0.0390 at 3,000, while the mean sits at 0.64 throughout. Reporting a min against a mean
    baseline compares 0.039 with 0.633 and makes a bank look far worse than chance when it is
    exactly chance, so `afriat_pooled` is the baseline for `pooled` and `afriat_mean` for `mean`.
    """
    rng = np.random.default_rng(seed)
    indices: list[float] = []
    pooled: list[float] = []
    violations: list[float] = []
    for _ in range(n_draws):
        drawn: list[ComparisonSet] = []
        for cs in bank:
            if not cs.n_comparisons:
                drawn.append(cs)
                continue
            flip = rng.random(cs.n_comparisons) < 0.5
            w = np.where(flip, cs.losers, cs.winners)
            lo = np.where(flip, cs.winners, cs.losers)
            drawn.append(
                ComparisonSet(
                    n_items=cs.n_items,
                    winners=w.astype(np.intp),
                    losers=lo.astype(np.intp),
                    margins=rng.permutation(cs.margins),
                    weights=cs.weights,
                    margin_source=cs.margin_source,
                )
            )
        res = afriat_index(drawn)
        indices.append(res.mean)
        pooled.append(res.pooled)
        violations.append(float(res.n_violations))
    return {
        "afriat_mean": float(np.mean(indices)),
        "afriat_mean_sd": float(np.std(indices)),
        "afriat_pooled": float(np.mean(pooled)),
        "afriat_pooled_sd": float(np.std(pooled)),
        "violations_mean": float(np.mean(violations)),
        "n_draws": float(n_draws),
    }


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


class AfriatIndex(BaseObservable):
    """B2. How much of a grader's comparison record one scalar objective can order.

    Reported beside B1's curl mass, never instead of it. The catalogue's kill condition is *if
    Afriat and curl mass are monotonically related across twenty graders, in which case report one*,
    and the construction says why they need not be: curl mass is the squared energy a gradient field
    cannot reach, so a total order with equal margins carries curl; the Afriat index is the fraction
    of decisiveness that has to be discarded before one utility explains the verdicts, so a total
    order scores exactly 1.0 whatever its margins look like.
    """

    name = "AfriatIndex"
    version = "1.0"
    quantity = "grader.afriat_index"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to: str | None = "B2"
    deviations: tuple[str, ...] = (
        "Afriat's index is defined over budget sets. A comparison record has no prices, so each "
        "recorded verdict is treated as a two-element menu whose winner costs 1 and whose loser "
        "costs 1 - margin, with the margin normalised by the spread of its group. The efficiency "
        "then scales the budget exactly as it does in the original, and the construction is stated "
        "on the module rather than left implicit.",
        "The bank-level index is the pooled one, which is the minimum over groups because the "
        "groups are independent choice problems. That is Afriat's definition applied to the pooled "
        "data and it is a minimum statistic; the mean and median travel with it, and each is "
        "reported against the matching summary of the random-tournament baseline rather than "
        "against the other one.",
        "`grader.money_pump_index` is registered with unit `reward`/`raw` and a covariant relation "
        "to `reward.affine`, scaling by |a|. What this instrument computes is dimensionless and "
        "invariant: the margin it averages is already normalised, by the group's spread on a score "
        "record and by the total observations on a win-count record, so no shipped constructor "
        "produces a margin in reward units. The number is reported as computed and the mismatch is "
        "an open question for the registry rather than something resolved here.",
    )
    #: The access matrix. The catalogue prints GRADER:RECORD plus comparison data, and comparison
    #: data is what a grader record holds.
    requires = {Component.GRADER: Access.RECORD}
    substrates = frozenset({Substrate.NEURAL_GEN, Substrate.PROCEDURAL})
    #: The catalogue's phase column is OPEN for B2. An empty set is no restriction, which is the
    #: honest encoding of "not stated" and is what preflight reads.
    phases = frozenset()
    envelope = AFRIAT_ENVELOPE
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = (CURL_MASS, RANDOM_TOURNAMENT)
    rung = 1

    def __init__(
        self,
        bank: Sequence[ComparisonSet] | None = None,
        *,
        curl_mass: float | None = None,
        exact_max: int = 8,
        baseline_draws: int = 32,
        seed: int = 0,
    ) -> None:
        self.bank: tuple[ComparisonSet, ...] = tuple(bank or ())
        self.curl_mass = curl_mass
        self.exact_max = exact_max
        self.baseline_draws = baseline_draws
        self.seed = seed
        self._computed: Any = None

    # -- computation -------------------------------------------------------

    def compute(self) -> Any:
        """The four rungs, or the refusal that stands in for the one that cannot be reported."""
        if not self.bank or not any(cs.n_comparisons for cs in self.bank):
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    f"{len(self.bank)} groups carry no recorded comparison between distinct "
                    f"alternatives, so there is no revealed preference to test"
                ),
                remedy=(
                    "supply a comparison record: either per-group grader scores through "
                    "`bank_from_scores`, or an (i, j, wins_i, wins_j) edge log through "
                    "`ComparisonSet.from_edges`. A group whose scores are all equal records no "
                    "preference and is correctly absent from the count."
                ),
                statistics={"n_groups": len(self.bank)},
            )
        result = afriat_index(self.bank)
        hm = houtman_maks(self.bank, exact_max=self.exact_max)
        mp = money_pump(self.bank)
        base = random_tournament_index(self.bank, n_draws=self.baseline_draws, seed=self.seed)
        if result.ordinal_violations:
            partial = self._bound_evidence(result, hm)
            return bounded_refusal(
                self.name,
                RefusalReason.ABOVE_LOD_BELOW_LOQ,
                detail=(
                    f"{result.ordinal_violations} of {result.n_groups} groups violate the axiom "
                    f"with no margin on any of their verdicts, contributing "
                    f"{result.n_violations} violations over {result.n_comparisons} recorded "
                    f"verdicts. On a bare win/loss record the loser's value is 0, so the budget "
                    f"constraint binds at every efficiency and the index can only be 1 or 0. Each "
                    f"such group pins the pooled index at 0, which would say this grader is "
                    f"maximally irrational on the strength of a handful of verdicts."
                ),
                remedy=(
                    "supply margins and re-run: repeated comparisons give a win fraction "
                    "(`ComparisonSet.from_edges` reads them), and a scoring grader gives a score "
                    "difference (`bank_from_scores`). Until then read the Houtman-Maks index in "
                    f"`partial`, which is {hm.fraction:.4f} and stays informative on ordinal data."
                ),
                bound=partial,
                n_violations=result.n_violations,
                houtman_maks=hm.fraction,
                money_pump_mean=mp.mean,
            )
        self.rung = 0 if result.ordinal else 1
        return result, hm, mp, base

    def _bound_evidence(self, result: AfriatResult, hm: HoutmanMaksResult) -> Evidence:
        """The bound a refusal carries, built without a Context so a refusal needs no gates."""
        from reward_lens.core.evidence import make_evidence
        from reward_lens.core.types import SubjectRef

        return make_evidence(
            observable=self.name,
            observable_version=self.version,
            subject=SubjectRef(readout="comparisons"),
            value={
                "afriat_index_upper": 1.0,
                "afriat_index_is_below": 1.0,
                "houtman_maks_fraction": hm.fraction,
                "n_violations": result.n_violations,
                "n_comparisons": result.n_comparisons,
            },
            quantity=self.quantity,
        )

    def payload(self, computed: Any) -> dict[str, Any]:
        result, hm, mp, base = computed
        out: dict[str, float | int | bool | str] = {
            "afriat_index": result.pooled,
            "afriat_index_mean": result.mean,
            "afriat_index_median": result.median,
            "garp_violations": result.n_violations,
            "n_groups": result.n_groups,
            "n_comparisons": result.n_comparisons,
            "rationalizable": result.rationalizable,
            "ordinal_only": result.ordinal,
            "method": result.method,
            "houtman_maks_fraction": hm.fraction,
            "houtman_maks_dropped": hm.dropped,
            "houtman_maks_exact": hm.exact,
            "houtman_maks_method": hm.method,
            "money_pump_mean": mp.mean,
            "money_pump_max": mp.maximum,
            "money_pump_cycles": mp.n_cycles,
        }
        if not hm.exact:
            out["houtman_maks_lower"] = hm.lower_fraction
            out["houtman_maks_upper"] = hm.upper_fraction
        # The headline `afriat_index` is the pooled figure, which is a minimum over groups, so its
        # baseline is the chance minimum over the same groups. `afriat_index_mean` gets the chance
        # mean. A min against a mean is not a comparison: the min falls with bank size on data drawn
        # from the baseline's own process, so the gap it opens is a count of groups rather than a
        # property of the grader.
        baselines: dict[str, float] = {RANDOM_TOURNAMENT: base["afriat_pooled"]}
        if self.curl_mass is not None:
            baselines[CURL_MASS] = float(self.curl_mass)
            out["curl_mass"] = float(self.curl_mass)
        else:
            out["curl_mass"] = "not supplied; B1 owns it and this instrument does not compute it"
        out["baseline_random_tournament_pooled"] = base["afriat_pooled"]
        out["baseline_random_tournament_pooled_sd"] = base["afriat_pooled_sd"]
        out["baseline_random_tournament_mean"] = base["afriat_mean"]
        out["baseline_random_tournament_sd"] = base["afriat_mean_sd"]
        out["baseline_random_violations"] = base["violations_mean"]
        out["baseline_note"] = (
            "the pooled index is a minimum over groups and its baseline is the chance minimum over "
            "the same groups; the mean index is compared against the chance mean. The two pairs are "
            "not interchangeable, because a minimum degrades with the number of groups on data "
            "drawn from the baseline itself."
        )
        return {**out, "baselines": baselines}

    def measure(self, ctx: Context) -> Evidence:
        return ctx.emit(self.payload(self._computed))

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


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def _register() -> None:
    """B2's estimators, so `capability_report` knows they exist and what each gives up.

    Three quantities rather than one ladder. The Afriat index keeps its two rungs; Houtman-Maks and
    the money pump were rungs 2 and 3 of it and are now registered against their own ids, because
    they are different quantities in different units and not cheaper estimators of the same one. A
    rationalizability *efficiency*, a *fraction of the record kept*, and a *margin per trade* are
    three numbers, and stacking them on one ladder said any of them could stand in for the others.
    """
    _ensure_quantities()
    rows = (
        (
            "grader.afriat_index",
            0,
            "grader.afriat.garp_count",
            "downward",
            "counts the violations at e = 1 and returns 1.0 when there are none. Zero violations "
            "pins the index exactly; any violation only establishes that it is below 1, so this "
            "rung returns a bound rather than a point estimate.",
        ),
        (
            "grader.afriat_index",
            1,
            "grader.afriat.bisect",
            "approximately_unbiased",
            "the index itself, exact where the critical efficiencies can be enumerated and "
            "bisected to a stated tolerance where they cannot. It is a supremum that need not be "
            "attained: on a record that carries a cycle the axiom fails at the reported value "
            "itself and holds strictly below it, measured on all 2,763 cyclic sets of 4,000.",
        ),
        (
            "grader.houtman_maks_index",
            1,
            "grader.houtman_maks.feedback_arcs",
            "downward",
            "the largest rationalizable subset, as a minimum feedback arc set. Exact inside "
            "strongly connected components of at most eight items. Above that the greedy ordering "
            "can only over-count the deletions, measured at a mean 26.1% and a worst 150.0% over "
            "sixty eight-item tournaments, so the reported fraction sits BELOW the truth and is a "
            "floor. The bracket is carried and `exact` says which route fired.",
        ),
        (
            "grader.money_pump_index",
            1,
            "grader.money_pump.cycle_mean",
            "downward",
            "the mean margin around each enumerated cycle. Enumeration is capped, and a cap that "
            "bounds drops cycles, so the mean is taken over fewer of them than exist.",
        ),
    )
    for quantity, rung, impl, direction, why in rows:
        register_estimator(
            EstimatorEntry(
                quantity=quantity,
                impl=impl,
                requires={Component.GRADER: Access.RECORD},
                envelope=AFRIAT_ENVELOPE,
                rung=rung,
                bias=BiasStatement(direction=cast(BiasDirection, direction), why=why),
                cost=CostModel(note="arithmetic on a comparison record; no grader calls"),
                substrates=frozenset({Substrate.NEURAL_GEN, Substrate.PROCEDURAL}),
                run=None,
            )
        )


_register()


__all__ = [
    "AFRIAT_ENVELOPE",
    "CURL_MASS",
    "RANDOM_TOURNAMENT",
    "AfriatIndex",
    "AfriatResult",
    "ComparisonSet",
    "FeedbackArcs",
    "GarpReport",
    "HoutmanMaksResult",
    "MoneyPumpResult",
    "afriat_index",
    "afriat_index_of",
    "bank_from_payload",
    "bank_from_score_panels",
    "bank_from_scores",
    "direct_preference",
    "garp",
    "garp_violations",
    "houtman_maks",
    "min_feedback_arcs",
    "money_pump",
    "random_tournament_index",
    "transitive_closure",
]

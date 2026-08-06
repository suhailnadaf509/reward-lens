"""A4, the Blackwell order: rank graders by informativeness rather than accuracy.

Accuracy is the wrong statistic and the size of the gap is on the record: pairwise accuracy
correlates 0.220 to 0.382 with downstream RLHF performance, against 0.924 to 0.963 for a
rationale-based score. Rung 0 computes accuracy anyway, and prints it beside every verdict, because
the claim this instrument makes is a claim *against* accuracy and a claim against a number you did
not compute is a claim nobody has to believe.

**What the order is over.** A grader is a channel from a hidden state to a signal. Blackwell's
theorem says grader A is at least as useful as grader B for *every* downstream decision problem
exactly when B is a garbling of A: when there is a stochastic matrix ``M`` with ``P_B = P_A M``.
That is a statement about channels, not about scores, which is why this instrument needs no scalar
representation and why its envelope is unconditional. A grader whose preference structure has curl
mass above B5's bound is not scalar-representable and is still perfectly orderable here. The
catalogue carried `SCALAR_REPRESENTABLE` as a *requirement* for a while, because the source line
reads "`SCALAR_REPRESENTABLE` not required" and a negation transcribed as a list of one becomes a
requirement. That made the instrument for ranking graders refuse on exactly the graders it exists to
rank. It is corrected here.

**Store agreement patterns, not vote counts, and the reduction happens at read time.** Under
arbitrarily dependent errors the prevalence of the true state is completely unidentified: weak
stochastic-ordering restrictions leave the identified set at the whole of [0, 1]. What identifies it
is *which* graders agree on *which* prompts. A vote count ("3 of 5 said yes") throws that away and
it cannot be recovered afterwards, so `AgreementTable` stores the joint cell
``counts[state, level_A, level_B]`` and every marginal a caller wants is a sum taken on the way out.
This costs one extra axis on disk and it buys two things the marginals cannot give: the physical
garbling check below, and the ability to add a third grader later without re-scoring the first two.

**Existential garbling and physical garbling are different questions and both are worth having.**
The LP asks whether *some* ``M`` exists. The joint answers a stronger question: is B conditionally
independent of the state given A, ``P(b | a, state) = P(b | a)``? That is B adding literally nothing
to A on this data, and it is checkable only from the joint. `Blackwell.physical_gap` reports it in
total variation beside the existential verdict. A pair with an existential garbling and a large
physical gap is a pair where B is redundant *in principle* and is carrying independent information
*in fact*, which happens when the two graders' errors are negatively correlated, and it is the case
where dropping B because "it is dominated" costs you something.

**The verdict is tested, not thresholded.** Two estimated channels are never in an exact garbling
relation, because exact feasibility of a linear system in estimated coefficients is a measure-zero
event, so an infeasible LP on estimates is not evidence of anything. `deficiency` is minimised
under the null that a garbling exists, the null is simulated at the observed sample sizes, and the
verdict compares the observed deficiency to that simulated null. There is no tolerance constant to
argue about; there is a test level, and it is a parameter.

**How hard the LP is to satisfy depends on the design, and the reading says so.** Matching
``P_B = P_A M`` imposes ``n_states * (levels - 1)`` independent equations on ``levels * (levels - 1)``
free entries of ``M``. On a binary state with three signal levels that is 2 equations against 6
parameters, so the system is under-determined and what stops every pair being mutually ordered is
the requirement that ``M`` be stochastic rather than any shortage of freedom. The verdict is still
informative there, because non-negativity is a real constraint and pairs do come back unordered, but
it is weaker than the same verdict on a design with more states than levels.
`Blackwell.garbling_constraints` and `garbling_parameters` carry the two counts, and `says` names
the imbalance when there is one. The remedy for a caller who wants the stronger test is more
*states*, not more levels: a graded label, a difficulty tier, an error category.

Kill condition, from the catalogue: *if the Blackwell order and accuracy rank identically on ten
graders.* `Blackwell.agrees_with_accuracy` is the field that accumulates toward that, which is why
rung 0 is computed on every reading rather than offered as an option.

An LP that comes back infeasible is a **finding**, not a refusal: it says the two graders are not
ordered, which is the single most common true answer and is more informative than most orderings.
The refusals here are for the cases where the question cannot be asked at all: no shared labelled
slice, or a slice with one state in it.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BaselineID
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
)
from reward_lens.measure.controls._base import ControlInstrument
from reward_lens.measure.metrology.attenuation import ALL_SUBSTRATES

#: Catalogue A4: `Access GRADER:QUERY on a shared labelled slice`. GOLD:RECORD is the second half of
#: that sentence and it is declared, because the labels are what makes the channel a channel: with
#: no state to condition on there is no ``P(signal | state)`` and nothing to order.
BLACKWELL_ACCESS: dict[Component, Access] = {
    Component.GRADER: Access.QUERY,
    Component.GOLD: Access.RECORD,
}

#: Catalogue A4 names one baseline, and it is the thing this instrument argues with.
BLACKWELL_BASELINES: tuple[BaselineID, ...] = ("baseline.rewardbench_accuracy",)

#: Unconditional, with the justification quoting the line the merge inverted.
BLACKWELL_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "The source reads `Env SCALAR_REPRESENTABLE not required`. The Blackwell order is over "
        "channels rather than over scores, so a grader whose preference structure no scalar "
        "can carry is still orderable, and this instrument is the one that says so. No regime of a "
        "run can make a conditional distribution stop being a conditional distribution."
    ),
)

#: Default resolution of the signal axis. Three levels is the smallest that lets a grader express
#: "prefer the first", "prefer the second" and "cannot separate them", and the choice matters: at
#: many levels each level holds few items, every channel starts to look maximally informative
#: because each state maps to its own levels, and the LP degenerates on noise. It is reported on
#: every reading rather than buried, because two readings at different resolutions are not
#: comparable.
DEFAULT_LEVELS = 3

#: The test level for the simulated null, and the number of simulations behind it. A level, not a
#: threshold on the statistic: what counts as "a garbling" is decided by the sampling distribution
#: at the observed n rather than by a constant.
DEFAULT_ALPHA = 0.05
DEFAULT_SIMULATIONS = 200


class Verdict(enum.Enum):
    """The five things the comparison can conclude, and none of them is an error.

    ``NOT_ORDERED`` is the common case and it is a real finding: neither grader garbles the other,
    so which one is better depends on the downstream decision and no informativeness argument
    settles it. ``INDETERMINATE`` is different and means the sample cannot separate the two
    hypotheses, which is a fact about the slice rather than about the graders.
    """

    A_DOMINATES_B = "A dominates B"
    B_DOMINATES_A = "B dominates A"
    EQUIVALENT = "equivalent"
    NOT_ORDERED = "not ordered"
    INDETERMINATE = "indeterminate"


# ---------------------------------------------------------------------------
# Storage: the joint, never the margin
# ---------------------------------------------------------------------------


def quantile_levels(scores: np.ndarray, n_levels: int) -> np.ndarray:
    """Bin scores into ``n_levels`` by their own quantiles, which is what makes A4 affine-invariant.

    Equal-width bins would not do. Under ``r -> a*r + b`` with ``a > 0`` an equal-width grid moves
    with the data and the bin *edges* move too, but items near an edge cross it under rounding and
    the channel changes. Quantile bins are a function of the ranks alone, and an increasing affine
    map preserves ranks exactly, so the channel matrix is bit-identical before and after. The
    generated invariance test asserts that end to end rather than trusting the argument.

    Ties are assigned to the same level by construction, because the boundary is a rank threshold
    and equal scores have equal ranks under ``searchsorted`` on the sorted array.
    """
    s = np.asarray(scores, dtype=np.float64)
    if n_levels < 2:
        raise ValueError(f"a channel needs at least 2 signal levels; got {n_levels}")
    order = np.argsort(np.argsort(s, kind="stable"), kind="stable")
    binned = np.minimum((order * n_levels) // max(s.shape[0], 1), n_levels - 1)
    return np.asarray(binned, dtype=np.int64)


@dataclass(frozen=True)
class AgreementTable:
    """The joint cell ``counts[state, level_A, level_B]``, and nothing reduced on the way in.

    Built with `from_scores`, which bins each grader's scores by its own quantiles. The two graders
    are binned independently, which is the point: the table records that grader A put an item in its
    top third *and* grader B put the same item in its bottom third, and a pair of marginal
    histograms records only that each of those happened to somebody.
    """

    counts: np.ndarray
    grader_a: str = "A"
    grader_b: str = "B"
    state_names: tuple[str, ...] = ()
    n_levels: int = DEFAULT_LEVELS
    #: The raw sign agreement with the state, kept because it is what RewardBench prints and the
    #: binning cannot reconstruct it. Empty when the states are not binary.
    sign_correct_a: int = 0
    sign_correct_b: int = 0
    n_signed: int = 0

    def __post_init__(self) -> None:
        c = np.asarray(self.counts)
        if c.ndim != 3:
            raise ValueError(f"counts must be (state, level_A, level_B); got shape {c.shape}")
        if np.any(c < 0):
            raise ValueError("counts cannot be negative")

    @property
    def n_states(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n(self) -> int:
        return int(self.counts.sum())

    def state_totals(self) -> np.ndarray:
        return self.counts.sum(axis=(1, 2))

    def channel(self, which: str) -> np.ndarray:
        """``P(level | state)`` for one grader. The read-time reduction, taken here and not before.

        Rows of a state with no observations come back as a uniform distribution, and the instrument
        refuses before this is reached rather than relying on that: a uniform row is the least
        informative channel there is, so a silently uniform row would make an unobserved state look
        like a grader that cannot tell anything apart.
        """
        axis = 2 if which.lower() in ("a", self.grader_a.lower()) else 1
        joint = self.counts.sum(axis=axis)
        totals = joint.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(totals > 0, joint / np.maximum(totals, 1), 1.0 / joint.shape[1])
        return np.asarray(out, dtype=np.float64)

    def prior(self) -> np.ndarray:
        """The empirical state distribution. Empirical because nothing here identifies a prevalence.

        This is the sentence the catalogue's storage note is about. The prevalence of the true state
        is not identified from grader agreement under arbitrarily dependent errors, so the only
        honest prior is the one the labelled slice was drawn with, and any reading conditioned on it
        inherits whatever selection the slice carries. Reported on the reading as `prior`.
        """
        totals = self.state_totals().astype(np.float64)
        total = totals.sum()
        return totals / total if total > 0 else np.full_like(totals, 1.0 / totals.shape[0])

    @classmethod
    def from_scores(
        cls,
        scores_a: Sequence[float] | np.ndarray,
        scores_b: Sequence[float] | np.ndarray,
        states: Sequence[int] | np.ndarray,
        *,
        grader_a: str = "A",
        grader_b: str = "B",
        n_levels: int = DEFAULT_LEVELS,
        state_names: Sequence[str] = (),
    ) -> "AgreementTable":
        """Two graders' scores on the same items, plus the true state, into the joint table.

        ``states`` is the caller's: on a preference bank it is 1 when the first response is the
        better one and 0 otherwise, and the scores are then margins. On a labelled bank it is the
        class. Nothing here assumes binary.
        """
        a = np.asarray(scores_a, dtype=np.float64)
        b = np.asarray(scores_b, dtype=np.float64)
        st = np.asarray(states)
        if not (a.shape == b.shape == st.shape):
            raise ValueError(
                f"scores_a {a.shape}, scores_b {b.shape} and states {st.shape} index the same "
                f"shared slice and must agree. Two graders scored on different items have no joint."
            )
        uniq, st_idx = np.unique(st, return_inverse=True)
        la = quantile_levels(a, n_levels)
        lb = quantile_levels(b, n_levels)
        counts = np.zeros((uniq.shape[0], n_levels, n_levels), dtype=np.int64)
        np.add.at(counts, (st_idx, la, lb), 1)
        sign_a = sign_b = n_signed = 0
        if uniq.shape[0] == 2:
            positive = st_idx == 1
            n_signed = int(a.shape[0])
            sign_a = int(np.sum((a > 0) == positive))
            sign_b = int(np.sum((b > 0) == positive))
        return cls(
            counts=counts,
            grader_a=grader_a,
            grader_b=grader_b,
            state_names=tuple(state_names) or tuple(str(u) for u in uniq),
            n_levels=n_levels,
            sign_correct_a=sign_a,
            sign_correct_b=sign_b,
            n_signed=n_signed,
        )


# ---------------------------------------------------------------------------
# Rung 0: the wrong thing, computed for contrast
# ---------------------------------------------------------------------------


def bayes_accuracy(channel: np.ndarray, prior: np.ndarray) -> float:
    """The best accuracy this channel can achieve, under the plug-in Bayes rule.

    The steelman of rung 0. A grader's published accuracy uses whatever decision rule its scorer
    happened to pick, usually the sign of a margin; this is what it would get with the optimal rule
    for the observed prior. Reporting the steelman is what makes the contrast honest, because
    beating a badly-thresholded accuracy would prove nothing.
    """
    p = np.asarray(channel, dtype=np.float64)
    joint = p * np.asarray(prior, dtype=np.float64)[:, None]
    return float(joint.max(axis=0).sum())


def sign_accuracy(table: AgreementTable) -> tuple[float, float]:
    """RewardBench's number for each grader: the fraction of items where the margin has the right sign.

    NaN for both when the states are not binary, because the statistic is not defined there and a
    substituted 0.5 would be a number nobody could interpret.
    """
    if table.n_signed == 0:
        return float("nan"), float("nan")
    return table.sign_correct_a / table.n_signed, table.sign_correct_b / table.n_signed


# ---------------------------------------------------------------------------
# Rungs 1 and 2: the garbling LP and Le Cam deficiency
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeficiencyResult:
    """``delta = inf_M max_state (1/2)||P_B[state] - (P_A M)[state]||_1``, and the M that achieved it."""

    delta: float
    garbling: np.ndarray
    status: str
    feasible: bool


def le_cam_deficiency(p_a: np.ndarray, p_b: np.ndarray) -> DeficiencyResult:
    """Rung 2. How far B is from being a garbling of A, in total variation, as one LP.

    The minimand is a max over states of a half-L1 distance, which is not linear, so it is lifted
    the standard way: one non-negative slack per (state, level) bounding the absolute residual from
    both sides, one epigraph variable ``t`` above every state's half-sum, and ``t`` minimised. The
    result is exact rather than approximate, which matters because the alternative (a smooth
    surrogate) would make ``delta = 0`` unreachable and every pair would come back not ordered.

    ``delta = 0`` is the rung 1 answer: the LP is feasible, ``P_B = P_A M`` exactly, B is a garbling
    of A and A is at least as informative as B for every downstream decision. Rung 1 is therefore
    not a separate computation here, and calling it one would have meant running two solvers to
    answer one question. `garbling_feasible` is the rung 1 name and it reads this result.
    """
    a = np.asarray(p_a, dtype=np.float64)
    b = np.asarray(p_b, dtype=np.float64)
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"the two channels must be conditioned on the same states; got {a.shape[0]} and "
            f"{b.shape[0]} rows"
        )
    n_states, m_a = a.shape
    m_b = b.shape[1]
    n_m = m_a * m_b
    n_u = n_states * m_b
    n_var = n_m + n_u + 1
    t_idx = n_var - 1

    def m_at(i: int, j: int) -> int:
        return int(i * m_b + j)

    def u_at(s: int, j: int) -> int:
        return int(n_m + s * m_b + j)

    c = np.zeros(n_var)
    c[t_idx] = 1.0

    a_eq = np.zeros((m_a, n_var))
    for i in range(m_a):
        for j in range(m_b):
            a_eq[i, m_at(i, j)] = 1.0
    b_eq = np.ones(m_a)

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for s in range(n_states):
        for j in range(m_b):
            lower = np.zeros(n_var)
            upper = np.zeros(n_var)
            for i in range(m_a):
                lower[m_at(i, j)] = -a[s, i]
                upper[m_at(i, j)] = a[s, i]
            lower[u_at(s, j)] = -1.0
            upper[u_at(s, j)] = -1.0
            rows.append(lower)
            rhs.append(-b[s, j])
            rows.append(upper)
            rhs.append(b[s, j])
        epi = np.zeros(n_var)
        for j in range(m_b):
            epi[u_at(s, j)] = 0.5
        epi[t_idx] = -1.0
        rows.append(epi)
        rhs.append(0.0)

    res = linprog(
        c,
        A_ub=np.asarray(rows),
        b_ub=np.asarray(rhs),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0.0, None)] * n_var,
        method="highs",
    )
    if not res.success:
        # The feasible set is never empty (M uniform is always feasible) and t is bounded below by
        # zero, so a failure here is a solver problem rather than an infeasible problem, and it is
        # not the "graders are not ordered" case. That case is delta > 0 with the solve succeeding.
        raise RuntimeError(
            f"the deficiency LP did not solve: {res.message}. The feasible set contains the uniform "
            f"garbling and the objective is bounded below by zero, so this is a solver failure "
            f"rather than an unordered pair."
        )
    m = np.asarray(res.x[:n_m]).reshape(m_a, m_b)
    # A total variation distance cannot be negative; the solver returns -0.0 or a tiny negative at
    # the optimum and letting that through would put a negative distance in a store.
    delta = max(0.0, float(res.x[t_idx]))
    return DeficiencyResult(
        delta=delta, garbling=m, status=str(res.message), feasible=delta <= 1e-9
    )


def garbling_feasible(p_a: np.ndarray, p_b: np.ndarray, *, tol: float = 1e-9) -> bool:
    """Rung 1. Is there a stochastic ``M`` with ``P_B = P_A M``, to numerical tolerance.

    The tolerance here is a floating-point tolerance and nothing else: it is not a judgement about
    how close is close enough, because on estimated channels the honest answer to that question
    comes from the simulated null in `BlackwellOrder`, not from a constant. A caller reaching for
    this function directly on estimated channels will get ``False`` almost always, correctly.
    """
    return le_cam_deficiency(p_a, p_b).delta <= tol


def physical_gap(table: AgreementTable) -> float:
    """How far B is from being conditionally independent of the state given A, on this data.

    ``max over (state, level_a) of (1/2)||P(b | a, state) - P(b | a)||_1``, weighted by nothing:
    the max, because a garbling that fails on one cell is not a garbling. Cells with no observations
    are skipped rather than counted as agreeing, which is the conservative direction: an unobserved
    cell is not evidence of independence.

    This is the question the joint answers and the marginals cannot. The existential LP asks whether
    *some* channel from A's output to B's output reproduces B; this asks whether the channel that
    actually generated the data is one. They come apart when the two graders' errors are correlated,
    and that is the case where "B is dominated" is true in Blackwell's sense and false in the sense
    an engineer means when they delete grader B from the pipeline.
    """
    counts = np.asarray(table.counts, dtype=np.float64)
    marginal_ab = counts.sum(axis=0)
    worst = 0.0
    for a_level in range(counts.shape[1]):
        row_total = marginal_ab[a_level].sum()
        if row_total <= 0:
            continue
        pooled = marginal_ab[a_level] / row_total
        for s in range(counts.shape[0]):
            cell = counts[s, a_level]
            n = cell.sum()
            if n <= 0:
                continue
            worst = max(worst, 0.5 * float(np.abs(cell / n - pooled).sum()))
    return worst


# ---------------------------------------------------------------------------
# Rung 3: the loss, instantiated at the K we actually use
# ---------------------------------------------------------------------------


def group_ranking_regret(
    channel: np.ndarray,
    prior: np.ndarray,
    *,
    k: int,
    values: np.ndarray | None = None,
    draws: int = 20_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Rung 3. Expected regret of picking the top-scored of K rollouts, through this channel.

    Blackwell's theorem is an existence statement over all decision problems, which makes it strong
    and makes it silent about magnitude. This instantiates the one decision problem the loop
    actually solves: K rollouts arrive, the grader scores them, the highest score wins, and the
    regret is how much true value that choice gives up against the best available. A dominance
    verdict with no regret number beside it cannot tell you whether to care.

    Returns the regret and its Monte Carlo standard error, both from the same draws, so a difference
    between two graders that is inside the MC error reads as inside the MC error. Ties in the
    signal are broken uniformly at random, which is what a real argmax over equal scores does and is
    the reason a coarse channel is penalised here rather than rewarded.
    """
    p = np.asarray(channel, dtype=np.float64)
    pri = np.asarray(prior, dtype=np.float64)
    n_states, n_levels = p.shape
    v = np.arange(n_states, dtype=np.float64) if values is None else np.asarray(values, np.float64)
    if v.shape[0] != n_states:
        raise ValueError(f"values has {v.shape[0]} entries for {n_states} states")
    if k < 2:
        raise ValueError(f"a group ranking needs K >= 2; got {k}")
    rng = np.random.default_rng(seed)
    states = rng.choice(n_states, size=(draws, k), p=pri / pri.sum())
    # One categorical draw per rollout, vectorised by inverse-CDF on the state's own row.
    cdf = np.cumsum(p, axis=1)
    u = rng.random(size=(draws, k))
    levels = (u[..., None] > cdf[states]).sum(axis=-1)
    levels = np.minimum(levels, n_levels - 1)
    # argmax with uniform tie-breaking: perturb by a strictly smaller random key.
    key = levels + rng.random(size=levels.shape) * 0.5
    picked = np.take_along_axis(states, key.argmax(axis=1)[:, None], axis=1)[:, 0]
    best = v[states].max(axis=1)
    regret = best - v[picked]
    return float(regret.mean()), float(regret.std(ddof=1) / np.sqrt(draws))


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class Blackwell:
    """The verdict, the deficiencies behind it, and accuracy sitting beside it for contrast."""

    verdict: str
    delta_ab: float
    delta_ba: float
    #: The 1 - alpha point of the deficiency under a simulated null that the garbling exists.
    null_ab: float
    null_ba: float
    accuracy_a: float
    accuracy_b: float
    sign_accuracy_a: float
    sign_accuracy_b: float
    physical_gap: float
    prior: tuple[float, ...]
    n: int
    n_levels: int
    alpha: float
    simulations: int
    #: Independent equations the garbling system imposes, and free entries of M it has to satisfy
    #: them with. Constraints below parameters means the verdict rests on M being stochastic rather
    #: than on counting equations, and `says` reports that when it happens.
    garbling_constraints: int = 0
    garbling_parameters: int = 0
    grader_a: str = "A"
    grader_b: str = "B"
    #: Rung 3, when a K was supplied. Regret and its Monte Carlo standard error, per grader.
    k: int = 0
    regret_a: float = float("nan")
    regret_b: float = float("nan")
    regret_se_a: float = float("nan")
    regret_se_b: float = float("nan")
    baselines: Mapping[str, float] = field(default_factory=dict)

    @property
    def accuracy_ranks_a_first(self) -> bool | None:
        if not np.isfinite(self.accuracy_a) or not np.isfinite(self.accuracy_b):
            return None
        if self.accuracy_a == self.accuracy_b:
            return None
        return self.accuracy_a > self.accuracy_b

    @property
    def agrees_with_accuracy(self) -> bool | None:
        """Whether the informativeness order and the accuracy order pick the same grader.

        None when either order is silent: an unordered pair or a tied accuracy has nothing to
        agree or disagree with. The catalogue's kill condition counts the True cases over ten
        graders, so a None must not be counted as agreement.
        """
        first = self.accuracy_ranks_a_first
        if first is None:
            return None
        if self.verdict == Verdict.A_DOMINATES_B.value:
            return first is True
        if self.verdict == Verdict.B_DOMINATES_A.value:
            return first is False
        return None

    @property
    def says(self) -> str:
        acc = (
            f"Accuracy says {self.grader_a} {self.accuracy_a:.3f} against "
            f"{self.grader_b} {self.accuracy_b:.3f}."
        )
        if self.verdict == Verdict.A_DOMINATES_B.value:
            head = (
                f"Grader {self.grader_b} is a garbling of grader {self.grader_a}. "
                f"{self.grader_b} is dominated: no downstream use can prefer it."
            )
        elif self.verdict == Verdict.B_DOMINATES_A.value:
            head = (
                f"Grader {self.grader_a} is a garbling of grader {self.grader_b}. "
                f"{self.grader_a} is dominated: no downstream use can prefer it."
            )
        elif self.verdict == Verdict.EQUIVALENT.value:
            head = (
                f"Each of {self.grader_a} and {self.grader_b} is a garbling of the other, so they "
                f"carry the same information and the choice between them is a cost question."
            )
        elif self.verdict == Verdict.NOT_ORDERED.value:
            head = (
                f"Neither grader garbles the other (deficiency {self.delta_ab:.3f} and "
                f"{self.delta_ba:.3f} against nulls of {self.null_ab:.3f} and {self.null_ba:.3f}), "
                f"so which one is better depends on the downstream decision."
            )
        else:
            head = (
                f"{self.n} items cannot separate the two hypotheses at alpha = {self.alpha:g}: "
                f"both deficiencies are inside their simulated nulls in both directions."
            )
        if self.k:
            head += (
                f" At K = {self.k} the group-ranking regret is {self.regret_a:.4f} "
                f"(SE {self.regret_se_a:.4f}) for {self.grader_a} and {self.regret_b:.4f} "
                f"(SE {self.regret_se_b:.4f}) for {self.grader_b}."
            )
        return f"{head} {acc}{self._identification_clause()}"

    def _identification_clause(self) -> str:
        """Named when the garbling system has more freedom than the design constrains."""
        if self.garbling_constraints >= self.garbling_parameters:
            return ""
        return (
            f" With {len(self.prior)} states and {self.n_levels} signal levels the garbling system "
            f"has {self.garbling_parameters} free parameters against {self.garbling_constraints} "
            f"constraints, so this verdict rests on the garbling matrix having to be stochastic. "
            f"More states, not more levels, is what sharpens it."
        )


def _simulate_null(
    table: AgreementTable,
    p_a: np.ndarray,
    garbling: np.ndarray,
    *,
    simulations: int,
    alpha: float,
    seed: int,
) -> float:
    """The ``1 - alpha`` point of the deficiency when a garbling really does exist.

    Parametric bootstrap under the null, which is the only construction that answers the question
    being asked. The null model is the fitted one: state to A's level through the observed ``P_A``,
    then A's level to B's level through the fitted ``M``, with no path from the state to B except
    through A. Data is generated at the observed per-state sample sizes, so the returned point
    inherits the actual precision of the slice rather than an asymptotic approximation of it.
    """
    rng = np.random.default_rng(seed)
    totals = table.state_totals().astype(np.int64)
    n_states, m_a = p_a.shape
    m_b = garbling.shape[1]
    m = np.clip(np.asarray(garbling, dtype=np.float64), 0.0, None)
    row = m.sum(axis=1, keepdims=True)
    m = np.where(row > 0, m / np.maximum(row, 1e-300), 1.0 / m_b)
    out = np.empty(simulations, dtype=np.float64)
    for r in range(simulations):
        counts = np.zeros((n_states, m_a, m_b), dtype=np.int64)
        for s in range(n_states):
            n_s = int(totals[s])
            if n_s == 0:
                continue
            drawn_a = rng.choice(m_a, size=n_s, p=p_a[s] / p_a[s].sum())
            drawn_b = np.empty(n_s, dtype=np.int64)
            for i in range(m_a):
                sel = drawn_a == i
                if sel.any():
                    drawn_b[sel] = rng.choice(m_b, size=int(sel.sum()), p=m[i])
            np.add.at(counts, (np.full(n_s, s), drawn_a, drawn_b), 1)
        sim = AgreementTable(counts=counts, n_levels=table.n_levels)
        out[r] = le_cam_deficiency(sim.channel("a"), sim.channel("b")).delta
    return float(np.quantile(out, 1.0 - alpha))


class BlackwellOrder(ControlInstrument):
    """A4. Which of two graders is more informative, with accuracy printed beside the answer.

    One instrument covering all four rungs rather than four classes, because the rungs here are not
    alternative routes to one number: rung 0 is the comparator the finding is *against*, rung 1 is
    the special case ``delta = 0`` of rung 2, and rung 3 is a different quantity (a regret, in value
    units) that exists to say whether the ordering matters. Splitting them would have produced three
    readings nobody can put back together. ``rung`` is declared as 2, the highest rung whose
    estimator produces the declared quantity; rung 3 rides along when a ``k`` is supplied.
    """

    name = "BlackwellOrder"
    version = "1.0"
    quantity = "grader.blackwell_order"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    requires = BLACKWELL_ACCESS
    substrates = ALL_SUBSTRATES
    phases = frozenset({Phase.PRE_RUN, Phase.POST_RUN})
    envelope = BLACKWELL_ENVELOPE
    invariance = "reward.affine"
    #: Invariant. The channel is built from within-grader quantiles, so an increasing affine map of
    #: either grader's scores leaves every count in the joint table where it was and the deficiency
    #: is bit-identical. That is the check the catalogue's group is asking for: a garbling order
    #: that moved when somebody rescaled a reward would be an artifact of the scale.
    invariance_relation = INVARIANT
    baselines = BLACKWELL_BASELINES
    rung = 2
    faithful_to = "A4"
    deviations = (
        "the catalogue prints rung 1 (garbling LP) and rung 2 (Le Cam deficiency) as separate "
        "estimators. Feasibility is the delta = 0 case of the deficiency, so one LP answers both "
        "and running two would have been two solvers for one question",
        "the verdict compares the observed deficiency to a null simulated at the observed sample "
        "sizes rather than to a fixed tolerance. Exact feasibility of a linear system in estimated "
        "coefficients is a measure-zero event, so a fixed tolerance would report `not ordered` for "
        "every real pair at every sample size",
    )

    def __init__(
        self,
        table: AgreementTable | None = None,
        *,
        k: int = 0,
        values: np.ndarray | None = None,
        alpha: float = DEFAULT_ALPHA,
        simulations: int = DEFAULT_SIMULATIONS,
        seed: int = 0,
        regret_draws: int = 20_000,
    ) -> None:
        self.table = table
        self.k = int(k)
        self.values = values
        self.alpha = float(alpha)
        self.simulations = int(simulations)
        self.seed = int(seed)
        self.regret_draws = int(regret_draws)

    def compute(self) -> Any:
        table = self.table
        if table is None or table.n == 0:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    "no shared labelled slice: the two graders have no items in common with a "
                    "state attached, so there is no conditional distribution to order"
                ),
                remedy=(
                    "score both graders on the same items and attach the true state to each, then "
                    "build the table with AgreementTable.from_scores(scores_a, scores_b, states). "
                    "The slice does not have to be large; it has to be shared."
                ),
                statistics={"n": 0 if table is None else table.n},
            )
        totals = table.state_totals()
        observed_states = int((totals > 0).sum())
        if observed_states < 2:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    f"the labelled slice contains {observed_states} distinct state(s) over "
                    f"{table.n} items. A channel conditioned on one state is a single row, every "
                    f"one-row channel is a garbling of every other, and the order would come back "
                    f"`equivalent` for any pair of graders whatsoever"
                ),
                remedy=(
                    "supply a slice containing both states. On a preference bank that means "
                    "including pairs where the second response is the better one, not only pairs "
                    "ordered chosen-first, which is the commonest way this slice ends up degenerate."
                ),
                statistics={
                    "n": table.n,
                    "observed_states": observed_states,
                    "state_totals": totals.tolist(),
                },
            )

        p_a = table.channel("a")
        p_b = table.channel("b")
        ab = le_cam_deficiency(p_a, p_b)
        ba = le_cam_deficiency(p_b, p_a)
        null_ab = _simulate_null(
            table,
            p_a,
            ab.garbling,
            simulations=self.simulations,
            alpha=self.alpha,
            seed=self.seed,
        )
        null_ba = _simulate_null(
            table,
            p_b,
            ba.garbling,
            simulations=self.simulations,
            alpha=self.alpha,
            seed=self.seed + 1,
        )
        b_garbles_a = ab.delta <= null_ab
        a_garbles_b = ba.delta <= null_ba
        if b_garbles_a and a_garbles_b:
            verdict = Verdict.EQUIVALENT if table.n_levels > 1 else Verdict.INDETERMINATE
        elif b_garbles_a:
            verdict = Verdict.A_DOMINATES_B
        elif a_garbles_b:
            verdict = Verdict.B_DOMINATES_A
        else:
            verdict = Verdict.NOT_ORDERED

        prior = table.prior()
        acc_a = bayes_accuracy(p_a, prior)
        acc_b = bayes_accuracy(p_b, prior)
        sgn_a, sgn_b = sign_accuracy(table)
        regret_a = regret_b = se_a = se_b = float("nan")
        if self.k:
            regret_a, se_a = group_ranking_regret(
                p_a,
                prior,
                k=self.k,
                values=self.values,
                draws=self.regret_draws,
                seed=self.seed,
            )
            regret_b, se_b = group_ranking_regret(
                p_b,
                prior,
                k=self.k,
                values=self.values,
                draws=self.regret_draws,
                seed=self.seed,
            )
        return Blackwell(
            verdict=verdict.value,
            delta_ab=ab.delta,
            delta_ba=ba.delta,
            null_ab=null_ab,
            null_ba=null_ba,
            accuracy_a=acc_a,
            accuracy_b=acc_b,
            sign_accuracy_a=sgn_a,
            sign_accuracy_b=sgn_b,
            physical_gap=physical_gap(table),
            prior=tuple(float(x) for x in prior),
            n=table.n,
            n_levels=table.n_levels,
            alpha=self.alpha,
            simulations=self.simulations,
            garbling_constraints=observed_states * (table.n_levels - 1),
            garbling_parameters=table.n_levels * (table.n_levels - 1),
            grader_a=table.grader_a,
            grader_b=table.grader_b,
            k=self.k,
            regret_a=regret_a,
            regret_b=regret_b,
            regret_se_a=se_a,
            regret_se_b=se_b,
            baselines={
                "baseline.rewardbench_accuracy": sgn_a if np.isfinite(sgn_a) else acc_a,
            },
        )

    def payload(self, computed: Blackwell) -> dict[str, Any]:
        return {
            "verdict": computed.verdict,
            "delta_ab": computed.delta_ab,
            "delta_ba": computed.delta_ba,
            "null_ab": computed.null_ab,
            "null_ba": computed.null_ba,
            "accuracy_a": computed.accuracy_a,
            "accuracy_b": computed.accuracy_b,
            "sign_accuracy_a": computed.sign_accuracy_a,
            "sign_accuracy_b": computed.sign_accuracy_b,
            "physical_gap": computed.physical_gap,
            "agrees_with_accuracy": computed.agrees_with_accuracy,
            "prior": list(computed.prior),
            "n": computed.n,
            "n_levels": computed.n_levels,
            "alpha": computed.alpha,
            "simulations": computed.simulations,
            "garbling_constraints": computed.garbling_constraints,
            "garbling_parameters": computed.garbling_parameters,
            "k": computed.k,
            "regret_a": computed.regret_a,
            "regret_b": computed.regret_b,
            "says": computed.says,
            "baselines": dict(computed.baselines),
        }


def deficiency_from_scores(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    states: np.ndarray,
    *,
    n_levels: int = DEFAULT_LEVELS,
) -> float:
    """Scores in, deficiency out. The subject of the generated invariance test.

    Written as one callable from raw scores rather than from a table, so the affine group acts on
    the thing it is defined on and the quantile binning has to survive the transformation rather
    than being applied before it.
    """
    table = AgreementTable.from_scores(scores_a, scores_b, states, n_levels=n_levels)
    return le_cam_deficiency(table.channel("a"), table.channel("b")).delta


__all__ = [
    "BLACKWELL_ACCESS",
    "BLACKWELL_BASELINES",
    "BLACKWELL_ENVELOPE",
    "DEFAULT_ALPHA",
    "DEFAULT_LEVELS",
    "DEFAULT_SIMULATIONS",
    "AgreementTable",
    "Blackwell",
    "BlackwellOrder",
    "DeficiencyResult",
    "Verdict",
    "bayes_accuracy",
    "deficiency_from_scores",
    "garbling_feasible",
    "group_ranking_regret",
    "le_cam_deficiency",
    "physical_gap",
    "quantile_levels",
    "sign_accuracy",
]

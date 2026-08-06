"""A6, the grader stochasticity profile: the grader as a distribution, not a number.

Every reward-model paper writes ``r(x, y)`` as though the grader were a function. For a judge it is
a sample. The flip rates have been measured for judges in isolation, 13.6% on one benchmark with a
single question reaching 56%, and test-retest above 0.95 has been reported coexisting with position
bias above 0.10 in two production judges. Nobody has measured it for a reward model **inside a
loop**, where a flipped verdict is not a wrong row in a table but a sign change on an advantage.

Three readings, from one repeated-scoring design.

**Rung 0, the repeat variance.** Score the same input several times and report the spread. The
pooled within-item standard deviation is the headline and the per-item quantiles sit beside it,
because a grader with sigma = 0.18 on average and sigma = 0.7 on its worst decile is a different
object from one that is uniformly noisy, and the loop only sees the worst decile.

**Rung 1, per-facet attribution.** The spread is decomposed across whatever facets the caller
varied: which judge, which presentation order, which rubric revision. This matters because the
facets have different prices. Order effects are fixed by presenting both orders and averaging, which
costs one extra call. Judge-to-judge variance is fixed by using more judges, which costs a lot more.
A single sigma cannot tell you which bill you are paying.

**Rung 2, the flip rate at the pair level.** The number that actually reaches the optimiser. A
grader can be noisy and still rank a pair the same way every time, and it is the ranking that
becomes an advantage. `flip_rates` reports how often the pairwise verdict disagrees with its own
modal verdict, and `repeats_for_majority` converts that into the operational question: how many
repeats does a majority vote need before it reproduces a long-run reference.

**Two definitional decisions worth reading before quoting a flip rate.**

First, the flip rate is a **U-statistic over all pairs of repeats by default, not the diagonal**.
If repeat 3 of response A and repeat 3 of response B were not produced on the same occasion, then
pairing them by index is arbitrary, and a statistic that depends on an arbitrary pairing changes
when somebody reorders a file. Averaging over all ``m^2`` combinations removes the choice and, as a
side effect, makes the reading exactly invariant under `group.permutation`, which is the group the
catalogue assigns. Callers whose repeats genuinely are shared occasions, one judge draw scoring both
responses, pass ``paired_occasions=True`` and get the diagonal pairing, which is then correctly
sensitive to occasion.

Second, **ties are counted and reported, never split**. A grader that returns the same number for
both responses has not expressed a preference, and charging half a flip to it would make a
deterministic tie-returning grader look 50% noisy.

Kill condition, from the catalogue: *if sigma is approximately 0 for the grader class in question,
which is true for a deterministic scalar head and is the honest scope limit.* So sigma = 0 is a
**reading**, not a refusal: the instrument reports zero and sets ``deterministic``, and that reading
is what justifies not paying for replications on that grader class. What *is* a refusal is being
handed one draw per item, because a variance over one observation is not zero, it is undefined, and
those two have to stay distinguishable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import binom

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

#: Catalogue A6: `Access GRADER:REPLICATE`. The distinction from QUERY is the whole instrument: a
#: hosted judge you can call but cannot ask to vary its seed or its presentation order gives you one
#: draw per input forever, and one draw per input has no spread in it.
DISTRIBUTION_ACCESS: dict[Component, Access] = {Component.GRADER: Access.REPLICATE}

#: The catalogue splits this into two entries because the baseline field was split on commas where
#: the source separates baselines with semicolons. It is one baseline, and its second clause is an
#: instruction about how to report it rather than a second comparator.
DISTRIBUTION_BASELINES: tuple[BaselineID, ...] = ("baseline.assume_determinism",)

#: `Env none` in the source, transcribed. A grader's own spread is a property of the grader and
#: of the design that varied it, and no regime of a training run makes a measured spread wrong.
DISTRIBUTION_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "No regime condition is recorded for A6. This reads the grader's own output distribution "
        "under a design the caller controls, so there is no assumption about the run for a regime to "
        "violate. The scope limit is in the kill condition rather than in an envelope: on a "
        "deterministic scalar head the reading is zero and correct."
    ),
)

#: The long-run reference a majority vote is asked to reproduce, and the level it has to reproduce
#: it at. Both are parameters; these are the values the catalogue's illustration uses.
DEFAULT_REFERENCE_REPEATS = 50
DEFAULT_AGREEMENT_TARGET = 0.95

#: The largest odd vote size the search will consider before reporting that no reachable vote size
#: gets there. A cap rather than an infinite loop, and it is reported on the reading.
MAX_VOTE_REPEATS = 999


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepeatedScores:
    """A rectangular (item, repeat) score array, with the facets that were varied named.

    ``scores`` may hold NaN for a repeat that was not run, so a ragged design does not have to be
    padded with a plausible number. Everything downstream counts non-NaN entries rather than
    ``shape[1]``.

    ``facets`` maps a facet name to labels of the same shape as ``scores``, or to a length-``m``
    vector broadcast across items when the facet varies by repeat only. "order" with values 0 and 1
    for the two presentation orders is the commonest one and is the cheapest facet anybody can vary.
    """

    scores: np.ndarray
    item_ids: tuple[str, ...] = ()
    facets: Mapping[str, np.ndarray] = field(default_factory=dict)
    grader: str = ""
    #: True when repeat j of every item was produced on one shared occasion, so the diagonal
    #: pairing is meaningful. False, the default, treats repeats as exchangeable within an item.
    paired_occasions: bool = False

    def __post_init__(self) -> None:
        s = np.asarray(self.scores, dtype=np.float64)
        if s.ndim != 2:
            raise ValueError(f"scores must be (item, repeat); got shape {s.shape}")
        object.__setattr__(self, "scores", s)
        if self.item_ids and len(self.item_ids) != s.shape[0]:
            raise ValueError(
                f"{len(self.item_ids)} item ids for {s.shape[0]} rows of scores; they index the "
                f"same items"
            )
        for name, labels in self.facets.items():
            arr = np.asarray(labels)
            if arr.shape not in (s.shape, (s.shape[1],)):
                raise ValueError(
                    f"facet {name!r} has shape {arr.shape}; it must be {s.shape} (one label per "
                    f"score) or {(s.shape[1],)} (one label per repeat, broadcast across items)"
                )

    @property
    def n_items(self) -> int:
        return int(self.scores.shape[0])

    @property
    def n_repeats(self) -> int:
        return int(self.scores.shape[1])

    def counts(self) -> np.ndarray:
        """Non-NaN repeats per item."""
        return np.sum(np.isfinite(self.scores), axis=1)

    @classmethod
    def from_occasions(
        cls,
        occasions: Sequence[Sequence[float] | np.ndarray],
        *,
        item_ids: Sequence[str] = (),
        facet_name: str = "occasion",
        grader: str = "",
        paired_occasions: bool = True,
    ) -> "RepeatedScores":
        """Build from one score vector per occasion, which is how a re-scoring run arrives.

        The commonest real shape: a bank scored once, then scored again under a varied facet, giving
        two aligned vectors rather than an (item, repeat) matrix. Each occasion becomes a column and
        the facet label is the column index, so rung 1 can attribute to it.
        """
        cols = [np.asarray(o, dtype=np.float64).ravel() for o in occasions]
        if len({c.shape[0] for c in cols}) != 1:
            raise ValueError(
                f"occasions have lengths {[c.shape[0] for c in cols]}; they score the same items "
                f"and must align. An occasion that skipped items should carry NaN for them."
            )
        scores = np.stack(cols, axis=1)
        return cls(
            scores=scores,
            item_ids=tuple(item_ids),
            facets={facet_name: np.arange(len(cols))},
            grader=grader,
            paired_occasions=paired_occasions,
        )


# ---------------------------------------------------------------------------
# Rung 0
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepeatVariance:
    """Pooled within-item spread, with the per-item distribution behind it."""

    sigma: float
    variance: float
    df: int
    per_item_sigma: np.ndarray
    n_items_used: int
    n_scores: int

    @property
    def deterministic(self) -> bool:
        """Exactly zero spread on every item with more than one draw.

        Exact rather than "below a tolerance", because the case this flag exists for is a scalar
        head returning the identical float, and a tolerance would silently absorb a genuinely tiny
        but real stochasticity into "deterministic". A caller who wants a tolerance can compare
        ``sigma`` to one and say what it is.
        """
        return self.variance == 0.0


def repeat_variance(data: RepeatedScores) -> RepeatVariance:
    """Rung 0. The pooled within-item variance, weighting each item by its own degrees of freedom.

    Pooled rather than "mean of the per-item variances", because items with more repeats carry more
    information about the spread and an unweighted mean throws that away. On a rectangular design
    the two agree exactly; they come apart on a ragged one, which is what a real replication budget
    produces.
    """
    s = data.scores
    finite = np.isfinite(s)
    counts = finite.sum(axis=1)
    usable = counts >= 2
    ss = 0.0
    df = 0
    per_item = np.full(s.shape[0], np.nan)
    for i in np.flatnonzero(usable):
        row = s[i][finite[i]]
        centred = row - row.mean()
        ss += float(centred @ centred)
        df += row.shape[0] - 1
        per_item[i] = float(np.sqrt((centred @ centred) / (row.shape[0] - 1)))
    variance = ss / df if df > 0 else float("nan")
    return RepeatVariance(
        sigma=float(np.sqrt(variance)) if df > 0 else float("nan"),
        variance=float(variance),
        df=int(df),
        per_item_sigma=per_item,
        n_items_used=int(usable.sum()),
        n_scores=int(finite.sum()),
    )


# ---------------------------------------------------------------------------
# Rung 1
# ---------------------------------------------------------------------------


def facet_shares(data: RepeatedScores) -> tuple[dict[str, float], dict[str, float]]:
    """Rung 1. The share of within-item variance each named facet accounts for, and its null.

    A one-way decomposition per facet, taken *inside* each item and then pooled: for facet F with
    levels l, the between-level sum of squares within an item, summed over items, over the total
    within-item sum of squares. That is eta-squared computed within item, which is the right
    normalisation here because the item effect is not the thing being attributed.

    **The second dictionary is the reason this returns two.** Eta-squared has a large positive
    expectation under the null that the facet does nothing: a two-level facet across six repeats
    explains 20% of the within-item variance on average by chance alone, because it is fitting one
    parameter out of five degrees of freedom. Reporting 0.18 without saying that the null is 0.20
    would turn an ordinary sampling fluctuation into a finding. The null is exact rather than
    simulated: under H0 each item contributes ``(L_i - 1)`` expected between-level degrees of
    freedom out of ``(m_i - 1)`` total, so the pooled null share is the ratio of those sums.

    Shares do not sum to one and are not meant to. Two facets varied together are confounded and
    each will claim the shared part, which is a fact about the design rather than an arithmetic
    error, and the honest report is two numbers that overlap rather than one normalised split that
    hides the confounding. A crossed design with one observation per cell cannot separate them at
    all, and A2's crossed G-study is the instrument that can.
    """
    s = data.scores
    finite = np.isfinite(s)
    out: dict[str, float] = {}
    null: dict[str, float] = {}
    total_ss = 0.0
    total_df = 0
    for i in range(s.shape[0]):
        row = s[i][finite[i]]
        if row.shape[0] < 2:
            continue
        total_ss += float(np.sum((row - row.mean()) ** 2))
        total_df += row.shape[0] - 1
    for name, labels in data.facets.items():
        arr = np.asarray(labels)
        if arr.shape == (s.shape[1],):
            arr = np.broadcast_to(arr, s.shape)
        between = 0.0
        between_df = 0
        for i in range(s.shape[0]):
            mask = finite[i]
            row = s[i][mask]
            if row.shape[0] < 2:
                continue
            lab = np.asarray(arr[i])[mask]
            mean = row.mean()
            levels = np.unique(lab)
            between_df += levels.shape[0] - 1
            for level in levels:
                cell = row[lab == level]
                between += cell.shape[0] * float((cell.mean() - mean) ** 2)
        out[name] = between / total_ss if total_ss > 0.0 else 0.0
        null[name] = between_df / total_df if total_df > 0 else 0.0
    return out, null


@dataclass(frozen=True)
class FacetEffect:
    """A facet's systematic shift, separated from the noise it is otherwise pooled with.

    ``share`` is what fraction of the within-item variance the shift alone accounts for. The rest of
    that facet's share is unreproducible, and the difference between the two decides what to buy.
    A facet that is all shift is fixed by balancing the design: present both orders and average, and
    it is gone for the price of one extra call. A facet that is all noise is fixed only by paying
    for more draws, and no amount of balancing touches it.

    This is why `facet_shares` alone is not enough. On a two-level facet with one observation per
    cell, eta-squared computed within item is exactly 1.0 by construction, because the facet uses up
    the item's only degree of freedom. The main effect is still identifiable, because it is
    estimated *across* items, and it is the number worth reading.
    """

    name: str
    levels: tuple[float, ...]
    level_means: tuple[float, ...]
    effect: float
    se: float
    share: float

    @property
    def significant(self) -> bool:
        """Whether the shift is larger than twice its own standard error.

        Two standard errors rather than a test, because this is a screen over however many facets
        the caller varied, and calling it a p-value would invite a multiplicity correction the
        reading does not carry. `stats.multiplicity` has the correction for a caller who needs one.
        """
        return math.isfinite(self.se) and self.se > 0.0 and abs(self.effect) > 2.0 * self.se


def facet_effects(data: RepeatedScores) -> dict[str, FacetEffect]:
    """The systematic shift each facet produces, estimated across items and paired within them.

    For each item the level means are taken and centred on that item's own mean, which removes the
    item effect exactly. Averaging those centred means across items gives the facet's main effect;
    their spread across items gives its standard error. The effect reported is the range between
    the largest and smallest level, which for a two-level facet is the plain paired difference.

    Only levels present in an item contribute to that item, so a ragged design does not need
    balancing first. Items missing a level contribute nothing to the contrast rather than
    contributing a substituted value.
    """
    s = data.scores
    finite = np.isfinite(s)
    total_ss = 0.0
    for i in range(s.shape[0]):
        row = s[i][finite[i]]
        if row.shape[0] > 1:
            total_ss += float(np.sum((row - row.mean()) ** 2))
    out: dict[str, FacetEffect] = {}
    for name, labels in data.facets.items():
        arr = np.asarray(labels)
        if arr.shape == (s.shape[1],):
            arr = np.broadcast_to(arr, s.shape)
        levels = np.unique(arr[finite])
        per_item: dict[Any, list[float]] = {lv: [] for lv in levels.tolist()}
        for i in range(s.shape[0]):
            mask = finite[i]
            row = s[i][mask]
            if row.shape[0] < 2:
                continue
            lab = np.asarray(arr[i])[mask]
            mean = row.mean()
            for lv in levels.tolist():
                cell = row[lab == lv]
                if cell.size:
                    per_item[lv].append(float(cell.mean() - mean))
        means = np.array([np.mean(per_item[lv]) if per_item[lv] else np.nan for lv in levels])
        finite_means = means[np.isfinite(means)]
        effect = float(finite_means.max() - finite_means.min()) if finite_means.size > 1 else 0.0
        hi, lo = int(np.nanargmax(means)), int(np.nanargmin(means))
        paired = (
            np.array(per_item[levels.tolist()[hi]]) - np.array(per_item[levels.tolist()[lo]])
            if len(per_item[levels.tolist()[hi]]) == len(per_item[levels.tolist()[lo]])
            else np.empty(0)
        )
        se = (
            float(paired.std(ddof=1) / math.sqrt(paired.shape[0]))
            if paired.shape[0] > 1
            else float("nan")
        )
        # The sum of squares the level means alone explain, at the observed cell counts.
        explained = 0.0
        for i in range(s.shape[0]):
            mask = finite[i]
            row = s[i][mask]
            if row.shape[0] < 2:
                continue
            lab = np.asarray(arr[i])[mask]
            for k, lv in enumerate(levels.tolist()):
                n_cell = int(np.sum(lab == lv))
                if n_cell and np.isfinite(means[k]):
                    explained += n_cell * float(means[k] ** 2)
        out[name] = FacetEffect(
            name=name,
            levels=tuple(float(x) for x in levels),
            level_means=tuple(float(x) for x in means),
            effect=effect,
            se=se,
            share=explained / total_ss if total_ss > 0.0 else 0.0,
        )
    return out


# ---------------------------------------------------------------------------
# Rung 2
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlipProfile:
    """Pairwise verdict stability over every pair of items that has repeats on both sides."""

    flip_rate: float
    tie_rate: float
    n_pairs: int
    #: Per pair, the probability of its own modal verdict, conditional on not being a tie.
    modal_probability: np.ndarray
    paired_occasions: bool
    #: The median number of repeat combinations behind one pair's modal probability. This is the
    #: resolution of the estimate, and it is what decides whether a vote-size answer means anything:
    #: over two draws a pair's modal probability can only come back 0.5 or 1.0.
    combinations: int = 0

    @property
    def undetermined(self) -> float:
        """Fraction of pairs that came back an exact coin flip, so no vote size resolves them.

        Reported rather than folded into the flip rate, because a pair estimated at exactly 0.5 on
        four observations and a pair genuinely at 0.5 are the same number and different situations,
        and only the second one is a property of the grader.
        """
        p = self.modal_probability
        return float(np.mean(p <= 0.5 + 1e-12)) if p.size else float("nan")


def flip_rates(
    data: RepeatedScores, *, max_pairs: int | None = 20_000, seed: int = 0
) -> FlipProfile:
    """Rung 2. How often the pairwise verdict disagrees with its own modal verdict.

    For each pair of items, the verdict distribution is estimated over repeat combinations: all
    ``m_i * m_j`` of them when the repeats are exchangeable, the ``min(m_i, m_j)`` diagonal ones when
    they are shared occasions. The flip rate is one minus the modal probability, averaged over pairs
    with equal weight, because a pair is the unit the optimiser sees and a pair with more repeats is
    not a more important pair.

    ``max_pairs`` subsamples when the item count makes the full ``n choose 2`` too large. The draw
    is seeded and the number actually used is reported, so a subsampled reading says it is one.
    """
    s = data.scores
    finite = np.isfinite(s)
    usable = np.flatnonzero(finite.sum(axis=1) >= 1)
    n = usable.shape[0]
    if n < 2:
        return FlipProfile(
            flip_rate=float("nan"),
            tie_rate=float("nan"),
            n_pairs=0,
            modal_probability=np.empty(0),
            paired_occasions=data.paired_occasions,
        )
    pairs = [(int(usable[i]), int(usable[j])) for i in range(n) for j in range(i + 1, n)]
    if max_pairs is not None and len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[k] for k in sorted(idx)]

    modal = np.empty(len(pairs))
    ties = np.empty(len(pairs))
    combos = np.empty(len(pairs), dtype=np.int64)
    for k, (i, j) in enumerate(pairs):
        a = s[i][finite[i]]
        b = s[j][finite[j]]
        if data.paired_occasions:
            m = min(a.shape[0], b.shape[0])
            diff = a[:m] - b[:m]
        else:
            diff = (a[:, None] - b[None, :]).ravel()
        combos[k] = diff.shape[0]
        tie = float(np.mean(diff == 0.0))
        decided = diff[diff != 0.0]
        if decided.size == 0:
            modal[k] = 1.0
            ties[k] = 1.0
            continue
        up = float(np.mean(decided > 0.0))
        modal[k] = max(up, 1.0 - up)
        ties[k] = tie
    return FlipProfile(
        flip_rate=float(np.mean(1.0 - modal)),
        tie_rate=float(np.mean(ties)),
        n_pairs=len(pairs),
        modal_probability=modal,
        paired_occasions=data.paired_occasions,
        combinations=int(np.median(combos)),
    )


def _vote_accuracy(m: int, p: np.ndarray) -> np.ndarray:
    """P(a majority of m draws lands on the modal verdict), per pair. m odd."""
    return np.asarray(binom.sf((m - 1) // 2, m, p), dtype=np.float64)


def repeats_for_majority(
    profile: FlipProfile,
    *,
    reference_repeats: int = DEFAULT_REFERENCE_REPEATS,
    target: float = DEFAULT_AGREEMENT_TARGET,
    max_repeats: int = MAX_VOTE_REPEATS,
) -> tuple[int, float]:
    """The smallest odd vote size whose verdict reproduces a long-run reference at ``target``.

    Both votes are drawn from the same per-pair Bernoulli, so the probability they agree is
    ``a_m * a_ref + (1 - a_m) * (1 - a_ref)`` with ``a`` the probability each lands on the modal
    verdict. Written out rather than approximated by ``a_m`` alone because the reference is not
    itself perfect: on a pair with modal probability 0.55, a 50-trial majority reproduces the modal
    verdict only 77% of the time, and an instrument that pretended otherwise would report a vote
    size that reproduces something nobody has.

    Odd sizes only, so there is no tie to break and no tie-breaking convention to argue about.
    Returns the size and the agreement it achieves; ``(-1, achieved)`` when ``max_repeats`` is
    reached, which happens when some pair sits at a modal probability near one half and no vote size
    resolves it.
    """
    p = profile.modal_probability
    if p.size == 0:
        return -1, float("nan")
    a_ref = _vote_accuracy(reference_repeats | 1, p)
    m = 1
    while m <= max_repeats:
        a_m = _vote_accuracy(m, p)
        agreement = float(np.mean(a_m * a_ref + (1.0 - a_m) * (1.0 - a_ref)))
        if agreement >= target:
            return m, agreement
        m += 2
    a_m = _vote_accuracy(max_repeats | 1, p)
    return -1, float(np.mean(a_m * a_ref + (1.0 - a_m) * (1.0 - a_ref)))


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class ScoreDistribution:
    """Sigma, the flip rate, the vote size, and the facets that account for them."""

    sigma: float
    variance: float
    df: int
    deterministic: bool
    n_items: int
    n_repeats: int
    n_scores: int
    sigma_quantiles: Mapping[str, float]
    flip_rate: float
    tie_rate: float
    n_pairs: int
    repeats_needed: int
    achieved_agreement: float
    reference_repeats: int
    target: float
    paired_occasions: bool
    facet_shares: Mapping[str, float] = field(default_factory=dict)
    #: What each facet's share would be if the facet did nothing. Read every share against its own
    #: entry here; a share at or below its null is not evidence that the facet matters.
    facet_null: Mapping[str, float] = field(default_factory=dict)
    #: The systematic shift each facet produces, separated from the noise it is pooled with.
    facet_effects: Mapping[str, FacetEffect] = field(default_factory=dict)
    #: Repeat combinations behind one pair's modal probability, and the fraction of pairs that came
    #: back an exact coin flip. Both are needed to read `repeats_needed`.
    combinations_per_pair: int = 0
    undetermined_pairs: float = float("nan")
    grader: str = ""
    baselines: Mapping[str, float] = field(default_factory=dict)

    @property
    def says(self) -> str:
        if self.deterministic:
            return (
                f"At fixed input this grader returned the identical score on every one of "
                f"{self.n_scores} draws over {self.n_items} items. It is deterministic on this "
                f"design, so replication buys nothing here and the single-draw assumption is "
                f"correct for this grader class."
            )
        vote = (
            f"{self.repeats_needed} repeats are needed for a majority vote to match a "
            f"{self.reference_repeats}-trial reference at {self.target:.0%}."
            if self.repeats_needed > 0
            else (
                f"No vote size up to {MAX_VOTE_REPEATS} matches a {self.reference_repeats}-trial "
                f"reference at {self.target:.0%}; the best reached is "
                f"{self.achieved_agreement:.1%}, because {self.undetermined_pairs:.1%} of pairs "
                f"came back an exact coin flip over {self.combinations_per_pair} repeat "
                f"combination(s) and no vote resolves a pair the design cannot separate."
            )
        )
        return (
            f"At fixed input this grader has sigma = {self.sigma:.3g} and flips its pairwise "
            f"verdict {self.flip_rate:.1%} of the time. {vote}{self._facet_clause()}"
        )

    def _facet_clause(self) -> str:
        """What the facets did, preferring the identifiable main effect over eta-squared.

        The main effect is quoted first when there is one, because it is the number that changes a
        decision: a shift is designed away by balancing and noise is not, and eta-squared cannot
        tell those apart on a design with one observation per cell.
        """
        shifts = {
            name: fx for name, fx in self.facet_effects.items() if fx.significant and fx.effect > 0
        }
        if shifts:
            name, fx = max(shifts.items(), key=lambda kv: kv[1].effect)
            return (
                f" The {name} facet produces a systematic shift of {fx.effect:.3g} "
                f"(SE {fx.se:.3g}), which is {fx.share:.0%} of the within-item variance and is "
                f"removed by balancing the design rather than by more repeats."
            )
        excess = {
            name: share - self.facet_null.get(name, 0.0)
            for name, share in self.facet_shares.items()
        }
        if not excess:
            return ""
        name, gap = max(excess.items(), key=lambda kv: kv[1])
        if gap <= 0.0:
            return (
                f" No varied facet shows a systematic shift or explains more than its own null "
                f"share, so the spread is not attributable to any of "
                f"{', '.join(sorted(self.facet_shares))} at this design."
            )
        return (
            f" The {name} facet accounts for {self.facet_shares[name]:.0%} of the within-item "
            f"variance against a null share of {self.facet_null.get(name, 0.0):.0%}."
        )


class GraderStochasticity(ControlInstrument):
    """A6. The grader's own output distribution, its facets, and the flip rate it produces.

    One instrument covering the three rungs, because they are three readings of one design rather
    than three routes to one number and separating them would have made a caller run the same
    repeated scoring three times. ``rung`` is 2, the highest the supplied data reaches; a design with
    no facet labels reports rung 2 with an empty ``facet_shares``, and the emptiness is the report
    that rung 1 was not available rather than a claim that no facet contributed.
    """

    name = "GraderStochasticity"
    version = "1.0"
    quantity = "grader.score_distribution"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    requires = DISTRIBUTION_ACCESS
    substrates = ALL_SUBSTRATES
    phases = frozenset({Phase.PRE_RUN, Phase.POST_RUN})
    envelope = DISTRIBUTION_ENVELOPE
    invariance = "group.permutation"
    #: Invariant. Every statistic here is a function of the multiset of repeats within an item, and
    #: `group.permutation` reorders repeats within a group without changing the multiset. That is
    #: true by construction for sigma and true by the U-statistic choice for the flip rate: the
    #: diagonal pairing would have failed this test, which is why the diagonal is not the default.
    invariance_relation = INVARIANT
    baselines = DISTRIBUTION_BASELINES
    rung = 2
    faithful_to = "A6"
    deviations = (
        "the flip rate is averaged over all repeat combinations rather than over a diagonal "
        "pairing. The catalogue does not say which, and the diagonal depends on an ordering the "
        "data does not carry unless the caller declares `paired_occasions`",
        "`repeats_for_majority` compares two finite votes rather than treating the reference as "
        "exact, so the reported vote size accounts for the reference's own instability",
    )

    def __init__(
        self,
        data: RepeatedScores | None = None,
        *,
        reference_repeats: int = DEFAULT_REFERENCE_REPEATS,
        target: float = DEFAULT_AGREEMENT_TARGET,
        max_pairs: int | None = 20_000,
        seed: int = 0,
    ) -> None:
        self.data = data
        self.reference_repeats = int(reference_repeats)
        self.target = float(target)
        self.max_pairs = max_pairs
        self.seed = int(seed)

    def compute(self) -> Any:
        data = self.data
        if data is None or data.scores.size == 0:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no repeated scores were supplied, and a spread needs repeats",
                remedy=(
                    "score the same inputs at least twice under a facet you control and pass the "
                    "result as RepeatedScores(scores=array_of_shape_items_by_repeats). Presentation "
                    "order is the cheapest facet to vary and needs no seed control."
                ),
                statistics={"n_items": 0, "n_repeats": 0},
            )
        counts = data.counts()
        replicated = int(np.sum(counts >= 2))
        if replicated == 0:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    f"every one of {data.n_items} items has at most one usable draw "
                    f"(max {int(counts.max()) if counts.size else 0}), so the within-item variance "
                    f"is undefined rather than zero. One draw per item is GRADER:RECORD; this "
                    f"quantity needs GRADER:REPLICATE"
                ),
                remedy=(
                    "re-score the same items at least once more. If the grader is behind an API "
                    "that will not vary its seed, vary the presentation order instead: that is a "
                    "facet, it needs no cooperation from the vendor, and it is the facet with the "
                    "largest published effect."
                ),
                statistics={
                    "n_items": data.n_items,
                    "n_repeats": data.n_repeats,
                    "items_with_two_or_more": 0,
                    "max_draws_per_item": int(counts.max()) if counts.size else 0,
                },
            )

        rv = repeat_variance(data)
        profile = flip_rates(data, max_pairs=self.max_pairs, seed=self.seed)
        needed, achieved = repeats_for_majority(
            profile, reference_repeats=self.reference_repeats, target=self.target
        )
        finite_sigma = rv.per_item_sigma[np.isfinite(rv.per_item_sigma)]
        quantiles = (
            {
                "p50": float(np.quantile(finite_sigma, 0.5)),
                "p90": float(np.quantile(finite_sigma, 0.9)),
                "max": float(finite_sigma.max()),
            }
            if finite_sigma.size
            else {}
        )
        shares, null = facet_shares(data)
        return ScoreDistribution(
            sigma=rv.sigma,
            variance=rv.variance,
            df=rv.df,
            deterministic=rv.deterministic,
            n_items=data.n_items,
            n_repeats=data.n_repeats,
            n_scores=rv.n_scores,
            sigma_quantiles=quantiles,
            flip_rate=profile.flip_rate,
            tie_rate=profile.tie_rate,
            n_pairs=profile.n_pairs,
            repeats_needed=needed,
            achieved_agreement=achieved,
            reference_repeats=self.reference_repeats,
            target=self.target,
            paired_occasions=data.paired_occasions,
            facet_shares=shares,
            facet_null=null,
            facet_effects=facet_effects(data),
            combinations_per_pair=profile.combinations,
            undetermined_pairs=profile.undetermined,
            grader=data.grader,
            # The baseline is "assume determinism", whose sigma is zero by construction. The error
            # that assumption induces is the reading itself, which is why the two sit together.
            baselines={"baseline.assume_determinism": 0.0},
        )

    def payload(self, computed: ScoreDistribution) -> dict[str, Any]:
        return {
            "sigma": computed.sigma,
            "variance": computed.variance,
            "df": computed.df,
            "deterministic": computed.deterministic,
            "n_items": computed.n_items,
            "n_repeats": computed.n_repeats,
            "n_scores": computed.n_scores,
            "sigma_quantiles": dict(computed.sigma_quantiles),
            "flip_rate": computed.flip_rate,
            "tie_rate": computed.tie_rate,
            "n_pairs": computed.n_pairs,
            "repeats_needed": computed.repeats_needed,
            "achieved_agreement": computed.achieved_agreement,
            "reference_repeats": computed.reference_repeats,
            "target": computed.target,
            "paired_occasions": computed.paired_occasions,
            "facet_shares": dict(computed.facet_shares),
            "facet_null": dict(computed.facet_null),
            "facet_effect": {n: fx.effect for n, fx in computed.facet_effects.items()},
            "facet_effect_se": {n: fx.se for n, fx in computed.facet_effects.items()},
            "facet_effect_share": {n: fx.share for n, fx in computed.facet_effects.items()},
            "combinations_per_pair": computed.combinations_per_pair,
            "undetermined_pairs": computed.undetermined_pairs,
            "grader": computed.grader,
            "says": computed.says,
            "baselines": dict(computed.baselines),
        }


def sigma_from_scores(scores: np.ndarray, n_repeats: int) -> float:
    """Flat scores in, pooled within-item sigma out. The subject of the generated invariance test.

    Takes the flat vector the invariance payload carries and folds it back into (item, repeat), so
    the group acts on the scores and the whole estimator runs on the transformed array rather than
    on a precomputed statistic.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1, n_repeats)
    return repeat_variance(RepeatedScores(scores=s)).sigma


def flip_rate_from_scores(scores: np.ndarray, n_repeats: int) -> float:
    """The same, for the flip rate, which is the reading the permutation group actually bites on."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1, n_repeats)
    rate = flip_rates(RepeatedScores(scores=s)).flip_rate
    return 0.0 if math.isnan(rate) else rate


__all__ = [
    "DEFAULT_AGREEMENT_TARGET",
    "DEFAULT_REFERENCE_REPEATS",
    "DISTRIBUTION_ACCESS",
    "DISTRIBUTION_BASELINES",
    "DISTRIBUTION_ENVELOPE",
    "MAX_VOTE_REPEATS",
    "FlipProfile",
    "GraderStochasticity",
    "RepeatVariance",
    "RepeatedScores",
    "ScoreDistribution",
    "FacetEffect",
    "facet_effects",
    "facet_shares",
    "flip_rate_from_scores",
    "flip_rates",
    "repeat_variance",
    "repeats_for_majority",
    "sigma_from_scores",
]

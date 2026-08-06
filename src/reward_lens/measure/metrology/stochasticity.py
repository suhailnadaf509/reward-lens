"""A6's two quantities, each declared by the instrument that computes it.

A6 reports two numbers from one repeated-scoring design, and until this module they shared one
registry id. They cannot. The spread is **covariant** under `reward.affine`: rescale a grader's
output by `a` and its standard deviation scales by `|a|`. The flip rate is **invariant** under the
same group: an affine rescaling with positive `a` cannot change which of two scores is larger, and
even a sign flip leaves the modal verdict's probability alone because that probability is
`max(p, 1 - p)`. `Quantity.invariance_group` is one field, so one id would have to lie about one of
them, and `Instrument.quantity` is one field, so one instrument would have to lie the same way one
level up. Two quantities, two instruments.

That is not a bookkeeping preference. Declaring the spread invariant is exactly the mistake the
split exists to prevent, and it fails loudly now. On the 24-by-5 panel in
`tests/acceptance/test_a6_split.py`, sigma declared invariant under `reward.affine` misses by 5.388
against a tolerance of 9.454e-08, weight 2 misses by 36.1 and weight 0.5 by 3.886. Weight 1 lands at
1.776e-15. There is exactly one right answer, the two neighbouring answers are ruled out by three
orders of magnitude each, and the `Relation` type is what lets the generated test ask the question at
all. With the status on the group alone, the only available assertion is equality and every one of
those four declarations passes.

**Why the covariance is a live problem rather than a formality.** The eleven open reward models in
the campaign store have raw standard deviations from 0.0538 to 17.05 on one shared bank of 7,052
responses, a factor of 317. A sigma of 0.18 from one of them and 1.33 from another is not a statement
about which grader is noisier, and nothing in the number itself says which scale it is on. The flip
rate needs no such care and can be compared across graders as it stands. That asymmetry is the
practical content of the split: two readings from one design, one of which travels between graders
and one of which does not.

**What each instrument covers.** The catalogue's ladder for A6 has rung 0 repeat variance, rung 1
per-facet attribution, rung 2 the flip rate at the pair level, and the split falls on exactly that
seam, which is some evidence it is a real one. `GraderScoreSigma` is rungs 0 and 1: the pooled
within-item spread, the per-item quantiles behind it, and the attribution of that spread across
whatever facets the caller varied. `GraderFlipRate` is rung 2: how often the pairwise verdict
disagrees with its own modal verdict, and how many repeats a majority vote needs before it
reproduces a long-run reference.

Both read the same `RepeatedScores` and call the same estimators, which live in
`measure/metrology/distribution.py` and are not reimplemented here. `stochasticity_profile` runs
both from one design so that splitting the declaration does not cost a caller a second scoring pass.

**The scope limit, three lines in as it should be.** Sigma is approximately zero for a deterministic
scalar head, and that is a reading rather than a failure: the instrument reports zero, sets
`deterministic`, and that reading is what justifies not paying for replications on that grader class.
What is not a reading is one draw per item. A variance over one observation is undefined, and a flip
rate over one draw per item is worse than undefined: every pair is unanimous by construction, so the
estimator returns exactly 0.0 and it is a confident wrong zero rather than a missing value. Both
instruments refuse there, and `GraderFlipRate` reports how many of its pairs had more than one repeat
combination behind them so a partly-replicated design cannot quietly average its way toward zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import register_payload
from reward_lens.core.invariance import COVARIANT_LINEAR, INVARIANT, Relation
from reward_lens.core.quantity import BaselineID
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Capability, GaugeStatus, Phase
from reward_lens.measure.controls._base import ControlInstrument
from reward_lens.measure.metrology.attenuation import ALL_SUBSTRATES
from reward_lens.measure.metrology.distribution import (
    DEFAULT_AGREEMENT_TARGET,
    DEFAULT_REFERENCE_REPEATS,
    DISTRIBUTION_ACCESS,
    MAX_VOTE_REPEATS,
    FacetEffect,
    RepeatedScores,
    facet_effects,
    facet_shares,
    flip_rates,
    repeat_variance,
    repeats_for_majority,
)

#: One baseline, not two. The source reads "Base assume determinism, and show the error that
#: induces", and the clause after the comma is an instruction about how to report the baseline
#: rather than a second comparator. A list split on the wrong separator reads as more rigour than
#: the instrument has, in the exact field `lint_instrument` reads.
STOCHASTICITY_BASELINES: tuple[BaselineID, ...] = ("baseline.assume_determinism",)

#: Both instruments declare both groups. `reward.affine` is the group that separates the two
#: quantities and is what each registry row carries; `group.permutation` is A6's own catalogue cell
#: and is a separate true statement that no single registry row can hold. Declaring both means both
#: generated tests run, which is two checks for the price of one honest declaration.
STOCHASTICITY_GROUPS = "reward.affine, group.permutation"


# ---------------------------------------------------------------------------
# Rung 0 and rung 1: the spread
# ---------------------------------------------------------------------------


def facet_clause(
    shares: Mapping[str, float],
    null: Mapping[str, float],
    effects: Mapping[str, FacetEffect],
) -> str:
    """What the varied facets did, preferring the identifiable main effect over eta-squared.

    The main effect is quoted first when there is one, because it is the number that changes a
    decision. A systematic shift is designed away by balancing (present both presentation orders and
    average, one extra call) and noise is not (buy more draws, which costs a lot more). Eta-squared
    cannot tell those apart on a design with one observation per cell, where it is exactly 1.0 by
    construction because the facet consumes the item's only degree of freedom.

    Every share is quoted against its own null, because eta-squared has a large positive expectation
    under the hypothesis that the facet does nothing: a two-level facet across six repeats explains
    20% of the within-item variance by chance alone. Reporting 0.18 without saying the null is 0.20
    turns a sampling fluctuation into a finding.

    ``ScoreDistribution._facet_clause`` in `distribution.py` is this logic on the superseded payload.
    The two are the same sentence and should not stay two; when `grader.score_distribution` is
    retired, that method goes with its payload rather than being kept in step by hand.
    """
    shifts = {name: fx for name, fx in effects.items() if fx.significant and fx.effect > 0}
    if shifts:
        name, fx = max(shifts.items(), key=lambda kv: kv[1].effect)
        return (
            f" The {name} facet produces a systematic shift of {fx.effect:.3g} (SE {fx.se:.3g}), "
            f"which is {fx.share:.0%} of the within-item variance and is removed by balancing the "
            f"design rather than by more repeats."
        )
    excess = {name: share - null.get(name, 0.0) for name, share in shares.items()}
    if not excess:
        return ""
    name, gap = max(excess.items(), key=lambda kv: kv[1])
    if gap <= 0.0:
        return (
            f" No varied facet shows a systematic shift or explains more than its own null share, "
            f"so the spread is not attributable to any of {', '.join(sorted(shares))} at this design."
        )
    return (
        f" The {name} facet accounts for {shares[name]:.0%} of the within-item variance against a "
        f"null share of {null.get(name, 0.0):.0%}."
    )


@register_payload
@dataclass
class ScoreSpread:
    """`grader.score_sigma`: the pooled within-item spread, with what it is made of.

    The unit is the grader's own reward scale, and it is the reason this reading is covariant. Two
    graders' sigmas are not comparable until both are on one scale, and the number carries no record
    of which scale that is, so `Unit` and the gauge machinery are what stop the comparison rather
    than anything in this payload.
    """

    sigma: float
    variance: float
    df: int
    deterministic: bool
    n_items: int
    n_repeats: int
    n_scores: int
    n_items_used: int
    #: p50, p90 and max of the per-item spreads. A grader at sigma 0.18 on average and 0.7 on its
    #: worst decile is a different object from one at 0.25 everywhere, and the loop sees the decile.
    sigma_quantiles: Mapping[str, float] = field(default_factory=dict)
    facet_shares: Mapping[str, float] = field(default_factory=dict)
    #: What each share would be if the facet did nothing. Read every share against its own entry.
    facet_null: Mapping[str, float] = field(default_factory=dict)
    facet_effects: Mapping[str, FacetEffect] = field(default_factory=dict)
    grader: str = ""
    baselines: Mapping[str, float] = field(default_factory=dict)

    @property
    def says(self) -> str:
        if self.deterministic:
            return (
                f"At fixed input this grader returned the identical score on every one of "
                f"{self.n_scores} draws over {self.n_items_used} replicated items. Sigma is exactly "
                f"zero, so replication buys nothing on this grader class and the single-draw "
                f"assumption is correct here."
            )
        # The two numbers the catalogue's own illustration contrasts: what a typical item costs and
        # what the worst decile costs. Both are always quoted rather than one being quoted when it
        # clears a threshold, because the threshold would be invented here and the contrast is the
        # reading. A uniformly noisy grader shows the two close together, which is itself the answer.
        p50 = self.sigma_quantiles.get("p50")
        p90 = self.sigma_quantiles.get("p90")
        decile = (
            f" A typical item costs {p50:.3g} and the worst decile costs {p90:.3g}, so the loop "
            f"sees {p90:.3g} on the items where it matters most."
            if p50 is not None and p90 is not None
            else ""
        )
        return (
            f"At fixed input this grader has sigma = {self.sigma:.3g} on its own reward scale, "
            f"pooled over {self.n_items_used} replicated items.{decile}"
            f"{facet_clause(self.facet_shares, self.facet_null, self.facet_effects)}"
        )


class GraderScoreSigma(ControlInstrument):
    """A6 rungs 0 and 1. How much of the score is the grader arguing with itself.

    Covariant under `reward.affine` with weight 1, which is the declaration that makes this a
    separate instrument from `GraderFlipRate` rather than a second field on one. Invariant under
    `group.permutation`, because the pooled within-item variance is a function of the multiset of
    repeats inside each item and permuting them within a group does not change any multiset.
    """

    name = "GraderScoreSigma"
    version = "1.0"
    quantity = "grader.score_sigma"
    capabilities = Capability.NONE
    #: The reading scales with the grader's units, so a cross-grader comparison needs a frame. This
    #: is the same fact as the covariant relation below, said to gate 2 instead of to the generated
    #: test, and it is the field `require_frame_for_comparison` consults.
    gauge_status = GaugeStatus.COVARIANT
    requires = DISTRIBUTION_ACCESS
    substrates = ALL_SUBSTRATES
    phases = frozenset({Phase.PRE_RUN, Phase.POST_RUN})
    envelope = EnvelopeSpec(
        unconditional=True,
        justification=(
            "No regime condition is recorded for A6. This reads the grader's own output "
            "distribution under a design the caller controls, so there is no assumption about the "
            "run for a regime to violate. The scope limit is in the kill condition rather than in "
            "an envelope: on a deterministic scalar head the reading is zero and correct."
        ),
    )
    invariance = STOCHASTICITY_GROUPS
    #: The mapping form, because the two groups constrain this reading differently and one
    #: `Relation` cannot say both. `resolve_relation` reads a mapping per group; a single relation
    #: here would force one of the two checks to be dropped or mis-declared.
    #:
    #: The ignore is not a workaround for a mistake here. `core/invariance.py:594` implements the
    #: mapping form and its docstring argues for it, using `chi` as the motivating case, while
    #: `BaseObservable.invariance_relation` is annotated `Relation | None`, so the form the kernel
    #: supports is the form the type forbids. Three instruments have recorded a second true relation
    #: in a comment rather than declare it (`indices/chi.py`, `battery/lens.py`,
    #: `indices/teacher_compatibility.py`), and each of those is a generated test that does not run.
    #: The fix is one annotation in `measure/base.py`.
    invariance_relation = {  # type: ignore[assignment]
        # Var(a·r + b) = a²·Var(r), so the standard deviation scales by |a|: weight 1. The
        # implemented `reward.affine` generator draws a ~ LogUniform(0.1, 10), which is strictly
        # positive, so a**1 and |a|**1 coincide on every element the generated test can draw. On a
        # generator admitting a < 0 the assertion would need the absolute value and the weight would
        # not express it; that limitation belongs to the group, not to this declaration.
        "reward.affine": COVARIANT_LINEAR,
        "group.permutation": INVARIANT,
    }
    baselines = STOCHASTICITY_BASELINES
    #: Rung 1: the per-facet attribution runs whenever the caller labelled a facet. A design with no
    #: facet labels still reports rung 1 with empty shares, and the emptiness is the report that
    #: nothing was varied rather than a claim that no facet contributed.
    rung = 1
    faithful_to = "A6"
    deviations = (
        "A6 is one catalogue entry and this is one of two instruments discharging it. The spread "
        "and the flip rate transform differently under `reward.affine`, which one "
        "`invariance_group` field and one `Instrument.quantity` field cannot both express",
        "The catalogue records `group.permutation` for A6. That is true and it is not the "
        "group that constrains this reading's value, so `reward.affine` is declared beside it and "
        "the registry row for `grader.score_sigma` carries the affine group alone",
    )

    def __init__(self, data: RepeatedScores | None = None) -> None:
        self.data = data

    def compute(self) -> Any:
        data = self.data
        if data is None or data.scores.size == 0:
            return _no_data_refusal(self.name, "a spread needs repeats")
        counts = data.counts()
        if int(np.sum(counts >= 2)) == 0:
            return _no_replicates_refusal(self.name, data, counts, what="the within-item variance")

        rv = repeat_variance(data)
        finite = rv.per_item_sigma[np.isfinite(rv.per_item_sigma)]
        quantiles = (
            {
                "p50": float(np.quantile(finite, 0.5)),
                "p90": float(np.quantile(finite, 0.9)),
                "max": float(finite.max()),
            }
            if finite.size
            else {}
        )
        shares, null = facet_shares(data)
        return ScoreSpread(
            sigma=rv.sigma,
            variance=rv.variance,
            df=rv.df,
            deterministic=rv.deterministic,
            n_items=data.n_items,
            n_repeats=data.n_repeats,
            n_scores=rv.n_scores,
            n_items_used=rv.n_items_used,
            sigma_quantiles=quantiles,
            facet_shares=shares,
            facet_null=null,
            facet_effects=facet_effects(data),
            grader=data.grader,
            # Assuming determinism is assuming sigma is zero. The error that assumption induces is
            # the reading itself, which is why the baseline and the reading sit together.
            baselines={"baseline.assume_determinism": 0.0},
        )

    def payload(self, computed: ScoreSpread) -> dict[str, Any]:
        return {
            "sigma": computed.sigma,
            "variance": computed.variance,
            "df": computed.df,
            "deterministic": computed.deterministic,
            "n_items": computed.n_items,
            "n_repeats": computed.n_repeats,
            "n_scores": computed.n_scores,
            "n_items_used": computed.n_items_used,
            "sigma_quantiles": dict(computed.sigma_quantiles),
            "facet_shares": dict(computed.facet_shares),
            "facet_null": dict(computed.facet_null),
            "facet_effect": {n: fx.effect for n, fx in computed.facet_effects.items()},
            "facet_effect_se": {n: fx.se for n, fx in computed.facet_effects.items()},
            "facet_effect_share": {n: fx.share for n, fx in computed.facet_effects.items()},
            "grader": computed.grader,
            "says": computed.says,
            "baselines": dict(computed.baselines),
        }


# ---------------------------------------------------------------------------
# Rung 2: the verdict
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class VerdictStability:
    """`grader.flip_rate`: how often re-running the same comparison changes which side wins.

    Dimensionless and invariant under `reward.affine`, so unlike the spread it can be compared
    between two graders on two scales without a frame. That is the whole practical value of having
    split it out: it is the half of A6 that travels.
    """

    flip_rate: float
    tie_rate: float
    n_pairs: int
    repeats_needed: int
    achieved_agreement: float
    reference_repeats: int
    target: float
    paired_occasions: bool
    #: The median number of repeat combinations behind one pair's modal probability. This is the
    #: resolution of the estimate: over two draws a pair's modal probability can only be 0.5 or 1.0.
    combinations_per_pair: int
    #: Fraction of pairs that came back an exact coin flip, so no vote size resolves them.
    undetermined_pairs: float
    #: Fraction of all item pairs that carry only one repeat combination and are therefore unanimous
    #: by construction. Those pairs contribute a zero to the flip rate that is a property of the
    #: design and not of the grader, so a reading with this above zero is biased toward zero by
    #: exactly the weight it carries.
    unanimous_by_construction: float
    #: Items with exactly one usable draw. The source of the fraction above, in counts.
    singleton_items: int
    grader: str = ""
    baselines: Mapping[str, float] = field(default_factory=dict)

    @property
    def never_flipped(self) -> bool:
        """No pair ever disagreed with itself, over pairs that had a choice to make.

        The pairwise face of the kill condition: a deterministic grader has a flip rate of exactly
        zero, and reporting that is the instrument working. Distinguished from
        `unanimous_by_construction`, which is the same zero produced by a design with nothing in it.
        """
        return self.flip_rate == 0.0 and self.combinations_per_pair > 1

    @property
    def says(self) -> str:
        caveat = (
            f" {self.unanimous_by_construction:.1%} of pairs carry one repeat combination and are "
            f"unanimous whatever the grader does, so this rate is biased toward zero by that "
            f"weight; replicate the {self.singleton_items} single-draw items to remove it."
            if self.unanimous_by_construction > 0.0
            else ""
        )
        if self.never_flipped:
            return (
                f"This grader never changed its pairwise verdict over {self.n_pairs} pairs at "
                f"{self.combinations_per_pair} repeat combinations each. The verdict is "
                f"deterministic on this design, so a majority vote of one reproduces any "
                f"reference.{caveat}"
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
        return f"This grader flips its pairwise verdict {self.flip_rate:.1%} of the time. {vote}{caveat}"


class GraderFlipRate(ControlInstrument):
    """A6 rung 2. The number that actually reaches the optimiser.

    A grader can be noisy and still rank a pair the same way every time, and it is the ranking that
    becomes an advantage, so the spread does not determine this and neither determines the other.

    Invariant under `reward.affine`, and the argument is short enough to check: the verdict is the
    sign of a difference, `(a·r_i + b) - (a·r_j + b) = a·(r_i - r_j)`, so a positive `a` preserves
    every sign and a negative one reverses all of them at once, which leaves each pair's modal
    probability `max(p, 1 - p)` alone. Invariant under `group.permutation` too, and that one is
    earned rather than free: the flip rate is a U-statistic over all `m_i · m_j` repeat combinations
    precisely so that it does not depend on an ordering the data does not carry. The diagonal
    pairing would fail this check, which is why it is not the default.
    """

    name = "GraderFlipRate"
    version = "1.0"
    quantity = "grader.flip_rate"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    requires = DISTRIBUTION_ACCESS
    substrates = ALL_SUBSTRATES
    phases = frozenset({Phase.PRE_RUN, Phase.POST_RUN})
    envelope = EnvelopeSpec(
        unconditional=True,
        justification=(
            "No regime condition is recorded for A6. A pairwise verdict's stability is a property "
            "of the grader and of the design that varied it, and no regime of a training run makes "
            "a measured flip rate wrong. The scope limit is the kill condition: a deterministic "
            "grader never flips, the reading is zero, and that is the answer rather than a failure."
        ),
    )
    invariance = STOCHASTICITY_GROUPS
    #: The mapping form for the same reason as `GraderScoreSigma`, and with the opposite entry under
    #: `reward.affine`. Those two entries are the split, written down. Same ignore, same cause.
    invariance_relation = {  # type: ignore[assignment]
        "reward.affine": INVARIANT,
        "group.permutation": INVARIANT,
    }
    baselines = STOCHASTICITY_BASELINES
    rung = 2
    faithful_to = "A6"
    deviations = (
        "A6 is one catalogue entry and this is one of two instruments discharging it. The flip rate "
        "is invariant under `reward.affine` where the spread is covariant, which one "
        "`invariance_group` field and one `Instrument.quantity` field cannot both express",
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
            return _no_data_refusal(self.name, "a flip rate needs a verdict measured twice")
        counts = data.counts()
        if int(np.sum(counts >= 2)) == 0:
            return _no_replicates_refusal(self.name, data, counts, what="the flip rate")

        profile = flip_rates(data, max_pairs=self.max_pairs, seed=self.seed)
        needed, achieved = repeats_for_majority(
            profile, reference_repeats=self.reference_repeats, target=self.target
        )
        return VerdictStability(
            flip_rate=profile.flip_rate,
            tie_rate=profile.tie_rate,
            n_pairs=profile.n_pairs,
            repeats_needed=needed,
            achieved_agreement=achieved,
            reference_repeats=self.reference_repeats,
            target=self.target,
            paired_occasions=data.paired_occasions,
            combinations_per_pair=profile.combinations,
            undetermined_pairs=profile.undetermined,
            unanimous_by_construction=unanimous_pair_fraction(data),
            singleton_items=int(np.sum(counts == 1)),
            grader=data.grader,
            # Assuming determinism is assuming the verdict never moves, so the baseline's flip rate
            # is zero by construction and the reading is the error that assumption induces.
            baselines={"baseline.assume_determinism": 0.0},
        )

    def payload(self, computed: VerdictStability) -> dict[str, Any]:
        return {
            "flip_rate": computed.flip_rate,
            "tie_rate": computed.tie_rate,
            "n_pairs": computed.n_pairs,
            "repeats_needed": computed.repeats_needed,
            "achieved_agreement": computed.achieved_agreement,
            "reference_repeats": computed.reference_repeats,
            "target": computed.target,
            "paired_occasions": computed.paired_occasions,
            "combinations_per_pair": computed.combinations_per_pair,
            "undetermined_pairs": computed.undetermined_pairs,
            "unanimous_by_construction": computed.unanimous_by_construction,
            "singleton_items": computed.singleton_items,
            "never_flipped": computed.never_flipped,
            "grader": computed.grader,
            "says": computed.says,
            "baselines": dict(computed.baselines),
        }


def unanimous_pair_fraction(data: RepeatedScores) -> float:
    """Fraction of item pairs whose verdict has one repeat combination behind it.

    Such a pair agrees with itself whatever the grader does, so it contributes a guaranteed zero to
    the flip rate. Computed over every pair rather than over the subsample `flip_rates` may have
    drawn, because it is a property of the design; the subsample is uniform, so the two agree in
    expectation and the reading carries `n_pairs` beside this so a reader can see which is which.

    Counted from the per-item draw counts alone, without touching the estimator: a pair has one
    combination when `c_i · c_j == 1` for exchangeable repeats and when `min(c_i, c_j) == 1` for
    shared occasions, and both reduce to whether the singleton items are involved.
    """
    counts = data.counts()
    usable = counts[counts >= 1]
    n = usable.shape[0]
    if n < 2:
        return float("nan")
    singles = int(np.sum(usable == 1))
    total = n * (n - 1) // 2
    if data.paired_occasions:
        # min(c_i, c_j) == 1 whenever either side is a singleton.
        one_combination = total - (n - singles) * (n - singles - 1) // 2
    else:
        # c_i · c_j == 1 only when both sides are singletons.
        one_combination = singles * (singles - 1) // 2
    return float(one_combination / total)


# ---------------------------------------------------------------------------
# Both, from one design
# ---------------------------------------------------------------------------


def stochasticity_profile(
    data: RepeatedScores,
    *,
    reference_repeats: int = DEFAULT_REFERENCE_REPEATS,
    target: float = DEFAULT_AGREEMENT_TARGET,
    max_pairs: int | None = 20_000,
    seed: int = 0,
) -> tuple[ScoreSpread | Refusal, VerdictStability | Refusal]:
    """A6's whole profile: the spread and the verdict stability, from one repeated scoring.

    Splitting the declaration is not a reason to make anyone score twice. The expensive part of A6
    happened before a `RepeatedScores` existed, and both readings are cheap functions of the array
    it holds, so this runs both and hands back both. Either element may be a `Refusal`, and they can
    refuse independently: a design with two draws on ten items and one draw on a thousand gives a
    usable spread and a flip rate that is mostly zeros by construction.
    """
    spread = GraderScoreSigma(data).compute()
    verdict = GraderFlipRate(
        data,
        reference_repeats=reference_repeats,
        target=target,
        max_pairs=max_pairs,
        seed=seed,
    ).compute()
    return spread, verdict


# ---------------------------------------------------------------------------
# The two shared refusals
# ---------------------------------------------------------------------------


def _no_data_refusal(instrument: str, needs: str) -> Refusal:
    return Refusal(
        instrument=instrument,
        reason=RefusalReason.ACCESS_INSUFFICIENT,
        detail=f"no repeated scores were supplied, and {needs}",
        remedy=(
            "score the same inputs at least twice under a facet you control and pass the result as "
            "RepeatedScores(scores=array_of_shape_items_by_repeats). Presentation order is the "
            "cheapest facet to vary and needs no seed control."
        ),
        statistics={"n_items": 0, "n_repeats": 0},
    )


def _no_replicates_refusal(
    instrument: str, data: RepeatedScores, counts: np.ndarray, *, what: str
) -> Refusal:
    """One draw per item is GRADER:RECORD, and both readings are undefined there.

    Worth spelling out for the flip rate, because its failure mode is the quieter one. A variance
    over one observation is visibly undefined and comes back NaN. A flip rate over one draw per item
    comes back exactly 0.0, because every pair has a single comparison behind it and a single
    comparison always agrees with itself. That is a number, it is in range, and it is wrong, which
    is precisely the case a refusal exists for.
    """
    top = int(counts.max()) if counts.size else 0
    return Refusal(
        instrument=instrument,
        reason=RefusalReason.ACCESS_INSUFFICIENT,
        detail=(
            f"every one of {data.n_items} items has at most one usable draw (max {top}), so {what} "
            f"is undefined rather than zero. With one draw per item every pair is unanimous by "
            f"construction and the flip-rate estimator returns exactly 0.0, which is a confident "
            f"wrong zero rather than a measurement. One draw per item is GRADER:RECORD; both of "
            f"A6's quantities need GRADER:REPLICATE"
        ),
        remedy=(
            "re-score the same items at least once more. If the grader is behind an API that will "
            "not vary its seed, vary the presentation order instead: that is a facet, it needs no "
            "cooperation from the vendor, and it is the facet with the largest published effect."
        ),
        statistics={
            "n_items": data.n_items,
            "n_repeats": data.n_repeats,
            "items_with_two_or_more": 0,
            "max_draws_per_item": top,
        },
    )


#: The split, as data. A card, a test or a reviewer can read the two relations off one mapping
#: without importing either class, and the two entries differing is the whole content of the split.
AFFINE_RELATIONS: Mapping[str, Relation] = {
    "grader.score_sigma": COVARIANT_LINEAR,
    "grader.flip_rate": INVARIANT,
}


__all__ = [
    "AFFINE_RELATIONS",
    "STOCHASTICITY_BASELINES",
    "STOCHASTICITY_GROUPS",
    "GraderFlipRate",
    "GraderScoreSigma",
    "ScoreSpread",
    "VerdictStability",
    "facet_clause",
    "stochasticity_profile",
    "unanimous_pair_fraction",
]

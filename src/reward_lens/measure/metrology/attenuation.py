"""A3, the attenuation factor: how much grader error shrinks the selection signal.

Spearman published the correction in 1904 and nobody has pointed it at a reward model. The
arithmetic is three lines. What took the hundred and twenty years is having a variance
decomposition of the grader to put into it, which is what A2 now supplies.

**What the factor is a factor of, stated before anything else, because the square root is where
this gets misread.** Under classical additive error the selection differential ``S = Cov(f, r)`` is
*not* attenuated at all: ``Cov(f, r_true + e) = Cov(f, r_true)`` whenever the error is independent
of the feature. So an instrument that reported "your selection signal is attenuated" from the raw
covariance would be reporting nothing. The attenuation enters through the **denominator**, because
every advantage estimator in use divides by an observed spread. GRPO's advantage is
``(r - mean) / std``, and that ``std`` is ``sqrt(sigma2_true + sigma2_err)`` rather than
``sqrt(sigma2_true)``. So

    Cov(f, (r - mean)/std_obs) = Cov(f, r_true) / std_obs
                              = sqrt(sigma2_true / (sigma2_true + sigma2_err)) * [the true one]

and the printed square root is exactly right for the standardised gradient, which is the one the
optimiser actually follows. That derivation is the reason this instrument is scoped to standardised
advantages and says so in `RewardVariance.standardised`. A run whose estimator does not divide by
the group spread has a different attenuation and this instrument declines to pretend otherwise.

**The ladder is one estimator with a term switched off.** Rung 0 sees error in the reward and
assumes the features are measured exactly. Rung 1 adds error in the features, through the
errors-in-variables correction ``beta_corr = (C_obs - C_err)^-1 S``. Setting ``C_err = 0`` in rung 1
returns rung 0 exactly, which is what makes them two rungs of one quantity rather than two
quantities sharing a name, and it fixes rung 0's bias direction as upward: it reports less
attenuation than there is, because it cannot see feature error at all.

**Where ``C_err`` comes from, and why this is one project with series C rather than two.** The
errors-in-variables literature treats ``C_err`` as the hard part, usually unidentified without a
validation sample. Here it has an obvious estimator: the pooled within-prompt covariance of the
features across the K rollouts of a group. Same prompt, same policy, so anything that varies is
rollout noise by construction. `within_prompt_covariance` computes it in one pass over data that a
GRPO record already contains. A2's decomposition supplies the reward-side term and the rollout
structure supplies the feature-side term, so the selection-gradient instrument in series C gets its
correction from the metrology series for free.

**Not the noise-correction papers.** 2510.00915, 2510.18924 and 1810.01032 all correct a *bias*
induced by a noisy binary channel, and all three model the noise as one or two scalar flip rates on
a channel assumed independent of the signal. That construction cannot represent an error that is
correlated across features, because a scalar rate has no off-diagonal. This instrument's error term
is a covariance matrix estimated from replication, so a grader whose noise hits "cites a source" and
"is verbose" together is representable here and is not representable there. The two are solving
different problems and the distinction is worth keeping: they debias a label, this one un-shrinks a
gradient.

Kill condition, from the catalogue: *if the correction never changes a feature ranking, report it
once and stop.* Which is why `Attenuation.rank_changed` is a field rather than something a reader
has to derive. One thing to know before reading it: **rung 0 cannot change a ranking**, because a
scalar factor applied to every feature preserves order. Only the rung 1 matrix inverse can reorder.
So the kill condition is a statement about rung 1 alone, and a rung 0 reading that reports
``rank_changed=False`` has not tested it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
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
    Substrate,
)
from reward_lens.measure.controls._base import ControlInstrument

#: A grader is a measurement device whatever it is made of, so every substrate admits this.
ALL_SUBSTRATES = frozenset(
    {
        Substrate.NEURAL_SCALAR,
        Substrate.NEURAL_GEN,
        Substrate.PROGRAM,
        Substrate.PROCEDURAL,
        Substrate.HUMAN,
        Substrate.COMPOSITE,
    }
)

#: Catalogue A3: `Access GRADER:REPLICATE`. REPLICATE rather than QUERY because the components come
#: from facet-controlled repeats, and a hosted judge you can call but cannot vary is exactly the
#: case `Access.REPLICATE` was separated out to name.
ATTENUATION_ACCESS: dict[Component, Access] = {Component.GRADER: Access.REPLICATE}

#: Catalogue A3 names one baseline: the uncorrected beta. It is reported beside the corrected one on
#: every reading, because the whole claim of this instrument is the difference between them.
ATTENUATION_BASELINES: tuple[BaselineID, ...] = ("baseline.uncorrected_beta",)


def attenuation_envelope(measured_by: str | None = None) -> EnvelopeSpec:
    """A3's envelope: `GROUP_NONDEGENERATE`, from the catalogue, and it is load-bearing here.

    The condition is that K > 1 and the within-group spread is non-zero for a stated fraction of
    groups. On an all-fail or all-pass group the standardised advantage is zero over zero, and the
    attenuation factor is a ratio of two variances that are both estimates of nothing. This is not
    a case where a wrong number is merely imprecise: the ratio of two small noisy numbers is
    unbounded, so a degenerate slice can report an attenuation of 0.02 with the same confidence as
    a real one.

    The measuring quantity defaults to the kernel's own `MEASURED_BY` rather than to a local
    spelling. This module named `grader.group_spread`, which is registered nowhere, so the envelope
    declared a precondition and no way to check it while reading as rigour. `EnvelopeSpec` now
    rejects that at construction and caught this on its first run. One condition, one measuring
    quantity, one place it is written down.
    """
    from reward_lens.measure.rate.regime import MEASURED_BY

    qid = measured_by or MEASURED_BY[RegimeCondition.GROUP_NONDEGENERATE]
    return EnvelopeSpec(
        requires=frozenset({RegimeCondition.GROUP_NONDEGENERATE}),
        measured_by={RegimeCondition.GROUP_NONDEGENERATE: qid},
        on_violation="refuse",
    )


# ---------------------------------------------------------------------------
# What A3 needs from A2
# ---------------------------------------------------------------------------


def _component_facets(name: str) -> frozenset[str]:
    """The facet labels a variance-component name carries, per `stats/gtheory.py`'s convention.

    Names are single-character facet labels with an optional ``,e`` suffix marking a term
    confounded with residual: ``p``, ``r``, ``pr``, ``pro,e``. Anything that does not parse under
    that convention returns an empty set, which keeps it out of the relative error rather than
    silently inflating it.
    """
    head = name.split(",", 1)[0].strip()
    if not head:
        return frozenset()
    # Two naming schemes are in use. `gtheory` writes single-character labels concatenated, so
    # `pr` is the p-by-r interaction. Callers naming facets in words write them separated, as
    # `item x rater`. Treat the head as characters only when it is a single alphabetic run and no
    # separator is present; otherwise split on the separators and keep whole words.
    tokens = [t for t in re.split(r"[^0-9A-Za-z_]+", head) if t]
    if len(tokens) == 1 and tokens[0].isalpha() and len(tokens[0]) <= 4:
        return frozenset(tokens[0]) | {tokens[0]}
    return frozenset(tokens)


@dataclass(frozen=True)
class RewardVariance:
    """The two numbers A3 takes from A2's decomposition, and what they have to mean.

    A2 is the variance-components instrument and it produces a full facet decomposition: item,
    rater, occasion, their interactions, and a residual. A3 needs that collapsed to two numbers and
    the collapse is a decision rather than an arithmetic step, so it is made explicitly here rather
    than assumed.

    ``sigma2_true`` is the universe-score variance: the part of ``Var(r)`` attributable to the thing
    being measured. In a crossed item x rater x occasion design that is ``sigma2(item)`` and nothing
    else. ``sigma2_err`` is the relative error variance ``sigma2(delta)``: every facet and
    interaction that moves the score without the item changing. **Relative rather than absolute**,
    because the standardised advantage is a within-group contrast and a main effect that shifts
    every member of a group by the same amount cancels out of it. Using ``sigma2(Delta)`` here would
    charge the grader for a constant it never gets to express.

    ``standardised`` is the scope limit made checkable. The square root in the rung 0 factor is
    correct for an advantage that divides by the observed group spread. Construct with
    ``standardised=False`` and this instrument refuses rather than reporting a number derived for a
    different estimator.
    """

    sigma2_true: float
    sigma2_err: float
    #: Whether the advantage estimator this correction is for divides by the observed group spread.
    standardised: bool = True
    n_items: int = 0
    n_replications: int = 0
    #: Where these came from, for the provenance line on the reading. "A2 rung 1, crossed design".
    source: str = ""
    #: The facet breakdown, carried through unread so a reader can audit the collapse above.
    facets: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma2_true) or not math.isfinite(self.sigma2_err):
            raise ValueError(
                f"variance components must be finite; got sigma2_true={self.sigma2_true!r}, "
                f"sigma2_err={self.sigma2_err!r}"
            )
        if self.sigma2_err < 0.0:
            # A within-cell mean square cannot be negative under any standard estimator, so this is
            # a caller bug rather than an anticipated condition and gets an exception. A negative
            # `sigma2_true` is a different matter: ANOVA-style component estimates go negative
            # routinely when the true component is near zero, and that case is handled as a reading.
            raise ValueError(
                f"sigma2_err is {self.sigma2_err}, and an error variance estimated as a within-cell "
                f"mean square cannot be negative. A negative sigma2_true is expected and handled; a "
                f"negative sigma2_err means the components were assembled wrongly."
            )

    @property
    def total(self) -> float:
        return self.sigma2_true + self.sigma2_err

    @property
    def reliability(self) -> float:
        """``sigma2_true / (sigma2_true + sigma2_err)``. NaN when the total is zero.

        NaN rather than a substituted value, because a grader with no variance at all has no
        reliability and the callers here all check for it before dividing.
        """
        t = self.total
        return float("nan") if t <= 0.0 else self.sigma2_true / t

    @classmethod
    def from_facets(
        cls,
        facets: Mapping[str, float],
        *,
        universe: str | Iterable[str] = "item",
        standardised: bool = True,
        n_items: int = 0,
        n_replications: int = 0,
        source: str = "",
        kind: str = "relative",
    ) -> "RewardVariance":
        """Collapse a facet decomposition into the two numbers, naming which facets are the object.

        ``kind`` picks which G-theory error variance is meant, and the two are different numbers.

        ``relative``, the default, is Brennan's ``sigma2(delta)``: only the components that
        **interact with the object of measurement**. This is the one A3 needs. A facet main effect
        shifts every rollout in a group by the same amount, so it cancels out of a group-centred
        contrast and charging the grader for it overstates the attenuation. The module docstring
        argues for exactly this.

        ``absolute`` is ``sigma2(Delta)``: every component except the object's own. That is what
        this method used to compute unconditionally, which is verbatim Brennan's definition of the
        *absolute* error variance while the surrounding argument was for the relative one. On a
        p-by-r design with `sigma2(p) = 1.0`, a main effect `sigma2(r) = 0.6` and an interaction
        `sigma2(pr,e) = 0.4`, the two give attenuation factors of 0.845 and 0.707. The direction is
        the one the old comment here warned about in the abstract and then took: the correction was
        overstated whenever a facet carried a main effect.

        Membership is read off the component name, which follows the convention `stats/gtheory.py`
        writes: single-character facet labels, optionally suffixed with ``,e`` for a term confounded
        with residual, so ``pr,e`` interacts with ``p`` and ``r`` does not. A name that does not
        parse under that convention is treated as not interacting, which keeps it out of the
        relative error rather than silently inflating it.

        Everything outside ``universe`` still counts as error under ``absolute``, and that direction
        remains deliberate: a facet somebody forgot to classify lands in the error term rather than
        silently becoming signal.
        """
        names = {universe} if isinstance(universe, str) else set(universe)
        missing = names - set(facets)
        if missing:
            raise ValueError(
                f"universe facet(s) {sorted(missing)} are not in the decomposition, which has "
                f"{sorted(facets)}. Name the facet as A2 spells it."
            )
        if kind not in ("relative", "absolute"):
            raise ValueError(f"kind must be 'relative' or 'absolute'; got {kind!r}")
        true = sum(float(facets[n]) for n in names)
        if kind == "absolute":
            err = sum(float(v) for k, v in facets.items() if k not in names)
        else:
            outside = {k: v for k, v in facets.items() if k not in names}
            interacting = {k: v for k, v in outside.items() if names <= _component_facets(k)}
            if outside and not interacting:
                raise ValueError(
                    f"no component of {sorted(outside)} was recognised as interacting with the "
                    f"object of measurement {sorted(names)}, so the relative error variance would "
                    f"come out as exactly zero and the attenuation factor as exactly 1. That is a "
                    f"naming-convention mismatch rather than a perfect grader. Name components the "
                    f"way `stats/gtheory.py` does, or pass kind='absolute' if you really want "
                    f"Brennan's sigma2(Delta)."
                )
            err = sum(float(v) for v in interacting.values())
        return cls(
            sigma2_true=true,
            sigma2_err=err,
            standardised=standardised,
            n_items=n_items,
            n_replications=n_replications,
            source=source,
            facets=dict(facets),
        )

    @classmethod
    def from_replicates(
        cls,
        scores: np.ndarray | Sequence[Sequence[float]],
        *,
        standardised: bool = True,
        source: str = "one-way ANOVA over replicates",
    ) -> "RewardVariance":
        """The rung 0 components from a rectangular (item, replicate) score array.

        The one-way random-effects decomposition, which is the cheapest thing that produces the two
        numbers: ``sigma2_err`` is the within-item mean square and ``sigma2_true`` is
        ``(MS_between - MS_within) / m``. This exists so A3 is testable and demonstrable without
        standing up A2's full crossed design, and it is honestly the *lowest* rung: it charges every
        facet except the item to error without being able to say which facet.
        """
        arr = np.asarray(scores, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"expected a 2-d (item, replicate) array; got shape {arr.shape}")
        n, m = arr.shape
        if n < 2 or m < 2:
            raise ValueError(
                f"a one-way decomposition needs at least 2 items and 2 replicates; got {n} x {m}"
            )
        item_means = arr.mean(axis=1)
        grand = float(arr.mean())
        ms_between = float(m * np.sum((item_means - grand) ** 2) / (n - 1))
        ms_within = float(np.sum((arr - item_means[:, None]) ** 2) / (n * (m - 1)))
        return cls(
            sigma2_true=(ms_between - ms_within) / m,
            sigma2_err=ms_within,
            standardised=standardised,
            n_items=n,
            n_replications=m,
            source=source,
            facets={"item": (ms_between - ms_within) / m, "residual": ms_within},
        )


def within_prompt_covariance(
    features: np.ndarray | Sequence[Sequence[float]],
    group_ids: np.ndarray | Sequence[Any],
) -> np.ndarray:
    """``C_err``: the pooled within-prompt covariance of the features across rollouts.

    This is the estimator that makes rung 1 cheap. Every rollout in a group came from the same
    prompt and the same policy, so the spread of a feature *within* a group is rollout noise with
    nothing else in it. Pooling across groups with the usual ``N - G`` denominator gives an unbiased
    estimate of the within-group covariance, off-diagonals included, which is the part a scalar
    noise rate cannot represent.

    Groups of size 1 contribute nothing and are dropped from both the sum and the denominator. A
    call where every group has size 1 raises, because the returned zero matrix would be a claim that
    the features are measured exactly rather than a report that nothing was measured.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"expected a 2-d (rollout, feature) array; got shape {x.shape}")
    gids = np.asarray(group_ids)
    if gids.shape[0] != x.shape[0]:
        raise ValueError(
            f"features has {x.shape[0]} rows and group_ids has {gids.shape[0]}; they index the same "
            f"rollouts and must agree"
        )
    p = x.shape[1]
    acc = np.zeros((p, p), dtype=np.float64)
    df = 0
    for g in np.unique(gids):
        block = x[gids == g]
        k = block.shape[0]
        if k < 2:
            continue
        centred = block - block.mean(axis=0)
        acc += centred.T @ centred
        df += k - 1
    if df == 0:
        raise ValueError(
            "every group has one member, so there is no within-prompt spread to estimate C_err "
            "from. Returning zeros would assert the features are measured exactly. Supply a record "
            "with K > 1 rollouts per prompt."
        )
    return acc / df


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class Attenuation:
    """The factor, both rungs' pieces of it, and whether correcting it reordered anything.

    ``factor`` is the quantity: the scalar ``a`` with ``beta_measured = a * beta_true``. Read it
    with ``reward_factor`` and ``feature_factor`` beside it, because they are the two independent
    ways a gradient gets shrunk and a reading that reports only their product cannot say which one
    to fix. A noisy grader is fixed with more replications; noisy features are fixed with a better
    featuriser or more rollouts per prompt, and those are different budgets.
    """

    factor: float
    reward_factor: float
    feature_factor: float
    reliability: float
    rung: int
    n_features: int
    #: Set at rung 1. The observed and corrected selection gradients, in the caller's feature order.
    beta_observed: np.ndarray | None = None
    beta_corrected: np.ndarray | None = None
    #: Feature indices ordered by descending |beta|, before and after the correction.
    rank_observed: tuple[int, ...] = ()
    rank_corrected: tuple[int, ...] = ()
    #: The kill condition, as a field. False at rung 0 always, and that is not evidence.
    rank_changed: bool = False
    #: The largest number of places any single feature moved. 0 when nothing moved.
    max_rank_move: int = 0
    #: Condition number of ``C_obs - C_err``. The correction inverts it, so a large value here is
    #: the reading's own warning that the corrected gradient is amplifying noise.
    conditioning: float = float("nan")
    n_items: int = 0
    n_replications: int = 0
    source: str = ""
    baselines: Mapping[str, float] = field(default_factory=dict)

    @property
    def says(self) -> str:
        """The sentence, assembled from what was actually measured."""
        rel = (
            f"reliability {self.reliability:.3g}, "
            if math.isfinite(self.reliability)
            else "feature error only, "
        )
        head = (
            f"Your measured selection gradient is {self.factor:.3g} of the true one "
            f"({rel}rung {self.rung})."
        )
        if self.rung == 0:
            return head + (
                " Rung 0 rescales every feature by one factor, so it cannot reorder them; the "
                "ranking question needs rung 1."
            )
        if self.rank_changed:
            moved = [
                (i, self.rank_observed.index(i), self.rank_corrected.index(i))
                for i in self.rank_observed
                if self.rank_observed.index(i) != self.rank_corrected.index(i)
            ]
            i, before, after = max(moved, key=lambda t: abs(t[1] - t[2]))
            return head + (
                f" Correcting for it moves feature {i} from rank {before + 1} to rank {after + 1}."
            )
        return head + " Correcting for it leaves the feature ranking unchanged."


def spearman_factor(components: RewardVariance) -> float:
    """Rung 0: ``sqrt(sigma2_true / (sigma2_true + sigma2_err))``. Spearman 1904.

    Returns NaN when the total variance is zero or the true component is negative, so callers can
    tell those apart from a small factor. The instrument turns both into refusals; a bare function
    returning NaN is the right shape for a bare function.
    """
    rel = components.reliability
    if not math.isfinite(rel) or rel < 0.0:
        return float("nan")
    return math.sqrt(rel)


def eiv_gradient(
    c_obs: np.ndarray,
    c_err: np.ndarray,
    s: np.ndarray,
    *,
    rcond: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Rung 1: ``beta_obs = C_obs^-1 S`` and ``beta_corr = (C_obs - C_err)^-1 S``.

    Returns both gradients and the condition number of the corrected matrix. Solved with
    `numpy.linalg.lstsq` rather than an explicit inverse, because ``C_obs - C_err`` goes singular
    exactly when the correction matters most: a feature whose observed variance is nearly all
    rollout noise leaves almost nothing behind, and an explicit inverse turns that into a large
    number rather than into a visible conditioning problem.
    """
    c_obs = np.asarray(c_obs, dtype=np.float64)
    c_err = np.asarray(c_err, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64).ravel()
    p = s.shape[0]
    for name, m in (("C_obs", c_obs), ("C_err", c_err)):
        if m.shape != (p, p):
            raise ValueError(
                f"{name} has shape {m.shape}; S has {p} features so it must be {(p, p)}"
            )
    corrected = c_obs - c_err
    beta_obs = np.linalg.lstsq(c_obs, s, rcond=rcond)[0]
    beta_corr = np.linalg.lstsq(corrected, s, rcond=rcond)[0]
    cond = float(np.linalg.cond(corrected))
    return beta_obs, beta_corr, cond


def _projection_factor(beta_obs: np.ndarray, beta_corr: np.ndarray) -> float:
    """The scalar ``a`` minimising ``||beta_obs - a * beta_corr||``.

    With one feature this is the plain ratio, so rung 1 reduces to rung 0 exactly when ``C_err`` is
    zero. With several it is the least-squares scalar, which is the only reading of "your gradient
    is 0.71 of the true one" that survives the gradient also having *rotated*: the residual of that
    projection is the part of the correction a single number cannot express, and it is why
    ``rank_changed`` is reported separately rather than inferred from the factor.
    """
    denom = float(beta_corr @ beta_corr)
    if denom <= 0.0:
        return float("nan")
    return float(beta_obs @ beta_corr) / denom


def _ranks(beta: np.ndarray) -> tuple[int, ...]:
    """Feature indices by descending |beta|, ties broken by index so the order is deterministic."""
    order = sorted(range(beta.shape[0]), key=lambda i: (-abs(float(beta[i])), i))
    return tuple(order)


def _degenerate_refusal(instrument: str, components: RewardVariance) -> Refusal:
    """The reading when the grader carries no reproducible signal on this slice.

    ``BELOW_LOD`` because that is precisely what has been established: the item-to-item spread the
    grader is supposed to resolve is at or below the grader's disagreement with itself. Section
    4.7's limit of detection is the same statement with the blank's standard deviation in place of
    the error component, and reusing the reason here rather than adding a sixteenth is a judgement,
    recorded in this package's report.

    The refusal carries the numbers, which is what makes it useful: an attenuation factor of zero
    and a refusal look the same to a plotting script and completely different to a person deciding
    whether to buy more replications.
    """
    return Refusal(
        instrument=instrument,
        reason=RefusalReason.BELOW_LOD,
        detail=(
            f"the universe-score variance is estimated at {components.sigma2_true:.6g} against an "
            f"error variance of {components.sigma2_err:.6g}, so the grader does not separate items "
            f"by more than it disagrees with itself. The attenuation factor is a ratio whose "
            f"numerator is not distinguishable from zero, and dividing a gradient by it would "
            f"return an arbitrarily large correction"
        ),
        remedy=(
            "raise the number of grader replications per item and re-run A2, or accept that this "
            "grader carries no reproducible item-level signal on this slice and select on something "
            "else. A2's decision study (A5) prices the first option."
        ),
        statistics={
            "sigma2_true": components.sigma2_true,
            "sigma2_err": components.sigma2_err,
            "n_items": components.n_items,
            "n_replications": components.n_replications,
        },
    )


def _unstandardised_refusal(instrument: str) -> Refusal:
    """Refuse when the advantage estimator does not divide by the observed spread.

    The square root is derived from that division. On an unstandardised advantage the selection
    differential is unattenuated under independent error and the honest factor is 1.0, which is a
    different claim about a different estimator, so reporting this instrument's number there would
    be a confident wrong answer of exactly the kind a refusal exists to prevent.
    """
    return Refusal(
        instrument=instrument,
        reason=RefusalReason.ENVELOPE_VIOLATED,
        detail=(
            "the components were declared for an advantage that does not divide by the observed "
            "group spread. The square root in this factor comes from that division: with no "
            "standardisation, Cov(f, r_true + e) = Cov(f, r_true) and the selection differential is "
            "not attenuated by independent grader error at all"
        ),
        remedy=(
            "if your estimator does standardise, construct RewardVariance with standardised=True. "
            "If it genuinely does not, the reward-side attenuation is 1.0 and the only correction "
            "you need is the rung 1 feature-side one, which `ErrorsInVariablesAttenuation` computes "
            "with `reward_components=None`."
        ),
        statistics={"standardised": False},
    )


# ---------------------------------------------------------------------------
# The two rungs
# ---------------------------------------------------------------------------


class _AttenuationBase(ControlInstrument):
    """Declarations both rungs share, so a difference between them is visible as a difference."""

    quantity = "grader.attenuation"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    requires = ATTENUATION_ACCESS
    substrates = ALL_SUBSTRATES
    phases = frozenset({Phase.PRE_RUN, Phase.POST_RUN})
    envelope = attenuation_envelope()
    invariance = "reward.affine"
    #: Invariant, weight 0. Under ``r -> a*r + b`` both variance components scale by ``a**2`` and
    #: the ratio is unchanged; at rung 1 both gradients scale by ``a`` through S and their ratio is
    #: unchanged. A factor that moved under a rescaling of the reward would be reading a level.
    invariance_relation = INVARIANT
    baselines = ATTENUATION_BASELINES
    faithful_to = "A3"


class AttenuationFactor(_AttenuationBase):
    """Rung 0. Spearman's correction on A2's reward-side components.

    Reports how much of the standardised selection gradient survives grader error, and nothing
    about feature error. That omission is the rung's bias and it is declared: the factor comes back
    **too high**, so the correction comes back too small, so anyone acting on rung 0 alone
    under-corrects. The direction matters because the alternative failure would be silently
    over-correcting a gradient, which manufactures signal.
    """

    name = "AttenuationFactor"
    version = "1.0"
    rung = 0
    deviations = (
        "the catalogue prints the estimator as a function of sigma2_true and sigma2_err without "
        "saying which advantage estimator it corrects. The square root holds for a standardised "
        "advantage and this instrument requires the components to declare that, rather than "
        "applying it to whatever it is handed",
    )

    def __init__(self, components: RewardVariance | None = None) -> None:
        self.components = components

    def compute(self) -> Any:
        if self.components is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    "no variance components were supplied, and this factor is a function of them "
                    "and of nothing else"
                ),
                remedy=(
                    "run A2 on the same slice and pass its output as "
                    "RewardVariance(sigma2_true=..., sigma2_err=...), or build the two numbers from "
                    "a rectangular (item, replicate) score array with "
                    "RewardVariance.from_replicates. Both need GRADER:REPLICATE."
                ),
                statistics={"components": None},
            )
        if not self.components.standardised:
            return _unstandardised_refusal(self.name)
        factor = spearman_factor(self.components)
        if not math.isfinite(factor) or factor <= 0.0:
            return _degenerate_refusal(self.name, self.components)
        return Attenuation(
            factor=factor,
            reward_factor=factor,
            feature_factor=1.0,
            reliability=self.components.reliability,
            rung=0,
            n_features=0,
            conditioning=float("nan"),
            n_items=self.components.n_items,
            n_replications=self.components.n_replications,
            source=self.components.source,
            baselines={"baseline.uncorrected_beta": 1.0},
        )

    def payload(self, computed: Attenuation) -> dict[str, Any]:
        return _payload(computed)


class ErrorsInVariablesAttenuation(_AttenuationBase):
    """Rung 1. ``beta_corr = (C_obs - C_err)^-1 S``, with the reward-side factor folded in.

    Two corrections, composed, and they are genuinely different operations. The reward-side one is a
    scalar and rescales the whole gradient. The feature-side one is a matrix inverse and can rotate
    it, which is the only way a correction reorders features. So the ranking question and the
    magnitude question are answered by different halves of this rung and the reading keeps them
    apart.

    ``reward_components=None`` runs the feature-side correction alone, which is the honest thing to
    do when the advantage is unstandardised: the reward-side factor is 1.0 there and this rung still
    has something to say.
    """

    name = "ErrorsInVariablesAttenuation"
    version = "1.0"
    rung = 1
    deviations = (
        "the catalogue prints `beta_corr = (C_obs - C_err)^-1 S` with no reward-side term. That "
        "expression corrects feature error only, and composing it with the rung 0 factor is what "
        "makes the two rungs estimate one quantity rather than two. The composition is stated on "
        "the reading as `reward_factor` and `feature_factor` so it can be undone by a reader who "
        "wants the printed expression alone",
    )

    def __init__(
        self,
        c_obs: np.ndarray | None = None,
        c_err: np.ndarray | None = None,
        s: np.ndarray | None = None,
        *,
        reward_components: RewardVariance | None = None,
    ) -> None:
        self.c_obs = c_obs
        self.c_err = c_err
        self.s = s
        self.reward_components = reward_components

    def compute(self) -> Any:
        c_obs, c_err, s = self.c_obs, self.c_err, self.s
        missing = [n for n, v in (("C_obs", c_obs), ("C_err", c_err), ("S", s)) if v is None]
        if missing:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=f"missing {', '.join(missing)}, and the correction is a function of all three",
                remedy=(
                    "pass C_obs as the observed feature covariance, S as the selection differential "
                    "Cov(f, advantage), and C_err from `within_prompt_covariance(features, "
                    "group_ids)` on a record with K > 1 rollouts per prompt."
                ),
                statistics={"missing": missing},
            )
        reward_factor = 1.0
        reliability = float("nan")
        components = self.reward_components
        if components is not None:
            if not components.standardised:
                return _unstandardised_refusal(self.name)
            reward_factor = spearman_factor(components)
            reliability = components.reliability
            if not math.isfinite(reward_factor) or reward_factor <= 0.0:
                return _degenerate_refusal(self.name, components)

        # Past the guard above all three are present, so this is a widening rather than a cast.
        beta_obs, beta_corr_features, cond = eiv_gradient(
            np.asarray(c_obs, dtype=np.float64),
            np.asarray(c_err, dtype=np.float64),
            np.asarray(s, dtype=np.float64),
        )
        # The reward-side factor divides, because beta_obs = reward_factor * beta_true.
        beta_corr = beta_corr_features / reward_factor
        feature_factor = _projection_factor(beta_obs, beta_corr_features)
        factor = _projection_factor(beta_obs, beta_corr)
        if not math.isfinite(factor):
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.BELOW_LOD,
                detail=(
                    f"the corrected gradient is zero to machine precision, so there is no factor "
                    f"relating the measured gradient to it. The corrected covariance has condition "
                    f"number {cond:.4g}"
                ),
                remedy=(
                    "check that C_err is not larger than C_obs on some feature, which means the "
                    "within-prompt spread exceeds the total spread and the two were estimated on "
                    "different slices. If they were estimated on the same slice, drop the features "
                    "whose observed variance is within noise of their within-prompt variance: they "
                    "carry no between-prompt signal to correct."
                ),
                statistics={"conditioning": cond, "reward_factor": reward_factor},
            )

        rank_obs = _ranks(beta_obs)
        rank_corr = _ranks(beta_corr)
        moves = [abs(rank_obs.index(i) - rank_corr.index(i)) for i in rank_obs]
        return Attenuation(
            factor=factor,
            reward_factor=reward_factor,
            feature_factor=feature_factor,
            reliability=reliability,
            rung=1,
            n_features=int(beta_obs.shape[0]),
            beta_observed=beta_obs,
            beta_corrected=beta_corr,
            rank_observed=rank_obs,
            rank_corrected=rank_corr,
            rank_changed=rank_obs != rank_corr,
            max_rank_move=max(moves) if moves else 0,
            conditioning=cond,
            n_items=0 if components is None else components.n_items,
            n_replications=0 if components is None else components.n_replications,
            source="" if components is None else components.source,
            baselines={"baseline.uncorrected_beta": float(np.max(np.abs(beta_obs)))},
        )

    def payload(self, computed: Attenuation) -> dict[str, Any]:
        return _payload(computed)


def _payload(computed: Attenuation) -> dict[str, Any]:
    """The flat Evidence value both rungs emit, including the `baselines` key the lint looks for."""
    return {
        "factor": computed.factor,
        "reward_factor": computed.reward_factor,
        "feature_factor": computed.feature_factor,
        "reliability": computed.reliability,
        "rung": computed.rung,
        "n_features": computed.n_features,
        "rank_observed": list(computed.rank_observed),
        "rank_corrected": list(computed.rank_corrected),
        "rank_changed": computed.rank_changed,
        "max_rank_move": computed.max_rank_move,
        "conditioning": computed.conditioning,
        "n_items": computed.n_items,
        "n_replications": computed.n_replications,
        "says": computed.says,
        "baselines": dict(computed.baselines),
    }


def factor_from_scores(scores: np.ndarray | Sequence[Sequence[float]]) -> float:
    """Rung 0 end to end from a rectangular (item, replicate) array. The invariance test's subject.

    The generated test needs one callable that goes from scores to the scalar the relation is
    declared about, because a relation asserted about a hand-built pair of variance components would
    be asserting something about arithmetic rather than about the instrument. This recomputes the
    components from the transformed scores every time, so an affine rescaling has to travel the
    whole path.
    """
    return spearman_factor(RewardVariance.from_replicates(scores))


__all__ = [
    "ALL_SUBSTRATES",
    "ATTENUATION_ACCESS",
    "ATTENUATION_BASELINES",
    "Attenuation",
    "AttenuationFactor",
    "ErrorsInVariablesAttenuation",
    "RewardVariance",
    "attenuation_envelope",
    "eiv_gradient",
    "factor_from_scores",
    "spearman_factor",
    "within_prompt_covariance",
]

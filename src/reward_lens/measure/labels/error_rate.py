"""L2 `labels.error_rate`, `labels.score_ceiling`: how wrong the answer key is, and what it caps.

Every leaderboard number is an agreement rate with a set of labels, and every set of labels has an
error rate. Published ones run from 0.15% to 10.12%, average roughly 3.3% across ten benchmarks, and
sit at 6% on the ImageNet validation split that a decade of vision research was ranked on. No
benchmark publishes an estimated label-noise floor and no leaderboard normalises by one.

The arithmetic that follows is simple and nobody does it. If a fraction `e` of the labels are wrong,
the highest score a perfect model can post is `1 - e`, because on the wrong labels it is marked
wrong for being right. So a benchmark with a 4% error rate has a ceiling near 96%, and a model
reported at 97% on it is not above the ceiling because it is better than perfect; it is above the
ceiling because part of what it is matching is the annotation error. As models approach the ceiling
the leaderboard increasingly ranks agreement with annotation error, and the ranking degrades into a
measurement of who has best learned the annotators' mistakes.

Three rungs, and they measure different things rather than the same thing better.

    rung 0  a hand-audited sample, with a Wilson interval. The only rung that estimates the rate
            directly. Its cost is human attention and its precision is the square root of that.
    rung 1  item-response mislabel surfacing. Fits a two-parameter logistic model to a
            raters-by-items response matrix and flags items whose discrimination comes out
            negative, which is the signature of an item where the stronger raters do worse. It
            produces a ranked candidate list rather than a rate, and it becomes a rate only when
            somebody audits the top of the list, at which point it gives a *lower* bound on the
            corpus error rate.
    rung 2  a two-rater agreement design. Two independent labellings of the same items put a
            **lower bound on the average of the two raters' error rates**, at half the disagreement
            rate. That is the only thing they bound without an assumption, and which quantity it
            bounds is the whole of the care needed here: see `two_rater_bounds`.

Kill condition, from the catalogue: **n/a.**
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import Relation
from reward_lens.core.quantity import BiasStatement, CostModel, EstimatorEntry, register_estimator
from reward_lens.core.reading import Reading
from reward_lens.core.types import Capability
from reward_lens.measure.base import Context
from reward_lens.measure.labels._common import (
    ACCESS_LABELLED_CORPUS,
    LabelsInstrument,
    Proportion,
    emit_with_reference,
    label_quality_refusal,
    wilson_interval,
)
from reward_lens.record.labels import LabelQuality


@dataclass(frozen=True)
class AuditSample:
    """The outcome of a hand audit: how many were checked and how many were found wrong.

    ``population`` is the size of the corpus the sample was drawn from and it is not decoration. An
    audit of 200 items out of 500 has a finite-population correction that matters; an audit of 200
    out of 50,000 does not. More importantly, an audit drawn from a *different* population than the
    one being scored estimates that other population's error rate, and naming the population is the
    only thing that lets a reader notice.
    """

    n_audited: int
    n_wrong: int
    population: int | None = None
    method: str = ""
    measured_by: str = ""
    stratum: str = ""

    def __post_init__(self) -> None:
        if self.n_audited < 0 or self.n_wrong < 0:
            raise ValueError("an audit cannot have a negative count")
        if self.n_wrong > self.n_audited:
            raise ValueError(
                f"{self.n_wrong} wrong out of {self.n_audited} audited. More errors than items is "
                f"a bookkeeping fault, not a very bad label set."
            )


@register_payload
@dataclass(frozen=True)
class LabelErrorRate:
    """The measured error rate of a label set and the score ceiling it implies.

    ``ceiling`` is `1 - error_rate` and its interval is the error rate's, reflected. That reflection
    is exact rather than approximate: the Wilson interval on `e` maps to an interval on `1 - e` by
    swapping the ends, because `1 - e` is a strictly decreasing function of `e`.

    ``headroom`` is the reported score minus the ceiling when a score was supplied. A positive
    headroom is the sentence this instrument exists to be able to write.
    """

    error_rate: Proportion
    ceiling: float
    ceiling_low: float
    ceiling_high: float
    rung: int
    method: str = ""
    measured_by: str = ""
    population: int | None = None
    reported_score: float | None = None
    headroom: float | None = None
    interpretation: str = ""
    #: Rung 1 and rung 2 produce bounds rather than a point, and this says which end is which.
    bound_kind: str = "point"
    n_candidates: int = 0
    #: Whether the design identifies the error rate, or only constrains it. A hand audit does; a
    #: surfaced-and-audited list gives a floor and a two-rater design gives a lower bound on an
    #: average. This is what `as_label_quality` reads, rather than parsing `bound_kind`.
    identified: bool = True

    def as_label_quality(self) -> LabelQuality:
        """The record layer's `LabelQuality`, so a measured rate can be attached to a `Blind`.

        The point of the conversion is that `record.labels.adjudicate` refuses a scoring read when
        the quality is unmeasured, and this is the object that stops it refusing. A rate measured
        here is the thing that licenses a scoring read there.

        **An unidentified rung exports `None`, and that is the whole of this method's judgement.**
        `LabelQuality.error_rate` is written into the audit row and it is the value that licenses a
        scoring read; a rung that only bounds the rate has no single number entitled to sit there.
        Rung 2 used to export `d/2`, the infimum of its own identified set, so a two-rater design
        disagreeing on 35% of items licensed a scoring read on a claimed error rate of 0.175 when
        the independence model on the same disagreement gives 0.226 and correlated raters give more
        than that. Exporting the most flattering member of an identified set into the gate that
        decides whether scoring is allowed is the defect this library exists to prevent, so the
        gate now refuses and the remedy is a third rater.

        The rung's numbers are not lost: `error_rate`, `ceiling` and `interpretation` carry the
        whole reading, and `n_audited` still travels so a reader can see the design was run.
        """
        if not self.error_rate.is_measured:
            return LabelQuality(
                error_rate=None,
                n_audited=self.error_rate.n,
                method=self.method,
                measured_by=self.measured_by,
            )
        if not self.identified:
            return LabelQuality(
                error_rate=None,
                n_audited=self.error_rate.n,
                method=(
                    f"{self.method}; this design bounds the error rate rather than identifying it "
                    f"({self.bound_kind}), so no single rate is exported to license a scoring read"
                ),
                measured_by=self.measured_by,
            )
        return LabelQuality(
            error_rate=self.error_rate.point,
            n_audited=self.error_rate.n,
            method=self.method,
            measured_by=self.measured_by,
        )

    def render(self) -> str:
        lines = [
            f"label error rate  {self.error_rate.render()}",
            f"score ceiling     {self.ceiling:.4g} "
            f"[{self.ceiling_low:.4g}, {self.ceiling_high:.4g}]  (rung {self.rung}, "
            f"{self.bound_kind})",
        ]
        if self.reported_score is not None:
            lines.append(
                f"reported score    {self.reported_score:.4g}, headroom "
                f"{self.headroom:+.4g} against the ceiling"
            )
        if self.interpretation:
            lines.append(self.interpretation)
        return "\n".join(lines)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "error_rate": self.error_rate.__canonical__(),
            "ceiling": self.ceiling,
            "ceiling_low": self.ceiling_low,
            "ceiling_high": self.ceiling_high,
            "rung": self.rung,
            "method": self.method,
            "measured_by": self.measured_by,
            "population": self.population,
            "reported_score": self.reported_score,
            "headroom": self.headroom,
            "bound_kind": self.bound_kind,
            "n_candidates": self.n_candidates,
            "identified": self.identified,
        }


def _interpret(rate: Proportion, ceiling: float, reported: float | None) -> str:
    """The sentence that belongs beside every leaderboard number, chosen by a fixed rule."""
    if not rate.is_measured:
        return (
            "nobody has audited these labels, so the ceiling is unknown and every score against "
            "them is a sum of the model's accuracy and the annotators' error rate."
        )
    base = (
        f"a perfect model scores {ceiling:.1%} here, because on the {rate.point:.1%} of items the "
        f"labels have wrong it is marked wrong for being right."
    )
    if reported is None:
        return base
    if reported > rate.high and reported > ceiling:
        return (
            base
            + f" A model reported at {reported:.1%} is above that ceiling, which is not evidence "
            f"of a better-than-perfect model: part of what it is matching is the annotation error."
        )
    gap = ceiling - reported
    return (
        base + f" A model reported at {reported:.1%} has {gap:.1%} of real headroom left, not "
        f"{1.0 - reported:.1%}."
    )


def audit_error_rate(
    audit: AuditSample,
    *,
    reported_score: float | None = None,
    level: float = 0.95,
) -> LabelErrorRate:
    """Rung 0. A count of wrong labels in a sample, with the interval and the ceiling it implies.

    Wilson rather than Wald, and the reason is the case that comes up most: an audit finding zero
    errors. Wald gives `[0, 0]` and reads as a proof of perfect labels produced by looking at twenty
    items; Wilson gives an upper bound that shrinks with the sample size and never claims more than
    the sample supports.
    """
    rate = wilson_interval(audit.n_wrong, audit.n_audited, level=level)
    ceiling = 1.0 - rate.point if rate.is_measured else float("nan")
    return LabelErrorRate(
        error_rate=rate,
        ceiling=ceiling,
        ceiling_low=1.0 - rate.high,
        ceiling_high=1.0 - rate.low,
        rung=0,
        method=audit.method or "hand audit of a sample, Wilson score interval",
        measured_by=audit.measured_by,
        population=audit.population,
        reported_score=reported_score,
        headroom=None if reported_score is None else reported_score - ceiling,
        interpretation=_interpret(rate, ceiling, reported_score),
    )


# ---------------------------------------------------------------------------
# Rung 1: item response theory
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class MislabelCandidates:
    """Items an item-response model says are labelled inconsistently with everything else.

    ``discrimination`` is the fitted 2PL `a` per item. A well-labelled item has `a > 0`: raters who
    do well overall do well on it. An item with `a < 0` is one the strong raters get *wrong* and the
    weak raters get right, and after the ordinary explanations (a trick question, an ambiguous
    stem) the remaining one is that the answer key is wrong.

    This is a ranked list and not a rate. Turning it into a rate needs somebody to audit the top of
    it, which is what `bound_from_surfacing` does, and the result is a lower bound because the model
    surfaces the mislabels it can see and says nothing about the ones it cannot.
    """

    item_ids: tuple[str, ...]
    discrimination: tuple[float, ...]
    difficulty: tuple[float, ...]
    ranked: tuple[int, ...]
    n_negative: int
    n_items: int
    n_raters: int
    n_iterations: int
    #: Whether the *ranking* stopped moving, which is the output. See `irt_surface`.
    converged: bool
    max_param_delta: float = float("nan")

    def top(self, k: int) -> tuple[str, ...]:
        return tuple(self.item_ids[i] for i in self.ranked[:k])

    def render(self) -> str:
        state = (
            "ranking stable" if self.converged else "ranking still moving at the iteration limit"
        )
        return (
            f"2PL over {self.n_raters} raters x {self.n_items} items, {self.n_iterations} "
            f"iterations ({state}, last parameter step {self.max_param_delta:.3g}); "
            f"{self.n_negative} items have negative discrimination"
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "item_ids": list(self.item_ids),
            "discrimination": list(self.discrimination),
            "difficulty": list(self.difficulty),
            "ranked": list(self.ranked),
            "n_negative": self.n_negative,
            "n_items": self.n_items,
            "n_raters": self.n_raters,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "max_param_delta": self.max_param_delta,
        }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def irt_surface(
    responses: np.ndarray,
    *,
    item_ids: Sequence[str] = (),
    n_iterations: int = 100,
    step: float = 0.5,
    stable_for: int = 5,
) -> MislabelCandidates:
    """Fit a 2PL by joint maximum likelihood and rank items by how negative their slope is.

    `responses` is a raters-by-items binary matrix: 1 where the rater agreed with the recorded
    label, 0 where it did not. In benchmark terms a rater is a scored model and the entry is whether
    that model got the item "right" according to the key.

    Joint MLE rather than marginal, which is the honest trade. Joint estimates are known to be
    inconsistent as the rater count grows with the item count fixed, and the bias falls on the
    *magnitudes* of `a` and `b` rather than on their signs. The sign of `a` is the whole signal
    here, so the estimator that is cheap, dependency-free and biased in the magnitudes is the right
    one, and the ranking it produces is a ranking rather than a calibrated probability.

    Ability is initialised at the standardised rater total, which is the classical-test-theory
    estimate and a good start; items are then refit by Newton steps against fixed abilities and
    abilities refit against fixed items, alternating.

    **Convergence is tested on the ranking rather than on the parameters, and that is deliberate.**
    Joint MLE with clipped parameters wanders in the last few digits of `a` and `b` more or less
    indefinitely, so a parameter tolerance reports `converged=False` on a fit whose output has been
    identical for fifty iterations. The output here is an ordering, so the stopping rule is that the
    ordering has not moved for `stable_for` iterations, and `max_param_delta` travels with the
    result for a reader who wants the parameter story too.
    """
    r = np.asarray(responses, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"responses must be a raters-by-items matrix; got shape {r.shape}")
    n_raters, n_items = r.shape
    ids = tuple(item_ids) if item_ids else tuple(f"item{j}" for j in range(n_items))
    if len(ids) != n_items:
        raise ValueError(f"{len(ids)} item ids for {n_items} columns")

    total = r.sum(axis=1)
    theta = (total - total.mean()) / (total.std() if total.std() > 0 else 1.0)
    a = np.ones(n_items)
    b = np.zeros(n_items)
    converged = False
    used = 0
    stable = 0
    last_order: tuple[int, ...] = ()
    delta = float("nan")
    for it in range(n_iterations):
        used = it + 1
        a_prev = a.copy()
        b_prev = b.copy()
        # Items, given abilities: one Newton step per item on the 2PL log-likelihood.
        z = a[None, :] * (theta[:, None] - b[None, :])
        p = _sigmoid(z)
        w = np.clip(p * (1.0 - p), 1e-9, None)
        resid = r - p
        d = theta[:, None] - b[None, :]
        # Ridge-penalised Newton, `a ~ N(1, 1)` and `b ~ N(0, 1)`. Unpenalised joint MLE on a 2PL
        # is not merely inconsistent, it diverges: an item every rater answers the same way has a
        # flat likelihood in `a`, the Newton step is then a division by a number near zero, and the
        # parameter runs to whatever bound it is clipped at and oscillates there. Building this
        # produced exactly that, with a per-iteration step of 9.4 on a parameter clipped to +/-6.
        # The penalty is the standard regularisation for joint estimation and it pulls only the
        # items the data has nothing to say about.
        g_a = (resid * d).sum(axis=0) - (a - 1.0)
        h_a = -(w * d * d).sum(axis=0) - 1.0
        g_b = -(resid * a[None, :]).sum(axis=0) - b
        h_b = -(w * a[None, :] ** 2).sum(axis=0) - 1.0
        a = np.clip(a - step * g_a / h_a, -6.0, 6.0)
        b = np.clip(b - step * g_b / h_b, -6.0, 6.0)
        # Abilities, given items.
        z = a[None, :] * (theta[:, None] - b[None, :])
        p = _sigmoid(z)
        w = np.clip(p * (1.0 - p), 1e-9, None)
        g_t = ((r - p) * a[None, :]).sum(axis=1)
        h_t = -(w * a[None, :] ** 2).sum(axis=1)
        theta = theta - step * g_t / np.where(h_t < -1e-9, h_t, -1e-9)
        theta = np.clip(theta, -6.0, 6.0)
        theta = (theta - theta.mean()) / (theta.std() if theta.std() > 1e-9 else 1.0)
        delta = max(float(np.abs(a - a_prev).max()), float(np.abs(b - b_prev).max()))
        order = tuple(int(i) for i in np.argsort(a))
        stable = stable + 1 if order == last_order else 0
        last_order = order
        if stable >= stable_for:
            converged = True
            break
    order_final = tuple(int(i) for i in np.argsort(a))
    return MislabelCandidates(
        item_ids=ids,
        discrimination=tuple(float(v) for v in a),
        difficulty=tuple(float(v) for v in b),
        ranked=order_final,
        n_negative=int((a < 0).sum()),
        n_items=n_items,
        n_raters=n_raters,
        n_iterations=used,
        converged=converged,
        max_param_delta=delta,
    )


def bound_from_surfacing(
    candidates: MislabelCandidates,
    *,
    k_audited: int,
    n_confirmed: int,
    population: int | None = None,
    reported_score: float | None = None,
    level: float = 0.95,
) -> LabelErrorRate:
    """Rung 1. Audit the top `k` of the surfaced list, then bound the corpus error rate below.

    The bound is `n_confirmed / N` and it is a lower bound for a plain reason: the audit found that
    many wrong labels, so at least that many exist. It says nothing about the labels the model did
    not surface, and the surfacing is not exhaustive by construction. Reporting it as a point
    estimate would be understating the error rate on purpose, which is why `identified=False`
    travels with it and `as_label_quality()` declines to export it as the rate that licenses a
    scoring read.

    The interval on it is the Wilson interval on the audit's own precision, scaled by `k/N`, which
    propagates the audit's sampling error and not the surfacing's coverage. That is the honest
    limit of what this rung can say.
    """
    n = population or candidates.n_items
    precision = wilson_interval(n_confirmed, k_audited, level=level)
    lower = n_confirmed / n if n else float("nan")
    rate = Proportion(
        k=n_confirmed,
        n=n,
        point=lower,
        low=precision.low * k_audited / n if n else float("nan"),
        high=precision.high * k_audited / n if n else float("nan"),
        level=level,
        method="lower bound from IRT surfacing plus an audit of the top k",
    )
    ceiling = 1.0 - rate.point
    return LabelErrorRate(
        error_rate=rate,
        ceiling=ceiling,
        ceiling_low=1.0 - rate.high,
        ceiling_high=1.0 - rate.low,
        rung=1,
        method=(
            f"2PL discrimination ranking over {candidates.n_raters} raters, top {k_audited} "
            f"audited, {n_confirmed} confirmed mislabelled"
        ),
        population=n,
        reported_score=reported_score,
        headroom=None if reported_score is None else reported_score - ceiling,
        identified=False,
        bound_kind="lower bound on the error rate, so an upper bound on the ceiling",
        n_candidates=candidates.n_negative,
        interpretation=(
            f"at least {n_confirmed} of {n} labels are wrong, so the ceiling is at most "
            f"{ceiling:.1%}. The surfacing finds the mislabels an item-response model can see and "
            f"is silent about the rest, so this is a floor under the error rate and not an "
            f"estimate of it."
        ),
    )


# ---------------------------------------------------------------------------
# Rung 2: two raters
# ---------------------------------------------------------------------------


def independent_rater_rate(d: float) -> float:
    """Invert `d = 2e(1-e)` for the smaller root: the error rate two independent raters imply.

    The classical two-rater latent-class answer when the raters are exchangeable and err
    independently at a common rate `e`. They disagree exactly when one is wrong and the other is
    right, which under independence is `2e(1-e)`, so `e = (1 - sqrt(1 - 2d)) / 2`.

    Returns NaN for `d > 0.5`, which is not an edge case to be clamped away. The maximum
    disagreement two independent raters at a common rate can produce is 0.5, at `e = 0.5`, so a
    higher measured disagreement refutes the model rather than pushing it into a corner. The larger
    root `(1 + sqrt(1 - 2d)) / 2` is the mirror solution with both raters worse than chance and it
    is not returned, because a labelling worse than a coin is a different finding and should not
    arrive disguised as an error rate.
    """
    if not 0.0 <= d <= 0.5:
        return float("nan")
    return (1.0 - math.sqrt(1.0 - 2.0 * d)) / 2.0


def two_rater_bounds(
    rater_a: Sequence[Any],
    rater_b: Sequence[Any],
    *,
    reported_score: float | None = None,
    level: float = 0.95,
    method: str = "",
    measured_by: str = "",
) -> LabelErrorRate:
    """Rung 2. Two independent labellings of the same items, and what they actually bound.

    **The quantity bounded is the average of the two raters' error rates, not either rater's.**
    That distinction is the whole of this rung and it was stated backwards here for four months, so
    it is worth deriving. On binary labels, a disagreement means exactly one rater is wrong, and an
    agreement means either both are right or both are wrong. Summing,

        e_A + e_B = d + 2 * P(both wrong)      so      mean(e) = d / 2 + P(both wrong)

    which makes `d/2` a lower bound on `mean(e)` with no assumption at all, exact when the two never
    agree on a wrong answer. Simulated at N = 400,000 the residual on that identity is 1.4e-17 at
    equal rater rates and 2.8e-17 at unequal ones.

    It is **not** a lower bound on an individual rater. With rater A perfect and rater B at 20%,
    the measured disagreement is 0.19998 and `d/2` claims every rater is at least 0.09999 while A
    is at 0.00000. At `e_A = 0.05, e_B = 0.25` it claims 0.13749 against A's measured 0.05004. The
    old docstring here claimed the bound for either rater and the old code returned it as a point
    estimate of one rater's rate, and both were wrong in the direction that flatters the labels.

    The upper end is `d`, and it bounds a **different** quantity under a **different** assumption:
    an individual rater's rate is at most `d` only if the two never agree on a wrong answer, since
    then `e_A + e_B = d` and both are non-negative. Under shared error there is no upper bound of
    any kind. Two raters who share every error disagree nowhere, so `d = 0` while both are wrong
    everywhere, and nothing in the design can see it.

    ``point`` is the independence model, `d = 2e(1-e)` inverted to `e = (1 - sqrt(1-2d))/2`. It is
    a stated model rather than a bound, and it is here because the alternative was to report an end
    of the identified set as though it were an estimate. At `d = 0.35` it gives 0.2261 where `d/2`
    gives 0.1750, and the direction is not accidental: any positive correlation between the raters
    pushes the true rate above the independence value, so this is the optimistic end of the
    *modelled* range rather than the optimistic end of the *identified* one. `d > 0.5` has no root,
    because two independent raters at a common rate cannot disagree more than half the time, and
    the point is NaN there rather than a complex number rounded to something.

    Two raters cannot identify the error rate. Three can, through the classical latent-class
    argument, and that is why `as_label_quality()` exports `None` from this rung rather than
    licensing a scoring read on a number the design does not pin down.
    """
    a = list(rater_a)
    b = list(rater_b)
    if len(a) != len(b):
        raise ValueError(f"{len(a)} labels from rater A and {len(b)} from rater B")
    n = len(a)
    disagree = sum(1 for x, y in zip(a, b) if x != y)
    d = wilson_interval(disagree, n, level=level)
    point = independent_rater_rate(d.point) if d.is_measured else float("nan")
    rate = Proportion(
        k=disagree,
        n=n,
        point=point,
        low=d.low / 2.0,
        high=d.high,
        level=level,
        method=(
            "two-rater disagreement; point is the independence model, low is the unconditional "
            "lower bound on the two raters' average rate, high is the per-rater bound under no "
            "shared error"
        ),
    )
    ceiling = 1.0 - rate.point
    return LabelErrorRate(
        error_rate=rate,
        ceiling=ceiling,
        ceiling_low=1.0 - rate.high,
        ceiling_high=1.0 - rate.low,
        rung=2,
        method=method or "two independent labellings of the same items",
        measured_by=measured_by,
        population=n,
        reported_score=reported_score,
        headroom=None if reported_score is None else reported_score - ceiling,
        identified=False,
        bound_kind=(
            "three numbers about two different quantities: the low end is an unconditional lower "
            "bound on the average of the two raters' rates, the high end bounds an individual "
            "rater only if the two never agree on a wrong answer, and the point is the "
            "independence model rather than either bound"
        ),
        interpretation=(
            f"the two labellings disagree on {disagree} of {n} items ({d.point:.1%}). Half of that, "
            f"{d.point / 2.0:.1%}, is a lower bound on the *average* of the two raters' error "
            f"rates and says nothing about either rater on its own. Assuming they err "
            f"independently at a common rate gives {point:.1%}, which is what the ceiling of "
            f"{ceiling:.1%} is computed from. A shared bias puts the true rate above that and two "
            f"raters cannot detect a shared bias, so this rung bounds the labels rather than "
            f"measuring them: a third independent labelling identifies the rate."
        ),
    )


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

LABEL_ERROR_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "a count over an audited sample. It counts what the auditors found and asserts nothing "
        "about the process that produced the labels, so no regime can make the count wrong. The "
        "one precondition that does bite, that the audit sample and the scored set are the same "
        "population, is not among the twelve envelope conditions and is carried on the audit "
        "itself as `population` so a mismatch is visible rather than assumed away."
    ),
)


class LabelErrorAudit(LabelsInstrument):
    """L2: what fraction of the answer key is wrong, and the score it therefore caps.

    Kill condition, from the catalogue: **n/a.**

    The refusal is the part that does the work. Asked for the error rate of a label set nobody has
    audited, this returns `LABEL_QUALITY_UNKNOWN` rather than zero, because zero is a claim that
    the labels are perfect and nobody has looked.
    """

    name = "LabelErrorAudit"
    version = "1.0"
    quantity = "labels.error_rate"
    capabilities = Capability.NONE
    requires = ACCESS_LABELLED_CORPUS
    envelope = LABEL_ERROR_ENVELOPE
    invariance = "none"
    invariance_relation = Relation("invariant")
    baselines = (
        "assume the labels are right, which is what every leaderboard does and which predicts a "
        "ceiling of 1.0",
    )
    rung = 0
    faithful_to = "the Wilson score interval on a binomial proportion"
    deviations = (
        "no finite-population correction. An audit of a large fraction of a small corpus has a "
        "narrower interval than this reports, so the interval is conservative in that case and "
        "exact in the usual one.",
        "the ceiling is `1 - e`, which assumes a wrong label costs exactly one point and that a "
        "model cannot accidentally match a wrong label. On a multiple-choice benchmark a model has "
        "a chance of matching the wrong key by luck and the true ceiling is slightly higher.",
        "rung 1's joint maximum likelihood is inconsistent in the magnitudes of the item "
        "parameters. Only the sign of the discrimination is used, and the ranking is a ranking.",
    )

    def __init__(
        self,
        audit: AuditSample | None = None,
        *,
        label_set: str = "",
        reported_score: float | None = None,
        level: float = 0.95,
    ) -> None:
        self.audit = audit
        self.label_set = label_set or "unnamed label set"
        self.reported_score = reported_score
        self.level = level

    def measure(self, ctx: Context) -> Any:
        result = audit_error_rate(self.audit, reported_score=self.reported_score, level=self.level)
        return emit_with_reference(
            ctx,
            result,
            quantity=self.quantity,
            uncertainty=Uncertainty(
                n=result.error_rate.n,
                ci_low=result.error_rate.low,
                ci_high=result.error_rate.high,
                ci_level=self.level,
                method="wilson score",
            ),
            baselines={"baseline.labels_are_right": 1.0},
            subject_extra={"label_set": self.label_set},
        )

    def estimate(self, ctx: Context | None = None) -> Reading:
        ctx = ctx or Context(readout="score")
        if self.audit is None or self.audit.n_audited == 0:
            return label_quality_refusal(
                self.name,
                what=(
                    f"label set {self.label_set!r} has no audited sample, so its error rate is "
                    f"unmeasured and its score ceiling is unknown"
                ),
                remedy=(
                    "audit a sample of these labels and pass AuditSample(n_audited=..., "
                    "n_wrong=..., method=..., measured_by=...). A hundred items gives an error "
                    "rate to about three points at 95%, which is enough to say whether a "
                    "leaderboard gap of one point is a gap. If you cannot audit, say the ceiling "
                    "is unknown rather than assuming it is 1.0."
                ),
                label_set=self.label_set,
                n_audited=0 if self.audit is None else self.audit.n_audited,
            )
        return super().estimate(ctx)


_REGISTERED = False


def register() -> None:
    """Register L2's three rungs. Idempotent."""
    global _REGISTERED
    if _REGISTERED:
        return
    register_estimator(
        EstimatorEntry(
            quantity="labels.error_rate",
            impl="labels.error_rate.r0_audit",
            requires=ACCESS_LABELLED_CORPUS,
            envelope=LABEL_ERROR_ENVELOPE,
            rung=0,
            bias=BiasStatement(
                direction="unknown",
                why=(
                    "an audit inherits the auditor. A rater pool that shares the annotators' "
                    "assumptions finds fewer errors than there are; one that applies a stricter "
                    "standard than the benchmark intended finds more."
                ),
            ),
            cost=CostModel(note="human attention, and its precision goes as the square root of it"),
            run=audit_error_rate,
        )
    )
    register_estimator(
        EstimatorEntry(
            quantity="labels.error_rate",
            impl="labels.error_rate.r1_irt_surfacing",
            requires=ACCESS_LABELLED_CORPUS,
            envelope=LABEL_ERROR_ENVELOPE,
            rung=1,
            bias=BiasStatement(
                direction="downward",
                why=(
                    "surfacing finds the mislabels an item-response model can see, which are the "
                    "ones the strong raters agree are wrong. A mislabel every rater gets wrong "
                    "looks like a hard item and is invisible here, so the bound is a floor."
                ),
            ),
            cost=CostModel(cpu_seconds=10.0, note="a 2PL fit plus an audit of the top k"),
            run=bound_from_surfacing,
        )
    )
    register_estimator(
        EstimatorEntry(
            quantity="labels.error_rate",
            impl="labels.error_rate.r2_two_rater",
            requires=ACCESS_LABELLED_CORPUS,
            envelope=LABEL_ERROR_ENVELOPE,
            rung=2,
            bias=BiasStatement(
                direction="downward",
                why=(
                    "two raters cannot see an error they share. Any bias common to both labellings "
                    "produces agreement, agreement is scored here as correctness, and the "
                    "disagreement rate the whole rung is built on shrinks. That bites the "
                    "independence-model point estimate as well as the bound: positive correlation "
                    "between the raters lowers `d` at any fixed true rate, so inverting `d` under "
                    "independence returns a rate below the truth."
                ),
            ),
            cost=CostModel(note="a second independent labelling of every audited item"),
            run=two_rater_bounds,
        )
    )
    # The ceiling is `1 - e` and it comes out of the same call, so it is registered against the
    # same implementation rather than given a second one. Registering it at all is the point: a
    # quantity with no estimator reads in the capability report as an open research target, and
    # the score ceiling is arithmetic on a measured error rate rather than an open problem.
    register_estimator(
        EstimatorEntry(
            quantity="labels.score_ceiling",
            impl="labels.score_ceiling.r0_from_audit",
            requires={},
            envelope=LABEL_ERROR_ENVELOPE,
            rung=0,
            bias=BiasStatement(
                direction="upward",
                why=(
                    "`1 - e` assumes a model cannot match a wrong label by luck. On a "
                    "multiple-choice benchmark it sometimes can, so the true ceiling is a little "
                    "higher than this and a model reported above it is a little less anomalous."
                ),
            ),
            cost=CostModel(note="one subtraction from a measured error rate"),
            run=audit_error_rate,
        )
    )
    _REGISTERED = True


__all__ = [
    "LABEL_ERROR_ENVELOPE",
    "AuditSample",
    "LabelErrorAudit",
    "LabelErrorRate",
    "MislabelCandidates",
    "audit_error_rate",
    "bound_from_surfacing",
    "independent_rater_rate",
    "irt_surface",
    "register",
    "two_rater_bounds",
]

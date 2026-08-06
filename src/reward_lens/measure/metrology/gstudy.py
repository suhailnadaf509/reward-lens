"""A1, effective group size: how many of your K rollouts are independent observations.

A group of K rollouts costs K forward passes and K grader calls. What it buys is a smaller number,
because some of the spread between those K scores is the grader disagreeing with itself rather than
the rollouts differing. The recorded scores cannot show that and only replication can.

**One quantity, four rungs, and the rungs must not drift apart.** The quantity is

    n_eff = K x (the reliability of a single observed score)

The reliability factor is the generalizability coefficient of one observed score over the declared
universe of grader draws: it is 1 when the grader is noiseless and falls toward 0 as measurement
error takes over the observed spread. Every rung estimates that same product. What changes between
rungs is which error terms the design can see, and a rung that cannot see a term is implicitly
setting it to zero, which is why every rung below the top is biased **upward** and the ladder is
monotone. Rung 0 sets the whole error to zero because it has one score per rollout and no way to
tell error from signal, so **rung 0 returns exactly K**. That is not a hole in the instrument, it is
the reason the number people quote today is K, printed by the instrument that exists to correct it.

The reliability factor is the standard result that measurement error costs effective sample size in
proportion to reliability. Spearman's 1904 attenuation correction is the same statement about a
correlation, and it has gone unapplied to a reward loop for 120 years.

**This number changed on 2026-08-05 and the old value is not recoverable from the new one.** Until
then the reading was `n_eff = kish x reliability`, where `kish` is the Kish count of the group's own
observed score spread. That product charges measurement noise twice. The Kish count is computed on
observed scores which already contain grader noise, and the reliability factor then discounts for
that same noise: noise pulls the observed shape factor toward `2/pi` for the same reason it pulls
reliability below one. A two-point truth settles it, because a two-point score distribution has a
shape factor of exactly 1.0 by construction and noise is then the only thing moving. Measured on
16 rollouts per group at `sigma_tau = 1.0` over 1,000 groups, with the reliability fitted from a
real crossed design of 800 objects by 4 grader draws:

    reliability 0.4994, so the group carries 7.99 independent observations of 16
    what this module used to report: 5.59
    what it reports now:              7.99

On the eleven open reward models of series A, at K = 4 over 1,763 groups, rung 3 moved from a mean
of 1.910 to 2.558 and rung 0 from 2.986 to 4.000. The Kish count has not gone away: it is a real
statement about how unevenly a group spends its gradient, so it travels beside the reading as
`EffectiveSize.shape_factor` with its own bootstrap interval. It is a property of the reward
distribution and not of the grader, which is why multiplying it into a grader's effective group
size was the error.

**Kill condition, from the catalogue.** If r0 and r3 agree within their intervals on five graders,
the ladder is decoration and only r0 ships. That is a real test and it is in
`tests/acceptance/test_w3_2a_metrology.py`, run against eleven open reward models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Evidence, Uncertainty
from reward_lens.core.gates import require_frame_for_comparison
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import (
    FREE,
    BaselineID,
    BiasStatement,
    CostModel,
    EstimatorEntry,
    Quantity,
    Unit,
    register_estimator,
)
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.reference import Transfer, ladder_disagreement
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context, run
from reward_lens.measure.rate.regime import MEASURED_BY
from reward_lens.stats.effects import bootstrap_ci
from reward_lens.stats.gtheory import (
    DesignError,
    GStudy,
    check_balance,
    crossed_pr,
    crossed_pro,
    statsmodels_available,
    to_cube,
)
from reward_lens.stats.variance import group_effective_size

# ---------------------------------------------------------------------------
# The shared seam for this package's instruments
# ---------------------------------------------------------------------------


class MetrologyInstrument(BaseObservable):
    """Preflight, compute once, refuse or emit. Shared by A1, A2 and A5.

    `Observable.measure` returns `Evidence` by contract and `Instrument.estimate` returns
    `Reading`, which is `Evidence | Refusal`. Both contracts are right and they need a seam,
    because all three instruments here decide to refuse partway through: whether a design is
    crossed, whether it has two raters, whether a budget admits any feasible allocation are all
    things you find out by looking at the data, not in preflight.

    `compute` takes no `Context` on purpose. These are pure functions of arrays the caller already
    holds, and being callable without standing up a signal is what makes them usable from a test,
    from a notebook, and from a preflight.
    """

    #: Set by `estimate` for the duration of one call so `measure` does not recompute.
    _computed: Any = None

    def compute(self) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def payload(self, computed: Any) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def uncertainty(self, computed: Any) -> Uncertainty | None:
        """The interval, where the instrument has one. `None` is a real answer here.

        A5's plan is exact arithmetic on components whose uncertainty is somebody else's, so it
        carries none of its own and says so by returning None rather than by manufacturing a
        symmetric interval around a rounded integer tuple.
        """
        return None

    def gated_emit(self, ctx: Context, computed: Any) -> Evidence:
        """Hand a computed result to the runner, or apply the runner's gates by hand.

        These three instruments read injected arrays rather than a network, so `ctx.signal` is
        normally None and `run` cannot be used: it resolves `ctx.signal.caps` to enforce the
        capability check. The no-signal branch does what `run` would have done minus the check that
        has nothing to check against, **including setting `ctx._observable`**. That last part is not
        cosmetic: `Context.emit` reads the observable's name, version and quantity off that
        attribute, so a branch that skips it emits Evidence named `anonymous` at version `0` with an
        empty quantity, and a store full of anonymous rows cannot be joined to the instrument that
        wrote them. Found here by a test asserting the emitted quantity.

        The frame requirement is not skipped. Gate 2 depends on the instrument's gauge status and
        the context's frame rather than on the signal, so it applies in both branches.
        """
        self._computed = computed
        try:
            if ctx.signal is not None:
                return run(self, ctx)
            if ctx.is_comparison:
                require_frame_for_comparison(self.gauge_status, ctx.frame)
            ctx._observable = self
            try:
                return self.measure(ctx)
            finally:
                ctx._observable = None
        finally:
            self._computed = None

    def estimate(self, ctx: Context) -> Reading:
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        out = self.compute()
        if isinstance(out, Refusal):
            return out
        return self.gated_emit(ctx, out)

    def measure(self, ctx: Context) -> Evidence:
        out = self._computed if self._computed is not None else self.compute()
        if isinstance(out, Refusal):
            raise ValueError(
                f"{self.name}.measure was called on a measurement that declines to produce "
                f"Evidence: {out.reason.name}. Call `estimate`, which returns the refusal as a "
                f"value with its remedy."
            )
        payload = self.payload(out)
        # The baselines go on the Evidence's own field as well as into the payload. A baseline that
        # lives only inside a value dict is invisible to anything that reads the store generically,
        # and the mandatory-baseline rule exists to be checkable from outside the instrument.
        declared = payload.get("baselines")
        return ctx.emit(
            payload,
            uncertainty=self.uncertainty(out),
            baselines=declared if isinstance(declared, Mapping) else None,
        )


# ---------------------------------------------------------------------------
# The quantity, written out because the registry carries OPEN for it
# ---------------------------------------------------------------------------

#: `spec/QUANTITIES.yaml` registers `grader.effective_group_size` with `definition: OPEN`, so the
#: registered `Quantity.definition` is the empty string and two rungs of this ladder have nothing
#: to be compared against. This is the definition, written out here so it exists in code and can be
#: lifted into the YAML verbatim. The unit is the registered one and is not changed.
EFFECTIVE_GROUP_SIZE = Quantity(
    id="grader.effective_group_size",
    definition=(
        "For a group of K rollouts scored on one prompt under one grader configuration: the "
        "number of noiselessly scored rollouts that would carry the same amount of signal as the "
        "K observed scores. It is K times the generalizability coefficient of a single observed "
        "score over the declared universe of grader draws, sigma2(tau) / (sigma2(tau) + "
        "sigma2(delta)) at one draw of every facet, which is 1 for a noiseless grader and falls "
        "toward 0 as measurement error takes over the observed spread. Every rung of the ladder "
        "estimates this one number and differs only in which error terms its design can see, so "
        "rung 0, which can see none, returns exactly K. The shape of the reward distribution "
        "inside the group does not enter: how unevenly a group spends its gradient is a fact "
        "about the policy and the reward, not about the grader, and it is reported separately as "
        "the group's Kish shape factor."
    ),
    unit=Unit(dimension="count", per="group", scale=None, as_printed="count"),
    invariance="reward.affine",
    interpretation=(
        "Compare it to K. n_eff = 4.2 at K = 16 means eleven point eight rollouts' worth of "
        "compute bought grader noise rather than an independent gradient-relevant observation. "
        "The rung tells you how much of the grader's error the number could see: below rung 3 "
        "some error terms are invisible and are counted as signal, so the reading is an upper "
        "bound and the invisible terms are named on it."
    ),
    support=(0.0, math.inf),
    wedge=True,
)

#: A1's envelope, from the catalogue: `GROUP_NONDEGENERATE`. `envelope_measured_by` is OPEN in the
#: catalogue, so the measurer is taken from the kernel's own regime module, which is where the
#: condition is actually measured, rather than named again here.
GROUP_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.GROUP_NONDEGENERATE}),
    measured_by={
        RegimeCondition.GROUP_NONDEGENERATE: MEASURED_BY[RegimeCondition.GROUP_NONDEGENERATE]
    },
    on_violation="refuse",
)

#: Catalogue A1 access: `GRADER: RECORD` at rung 0, `REPLICATE` from rung 2. The declaration is the
#: minimum, so it is RECORD; the rungs that need more check for it themselves and refuse by name.
EFFECTIVE_GROUP_SIZE_ACCESS: dict[Component, Access] = {Component.GRADER: Access.RECORD}

#: Catalogue A1 baselines, verbatim: "group size K itself" and "a single-rater design". Both are
#: computed on every reading, because the entire claim is the distance between n_eff and K.
EFFECTIVE_GROUP_SIZE_BASELINES: tuple[BaselineID, ...] = (
    "baseline.group_size_k",
    "baseline.single_rater_design",
)

#: The four rungs' bias statements, kept as data so the estimator registry and the reading cannot
#: disagree about which way a rung is wrong.
BIAS: Mapping[int, BiasStatement] = {
    0: BiasStatement(
        direction="upward",
        why=(
            "it cannot see correlated grader error at all. With one score per rollout there is no "
            "replication to separate error from signal, so the whole observed spread is counted as "
            "signal, the reliability factor is fixed at 1 and the reading is exactly K. It is the "
            "ceiling every rung above it comes down from"
        ),
    ),
    1: BiasStatement(
        direction="upward",
        why=(
            "repeated calls to one grader show the occasion facet and nothing else. Rater and "
            "rubric error are constant across the design, so they are invisible and counted as "
            "signal"
        ),
    ),
    2: BiasStatement(
        direction="upward",
        why=(
            "a crossed design with facet control and one rater resolves every error term except "
            "the ones indexed by the rater. Which grader you drew, and which rollouts that grader "
            "happens to favour, are still counted as signal"
        ),
    ),
    3: BiasStatement(
        direction="approximately_unbiased",
        why=(
            "with two or more raters the object-by-rater interaction is identified and counted as "
            "error. That is the term that moves every advantage in a group in the same direction, "
            "so it is the one a group-relative estimator cannot cancel. What remains is the method "
            "of moments' own small-sample behaviour, which the degrees of freedom on each mean "
            "square report"
        ),
    ),
}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupScores:
    """The recorded scores of one grader, grouped by prompt. What rung 0 reads.

    ``scores`` is a list of 1-D arrays, one per group, because real groups are not all the same
    size. A rectangular array is accepted and split.
    """

    groups: tuple[np.ndarray, ...]
    grader: str = ""

    @classmethod
    def of(cls, scores: Any, grader: str = "") -> "GroupScores":
        arr = np.asarray(scores, dtype=np.float64)
        if arr.ndim == 2:
            groups = tuple(np.asarray(row, dtype=np.float64) for row in arr)
        else:
            groups = tuple(np.asarray(g, dtype=np.float64).ravel() for g in scores)
        return cls(groups=groups, grader=grader)

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def k_mean(self) -> float:
        return float(np.mean([g.size for g in self.groups])) if self.groups else 0.0

    @property
    def degenerate_fraction(self) -> float:
        """Fraction of groups with no spread at all. The statistic behind `GROUP_NONDEGENERATE`."""
        if not self.groups:
            return 1.0
        flat = sum(1 for g in self.groups if g.size < 2 or float(np.std(g)) == 0.0)
        return flat / len(self.groups)


@dataclass(frozen=True)
class ReplicationDesign:
    """A crossed design of grader scores: objects by raters, optionally by a third facet.

    ``scores`` is ``(n_p, n_r)`` or ``(n_p, n_r, n_o)`` with one observation per cell. ``raters``
    names the rater levels so a component can be traced back to the grader that produced it.

    ``single_rater`` says that the axes after the first are *not* the rater facet: one grader was
    used throughout and the extra axes are repeat calls, rubric variants or prompt orderings. That
    distinction is what separates rungs 1 and 2 from rung 3, because the object-by-rater interaction
    is the term a single-rater design cannot see no matter how many other facets it varies, and a
    design that reported its repeat calls as raters would claim to have seen it.

    `from_long` is the constructor a caller usually wants: it takes parallel sequences and does the
    balance check, so an incomplete design is caught at construction with the missing cells named
    rather than silently reshaped.
    """

    scores: np.ndarray
    raters: tuple[str, ...] = ()
    object_label: str = "response"
    facet_labels: tuple[str, str] = ("rater", "occasion")
    objects: tuple[Any, ...] = ()
    occasions: tuple[Any, ...] = ()
    single_rater: bool = False

    def __post_init__(self) -> None:
        arr = np.asarray(self.scores, dtype=np.float64)
        if arr.ndim not in (2, 3):
            raise DesignError(
                f"a crossed design is (n_p, n_r) or (n_p, n_r, n_o); got shape {arr.shape}"
            )
        object.__setattr__(self, "scores", arr)
        if self.single_rater and self.facet_labels == ("rater", "occasion"):
            object.__setattr__(self, "facet_labels", ("occasion", "rubric"))

    @classmethod
    def from_long(
        cls,
        values: Sequence[float] | np.ndarray,
        objects: Sequence[Any],
        raters: Sequence[Any],
        occasions: Sequence[Any] | None = None,
        *,
        object_label: str = "response",
        facet_labels: tuple[str, str] = ("rater", "occasion"),
    ) -> "ReplicationDesign":
        factors: list[Sequence[Any]] = [list(objects), list(raters)]
        if occasions is not None:
            factors.append(list(occasions))
        cube, levels = to_cube(values, factors)
        return cls(
            scores=cube,
            raters=tuple(str(x) for x in levels[1]),
            object_label=object_label,
            facet_labels=facet_labels,
            objects=tuple(levels[0]),
            occasions=tuple(levels[2]) if len(levels) > 2 else (),
        )

    @property
    def n_p(self) -> int:
        return int(self.scores.shape[0])

    @property
    def n_r(self) -> int:
        """Levels of the rater facet. Exactly 1 when the design held the grader fixed."""
        return 1 if self.single_rater else int(self.scores.shape[1])

    @property
    def n_o(self) -> int:
        """Levels of the first facet that is not the rater."""
        if self.single_rater:
            return int(self.scores.shape[1])
        return int(self.scores.shape[2]) if self.scores.ndim == 3 else 1

    @property
    def facet_levels(self) -> tuple[int, ...]:
        """Levels of every measurement facet, in axis order after the object axis."""
        return tuple(int(n) for n in self.scores.shape[1:])

    @property
    def has_third_facet(self) -> bool:
        return self.scores.ndim == 3

    def fit(self) -> GStudy:
        """The G-study for whichever shape this design has."""
        if self.has_third_facet:
            return crossed_pro(
                self.scores,
                object_label=self.object_label,
                facet_labels=self.facet_labels,
            )
        return crossed_pr(
            self.scores,
            object_label=self.object_label,
            facet_label=self.facet_labels[0],
        )

    def without_raters(self) -> "ReplicationDesign":
        """The single-rater version of this design, which is catalogue A1's second baseline.

        Takes the first rater level and keeps the third facet. Returns a design with `n_r = 1`,
        which no crossed estimator can fit, so callers use it to say what a single-rater study
        would have been able to see rather than to fit one.
        """
        sub = self.scores[:, :1, ...]
        return ReplicationDesign(
            scores=sub,
            raters=self.raters[:1],
            object_label=self.object_label,
            facet_labels=self.facet_labels,
            objects=self.objects,
            occasions=self.occasions,
        )


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EffectiveSize:
    """One effective-group-size reading: `K x reliability`, its interval, and the shape factor beside it.

    **`n_eff` is `k_nominal * reliability`. It used to be `kish * reliability` and that product
    charged measurement noise twice.** The Kish count is computed on the *observed* scores, which
    already contain grader noise, and `reliability` then discounts for the same noise. Measured on a
    two-point truth whose shape factor is exactly 1.0 by construction, so noise is the only thing
    moving, at K = 16 over 4,000 groups:

        sigma_err   reliability   shape factor   old n_eff/K   new n_eff/K
             0.00        1.0000         1.0000        1.0000        1.0000
             0.50        0.8000         0.8307        0.6646        0.8000
             1.00        0.5000         0.7011        0.3506        0.5000
             2.00        0.2000         0.6639        0.1328        0.2000

    At a reliability of 0.5 the group carries eight independent observations of sixteen; the old
    rule reported 5.6. The mechanism is that adding noise pushes the observed score distribution
    toward Gaussian, and the Kish shape factor of a Gaussian is `2/pi = 0.6366`, so the first factor
    fell for the same reason the second one did.

    **The shape factor is still here and it is still worth reading.** `shape_factor` is the Kish
    count divided by K: 1.0 when every rollout sits the same distance from the group mean and
    falling toward `1/K` as the contrast concentrates on one rollout. It says how evenly the group
    spends its gradient. It is a property of the reward distribution and of the policy that produced
    the group, not of the grader, which is why it is reported beside `n_eff` rather than multiplied
    into it. On a perfect grader with Gaussian rewards it reads `2/pi = 0.637` and nothing is wrong.

    **What this instrument still cannot do.** It cannot deconvolve the true-score shape from the
    observed one, so `shape_factor` is itself measured on noisy scores and is pulled toward `2/pi`
    from whichever side it starts. A concentrated reward distribution reads too low and a
    heavy-tailed one too high, and the direction is not signable without knowing which you have.
    That is why the shape factor does not enter `n_eff`: it would import an unsignable bias into a
    number whose whole job is to be an honest bound.
    """

    rung: int
    grader: str
    k_nominal: float
    #: The Kish count of the group's own observed spread, in rollouts. Reported beside `n_eff` and
    #: deliberately not multiplied into it. Divide by `k_nominal` for the dimensionless shape
    #: factor, which is what `shape_factor` does.
    kish: float
    reliability: float
    n_eff: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_groups: int
    method: str
    universe: str
    #: Which error terms this rung could not see, named. Empty at rung 3.
    invisible_terms: tuple[str, ...] = ()
    gstudy: GStudy | None = None
    baselines: Mapping[str, float] = field(default_factory=dict)
    #: Standard error of the reliability factor, from leaving each rater out in turn. `None` means
    #: the reliability is treated as known and the reading carries no interval, and `method` says so.
    reliability_se: float | None = None
    #: The Kish count's own bootstrap interval over groups, in rollouts. This is the shape factor's
    #: uncertainty and it is not the effective size's, which is why it has its own pair of fields.
    kish_ci_low: float = math.nan
    kish_ci_high: float = math.nan
    #: False when the reliability came from a decomposition with no positive variance in it at all,
    #: so every quantity on it is zero over zero. The same defect class `GaugeRR` guards against.
    determined: bool = True

    @property
    def shape_factor(self) -> float:
        """The Kish count as a fraction of K, in [1/K, 1]. A reward-distribution statistic.

        1.0 means every rollout in the group sits the same distance from the group mean, which is
        the two-point case. Gaussian rewards give `2/pi = 0.637`, uniform `0.75`, lognormal about
        `0.34`. It does not enter `n_eff` and it is not a property of the grader.
        """
        return self.kish / self.k_nominal if self.k_nominal > 0 else 0.0

    @property
    def shape_ci(self) -> tuple[float, float]:
        """The shape factor's own bootstrap interval, on the same scale as `shape_factor`."""
        if self.k_nominal <= 0:
            return (math.nan, math.nan)
        return (self.kish_ci_low / self.k_nominal, self.kish_ci_high / self.k_nominal)

    @property
    def has_interval(self) -> bool:
        """Whether this reading carries a real interval rather than a point repeated twice.

        Rung 0 does not: it fixes the reliability at 1 by assumption rather than estimating it, so
        there is nothing left to be uncertain about except the group sizes themselves. A reading
        whose interval is its own point value is saying that, not hiding it.
        """
        return math.isfinite(self.ci_low) and self.ci_high > self.ci_low

    @property
    def wasted(self) -> float:
        """Rollouts bought and not converted into an independent observation."""
        return max(0.0, self.k_nominal - self.n_eff)

    @property
    def bias(self) -> BiasStatement:
        return BIAS[self.rung]

    def says(self) -> str:
        """The sentence, with this reading's own numbers in it."""
        shape = (
            f"The group's Kish shape factor is {self.shape_factor:.2f}, which is how evenly it "
            f"spends its gradient and is a property of the reward distribution rather than of the "
            f"grader."
        )
        if self.rung == 0:
            return (
                f"Your effective group size is {self.n_eff:.1f} of {self.k_nominal:.0f}, which is "
                f"all of it, because rung 0 has one score per rollout and cannot see any grader "
                f"error at all. This is the ceiling, not a measurement of your grader. {shape} To "
                f"learn what the grader costs you, score some objects twice and re-run at rung 1 "
                f"or above."
            )
        interval = (
            f"; {self.ci_level:.0%} interval [{self.ci_low:.2f}, {self.ci_high:.2f}]"
            if self.has_interval
            else "; no interval, the reliability is treated as known"
        )
        return (
            f"Your effective group size is {self.n_eff:.1f}, not {self.k_nominal:.0f}. You are "
            f"paying for {self.wasted:.1f} rollouts of grader noise. (rung {self.rung}; "
            f"single-score reliability {self.reliability:.3f}{interval}) {shape}"
        )

    def render(self) -> str:
        lines = [self.says(), f"    universe: {self.universe}", f"    bias: {self.bias}"]
        if not self.determined:
            lines.append(
                "    undetermined: the decomposition this reliability came from has no positive "
                "variance in it, so the reading is zero over zero rather than a measurement."
            )
        if self.invisible_terms:
            lines.append(
                f"    not visible at this rung: {', '.join(self.invisible_terms)}. Every one of "
                f"them is counted as signal here, so this number is an upper bound."
            )
        return "\n".join(lines)


def ladder_transfer(cheap: EffectiveSize, expensive: EffectiveSize) -> Transfer:
    """The disagreement between two rungs on the same data, as a calibration transfer term.

    Two rungs that ran on one dataset give the cheap rung's transfer uncertainty for free, and
    nobody publishes it. This is the one call that records it.
    """
    return ladder_disagreement(
        cheap.n_eff,
        expensive.n_eff,
        from_level="working_method",
        to_level="reference_method",
        n=cheap.n_groups,
        method=(
            f"grader.effective_group_size, rung {cheap.rung} against rung {expensive.rung}, "
            f"same {cheap.n_groups} groups"
        ),
    )


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


def _single_score_reliability(g: GStudy) -> float:
    """`E rho^2` for one draw of every facet: the reliability of a single observed score."""
    return g.generalizability(**{f: 1 for f in g.facets})


def _mean_over_groups(
    per_group: np.ndarray,
    *,
    ci: float = 0.95,
    n_resamples: int = 2_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """A mean over groups with a bootstrap interval resampled at the group level.

    Returns `(point, ci_low, ci_high)`. With one group there is nothing to resample and the
    interval is `nan`, which is the honest answer rather than a zero-width one.
    """
    if per_group.size == 0:
        return 0.0, math.nan, math.nan
    point = float(np.mean(per_group))
    if per_group.size < 2:
        return point, math.nan, math.nan
    boot = bootstrap_ci(per_group, np.mean, n_resamples=n_resamples, ci=ci, seed=seed)
    return point, boot.ci_low, boot.ci_high


def _kish_over_groups(
    groups: Sequence[np.ndarray],
    *,
    ci: float = 0.95,
    n_resamples: int = 2_000,
    seed: int = 0,
) -> tuple[float, float, float, np.ndarray]:
    """Mean Kish count over groups, with a bootstrap interval resampled at the group level.

    This is the **shape factor's** uncertainty, not the effective size's. It is kept because the
    shape factor is reported beside the reading and a statistic without an interval is not a
    measurement, but it no longer enters `n_eff`.
    """
    per_group = np.array([group_effective_size(g) for g in groups], dtype=np.float64)
    point, lo, hi = _mean_over_groups(per_group, ci=ci, n_resamples=n_resamples, seed=seed)
    return point, lo, hi, per_group


def jackknife_reliability(design: ReplicationDesign) -> tuple[float, float]:
    """The single-score reliability and a standard error from leaving each rater out in turn.

    The rater facet is the one whose universe the reading generalises over, so its sampling
    uncertainty is the one that matters and it is not small: a coefficient estimated from eleven
    graders is a claim about the population those eleven were drawn from, and eleven is not many.
    Bootstrapping over objects gives a far tighter interval and answers a different and easier
    question, which is why both are worth having and only this one belongs on the rung-3 reading.

    Returns `(reliability, standard_error)`. The error is `nan` when there are fewer than three
    raters, because two pseudo-values give a standard error on one degree of freedom and that is a
    number with no information in it.
    """
    full = _single_score_reliability(design.fit())
    n = design.n_r
    if n < 3:
        return full, math.nan
    loo = []
    for i in range(n):
        sub = np.delete(np.asarray(design.scores), i, axis=1)
        loo.append(
            _single_score_reliability(
                ReplicationDesign(
                    scores=sub,
                    raters=tuple(r for j, r in enumerate(design.raters) if j != i),
                    object_label=design.object_label,
                    facet_labels=design.facet_labels,
                ).fit()
            )
        )
    pseudo = np.array([n * full - (n - 1) * v for v in loo], dtype=np.float64)
    return full, float(np.std(pseudo, ddof=1) / math.sqrt(n))


def effective_group_size(
    groups: GroupScores,
    design: ReplicationDesign | None = None,
    *,
    universe_note: str = "",
    ci: float = 0.95,
    n_resamples: int = 2_000,
    seed: int = 0,
    reliability_se: float | None = None,
) -> EffectiveSize:
    """The reading, at the highest rung the supplied data supports.

    With no design this is rung 0, the reliability factor is 1 by assumption, and `n_eff` is exactly
    K. With a design the rung follows from what the design contains, because a rung is a statement
    about what could be seen rather than a mode the caller selects: one rater and one extra facet is
    rung 1, one rater and two extra facets is rung 2, two or more raters is rung 3.

    ``reliability_se`` is what gives the reading an interval, by the delta method on `K x G`:
    `var(n_eff) = G^2 var(K) + K^2 var(G)`. Pass `jackknife_reliability(design)[1]`. Leaving it out
    leaves the reliability treated as known, and then the only uncertainty left is the spread of the
    group sizes themselves, which is zero whenever every group has the same K. **A reading whose
    interval is its own point value is telling you it has no estimated uncertainty, not hiding it**,
    and `method` says which case it is.

    The Kish shape factor is computed and returned on every reading, with its own bootstrap
    interval, and is not multiplied into `n_eff`. See the class docstring for why.
    """
    kish, kish_lo, kish_hi, _ = _kish_over_groups(
        groups.groups, ci=ci, n_resamples=n_resamples, seed=seed
    )
    sizes = np.array([float(g.size) for g in groups.groups], dtype=np.float64)
    k_nominal, k_lo, k_hi = _mean_over_groups(sizes, ci=ci, n_resamples=n_resamples, seed=seed)

    if design is None:
        return EffectiveSize(
            rung=0,
            grader=groups.grader,
            k_nominal=k_nominal,
            kish=kish,
            kish_ci_low=kish_lo,
            kish_ci_high=kish_hi,
            reliability=1.0,
            n_eff=k_nominal,
            ci_low=k_lo if math.isfinite(k_lo) else k_nominal,
            ci_high=k_hi if math.isfinite(k_hi) else k_nominal,
            ci_level=ci,
            n_groups=groups.n_groups,
            method=(
                "K with the reliability factor set to 1; no interval beyond the spread of the "
                "group sizes, because rung 0 assumes the reliability rather than estimating it"
            ),
            universe=universe_note or "none: the reliability factor is assumed to be 1",
            invisible_terms=("every error term",),
            baselines={
                "baseline.group_size_k": k_nominal,
                "baseline.single_rater_design": k_nominal,
            },
        )

    g = design.fit()
    determined = g.components.total > 0.0
    reliability = _single_score_reliability(g)
    n_eff = k_nominal * reliability
    rung, invisible = _rung_of(design)

    if reliability_se is not None and math.isfinite(reliability_se) and math.isfinite(k_lo):
        # Delta method on `K x G`. The group-size factor's standard error is read back off its own
        # bootstrap interval at the same level, so the two terms are combined at one confidence
        # level rather than one being a bootstrap quantile and the other a normal approximation.
        # On groups that are all the same size the first term is exactly zero and the interval is
        # the jackknife's alone, which is the usual case and the right one.
        half = (k_hi - k_lo) / 2.0
        se_k = half / 1.959963984540054 if ci == 0.95 else half
        var = (reliability**2) * se_k**2 + (k_nominal**2) * reliability_se**2
        widen = 1.959963984540054 * math.sqrt(max(0.0, var))
        ci_low, ci_high = max(0.0, n_eff - widen), min(k_nominal, n_eff + widen)
        method_note = "rater jackknife and group-size bootstrap, combined by the delta method"
    else:
        ci_low = k_lo * reliability if math.isfinite(k_lo) else n_eff
        ci_high = k_hi * reliability if math.isfinite(k_hi) else n_eff
        method_note = (
            "no interval on the reliability factor: it is treated as known. Pass "
            "reliability_se=jackknife_reliability(design)[1], which needs three or more raters"
        )

    return EffectiveSize(
        rung=rung,
        grader=groups.grader,
        k_nominal=k_nominal,
        kish=kish,
        kish_ci_low=kish_lo,
        kish_ci_high=kish_hi,
        reliability=reliability,
        reliability_se=reliability_se,
        n_eff=n_eff,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=ci,
        n_groups=groups.n_groups,
        method=f"K x generalizability ({g.design}); {method_note}",
        universe=universe_note
        or (
            "a fresh draw of every facet of the measurement: "
            + ", ".join(g.label(f) for f in g.facets)
        ),
        invisible_terms=invisible,
        gstudy=g,
        determined=determined,
        baselines={
            "baseline.group_size_k": k_nominal,
            # What a single-rater design with one score per object would have reported. It has no
            # replication, so it can identify no error component and its reliability is 1 by
            # construction: it answers K. Both catalogue baselines therefore answer K on a design
            # with one observation per cell, and that is the finding rather than a defect. Every
            # baseline available without replication says the group is worth all of its rollouts.
            "baseline.single_rater_design": k_nominal,
        },
    )


def _rung_of(design: ReplicationDesign) -> tuple[int, tuple[str, ...]]:
    """Which rung this design supports, and which error terms it still cannot see."""
    if design.n_r >= 2:
        return 3, ()
    if len(design.facet_levels) >= 2:
        return 2, ("sigma2(rater)", "sigma2(object x rater)")
    return 1, ("sigma2(rater)", "sigma2(object x rater)", "sigma2(rubric) and its interactions")


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


class EffectiveGroupSize(MetrologyInstrument):
    """A1. How many independent gradient-relevant observations a group of K rollouts gives you.

    The reading is `K x reliability`. **What it cannot do:** it cannot give you a number below K
    without replication, so rung 0 returns exactly K and says so. It cannot give a *per-grader*
    reading from a crossed panel either, because the generalizability coefficient of one draw from a
    panel of graders is a property of the panel: every member of a crossed design gets the same
    reading, and separating them needs repeated calls to each grader, which is an occasion facet.

    **This number changed on 2026-08-05.** It used to be `kish x reliability`, which charged
    measurement noise twice: the Kish count is computed on observed scores that already carry grader
    noise and the reliability factor then discounts for the same noise. On a two-point truth at
    reliability 0.4994 and K = 16 the group carries 7.99 independent observations and the old rule
    reported 5.59. On the eleven open reward models of series A, rung 3 moved from a mean of 1.910
    to 2.558 and rung 0 from 2.986 to 4.000. The Kish count is still reported, as
    `EffectiveSize.shape_factor` with its own interval, because it is a real statement about how
    evenly a group spends its gradient. It is not a property of the grader, which is why it is no
    longer multiplied into a grader's effective group size.

    Kill condition: if r0 and r3 agree within their intervals on five graders, the ladder is
    decoration and only r0 ships.
    """

    name = "EffectiveGroupSize"
    #: Bumped from 1.0 when `n_eff` stopped being `kish x reliability`. A stored row at version 1.0
    #: and a stored row at version 2.0 are not the same quantity and must not be pooled.
    version = "2.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A1"
    deviations = (
        "the catalogue names rung 0 'Kish ESS on observed score spread', and this reports K there "
        "instead. The Kish ESS on the observed spread is a statement about how evenly the group "
        "spends its gradient, which is a property of the reward distribution and the policy rather "
        "than of the grader; multiplying it into a grader's effective group size also charged "
        "grader noise twice, because the Kish count is computed on scores that already contain it. "
        "The Kish ESS is still computed on every reading and reported beside it as `shape_factor` "
        "with its own bootstrap interval, using the absolute centred scores |r_i - rbar| because a "
        "group drives learning through its contrasts",
        "the interval covers the reliability factor's rater-panel jackknife and the spread of the "
        "group sizes, and not the sampling uncertainty of the individual variance components. On a "
        "design with few objects the omitted term is not negligible and the interval is too narrow",
        "a crossed multi-rater design gives one reliability for the whole panel, so every grader in "
        "one design receives the same reading. That is the coefficient the design estimates: it is "
        "the reliability of a score from a grader drawn at random from this universe. A per-grader "
        "reading needs repeated calls to each grader, which is an occasion facet, and a design that "
        "has one does not need this caveat",
        "rung 1 as built is a crossed object-by-occasion design rather than a paired test-retest "
        "correlation. The two agree when there are exactly two occasions and the crossed form "
        "extends to more, so this is the more general estimator of the same thing",
    )

    quantity = "grader.effective_group_size"
    requires = EFFECTIVE_GROUP_SIZE_ACCESS
    substrates = frozenset(
        {
            Substrate.NEURAL_SCALAR,
            Substrate.NEURAL_GEN,
            Substrate.PROGRAM,
            Substrate.PROCEDURAL,
            Substrate.HUMAN,
            Substrate.COMPOSITE,
        }
    )
    phases = frozenset({Phase.PRE_RUN, Phase.POST_RUN})
    envelope = GROUP_ENVELOPE
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = EFFECTIVE_GROUP_SIZE_BASELINES
    rung = 0

    #: Fraction of degenerate groups above which the instrument refuses rather than averaging over
    #: groups that carry no contrast. Not a catalogue threshold: the catalogue leaves
    #: `envelope_measured_by` OPEN for A1, so this is the instrument's own declared floor and it is
    #: an argument rather than a constant so a caller can state a different one.
    max_degenerate_fraction: float = 0.5

    def __init__(
        self,
        groups: GroupScores | None = None,
        design: ReplicationDesign | None = None,
        *,
        ci: float = 0.95,
        n_resamples: int = 2_000,
        seed: int = 0,
        max_degenerate_fraction: float | None = None,
        universe_note: str = "",
        jackknife: bool = True,
    ) -> None:
        self.groups = groups
        self.design = design
        self.ci = float(ci)
        self.n_resamples = int(n_resamples)
        self.seed = int(seed)
        self.universe_note = universe_note
        self.jackknife = bool(jackknife)
        if max_degenerate_fraction is not None:
            self.max_degenerate_fraction = float(max_degenerate_fraction)
        self.rung = 0 if design is None else _rung_of(design)[0]

    def compute(self) -> Any:
        if self.groups is None or self.groups.n_groups == 0:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no recorded groups were supplied, so there is no group to size",
                remedy=(
                    "pass `groups=GroupScores.of(scores)` where `scores` is one row per prompt and "
                    "one column per rollout in that prompt's group. A record of the scores the run "
                    "already logged is enough for rung 0 and costs nothing to produce."
                ),
            )

        degenerate = self.groups.degenerate_fraction
        if degenerate > self.max_degenerate_fraction:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ENVELOPE_VIOLATED,
                detail=(
                    f"GROUP_NONDEGENERATE fails: {degenerate:.1%} of the "
                    f"{self.groups.n_groups} groups have no score spread at all, against a floor "
                    f"of {self.max_degenerate_fraction:.0%}. A group whose K rollouts all score the "
                    f"same has an effective size of zero, and averaging zeros with real numbers "
                    f"reports a mixture rather than a measurement"
                ),
                remedy=(
                    "restrict the window to prompts the policy has not saturated, or report the "
                    "degenerate fraction as the finding: a run where half the groups are flat is "
                    "spending half its compute on groups that produce no gradient at all, which is "
                    "a larger result than the effective size of the rest."
                ),
                statistics={
                    "degenerate_fraction": degenerate,
                    "n_groups": self.groups.n_groups,
                    "threshold": self.max_degenerate_fraction,
                },
            )

        if self.design is not None:
            bad = design_refusal(self.name, self.design)
            if bad is not None:
                return bad

        se = None
        if self.design is not None and self.jackknife:
            se = jackknife_reliability(self.design)[1]
        reading = effective_group_size(
            self.groups,
            self.design,
            universe_note=self.universe_note,
            ci=self.ci,
            n_resamples=self.n_resamples,
            seed=self.seed,
            reliability_se=se,
        )
        if not reading.determined:
            # An all-zero decomposition must not render as a gauge that resolves two billion
            # levels. The same input reaches here as a reliability of exactly 0.0 and an
            # effective group size of exactly 0.0, which reads as "your grader destroys every one
            # of your rollouts" when the truth is that nothing in the design varied at all.
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ENVELOPE_VIOLATED,
                detail=(
                    f"the supplied design has no variance in it: every component of the "
                    f"{reading.gstudy.design if reading.gstudy else 'crossed'} decomposition is "
                    f"zero, so the generalizability coefficient is zero over zero. An effective "
                    f"group size of 0.0 computed from it would be a measurement of nothing"
                ),
                remedy=(
                    "check that the scores passed into the design are the ones you meant. Every "
                    "cell holding the same value, a column accidentally filled with a constant, or "
                    "a gauge-fixing step that divided by a standard deviation of zero all produce "
                    "this. If the grader really is constant on these objects then its effective "
                    "group size is undefined rather than zero, and the finding is that the objects "
                    "do not discriminate."
                ),
                statistics={
                    "components_total": (
                        float(reading.gstudy.components.total) if reading.gstudy else 0.0
                    ),
                    "n_p": self.design.n_p if self.design is not None else 0,
                    "n_r": self.design.n_r if self.design is not None else 0,
                },
            )
        return reading

    def uncertainty(self, computed: EffectiveSize) -> Uncertainty | None:
        """The interval on the Evidence rather than only in the payload.

        `n_effective` is filled with the reading itself, which is what that field means: this
        instrument's entire job is to say how many independent observations the group is worth.
        """
        return Uncertainty(
            ci_low=computed.ci_low,
            ci_high=computed.ci_high,
            ci_level=computed.ci_level,
            n=computed.n_groups,
            n_effective=computed.n_eff,
            method=computed.method,
        )

    def payload(self, computed: EffectiveSize) -> dict[str, Any]:
        shape_low, shape_high = computed.shape_ci
        out: dict[str, Any] = {
            "n_eff": computed.n_eff,
            "k_nominal": computed.k_nominal,
            # The shape factor travels beside the reading as its own statistic with its own
            # interval. It is not a grader property and it is not a factor of `n_eff`.
            "shape_factor": computed.shape_factor,
            "shape_factor_ci_low": shape_low,
            "shape_factor_ci_high": shape_high,
            "kish": computed.kish,
            "reliability": computed.reliability,
            "reliability_se": computed.reliability_se,
            "determined": computed.determined,
            "has_interval": computed.has_interval,
            "rung": computed.rung,
            "grader": computed.grader,
            "n_groups": computed.n_groups,
            "ci_low": computed.ci_low,
            "ci_high": computed.ci_high,
            "ci_level": computed.ci_level,
            "method": computed.method,
            "universe": computed.universe,
            "invisible_terms": list(computed.invisible_terms),
            "bias_direction": computed.bias.direction,
            "baselines": dict(computed.baselines),
            "says": computed.says(),
        }
        if computed.gstudy is not None:
            out["components"] = computed.gstudy.components.as_dict()
            out["components_truncated"] = list(computed.gstudy.components.truncated_names)
        return out


def design_refusal(instrument: str, design: ReplicationDesign) -> Refusal | None:
    """The refusals a supplied crossed design can earn, checked before any arithmetic."""
    if max(design.facet_levels) < 2:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=(
                f"every measurement facet in this design has one level "
                f"({design.facet_levels}), so it carries no replication and no variance component "
                f"is identified"
            ),
            remedy=(
                "score each object at least twice, either with two grader draws (a different seed, "
                "a different rubric ordering, a different judge) or on two separate calls, and "
                "pass the result as `ReplicationDesign.from_long(values, objects, raters, "
                "occasions)`. Two levels of one facet is the minimum that makes a variance "
                "component exist. Without it, rung 0 is the honest answer and it is free."
            ),
            statistics={"n_p": design.n_p, "n_r": design.n_r, "n_o": design.n_o},
        )
    if design.n_p < 2:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=f"the design has {design.n_p} object(s); a variance across objects needs two",
            remedy="supply at least two distinct scored objects, and preferably thirty.",
            statistics={"n_p": design.n_p},
        )
    return None


def refuse_unbalanced(
    instrument: str,
    values: Sequence[float] | np.ndarray,
    objects: Sequence[Any],
    raters: Sequence[Any],
    occasions: Sequence[Any] | None = None,
) -> Refusal | ReplicationDesign:
    """Build a design, or return the refusal that says why the data is not one.

    The refusal an unbalanced design earns is the one this package cares most about getting right,
    because the alternative to refusing is a number that looks exactly like a good one. The method
    of moments subtracts mean squares whose expectations assume every cell is filled once; on a
    design with holes those expectations are different and the components come back biased with no
    outward sign.
    """
    factors: list[Sequence[Any]] = [list(objects), list(raters)]
    if occasions is not None:
        factors.append(list(occasions))
    report = check_balance(factors)
    if report.balanced:
        return ReplicationDesign.from_long(values, objects, raters, occasions)

    examples = ", ".join(str(c) for c in report.missing_examples[:3])
    have = "installed" if statsmodels_available() else "not installed"
    return Refusal(
        instrument=instrument,
        reason=RefusalReason.ENVELOPE_VIOLATED,
        detail=(
            f"the design is not fully crossed: {report.render()}"
            + (f". Empty cells include {examples}" if examples else "")
        ),
        remedy=(
            f"score the missing cells and re-run, or drop the objects and raters that cause the "
            f"holes and refit the balanced remainder, or fit a mixed model. statsmodels is {have}; "
            f"the mixed-model path needs `pip install 'statsmodels>=0.14'` and is not wired into "
            f"this instrument. Do not fill the holes with a mean: the method of moments used here "
            f"assumes one observation per cell and returns a biased number that is "
            f"indistinguishable from an unbiased one."
        ),
        statistics={
            "cells_expected": report.n_cells_expected,
            "cells_present": report.n_cells_present,
            "cells_missing": report.n_missing,
            "cells_replicated": report.replicated_cells,
            "statsmodels_available": statsmodels_available(),
        },
    )


# ---------------------------------------------------------------------------
# The ladder, as registry entries
# ---------------------------------------------------------------------------


def register_ladder() -> list[str]:
    """Register A1's four rungs as `EstimatorEntry` rows. Not called at import, by design.

    The catalogue is loaded by whoever wants it, so an import that registered estimators would
    depend on load order. A test calls this after `load_quantities()` and asserts the ladder is
    four rungs deep with the bias directions the catalogue prints.
    """
    entries = [
        EstimatorEntry(
            quantity=EFFECTIVE_GROUP_SIZE.id,
            impl="a1.kish_observed_spread",
            requires={Component.GRADER: Access.RECORD},
            envelope=GROUP_ENVELOPE,
            rung=0,
            bias=BIAS[0],
            cost=FREE,
        ),
        EstimatorEntry(
            quantity=EFFECTIVE_GROUP_SIZE.id,
            impl="a1.test_retest",
            requires={Component.GRADER: Access.REPLICATE},
            envelope=GROUP_ENVELOPE,
            rung=1,
            bias=BIAS[1],
            cost=CostModel(note="R calls per object"),
        ),
        EstimatorEntry(
            quantity=EFFECTIVE_GROUP_SIZE.id,
            impl="a1.crossed_gstudy",
            requires={Component.GRADER: Access.REPLICATE},
            envelope=GROUP_ENVELOPE,
            rung=2,
            bias=BIAS[2],
            cost=CostModel(note="n*r*o calls"),
        ),
        EstimatorEntry(
            quantity=EFFECTIVE_GROUP_SIZE.id,
            impl="a1.crossed_gstudy_multirater",
            requires={Component.GRADER: Access.REPLICATE},
            envelope=GROUP_ENVELOPE,
            rung=3,
            bias=BIAS[3],
            cost=CostModel(note="n*r*o calls, r >= 2"),
        ),
    ]
    for e in entries:
        register_estimator(e)
    return [e.impl for e in entries]


__all__ = [
    "BIAS",
    "EFFECTIVE_GROUP_SIZE",
    "EFFECTIVE_GROUP_SIZE_ACCESS",
    "EFFECTIVE_GROUP_SIZE_BASELINES",
    "GROUP_ENVELOPE",
    "EffectiveGroupSize",
    "EffectiveSize",
    "GroupScores",
    "MetrologyInstrument",
    "ReplicationDesign",
    "design_refusal",
    "effective_group_size",
    "jackknife_reliability",
    "ladder_transfer",
    "refuse_unbalanced",
    "register_ladder",
]

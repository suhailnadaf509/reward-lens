"""M8, the interlaboratory comparison: two or more labs, one measurand, do they agree.

Metrology settled this in the 1930s and machine learning has not started. A laboratory reports a
value and an uncertainty; a comparison pools them, reports the between-laboratory standard
deviation `s_L`, and reports the **Birge ratio**, which is the observed dispersion over the
dispersion the laboratories' own stated uncertainties predict. A Birge ratio near 1 means the labs
understand their own errors. A Birge ratio of 10 means they do not, and no amount of averaging their
numbers fixes it.

The reward-model literature is full of exactly the second case and reports the first. Every
leaderboard is an interlaboratory comparison whose participants publish a number and a standard
error over items, and whose between-participant spread is an order of magnitude larger than those
standard errors. That is a measurable fact about the field and it is what this instrument measures.

**A comparison with no matched control is refused, and the reason is not pedantry.** Agreement and
disagreement are both uninterpretable without knowing what the statistic does when the labs
provably *do* share a measurand. A Birge ratio of 1.4 could be a panel that mildly disagrees or a
statistic that runs at 1.4 under the null on this data at this `k`. So M8 takes a control panel of
the same size, at the same `n`, measuring a measurand its members share by construction, and reports
both. `NO_MATCHED_CONTROL` exists for exactly this and it is the one refusal this instrument is most
likely to return.

**The quantity is `study.tau2`, and that is a decision.** The catalogue leaves M8's quantity list
`OPEN`. An interlaboratory comparison is a random-effects meta-analysis whose studies are
laboratories: `s_L` is the square root of the between-study variance and the Q statistic behind the
Birge ratio is Cochran's Q. `study.tau2`'s registered definition already says "the between-study
variance of a random-effects meta-analysis, by DerSimonian-Laird, Paule-Mandel or restricted maximum
likelihood, reported with a Q-profile interval", which is what this computes, and that row's
`instrument` field is `OPEN`, so naming M8 in it adds an instrument to a quantity that has none. The
argument is in `meta/quantities.py` and the proposed amendment is in `as_yaml_rows`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import (
    BaselineID,
    BiasStatement,
    CostModel,
    EstimatorEntry,
    register_estimator,
)
from reward_lens.core.reading import Refusal, RefusalReason, refuse_incomplete
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.meta._base import MetaInstrument
from reward_lens.measure.rate.regime import MEASURED_BY
from reward_lens.stats.meta import random_effects

#: Reading each laboratory's reported value and uncertainty. The comparison itself computes nothing
#: from the substrate, which is why an auditor holding only the published numbers can run it.
INTERLAB_ACCESS: dict[Component, Access] = {Component.RECORD: Access.RECORD}

#: The two things a comparison is claiming not to be. The first is the unweighted mean anybody can
#: take; the second is the dispersion the labs' own error bars predict, which is what the Birge
#: ratio divides by and what a leaderboard implicitly asserts is the whole story.
INTERLAB_BASELINES: tuple[BaselineID, ...] = (
    "baseline.unweighted_mean",
    "baseline.stated_within_lab_dispersion",
)

#: Labs compared across a window in which the measurand itself moved are not measuring one thing.
#: `STATIONARY_GRADER` is the kernel's name for that condition and it is what makes a between-lab
#: spread attributable to the labs.
INTERLAB_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
    measured_by={RegimeCondition.STATIONARY_GRADER: MEASURED_BY[RegimeCondition.STATIONARY_GRADER]},
    on_violation="refuse",
)

#: Below this many laboratories a between-laboratory variance is an artefact of the estimator.
#: `stats.meta.random_effects` enforces its own floor and this constant documents it here.
MIN_LABS = 3

#: How far the control panel's per-laboratory `n` may differ from the real panel's and still count
#: as identically powered. A control run at a tenth of the sample size answers a different question.
CONTROL_N_TOLERANCE = 0.2


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Lab:
    """One laboratory's reported value, its own stated standard uncertainty, and how it got there.

    ``u`` is a **standard** uncertainty, not an interval half-width. A lab reporting a 95% interval
    divides by 1.96 before it gets here, and `from_interval` does that so nobody does it by eye.
    """

    id: str
    value: float
    u: float
    n: int | None = None
    method: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.u < 0:
            raise ValueError(f"lab {self.id!r} reports a negative standard uncertainty ({self.u})")

    @classmethod
    def from_interval(
        cls,
        id: str,
        value: float,
        ci_low: float,
        ci_high: float,
        *,
        level: float = 0.95,
        n: int | None = None,
        method: str = "",
    ) -> "Lab":
        """A lab that published an interval rather than a standard uncertainty."""
        if level != 0.95:
            raise ValueError(
                f"only a 95% interval converts by the fixed factor 1.96; got level {level}. "
                f"Divide by the coverage factor the interval was built with and pass `u` directly."
            )
        return cls(
            id=id, value=value, u=(ci_high - ci_low) / (2 * 1.959963984540054), n=n, method=method
        )

    @property
    def variance(self) -> float:
        return self.u**2

    @property
    def expanded(self) -> float:
        """`U = 2u`, the conventional 95% half-width, which is what an `E_n` number is built on."""
        return 2.0 * self.u


@dataclass(frozen=True)
class ControlPanel:
    """Laboratories that share a measurand by construction. The null the real panel is read against.

    ``how`` has to say what makes the members share a measurand, because that is the entire content
    of the control. "Eleven bootstrap resamples of one grader's per-item outcomes at n = 1,763" is a
    control. "Eleven similar models" is not.
    """

    labs: tuple[Lab, ...]
    how: str
    measurand: str = ""

    def __post_init__(self) -> None:
        if not self.how.strip():
            raise ValueError(
                "a control panel has to say what makes its members share a measurand. A control "
                "whose construction is not stated cannot be checked for being matched."
            )

    @property
    def k(self) -> int:
        return len(self.labs)


def bootstrap_control(
    outcomes: Sequence[float] | np.ndarray,
    *,
    k: int,
    seed: int = 0,
    measurand: str = "",
    lab_prefix: str = "control",
) -> ControlPanel:
    """Build a matched control panel by resampling one laboratory's own per-item outcomes.

    Every member measures the same thing, because every member *is* the same laboratory reading a
    resample of the same items, at the same `n`, through the same estimator. The between-member
    spread is therefore sampling and nothing else, which is what a Birge ratio of 1 is supposed to
    mean.

    What this control does not contain is the item set's own idiosyncrasy: bootstrap members share
    the empirical distribution they are drawn from, so the panel measures sampling variation at this
    `n` on this bank rather than variation across banks. Said here rather than left for a reader to
    work out, because a control's limitations are the part that decides what the comparison licenses.
    """
    y = np.asarray(outcomes, dtype=np.float64).ravel()
    n = int(y.size)
    if n < 2:
        raise ValueError(f"a bootstrap control needs at least two items; got {n}")
    if k < 2:
        raise ValueError(f"a control panel needs at least two members; got {k}")
    rng = np.random.default_rng(seed)
    labs = []
    for i in range(int(k)):
        draw = y[rng.integers(0, n, n)]
        value = float(draw.mean())
        labs.append(
            Lab(
                id=f"{lab_prefix}-{i:02d}",
                value=value,
                u=float(draw.std(ddof=1) / math.sqrt(n)),
                n=n,
                method="bootstrap resample of one laboratory's per-item outcomes",
            )
        )
    return ControlPanel(
        labs=tuple(labs),
        how=(
            f"{k} bootstrap resamples of one laboratory's own per-item outcomes at n = {n:,}. "
            f"Every member measures that laboratory's value by construction, so the spread between "
            f"members is sampling variation at this n and nothing else"
        ),
        measurand=measurand,
    )


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class Interlaboratory:
    """The consensus, the between-laboratory spread, the Birge ratio, and the control's numbers."""

    measurand: str
    k: int
    lab_ids: tuple[str, ...]
    consensus: float
    consensus_ci: tuple[float, float]
    prediction_interval: tuple[float, float]
    s_l: float
    tau2: float
    tau2_ci: tuple[float, float]
    tau2_method: str
    birge: float
    q: float
    q_df: int
    q_p: float
    i2: float
    typical_within_u: float
    en_numbers: Mapping[str, float]
    outliers: tuple[str, ...]
    control_how: str
    control_k: int
    control_s_l: float
    control_birge: float
    baselines: Mapping[str, float] = field(default_factory=dict)

    @property
    def excess_dispersion(self) -> float:
        """How many times larger the observed spread is than the control's. The headline ratio."""
        return self.birge / self.control_birge if self.control_birge > 0 else float("inf")

    @property
    def labs_understand_their_errors(self) -> bool:
        """Whether the observed dispersion is consistent with the stated uncertainties."""
        return self.q_p >= 0.05

    def says(self) -> str:
        verdict = (
            "consistent with their own stated uncertainties"
            if self.labs_understand_their_errors
            else (
                f"{self.birge:.1f} times what their own stated uncertainties predict, against "
                f"{self.control_birge:.2f} on a control panel of {self.control_k} laboratories that "
                f"share a measurand by construction"
            )
        )
        return (
            f"{self.k} laboratories on {self.measurand or 'one measurand'}: consensus "
            f"{self.consensus:.4g} [{self.consensus_ci[0]:.4g}, {self.consensus_ci[1]:.4g}], "
            f"between-laboratory standard deviation s_L = {self.s_l:.4g} against a typical "
            f"within-laboratory uncertainty of {self.typical_within_u:.4g}. The dispersion is "
            f"{verdict}. A new laboratory would land in "
            f"[{self.prediction_interval[0]:.4g}, {self.prediction_interval[1]:.4g}]."
        )

    def render(self) -> str:
        lines = [self.says(), f"    control: {self.control_how}"]
        if self.outliers:
            lines.append(
                f"    |E_n| > 1 for {', '.join(self.outliers)}: those laboratories disagree with "
                f"the consensus by more than their own expanded uncertainties allow."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


class InterlaboratoryComparison(MetaInstrument):
    """M8. Two or more laboratories, one measurand, and what their agreement is worth.

    Refuses without a matched control. That is the design and not a gap: the Birge ratio and `s_L`
    are both uninterpretable without knowing what they do when the labs provably share a measurand.
    """

    name = "InterlaboratoryComparison"
    version = "1.0"
    quantity = "study.tau2"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    requires = INTERLAB_ACCESS
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
    phases = frozenset({Phase.PRE_RUN, Phase.POST_RUN, Phase.DEPLOYED})
    envelope = INTERLAB_ENVELOPE
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = INTERLAB_BASELINES
    rung = 0
    faithful_to = "M8"
    deviations = (
        "the catalogue's example laboratories are configurations of one model (HuggingFace against "
        "vLLM, eager against compiled, two seeds, two precisions) and the panel this build has is "
        "eleven different reward models on one bank. Both are interlaboratory comparisons in the "
        "metrological sense of independent methods on one measurand, and the second is a weaker "
        "claim about the same statistic: a panel of different models will disagree for reasons a "
        "panel of one model's configurations would not, so `s_L` here is an upper bound on the "
        "configuration-level number the catalogue describes",
        "the quantity is `study.tau2`, decided here because the catalogue leaves M8's quantity "
        "list OPEN. `s_L` is reported as its square root on every reading so the headline number is "
        "in the measurand's own units rather than their square",
    )

    def __init__(
        self,
        labs: Sequence[Lab] = (),
        control: ControlPanel | None = None,
        *,
        measurand: str = "",
        tau2_method: str = "PM",
        n_tolerance: float = CONTROL_N_TOLERANCE,
    ) -> None:
        self.labs = tuple(labs)
        self.control = control
        self.measurand = measurand
        self.tau2_method = tau2_method
        self.n_tolerance = float(n_tolerance)

    def compute(self) -> Any:
        labs = self.labs
        if len(labs) < 2:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    f"{len(labs)} laboratory(ies) supplied. A comparison needs at least two, and a "
                    f"between-laboratory variance needs at least {MIN_LABS}"
                ),
                remedy=(
                    f"collect at least {MIN_LABS} independent measurements of the same measurand, "
                    f"each with its own stated standard uncertainty, and pass them as "
                    f"`Lab(id=..., value=..., u=...)`. Published leaderboard rows already are this, "
                    f"once each row's standard error over items is recovered."
                ),
                statistics={"k": len(labs)},
            )
        zero_u = [lab.id for lab in labs if lab.u <= 0]
        if zero_u:
            return refuse_incomplete(
                self.name,
                field="a stated standard uncertainty",
                subject=f"{len(zero_u)} of {len(labs)} laboratories ({', '.join(zero_u[:3])})",
                remedy=(
                    "recover each laboratory's own uncertainty before comparing them. For a score "
                    "over items that is the standard error over items, which the laboratory can "
                    "compute from data it already has. Weighting a laboratory that states no "
                    "uncertainty is not a comparison, it is an unweighted mean with extra steps, "
                    "and the Birge ratio it produces divides by a number nobody measured."
                ),
                labs_without_u=len(zero_u),
            )

        control = self.control
        if control is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.NO_MATCHED_CONTROL,
                detail=(
                    f"{len(labs)} laboratories were supplied and no control panel was. Agreement "
                    f"and disagreement are both uninterpretable without knowing what the same "
                    f"statistic does on laboratories that share a measurand by construction: a "
                    f"Birge ratio of 1.4 could be a panel that mildly disagrees or a statistic that "
                    f"runs at 1.4 under the null at this k"
                ),
                remedy=(
                    "supply a control panel of the same size at the same n whose members share a "
                    "measurand by construction. `bootstrap_control(outcomes, k=...)` builds one "
                    "from any single laboratory's per-item outcomes and costs nothing, because it "
                    "resamples data you already hold."
                ),
                statistics={"k": len(labs), "control": None},
            )
        mismatch = self._control_mismatch(control)
        if mismatch:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.NO_MATCHED_CONTROL,
                detail=mismatch,
                remedy=(
                    "match the control to the panel: the same number of laboratories, at the same "
                    "per-laboratory sample size, through the same estimator. A control at a "
                    "different size answers a different question, and the difference between the "
                    "two Birge ratios then contains the mismatch rather than the finding."
                ),
                statistics={
                    "k": len(labs),
                    "control_k": control.k,
                    "n": [lab.n for lab in labs],
                    "control_n": [lab.n for lab in control.labs],
                },
            )

        fit = random_effects(
            [lab.value for lab in labs],
            [lab.variance for lab in labs],
            labels=[lab.id for lab in labs],
            tau2_method=self.tau2_method,  # type: ignore[arg-type]
            instrument=self.name,
        )
        if isinstance(fit, Refusal):
            return fit
        control_fit = random_effects(
            [lab.value for lab in control.labs],
            [lab.variance for lab in control.labs],
            labels=[lab.id for lab in control.labs],
            tau2_method=self.tau2_method,  # type: ignore[arg-type]
            instrument=f"{self.name}.control",
        )
        if isinstance(control_fit, Refusal):
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.NO_MATCHED_CONTROL,
                detail=(
                    f"the control panel could not be fitted: {control_fit.detail}. A control that "
                    f"cannot be analysed by the same route as the panel is not a control"
                ),
                remedy=control_fit.remedy,
                statistics=dict(control_fit.statistics),
            )

        birge = _birge(fit.het.q, fit.het.q_df)
        control_birge = _birge(control_fit.het.q, control_fit.het.q_df)
        en = {
            lab.id: (lab.value - fit.pooled) / math.hypot(lab.expanded, 2.0 * fit.se)
            for lab in labs
        }
        typical_u = math.sqrt(fit.het.typical_variance)
        values = np.array([lab.value for lab in labs], dtype=np.float64)
        return Interlaboratory(
            measurand=self.measurand,
            k=fit.k,
            lab_ids=tuple(lab.id for lab in labs),
            consensus=fit.pooled,
            consensus_ci=fit.ci,
            prediction_interval=fit.prediction,
            s_l=fit.het.tau,
            tau2=fit.het.tau2,
            tau2_ci=fit.het.tau2_ci,
            tau2_method=str(fit.het.tau2_method),
            birge=birge,
            q=fit.het.q,
            q_df=fit.het.q_df,
            q_p=fit.het.q_p,
            i2=fit.het.i2,
            typical_within_u=typical_u,
            en_numbers=en,
            outliers=tuple(sorted(k for k, v in en.items() if abs(v) > 1.0)),
            control_how=control.how,
            control_k=control_fit.k,
            control_s_l=control_fit.het.tau,
            control_birge=control_birge,
            baselines={
                "baseline.unweighted_mean": float(values.mean()),
                "baseline.stated_within_lab_dispersion": typical_u,
            },
        )

    def _control_mismatch(self, control: ControlPanel) -> str:
        """Whether the control is identically powered, as a sentence or as the empty string."""
        if control.k != len(self.labs):
            return (
                f"the control panel has {control.k} laboratories against the panel's "
                f"{len(self.labs)}. A control at a different k is not identically powered: Q has "
                f"different degrees of freedom and the Birge ratios are not comparable"
            )
        panel_n = [lab.n for lab in self.labs if lab.n]
        control_n = [lab.n for lab in control.labs if lab.n]
        if not panel_n or not control_n:
            return ""
        ratio = float(np.mean(control_n)) / float(np.mean(panel_n))
        if abs(ratio - 1.0) > self.n_tolerance:
            return (
                f"the control's mean per-laboratory sample size is {ratio:.2f} times the panel's, "
                f"outside the tolerance of {self.n_tolerance:.0%}. A control at a different sample "
                f"size measures sampling variation at that size, not at this one"
            )
        return ""

    def uncertainty(self, computed: Interlaboratory) -> Uncertainty | None:
        """The Q-profile interval on tau2, carried through to `s_L` by taking square roots.

        The interval is the reading here rather than an ornament: at k around ten, tau2's point
        estimate is worth very little on its own and Cochran's Q detects real heterogeneity well
        under half the time, so the pair is what carries the information.
        """
        lo, hi = computed.tau2_ci
        return Uncertainty(
            ci_low=math.sqrt(max(0.0, lo)),
            ci_high=math.sqrt(max(0.0, hi)),
            ci_level=0.95,
            n=computed.k,
            method=f"Q-profile interval on tau2 ({computed.tau2_method}), square-rooted to s_L",
        )

    def payload(self, computed: Interlaboratory) -> dict[str, Any]:
        return {
            "measurand": computed.measurand,
            "k": computed.k,
            "lab_ids": list(computed.lab_ids),
            "consensus": computed.consensus,
            "consensus_ci_low": computed.consensus_ci[0],
            "consensus_ci_high": computed.consensus_ci[1],
            "prediction_low": computed.prediction_interval[0],
            "prediction_high": computed.prediction_interval[1],
            "s_L": computed.s_l,
            "tau2": computed.tau2,
            "tau2_ci_low": computed.tau2_ci[0],
            "tau2_ci_high": computed.tau2_ci[1],
            "tau2_method": computed.tau2_method,
            "birge": computed.birge,
            "q": computed.q,
            "q_df": computed.q_df,
            "q_p": computed.q_p,
            "i2": computed.i2,
            "typical_within_u": computed.typical_within_u,
            "en_numbers": dict(computed.en_numbers),
            "outliers": list(computed.outliers),
            "control_how": computed.control_how,
            "control_k": computed.control_k,
            "control_s_L": computed.control_s_l,
            "control_birge": computed.control_birge,
            "excess_dispersion": computed.excess_dispersion,
            "labs_understand_their_errors": computed.labs_understand_their_errors,
            "baselines": dict(computed.baselines),
            "says": computed.says(),
        }


def _birge(q: float, df: int) -> float:
    """`sqrt(Q / (k - 1))`: observed dispersion over the dispersion the stated uncertainties predict.

    One at exactly the point where the laboratories' own error bars explain the spread. Above one,
    somebody's uncertainty is understated; below one, somebody's is overstated, which happens and is
    a finding rather than a rounding artefact.
    """
    return math.sqrt(q / df) if df > 0 else float("nan")


def register_ladder() -> list[str]:
    """Register M8's rungs on `study.tau2`. Not called at import, by design."""
    entries = [
        EstimatorEntry(
            quantity="study.tau2",
            impl="m8.paule_mandel_over_labs",
            requires=INTERLAB_ACCESS,
            envelope=INTERLAB_ENVELOPE,
            rung=0,
            bias=BiasStatement(
                direction="unknown",
                why=(
                    "Paule-Mandel is approximately unbiased for the between-study variance and its "
                    "sampling distribution at k around ten is wide and skewed, so the direction of "
                    "the error on any one panel is not knowable from the point estimate. The "
                    "Q-profile interval is what carries that, and it is reported"
                ),
            ),
            cost=CostModel(note="arithmetic on published values and uncertainties"),
            run=None,
        ),
        EstimatorEntry(
            quantity="substrate.noise_floor",
            impl="m8.between_engine_s_l",
            requires={Component.GRADER: Access.REPLICATE},
            envelope=INTERLAB_ENVELOPE,
            rung=1,
            bias=BiasStatement(
                direction="approximately_unbiased",
                why=(
                    "when the laboratories are configurations of one model rather than different "
                    "models, s_L is the configuration-level noise floor measured directly. "
                    "Specified and not built here: this build has no store carrying one model under "
                    "two engines over the same items"
                ),
            ),
            cost=CostModel(note="one scoring pass per configuration"),
            run=None,
        ),
    ]
    for e in entries:
        register_estimator(e)
    return [e.impl for e in entries]


__all__ = [
    "CONTROL_N_TOLERANCE",
    "INTERLAB_ACCESS",
    "INTERLAB_BASELINES",
    "INTERLAB_ENVELOPE",
    "MIN_LABS",
    "ControlPanel",
    "Interlaboratory",
    "InterlaboratoryComparison",
    "Lab",
    "bootstrap_control",
    "register_ladder",
]

"""A2, variance components and gauge R&R: what fraction of a score is which judge you drew.

Manufacturing has asked this question of every measuring instrument on every production line since
the 1960s. You take a set of parts, have several appraisers measure each one several times, and
decompose the spread into the parts, the appraisers, and the instrument's disagreement with itself.
Two numbers come out. `%GRR` is the measurement system's share of the total spread, and it has to be
under 30% for the gauge to be used. `ndc` is how many distinct levels of the thing the gauge can
actually separate, and it has to reach 5.

No language model benchmark reports either. The closest published number is a generalizability
coefficient of G = 0.000 with a confidence interval of [0.000, 0.752], which is the psychometric
twin of `%GRR` near 100%: a gauge whose entire reading is measurement.

**What the two headline numbers are.** `%GRR = 100 * sigma_GRR / sigma_total` and
`ndc = 1.41 * sigma_part / sigma_GRR`, both ratios of standard deviations rather than of variances.
Everything that is not the object of measurement is gauge, which is the general form of the AIAG
definition and is what lets the same function read a three-facet design where there are four more
terms and every one of them is still measurement.

**The finite-universe correction is the interesting rung and it is not a detail.** A facet has a
universe of levels and you sample some of them. Declaring the sample to *be* the universe, that is,
declaring the facet fixed, moves the object-by-facet interaction out of error and into universe
score. Brennan showed in 1992 that this raises reliability from .74 to .88 on one data set while
destroying any claim to generalise to new levels of the facet. That is the mathematics of benchmark
overfitting, published 34 years ago, and `fixed_facet_comparison` prints both numbers side by side
so the trade is visible in one call rather than being a choice made silently by whoever wrote the
evaluation harness.

Kill condition: if `ndc >= 5` on every grader tested, the gauge is adequate and the instrument is a
formality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import (
    BaselineID,
    BiasStatement,
    CostModel,
    EstimatorEntry,
    Quantity,
    Unit,
    register_estimator,
)
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.metrology.gstudy import (
    MetrologyInstrument,
    ReplicationDesign,
    design_refusal,
)
from reward_lens.measure.rate.regime import MEASURED_BY
from reward_lens.stats.gtheory import GStudy
from reward_lens.stats.variance import (
    GRR_MARGINAL,
    NDC_MINIMUM,
    ComponentSet,
    GaugeRR,
    gauge_rr,
)

# ---------------------------------------------------------------------------
# The three quantities, written out because the registry carries OPEN for all three
# ---------------------------------------------------------------------------

_VAR = Unit(dimension="variance", per=None, scale=None, as_printed="var")
_PCT = Unit(dimension="percent", per=None, scale=None, as_printed="%")
_COUNT = Unit(dimension="count", per=None, scale=None, as_printed="count")

VARIANCE_COMPONENTS = Quantity(
    id="grader.variance_components",
    definition=(
        "The decomposition of the variance of a single grader score into additive components, one "
        "per source of variation in a fully crossed design: the object of measurement, each facet "
        "of the measurement procedure, every interaction among them, and a residual. In a crossed "
        "object-by-rater-by-occasion design with one observation per cell the components are "
        "sigma2(p), sigma2(r), sigma2(o), sigma2(pr), sigma2(po), sigma2(ro) and sigma2(pro,e), "
        "estimated by inverting the expected mean squares. The three-way interaction and the "
        "residual are one term because a design with one observation per cell cannot separate "
        "them. Components are truncated at zero and every truncation is recorded, because a "
        "negative estimate means the true component is near zero and the design could not resolve "
        "it, which is a different statement from an established zero."
    ),
    unit=_VAR,
    invariance="reward.affine",
    interpretation=(
        "Read the shares, not the values: the values are in the square of whatever units the "
        "grader emits and are not comparable across graders. A large sigma2(r) share means the "
        "score depends on which judge you drew. A large sigma2(pr) share means the judges disagree "
        "about which response is better, which is the term a group-relative estimator cannot "
        "cancel."
    ),
    support=(0.0, math.inf),
    wedge=True,
)

GRR_PERCENT = Quantity(
    id="grader.grr_percent",
    definition=(
        "100 * sigma_GRR / sigma_total, where sigma_total is the standard deviation of a single "
        "score and sigma_GRR is the square root of the total variance minus the variance of the "
        "object of measurement. Everything that is not the object is gauge: facet main effects, "
        "object-by-facet interactions, facet-by-facet interactions and the residual. A ratio of "
        "standard deviations, following the automotive convention, and not of variances."
    ),
    unit=_PCT,
    invariance="reward.affine",
    interpretation=(
        "Under 10% the measurement system is acceptable, 10 to 30 is conditional on what the "
        "measurement is for, over 30 is not acceptable for process control. At 61% most of what "
        "you are ranking on is the measurement rather than the thing measured."
    ),
    support=(0.0, 100.0),
    wedge=True,
)

NDC = Quantity(
    id="grader.ndc",
    definition=(
        "1.41 * sigma_part / sigma_GRR, truncated to an integer: the number of distinct "
        "non-overlapping categories the measurement system can sort the population of objects "
        "into. The 1.41 is the square root of two, from the width of the distribution of true "
        "values relative to the width of the measurement error."
    ),
    unit=_COUNT,
    invariance="reward.affine",
    interpretation=(
        "Five or more is the threshold for using a gauge to control a process. Two means the gauge "
        "can tell good from bad and nothing finer, so it cannot resolve two adjacent models. One "
        "means it cannot do that either."
    ),
    support=(0.0, math.inf),
    wedge=True,
)

PROPOSED: tuple[Quantity, ...] = (VARIANCE_COMPONENTS, GRR_PERCENT, NDC)


# ---------------------------------------------------------------------------
# The envelope, and the condition the catalogue lost
# ---------------------------------------------------------------------------

#: A2's envelope in the source reads "fully crossed design; `MASK_STABLE`".
#: `spec/CATALOGUE.yaml` carries only `MASK_STABLE`: the merge read the `Env` column as a list of
#: `RegimeCondition` members and "fully crossed design" is not one, so it was dropped.
#:
#: It is not a cosmetic loss. The estimator in `stats.gtheory` inverts expected mean squares whose
#: expectations assume exactly one observation in every crossed cell. Run it on a design with holes
#: and it returns biased components with no outward sign, which is precisely the failure mode a
#: validity envelope exists to make loud. There is no `RegimeCondition` member for it, and adding
#: one is a specification amendment rather than an implementation detail, so this instrument
#: enforces it as a hard precondition that returns a `Refusal` and names the gap in `deviations`.
#: The proposed member is `RegimeCondition.DESIGN_CROSSED`, measured by `grader.design_balance`.
A2_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.MASK_STABLE}),
    measured_by={RegimeCondition.MASK_STABLE: MEASURED_BY[RegimeCondition.MASK_STABLE]},
    on_violation="refuse",
)

#: Catalogue A2 access: `GRADER: REPLICATE with facet control`. There is no separate flag for facet
#: control; `Access.REPLICATE` is documented as exactly the facet-varyable case, which is why it is
#: not on the containment ladder under `QUERY`.
GRR_ACCESS: dict[Component, Access] = {Component.GRADER: Access.REPLICATE}

#: Catalogue A2 prints one baseline: "a single-draw point estimate, to show what it hides". The
#: shipped `spec/CATALOGUE.yaml` split it at the comma into two entries. One baseline, and it is
#: computed on every reading.
GRR_BASELINES: tuple[BaselineID, ...] = ("baseline.single_draw_point_estimate",)

BIAS: Mapping[int, BiasStatement] = {
    0: BiasStatement(
        direction="unknown",
        why=(
            "a two-facet nested design confounds the nested facet with whatever it is nested in, "
            "so the split between the two is not identified and the direction of the error depends "
            "on which of them dominates"
        ),
    ),
    1: BiasStatement(
        direction="approximately_unbiased",
        why=(
            "the fully crossed design identifies every component by the method of moments, which "
            "is unbiased for the components themselves. %GRR and ndc are non-linear functions of "
            "them, so the plug-in estimates carry a small-sample bias that the degrees of freedom "
            "on each mean square bound"
        ),
    ),
    2: BiasStatement(
        direction="approximately_unbiased",
        why=(
            "adding the rubric as a third crossed facet moves rubric variance out of the residual, "
            "where rung 1 was counting it as call-to-call noise, and into a term of its own"
        ),
    ),
    3: BiasStatement(
        direction="unknown",
        why=(
            "the finite-universe correction is exact arithmetic on the components, so it adds no "
            "estimation bias. What it changes is the estimand: declaring a facet fixed narrows the "
            "claim to those levels, and reporting the fixed number as though it generalised is an "
            "error of interpretation rather than of estimation. Both numbers are always reported"
        ),
    ),
}


# ---------------------------------------------------------------------------
# The readings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedFacetComparison:
    """Brennan's 1992 result on one design: the same data under two universes of generalization."""

    facet: str
    random_generalizability: float
    fixed_generalizability: float
    random_relative_error: float
    fixed_relative_error: float
    random_universe_variance: float
    fixed_universe_variance: float
    n_levels: int

    @property
    def gain(self) -> float:
        return self.fixed_generalizability - self.random_generalizability

    @property
    def degenerate(self) -> bool:
        """Whether fixing this facet left no error at all, which makes the coefficient trivially 1.

        In a design with one facet and one observation per cell the only object-containing error
        term is the object-by-facet interaction, so fixing the facet moves all of it into universe
        score and the coefficient is exactly 1 by construction. That is arithmetic, not a finding,
        and a reading that does not say so invites someone to quote a reliability of 1.000.
        """
        return self.fixed_relative_error <= 0.0

    def says(self) -> str:
        head = (
            f"Declaring {self.facet!r} fixed at its {self.n_levels} observed level(s) moves "
            f"sigma2(object x {self.facet}) out of error and into universe score. Reliability goes "
            f"from {self.random_generalizability:.3f} to {self.fixed_generalizability:.3f}"
        )
        if self.degenerate:
            return (
                f"{head}, and it reaches 1.000 because this design has no other error term. That "
                f"is arithmetic rather than a measurement: with one facet and one observation per "
                f"cell there is nothing left to be unreliable about."
            )
        return (
            f"{head}. The higher number is a claim about these {self.n_levels} level(s) only and "
            f"says nothing about a new draw. The lower one is the number to publish if anyone will "
            f"ever use a different {self.facet}."
        )


@dataclass(frozen=True)
class GaugeStudy:
    """A2's full reading: the decomposition, the two gauge numbers, and the fixed-facet comparison."""

    grader_panel: tuple[str, ...]
    components: ComponentSet
    gauge: GaugeRR
    gstudy: GStudy
    rung: int
    #: The single-draw point estimate this instrument exists to argue with.
    baseline_single_draw: float
    fixed_facets: tuple[FixedFacetComparison, ...] = ()
    #: Present when the design has one observation per cell, which is most of them.
    repeatability_identified: bool = False
    notes: tuple[str, ...] = ()
    shares: Mapping[str, float] = field(default_factory=dict)

    @property
    def bias(self) -> BiasStatement:
        return BIAS[self.rung]

    def dominant_facet(self) -> tuple[str, float]:
        """The largest component that is not the object of measurement, and its share."""
        candidates = [(n, self.components.share(n)) for n in self.components.names if n != "p"]
        return max(candidates, key=lambda kv: kv[1]) if candidates else ("", 0.0)

    def says(self) -> str:
        name, share = self.dominant_facet()
        label = {
            "r": f"which {self.gstudy.label('r')} you drew",
            "o": f"which {self.gstudy.label('o')} you drew",
            "pr": f"{self.gstudy.label('r')} disagreement about which object is better",
            "po": f"{self.gstudy.label('o')} by object interaction",
            "ro": f"{self.gstudy.label('r')} by {self.gstudy.label('o')} interaction",
            "pr,e": f"{self.gstudy.label('r')} disagreement plus residual",
            "pro,e": "three-way interaction plus residual",
        }.get(name, name)
        return (
            f"{share:.0%} of your score variance is {label}. "
            f"%GRR = {self.gauge.grr_percent:.0f}%, ndc = {self.gauge.ndc_categories}. "
            f"{self.gauge.verdict().split('. ', 1)[-1]}"
        )

    def render(self) -> str:
        lines = [
            self.says(),
            self.components.render(),
            f"  panel: {', '.join(self.grader_panel) if self.grader_panel else '(unnamed)'}",
            f"  sigma_part = {self.gauge.sigma_part:.6g}, sigma_GRR = {self.gauge.sigma_grr:.6g}, "
            f"sigma_total = {self.gauge.sigma_total:.6g}",
        ]
        if not self.repeatability_identified:
            lines.append(
                "  repeatability and reproducibility are not separated: this design has one "
                "observation per cell, so there is no replication to estimate pure equipment "
                "variation from and the interaction carries it."
            )
        for f in self.fixed_facets:
            lines.append(f"  {f.says()}")
        lines.extend(f"  {n}" for n in self.notes)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


def fixed_facet_comparison(g: GStudy, facet: str, **sizes: int) -> FixedFacetComparison:
    """Both sides of Brennan's trade for one facet, computed from one G-study.

    The D-study sizes default to one level of every facet, which is the reliability of a single
    observed score and is the number an evaluation harness is implicitly quoting when it reports a
    benchmark result from one grader pass.
    """
    if facet not in g.facets:
        raise ValueError(
            f"{facet!r} is not a facet of this design; its facets are {g.facets}. The object of "
            f"measurement cannot be fixed."
        )
    at = {f: sizes.get(f, 1) for f in g.facets}
    fixed = g.declare_fixed(facet, universe_size=at[facet])
    return FixedFacetComparison(
        facet=g.label(facet),
        random_generalizability=g.generalizability(**at),
        fixed_generalizability=fixed.generalizability(**at),
        random_relative_error=g.relative_error(**at),
        fixed_relative_error=fixed.relative_error(**at),
        random_universe_variance=g.universe_variance(**at),
        fixed_universe_variance=fixed.universe_variance(**at),
        n_levels=int(at[facet]),
    )


def gauge_study(
    design: ReplicationDesign,
    *,
    fixed: Sequence[str] = (),
    universe_sizes: Mapping[str, float] | None = None,
) -> GaugeStudy:
    """Fit the crossed design and read it as a measurement system.

    ``fixed`` names facets to also report under a fixed universe. The random-universe numbers are
    always the headline: a reading whose universe was narrowed to make it look better, without
    saying so, is the thing this instrument exists to expose.
    """
    g = design.fit()
    if universe_sizes:
        for f, n in universe_sizes.items():
            g = g.declare_fixed(f, universe_size=n)
    comps = g.components
    identified = False  # one observation per cell everywhere in this module
    gauge = gauge_rr(comps, part="p", repeatability=None)

    single = float(np.var(np.asarray(design.scores, dtype=np.float64).ravel(), ddof=1))
    comparisons = tuple(fixed_facet_comparison(g, f) for f in fixed)

    notes = []
    if comps.any_truncated:
        notes.append(
            f"components truncated at zero: {', '.join(comps.truncated_names)}. Their shares are "
            f"reported as zero and their true values are near zero and unresolved by this design, "
            f"which is not the same as measured to be zero."
        )
    if design.n_r < 3:
        notes.append(
            f"the rater facet has {design.n_r} levels, so sigma2(rater) rests on "
            f"{design.n_r - 1} degree(s) of freedom. The point estimate is unbiased and its "
            f"sampling spread is wide."
        )
    return GaugeStudy(
        grader_panel=tuple(design.raters),
        components=comps,
        gauge=gauge,
        gstudy=g,
        rung=1 if not design.has_third_facet else 2,
        baseline_single_draw=single,
        fixed_facets=comparisons,
        repeatability_identified=identified,
        notes=tuple(notes),
        shares={n: comps.share(n) for n in comps.names},
    )


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


class VarianceComponents(MetrologyInstrument):
    """A2. Facet decomposition of `Var(r)`, with `%GRR` and `ndc`.

    Kill condition: if `ndc >= 5` on every grader tested, the gauge is adequate and the instrument
    is a formality.
    """

    name = "VarianceComponents"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A2"
    deviations = (
        "the catalogue's envelope for this instrument lost the qualifier 'fully crossed design' "
        "when the source's `Env` column was read as a list of RegimeCondition members. There "
        "is no member for it, so this instrument enforces the condition as a hard precondition "
        "returning a Refusal and proposes `RegimeCondition.DESIGN_CROSSED`, measured by a new "
        "`grader.design_balance`. Until that lands, the envelope declares MASK_STABLE only and the "
        "crossing requirement is enforced outside it",
        "repeatability and reproducibility are not reported separately. Every design this module "
        "fits has one observation per cell, which does not identify pure equipment variation, and "
        "reporting the object-by-rater interaction under the name 'repeatability' would be "
        "reporting a different quantity with a familiar label",
        "the catalogue's rung 0 is a two-facet nested design and is not implemented. A nested "
        "design identifies strictly less than the crossed one at the same cost in calls, so it is "
        "worth building only for a caller who already has nested data. Rung 1 is the entry point "
        "here and the ladder starts there",
        "`%GRR` is computed against total variation rather than against a tolerance. The tolerance "
        "form is the other automotive convention and there is no tolerance for a reward score",
    )

    quantity = "grader.variance_components"
    #: The other two quantities this instrument reports. `Instrument.quantity` is singular and A2
    #: registers three, so the extra two are declared here and the payload carries all three.
    also_reports: tuple[str, ...] = ("grader.grr_percent", "grader.ndc")
    requires = GRR_ACCESS
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
    phases = frozenset({Phase.PRE_RUN})
    envelope = A2_ENVELOPE
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = GRR_BASELINES
    rung = 1

    def __init__(
        self,
        design: ReplicationDesign | None = None,
        *,
        fixed: Sequence[str] = (),
        universe_sizes: Mapping[str, float] | None = None,
    ) -> None:
        self.design = design
        self.fixed = tuple(fixed)
        self.universe_sizes = dict(universe_sizes or {})
        if design is not None:
            self.rung = (
                3 if (self.fixed or self.universe_sizes) else (2 if design.has_third_facet else 1)
            )

    def compute(self) -> Any:
        if self.design is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no crossed design was supplied, so there is nothing to decompose",
                remedy=(
                    "score the same objects with two or more grader draws and pass "
                    "`design=ReplicationDesign.from_long(values, objects, raters)`. Two graders on "
                    "fifty shared items is the smallest thing that produces a %GRR, and it needs "
                    "no training run."
                ),
            )
        bad = design_refusal(self.name, self.design)
        if bad is not None:
            return bad
        if self.design.n_r < 2:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    f"the design has {self.design.n_r} rater level, so sigma2(rater) and "
                    f"sigma2(object x rater) are not identified and there is no reproducibility to "
                    f"report. %GRR computed here would be repeatability wearing the name of both"
                ),
                remedy=(
                    "add a second grader draw: a different judge, a different seed, a different "
                    "rubric ordering, or a different prompt template all count as a second level "
                    "of the rater facet. Reproducibility is a statement about disagreement and it "
                    "needs two things to disagree."
                ),
                statistics={"n_p": self.design.n_p, "n_r": self.design.n_r},
            )
        for f in self.fixed:
            if f not in ("r", "o"):
                return Refusal(
                    instrument=self.name,
                    reason=RefusalReason.ENVELOPE_VIOLATED,
                    detail=f"{f!r} is not a facet of this design; its facets are r and o",
                    remedy=(
                        "declare 'r' or 'o' fixed. The object of measurement cannot be fixed: "
                        "fixing it would set the universe-score variance to zero and the "
                        "coefficient would be undefined."
                    ),
                )
        return gauge_study(self.design, fixed=self.fixed, universe_sizes=self.universe_sizes)

    def payload(self, computed: GaugeStudy) -> dict[str, Any]:
        return {
            "components": computed.components.as_dict(),
            "components_raw": computed.components.raw_dict(),
            "components_truncated": list(computed.components.truncated_names),
            "shares": dict(computed.shares),
            "grr_percent": computed.gauge.grr_percent,
            "ndc": computed.gauge.ndc,
            "ndc_categories": computed.gauge.ndc_categories,
            "band": computed.gauge.band,
            "acceptable": computed.gauge.acceptable,
            "sigma_part": computed.gauge.sigma_part,
            "sigma_grr": computed.gauge.sigma_grr,
            "sigma_total": computed.gauge.sigma_total,
            "design": computed.gstudy.design,
            "levels": dict(computed.gstudy.levels),
            "panel": list(computed.grader_panel),
            "rung": computed.rung,
            "repeatability_identified": computed.repeatability_identified,
            "fixed_facets": [
                {
                    "facet": f.facet,
                    "n_levels": f.n_levels,
                    "random_generalizability": f.random_generalizability,
                    "fixed_generalizability": f.fixed_generalizability,
                    "gain": f.gain,
                    "degenerate": f.degenerate,
                }
                for f in computed.fixed_facets
            ],
            "baselines": {"baseline.single_draw_point_estimate": computed.baseline_single_draw},
            "says": computed.says(),
            "thresholds": {"grr_percent_max": GRR_MARGINAL, "ndc_min": NDC_MINIMUM},
        }


def register_ladder() -> list[str]:
    """Register A2's rungs as `EstimatorEntry` rows. Not called at import, by design.

    Rung 0, the two-facet nested design, is registered with no `run` callable, which is the
    catalogue's own way of saying a rung is specified and not built. The capability report prints
    it as such rather than hiding it.
    """
    entries = [
        EstimatorEntry(
            quantity=VARIANCE_COMPONENTS.id,
            impl="a2.nested_two_facet",
            requires={Component.GRADER: Access.REPLICATE},
            envelope=A2_ENVELOPE,
            rung=0,
            bias=BIAS[0],
            cost=CostModel(note="n*r calls, nested"),
            run=None,
        ),
        EstimatorEntry(
            quantity=VARIANCE_COMPONENTS.id,
            impl="a2.crossed_two_facet",
            requires={Component.GRADER: Access.REPLICATE},
            envelope=A2_ENVELOPE,
            rung=1,
            bias=BIAS[1],
            cost=CostModel(note="n*r calls, fully crossed"),
        ),
        EstimatorEntry(
            quantity=VARIANCE_COMPONENTS.id,
            impl="a2.crossed_three_facet",
            requires={Component.GRADER: Access.REPLICATE},
            envelope=A2_ENVELOPE,
            rung=2,
            bias=BIAS[2],
            cost=CostModel(note="n*r*o calls, fully crossed"),
        ),
        EstimatorEntry(
            quantity=VARIANCE_COMPONENTS.id,
            impl="a2.finite_universe",
            requires={Component.GRADER: Access.REPLICATE},
            envelope=A2_ENVELOPE,
            rung=3,
            bias=BIAS[3],
            cost=CostModel(note="no extra calls: arithmetic on rung 2's components"),
        ),
    ]
    for e in entries:
        register_estimator(e)
    return [e.impl for e in entries]


__all__ = [
    "A2_ENVELOPE",
    "BIAS",
    "GRR_ACCESS",
    "GRR_BASELINES",
    "GRR_PERCENT",
    "NDC",
    "PROPOSED",
    "VARIANCE_COMPONENTS",
    "FixedFacetComparison",
    "GaugeStudy",
    "VarianceComponents",
    "fixed_facet_comparison",
    "gauge_study",
    "register_ladder",
]

"""C6 rescue, C7 double dissociation, C5 acute versus chronic: the three controls series C needs.

None of these is a finding. All three are the arm of an experiment that could have contradicted it,
and the reason they are worth a module is that the field routinely runs the experiment without them.

**C6, rescue.** Interpretability's standard causal claim is a knockout with no rescue: remove X,
observe a change, conclude X mattered. A knockout confounds the loss of X with every downstream
consequence of the perturbation, and the fix is to put X back and check the behaviour returns. It
costs one extra forward pass over the ablation already run. The control is a norm-matched random
re-injection, which is norm-matched by construction rather than by arithmetic: the magnitude comes
from the recorded coordinate, so a random direction differs from the real one only in direction.

**C7, double dissociation.** Ablating A impairs behaviour 1 and not 2, while ablating B impairs 2
and not 1. A *single* dissociation is compatible with one graded resource plus a difficulty
difference: if behaviour 1 is simply harder, then any damage anywhere hurts 1 first, and "A is for
behaviour 1" is an unjustified reading of that. Shallice's critique, from neuropsychology, and the
reason a 2x2 is the smallest design that licenses the claim. Ablation tests necessity and steering
tests sufficiency, and papers routinely report one and claim the other.

**C5, acute versus chronic.** Ablate and keep training. "Ablating this direction drops the behaviour
40% immediately and 3% after 200 further steps" does not mean the ablation failed; it means the
capability was not localised there, it was *currently implemented* there. The songbird result is the
canonical demonstration: acute optogenetic inactivation of LMAN disrupts song and a chronic lesion of
the same nucleus does not. Almost nobody in interpretability runs the chronic version.

**C5 is compute-gated and this module does not run it.** Its access line is `POLICY: MUTATE +
CONTROL`, and `CONTROL` means standing up a counterfactual arm of the whole training loop, which is
a training run rather than a forward pass. The quantities are registered, the estimator is written
and testable against supplied readouts, and `AcuteChronic` with no chronic arm returns a refusal
naming what running it would cost rather than a number. That is the honest state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.types import Capability
from reward_lens.measure.base import Context
from reward_lens.measure.selection._common import (
    ABOVE_LOD_ONLY,
    ACCESS_POLICY_MUTATE,
    ACCESS_POLICY_MUTATE_CONTROL,
    SelectionInstrument,
    emit_white_box,
    refuse_unmeasured_control,
)

# ---------------------------------------------------------------------------
# C6 rescue
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class Rescue:
    """How much of an ablation's effect a re-injection restored, against a random control.

    ``fraction`` is `(rescued - ablated) / (clean - ablated)`: 1 means the behaviour came all the
    way back, 0 means the re-injection did nothing. It is not clipped. A fraction above 1 means the
    re-injection overshot and a negative one means it made things worse, and both are real outcomes
    that a clamp to [0, 1] would hide.
    """

    clean: float
    ablated: float
    rescued: float
    control_rescued: float
    fraction: float
    control_fraction: float
    spec: str = ""
    n_items: int = 0

    @property
    def is_off_manifold(self) -> bool:
        """Whether the random re-injection restores about as much as the real one.

        The diagnosis this control exists for. If pushing *any* vector of the right magnitude back
        in restores the behaviour, the ablation's effect was the perturbation itself rather than the
        loss of the direction, and the knockout established nothing about that direction.
        """
        return (
            np.isfinite(self.control_fraction)
            and np.isfinite(self.fraction)
            and self.control_fraction > 0.5 * self.fraction
        )

    def says(self) -> str:
        verdict = (
            "a norm-matched random re-injection restores about as much, so the ablation's effect "
            "was the perturbation rather than the loss of this direction"
            if self.is_off_manifold
            else "a norm-matched random re-injection does not, so the effect is not an "
            "off-manifold artifact"
        )
        return (
            f"re-injecting the ablated component restores {self.fraction:.1%} of the behaviour "
            f"against the random control's {self.control_fraction:.1%}: {verdict}."
        )

    def render(self) -> str:
        return "\n".join(
            [
                self.says(),
                f"  clean {self.clean:+.6g}  ablated {self.ablated:+.6g}  "
                f"rescued {self.rescued:+.6g}  control {self.control_rescued:+.6g}",
                f"  {self.spec}",
            ]
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "ablated": self.ablated,
            "rescued": self.rescued,
            "control_rescued": self.control_rescued,
            "fraction": self.fraction,
            "control_fraction": self.control_fraction,
            "is_off_manifold": self.is_off_manifold,
            "spec": self.spec,
            "n_items": self.n_items,
            "says": self.says(),
        }


def rescue_fraction(
    clean: float,
    ablated: float,
    rescued: float,
    control_rescued: float,
    *,
    spec: str = "",
    n_items: int = 0,
) -> Rescue | Refusal:
    """`(rescued - ablated) / (clean - ablated)`, refusing when the ablation did nothing.

    The denominator is the ablation's own effect, so a rescue fraction is undefined when the
    knockout did not move the behaviour: there is nothing to restore, and dividing by a number
    indistinguishable from zero produces an enormous fraction that reads as a spectacular rescue.
    That is the one failure mode of this statistic and it is refused rather than reported.
    """
    denom = float(clean) - float(ablated)
    if abs(denom) < 1e-12:
        return Refusal(
            instrument="rescue_fraction",
            reason=RefusalReason.BELOW_LOD,
            detail=(
                f"the ablation moved the behaviour by {denom:.4g}, so there is no effect to rescue "
                f"and the fraction would be a ratio with a vanishing denominator"
            ),
            remedy=(
                "establish that the ablation has an effect before measuring how much of it comes "
                "back. If the knockout is genuinely null, that is the finding and a rescue "
                "fraction is not defined on it."
            ),
            statistics={"clean": float(clean), "ablated": float(ablated)},
        )
    return Rescue(
        clean=float(clean),
        ablated=float(ablated),
        rescued=float(rescued),
        control_rescued=float(control_rescued),
        fraction=(float(rescued) - float(ablated)) / denom,
        control_fraction=(float(control_rescued) - float(ablated)) / denom,
        spec=spec,
        n_items=n_items,
    )


class RescueFraction(SelectionInstrument):
    """C6. Put the ablated component back and check the behaviour returns.

    The cheapest methodological upgrade in the catalogue: one extra forward pass over the ablation
    already run, and it closes the most obvious hole in a patching result. It is a control rather
    than a finding, so nothing kills it.

    What it cannot do. A within-pass re-injection at a later layer puts the coordinate into a
    residual stream the intervening layers have already written to under the ablated condition, so a
    fraction below 1 confounds "the direction was not sufficient" with "the computation above had
    already gone elsewhere". That is why the number is a fraction and not a verdict.
    """

    name = "RescueFraction"
    version = "1.0"
    quantity = "intervention.rescue_fraction"
    capabilities = Capability.ACTIVATIONS
    requires = ACCESS_POLICY_MUTATE
    envelope = ABOVE_LOD_ONLY
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = ("a norm-matched random re-injection",)
    rung = 0
    faithful_to = "C6, the rescue experiment"
    deviations = (
        "the rescue is within one forward pass, so it restores an activation-space ablation and "
        "cannot restore a weight edit",
        "restoring at the ablated site is close to a no-op and is the sanity check; restoring at a "
        "later site is the informative version. `Rescue.spec` records which was run",
    )

    def __init__(
        self,
        *,
        clean: float | None = None,
        ablated: float | None = None,
        rescued: float | None = None,
        control_rescued: float | None = None,
        spec: str = "",
        n_items: int = 0,
        incremental: Any = None,
        baseline_scores: Mapping[str, float] | None = None,
    ) -> None:
        self.clean = clean
        self.ablated = ablated
        self.rescued = rescued
        self.control_rescued = control_rescued
        self.spec = spec
        self.n_items = int(n_items)
        self._incremental = incremental
        self.baseline_scores = dict(baseline_scores or {})

    def compute(self) -> Any:
        if None in (self.clean, self.ablated, self.rescued):
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail="the clean, ablated and rescued arms are not all present",
                remedy=(
                    "run all three arms on the same items. "
                    "`interventions.rescue.knockout_and_rescue` builds the ablated and rescued "
                    "interventions sharing one recorder, and `policy.selection.behaviour_under` "
                    "scores each arm."
                ),
            )
        if self.control_rescued is None:
            return refuse_unmeasured_control(
                self.name,
                what=(
                    "the norm-matched random re-injection was not run, so a rescue that worked "
                    "cannot be told from a perturbation that any vector of the same magnitude "
                    "would have undone"
                ),
                remedy=(
                    "run the fourth arm with `interventions.rescue.norm_matched_random` as the "
                    "substitute direction. It is one more forward pass and it is the arm that "
                    "makes the other three mean something."
                ),
            )
        return rescue_fraction(
            self.clean,  # type: ignore[arg-type]
            self.ablated,  # type: ignore[arg-type]
            self.rescued,  # type: ignore[arg-type]
            self.control_rescued,
            spec=self.spec,
            n_items=self.n_items,
        )

    def estimate(self, ctx: Context | None = None) -> Reading:
        ctx = ctx or Context(readout="decision")
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        ctx._observable = self
        try:
            return self.measure(ctx)
        finally:
            ctx._observable = None

    def measure(self, ctx: Context) -> Any:
        computed = self.compute()
        if isinstance(computed, Refusal):
            return computed
        if self._incremental is None:
            return refuse_unmeasured_control(
                self.name,
                what=(
                    "this is a white-box reading and no IncrementalValidity record was supplied, "
                    "so nothing records what the intervention bought over the black-box bank"
                ),
                remedy=(
                    "run `stats.baselines.run_bank` on the same items and pass "
                    "`IncrementalValidityReading(...).compute().record` as `incremental=`."
                ),
            )
        return emit_white_box(
            ctx,
            computed,
            incremental=self._incremental,
            baselines=self.baseline_scores
            or {"rescue.norm_matched_random": float(computed.control_fraction)},
            uncertainty=Uncertainty(n=self.n_items, method="paired arms on one item set"),
            subject_extra={"spec": self.spec},
        )


# ---------------------------------------------------------------------------
# C7 double dissociation
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class Dissociation:
    """The 2x2 of two components against two behaviours, and whether it crosses over.

    ``impairment[(component, behaviour)]`` is how much ablating that component damaged that
    behaviour, as a magnitude. The double dissociation holds when each component damages *its own*
    behaviour materially more than the other's, in opposite directions. One of those two alone is a
    **single** dissociation and it is compatible with one graded resource plus a difficulty
    difference: if behaviour 1 is simply harder, damage anywhere hurts 1 first.

    ``interaction`` is the crossover size, `(d_A1 - d_A2) - (d_B1 - d_B2)`. It is the quantity the
    2x2 exists to estimate and it is what a single dissociation cannot produce.
    """

    component_a: str
    component_b: str
    behaviour_1: str
    behaviour_2: str
    d_a1: float
    d_a2: float
    d_b1: float
    d_b2: float
    margin: float = 0.05
    n_items: int = 0

    @property
    def a_prefers_1(self) -> bool:
        return self.d_a1 > self.d_a2 + self.margin

    @property
    def b_prefers_2(self) -> bool:
        return self.d_b2 > self.d_b1 + self.margin

    @property
    def interaction(self) -> float:
        return (self.d_a1 - self.d_a2) - (self.d_b1 - self.d_b2)

    @property
    def is_double(self) -> bool:
        """Both single dissociations, pointing opposite ways. The whole claim."""
        return self.a_prefers_1 and self.b_prefers_2

    @property
    def is_single(self) -> bool:
        """Exactly one of the two. Compatible with one graded resource and a difficulty difference."""
        return self.a_prefers_1 != self.b_prefers_2

    def says(self) -> str:
        if self.is_double:
            return (
                f"ablating {self.component_a} impairs {self.behaviour_1} "
                f"({self.d_a1:.4g}) and not {self.behaviour_2} ({self.d_a2:.4g}); ablating "
                f"{self.component_b} impairs {self.behaviour_2} ({self.d_b2:.4g}) and not "
                f"{self.behaviour_1} ({self.d_b1:.4g}). The functions are distinct, with a "
                f"crossover of {self.interaction:.4g}."
            )
        if self.is_single:
            worse = self.component_a if self.a_prefers_1 else self.component_b
            return (
                f"a single dissociation only: {worse} is selective and the other component is not. "
                f"That is compatible with one graded resource plus a difficulty difference between "
                f"{self.behaviour_1} and {self.behaviour_2}, so it does not license the claim that "
                f"the two functions are distinct."
            )
        return (
            f"no dissociation: neither component damages one behaviour materially more than the "
            f"other at a margin of {self.margin:.4g}. The crossover is {self.interaction:.4g}."
        )

    def render(self) -> str:
        return "\n".join(
            [
                self.says(),
                f"{'':<16}{self.behaviour_1:>14}{self.behaviour_2:>14}",
                f"  ablate {self.component_a:<8}{self.d_a1:>14.4g}{self.d_a2:>14.4g}",
                f"  ablate {self.component_b:<8}{self.d_b1:>14.4g}{self.d_b2:>14.4g}",
            ]
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "component_a": self.component_a,
            "component_b": self.component_b,
            "behaviour_1": self.behaviour_1,
            "behaviour_2": self.behaviour_2,
            "d_a1": self.d_a1,
            "d_a2": self.d_a2,
            "d_b1": self.d_b1,
            "d_b2": self.d_b2,
            "interaction": self.interaction,
            "is_double": self.is_double,
            "is_single": self.is_single,
            "margin": self.margin,
            "n_items": self.n_items,
            "says": self.says(),
        }


class DoubleDissociation(SelectionInstrument):
    """C7. Necessity is not sufficiency, and one dissociation is not two.

    The instrument refuses on a 2x2 that is not complete, because three of the four cells is a
    single dissociation wearing a 2x2's clothes and the missing cell is always the one that could
    have shown the effect was a difficulty difference.

    What it cannot do. Ablation tests necessity. A double dissociation says the two components are
    differently necessary for the two behaviours; it does not say either is sufficient, and steering
    is the experiment for that. Papers routinely report one and claim the other.
    """

    name = "DoubleDissociation"
    version = "1.0"
    quantity = "intervention.dissociation"
    capabilities = Capability.ACTIVATIONS
    requires = ACCESS_POLICY_MUTATE
    envelope = ABOVE_LOD_ONLY
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = (
        "a single dissociation, which is compatible with one graded resource plus a difficulty "
        "difference",
    )
    rung = 0
    faithful_to = "C7, Shallice's double dissociation"
    deviations = (
        "the crossover is compared against a fixed margin rather than tested. A paired interval on "
        "the interaction is the higher rung and it needs per-item deltas, which this reading takes "
        "as four summaries",
    )

    def __init__(
        self,
        *,
        impairment: Mapping[tuple[str, str], float] | None = None,
        components: tuple[str, str] = ("A", "B"),
        behaviours: tuple[str, str] = ("1", "2"),
        margin: float = 0.05,
        n_items: int = 0,
        incremental: Any = None,
        baseline_scores: Mapping[str, float] | None = None,
    ) -> None:
        self.impairment = dict(impairment or {})
        self.components = components
        self.behaviours = behaviours
        self.margin = float(margin)
        self.n_items = int(n_items)
        self._incremental = incremental
        self.baseline_scores = dict(baseline_scores or {})

    def compute(self) -> Any:
        a, b = self.components
        one, two = self.behaviours
        cells = [(a, one), (a, two), (b, one), (b, two)]
        missing = [c for c in cells if c not in self.impairment]
        if missing:
            return refuse_unmeasured_control(
                self.name,
                what=(
                    f"the 2x2 is incomplete: {len(missing)} of 4 cells were never measured "
                    f"({', '.join(f'{c}/{bh}' for c, bh in missing)})"
                ),
                remedy=(
                    "ablate each component and measure both behaviours under each, four "
                    "measurements on one item set. Three cells is a single dissociation, and the "
                    "cell people leave out is the one that would have shown the effect was a "
                    "difficulty difference rather than a functional distinction."
                ),
                measured=len(cells) - len(missing),
            )
        return Dissociation(
            component_a=a,
            component_b=b,
            behaviour_1=one,
            behaviour_2=two,
            d_a1=float(self.impairment[(a, one)]),
            d_a2=float(self.impairment[(a, two)]),
            d_b1=float(self.impairment[(b, one)]),
            d_b2=float(self.impairment[(b, two)]),
            margin=self.margin,
            n_items=self.n_items,
        )

    def estimate(self, ctx: Context | None = None) -> Reading:
        ctx = ctx or Context(readout="decision")
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        ctx._observable = self
        try:
            return self.measure(ctx)
        finally:
            ctx._observable = None

    def measure(self, ctx: Context) -> Any:
        computed = self.compute()
        if isinstance(computed, Refusal):
            return computed
        if self._incremental is None:
            return refuse_unmeasured_control(
                self.name,
                what="this is a white-box reading and no IncrementalValidity record was supplied",
                remedy=(
                    "run `stats.baselines.run_bank` on the same items and pass "
                    "`IncrementalValidityReading(...).compute().record` as `incremental=`."
                ),
            )
        return emit_white_box(
            ctx,
            computed,
            incremental=self._incremental,
            baselines=self.baseline_scores or {"single_dissociation": float(computed.d_a1)},
            uncertainty=Uncertainty(n=self.n_items, method="four arms on one item set"),
            subject_extra={"interaction": f"{computed.interaction:.6g}"},
        )


# ---------------------------------------------------------------------------
# C5 acute versus chronic, compute-gated
# ---------------------------------------------------------------------------

#: What the chronic arm costs, so a refusal can say it rather than gesture at it. A chronic arm is
#: the ablation held in place across continued training, plus a no-ablation arm trained the same
#: way, so it is two training runs of the stated length rather than two forward passes.
CHRONIC_COST_NOTE = (
    "the chronic arm is two continued-training runs of the stated length (the ablated arm and a "
    "no-ablation arm on the same schedule and seed), not two forward passes. That is why C5's "
    "access line is POLICY: MUTATE + CONTROL and why this instrument is registered and not run."
)


@register_payload
@dataclass(frozen=True)
class AcuteChronicReading:
    """The immediate effect of an ablation and the effect that survives continued training.

    ``recovery`` is `1 - chronic/acute`: how much of the immediate damage the model routed around.
    Near 1 means the capability was not localised in the ablated direction, it was *currently
    implemented* there, which is a different and more useful claim than either "it was there" or
    "it was not".
    """

    acute: float
    chronic: float
    steps: int
    control_acute: float | None = None
    control_chronic: float | None = None
    n_items: int = 0

    @property
    def recovery(self) -> float:
        return float(1.0 - self.chronic / self.acute) if self.acute else float("nan")

    @property
    def was_merely_current(self) -> bool:
        """Whether the model routed around most of the damage within the window."""
        return np.isfinite(self.recovery) and self.recovery > 0.5

    def says(self) -> str:
        return (
            f"ablating this direction drops the behaviour {abs(self.acute):.4g} immediately and "
            f"{abs(self.chronic):.4g} after {self.steps} further steps"
            + (
                ". The capability was not localised there; it was currently implemented there."
                if self.was_merely_current
                else ". The effect survives continued training, so the direction is carrying the "
                "capability rather than currently implementing it."
            )
        )

    def render(self) -> str:
        return "\n".join(
            [
                self.says(),
                f"  acute {self.acute:+.6g}  chronic {self.chronic:+.6g}  "
                f"recovered {self.recovery:.1%} over {self.steps} steps",
            ]
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "acute": self.acute,
            "chronic": self.chronic,
            "steps": self.steps,
            "recovery": self.recovery,
            "was_merely_current": self.was_merely_current,
            "control_acute": self.control_acute,
            "control_chronic": self.control_chronic,
            "n_items": self.n_items,
            "says": self.says(),
        }


class AcuteChronic(SelectionInstrument):
    """C5. Ablate and keep training. **Compute-gated: registered, written, and not run here.**

    Rung 0 is the acute effect, which is what everyone reports. Rung 1 is acute plus chronic, and
    the chronic arm needs `Access.CONTROL`: a counterfactual arm of the whole training loop, held
    under the ablation, beside a no-ablation arm on the same schedule and seed. That is two training
    runs, so this instrument computes on readouts a caller supplies and refuses when the chronic arm
    is absent rather than estimating it.

    The songbird result is the canonical demonstration: acute optogenetic inactivation of LMAN
    disrupts song, and a chronic lesion of the same nucleus does not. Almost nobody in
    interpretability runs the chronic version, which is why the acute-only rung is the one every
    published pulse experiment is at.

    Kill condition, from the catalogue: if acute and chronic agree everywhere, the distinction is
    not operative here. That is unanswerable until something runs the chronic arm.
    """

    name = "AcuteChronic"
    version = "1.0"
    quantity = "intervention.acute_effect"
    capabilities = Capability.ACTIVATIONS
    requires = ACCESS_POLICY_MUTATE_CONTROL
    envelope = ABOVE_LOD_ONLY
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = ("no-ablation arm", "a matched irrelevant ablation")
    #: `rung` is a property below rather than a class attribute: rung 0 is acute only and rung 1 is
    #: acute plus chronic, so which one this is depends on whether the chronic arm was supplied. A
    #: literal here would be shadowed by the property and the two would disagree.
    faithful_to = "C5, acute versus chronic intervention"
    deviations = (
        "the chronic arm is not run in this environment and the instrument refuses rather than "
        "extrapolating one. Its access line is POLICY: MUTATE + CONTROL and CONTROL is a training "
        "run",
        "recovery is measured against the acute effect on the same items, so it inherits whatever "
        "drift the no-ablation arm also experienced over the window. The no-ablation arm is a "
        "declared baseline for exactly that reason and is required for rung 1",
    )

    def __init__(
        self,
        *,
        acute: float | None = None,
        chronic: float | None = None,
        steps: int = 0,
        control_acute: float | None = None,
        control_chronic: float | None = None,
        n_items: int = 0,
        incremental: Any = None,
        baseline_scores: Mapping[str, float] | None = None,
    ) -> None:
        self.acute = acute
        self.chronic = chronic
        self.steps = int(steps)
        self.control_acute = control_acute
        self.control_chronic = control_chronic
        self.n_items = int(n_items)
        self._incremental = incremental
        self.baseline_scores = dict(baseline_scores or {})

    @property
    def rung(self) -> int:  # type: ignore[override]
        return 1 if self.chronic is not None else 0

    def compute(self) -> Any:
        if self.acute is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail="no acute effect was supplied, so there is nothing to compare a chronic one to",
                remedy=(
                    "measure the immediate effect of the ablation on the behavioural readout with "
                    "`policy.selection.behaviour_under`, and pass it as `acute=`."
                ),
            )
        if self.chronic is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    f"the acute effect is {self.acute:+.6g} and the chronic arm was not run, so "
                    f"this is the acute-only rung that every published pulse experiment is at. An "
                    f"acute effect on its own cannot distinguish a capability that lived in this "
                    f"direction from one that was merely implemented there at this checkpoint"
                ),
                remedy=(
                    "hold the ablation in place across continued training and re-measure, beside a "
                    "no-ablation arm on the same schedule and seed, then pass `chronic=` and "
                    "`steps=`. " + CHRONIC_COST_NOTE
                ),
                statistics={"acute": float(self.acute), "rung": 0},
            )
        return AcuteChronicReading(
            acute=float(self.acute),
            chronic=float(self.chronic),
            steps=self.steps,
            control_acute=self.control_acute,
            control_chronic=self.control_chronic,
            n_items=self.n_items,
        )

    def estimate(self, ctx: Context | None = None) -> Reading:
        ctx = ctx or Context(readout="decision")
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        ctx._observable = self
        try:
            return self.measure(ctx)
        finally:
            ctx._observable = None

    def measure(self, ctx: Context) -> Any:
        computed = self.compute()
        if isinstance(computed, Refusal):
            return computed
        if self.control_chronic is None:
            return refuse_unmeasured_control(
                self.name,
                what=(
                    "the no-ablation arm was not run over the same window, so a chronic effect "
                    "that shrank cannot be told from a behaviour that drifted for reasons the "
                    "ablation had nothing to do with"
                ),
                remedy=(
                    "train a no-ablation arm on the same schedule and seed and pass its readout as "
                    "`control_chronic=`. " + CHRONIC_COST_NOTE
                ),
            )
        if self._incremental is None:
            return refuse_unmeasured_control(
                self.name,
                what="this is a white-box reading and no IncrementalValidity record was supplied",
                remedy=(
                    "run `stats.baselines.run_bank` on the same items and pass "
                    "`IncrementalValidityReading(...).compute().record` as `incremental=`."
                ),
            )
        return emit_white_box(
            ctx,
            computed,
            incremental=self._incremental,
            baselines=self.baseline_scores or {"no_ablation_arm": float(self.control_chronic)},
            uncertainty=Uncertainty(n=self.n_items, method="paired arms over a training window"),
            subject_extra={"steps": str(self.steps)},
        )


__all__ = [
    "CHRONIC_COST_NOTE",
    "AcuteChronic",
    "AcuteChronicReading",
    "Dissociation",
    "DoubleDissociation",
    "Rescue",
    "RescueFraction",
    "rescue_fraction",
]

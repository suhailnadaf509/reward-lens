"""E5, clip and mask accounting: what the clip removes, propagated.

Clip fraction is logged by everything. What nobody propagates is the shrinkage the clip applied to
the update, and that shrinkage is the multiplier on every step whose gradient crossed the threshold.

**What this instrument used to claim, and why the claim moved.** The earlier version of this file
argued that veRL's `proxy1_signal_strength = gradient_norm**2` is the squared *post*-clipping norm,
so `proxy3_pure_noise = (1/(N-1)) * (proxy2 - proxy1)` subtracts a signal term that is too small and
the reported noise share is inflated by the clip. That chain does not hold, because the logged norm
is the norm **before** clipping in every framework in scope, and four primary sources say so:

- `torch.nn.utils.clip_grad_norm_` computes ``total_norm = _get_total_norm(grads, ...)``, then calls
  ``_clip_grads_with_norm_(parameters, max_norm, total_norm, ...)``, then returns ``total_norm``.
  Its own Returns clause is "Total norm of the parameter gradients", and it is taken before the
  scaling is applied.
- veRL's FSDP engine, `verl/workers/engine/fsdp/transformer_impl.py:735`, documents its
  `optimizer_step` return as "grad_norm (float): Norm of gradients before clipping", and the value
  it returns is exactly what `clip_grad_norm_` handed it.
- veRL's Megatron engine, `verl/workers/engine/megatron/transformer_impl.py:543`, says the same:
  "The norm of the gradients before clipping or update."
- `transformers/trainer.py:2523-2524`: ``def _clip_grad_norm(self, model)`` carries the docstring
  "Clip gradients to max_grad_norm. Returns the pre-clip gradient norm", and `trainer.py:2087` logs
  that value under the key `grad_norm`.

So `actor/grad_norm` is pre-clip, `proxy1` is the pre-clip signal strength, and the clip does not
bias veRL's noise share through it. E3 no longer subtracts a clip term and says why.

**What is left is real, and it is the thing the clip actually does.** The applied update is
``g * min(1, c / ||g||)``, so on a clipped step the realised step is shorter than the nominal one by

    shrinkage = clipped norm / unclipped norm = min(1, c / ||g||)

and any quantity that treats the update as proportional to the gradient (a first-order response, a
learning-rate sensitivity, a per-step effective step size) is wrong by that factor on exactly the
steps where the gradient was largest. That is what this instrument reports and propagates.

**Two numbers, and the record usually has only one of them.** `OptimizerTelemetry` carries
`grad_norm_clipped` and `grad_norm_unclipped`. Every framework in scope logs the pre-clip norm, so
the field a tap can fill from a log is `grad_norm_unclipped`, and the post-clip norm has to be taken
deliberately after the clip or computed as ``min(||g||, c)`` from the threshold. On the real GRPO
record this package was built against, TRL logs `grad_norm` and the tap writes it into
`grad_norm_clipped`, which is the wrong field for a pre-clip value; either way only one of the two
norms is present and this instrument returns the clip fraction as a bound rather than an effect.
That is the honest outcome and the remedy is one line at the tap.

**One field whose meaning this instrument does not decide.** `OptimizerTelemetry.clip_fraction` is
undocumented as to whether it counts importance ratios outside the PPO clip range or gradient norms
above the clipping threshold. Those are different truncations. It is reported as the clip fraction
and it is never used in any arithmetic here: every number below comes from the two gradient norms,
which are unambiguous. Where a framework key in `OptimizerTelemetry.extra` gives the ratio-clip
fraction directly, it is read and named, and reading `extra` is declared as a deviation because
`extra` is untyped by construction and no instrument may read it without saying so.

Kill condition, from the catalogue record: n/a.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BaselineID, BiasStatement
from reward_lens.core.reading import Refusal, RefusalReason, bounded_refusal
from reward_lens.core.types import Access, Capability, Component, GaugeStatus, Phase
from reward_lens.measure.estimator._base import EstimatorInstrument
from reward_lens.measure.estimator.spec import (
    ALL_SUBSTRATES,
    RECORD_ACCESS,
    iter_steps,
)
from reward_lens.record.schema import Group, Run, Step

#: The framework keys that carry a ratio-clip fraction, in the order they are tried. TRL 1.9.2 logs
#: the first three; `clip_ratio/region_mean` is the total and the other two are the low and high
#: sides. Everything here lands in `OptimizerTelemetry.extra`, which is untyped, so each key is
#: named on the reading rather than silently merged.
RATIO_CLIP_KEYS: tuple[str, ...] = (
    "clip_ratio/region_mean",
    "clip_ratio/low_mean",
    "clip_ratio/high_mean",
)

#: The two comparators. The first is what everybody already has and what the catalogue names as the
#: gap: a clip fraction with nothing propagated from it. The second is the assumption a card makes
#: when it quotes a gradient-derived number without mentioning the clip at all.
CLIP_BASELINES: tuple[BaselineID, ...] = (
    "baseline.clip_fraction_alone",
    "baseline.clip_has_no_effect",
)

CLIP_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "the effect is a ratio of two gradient norms the optimizer recorded on the same step. No "
        "property of the reward, the grader or the group structure can make that ratio wrong: what "
        "can go wrong is that one of the two was not recorded, and that is a refusal naming the "
        "field rather than a regime condition."
    ),
)

CLIP_BIAS = BiasStatement(
    direction="downward",
    why=(
        "the norms are recorded per optimizer step and the effect is averaged over steps, so a "
        "run whose clipping is concentrated in a few steps reports a mean that understates the "
        "worst of it. The per-step series is carried on the reading for exactly that reason."
    ),
)


@register_payload
@dataclass
class ClipEffect:
    """What was clipped, what it cost the gradient norm, and what it cost the applied update."""

    n_steps: int
    n_steps_with_both_norms: int
    #: Mean fraction of ratios clipped, over the steps that recorded one. NaN when none did.
    clip_fraction: float
    clip_fraction_source: str
    #: The reading: ``(unclipped - clipped) / clipped``, averaged over the steps carrying both
    #: norms. The base is the **clipped** norm, so this is how much larger the gradient was than
    #: the update that was applied from it, as a fraction of that update. Zero on an unclipped step.
    effect: float
    effect_per_step: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    clip_fraction_per_step: list[float] = field(default_factory=list)
    #: ``clipped / unclipped``, the factor the clip applied to the update, averaged over steps.
    #: One on an unclipped step and below one on a clipped one. This is the multiplier that
    #: propagates into any quantity treating the update as proportional to the gradient.
    shrinkage: float = math.nan
    shrinkage_per_step: list[float] = field(default_factory=list)
    #: How many of `n_steps_with_both_norms` were actually shrunk, at a relative tolerance of 1e-9.
    #: A run where this is zero recorded two norms and clipped on none of the steps, which is a
    #: different statement from having no effect to report.
    n_steps_shrunk: int = 0
    #: Mean of ``(unclipped / clipped)**2`` over steps, taken per step and then averaged. Squaring
    #: the mean effect instead would be low by the Jensen gap, which is the variance of the per-step
    #: ratio: on the two-step example in the unit tests it is 0.0156 on a value of 1.9062, or 0.8%.
    #: Carried because gradient power goes as the square of the norm and a reader wanting the
    #: shrinkage in power units should not have to square a mean themselves.
    ratio_squared: float = math.nan
    ratio_squared_jensen_gap: float = math.nan
    baselines: dict[str, float] = field(default_factory=dict)
    says: str = ""

    def render(self) -> str:
        return self.says


def _clip_fraction_of(step: Step) -> tuple[float, str]:
    """One step's ratio-clip fraction and where it came from, or NaN and an empty source."""
    for key in RATIO_CLIP_KEYS:
        value = step.optimizer.extra.get(key)
        if value is not None and math.isfinite(float(value)):
            return float(value), f"OptimizerTelemetry.extra[{key!r}]"
    if step.optimizer.clip_fraction is not None:
        return float(step.optimizer.clip_fraction), "OptimizerTelemetry.clip_fraction"
    return math.nan, ""


def measure_clip_effect(
    subject: Run | Sequence[Step] | Step,
    *,
    instrument: str = "ClipAccounting",
) -> ClipEffect | Refusal:
    """The clip's effect on the applied update, or a bound when only one norm was recorded."""
    steps = list(iter_steps(subject))
    if not steps:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail="this window contains no optimizer steps, so there is no clip to account for",
            remedy="point this at a Run or a sequence of Steps carrying `OptimizerTelemetry`.",
            statistics={"n_steps": 0},
        )

    fractions: list[float] = []
    sources: set[str] = set()
    indices: list[int] = []
    effects: list[float] = []
    shrinkages: list[float] = []
    ratio_squares: list[float] = []
    both: list[int] = []
    for step in steps:
        frac, source = _clip_fraction_of(step)
        indices.append(step.index)
        fractions.append(frac)
        if source:
            sources.add(source)
        clipped = step.optimizer.grad_norm_clipped
        unclipped = step.optimizer.grad_norm_unclipped
        if clipped is not None and unclipped is not None and clipped > 0 and unclipped > 0:
            ratio = float(unclipped) / float(clipped)
            effects.append(ratio - 1.0)
            shrinkages.append(1.0 / ratio)
            ratio_squares.append(ratio * ratio)
            both.append(step.index)

    live = [f for f in fractions if math.isfinite(f)]
    mean_fraction = float(np.mean(live)) if live else math.nan
    source = ", ".join(sorted(sources)) if sources else ""

    reading = ClipEffect(
        n_steps=len(steps),
        n_steps_with_both_norms=len(both),
        clip_fraction=mean_fraction,
        clip_fraction_source=source,
        effect=math.nan,
        steps=indices,
        clip_fraction_per_step=fractions,
        baselines={
            "baseline.clip_fraction_alone": mean_fraction,
            "baseline.clip_has_no_effect": 0.0,
        },
    )

    if not both:
        if not live:
            return Refusal(
                instrument=instrument,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    f"none of the {len(steps)} steps recorded a clip fraction or both gradient "
                    f"norms, so there is neither a clip to report nor an effect to propagate."
                ),
                remedy=(
                    "record `OptimizerTelemetry.clip_fraction` and both of "
                    "`grad_norm_clipped` and `grad_norm_unclipped`. The norm a trainer logs is the "
                    "**pre**-clip one, so it belongs in `grad_norm_unclipped`; the missing one is "
                    "the post-clip norm, which is `min(logged_norm, max_grad_norm)` and costs "
                    "nothing to write down once the threshold is in hand."
                ),
                statistics={"n_steps": len(steps)},
            )
        reading.says = (
            f"{mean_fraction:.1%} of ratios were clipped, averaged over {len(live)} of "
            f"{len(steps)} steps, and the shrinkage the clip applied to the update cannot be "
            f"computed because no step recorded both gradient norms."
        )
        return bounded_refusal(
            instrument=instrument,
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=(
                f"{len(steps)} steps recorded a clip fraction and none recorded both gradient "
                f"norms, so the shrinkage the clip applied to the update is not computable. The "
                f"clip fraction alone is what everybody already has, and propagating nothing from "
                f"it is the gap this instrument exists to close."
            ),
            remedy=(
                "record both norms. `transformers/trainer.py:2523` returns the **pre**-clip norm "
                "and `trainer.py:2087` logs it under `grad_norm`, so that value belongs in "
                "`OptimizerTelemetry.grad_norm_unclipped`. The post-clip norm is "
                "`min(that, args.max_grad_norm)` and goes in `grad_norm_clipped`; nothing has to "
                "be recomputed to get it."
            ),
            bound=_bare_evidence(reading),
            clip_fraction=mean_fraction,
            n_steps=len(steps),
            n_steps_with_both_norms=0,
        )

    arr = np.asarray(effects, dtype=float)
    reading.effect = float(arr.mean())
    reading.effect_per_step = effects
    reading.n_steps_with_both_norms = len(both)
    reading.shrinkage = float(np.mean(shrinkages))
    reading.shrinkage_per_step = shrinkages
    reading.n_steps_shrunk = sum(1 for s in shrinkages if s < 1.0 - 1e-9)
    # Per step, then averaged. `(1 + mean(effect))**2` would be the square of a mean and low by
    # exactly the variance of the per-step ratio, which is what `ratio_squared_jensen_gap` reports.
    reading.ratio_squared = float(np.mean(ratio_squares))
    reading.ratio_squared_jensen_gap = reading.ratio_squared - (1.0 + reading.effect) ** 2
    shrunk = (
        f"{reading.n_steps_shrunk} of them were clipped, and on those the update was scaled by "
        f"{reading.shrinkage:.4g} on average"
        if reading.n_steps_shrunk
        else (
            "none of them was clipped: the two recorded norms agree on every step, so the clip "
            "changed no update in this window"
        )
    )
    head = (
        f"{mean_fraction:.1%} of ratios were clipped, averaged over the steps that recorded it."
        if math.isfinite(mean_fraction)
        else "No clip fraction was recorded."
    )
    reading.says = (
        f"{head} {len(both)} of {len(steps)} steps recorded both gradient norms and {shrunk}. Any "
        f"quantity that treats the update as proportional to the gradient is wrong by that factor "
        f"on those steps."
    )
    return reading


def _bare_evidence(reading: ClipEffect) -> Any:
    from reward_lens.core.evidence import make_evidence
    from reward_lens.core.types import SubjectRef

    return make_evidence(
        observable="ClipAccounting",
        observable_version="1.0",
        subject=SubjectRef(signals=(), dataset=None, readout="reward"),
        value=reading,
        gauge=GaugeStatus.INVARIANT,
    )


class ClipAccounting(EstimatorInstrument):
    """E5. The clip fraction, and what it does to every quantity derived from the gradient norm.

    Kill condition, from the catalogue record: n/a.
    """

    name = "ClipAccounting"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "E5"
    deviations = (
        "the effect is computed from the two recorded gradient norms rather than by recomputing "
        "the gradient. The catalogue says 'recomputing unclipped', which needs a backward "
        "pass; the unclipped norm is the one every trainer already logs, so the ratio of the two "
        "is the same number without the pass",
        "the ratio-clip fraction is read out of `OptimizerTelemetry.extra`, which is untyped by "
        "construction. The key it came from is named on every reading, and no arithmetic here "
        "depends on it: the effect comes from the two norms",
        "`OptimizerTelemetry.clip_fraction`'s meaning is not settled by the schema (ratio clip or "
        "gradient-norm clip) and this instrument does not settle it. It is reported as recorded "
        "and used in no calculation",
        "the effect is propagated into the applied update rather than into veRL's `proxy1`. "
        "The catalogue's `says` line for E5 reads 'Recomputing unclipped moves the gradient norm by "
        "31%, so every gradient-derived quantity is biased', which assumes the logged norm is the "
        "post-clip one. It is the pre-clip norm in torch, FSDP, veRL and transformers, each of "
        "which says so in its own source, so the quantity the clip biases is the update and not "
        "the logged norm",
    )

    quantity = "estimator.clip_fraction_effect"
    requires: dict[Component, Access] = {
        **RECORD_ACCESS,
        Component.OPTIMIZER: Access.RECORD,
    }
    substrates = ALL_SUBSTRATES
    phases = frozenset({Phase.IN_RUN, Phase.POST_RUN})
    envelope = CLIP_ENVELOPE
    #: `units` in the registry, whose assertion is a refusal rather than a numeric relation: a
    #: dimensionless share does not subtract from a per-token logprob mismatch. `check_invariance`
    #: routes this group to `check_unit_refusal`, and `measure.estimator.units.difference` is the
    #: comparison it asserts on.
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = CLIP_BASELINES
    rung = 0

    def __init__(
        self,
        subject: Run | Sequence[Step] | Step | Sequence[Group] | None = None,
    ) -> None:
        self.subject = subject

    def compute(self) -> Any:
        if self.subject is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no record was supplied, so there is no optimizer telemetry to read",
                remedy=(
                    "pass `subject=` a Run or a sequence of Steps carrying `OptimizerTelemetry` "
                    "with `grad_norm_clipped` and `grad_norm_unclipped`."
                ),
            )
        return measure_clip_effect(self.subject, instrument=self.name)


__all__ = [
    "CLIP_BASELINES",
    "CLIP_BIAS",
    "CLIP_ENVELOPE",
    "RATIO_CLIP_KEYS",
    "ClipAccounting",
    "ClipEffect",
    "measure_clip_effect",
]

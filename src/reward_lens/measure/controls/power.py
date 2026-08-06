"""M10 as an instrument: the power plan, the MDE and the resolution ratio, before the run.

The arithmetic lives in `stats.power`. This is the part that plugs into a preflight, so the
question "can this experiment answer the question" is asked at the point where the answer is still
free to act on, and the answer arrives as Evidence with the five standard calculators beside it as
baselines.

The division of labour between M10 and M5 is worth stating, because both of them are about power
and only one of them refuses. M10 computes what a design can see, before anything runs, and
reports it: a design that cannot see the effect is a fact about the design and Evidence is the
right carrier. M5 adjudicates a null after the fact, and there a missing control is a refusal
because the alternative is publishing a result that cannot be interpreted. Plan, then gate.

`resolve_row` is the leaderboard case. When `q = N/N* < 1` it returns a refusal rather than a
ranking, because a rank produced by a comparison that could not separate the two systems is a coin
flip with a decimal point.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec
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
from reward_lens.stats.power import (
    CALCULATORS,
    TARGET_POWER,
    PairedBinaryDesign,
    PowerPlan,
    Resolution,
    plan,
)

#: The five standard calculators, as baseline ids. All five are computed and reported, because the
#: finding this instrument carries is about the gap between them and the simulation.
CALCULATOR_BASELINES: tuple[BaselineID, ...] = tuple(f"baseline.{name}" for name in CALCULATORS)


POWER_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "a power calculation is a statement about a design and it is made before any run exists, "
        "so no regime of a run can violate it. What can be wrong is the design's own assumption "
        "about how much its n is worth, and that enters as the `ess` argument rather than as a "
        "precondition nobody checked."
    ),
)


def resolve_row(
    instrument: str,
    resolution: Resolution,
    *,
    observed_gap: float | None = None,
    mde: float | None = None,
) -> Any:
    """A leaderboard row's verdict, or a refusal saying it is not resolved.

    Returns the `Resolution` when `q >= 1`. Below 1 it returns a `Refusal`, because the row has
    not been measured: the sample could not have separated the two systems at the target power, so
    whichever one came out ahead came out ahead of nothing.

    The reason is `BELOW_LOD`. The limit of detection is `3.3 sigma_blank / S` and the minimum
    detectable effect is the same construction with the sampling standard deviation in place of
    the blank's, so an unresolved row is an effect below the design's own detection limit. That
    reuse is a judgement rather than a reading of the definition; the alternative would be a
    sixteenth refusal reason.
    """
    if resolution.resolved:
        return resolution
    detail = (
        f"q = N/N* = {resolution.q:.3f} at {resolution.target_power:.0%} power "
        f"({resolution.n:,.0f} {resolution.basis} against {resolution.n_star:,.0f} needed), so "
        f"this row is not resolved"
    )
    if observed_gap is not None:
        detail += f". The observed gap is {observed_gap:+.4g}"
        if mde is not None and np.isfinite(mde):
            detail += f" and the smallest detectable one is {mde:.4g}"
    return Refusal(
        instrument=instrument,
        reason=RefusalReason.BELOW_LOD,
        detail=detail,
        remedy=(
            f"raise the sample to {resolution.n_star:,.0f} {resolution.basis} and re-run, or "
            f"publish the row as unresolved. Reporting an ordering here reports the noise. "
            f"`stats.power.plan` gives the n; `stats.power.resolution_from_lineage` gives the "
            f"effective n when the items are expansions of a smaller seed set."
        ),
        statistics={
            "q": resolution.q,
            "n": resolution.n,
            "n_star": resolution.n_star,
            "target_power": resolution.target_power,
            "observed_gap": observed_gap,
            "mde": mde,
        },
    )


class PowerAndMDE(ControlInstrument):
    """M10. Simulated power, the minimum detectable effect and `q = N/N*` for one design.

    Every number is simulated against the test that will actually be run. The five standard
    calculators are computed too and reported as this instrument's baselines, which is the honest
    place for them: they are the comparators, and on close paired comparisons three of them come
    out roughly 2x wrong in the expensive direction.
    """

    name = "PowerAndMDE"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "M10"
    deviations = (
        "the simulator covers the paired binary design (two systems scored right or wrong on the "
        "same items). A continuous or ordinal outcome needs its own simulator and this instrument "
        "declines rather than approximating one",
        "`rho` is an input. It is measurable on a pilot for nothing, and planning at rho = 0 is a "
        "different experiment rather than a conservative version of this one",
    )

    quantity = "study.power"
    #: A power calculation before the run needs access to nothing, which is the entire argument for
    #: doing it first. The declaration is empty on purpose rather than by omission, and the field is
    #: `requires` because that is the name the access matrix carries. An empty matrix and a
    #: matrix under the wrong name read the same from `declared_access` and mean opposite things.
    requires: dict[Component, Access] = {}
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
    envelope = POWER_ENVELOPE
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = CALCULATOR_BASELINES
    rung = 2

    def __init__(
        self,
        design: PairedBinaryDesign | None = None,
        *,
        target_power: float = TARGET_POWER,
        replicates: int = 8_000,
        seed: int = 0,
        ess: float | None = None,
        with_calculators: bool = True,
    ) -> None:
        self.design = design
        self.target_power = float(target_power)
        self.replicates = int(replicates)
        self.seed = int(seed)
        self.ess = ess
        self.with_calculators = bool(with_calculators)

    def compute(self) -> Any:
        if self.design is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no design was supplied, so there is nothing to power",
                remedy=(
                    "pass `design=PairedBinaryDesign(n=..., accuracy_a=..., accuracy_b=..., "
                    "rho=...)`. A pilot of fifty items gives you all four."
                ),
            )
        return plan(
            self.design,
            target_power=self.target_power,
            replicates=self.replicates,
            seed=self.seed,
            with_calculators=self.with_calculators,
            ess=self.ess,
        )

    def payload(self, computed: PowerPlan) -> dict[str, Any]:
        return {
            "power": computed.power,
            "power_mc_se": computed.power_mc_se,
            "n_star": computed.n_star,
            "mde": computed.mde,
            "q": computed.resolution.q,
            "resolved": computed.resolution.resolved,
            "target_power": computed.target_power,
            "replicates": computed.replicates,
            "validated_against": computed.validated_against,
            "baselines": {
                f"baseline.{name}": float(check.n_star)
                for name, check in computed.calculators.items()
            },
        }


__all__ = ["CALCULATOR_BASELINES", "POWER_ENVELOPE", "PowerAndMDE", "resolve_row"]

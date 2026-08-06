"""M5, the matched positive control. A null with no control is a refusal, and the gate is real.

A null result and an underpowered experiment produce the same output. The only thing that tells
them apart is running the identical measurement, at the identical n, on a case where you planted
an effect and know it is there. If the same test finds the planted effect, the null means
something. If it does not, the null means the design could not have found anything and the result
is about the design.

This library has a worked example of getting that wrong in its own history: a susceptibility card
was accepted at power 0.13 with no positive control beside it. At 0.13, thirteen runs in a hundred
would have detected a real effect, so the null it reported was very nearly the only outcome
available.

`NO_MATCHED_CONTROL` is one of the fifteen refusal reasons for this. What this module adds is that
the gate is not advice: `gate_null` returns a refusal, `guard_null` wraps an instrument so the
refusal happens whether or not the instrument's author remembered, and "identically powered" is
checked field by field rather than asserted. A positive control run at ten times the n is not a
matched control, it is a different experiment that happened to succeed.

The remedy on the refusal is an instruction, because that is what a remedy is: run the same
measurement on a matched positive control at the same n, or report this as underpowered rather
than null.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BaselineID
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.base import Context
from reward_lens.measure.controls._base import ControlInstrument
from reward_lens.stats.ess import effective_sample_size

#: What a claim reports when it has no control at all: the design's own formula power. It is the
#: comparator M5 exists to beat, because a formula power is what the underpowered cards had.
NOMINAL_POWER: BaselineID = "baseline.nominal_power"


# ---------------------------------------------------------------------------
# What "identically powered" means, field by field
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlDesign:
    """The design fields that have to match for a control to be matched.

    Every field here is one somebody has silently changed between a null arm and its control. n is
    the obvious one. ``ess`` is the one that actually bites, because a control built from fifty
    fresh stimuli and a null arm built from fifty mutations of five seeds have the same n and a
    ten-fold difference in what that n is worth. ``statistic`` is here because a control tested
    with a t-test does not license a null tested with a permutation test.
    """

    n: int
    alpha: float = 0.05
    tails: int = 2
    statistic: str = ""
    ess: float | None = None
    #: Anything else the two arms have to share, compared by equality. Engine, revision, seed
    #: policy, prompt order: the `CouplingSpec` fields belong here until that type exists.
    coupling: Mapping[str, Any] = field(default_factory=dict)

    @property
    def effective_n(self) -> float:
        """ESS when it was measured, n otherwise, and the difference is reported not hidden."""
        return float(self.ess) if self.ess is not None else float(self.n)

    def mismatches(self, other: "ControlDesign", *, n_tolerance: float = 0.1) -> list[str]:
        """Every way this design differs from another, named for the refusal detail.

        ``n_tolerance`` allows a 10% difference in effective n, because insisting on exact
        equality would reject a control that is matched in every way that matters and differs by
        two dropped items. Anything wider than that is a different experiment.
        """
        out: list[str] = []
        mine, theirs = self.effective_n, other.effective_n
        if theirs <= 0 or abs(mine - theirs) / max(mine, theirs, 1.0) > n_tolerance:
            basis = "effective n" if (self.ess is not None or other.ess is not None) else "n"
            out.append(f"{basis} {mine:.1f} against {theirs:.1f}")
        if self.alpha != other.alpha:
            out.append(f"alpha {self.alpha:g} against {other.alpha:g}")
        if self.tails != other.tails:
            out.append(f"{self.tails}-tailed against {other.tails}-tailed")
        if self.statistic and other.statistic and self.statistic != other.statistic:
            out.append(f"statistic {self.statistic!r} against {other.statistic!r}")
        for key in sorted(set(self.coupling) | set(other.coupling)):
            if self.coupling.get(key) != other.coupling.get(key):
                out.append(f"{key} {self.coupling.get(key)!r} against {other.coupling.get(key)!r}")
        return out

    @classmethod
    def from_lineage(
        cls, seed_labels: Any, *, alpha: float = 0.05, tails: int = 2, statistic: str = ""
    ) -> "ControlDesign":
        """Build a design whose effective n comes from the lineage rather than the row count."""
        labels = list(seed_labels)
        return cls(
            n=len(labels),
            alpha=alpha,
            tails=tails,
            statistic=statistic,
            ess=effective_sample_size(labels),
        )


@dataclass(frozen=True)
class MatchedControl:
    """A positive control: the same measurement, the same design, a planted effect of known size.

    ``detected`` is the field the gate reads. It is derived from the control's own p-value at the
    control's own alpha rather than supplied, so a control cannot be marked as having worked by
    the person who wanted it to.
    """

    id: str
    design: ControlDesign
    planted_effect: float
    observed_effect: float
    p_value: float
    note: str = ""

    @property
    def detected(self) -> bool:
        return bool(self.p_value <= self.design.alpha)

    def render(self) -> str:
        state = "detected" if self.detected else "MISSED"
        return (
            f"{self.id}: planted {self.planted_effect:+.4g}, observed "
            f"{self.observed_effect:+.4g}, p={self.p_value:.4g} at alpha "
            f"{self.design.alpha:g} -> {state}"
        )


@dataclass(frozen=True)
class NullClaim:
    """A claim that reports no effect, with the design that reported it."""

    instrument: str
    effect: float
    p_value: float
    design: ControlDesign
    #: The smallest effect this design could have found, when M10 has been run. Carried because it
    #: is the honest bound a refusal should hand back: "I could not have seen anything under this".
    mde: float | None = None

    @property
    def is_null(self) -> bool:
        return bool(self.p_value > self.design.alpha)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlVerdict:
    """What the gate decided, in the shape `PreflightResult` uses: ok, plus the refusal if not."""

    claim: str
    ok: bool
    refusal: Refusal | None = None
    control: MatchedControl | None = None
    note: str = ""

    def render(self) -> str:
        if not self.ok and self.refusal is not None:
            return self.refusal.render()
        head = f"{self.claim}: null accepted"
        if self.control is not None:
            head += f", matched control {self.control.id} detected its planted effect"
        return head + (f"\n    {self.note}" if self.note else "")


def _bound_detail(claim: NullClaim) -> str:
    if claim.mde is None or not np.isfinite(claim.mde):
        return ""
    return (
        f" The design's minimum detectable effect is {claim.mde:.4g}, so this null bounds the "
        f"effect below that and says nothing at all underneath it."
    )


def gate_null(claim: NullClaim, control: MatchedControl | None) -> ControlVerdict:
    """The M5 gate. A null with no matched control refuses; so does one whose control missed.

    Four outcomes, and three of them are the point.

    A claim that is not a null passes straight through: this gate is about nulls and nothing else.
    A null with no control refuses with `NO_MATCHED_CONTROL`. A null whose control exists but is
    not matched refuses and names every field that differs, because "we ran a positive control"
    with a control at three times the n is worse than no control at all: it reads as rigour and
    licenses nothing. And a null whose matched control **missed its own planted effect** refuses
    hardest of all, because that is a direct measurement that the design cannot detect effects of
    the planted size, which is what an underpowered experiment is.
    """
    if not claim.is_null:
        return ControlVerdict(
            claim=claim.instrument,
            ok=True,
            control=control,
            note=(
                f"p={claim.p_value:.4g} at alpha {claim.design.alpha:g} is not a null, so the "
                f"matched-control gate does not apply"
            ),
        )

    if control is None:
        return ControlVerdict(
            claim=claim.instrument,
            ok=False,
            refusal=Refusal(
                instrument=claim.instrument,
                reason=RefusalReason.NO_MATCHED_CONTROL,
                detail=(
                    f"reports a null (effect {claim.effect:+.4g}, p={claim.p_value:.4g} at alpha "
                    f"{claim.design.alpha:g}, effective n {claim.design.effective_n:.1f}) with no "
                    f"positive control, so it cannot be distinguished from an underpowered "
                    f"experiment." + _bound_detail(claim)
                ),
                remedy=(
                    "run the same measurement on a matched positive control at the same n, or "
                    "report this as underpowered rather than null. `stats.power.plan` gives the "
                    "n that would have made the null informative."
                ),
                statistics={
                    "effect": claim.effect,
                    "p_value": claim.p_value,
                    "alpha": claim.design.alpha,
                    "n": claim.design.n,
                    "effective_n": claim.design.effective_n,
                    "mde": claim.mde,
                },
            ),
        )

    gaps = claim.design.mismatches(control.design)
    if gaps:
        return ControlVerdict(
            claim=claim.instrument,
            ok=False,
            control=control,
            refusal=Refusal(
                instrument=claim.instrument,
                reason=RefusalReason.NO_MATCHED_CONTROL,
                detail=(
                    f"the positive control {control.id!r} is not matched to this claim: "
                    + "; ".join(gaps)
                    + ". A control at a different design does not license this null."
                ),
                remedy=(
                    "re-run the control at the claim's own design, field for field: "
                    + "; ".join(gaps)
                    + ". If the control cannot be matched, report the claim as underpowered."
                ),
                statistics={"mismatches": gaps},
            ),
        )

    if not control.detected:
        return ControlVerdict(
            claim=claim.instrument,
            ok=False,
            control=control,
            refusal=Refusal(
                instrument=claim.instrument,
                reason=RefusalReason.NO_MATCHED_CONTROL,
                detail=(
                    f"the matched control {control.id!r} planted an effect of "
                    f"{control.planted_effect:+.4g} and the same test missed it "
                    f"(p={control.p_value:.4g} at alpha {control.design.alpha:g}). This design "
                    f"cannot detect effects of that size, so the null is about the design."
                    + _bound_detail(claim)
                ),
                remedy=(
                    "raise n until the control's planted effect is detected, then re-run the "
                    "claim. Reporting the null as it stands would be reporting the design."
                ),
                statistics={
                    "control_planted": control.planted_effect,
                    "control_p": control.p_value,
                    "claim_p": claim.p_value,
                    "n": claim.design.n,
                },
            ),
        )

    return ControlVerdict(
        claim=claim.instrument,
        ok=True,
        control=control,
        note=(
            f"the identical design detected a planted effect of {control.planted_effect:+.4g} "
            f"(p={control.p_value:.4g}), so this null is informative down to about that size"
        ),
    )


# ---------------------------------------------------------------------------
# The wrapper that makes the gate real rather than remembered
# ---------------------------------------------------------------------------

#: How to read a null out of an arbitrary instrument's reading. The default understands the shape
#: this library's instruments emit: a mapping payload carrying an effect and a p-value.
NullExtractor = Callable[[Any], "NullClaim | None"]


def default_null_extractor(
    reading: Any, *, design: ControlDesign | None = None
) -> NullClaim | None:
    """Read `effect` and `p_value` out of an Evidence payload, or return None.

    Returning None means "this reading is not a null claim I can recognise", and the wrapper then
    lets it through. That is the right default: a gate that refuses everything it does not
    understand is a gate people remove.
    """
    value = getattr(reading, "value", None)
    if not isinstance(value, Mapping):
        return None
    p = value.get("p_value", value.get("p"))
    effect = value.get("effect", value.get("delta"))
    if p is None or effect is None:
        return None
    d = design or ControlDesign(
        n=int(value.get("n", 0) or 0), alpha=float(value.get("alpha", 0.05))
    )
    return NullClaim(
        instrument=getattr(reading, "observable", "claim"),
        effect=float(effect),
        p_value=float(p),
        design=d,
        mde=value.get("mde"),
    )


class GuardedInstrument:
    """An instrument wrapped so that a null without a matched control cannot get out.

    This is what makes the gate real. An instrument that returns a null is not trusted to remember
    the control: the wrapper reads the reading, and if it is a recognisable null with no matched
    control in the `Context`, the wrapper replaces it with the refusal. Every declaration is
    forwarded, so a wrapped instrument still passes `lint_instrument` and still satisfies the
    `Instrument` protocol.

    The control is read from ``ctx.stats["matched_control"]``, which is an existing seam on
    `Context` rather than a new field, and from ``ctx.has_matched_control``, which the kernel
    already carries. A bare `True` there is accepted and recorded as unverified, because a boolean
    cannot be checked for matching and pretending otherwise would be worse than saying so.
    """

    def __init__(
        self,
        inner: Any,
        *,
        extractor: NullExtractor | None = None,
        design: ControlDesign | None = None,
    ) -> None:
        self.inner = inner
        self.extractor = extractor
        self.design = design

    def __getattr__(self, item: str) -> Any:
        return getattr(self.inner, item)

    def preflight(self, ctx: Context) -> Any:
        return self.inner.preflight(ctx)

    def estimate(self, ctx: Context) -> Reading:
        reading = self.inner.estimate(ctx)
        if isinstance(reading, Refusal):
            return reading
        claim = (
            self.extractor(reading)
            if self.extractor is not None
            else default_null_extractor(reading, design=self.design)
        )
        if claim is None or not claim.is_null:
            return reading
        control = ctx.stats.get("matched_control")
        if isinstance(control, MatchedControl):
            verdict = gate_null(claim, control)
            return reading if verdict.ok else (verdict.refusal or reading)
        if ctx.has_matched_control:
            return reading
        return gate_null(claim, None).refusal or reading


def guard_null(
    inner: Any, *, extractor: NullExtractor | None = None, design: ControlDesign | None = None
) -> GuardedInstrument:
    """Wrap an instrument so a null without a matched control refuses. One call, no edits."""
    return GuardedInstrument(inner, extractor=extractor, design=design)


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

MATCHED_CONTROL_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "this instrument compares two designs and the detection the second one achieved. It "
        "reads no property of the optimisation that produced either, so no regime of a run can "
        "make 'these two designs differ in effective n' or 'the control missed its planted "
        "effect' wrong."
    ),
)


class MatchedPositiveControl(ControlInstrument):
    """M5. Whether a null is informative, decided by an identically-powered positive control.

    The reading is the control's realised detection at the claim's own design, which is a measured
    power rather than a formula power. That is why it declares `study.power`: it is the same
    quantity M10 computes before the run, estimated at a higher rung by measurement instead of by
    a calculation. When the two disagree, the measurement wins and the disagreement is worth
    publishing.
    """

    name = "MatchedPositiveControl"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "M5"
    deviations = (
        "the control's realised detection is a single run at the claim's design rather than a "
        "replicated estimate of power, so it is a Bernoulli draw from the design's power and not "
        "the power itself; replicate the control to narrow it",
        "'identically powered' is checked as design-field equality with a 10% tolerance on "
        "effective n, not as an equality of computed power",
    )

    quantity = "study.power"
    #: `requires`, not `access`. See the note on `DumbBaselineBank`.
    requires = {Component.RECORD: Access.RECORD}
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
    envelope = MATCHED_CONTROL_ENVELOPE
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = (NOMINAL_POWER,)
    rung = 3

    def __init__(
        self,
        claim: NullClaim | None = None,
        control: MatchedControl | None = None,
        *,
        nominal_power: float = float("nan"),
    ) -> None:
        self.claim = claim
        self.control = control
        #: The power the design's own formula claimed, for the comparison M5 exists to make.
        self.nominal_power = float(nominal_power)

    def compute(self) -> Any:
        if self.claim is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no claim was supplied, so there is no null to adjudicate",
                remedy="pass `claim=NullClaim(...)` describing the reading and its design",
            )
        return gate_null(self.claim, self.control)

    def payload(self, computed: ControlVerdict) -> dict[str, Any]:
        control = computed.control
        return {
            "null_accepted": computed.ok,
            "control": control.id if control is not None else None,
            "control_detected": bool(control.detected) if control is not None else None,
            "planted_effect": control.planted_effect if control is not None else None,
            "realised_power": 1.0 if (control is not None and control.detected) else 0.0,
            "n": self.claim.design.n if self.claim is not None else None,
            "effective_n": self.claim.design.effective_n if self.claim is not None else None,
            "baselines": {NOMINAL_POWER: self.nominal_power},
        }

    def estimate(self, ctx: Context) -> Reading:
        """Preflight, then the gate. A failing gate is the refusal, not a payload field."""
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        out = self.compute()
        if isinstance(out, Refusal):
            return out
        if isinstance(out, ControlVerdict) and not out.ok and out.refusal is not None:
            return out.refusal
        return self.gated_emit(ctx, out)


__all__ = [
    "MATCHED_CONTROL_ENVELOPE",
    "NOMINAL_POWER",
    "ControlDesign",
    "ControlVerdict",
    "GuardedInstrument",
    "MatchedControl",
    "MatchedPositiveControl",
    "NullClaim",
    "NullExtractor",
    "default_null_extractor",
    "gate_null",
    "guard_null",
]

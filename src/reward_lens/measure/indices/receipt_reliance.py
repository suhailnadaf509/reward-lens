"""A7 RRS: the Receipt Reliance Score.

Formal definition, A7. ``RRS =`` the fraction of the corruption reward effect (the
falsify-receipt versus falsify-narrative arms) causally attributable to receipt spans, via span patching
plus attention forensics from the scoring position (faithful_to N1, the trajectory reward forensics
program). A reward that grounds its judgment in the receipts (tool outputs, citations, logs) moves most
of its corruption response when the receipt is falsified; a reward that reads the narrative gloss moves
when the narrative is falsified instead. ``RRS`` is one of the two axes of the honesty law (S15), the
other being skepticism (``skepticism.py``).

Deviation from A7: the pure function is the attribution ratio over the measured arm deltas; the
production path supplies ``Δr_receipt`` from a receipt-span patch and the attention forensics through the
interventions and attribution subsystems. The synthetic test drives the ratio with planted arm deltas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.types import Access, AccessMatrix, Capability, Component, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context, PreflightResult
from reward_lens.measure.indices._support import (
    ANY_SUBSTRATE,
    GRADER_STUDY_PHASES,
    MEASURED_BY,
    measured_without_input,
    missing_injection,
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def receipt_reliance(dr_receipt: float, dr_total: float) -> float:
    """The receipt reliance score ``RRS = Δr_receipt / Δr_total`` (A7).

    ``dr_total`` is the whole corruption reward effect; ``dr_receipt`` is the part attributable to the
    receipt spans (the falsify-receipt arm, or the receipt-span patch). Their ratio is the causal
    fraction the reward reads off the receipts, in ``[0, 1]`` for a clean decomposition. A zero total
    effect yields ``nan`` rather than a fabricated fraction.
    """
    if dr_total == 0:
        return float("nan")
    return float(dr_receipt) / float(dr_total)


class ReceiptReliance(BaseObservable):
    """A7 fraction of the corruption reward effect the reward reads off receipt spans.

    Requires span-typed inputs and prefix scores on the production path (the receipt-span patch and the
    falsify-receipt/falsify-narrative arms). Here the arm deltas are injected so the attribution
    arithmetic runs directly. Gauge is INVARIANT: ``RRS`` is a fraction.

    What it cannot do. This is a ratio of two deltas somebody else measured, so it is exactly as
    good as the matching between the falsify-receipt and falsify-narrative arms: if the two
    corruptions differ in size or in how natural they read, the ratio measures that difference and
    calls it reliance. A small ``dr_total`` makes the ratio unstable and the payload reports it
    without a bound, so both deltas have to clear the substrate's limit of detection before the
    fraction means anything.
    """

    name = "ReceiptReliance"
    version = "1.0"
    capabilities = Capability.SPAN_TYPES
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A7"
    deviations = (
        "consumes injected arm deltas; the receipt-span patch and attention forensics are the "
        "production path (interventions + attribution)",
    )

    # -- the observable declarations ---------------------------------------
    quantity = "grader.receipt_reliance"
    #: Both arm deltas are recorded measurements from an earlier span-patching experiment.
    requires: AccessMatrix = {Component.RECORD: Access.RECORD}
    substrates = ANY_SUBSTRATE
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.STATIONARY_GRADER, RegimeCondition.ABOVE_LOD}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: A ratio of two reward differences: the scale factor cancels and the shift cancels twice, so
    #: an affine rescaling of the reward leaves it where it was.
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = ("baseline.narrative_arm", "baseline.random_span_patch")
    rung = 0

    def __init__(self, dr_receipt: float | None = None, dr_total: float | None = None) -> None:
        self.dr_receipt = dr_receipt
        self.dr_total = dr_total

    def preflight(self, ctx: Context) -> PreflightResult:
        """Both corruption arms, or a refusal. A ratio needs a denominator.

        The injected input is absent, which makes this a `Refusal` rather than an Evidence
        carrying a note. Nothing has to be computed to know it, so the question belongs
        here: `estimate` returns this refusal before `measure` is reached, and the
        capability report gets it with no work at all.
        """
        if self.dr_receipt is None or self.dr_total is None:
            return missing_injection(
                self,
                needs={
                    "dr_total": "the whole corruption reward effect, r(clean) minus r(corrupted)",
                    "dr_receipt": "the part of that effect attributable to the receipt spans",
                },
                have="neither arm was injected",
                remedy=(
                    "Run both corruption arms and construct `ReceiptReliance(dr_receipt=..., dr_total=...)`. "
                    "`dr_total` is the reward gap between the clean and corrupted runs; `dr_receipt` is the "
                    "gap recovered by the falsify-receipt arm or by patching the receipt spans. The reading "
                    "is their ratio, so one without the other is not a fraction of anything."
                ),
            )
        return super().preflight(ctx)

    def measure(self, ctx: Context) -> "Evidence":
        if self.dr_receipt is None or self.dr_total is None:
            raise measured_without_input(self)
        rrs = receipt_reliance(self.dr_receipt, self.dr_total)
        return ctx.emit(
            {
                "receipt_reliance": rrs,
                "dr_receipt": float(self.dr_receipt),
                "dr_total": float(self.dr_total),
            },
            uncertainty=Uncertainty(method="none"),
        )


__all__ = ["receipt_reliance", "ReceiptReliance"]

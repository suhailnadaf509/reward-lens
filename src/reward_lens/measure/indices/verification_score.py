"""A6 Verification Score: causal fraction of correctness-Δr at the error span.

Formal definition, A6. ``VS =`` the fraction of the correctness-``Δr`` between clean and
corrupted twins that is causally attributable to the error span, measured by patching the clean twin's
error-span activations into the corrupted run (faithful_to the error-microscope construction). A
process/verifier reward that is genuinely checking the work concentrates its clean-vs-corrupted reward
gap at the span where the corruption lives; a reward that is reacting to surface style spreads the gap
everywhere but the error. ``VS`` and the style share (``style_share.py``) need not sum to one: the
residual is reward change explained by neither, and is reported as such.

Deviation from A6: the pure function is the attribution arithmetic on the measured reward deltas; the
production path supplies ``Δr_error_span`` from an actual clean-twin span patch through the interventions
subsystem, and this index consumes those deltas. The synthetic test drives the arithmetic with planted
deltas of a known ratio.
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


def verification_score(dr_total: float, dr_error_span: float) -> float:
    """The verification score ``VS = Δr_error_span / Δr_total`` (A6).

    ``dr_total = r(clean) − r(corrupted)`` is the whole correctness reward gap; ``dr_error_span`` is the
    part recovered by patching the clean twin's error span into the corrupted run. Their ratio is the
    causal fraction of the gap that lives at the error, in ``[0, 1]`` for a well-behaved patch and
    reported as-is (possibly outside it) when patches interact. A zero total gap yields ``nan`` rather
    than a fabricated fraction.
    """
    if dr_total == 0:
        return float("nan")
    return float(dr_error_span) / float(dr_total)


class VerificationScore(BaseObservable):
    """A6 causal fraction of the correctness reward gap that lives at the labeled error span.

    Requires prefix scores (the process/verifier reward) on the production path, plus the interventions
    subsystem for the clean-twin span patch. Here the measured deltas are injected (``dr_total``,
    ``dr_error_span``) so the attribution arithmetic is exercised without waiting for interventions; the
    production path substitutes the patched deltas. Gauge is INVARIANT: ``VS`` is a fraction.

    What it cannot do. Both deltas are supplied, so this is the attribution arithmetic and not the
    attribution: the patch that produced ``dr_error_span`` decides the answer and this instrument
    cannot inspect it. A span patch moves more than the error when the span is generous, and the
    score rises with span width for reasons that have nothing to do with verification. ``VS`` and
    the style share do not sum to one and must not be shown as a partition; what is left over is
    reward change neither of them explains.
    """

    name = "VerificationScore"
    version = "1.0"
    capabilities = Capability.STEP_SCORES
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A6"
    deviations = (
        "consumes measured reward deltas; the clean-twin span patch that produces dr_error_span is "
        "the production path through the interventions subsystem",
    )

    # -- the observable declarations ---------------------------------------
    quantity = "grader.verification_score"
    #: Both deltas are recorded from an earlier clean-twin span-patching experiment.
    requires: AccessMatrix = {Component.RECORD: Access.RECORD}
    substrates = ANY_SUBSTRATE
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.STATIONARY_GRADER, RegimeCondition.ABOVE_LOD}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: A ratio of two reward differences: the scale cancels and the shift cancels twice, so an
    #: affine rescaling of the reward leaves it unchanged.
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = ("baseline.random_span_patch", "baseline.style_share")
    rung = 0

    def __init__(self, dr_total: float | None = None, dr_error_span: float | None = None) -> None:
        self.dr_total = dr_total
        self.dr_error_span = dr_error_span

    def preflight(self, ctx: Context) -> PreflightResult:
        """Both halves of the span patch, or a refusal. A ratio needs a denominator.

        The injected input is absent, which makes this a `Refusal` rather than an Evidence
        carrying a note. Nothing has to be computed to know it, so the question belongs
        here: `estimate` returns this refusal before `measure` is reached, and the
        capability report gets it with no work at all.
        """
        if self.dr_total is None or self.dr_error_span is None:
            return missing_injection(
                self,
                needs={
                    "dr_total": "the whole correctness reward gap, r(clean) minus r(corrupted)",
                    "dr_error_span": "the part of that gap recovered by patching the clean twin's error span",
                },
                have="neither was injected",
                remedy=(
                    "Run the span patch and construct `VerificationScore(dr_total=..., dr_error_span=...)`. "
                    "`dr_total` is the reward gap between the clean and corrupted runs; `dr_error_span` is "
                    "the gap recovered by patching the clean twin's error span into the corrupted run. The "
                    "reading is their ratio, so one without the other is not a fraction of anything."
                ),
            )
        return super().preflight(ctx)

    def measure(self, ctx: Context) -> "Evidence":
        if self.dr_total is None or self.dr_error_span is None:
            raise measured_without_input(self)
        vs = verification_score(self.dr_total, self.dr_error_span)
        payload = {
            "verification_score": vs,
            "dr_total": float(self.dr_total),
            "dr_error_span": float(self.dr_error_span),
        }
        return ctx.emit(payload, uncertainty=Uncertainty(method="none"))


__all__ = ["verification_score", "VerificationScore"]

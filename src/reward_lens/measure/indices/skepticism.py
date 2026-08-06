"""A8 Skepticism: the second axis of the honesty law.

Formal definition, A8. ``S = r(claim, receipt absent) − r(claim, receipt showing failure)`` at
matched narrative (faithful_to the Grossman-Milgrom unraveling result). ``S ≈ 0`` means the reward
treats a missing receipt the same as a receipt that shows failure: silence is read as failure, the
skeptical stance the unraveling argument says a sound evaluator must take. ``S ≫ 0`` means the reward
scores the claim higher when the receipt is simply absent than when it is present-and-damning: silence
is rewarded over disclosure, the credulous stance that lets an agent win by hiding its receipts.
Skepticism is the second axis of the honesty law (S15); receipt reliance (``receipt_reliance.py``) is the
first.

Deviation from A8: none in the definition. The pure function is the matched-narrative reward
difference; the production path supplies the two scores from the receipt-absent and receipt-failure
conditions of the same narrative. The synthetic test drives it with planted scores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty
from reward_lens.core.invariance import COVARIANT_LINEAR
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


def skepticism(r_receipt_absent: float, r_receipt_failure: float) -> float:
    """The skepticism statistic ``S = r(receipt absent) − r(receipt showing failure)`` (A8).

    Positive ``S`` is credulous (silence scored above a damning receipt); ``S ≈ 0`` is skeptical
    (silence treated as failure). At matched narrative the difference isolates how the reward treats the
    absence of a receipt, which is exactly the quantity the unraveling law constrains.
    """
    return float(r_receipt_absent) - float(r_receipt_failure)


def skepticism_batch(
    r_receipt_absent: np.ndarray, r_receipt_failure: np.ndarray
) -> dict[str, float]:
    """Mean skepticism and its spread over a batch of matched narratives (A8).

    Averages the per-narrative ``S`` and reports the standard deviation, so a card can show whether the
    credulity is systematic or noisy. Both inputs are length-``n`` reward vectors, aligned by narrative.
    """
    a = np.asarray(r_receipt_absent, dtype=np.float64).ravel()
    f = np.asarray(r_receipt_failure, dtype=np.float64).ravel()
    if a.size != f.size:
        raise ValueError(f"absent ({a.size}) and failure ({f.size}) arrays must align by narrative")
    s = a - f
    return {
        "skepticism": float(np.mean(s)),
        "skepticism_std": float(np.std(s, ddof=0)),
        "n": a.size,
    }


class Skepticism(BaseObservable):
    """A8 whether the reward treats a missing receipt as failure (skeptical) or reward (credulous).

    Requires span-typed inputs on the production path (the receipt-absent and receipt-failure conditions
    of matched narratives). Here the two score vectors are injected so the difference is exercised
    directly. Gauge is INVARIANT with respect to representation, though ``S`` carries reward-scale units,
    noted as a deviation.

    What it cannot do. ``S`` is a raw reward difference, so its magnitude means nothing without
    the reward's scale beside it and two graders' ``S`` values are not comparable numbers. Only the
    sign and the distance from zero carry scale-free content, which is why this instrument declares
    itself covariant rather than invariant. The matched-narrative condition is an assumption about
    the inputs that nothing here verifies: if the receipt-absent and receipt-failure conditions
    differ in anything but the receipt, ``S`` measures that instead.
    """

    name = "Skepticism"
    version = "1.0"
    capabilities = Capability.SPAN_TYPES
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A8"
    deviations = (
        "S carries reward-scale units; the skeptical-vs-credulous sign and the S~0 boundary are the "
        "scale-free content",
    )

    # -- the observable declarations ---------------------------------------
    quantity = "grader.skepticism"
    #: Both score vectors are recorded from the receipt-absent and receipt-failure conditions of an
    #: earlier matched-narrative experiment.
    requires: AccessMatrix = {Component.RECORD: Access.RECORD}
    substrates = ANY_SUBSTRATE
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.STATIONARY_GRADER, RegimeCondition.ABOVE_LOD}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: ``S`` is a difference of two rewards, so under ``r -> a*r + b`` the shift cancels and the
    #: scale does not: ``S -> a*S``. Covariant with weight 1, and declaring it invariant would be
    #: the mis-declaration `Relation` exists to catch.
    invariance = "reward.affine"
    invariance_relation = COVARIANT_LINEAR
    baselines = ("baseline.matched_narrative_null", "baseline.receipt_present_success")
    rung = 0

    def __init__(
        self,
        r_receipt_absent: np.ndarray | None = None,
        r_receipt_failure: np.ndarray | None = None,
    ) -> None:
        self.r_receipt_absent = r_receipt_absent
        self.r_receipt_failure = r_receipt_failure

    def preflight(self, ctx: Context) -> PreflightResult:
        """Both arms, matched by narrative, or a refusal.

        The injected input is absent, which makes this a `Refusal` rather than an Evidence
        carrying a note. Nothing has to be computed to know it, so the question belongs
        here: `estimate` returns this refusal before `measure` is reached, and the
        capability report gets it with no work at all.
        """
        if self.r_receipt_absent is None or self.r_receipt_failure is None:
            return missing_injection(
                self,
                needs={
                    "r_receipt_absent": "length-n rewards for the receipt-absent arm",
                    "r_receipt_failure": "length-n rewards for the receipt-failure arm, matched by narrative",
                },
                have="neither arm was injected",
                remedy=(
                    "Score the same narratives twice, once with the receipt absent and once with it present "
                    "and failing, then construct `Skepticism(r_receipt_absent=..., r_receipt_failure=...)`. "
                    "The two vectors are aligned by narrative, so they have to be the same length and in the "
                    "same order: a mismatched pairing measures the narratives rather than the credulity."
                ),
            )
        return super().preflight(ctx)

    def measure(self, ctx: Context) -> "Evidence":
        if self.r_receipt_absent is None or self.r_receipt_failure is None:
            raise measured_without_input(self)
        absent = np.atleast_1d(np.asarray(self.r_receipt_absent, dtype=np.float64))
        failure = np.atleast_1d(np.asarray(self.r_receipt_failure, dtype=np.float64))
        report = skepticism_batch(absent, failure)
        report["credulous"] = bool(report["skepticism"] > 0)
        return ctx.emit(report, uncertainty=Uncertainty(n=int(report["n"]), method="none"))


__all__ = ["skepticism", "skepticism_batch", "Skepticism"]

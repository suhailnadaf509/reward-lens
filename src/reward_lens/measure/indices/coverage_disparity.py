"""Coverage disparity: v1's coverage statistic under its honest name (A2, deviation note).

This is NOT the Wang-Huang distortion index. It is the coverage statistic v1's E10 actually computed
while citing Wang-Huang 2603.28063, the exact operationalization drift A2 calls out. The real
Wang-Huang per-dimension distortion lives in ``distortion.py`` (A2); this module keeps v1's statistic so
its numbers remain reproducible, but names it for what it measures: the disparity in reward coverage
across a set of dimensions or groups, not a distortion.

``coverage(P)`` is the fraction of a property's reward-relevant signal captured by the named/intended
channels. The disparity is the spread of that coverage across the battery: a large disparity means the
reward covers some properties well and others poorly, which is a real and reportable inequality, just
not Wang-Huang's object. Keeping it here, honestly labelled, is the structural fix that makes the
v1 drift impossible to repeat silently: a card consuming this sees ``coverage_disparity``,
never ``distortion``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

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


def coverage_disparity(coverage: Sequence[float]) -> dict[str, float]:
    """The disparity in reward coverage across a battery (v1's honestly-named coverage statistic).

    Returns the spread of the coverage values: the range ``max − min``, the standard deviation, and the
    Gini-style mean absolute difference, all summarizing how unequally the reward covers the properties.
    ``coverage`` is a length-``K`` vector in ``[0, 1]``. Uniform coverage gives zero disparity by every
    measure; a reward that covers one property fully and another not at all gives a range of one.
    """
    c = np.clip(np.asarray(coverage, dtype=np.float64).ravel(), 0.0, 1.0)
    if c.size == 0:
        return {"range": 0.0, "std": 0.0, "mean_abs_diff": 0.0, "mean_coverage": float("nan")}
    range_ = float(c.max() - c.min())
    std = float(np.std(c, ddof=0))
    mad = float(np.mean(np.abs(c[:, None] - c[None, :])))
    return {
        "range": range_,
        "std": std,
        "mean_abs_diff": mad,
        "mean_coverage": float(c.mean()),
    }


class CoverageDisparity(BaseObservable):
    """v1's coverage statistic, kept reproducible under its honest name (not Wang-Huang distortion).

    Takes a battery of per-property coverage values (injected; the concept/KUI layer supplies them in
    production) and reports their disparity. ``faithful_to`` is None on purpose: this instantiates
    no theory object, it preserves a v1 statistic, and the deviation note says so. Gauge is
    INVARIANT.

    What it cannot do. Every number here is a spread of the coverage vector it was handed, so
    the reading is a restatement of somebody else's measurement and inherits every weakness of it
    without recording which one produced it. Nothing checks that the battery is a sample of anything:
    a disparity over three hand-picked properties is a fact about the three, and it will render
    beside a disparity over three hundred with nothing to distinguish them. This is deliberately not
    Wang-Huang distortion and must never be cited as it; ``distortion.py`` is that object.
    """

    name = "CoverageDisparity"
    version = "1.0"
    capabilities = Capability.SCORES
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = None
    deviations = (
        "this is v1/E10's coverage statistic, NOT Wang-Huang distortion (A2); the distortion index "
        "lives in distortion.py. Kept under its honest name to preserve v1's numbers without the "
        "operationalization drift.",
    )

    # -- the observable declarations ---------------------------------------
    quantity = "grader.coverage_disparity"
    #: The coverage battery is a recorded measurement from the concept or KUI layer. Nothing here
    #: calls the grader.
    requires: AccessMatrix = {Component.RECORD: Access.RECORD}
    substrates = ANY_SUBSTRATE
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: None of the seven groups act on a vector of coverage fractions: an affine rescaling of the
    #: reward does not reach it, and no change of representation basis does either. `trivial` is the
    #: answer rather than an omission, and its generated test passes vacuously, which is honest.
    invariance = "trivial"
    invariance_relation = INVARIANT
    baselines = ("baseline.uniform_coverage", "baseline.shuffled_property_labels")
    rung = 0

    def __init__(self, coverage: Sequence[float] | None = None) -> None:
        self.coverage = coverage

    def preflight(self, ctx: Context) -> PreflightResult:
        """The battery or a refusal. A disparity across nothing is not zero disparity.

        The injected input is absent, which makes this a `Refusal` rather than an Evidence
        carrying a note. Nothing has to be computed to know it, so the question belongs
        here: `estimate` returns this refusal before `measure` is reached, and the
        capability report gets it with no work at all.
        """
        if self.coverage is None:
            return missing_injection(
                self,
                needs={
                    "coverage": "a length-K vector of per-property reward coverage",
                },
                have="no battery was injected",
                remedy=(
                    "Construct `CoverageDisparity(coverage=[...])` with one coverage value per property in "
                    "the battery. The reading is the spread across those values, so a battery of one "
                    "property has no disparity to report and at least two are needed."
                ),
            )
        return super().preflight(ctx)

    def measure(self, ctx: Context) -> "Evidence":
        if self.coverage is None:
            raise measured_without_input(self)
        report: dict[str, Any] = coverage_disparity(self.coverage)
        report["n_properties"] = int(np.asarray(self.coverage).size)
        return ctx.emit(report, uncertainty=Uncertainty(n=report["n_properties"], method="none"))


__all__ = ["coverage_disparity", "CoverageDisparity"]

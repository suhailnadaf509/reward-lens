"""Void: the run was not readable, which is not the same as a negative result.

A study can end three ways and only two of them are science. ``RESULT`` means the registered
predictions were adjudicated against metrics that were actually computed. ``NULL`` means the same
thing with the predictions not confirmed. ``VOID`` means the study could not be read at all, so
nothing about the hypothesis was learned and the correct response is to fix the instrument and run
it again.

The distinction earns its place because of what happened without it. The campaign that produced
this library's evidence base ran its studies through a runner that turned a missing metric into
the string "inconclusive", and turned a kill criterion whose metric was missing into a criterion
that quietly did not fire. Ten cards came back inconclusive and in every case the cause was
infrastructural: an arc had not run, or its shard had not merged, or ``resolve_subjects`` hit a
``PermissionError``. A registered kill that could not be evaluated was, in the output, exactly as
reassuring as one that was evaluated and passed.

Void has a fix and inconclusive does not. That is the whole content of the distinction: every
``VoidReason`` below carries a remedy naming the thing to repair, and a void study is a work item
rather than a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class StudyOutcome(Enum):
    """How a study ended.

    ``RESULT`` and ``NULL`` are both readings: the metrics existed and the frozen predictions were
    checked against them. ``VOID`` is not a reading. A void study says nothing about its
    hypotheses, and reporting it beside a null would be a category error.
    """

    RESULT = "result"
    NULL = "null"
    VOID = "void"


class VoidReason(Enum):
    """The seven declared void conditions, the arc-level one that subsumes condition 5, and one more.

    Conditions 1 through 7 are the list this project adopted from a published preregistration that
    got the distinction right first. They are declared here rather than discovered per study so
    that a void is always a named condition and never an improvisation.

    Condition 8 is ours and is an amendment rather than an adoption. The seven were written for a
    two-arm training comparison where the anticipated failure is arms drifting apart, which is
    condition 4. Where the contrast is applied by configuration override rather than by hand, the
    symmetric failure is the override never reaching the trainer, and that one is strictly worse:
    an arm divergence announces itself, while a contrast that never applied produces a tidy null
    that reads as a result.
    """

    #: 1. Either arm's training collapsed: a reward or entropy pathology. Fix the loop, do not read.
    ARM_COLLAPSE = "arm_collapse"

    #: 2. Scorer disagreement on a hand-audited sample exceeded the frozen threshold.
    SCORER_DISAGREEMENT = "scorer_disagreement"

    #: 3. Decontamination breach: test material reached the training distribution.
    DECONTAMINATION_BREACH = "decontamination_breach"

    #: 4. Two arms of a controlled comparison differ in configuration beyond the declared contrast.
    ARM_DIVERGENCE = "arm_divergence"

    #: 5. A registered metric could not be computed. Never a pass, and never a silent non-firing.
    METRIC_ABSENT = "metric_absent"

    #: 6. A regime condition the primary instrument requires failed over more than the stated
    #: fraction of the measurement window.
    REGIME_VIOLATED = "regime_violated"

    #: 7. The instrument effect exceeded its budget over more than the stated fraction of steps, so
    #: the measurement perturbed the run it was measuring.
    INSTRUMENT_PERTURBED = "instrument_perturbed"

    #: 8. The declared contrast did not differ between the arms, so the comparison has no contrast.
    #: Condition 4 with the sign flipped, and added here rather than adopted.
    CONTRAST_INERT = "contrast_inert"


#: What to do about each condition. These strings are read by whoever is holding the failed run, so
#: they are written as instructions rather than as diagnoses. "Envelope violated" is not a remedy.
DEFAULT_REMEDY: dict[VoidReason, str] = {
    VoidReason.ARM_COLLAPSE: (
        "Repair the training loop before reading this study. Check the reward and entropy traces "
        "for the collapsing arm, then re-run at the frozen n and steps."
    ),
    VoidReason.SCORER_DISAGREEMENT: (
        "The scorer disagrees with the hand audit beyond the frozen threshold, so its scores are "
        "not a usable measurement here. Re-audit a fresh sample, or replace the scorer and "
        "re-freeze."
    ),
    VoidReason.DECONTAMINATION_BREACH: (
        "Test material reached the training distribution. Rebuild the split, re-run the "
        "decontamination check, and re-freeze the study."
    ),
    VoidReason.ARM_DIVERGENCE: (
        "The arms differ outside the declared contrast, so any difference between them is not "
        "attributable. Re-run with the diverging keys held fixed, or widen the declared contrast "
        "and re-freeze."
    ),
    VoidReason.METRIC_ABSENT: (
        "The arc that produces this metric did not run, or its shard was not merged. Re-run that "
        "arc and merge its shard, then re-adjudicate; the study does not need to be re-frozen."
    ),
    VoidReason.REGIME_VIOLATED: (
        "The primary instrument's regime precondition failed over too much of the window. "
        "Restrict the window to the steps where it holds, or use an estimator whose envelope "
        "admits this regime."
    ),
    VoidReason.INSTRUMENT_PERTURBED: (
        "The measurement exceeded its own overhead budget, so it perturbed the run. Lower the "
        "capture rate or move the measurement out of the hot path, then re-run."
    ),
    VoidReason.CONTRAST_INERT: (
        "The keys you declared as the contrast hold the same value in both arms, so nothing was "
        "varied and any difference between them is sampling noise. Check that the override reached "
        "the trainer, then re-run; the study does not need to be re-frozen."
    ),
}


@dataclass(frozen=True)
class Void:
    """A named, remediable reason that something could not be read.

    ``detail`` carries the numbers and the names: which metric was absent, which hypothesis or
    kill criterion wanted it, which arc was supposed to produce it. ``arc`` is the field that turns
    a void from a complaint into a work item, and it is why the runner records, per prediction,
    which arc was meant to produce its metric.
    """

    reason: VoidReason
    detail: str
    remedy: str = ""
    arc: str | None = None

    def __post_init__(self) -> None:
        if not self.remedy:
            object.__setattr__(self, "remedy", DEFAULT_REMEDY[self.reason])

    def __str__(self) -> str:
        arc = f" [arc: {self.arc}]" if self.arc else ""
        return f"VOID({self.reason.value}){arc}: {self.detail} Remedy: {self.remedy}"


#: Per-hypothesis adjudication. There is deliberately no "inconclusive" member: a hypothesis whose
#: metric was not computed is ``void``, and a void carries a reason and a remedy.
HypothesisOutcome = Literal["confirmed", "refuted", "void"]

#: Per-kill-criterion adjudication. ``passed`` means the criterion was evaluated and did not fire.
#: ``void`` means it could not be evaluated, which is the case the old runner rendered as a
#: non-firing and is the single most damaging thing it did.
KillOutcome = Literal["fired", "passed", "void"]


__all__ = [
    "DEFAULT_REMEDY",
    "HypothesisOutcome",
    "KillOutcome",
    "StudyOutcome",
    "Void",
    "VoidReason",
]

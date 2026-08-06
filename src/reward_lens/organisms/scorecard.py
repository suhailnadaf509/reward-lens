"""`MethodScorecard`: the ROC of an instrument recovering a planted structure.

Calibrating an instrument means grading it against an answer key: how well does it recover the
structure the foundry planted? That is an ROC/PR question, and this module is where an instrument
earns its calibration. `MethodScorecard.evaluate` takes an instrument's scores on organism data plus
the `AnswerKey`, computes the answer-key ROC via `reward_lens.stats.roc`, and returns a
`ScorecardEntry` carrying the AUC per dose, the operating points ("flags biases stronger than
rho=0.75 at 90% TPR / 5% FPR"), whether the recovery is monotone in the dose, the stored `Evidence`,
and the `CalibrationRef` a downstream measurement cites to satisfy gate 1.

Two properties make a scorecard trustworthy and are the object of the pure test
(`tests/test_organisms_scorecard.py`): a stronger planted signal must be easier to recover, so the
AUC is **monotone in the dose rho**; and the operating points must be read correctly off the curve.
Both are testable now with a synthetic detector whose separability scales with rho
(`synthetic_dose_detector`); the real DLA/patching scorecard is wired to the battery, when its
Observables exist to be graded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from reward_lens.core import (
    CalibrationRef,
    Evidence,
    GaugeStatus,
    SubjectRef,
    Uncertainty,
    capture_provenance,
    make_evidence,
    register_payload,
)
from reward_lens.organisms.spec import AnswerKey
from reward_lens.stats.roc import operating_point, roc_pr

# A detector's readout on one regime: per-item real-valued scores (higher meaning "more likely the
# planted structure is present") and the binary answer-key labels (1 = planted, 0 = not).
DetectorReadout = tuple[np.ndarray, np.ndarray]

_SCORECARD_VERSION = "1.0"


@register_payload
@dataclass
class ScorecardSummary:
    """The stored numeric scorecard of an instrument against a planted structure.

    This is the Evidence payload. ``doses`` are the planted dose levels (rho) in ascending order;
    ``aucs`` and ``average_precisions`` are the answer-key ROC-AUC and PR-AP at each dose;
    ``operating_points`` is the per-dose ``{rho, tpr, fpr, threshold}`` at the target FPR;
    ``is_monotone`` is whether the AUC is non-decreasing in rho (a stronger signal is easier to
    recover, the property that makes a scorecard sane); ``monotone_spearman`` is the rank correlation
    between rho and AUC; ``detects_rho_at`` is the smallest dose the instrument flags at the target
    operating point, the number behind the headline claim.
    """

    observable: str
    organism_family: str
    doses: list[float]
    aucs: list[float]
    average_precisions: list[float]
    operating_points: list[dict[str, float]]
    is_monotone: bool
    monotone_spearman: float
    detects_rho_at: float | None
    target_tpr: float
    target_fpr: float
    n_per_dose: list[int] = field(default_factory=list)

    def __canonical__(self) -> dict:
        return {
            "observable": self.observable,
            "organism_family": self.organism_family,
            "doses": [float(d) for d in self.doses],
            "aucs": [float(a) for a in self.aucs],
            "average_precisions": [float(a) for a in self.average_precisions],
            "operating_points": [
                {k: float(v) for k, v in op.items()} for op in self.operating_points
            ],
            "is_monotone": bool(self.is_monotone),
            "monotone_spearman": float(self.monotone_spearman),
            "detects_rho_at": None if self.detects_rho_at is None else float(self.detects_rho_at),
            "target_tpr": float(self.target_tpr),
            "target_fpr": float(self.target_fpr),
            "n_per_dose": [int(n) for n in self.n_per_dose],
        }


@dataclass(frozen=True)
class ScorecardEntry:
    """The result bundle from `MethodScorecard.evaluate`.

    Wraps the stored ``summary`` (the Evidence payload) with the ``evidence`` object it is stored as
    and the ``calibration_ref`` a downstream measurement cites to become CALIBRATED (gate 1). The
    passthrough properties expose the summary's headline numbers so a caller reads ``entry.aucs`` and
    ``entry.is_monotone`` directly.
    """

    summary: ScorecardSummary
    evidence: Evidence
    calibration_ref: CalibrationRef

    @property
    def aucs(self) -> list[float]:
        return self.summary.aucs

    @property
    def doses(self) -> list[float]:
        return self.summary.doses

    @property
    def is_monotone(self) -> bool:
        return self.summary.is_monotone

    @property
    def monotone_spearman(self) -> float:
        return self.summary.monotone_spearman

    @property
    def operating_points(self) -> list[dict[str, float]]:
        return self.summary.operating_points

    @property
    def detects_rho_at(self) -> float | None:
        return self.summary.detects_rho_at


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation between two vectors (NaN-safe, ties by average rank).

    Computed as the Pearson correlation of the average ranks. Returns ``nan`` if fewer than two finite
    pairs are present or either ranked vector is constant (correlation undefined there).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return float("nan")
    rx = _avg_rank(x)
    ry = _avg_rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _avg_rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(a.size, dtype=np.float64)
    # Average ties.
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size, dtype=np.float64)
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return np.asarray(avg[inv], dtype=np.float64)


def _is_nondecreasing(values: list[float], tol: float) -> bool:
    """Whether ``values`` is non-decreasing within ``tol`` (ignoring NaNs)."""
    finite = [v for v in values if np.isfinite(v)]
    return all(finite[i + 1] >= finite[i] - tol for i in range(len(finite) - 1))


class MethodScorecard:
    """Grades an instrument against an organism's answer key.

    An instance names the instrument being graded (``observable``). `evaluate` consumes the
    instrument's per-item scores, either on a single regime or across a dose sweep, and produces a
    `ScorecardEntry`. The scorecard is agnostic to *what* produced the scores: the pure test grades a
    synthetic detector, the battery grades DLA and patching, and the code path is identical.
    """

    def __init__(self, observable: str, *, version: str = _SCORECARD_VERSION) -> None:
        self.observable = observable
        self.version = version

    def evaluate(
        self,
        observable_scores: DetectorReadout | Mapping[float, DetectorReadout],
        answer_key: AnswerKey,
        doses: list[float] | None = None,
        *,
        target_tpr: float = 0.90,
        target_fpr: float = 0.05,
        monotone_tol: float = 0.02,
    ) -> ScorecardEntry:
        """Compute the answer-key ROC of the instrument and return a `ScorecardEntry`.

        Args:
            observable_scores: Either a single ``(scores, labels)`` readout, or a mapping from dose
                (rho) to a ``(scores, labels)`` readout for a dose sweep. Scores are real-valued with
                higher meaning "the planted structure is present"; labels are binary answer-key truth.
            answer_key: The organism's `AnswerKey` (names the family the scorecard certifies).
            doses: Explicit dose ordering; defaults to the sorted keys of a mapping, or ``[1.0]`` for a
                single readout.
            target_tpr: The TPR the operating-point / detection claim is stated at.
            target_fpr: The FPR ceiling the operating point respects.
            monotone_tol: Tolerance for the non-decreasing check on AUC across doses.

        Returns:
            A `ScorecardEntry` whose AUC is monotone in the dose for a well-behaved instrument, with
            per-dose operating points, a stored `Evidence`, and a `CalibrationRef` for gate 1.
        """
        readouts, dose_values = self._normalize(observable_scores, doses)

        aucs: list[float] = []
        aps: list[float] = []
        ops: list[dict[str, float]] = []
        n_per_dose: list[int] = []
        detects_rho_at: float | None = None
        detect_op: dict[str, float] | None = None

        for rho, (scores, labels) in zip(dose_values, readouts):
            scores = np.asarray(scores, dtype=np.float64).ravel()
            labels = np.asarray(labels, dtype=np.float64).ravel()
            roc = roc_pr(scores, labels)
            op = operating_point(roc, target_fpr=target_fpr)
            aucs.append(float(roc.auc))
            aps.append(float(roc.average_precision))
            op_row = {
                "rho": float(rho),
                "tpr": op["tpr"],
                "fpr": op["fpr"],
                "threshold": op["threshold"],
            }
            ops.append(op_row)
            n_per_dose.append(int(labels.size))
            if (
                detects_rho_at is None
                and np.isfinite(op["tpr"])
                and op["tpr"] >= target_tpr
                and op["fpr"] <= target_fpr + 1e-9
            ):
                detects_rho_at = float(rho)
                detect_op = op_row

        monotone_spearman = _spearman(np.array(dose_values), np.array(aucs))
        is_monotone = _is_nondecreasing(aucs, monotone_tol)

        summary = ScorecardSummary(
            observable=self.observable,
            organism_family=answer_key.family,
            doses=[float(d) for d in dose_values],
            aucs=aucs,
            average_precisions=aps,
            operating_points=ops,
            is_monotone=is_monotone,
            monotone_spearman=float(monotone_spearman),
            detects_rho_at=detects_rho_at,
            target_tpr=target_tpr,
            target_fpr=target_fpr,
            n_per_dose=n_per_dose,
        )

        evidence = self._build_evidence(summary, answer_key)
        calibration_ref = CalibrationRef(
            scorecard_entry=evidence.id,
            organism_family=answer_key.family,
            regime_match="exact",
            operating_point=detect_op,
        )
        return ScorecardEntry(summary=summary, evidence=evidence, calibration_ref=calibration_ref)

    def _normalize(
        self,
        observable_scores: DetectorReadout | Mapping[float, DetectorReadout],
        doses: list[float] | None,
    ) -> tuple[list[DetectorReadout], list[float]]:
        """Resolve the input into an ordered list of readouts and their dose values."""
        if isinstance(observable_scores, Mapping):
            keys = doses if doses is not None else sorted(observable_scores.keys())
            return [observable_scores[k] for k in keys], [float(k) for k in keys]
        # Single readout: one implicit dose (1.0 unless a dose was named).
        dose = float(doses[0]) if doses else 1.0
        return [observable_scores], [dose]

    def _build_evidence(self, summary: ScorecardSummary, answer_key: AnswerKey) -> Evidence:
        """Wrap a `ScorecardSummary` as INVARIANT Evidence (an AUC is gauge-invariant)."""
        subject = SubjectRef(
            extra={
                "organism_family": answer_key.family,
                "observable": self.observable,
                "kind": "method-scorecard",
            }
        )
        provenance = capture_provenance(
            config={"observable": self.observable, "version": self.version},
        )
        return make_evidence(
            observable=f"MethodScorecard[{self.observable}]",
            observable_version=self.version,
            subject=subject,
            value=summary,
            uncertainty=Uncertainty(
                n=int(sum(summary.n_per_dose)) if summary.n_per_dose else None,
                method="answer-key-roc",
            ),
            gauge=GaugeStatus.INVARIANT,
            provenance=provenance,
        )


def synthetic_dose_detector(
    rho: float, *, n: int = 400, seed: int = 0, slope: float = 6.0
) -> DetectorReadout:
    """A stand-in detector whose separability scales with the dose rho.

    Produces ``n`` positive and ``n`` negative items. Positive scores are drawn from ``N(mu, 1)`` and
    negatives from ``N(0, 1)`` with ``mu = slope * (rho - 0.5)``: at ``rho = 0.5`` the classes overlap
    (AUC near 0.5) and the separation grows with rho. This is the fixture that proves the scorecard
    machinery is monotone in rho before any real instrument exists, and the calibration reference the
    micro-organism can fall back on. It is not a claim about any real detector.
    """
    rng = np.random.default_rng([int(seed), int(round(rho * 1000))])
    mu = slope * (float(rho) - 0.5)
    pos = rng.normal(mu, 1.0, size=n)
    neg = rng.normal(0.0, 1.0, size=n)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    return scores, labels


__all__ = [
    "DetectorReadout",
    "ScorecardSummary",
    "ScorecardEntry",
    "MethodScorecard",
    "synthetic_dose_detector",
]

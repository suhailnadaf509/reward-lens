"""ROC/PR curves, calibration, and operating points: the scorecard's engine.

The organism foundry certifies an instrument by asking how well it recovers a
planted structure it already knows the answer to. That question is an ROC/PR
question, and this module is where those numbers are computed:
answer-key ROC and precision-recall curves, reliability (calibration) diagrams,
and the operating-point lookup behind statements like "flags biases stronger
than rho=0.75 at 90% TPR / 5% FPR".

The AUC is computed from the rank / Mann-Whitney identity rather than by
integrating the curve, because the rank form is exact under ties and needs no
threshold grid. scikit-learn is used for the curve arrays when it is installed,
but the module imports and runs without it: a self-contained numpy path
produces the same results, so the epistemics layer keeps its light dependency
footprint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reward_lens.stats.effects import _rankdata

try:  # scikit-learn is an optional convenience, never a hard dependency (R14).
    from sklearn import metrics as _sk_metrics

    _HAVE_SKLEARN = True
except ImportError:  # pragma: no cover - exercised only in sklearn-free envs
    _sk_metrics = None
    _HAVE_SKLEARN = False


@dataclass
class ROCResult:
    """ROC and precision-recall curves with their summary scalars.

    Attributes:
        fpr: False-positive rate at each ROC threshold, ascending from 0.
        tpr: True-positive rate at each ROC threshold, ascending from 0.
        thresholds: Score threshold for each ROC point; the first is above the
            maximum score so the curve starts at the (0, 0) origin.
        auc: Area under the ROC curve via the Mann-Whitney identity;
            ``nan`` if either class is empty.
        precision: Precision at each PR threshold, in order of increasing recall.
        recall: Recall at each PR threshold, ascending.
        average_precision: Step-wise area under the PR curve;
            ``nan`` if the positive class is empty.
    """

    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float
    precision: np.ndarray
    recall: np.ndarray
    average_precision: float


def _auc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via the rank identity: the probability a random positive outranks a random negative.

    ``AUC = (sum of ranks of positives - P(P+1)/2) / (P * N)`` using average
    ranks, so ties contribute exactly 0.5. Returns ``nan`` if either class is
    empty (the AUC is undefined there rather than 0 or 1).
    """
    n_pos = float(np.sum(labels == 1))
    n_neg = float(labels.size) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    rank_sum_pos = float(np.sum(ranks[labels == 1]))
    return (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def _curves_numpy(scores: np.ndarray, labels: np.ndarray):
    """ROC and PR curve arrays in pure numpy.

    Sorts by descending score and accumulates true/false positives, keeping one
    point per distinct score (the last row of each tie run). Returns
    ``(fpr, tpr, thresholds, precision, recall, average_precision)``.
    """
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = labels[order].astype(np.float64)

    tps = np.cumsum(y)
    fps = np.cumsum(1.0 - y)
    # Keep the last index of each run of equal scores: those are the real thresholds.
    threshold_idx = np.r_[np.where(np.diff(s))[0], s.size - 1]
    tps = tps[threshold_idx]
    fps = fps[threshold_idx]
    thr = s[threshold_idx]

    n_pos = tps[-1]
    n_neg = fps[-1]

    tpr = tps / n_pos
    fpr = fps / n_neg
    # Prepend the origin with a threshold above the max score, matching the
    # sklearn convention so the ROC starts at (0, 0).
    fpr = np.r_[0.0, fpr]
    tpr = np.r_[0.0, tpr]
    thresholds = np.r_[thr[0] + 1.0, thr]

    precision = tps / (tps + fps)
    recall = tps / n_pos
    # Step-wise average precision: sum of precision * (delta recall).
    recall_with_origin = np.r_[0.0, recall]
    average_precision = float(np.sum(np.diff(recall_with_origin) * precision))

    return fpr, tpr, thresholds, precision, recall, average_precision


def roc_pr(scores: np.ndarray, labels: np.ndarray) -> ROCResult:
    """Compute ROC and PR curves from scores and binary labels.

    Args:
        scores: Real-valued scores, higher meaning "more positive".
        labels: Binary labels in {0, 1}; label 1 is the positive class.

    Returns:
        A ``ROCResult``. Non-finite scores or labels are dropped first. If after
        that either class is empty, the AUC and average precision are ``nan`` and
        the curve arrays are a single ``nan`` (the curves are undefined without
        both classes), so the caller never divides by an empty class.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    if scores.size != labels.size:
        raise ValueError(f"scores and labels must align; got {scores.size} vs {labels.size}")

    mask = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[mask]
    labels = labels[mask]

    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if scores.size == 0 or n_pos == 0 or n_neg == 0:
        nan1 = np.array([np.nan], dtype=np.float64)
        return ROCResult(
            fpr=nan1.copy(),
            tpr=nan1.copy(),
            thresholds=nan1.copy(),
            auc=float("nan"),
            precision=nan1.copy(),
            recall=nan1.copy(),
            average_precision=float("nan"),
        )

    auc = _auc_mann_whitney(scores, labels)

    if _HAVE_SKLEARN:
        fpr, tpr, thresholds = _sk_metrics.roc_curve(labels, scores)
        precision, recall, _ = _sk_metrics.precision_recall_curve(labels, scores)
        average_precision = float(_sk_metrics.average_precision_score(labels, scores))
    else:
        fpr, tpr, thresholds, precision, recall, average_precision = _curves_numpy(scores, labels)

    return ROCResult(
        fpr=np.asarray(fpr, dtype=np.float64),
        tpr=np.asarray(tpr, dtype=np.float64),
        thresholds=np.asarray(thresholds, dtype=np.float64),
        auc=float(auc),
        precision=np.asarray(precision, dtype=np.float64),
        recall=np.asarray(recall, dtype=np.float64),
        average_precision=float(average_precision),
    )


def calibration_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Reliability-diagram data and the expected calibration error.

    Bins the predicted probabilities into ``n_bins`` equal-width bins over
    [0, 1] and, per bin, reports the empirical accuracy (mean label) against the
    mean confidence (mean probability). A perfectly calibrated model has
    accuracy equal to confidence in every bin.

    Args:
        probs: Predicted probabilities in [0, 1].
        labels: Binary outcomes in {0, 1}.
        n_bins: Number of equal-width probability bins.

    Returns:
        A dict with ``bin_centers``, ``bin_accuracy``, ``bin_confidence`` (each
        length ``n_bins``, with ``nan`` for empty bins), ``bin_counts``, and
        ``ece``. The expected calibration error is the count-weighted mean gap
        ``sum_b (n_b / N) * |acc_b - conf_b|`` over non-empty bins, which lies in
        [0, 1].
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    mask = np.isfinite(probs) & np.isfinite(labels)
    probs = probs[mask]
    labels = labels[mask]

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # digitize against the interior edges yields bin indices 0..n_bins-1;
    # clip keeps prob == 1.0 in the last bin rather than spilling past it.
    bin_of = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)

    accuracy = np.full(n_bins, np.nan, dtype=np.float64)
    confidence = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=int)
    total = probs.size
    ece = 0.0
    for b in range(n_bins):
        sel = bin_of == b
        c = int(np.sum(sel))
        counts[b] = c
        if c > 0:
            accuracy[b] = float(np.mean(labels[sel]))
            confidence[b] = float(np.mean(probs[sel]))
            ece += (c / total) * abs(accuracy[b] - confidence[b])

    return {
        "bin_centers": centers,
        "bin_accuracy": accuracy,
        "bin_confidence": confidence,
        "bin_counts": counts,
        "ece": float(ece),
    }


def operating_point(roc: ROCResult, target_fpr: float = 0.05) -> dict:
    """The operating point that maximizes TPR subject to an FPR cap.

    Among the ROC points whose false-positive rate is at most ``target_fpr``,
    return the one with the highest true-positive rate. Ties in TPR are broken
    toward the larger threshold (the more conservative score cutoff that still
    reaches that TPR), which is why the search walks the ascending-FPR arrays and
    takes the first maximizer. This is the operating point behind scorecard
    claims of the form "detects the planted effect at X% TPR and Y% FPR".

    Args:
        roc: A ``ROCResult`` from ``roc_pr``.
        target_fpr: The false-positive-rate ceiling.

    Returns:
        A dict with ``threshold``, ``tpr``, and ``fpr`` at the chosen point, or
        all-``nan`` if the ROC is degenerate (an empty class).
    """
    fpr = np.asarray(roc.fpr, dtype=np.float64)
    tpr = np.asarray(roc.tpr, dtype=np.float64)
    thresholds = np.asarray(roc.thresholds, dtype=np.float64)

    finite = np.isfinite(fpr) & np.isfinite(tpr)
    valid = np.where(finite & (fpr <= target_fpr))[0]
    if valid.size == 0:
        return {"threshold": float("nan"), "tpr": float("nan"), "fpr": float("nan")}
    best = valid[int(np.argmax(tpr[valid]))]
    return {
        "threshold": float(thresholds[best]),
        "tpr": float(tpr[best]),
        "fpr": float(fpr[best]),
    }


__all__ = [
    "ROCResult",
    "roc_pr",
    "calibration_curve",
    "operating_point",
]

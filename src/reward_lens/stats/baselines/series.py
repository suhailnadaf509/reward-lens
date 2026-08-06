"""The gradient-norm peak: the sixth baseline, and the only one that reads a number not a string.

Anyone who has watched a training run knows the gradient norm spikes when something changes. That
is the whole baseline, and it is the one an onset detector or a monitor has to beat: if a CUSUM on
a learned signal fires no earlier than the point where the gradient norm peaked, the learned
signal contributed nothing an existing dashboard did not already show.

This reads a **logged** scalar series. It does not compute gradients and this package imports no
autodiff framework to do so, which is why the baseline is available at RECORD access on a run
somebody else trained.

Two readings come out of it, and they answer different questions. Scored per item against a label
it is a discrimination baseline like the other five, so it fits the same bank. Scored as an onset
it returns a `ChangePoint` that composes with `stats.changepoint.onset_lead_time`, which is what a
later monitoring claim is measured against.
"""

from __future__ import annotations

import numpy as np

from reward_lens.core.quantity import CostModel
from reward_lens.stats.baselines.base import (
    BaseBaseline,
    BaselineScore,
    DetectionTask,
    accuracy_at_midpoint,
    auroc,
    oriented_score,
)
from reward_lens.stats.changepoint import ChangePoint


def smooth(series: np.ndarray, window: int = 5) -> np.ndarray:
    """A centred moving average, so a single noisy step does not become the peak.

    Reflect-padded at both ends, which keeps the smoothed series the same length as the input and
    stops the first and last few steps being pulled toward zero by padding that is not data.
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    w = max(1, int(window))
    if w <= 1 or x.size < 2:
        return x.copy()
    w = min(w, x.size)
    pad = w // 2
    padded = np.pad(x, pad, mode="reflect")
    kernel = np.ones(w, dtype=np.float64) / w
    out = np.convolve(padded, kernel, mode="same")[pad : pad + x.size]
    return out


def gradnorm_peak(series: np.ndarray, *, window: int = 5) -> ChangePoint:
    """Where the smoothed logged series peaks, with the peak's height in baseline sigmas.

    ``strength`` is the peak expressed in standard deviations above the series median, so a run
    whose gradient norm is flat reports a peak with a strength near zero and a reader can see that
    the baseline had nothing to say rather than reading its argmax as a detection.
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    if x.size < 3 or not np.any(np.isfinite(x)):
        return ChangePoint(None, 0.0, "gradnorm-peak")
    s = smooth(x, window)
    index = int(np.nanargmax(s))
    centre = float(np.nanmedian(s))
    spread = float(np.nanstd(s))
    strength = (float(s[index]) - centre) / spread if spread > 0 else 0.0
    return ChangePoint(index, strength, "gradnorm-peak")


class GradNormPeak(BaseBaseline):
    """The logged gradient norm, as a per-item score. One fitted parameter: the sign.

    The sign is fitted out of fold because "a bigger gradient means trouble" is an assumption
    rather than a fact: on some runs the interesting steps are the ones where the norm collapsed.
    Letting the data pick the direction and paying one parameter for it is more honest than
    hard-coding the direction everyone expects.
    """

    id = "baseline.gradnorm_peak"
    name = "gradient-norm peak"
    version = "1.0"
    reads = ("series",)
    cost = CostModel(note="free; reads a scalar already in the log")
    supply = (
        "pass `series=` on the DetectionTask: one logged scalar per item, usually the optimizer "
        "gradient norm already in your training log. Nothing here differentiates anything, so "
        "RECORD access to somebody else's run is enough."
    )

    def score(self, task: DetectionTask) -> BaselineScore:
        raw = np.asarray(task.series, dtype=np.float64)
        scores, direction = oriented_score(raw, task.labels, seed=0)
        peak = gradnorm_peak(raw)
        return BaselineScore(
            baseline=self.id,
            auroc=auroc(scores, task.labels),
            accuracy=accuracy_at_midpoint(scores, task.labels),
            n=task.n,
            n_parameters=1,
            scores=scores,
            direction=direction,
            fitted=True,
            detail=(
                f"logged scalar per item; peak of the smoothed series at index {peak.index} "
                f"at {peak.strength:.2f} sigma; sign fitted out of fold "
                f"({'higher means positive' if direction > 0 else 'lower means positive'})"
            ),
        )


__all__ = ["GradNormPeak", "gradnorm_peak", "smooth"]

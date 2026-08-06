"""Online changepoint and onset detection.

The recorder needs to answer "when did this feature start drifting?" before the reward and KL
curves visibly move, which is the whole point of the flight recorder: catch the onset early. Two
complementary detectors live here. Page's CUSUM is a fast sequential test for the first sample at
which a running sum of deviations crosses a threshold; it is what an `OnsetAlarm` runs online.
Bayesian online changepoint detection (Adams and MacKay 2007) maintains a posterior over the
run length since the last change and is the more careful retrospective tool, reporting the most
probable changepoint and a confidence.

Both are pure numpy and NaN-safe. They return the onset index (or None) plus a strength, so a
caller can compare the onset of a monitored feature against the onset of a gold-reward divergence
and report the lead time between them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChangePoint:
    """A detected onset: the index it began, a detection strength, and the method used."""

    index: int | None
    strength: float
    method: str

    @property
    def detected(self) -> bool:
        return self.index is not None


def cusum(
    series: np.ndarray | list[float],
    threshold: float = 5.0,
    drift: float = 0.5,
    baseline: int | None = None,
) -> ChangePoint:
    """Page's two-sided CUSUM onset detector.

    Standardizes the series against a baseline window (the first ``baseline`` samples, or the whole
    series if None), then accumulates positive and negative deviations beyond a small ``drift``
    slack. The onset is the first index at which either accumulator exceeds ``threshold``. This is
    the sequential test the recorder runs online: it reflects the true start of the drift, not the
    point at which the drift became large, which is what makes the reported lead time honest.
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = x.size
    if n < 3:
        return ChangePoint(None, 0.0, "cusum")
    b = x[: baseline if baseline else n]
    mu = float(np.nanmean(b))
    sd = float(np.nanstd(b))
    # Degenerate has to mean "degenerate to floating point", not "exactly zero". A run of identical
    # float64 values has a computed standard deviation around 1.7e-18 rather than 0.0, so an
    # equality guard misses it, the standardisation divides by numerical noise, and the accumulator
    # crosses on rounding error. Measured: a series constant at 0.01 for 120 steps and then 0.6
    # reports an onset at index 10 with `baseline=60`, which is a confident changepoint inside the
    # flat part. Scaled to the data, because 1e-18 is degenerate at unit scale and ordinary at 1e-15.
    scale = float(np.nanmax(np.abs(b))) if b.size else 0.0
    if not np.isfinite(sd) or sd <= 1e-12 * max(scale, 1.0):
        sd = float(np.nanstd(x))
        if not np.isfinite(sd) or sd <= 1e-12 * max(scale, 1.0):
            sd = 1.0
    z = (x - mu) / sd
    hi = lo = 0.0
    peak = 0.0
    for i in range(n):
        zi = z[i] if np.isfinite(z[i]) else 0.0
        hi = max(0.0, hi + zi - drift)
        lo = min(0.0, lo + zi + drift)
        peak = max(peak, hi, -lo)
        if hi > threshold or -lo > threshold:
            return ChangePoint(i, float(peak), "cusum")
    return ChangePoint(None, float(peak), "cusum")


def bocpd(
    series: np.ndarray | list[float],
    hazard: float = 1.0 / 50.0,
    mean0: float = 0.0,
    var0: float = 1.0,
    obs_var: float = 1.0,
) -> ChangePoint:
    """Bayesian online changepoint detection (Adams and MacKay 2007) with a Gaussian model.

    Maintains the run-length posterior under a constant hazard and a conjugate Gaussian
    observation model, then reports the single most probable changepoint (the index at which the
    expected run length drops sharply) and the posterior mass there as the strength. This is the
    retrospective, uncertainty-aware complement to CUSUM: where CUSUM says "an onset happened by
    here", BOCPD says "the change was most probably here, with this confidence".
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = x.size
    if n < 3:
        return ChangePoint(None, 0.0, "bocpd")
    # Run-length distribution R[t]; track sufficient statistics per run length.
    max_rl = n + 1
    R = np.zeros(max_rl)
    R[0] = 1.0
    means = np.full(max_rl, mean0)
    vars_ = np.full(max_rl, var0)
    counts = np.zeros(max_rl)
    expected_rl = np.zeros(n)
    for t in range(n):
        xt = x[t] if np.isfinite(x[t]) else means[0]
        rl = min(t + 1, max_rl - 1)
        pred_var = vars_[:rl] + obs_var
        pred = np.exp(-0.5 * (xt - means[:rl]) ** 2 / pred_var) / np.sqrt(2 * np.pi * pred_var)
        growth = R[:rl] * pred * (1 - hazard)
        cp = float(np.sum(R[:rl] * pred * hazard))
        newR = np.zeros(max_rl)
        newR[1 : rl + 1] = growth
        newR[0] = cp
        s = newR.sum()
        R = newR / s if s > 0 else newR
        # update conjugate posteriors: shift stats up by one run length
        new_means = np.full(max_rl, mean0)
        new_vars = np.full(max_rl, var0)
        new_counts = np.zeros(max_rl)
        for k in range(rl, 0, -1):
            c = counts[k - 1] + 1
            m = (means[k - 1] * counts[k - 1] + xt) / c
            new_counts[k] = c
            new_means[k] = m
            new_vars[k] = var0 / c
        new_counts[0] = 0
        counts, means, vars_ = new_counts, new_means, new_vars
        expected_rl[t] = float(np.sum(np.arange(max_rl) * R))
    # A changepoint shows as a drop in expected run length below its running max.
    drops = np.maximum.accumulate(expected_rl) - expected_rl
    idx = int(np.argmax(drops))
    strength = float(drops[idx] / (np.max(expected_rl) + 1e-9))
    if strength < 0.25:
        return ChangePoint(None, strength, "bocpd")
    return ChangePoint(idx, strength, "bocpd")


def onset_lead_time(monitor: ChangePoint, gold: ChangePoint) -> int | None:
    """The number of steps by which a monitor's onset precedes a gold-divergence onset.

    Positive means the monitor fired earlier than the behavioural divergence, which is the recorder
    hypothesis worth its name. Returns None if either onset was not detected.
    """
    if monitor.index is None or gold.index is None:
        return None
    return gold.index - monitor.index


__all__ = ["ChangePoint", "cusum", "bocpd", "onset_lead_time"]

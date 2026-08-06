"""Onset detection recovers a planted changepoint.

The recorder's lead-time claim rests on detecting the start of a drift, not the point at which it
became large. These tests plant a changepoint at a known index in an otherwise-stationary series
and confirm both detectors find it, and that a monitored feature onset that precedes a gold onset
yields a positive lead time.
"""

from __future__ import annotations

import numpy as np

from reward_lens.stats.changepoint import bocpd, cusum, onset_lead_time


def _series_with_change(onset: int, n: int = 120, shift: float = 3.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x[onset:] += shift
    return x


def test_cusum_finds_planted_onset():
    # A clean, well-separated change with a conservative threshold: the ARL to false alarm is long,
    # so the accumulator stays quiet through the pre-change region and fires shortly after the true
    # onset rather than on a noise run.
    x = _series_with_change(onset=60, shift=4.0)
    cp = cusum(x, threshold=8.0, baseline=50)
    assert cp.detected
    # Detected near the true onset (CUSUM fires shortly after the change begins).
    assert 58 <= cp.index <= 80


def test_cusum_quiet_on_stationary_series():
    rng = np.random.default_rng(1)
    cp = cusum(rng.standard_normal(120), threshold=6.0)
    assert not cp.detected


def test_bocpd_finds_planted_onset():
    x = _series_with_change(onset=70, shift=4.0)
    cp = bocpd(x, hazard=1 / 50)
    assert cp.detected
    assert abs(cp.index - 70) <= 12


def test_lead_time_is_positive_when_monitor_precedes_gold():
    monitor = cusum(_series_with_change(onset=40), baseline=30)
    gold = cusum(_series_with_change(onset=55), baseline=30)
    lead = onset_lead_time(monitor, gold)
    assert lead is not None and lead > 0

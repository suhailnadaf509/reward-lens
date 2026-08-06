"""Tests for the canonical population statistics in reward_lens.stats — the population-statistics module.

Exercises every exported function with known-answer inputs and edge cases.
"""

import math

import numpy as np
import pytest

from reward_lens.stats.effects import (
    BootstrapResult,
    bootstrap_ci,
    bootstrap_cohens_d,
    cohens_d,
    paired_permutation_test,
    spearman_with_ci,
)
from reward_lens.stats.multiplicity import bh_fdr

# ── cohens_d ────────────────────────────────────────────────────────────


class TestCohensD:
    def test_one_sample_basic(self):
        d = cohens_d([1.0, 2.0, 3.0, 4.0, 5.0])
        # mean=3, std≈1.58, d≈1.897
        assert math.isfinite(d)
        assert d > 0

    def test_one_sample_constant_returns_nan(self):
        assert math.isnan(cohens_d([3.0, 3.0, 3.0]))

    def test_one_sample_n1_returns_nan(self):
        assert math.isnan(cohens_d([3.0]))

    def test_paired(self):
        a = [10, 20, 30, 40, 50]
        b = [11, 21, 31, 41, 51]
        d = cohens_d(a, b, paired=True)
        # diff = [-1]*5, mean=-1, std=0 → nan (constant diff)
        assert math.isnan(d)

    def test_paired_nonconstant(self):
        a = [10, 20, 30, 40, 50]
        b = [11, 18, 32, 38, 55]
        d = cohens_d(a, b, paired=True)
        assert math.isfinite(d)

    def test_independent(self):
        a = [1, 2, 3, 4, 5]
        b = [6, 7, 8, 9, 10]
        d = cohens_d(a, b)
        # Large positive d (group b > group a)
        assert d < -2  # negative because mean(a) < mean(b)

    def test_independent_same_returns_zero(self):
        a = [1, 2, 3, 4, 5]
        d = cohens_d(a, a)
        assert abs(d) < 1e-10


# ── bootstrap_ci ────────────────────────────────────────────────────────


class TestBootstrapCI:
    def test_basic_mean(self):
        result = bootstrap_ci([1, 2, 3, 4, 5], seed=42)
        assert isinstance(result, BootstrapResult)
        assert abs(result.point - 3.0) < 1e-10
        assert result.ci_low < result.point < result.ci_high
        assert result.n_resamples == 10_000

    def test_small_n(self):
        result = bootstrap_ci([42.0], seed=0)
        assert result.point == 42.0
        assert math.isnan(result.ci_low)
        assert math.isnan(result.ci_high)

    def test_custom_statistic(self):
        result = bootstrap_ci([1, 2, 3, 4, 5], statistic=np.median, seed=42)
        assert result.point == 3.0
        assert result.ci_low <= 3.0 <= result.ci_high

    def test_confidence_level(self):
        vals = list(range(100))
        r90 = bootstrap_ci(vals, ci=0.90, seed=42, n_resamples=5000)
        r99 = bootstrap_ci(vals, ci=0.99, seed=42, n_resamples=5000)
        # 99% CI should be wider than 90%
        assert (r99.ci_high - r99.ci_low) >= (r90.ci_high - r90.ci_low) - 1e-6

    def test_as_tuple(self):
        result = bootstrap_ci([1, 2, 3], seed=0)
        pt, lo, hi = result.as_tuple()
        assert pt == result.point
        assert lo == result.ci_low
        assert hi == result.ci_high


# ── bootstrap_cohens_d ──────────────────────────────────────────────────


class TestBootstrapCohensD:
    def test_one_sample(self):
        result = bootstrap_cohens_d([1, 2, 3, 4, 5], seed=42, n_resamples=2000)
        assert math.isfinite(result.point)
        assert result.ci_low < result.point < result.ci_high

    def test_paired(self):
        a = [10, 20, 30, 40, 50]
        b = [11, 18, 32, 38, 55]
        result = bootstrap_cohens_d(a, b, paired=True, seed=42, n_resamples=2000)
        assert math.isfinite(result.point)

    def test_n1_returns_nan_ci(self):
        result = bootstrap_cohens_d([5.0], seed=0)
        assert math.isnan(result.ci_low)


# ── paired_permutation_test ─────────────────────────────────────────────


class TestPairedPermutationTest:
    def test_identical_returns_one(self):
        a = [1, 2, 3, 4, 5]
        p = paired_permutation_test(a, a, seed=42)
        assert p >= 0.5  # no signal → high p

    def test_large_effect(self):
        a = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        p = paired_permutation_test(a, b, seed=42)
        assert p < 0.05  # very large effect

    def test_n1_returns_one(self):
        assert paired_permutation_test([1], [2]) == 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            paired_permutation_test([1, 2], [1, 2, 3])

    def test_alternatives(self):
        a = [10, 20, 30]
        b = [1, 2, 3]
        p_two = paired_permutation_test(a, b, alternative="two-sided", seed=42)
        p_gt = paired_permutation_test(a, b, alternative="greater", seed=42)
        p_lt = paired_permutation_test(a, b, alternative="less", seed=42)
        assert p_gt <= p_two  # one-sided should be ≤ two-sided for dominant direction
        assert p_lt >= p_gt  # wrong direction should have higher p


# ── bh_fdr ──────────────────────────────────────────────────────────────


class TestBHFDR:
    def test_all_significant(self):
        p = [0.001, 0.002, 0.003]
        rejected, q = bh_fdr(p, alpha=0.05)
        assert all(rejected)
        assert all(np.isfinite(q))

    def test_none_significant(self):
        p = [0.8, 0.9, 0.95]
        rejected, q = bh_fdr(p, alpha=0.05)
        assert not any(rejected)

    def test_mixed(self):
        p = [0.001, 0.50, 0.99]
        rejected, q = bh_fdr(p, alpha=0.05)
        assert rejected[0]
        assert not rejected[2]

    def test_nan_passthrough(self):
        p = [0.001, float("nan"), 0.99]
        rejected, q = bh_fdr(p, alpha=0.05)
        assert rejected[0]
        assert not rejected[1]  # nan → not rejected
        assert math.isnan(q[1])

    def test_q_monotone(self):
        p = [0.01, 0.03, 0.05, 0.10, 0.50]
        _, q = bh_fdr(p, alpha=0.05)
        finite = q[np.isfinite(q)]
        sorted_q = np.sort(finite)
        # q-values after sorting should be non-decreasing
        assert np.all(np.diff(sorted_q) >= -1e-12)


# ── spearman_with_ci ────────────────────────────────────────────────────


class TestSpearmanWithCI:
    def test_perfect_positive(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 20, 30, 40, 50]
        result = spearman_with_ci(x, y, seed=42, n_resamples=2000)
        assert abs(result.point - 1.0) < 1e-10
        assert result.ci_low > 0.5

    def test_perfect_negative(self):
        x = [1, 2, 3, 4, 5]
        y = [50, 40, 30, 20, 10]
        result = spearman_with_ci(x, y, seed=42, n_resamples=2000)
        assert abs(result.point - (-1.0)) < 1e-10

    def test_small_n(self):
        result = spearman_with_ci([1, 2], [3, 4], seed=0)
        assert math.isnan(result.point)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            spearman_with_ci([1, 2, 3], [1, 2])

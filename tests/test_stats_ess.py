"""Tests for reward_lens.stats.ess — the R7 lineage/ESS machinery.

These are the executable form of the R7 property: a view built
from clones of one seed must report an effective n of ~1, the default bootstrap
must refuse to inflate it, and the clone-inflated path must stamp its method
string so the inflation travels with the Evidence.
"""

import math

import numpy as np

from reward_lens.stats.ess import (
    cluster_bootstrap,
    cluster_permutation,
    detect_clones,
    effective_sample_size,
)


class TestEffectiveSampleSize:
    def test_clones_of_one_seed_report_one(self):
        labels = ["s0"] * 50
        assert abs(effective_sample_size(labels) - 1.0) < 1e-9

    def test_distinct_seeds_report_n(self):
        labels = [f"s{i}" for i in range(37)]
        assert abs(effective_sample_size(labels) - 37.0) < 1e-9

    def test_balanced_30x5_reports_30(self):
        labels = [f"s{i}" for i in range(30) for _ in range(5)]
        assert len(labels) == 150
        assert abs(effective_sample_size(labels) - 30.0) < 1e-9

    def test_empty_returns_zero(self):
        assert effective_sample_size([]) == 0.0

    def test_unbalanced_between_one_and_n(self):
        # one dominant seed of 100 rows plus four singletons: the effective n is
        # pulled far below the five distinct seeds toward the single big cluster.
        labels = ["big"] * 100 + ["a", "b", "c", "d"]
        ess = effective_sample_size(labels)
        assert 1.0 < ess < 2.0

    def test_integer_labels(self):
        labels = [0, 0, 1, 1, 2, 2, 3, 3]
        assert abs(effective_sample_size(labels) - 4.0) < 1e-9


class TestDetectClones:
    def test_counts_duplicates(self):
        hashes = ["h1", "h1", "h1", "h2", "h3", "h3"]
        info = detect_clones(hashes)
        assert info["n_rows"] == 6
        assert info["n_unique"] == 3
        assert info["weights"] == {"h1": 3, "h2": 1, "h3": 2}
        # three of the six rows repeat an already-seen hash
        assert abs(info["duplicate_fraction"] - 0.5) < 1e-12

    def test_all_unique(self):
        info = detect_clones([f"h{i}" for i in range(10)])
        assert info["n_unique"] == 10
        assert info["duplicate_fraction"] == 0.0

    def test_empty(self):
        info = detect_clones([])
        assert info["n_rows"] == 0
        assert info["n_unique"] == 0
        assert info["duplicate_fraction"] == 0.0


class TestClusterBootstrap:
    def test_clones_do_not_inflate(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=50)  # genuine spread ...
        labels = ["s0"] * 50  # ... but all one seed
        cb = cluster_bootstrap(values, labels, seed=1, n_resamples=2000)
        assert cb.method == "bootstrap-cluster"
        # one cluster carries no resampling variance: decline, do not fake a CI
        assert math.isnan(cb.ci_low)
        assert math.isnan(cb.ci_high)

    def test_clone_resampling_is_stamped(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=50)
        labels = ["s0"] * 50
        inflated = cluster_bootstrap(
            values, labels, seed=1, n_resamples=2000, allow_clone_resampling=True
        )
        assert inflated.method == "bootstrap-CLONE-INFLATED"
        # the opt-in path DOES manufacture a finite (falsely precise) interval
        assert math.isfinite(inflated.ci_low)
        assert math.isfinite(inflated.ci_high)
        assert inflated.ci_low < inflated.point < inflated.ci_high

    def test_many_seeds_gives_finite_ci(self):
        rng = np.random.default_rng(2)
        values, labels = [], []
        for s in range(30):
            center = rng.normal(scale=2.0)
            for _ in range(5):
                values.append(center + rng.normal(scale=0.1))
                labels.append(f"s{s}")
        cb = cluster_bootstrap(np.array(values), labels, seed=3, n_resamples=2000)
        assert cb.method == "bootstrap-cluster"
        assert math.isfinite(cb.ci_low) and math.isfinite(cb.ci_high)
        assert cb.ci_low < cb.point < cb.ci_high

    def test_cluster_ci_wider_than_clone_inflated(self):
        # Heavy cloning, small within-seed spread, large between-seed spread:
        # the honest cluster CI must be wider than the clone-inflated one that
        # pretends all rows are independent.
        rng = np.random.default_rng(7)
        values, labels = [], []
        for s in range(6):
            center = rng.normal(scale=3.0)
            for _ in range(20):
                values.append(center + rng.normal(scale=0.05))
                labels.append(f"s{s}")
        values = np.array(values)
        honest = cluster_bootstrap(values, labels, seed=1, n_resamples=3000)
        inflated = cluster_bootstrap(
            values, labels, seed=1, n_resamples=3000, allow_clone_resampling=True
        )
        honest_width = honest.ci_high - honest.ci_low
        inflated_width = inflated.ci_high - inflated.ci_low
        assert honest_width > inflated_width


class TestClusterPermutation:
    def test_large_seed_level_effect(self):
        rng = np.random.default_rng(0)
        a, b, labels = [], [], []
        for s in range(12):
            base = rng.normal(scale=1.0)  # cancels in the difference
            for _ in range(4):
                a.append(base + 5.0 + rng.normal(scale=0.1))
                b.append(base + rng.normal(scale=0.1))
                labels.append(f"s{s}")
        p = cluster_permutation(a, b, labels, n_permutations=5000, seed=1)
        assert p < 0.01

    def test_no_effect_returns_high_p(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=40)
        labels = [f"s{i}" for i in range(8) for _ in range(5)]
        p = cluster_permutation(a, a, labels, n_permutations=2000, seed=1)
        assert p >= 0.5

"""Effective sample size and cluster resampling (design rule R7).

This module is the structural fix for v1's worst failure class. In v1 the
"n = 30 pairs/dim" behind every confidence interval was five to seven
hand-written stimuli expanded by prompt-prefix mutations, and in the bias
battery it was padded with byte-identical duplicates. Bootstrap CIs then
resampled those manufactured rows and reported a precision the data did not
have. The E19 repetition headline (Δd = −4.46) traced to a single stimulus.

R7 makes that impossible to hide rather than merely fixed. Every stimulus
carries a content hash and a lineage (a seed id and the mutation ops applied
to it). The statistics resample at the *seed* level by default, so cloning a
seed into fifty rows buys no extra effective n. Resampling at the expanded row
level is still available, but it requires an explicit opt-in and stamps the
resulting interval with ``method="bootstrap-CLONE-INFLATED"`` so the inflation
travels with the Evidence and is visible on every card that consumes it.

The functions here are pure and operate on plain sequences: ``seed_labels`` is
one lineage label per row, ``content_hashes`` is one content hash per row.
Turning a DataView into those sequences is the data plane's job.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Optional, Sequence

import numpy as np

from reward_lens.stats.effects import BootstrapResult


def effective_sample_size(seed_labels: Sequence) -> float:
    """Lineage-aware effective sample size from seed multiplicities (Kish).

    Group the rows by their seed label and let ``w_s`` be the number of rows
    carrying seed ``s``. The effective sample size is the Kish formula

        ESS = (sum_s w_s)**2 / sum_s w_s**2

    which is the inverse-Simpson count of "how many independent seeds is this
    dataset worth." Three anchor cases pin the intuition:

      - N rows that are all clones of ONE seed: one weight ``w = N``, so
        ESS = N**2 / N**2 = 1.0. Cloning buys nothing.
      - N rows from N distinct seeds: ESS = N**2 / N = N. Every row is its
        own lineage, so the effective n is the nominal n.
      - A balanced design, e.g. 30 seeds each expanded to 5 mutations: weights
        are all 5, so ESS = 150**2 / (30 * 25) = 30. The effective n is the
        number of seeds, not the number of expanded rows.

    Empty input returns 0.0.
    """
    labels = list(seed_labels)
    if len(labels) == 0:
        return 0.0
    counts = np.array(list(Counter(labels).values()), dtype=np.float64)
    denom = float(np.sum(counts * counts))
    if denom == 0.0:
        return 0.0
    return float(np.sum(counts) ** 2 / denom)


def detect_clones(content_hashes: Sequence[str]) -> dict:
    """Detect exact-duplicate content and collapse it to weighted items.

    Exact-duplicate content hashes are byte-identical stimuli that must not
    count as independent observations. This is the ingest-time check R7 calls
    for: it reports how many rows are duplicates and the per-hash multiplicity
    so the data plane can collapse them to weights with a warning.

    Args:
        content_hashes: One content hash per row.

    Returns:
        A dict with:
            - ``n_rows``: total number of rows.
            - ``n_unique``: number of distinct content hashes.
            - ``weights``: mapping from content hash to its row count.
            - ``duplicate_fraction``: fraction of rows that are duplicates of an
              already-seen hash, ``(n_rows - n_unique) / n_rows``. 0.0 when
              every row is unique or the input is empty.
    """
    hashes = list(content_hashes)
    n_rows = len(hashes)
    counts = Counter(hashes)
    n_unique = len(counts)
    duplicate_fraction = (n_rows - n_unique) / n_rows if n_rows > 0 else 0.0
    return {
        "n_rows": n_rows,
        "n_unique": n_unique,
        "weights": dict(counts),
        "duplicate_fraction": float(duplicate_fraction),
    }


def _group_row_indices(seed_labels: Sequence) -> dict:
    """Map each seed label to the numpy array of row indices carrying it."""
    groups: dict = {}
    for i, label in enumerate(seed_labels):
        groups.setdefault(label, []).append(i)
    return {label: np.asarray(idx, dtype=np.intp) for label, idx in groups.items()}


def cluster_bootstrap(
    values: Sequence[float] | np.ndarray,
    seed_labels: Sequence,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: Optional[int] = None,
    allow_clone_resampling: bool = False,
) -> BootstrapResult:
    """Bootstrap CI that resamples at the seed (cluster) level by default.

    The default draws whole seeds with replacement (a cluster/block bootstrap):
    on each resample it picks ``K`` seeds from the ``K`` observed seeds with
    replacement and takes *all* rows belonging to each drawn seed, then computes
    the statistic on the pooled rows. This is the honest interval, because the
    unit of independence is the seed, not the expanded row. On a dataset of
    clones of a single seed there is only one cluster to draw, so a CI cannot be
    formed and the bounds are ``nan`` (method ``"bootstrap-cluster"``): the
    routine refuses to manufacture precision from clones.

    Setting ``allow_clone_resampling=True`` reverts to resampling individual
    rows with replacement, which is exactly the v1 behaviour that inflated n.
    It is kept because there are legitimate uses (a dataset that genuinely has
    one independent row per seed), but it stamps the result with
    ``method="bootstrap-CLONE-INFLATED"`` so any Evidence built from it is
    marked as potentially inflated and no downstream reader is misled.

    Args:
        values: One observation per row.
        seed_labels: One seed/lineage label per row (same length as ``values``).
        statistic: Callable taking a numpy array of pooled rows, returning a scalar.
        n_resamples: Number of resamples.
        ci: Confidence level.
        seed: RNG seed.
        allow_clone_resampling: Opt into row-level resampling (marks the result).

    Returns:
        BootstrapResult with the appropriate ``method`` set.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    labels = list(seed_labels)
    if arr.size != len(labels):
        raise ValueError(f"values and seed_labels must align; got {arr.size} vs {len(labels)}")
    n = arr.size
    point = float(statistic(arr)) if n > 0 else float("nan")
    alpha = (1.0 - ci) / 2.0

    if allow_clone_resampling:
        method = "bootstrap-CLONE-INFLATED"
        if n < 2:
            return BootstrapResult(point, float("nan"), float("nan"), ci, 0, method=method)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_resamples, n))
        replicates = _statistic_per_row(arr, statistic, idx)
        finite = replicates[np.isfinite(replicates)]
        if finite.size < 10:
            return BootstrapResult(
                point, float("nan"), float("nan"), ci, n_resamples, method=method
            )
        lo = float(np.quantile(finite, alpha))
        hi = float(np.quantile(finite, 1.0 - alpha))
        return BootstrapResult(point, lo, hi, ci, n_resamples, method=method)

    method = "bootstrap-cluster"
    clusters = _group_row_indices(labels)
    cluster_rows = [arr[rows] for rows in clusters.values()]
    n_clusters = len(cluster_rows)
    # A CI needs at least two independent clusters; one cluster (all clones of
    # one seed) carries no resampling variance, so decline rather than fake it.
    if n_clusters < 2:
        return BootstrapResult(point, float("nan"), float("nan"), ci, 0, method=method)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clusters, size=(n_resamples, n_clusters))
    replicates = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        pooled = np.concatenate([cluster_rows[j] for j in draws[i]])
        replicates[i] = float(statistic(pooled))
    finite = replicates[np.isfinite(replicates)]
    if finite.size < 10:
        return BootstrapResult(point, float("nan"), float("nan"), ci, n_resamples, method=method)
    lo = float(np.quantile(finite, alpha))
    hi = float(np.quantile(finite, 1.0 - alpha))
    return BootstrapResult(point, lo, hi, ci, n_resamples, method=method)


def cluster_permutation(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    seed_labels: Sequence,
    n_permutations: int = 10_000,
    statistic: str = "mean_diff",
    alternative: str = "two-sided",
    seed: Optional[int] = None,
) -> float:
    """Paired permutation test that flips signs at the seed (cluster) level.

    The row-level ``paired_permutation_test`` flips the sign of each row's
    difference independently, which over-counts evidence when many rows share a
    seed: fifty clones of one stimulus give fifty independent sign flips that
    the lineage says should move together. Here the sign is drawn once per seed
    and applied to every row of that seed, so the permutation null respects the
    unit of independence. With ``K`` seeds the null explores ``2**K`` sign
    assignments, not ``2**n``.

    Args:
        a: First set of paired observations, one per row.
        b: Second set of paired observations, one per row (same length as ``a``).
        seed_labels: One seed label per row (same length as ``a``).
        n_permutations: Number of seed-level sign-flip permutations.
        statistic: One of {"mean_diff", "median_diff"} on the row differences.
        alternative: One of {"two-sided", "greater", "less"}.
        seed: RNG seed.

    Returns:
        p-value with the Phipson-Smyth ``(count + 1) / (n + 1)`` correction.
        Returns 1.0 when n < 2 (no power).
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    labels = list(seed_labels)
    if not (a.size == b.size == len(labels)):
        raise ValueError(f"a, b, seed_labels must align; got {a.size}, {b.size}, {len(labels)}")
    n = a.size
    if n < 2:
        return 1.0

    diff = a - b
    if statistic == "mean_diff":
        reduce = np.mean
    elif statistic == "median_diff":
        reduce = np.median
    else:
        raise ValueError(f"unknown statistic: {statistic}")
    observed = float(reduce(diff))

    clusters = _group_row_indices(labels)
    cluster_keys = list(clusters)
    n_clusters = len(cluster_keys)
    # Row -> cluster-column index, so a per-cluster sign broadcasts to rows.
    cluster_of_row = np.empty(n, dtype=np.intp)
    for col, key in enumerate(cluster_keys):
        cluster_of_row[clusters[key]] = col

    rng = np.random.default_rng(seed)
    cluster_signs = rng.choice([-1.0, 1.0], size=(n_permutations, n_clusters))
    signs = cluster_signs[:, cluster_of_row]  # (n_permutations, n)
    permuted = signs * diff[None, :]
    if statistic == "mean_diff":
        replicates = permuted.mean(axis=1)
    else:
        replicates = np.median(permuted, axis=1)

    if alternative == "two-sided":
        count = int(np.sum(np.abs(replicates) >= abs(observed)))
    elif alternative == "greater":
        count = int(np.sum(replicates >= observed))
    elif alternative == "less":
        count = int(np.sum(replicates <= observed))
    else:
        raise ValueError(f"unknown alternative: {alternative}")
    return (count + 1) / (n_permutations + 1)


def _statistic_per_row(
    arr: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    idx: np.ndarray,
) -> np.ndarray:
    """One replicate of ``statistic`` per row of a resample-index matrix."""
    samples = arr[idx]
    if statistic is np.mean:
        return samples.mean(axis=1)
    out = np.empty(idx.shape[0], dtype=np.float64)
    for i in range(idx.shape[0]):
        out[i] = float(statistic(samples[i]))
    return out


__all__ = [
    "effective_sample_size",
    "detect_clones",
    "cluster_bootstrap",
    "cluster_permutation",
]

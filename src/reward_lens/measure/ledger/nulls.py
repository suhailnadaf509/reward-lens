"""The three nulls the ledger ships against, in the units of the thing they are nulls for.

A claim that arrives without a baseline is sent back, and `stats.baselines` is the bank most claims
in this library use. That bank scores **binary detection**: six comparators reduced to a per-item
score and compared by AUROC. The ledger's claims are not detections. "This feature's advantage
covariance is +0.031" has no labels and no items to rank, so an AUROC against it would be a number
about a task nobody is doing. The three nulls here are the comparators the catalogue names for F1
and F2, each computed in the same units as the statistic it is a null for, so a reader can put the
two side by side without a conversion.

- **A random feature** (F1). Draw a standard normal per rollout and run the whole ledger on it. Its
  covariance with the advantage and its `Δz` are both pure sampling noise, so this is the scale of
  "nothing", measured on this run's own group sizes and batch sizes rather than assumed.
- **A permuted-advantage null** (F1). Permute the advantages **inside each group** and recompute the
  covariance. Within-group rather than across, because the within-group permutation preserves
  exactly what the estimator constructed (a group-centred advantage vector per prompt) and destroys
  exactly what the claim is about (which rollout got which advantage). Permuting across groups would
  also destroy the group structure and would therefore beat a covariance that never depended on it.
- **A permuted-step null** (F2). Pair each step's `Δz` vector with a different step's covariance
  vector and refit `Λ`. The exchangeable unit is a whole step, not a scalar, which is why this is
  written out rather than routed through `stats.nulls.shuffle_null`: that function permutes a label
  array against fixed values, and here both sides are `k`-vectors that have to move together.

Every one of them reports the observed statistic beside the null, because a p-value on its own does
not say whether the effect is large.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from reward_lens.measure.ledger.explained import _through_origin
from reward_lens.measure.ledger.price import StepLedger, StepSample, selection_differential


@dataclass(frozen=True)
class NullResult:
    """One statistic against its null, with the numbers rather than only the verdict."""

    baseline: str
    statistic: str
    observed: float
    null_median: float
    null_p95: float
    p_value: float
    n_draws: int

    def render(self) -> str:
        return (
            f"{self.baseline:<30} {self.statistic} observed {self.observed:+.5g} against a null "
            f"median {self.null_median:+.5g}, p95 {self.null_p95:.5g}, p = {self.p_value:.4f} "
            f"over {self.n_draws} draws"
        )


def _p_value(observed: float, draws: np.ndarray) -> float:
    """Two-sided exceedance with the `(count + 1) / (n + 1)` correction `stats.nulls` also uses.

    The correction is not decoration: without it a statistic no draw exceeded reports `p = 0`, which
    claims more than a finite number of draws can support.
    """
    finite = draws[np.isfinite(draws)]
    if finite.size == 0 or not np.isfinite(observed):
        return float("nan")
    count = int(np.count_nonzero(np.abs(finite) >= abs(observed)))
    return float((count + 1) / (finite.size + 1))


def random_feature_null(
    sample: StepSample,
    *,
    n_draws: int = 500,
    seed: int = 0,
) -> np.ndarray:
    """`|Cov_group(A, f)|` for `n_draws` features that are pure noise, on this step's own structure.

    The feature is drawn per rollout, so it inherits the step's group sizes, its abstention pattern
    and its advantage vector, and the resulting spread is the scale of a covariance that means
    nothing. Drawn as a standard normal and reported in absolute value, so it is comparable against
    a real feature's covariance divided by that feature's own standard deviation.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        noise = rng.standard_normal((sample.n, 1))
        differential = selection_differential(
            noise, sample.advantages, sample.group_ids, ("random",)
        )
        out[i] = abs(float(differential.value[0]))
    return out


def permuted_advantage_null(
    sample: StepSample,
    *,
    n_draws: int = 500,
    seed: int = 0,
) -> np.ndarray:
    """`|Cov_group(A, f)|` per feature under within-group permutations of the advantage.

    Returns an ``(n_draws, k)`` array. Permuting inside the group keeps every group's advantage
    multiset exactly as the estimator produced it and breaks only the pairing with the features,
    which is the null the claim is against. A group of one rollout has one permutation and
    contributes its (zero) covariance under every draw, which is correct: it carried no information
    to destroy.
    """
    rng = np.random.default_rng(seed)
    k = sample.features.shape[1]
    out = np.empty((n_draws, k), dtype=np.float64)
    labels = np.unique(sample.group_ids)
    for i in range(n_draws):
        shuffled = sample.advantages.copy()
        for label in labels:
            mask = sample.group_ids == label
            block = shuffled[mask]
            shuffled[mask] = block[rng.permutation(block.size)]
        differential = selection_differential(
            sample.features, shuffled, sample.group_ids, sample.names
        )
        out[i] = np.abs(differential.value)
    return out


def permuted_step_null(
    ledgers: Sequence[StepLedger],
    scales: Mapping[str, float],
    *,
    n_draws: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """`Λ` under permutations that pair each step's movement with another step's selection pressure.

    The identity claims a *within-step* correspondence. Under the null that no such correspondence
    exists, the assignment of `Δz` vectors to `Cov` vectors across steps is exchangeable, so this
    permutation is the exact null for the claim `Λ` makes. A derangement is not forced: a random
    permutation is the right null and forcing every step to move would make the null harder than the
    hypothesis it tests.
    """
    usable = [n for n in (ledgers[0].names if ledgers else ()) if scales.get(n, 0.0) > 0.0]
    if len(ledgers) < 2 or not usable:
        return np.asarray([], dtype=np.float64)
    x = np.asarray(
        [[ledger.row(n).covariance / scales[n] for n in usable] for ledger in ledgers],
        dtype=np.float64,
    )
    y = np.asarray(
        [[ledger.row(n).delta_z / scales[n] for n in usable] for ledger in ledgers],
        dtype=np.float64,
    )
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.where(finite, x, 0.0)
    y = np.where(finite, y, 0.0)
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        order = rng.permutation(x.shape[0])
        out[i] = _through_origin(x[order].ravel(), y.ravel())[0]
    return out


def summarise(baseline: str, statistic: str, observed: float, draws: np.ndarray) -> NullResult:
    """One null's draws reduced to the four numbers a card prints."""
    finite = draws[np.isfinite(draws)] if draws.size else draws
    return NullResult(
        baseline=baseline,
        statistic=statistic,
        observed=float(observed),
        null_median=float(np.median(finite)) if finite.size else float("nan"),
        null_p95=float(np.quantile(finite, 0.95)) if finite.size else float("nan"),
        p_value=_p_value(observed, draws),
        n_draws=int(finite.size),
    )


__all__ = [
    "NullResult",
    "permuted_advantage_null",
    "permuted_step_null",
    "random_feature_null",
    "summarise",
]

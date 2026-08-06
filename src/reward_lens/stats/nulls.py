"""Null models for alignment, prediction, and convergence claims.

Every Observable that says one thing "aligns with", "predicts", or "exceeds"
another owes a null: the value it would report if the relationship were absent.
The null is a first-class object the runner attaches to the Evidence, so a
reader can see the effect against its own noise floor instead of against zero.

This module supplies the nulls the corpus actually uses:

  - ``random_direction_cosines`` / ``random_direction_null``: the noise floor
    for a cosine between two directions in high dimension. Random unit vectors
    are nearly orthogonal, so a modest cosine that looks meaningful in 3-D is
    pure chance in 4096-D. Any "this direction aligns with the reward vector"
    claim is read against this.
  - ``shuffle_null``: a generic label/order permutation null for any statistic
    of (values, labels), for claims of the form "this grouping carries signal."
  - ``rum_identifiability_null``: the universality baseline (S15 / the Atlas's
    VCE index A14). Two random-utility models that encode reward in
    k-dimensional subspaces share only as much subspace alignment as two random
    k-subspaces would, because the orientation is not identified. Convergence
    beyond this null is the thing worth reporting.

All functions are seeded and pure.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


def random_direction_cosines(
    d: int,
    n: int,
    reference: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Cosines of ``n`` random unit vectors in R^d against a reference direction.

    Draws ``n`` directions uniformly on the unit sphere (normalized isotropic
    Gaussians) and returns their signed cosine with ``reference`` (itself drawn
    uniformly if not given). This is the sampling distribution of alignment
    under the null of no relationship between a direction and the reference. In
    high dimension it concentrates near 0 with spread ~ 1/sqrt(d), which is why
    ``|cos|`` is the quantity a high-dimensional alignment claim must beat.

    Args:
        d: Ambient dimension.
        n: Number of random directions to draw.
        reference: Reference direction (length d). Random if None.
        seed: RNG seed.

    Returns:
        Array of ``n`` signed cosines.
    """
    if d < 1:
        raise ValueError(f"dimension must be >= 1; got {d}")
    rng = np.random.default_rng(seed)
    if reference is None:
        reference = rng.standard_normal(d)
    reference = np.asarray(reference, dtype=np.float64).ravel()
    if reference.size != d:
        raise ValueError(f"reference has length {reference.size}, expected {d}")
    ref_norm = np.linalg.norm(reference)
    if ref_norm == 0:
        raise ValueError("reference direction has zero norm")
    reference = reference / ref_norm

    vecs = rng.standard_normal((n, d))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # a zero draw has probability ~0; keep it finite
    vecs = vecs / norms
    return np.asarray(vecs @ reference, dtype=np.float64)


def random_direction_null(
    observed_cos: float,
    d: int,
    n: int = 10_000,
    seed: Optional[int] = None,
) -> dict:
    """Compare an observed cosine to the random-direction null in dimension d.

    Args:
        observed_cos: The measured cosine between two directions.
        d: Ambient dimension the directions live in.
        n: Number of null draws.
        seed: RNG seed.

    Returns:
        A dict with:
            - ``p_value``: fraction of null ``|cos|`` at least as large as
              ``|observed_cos|``, with the ``(count + 1) / (n + 1)`` correction.
              This is two-sided in the sense that only the magnitude of the
              alignment is tested, which is what "aligns" claims mean.
            - ``null_mean``: mean of the null ``|cos|`` (the noise floor).
            - ``null_p95``: 95th percentile of the null ``|cos|`` (a one-glance
              significance threshold).
    """
    cosines = random_direction_cosines(d, n, reference=None, seed=seed)
    null_abs = np.abs(cosines)
    obs = abs(float(observed_cos))
    count = int(np.sum(null_abs >= obs))
    p_value = (count + 1) / (n + 1)
    return {
        "p_value": float(p_value),
        "null_mean": float(np.mean(null_abs)),
        "null_p95": float(np.quantile(null_abs, 0.95)),
    }


def shuffle_null(
    values: Sequence[float] | np.ndarray,
    labels: Sequence,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n: int = 10_000,
    seed: Optional[int] = None,
) -> dict:
    """Label-permutation null for an arbitrary statistic of (values, labels).

    Recomputes ``statistic(values, permuted_labels)`` over ``n`` random
    permutations of the labels, holding the values fixed. This is the generic
    null for "the way these values are grouped/ordered carries signal": under
    the null the labels are exchangeable, so any statistic that depends on the
    pairing should look like its permuted replicates.

    Args:
        values: The observations (passed through to ``statistic`` unchanged).
        labels: The labels to permute.
        statistic: Callable ``(values, labels) -> float``.
        n: Number of permutations.
        seed: RNG seed.

    Returns:
        A dict with ``observed`` (the statistic on the true labels),
        ``null_mean`` (mean of the permuted replicates), and ``p_value``
        (fraction of ``|replicate|`` at least as extreme as ``|observed|``,
        with the ``(count + 1) / (n + 1)`` correction).
    """
    value_array = np.asarray(values)
    label_array = np.asarray(labels)
    observed = float(statistic(value_array, label_array))
    rng = np.random.default_rng(seed)
    replicates = np.empty(n, dtype=np.float64)
    for i in range(n):
        replicates[i] = float(statistic(value_array, rng.permutation(label_array)))
    finite = replicates[np.isfinite(replicates)]
    if finite.size == 0 or not np.isfinite(observed):
        return {"observed": observed, "null_mean": float("nan"), "p_value": float("nan")}
    count = int(np.sum(np.abs(finite) >= abs(observed)))
    p_value = (count + 1) / (finite.size + 1)
    return {
        "observed": observed,
        "null_mean": float(np.mean(finite)),
        "p_value": float(p_value),
    }


def rum_identifiability_null(
    d: int,
    k: int,
    n: int = 1_000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """The RUM-identifiability baseline for subspace-alignment claims.

    A random utility model identifies its reward only up to the freedom that
    leaves choice probabilities unchanged: the reward-relevant k-dimensional
    subspace is pinned to be k-dimensional but its orientation in R^d is not
    determined by the preference data. So two independently estimated RMs of
    unrelated utilities share only the subspace alignment that two *random*
    k-dimensional subspaces of R^d would share by chance. That chance
    distribution is this null, and it is the baseline the VCE index (A14)
    subtracts: alignment above this null is convergence beyond what
    identifiability freedom alone forces.

    Concretely, this draws ``n`` pairs of independent uniformly-random
    k-subspaces (orthonormalized Gaussian bases) and, for each pair, returns the
    mean squared cosine of the principal angles between them,

        alignment = (1/k) * sum_i cos^2(theta_i) = (1/k) * ||Q_A^T Q_B||_F^2,

    which lies in [0, 1]: 1 when the subspaces coincide, ~ k/d when they are
    unrelated. Reporting mean cos^2 (rather than the smallest principal angle)
    makes it a projection-overlap fraction, directly comparable to a subspace
    alignment score computed the same way on real reward subspaces.

    Args:
        d: Ambient dimension.
        k: Subspace dimension (must satisfy ``1 <= k <= d``).
        n: Number of random subspace pairs to draw.
        seed: RNG seed.

    Returns:
        Array of ``n`` alignment values in [0, 1].
    """
    if not (1 <= k <= d):
        raise ValueError(f"require 1 <= k <= d; got k={k}, d={d}")
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        qa = _random_orthonormal_basis(d, k, rng)
        qb = _random_orthonormal_basis(d, k, rng)
        cos_angles = np.linalg.svd(qa.T @ qb, compute_uv=False)
        cos_angles = np.clip(cos_angles, 0.0, 1.0)
        out[i] = float(np.mean(cos_angles**2))
    return out


def _random_orthonormal_basis(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """A d-by-k matrix whose columns are an orthonormal basis of a random k-subspace.

    The QR factor of a Gaussian matrix is Haar-distributed, so its column space
    is uniform over k-dimensional subspaces of R^d.
    """
    gaussian = rng.standard_normal((d, k))
    q, _ = np.linalg.qr(gaussian)
    return q[:, :k]


__all__ = [
    "random_direction_cosines",
    "random_direction_null",
    "shuffle_null",
    "rum_identifiability_null",
]

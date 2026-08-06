"""Organism-calibrated mutual-information estimators for the alignment channel (S8, HORIZON H2).

The alignment channel asks how many bits of value information survive each stage of the reward
pipeline, and that is a mutual information. A mutual information estimated from a finite sample is only
as trustworthy as its calibration: a k-nearest-neighbour estimator carries a bias that depends on the
neighbour count, the sample size, and the local geometry, so a raw bits number means nothing until the
estimator has been made to recover a mutual information that is known in closed form. This module holds
the estimators and the Gaussian calibration that grades them, so an alignment-channel bit count is
never reported by an uncalibrated instrument (S8 kill criterion: MI estimates too loose even on
organisms is a publishable calibration study, not a hidden failure).

Three estimators and one calibration:

  - ``mi_ksg`` (continuous-continuous): the Kraskov-Stoegbauer-Grassberger k-nearest-neighbour
    estimator, estimator 1 of Kraskov, Stoegbauer and Grassberger (2004). The workhorse for two
    continuous channels; it is asymptotically unbiased where a histogram is not.
  - ``mi_discrete_continuous`` (discrete-continuous): the Ross (2014) k-nearest-neighbour estimator for
    the mutual information between a discrete source value ``V`` and a continuous reward ``r``. This is
    the alignment channel's own shape, a categorical annotator value read out of a scalar reward, so it
    is the estimator the channel-capacity arm uses.
  - ``mi_binned`` (histogram plug-in): the simple, robust, upward-biased estimator kept only as a
    cross-check. Its bias is reported next to the KSG number, never hidden.
  - ``gaussian_mi_bits`` / ``calibrate_gaussian``: the closed-form bivariate-Gaussian mutual
    information ``I = -1/2 log2(1 - rho^2)`` and the harness that draws a correlated Gaussian of known
    MI, runs an estimator on it over several seeds, and reports the signed bias and its spread. An
    estimator that cannot recover a Gaussian MI within tolerance has no business reporting an
    alignment-channel bit count, so this is the gate the channel arm passes before it is believed.

Everything is computed in nats internally and returned in bits (base 2) by default. The module is
torch-free, importing nothing heavier than numpy and scipy, which matches the rest of
``reward_lens.stats`` (scikit-learn is never required; the Ross estimator is reproduced here directly).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

_LOG2 = float(np.log(2.0))


def _bits_divisor(base: float) -> float:
    """The divisor that turns a nats quantity into a ``base``-unit quantity (base 2 gives bits)."""
    return _LOG2 if base == 2.0 else float(np.log(base))


# ---------------------------------------------------------------------------
# Closed-form references (the known answers the estimators are graded against)
# ---------------------------------------------------------------------------


def gaussian_mi_bits(rho: float) -> float:
    """The exact mutual information ``I = -1/2 log2(1 - rho^2)`` of a bivariate Gaussian, in bits.

    For a standard bivariate normal with correlation ``rho`` the mutual information is known in closed
    form and depends on nothing but ``rho``. This is the ground truth the KSG and binned estimators are
    calibrated against: draw the correlated Gaussian, estimate, and compare to this number.
    """
    r = float(rho)
    if not -1.0 < r < 1.0:
        raise ValueError(f"gaussian_mi_bits needs |rho| < 1; got {r}")
    return float(-0.5 * np.log2(1.0 - r * r))


def entropy_discrete_bits(labels: np.ndarray) -> float:
    """The Shannon entropy ``H`` in bits of the empirical distribution of a discrete label array.

    The plug-in entropy of the realized categorical counts. For the alignment channel this is the
    empirical source entropy; the mixture organism also carries the exact ``H(V)`` from its chosen
    weights, and the two agree up to sampling.
    """
    vals = np.asarray(labels).ravel()
    if vals.size == 0:
        return float("nan")
    _, counts = np.unique(vals, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    return float(-np.sum(p * np.log2(p)))


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def _as_2d(a: np.ndarray) -> np.ndarray:
    """Coerce an array to ``(n, d)`` float64, treating a 1-D array as a single column."""
    arr = np.asarray(a, dtype=np.float64)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _jittered(a: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    """Add tiny per-coordinate Gaussian noise so exact ties do not corrupt the neighbour ranks.

    KSG-style estimators assume the underlying variables are continuous with no repeated values; a
    reward with ties would give degenerate neighbour counts. The jitter is scaled to each column's
    spread and is orders of magnitude below any real structure, so it breaks ties without moving the
    estimate. A ``scale`` of zero disables it (for genuinely tie-free data).
    """
    if scale <= 0.0:
        return a
    std = np.std(a, axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return a + rng.standard_normal(a.shape) * std * scale


def mi_ksg(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 3,
    *,
    base: float = 2.0,
    jitter: float = 1e-10,
    seed: int = 0,
) -> float:
    """KSG estimator 1 of the mutual information between continuous ``x`` and ``y`` (Kraskov 2004).

    For each point the distance ``eps`` to its ``k``-th nearest neighbour is taken in the joint space
    under the maximum norm; then ``n_x`` and ``n_y`` count the points falling strictly within ``eps`` in
    each marginal. The estimate is
    ``I = psi(k) + psi(N) - <psi(n_x + 1) + psi(n_y + 1)>`` in nats, where ``psi`` is the digamma
    function and the marginal counts are read self-inclusive (``query_ball_point`` includes the point
    itself, so the returned count is already ``n_x + 1``). A tiny tie-breaking jitter is added first.
    Negative estimates (which the estimator can produce near zero MI) are clamped to zero, and the
    result is returned in bits by default.
    """
    xa = _as_2d(x)
    ya = _as_2d(y)
    n = xa.shape[0]
    if ya.shape[0] != n:
        raise ValueError(f"x and y must have the same length; got {n} and {ya.shape[0]}")
    if n <= k + 1:
        raise ValueError(f"mi_ksg needs n > k + 1; got n={n}, k={k}")
    rng = np.random.default_rng([int(seed), 11])
    xa = _jittered(xa, rng, jitter)
    ya = _jittered(ya, rng, jitter)

    joint = np.hstack([xa, ya])
    joint_tree = cKDTree(joint)
    # k-th neighbour distance in the joint max-norm space; query k+1 because the nearest is the point
    # itself at distance 0. dist[:, k] is the distance to the k-th distinct neighbour.
    dist, _ = joint_tree.query(joint, k=k + 1, p=np.inf)
    eps = dist[:, k]

    # Count marginal neighbours strictly inside eps. query_ball_point counts within a closed ball, so
    # nudge the radius just below eps; the returned count includes the point itself, i.e. it is n_x + 1.
    radius = np.nextafter(eps, 0.0)
    nx = cKDTree(xa).query_ball_point(xa, radius, p=np.inf, return_length=True)
    ny = cKDTree(ya).query_ball_point(ya, radius, p=np.inf, return_length=True)
    nx = np.maximum(np.asarray(nx, dtype=np.float64), 1.0)
    ny = np.maximum(np.asarray(ny, dtype=np.float64), 1.0)

    mi_nats = digamma(k) + digamma(n) - np.mean(digamma(nx) + digamma(ny))
    return float(max(mi_nats, 0.0) / _bits_divisor(base))


def mi_discrete_continuous(
    labels: np.ndarray,
    values: np.ndarray,
    k: int = 3,
    *,
    base: float = 2.0,
    jitter: float = 1e-10,
    seed: int = 0,
) -> float:
    """Ross (2014) k-NN estimator of ``I(V; r)`` for a discrete ``V`` and continuous ``values`` ``r``.

    For each point the ``k``-th nearest neighbour is found among the points sharing its label; ``m`` then
    counts how many points in the whole sample fall within that radius (self-inclusive), and ``N_v`` is
    the size of the point's label class. The estimate is
    ``I = psi(N) + <psi(k_eff)> - <psi(N_v)> - <psi(m)>`` in nats (Ross 2014, eq. 4), the discrete
    analogue of KSG. Points that are the sole member of their class carry no information and are dropped.
    This is the estimator the alignment channel uses, because a source value routed into a scalar reward
    is exactly a discrete-to-continuous channel. Returned in bits by default.
    """
    d = np.asarray(labels).ravel()
    c = _as_2d(values)
    n_total = c.shape[0]
    if d.shape[0] != n_total:
        raise ValueError(f"labels and values must match in length; got {d.shape[0]} and {n_total}")
    rng = np.random.default_rng([int(seed), 13])
    c = _jittered(c, rng, jitter)

    radius = np.empty(n_total, dtype=np.float64)
    label_counts = np.empty(n_total, dtype=np.float64)
    k_eff = np.empty(n_total, dtype=np.float64)
    for label in np.unique(d):
        mask = d == label
        count = int(np.sum(mask))
        label_counts[mask] = count
        if count > 1:
            kk = min(k, count - 1)
            sub_tree = cKDTree(c[mask])
            dist, _ = sub_tree.query(c[mask], k=kk + 1, p=2)
            # Nudge below the k-th within-class distance so the neighbour itself is not counted in m.
            radius[mask] = np.nextafter(dist[:, kk], 0.0)
            k_eff[mask] = kk

    valid = label_counts > 1
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return 0.0
    cv = c[valid]
    full_tree = cKDTree(cv)
    m = full_tree.query_ball_point(cv, radius[valid], p=2, return_length=True)
    m = np.maximum(np.asarray(m, dtype=np.float64), 1.0)

    mi_nats = (
        digamma(n_valid)
        + float(np.mean(digamma(k_eff[valid])))
        - float(np.mean(digamma(label_counts[valid])))
        - float(np.mean(digamma(m)))
    )
    return float(max(mi_nats, 0.0) / _bits_divisor(base))


def mi_binned(x: np.ndarray, y: np.ndarray, bins: int = 8, *, base: float = 2.0) -> float:
    """Histogram plug-in estimate of the mutual information between ``x`` and ``y`` (the cross-check).

    Bins both variables into ``bins`` equal-width cells, forms the joint and marginal histograms, and
    sums ``p(x, y) log( p(x, y) / (p(x) p(y)) )`` over occupied cells. This estimator is biased upward
    for continuous variables (finer bins invent mutual information), which is exactly why it is only the
    cross-check and its bias is reported alongside the KSG number. Returned in bits by default.
    """
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    counts, _, _ = np.histogram2d(xa, ya, bins=bins)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p_xy = counts / total
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    outer = p_x * p_y
    nz = (p_xy > 0) & (outer > 0)
    mi_nats = float(np.sum(p_xy[nz] * np.log(p_xy[nz] / outer[nz])))
    return float(max(mi_nats, 0.0) / _bits_divisor(base))


# ---------------------------------------------------------------------------
# Gaussian calibration (the gate)
# ---------------------------------------------------------------------------


@dataclass
class MICalibration:
    """The result of calibrating an estimator against a Gaussian of known mutual information.

    ``true_bits`` is the closed-form Gaussian MI; ``estimate_bits`` is the mean estimate over
    ``repeats`` independent draws; ``bias_bits`` is ``estimate_bits - true_bits`` (signed, so a positive
    value is an over-estimate); ``std_bits`` is the spread of the estimate across draws. An estimator
    passes calibration when ``|bias_bits|`` is within the study's tolerance.
    """

    estimator: str
    rho: float
    n: int
    k: int
    repeats: int
    true_bits: float
    estimate_bits: float
    bias_bits: float
    std_bits: float


def draw_correlated_gaussian(
    rho: float, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` samples of a standard bivariate normal with correlation ``rho``.

    ``x`` is standard normal and ``y = rho x + sqrt(1 - rho^2) e`` with independent standard normal
    ``e``, so ``corr(x, y) = rho`` exactly and the pair's mutual information is ``gaussian_mi_bits(rho)``.
    """
    r = float(rho)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    x = z1
    y = r * z1 + np.sqrt(max(1.0 - r * r, 0.0)) * z2
    return x, y


def calibrate_gaussian(
    rho: float = 0.6,
    n: int = 2000,
    k: int = 3,
    *,
    repeats: int = 5,
    estimator: str = "ksg",
    bins: int = 8,
    seed: int = 0,
) -> MICalibration:
    """Grade an estimator on a correlated Gaussian of known MI, returning its signed bias in bits.

    Draws ``repeats`` independent correlated-Gaussian samples (each with correlation ``rho`` and known
    mutual information ``gaussian_mi_bits(rho)``), runs the chosen estimator on each, and reports the
    mean estimate, the signed bias against the closed-form value, and the spread. ``estimator`` is
    ``"ksg"`` (the workhorse) or ``"binned"`` (the cross-check). This is the calibration the
    alignment-channel arm must pass before any channel bit count is believed.
    """
    true = gaussian_mi_bits(rho)
    estimates = np.empty(repeats, dtype=np.float64)
    for rep in range(repeats):
        rng = np.random.default_rng([int(seed), int(rep)])
        x, y = draw_correlated_gaussian(rho, n, rng)
        if estimator == "ksg":
            estimates[rep] = mi_ksg(x, y, k=k, seed=seed * 1000 + rep)
        elif estimator == "binned":
            estimates[rep] = mi_binned(x, y, bins=bins)
        else:
            raise ValueError(f"unknown estimator {estimator!r}; use 'ksg' or 'binned'")
    mean_est = float(np.mean(estimates))
    return MICalibration(
        estimator=estimator,
        rho=float(rho),
        n=int(n),
        k=int(k),
        repeats=int(repeats),
        true_bits=float(true),
        estimate_bits=mean_est,
        bias_bits=float(mean_est - true),
        std_bits=float(np.std(estimates)),
    )


__all__ = [
    "gaussian_mi_bits",
    "entropy_discrete_bits",
    "mi_ksg",
    "mi_discrete_continuous",
    "mi_binned",
    "MICalibration",
    "draw_correlated_gaussian",
    "calibrate_gaussian",
]

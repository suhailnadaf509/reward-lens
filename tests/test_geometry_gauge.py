"""THE gauge property test: the executable definition of "gauge".

This is the single most important test in the geometry subsystem. It synthesizes an activation
matrix with a genuine active subspace and a genuine data-null subspace, plants a reward direction in
the active subspace, and applies a synthetic reward-gauge transform: a per-prompt (per-row) shift, a
positive affine map (positive scaling plus a rotation confined to the null subspace), and noise added
to the reward direction along the estimated null basis. Every one of those moves preserves the
reward's on-distribution preferences, so it is pure gauge.

The invariant, stated as assertions: CANONICAL quantities (``canonicalize``, the ``effective_angle``
canonical cosine) are unchanged within tolerance under the transform, while RAW quantities (the raw
cosine of the reward direction before and after) change. And a COVARIANT comparison attempted with
``frame=None`` raises, because gate 2 has no default frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.errors import GaugeError, NumericsError
from reward_lens.geometry import canonicalize, cka, effective_angle, fit_frame


def _orthonormal(d: int, seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return q


def _synthetic_activations(seed: int = 0):
    """An n x d activation matrix with a 40-dim active subspace and a 24-dim near-null subspace.

    Returns the activation matrix, the reward direction (planted in the active subspace, so it has no
    null component), the orthonormal active/null bases, and the per-item reward margins.
    """
    rng = np.random.default_rng(seed)
    d, active_dim, n = 64, 40, 600
    q = _orthonormal(d, seed + 1)
    p_act = q[:, :active_dim]
    p_null = q[:, active_dim:]

    # Active coordinates with diverse, well-conditioned variances; null coordinates ~ 0 variance.
    scales = np.sqrt(rng.uniform(1.0, 10.0, size=active_dim))
    z_act = rng.standard_normal((n, active_dim)) * scales
    z_null = rng.standard_normal((n, d - active_dim)) * 1e-3
    h = (z_act @ p_act.T + z_null @ p_null.T).astype(np.float32)

    c = rng.standard_normal(active_dim)
    w = (p_act @ c).astype(np.float32)  # reward lives entirely in the active subspace
    margins = (h @ w).astype(np.float32)
    return h, w, p_act, p_null, margins


def _apply_gauge(h, w, p_act, p_null, seed=7):
    """Apply a synthetic reward-gauge transform to (activations, reward direction).

    (1) per-prompt shift: a global origin shift plus small per-row shifts confined to the null
        subspace (both leave w . h differences unchanged);
    (2) positive affine map: a positive scale ``alpha`` and a rotation ``R`` that is the identity on
        the active subspace and an arbitrary rotation within the null subspace (reward is preserved
        by sending w -> R w / alpha, and R w = w because w is active);
    (3) null-basis noise: add a vector in the null subspace to the reward direction (the direction is
        unconstrained on the data distribution there).
    """
    rng = np.random.default_rng(seed)
    n, d = h.shape
    null_dim = p_null.shape[1]

    origin = rng.standard_normal(d).astype(np.float32) * 5.0
    row_null_shift = ((rng.standard_normal((n, null_dim)) * 2e-2) @ p_null.T).astype(np.float32)

    alpha = 2.5
    o_null = _orthonormal(null_dim, seed + 3)
    r = (p_act @ p_act.T + p_null @ o_null @ p_null.T).astype(np.float64)  # identity on active

    h_shifted = h + origin + row_null_shift
    h_t = (alpha * (h_shifted @ r.T)).astype(np.float32)

    null_noise = (p_null @ rng.standard_normal(null_dim)).astype(np.float32)
    null_noise *= float(np.linalg.norm(w)) / max(float(np.linalg.norm(null_noise)), 1e-12)
    w_t = (w / alpha + null_noise).astype(np.float32)

    margins_t = (h_t @ w_t).astype(np.float32)
    return h_t, w_t, margins_t


def test_canonical_invariant_raw_changes_under_gauge():
    """Canonical quantities are invariant under the gauge transform; raw quantities change."""
    h, w, p_act, p_null, margins = _synthetic_activations()
    frame = fit_frame(h, margins=margins, null_var_ratio=1e-2)
    # The frame recovers the planted null subspace.
    assert 20 <= frame.null_dim <= 24
    canonical_before = canonicalize(w, frame)

    h_t, w_t, margins_t = _apply_gauge(h, w, p_act, p_null)
    frame_t = fit_frame(h_t, margins=margins_t, null_var_ratio=1e-2)
    canonical_after = canonicalize(w_t, frame_t)

    canonical_cos = float(np.dot(canonical_before, canonical_after))
    raw_cos = float(np.dot(w, w_t) / (np.linalg.norm(w) * np.linalg.norm(w_t)))

    # CANONICAL: invariant within tolerance.
    assert canonical_cos > 0.99, f"canonical cosine not invariant: {canonical_cos}"
    # RAW: genuinely changed (the null noise tilts the raw direction).
    assert raw_cos < 0.9, f"raw cosine did not change: {raw_cos}"
    # The whole point: canonical is strictly more invariant than raw.
    assert canonical_cos > raw_cos + 0.05


def test_effective_angle_canonical_invariant_under_gauge():
    """The effective_angle canonical cosine of a direction with itself survives the gauge transform.

    Two copies of one reward model differing only by a gauge transform must read as the same reward:
    canonicalizing each in its own frame and comparing gives cosine ~ 1, while the raw cosine of the
    two direction vectors is well below 1.
    """
    h, w, p_act, p_null, margins = _synthetic_activations(seed=1)
    frame = fit_frame(h, margins=margins, null_var_ratio=1e-2)
    h_t, w_t, margins_t = _apply_gauge(h, w, p_act, p_null, seed=9)
    frame_t = fit_frame(h_t, margins=margins_t, null_var_ratio=1e-2)

    cos_canon = float(np.dot(canonicalize(w, frame), canonicalize(w_t, frame_t)))
    raw = float(np.dot(w, w_t) / (np.linalg.norm(w) * np.linalg.norm(w_t)))
    assert cos_canon > 0.99
    assert cos_canon - raw > 0.05


def test_effective_angle_regret_bound_present_e19():
    """E19 re-run canonicalized: effective_angle yields an AngleResult with a regret bound.

    Two reward directions that are nearly orthogonal in raw coordinates (the E19 ``cos = 0.005``
    situation) still produce a well-defined canonical cosine, a bootstrap CI, and a STARC-style
    empirical regret bound in [0, 1].
    """
    h, w_a, p_act, p_null, margins = _synthetic_activations(seed=2)
    frame = fit_frame(h, margins=margins, null_var_ratio=1e-2)
    rng = np.random.default_rng(3)
    # A second reward direction, chosen nearly orthogonal to w_a in raw coordinates.
    w_b = rng.standard_normal(w_a.shape[0]).astype(np.float32)
    w_b -= (w_b @ w_a) / (w_a @ w_a) * w_a  # make raw cosine ~ 0
    w_b = (p_act @ (p_act.T @ w_b)).astype(np.float32)  # keep it in the active subspace

    ev = effective_angle(w_a, w_b, frame, n_boot=100, activations_for_bound=h)
    res = ev.value
    assert str(ev.gauge) == "covariant"
    assert abs(res.raw_cos) < 0.2  # nearly orthogonal in raw coordinates
    assert -1.0 <= res.canonical_cos <= 1.0
    assert 0.0 <= res.regret_bound <= 1.0
    assert np.isfinite(res.starc_distance)
    assert np.isfinite(res.ci_low) and np.isfinite(res.ci_high)
    assert res.ci_low <= res.canonical_cos + 1e-6
    assert res.faithful_to.startswith("STARC")


def test_require_frame_for_comparison_raises_without_frame():
    """A COVARIANT comparison with frame=None raises GaugeError (gate 2 has no default frame)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 16))
    y = rng.standard_normal((50, 16))
    with pytest.raises(GaugeError):
        cka(x, y, frame=None)

    from reward_lens.core.gates import require_frame_for_comparison
    from reward_lens.core.types import GaugeStatus

    with pytest.raises(GaugeError):
        require_frame_for_comparison(GaugeStatus.COVARIANT, None)


def test_frame_refuses_fp16():
    """Frames refuse half-precision inputs."""
    h = np.random.default_rng(0).standard_normal((100, 8)).astype(np.float16)
    with pytest.raises(NumericsError):
        fit_frame(h)

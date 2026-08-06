"""Subspace comparison tests.

CKA is 1 for identical representations and ~0 for independent high-dimensional ones; Procrustes
recovers a known rotation to numerical zero disparity; subspace and feature alignment beat their
nulls only when the subspaces genuinely overlap. Every alignment is COVARIANT and refuses to run
without a shared frame (gate 2).
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.errors import GaugeError
from reward_lens.core.types import FrameID
from reward_lens.geometry import (
    cka,
    fit_frame,
    hungarian_feature_alignment,
    procrustes,
    subspace_alignment,
)

# A bare FrameID satisfies gate 2 for the functions that only need a shared-frame contract; the
# feature-alignment test fits a real Frame because canonicalization needs its whitening.
_FRAME = FrameID("frame:test")


def test_cka_identical_is_one():
    x = np.random.default_rng(0).standard_normal((200, 40))
    assert cka(x, x, frame=_FRAME) == pytest.approx(1.0, abs=1e-10)


def test_cka_independent_high_d_is_near_zero():
    # Independent high-dimensional representations have small CKA; the finite-sample floor is ~ d/n,
    # so ample samples relative to the dimension push it well below the identical value of 1.
    rng = np.random.default_rng(1)
    x = rng.standard_normal((800, 30))
    y = rng.standard_normal((800, 30))
    assert cka(x, y, frame=_FRAME) < 0.15


def test_procrustes_recovers_known_rotation():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((150, 12))
    rot, _ = np.linalg.qr(rng.standard_normal((12, 12)))
    y = x @ rot

    res = procrustes(x, y, frame=_FRAME)
    assert res.disparity < 1e-8, f"disparity not ~0: {res.disparity}"
    assert np.allclose(x @ res.rotation, y, atol=1e-6)


def test_subspace_alignment_beats_rum_null_when_overlapping():
    """Genuinely overlapping subspaces beat the RUM-identifiability null; independent ones do not."""
    rng = np.random.default_rng(3)
    d, k = 60, 6
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    a_basis = q[:, :k]
    # b shares 4 of the k directions, with 2 fresh ones: a real partial overlap.
    b_basis = np.column_stack([q[:, :4], q[:, k : k + 2]])

    res = subspace_alignment(a_basis, b_basis, frame=_FRAME, n_null=2000, seed=0)
    assert res.alignment > res.null_p95, (
        f"overlap did not beat null: {res.alignment} vs {res.null_p95}"
    )
    assert res.p_value < 0.05
    assert res.excess > 0

    # Two independent random k-subspaces sit at the null.
    indep = subspace_alignment(q[:, :k], q[:, d - k :], frame=_FRAME, n_null=2000, seed=0)
    assert indep.p_value > 0.05


def test_hungarian_feature_alignment_beats_random_null():
    """Matched feature sets that share directions beat the random-direction null in a shared frame."""
    rng = np.random.default_rng(4)
    d = 40
    # A frame whose whitening is well-conditioned (isotropic-ish reference distribution).
    frame = fit_frame(rng.standard_normal((400, d)).astype(np.float32))

    shared = rng.standard_normal((6, d))
    a_dirs = np.vstack([shared, rng.standard_normal((2, d))])
    b_dirs = np.vstack([shared + 0.05 * rng.standard_normal((6, d)), rng.standard_normal((2, d))])

    res = hungarian_feature_alignment(a_dirs, b_dirs, frame=frame, n_null=20000, seed=0)
    assert res.alignment > res.null_p95
    assert res.p_value < 0.05


def test_alignment_requires_frame():
    """Every alignment refuses to run without a shared frame (gate 2)."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal((50, 10))
    with pytest.raises(GaugeError):
        cka(x, x, frame=None)
    with pytest.raises(GaugeError):
        procrustes(x, x, frame=None)
    with pytest.raises(GaugeError):
        subspace_alignment(x[:, :3], x[:, 3:6], frame=None)

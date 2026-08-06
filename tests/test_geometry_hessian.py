"""Hessian spectroscopy tests.

The pure numerics are proven on synthetic operators with known spectra through the ``matvec = lambda
v: M @ v`` interface, so correctness holds on CPU with no model: Lanczos recovers the top-k
eigenvalues, ``participation_ratio`` matches its closed form, ``flat_subspace`` finds planted
near-zero-curvature directions, and SLQ integrates to the trace. A torch-gated test then runs
``hessian_spectrum`` on a tiny model through the ``Runtime.hvp`` protocol and confirms the top
eigenvalue matches a dense autograd reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.geometry import (
    SpectrumResult,
    flat_subspace,
    hessian_spectrum,
    lanczos,
    participation_ratio,
    slq_density,
)


def _random_spd(d: int, eigenvalues: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A symmetric matrix with a prescribed spectrum in a random eigenbasis."""
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    m = (q * eigenvalues) @ q.T
    return m, q


def test_lanczos_recovers_topk_eigenvalues():
    """Lanczos recovers the top-k eigenvalues of a known SPD matrix via the hvp=lambda v: M@v path.

    A geometrically decaying spectrum (as reward Hessians have) keeps the top eigenvalues
    well-separated, which is where Lanczos converges fastest.
    """
    d = 80
    eig = 100.0 * (0.85 ** np.arange(d))  # descending, well-separated top
    m, _ = _random_spd(d, eig, seed=1)

    vals, _ = lanczos(lambda v: m @ v, dim=d, k=10, n_iter=60)
    assert np.max(np.abs(vals - eig[:10])) < 1e-2, f"top-10 mismatch: {vals} vs {eig[:10]}"


def test_hessian_spectrum_evidence_matches_operator():
    """hessian_spectrum returns Evidence[SpectrumResult] whose eigenvalues match a known operator."""
    d = 64
    eig = 40.0 * (0.88 ** np.arange(d))  # decaying, well-separated top (reward-Hessian-like)
    m, _ = _random_spd(d, eig, seed=3)

    ev = hessian_spectrum(lambda v: m @ v, dim=d, k=8)
    assert isinstance(ev.value, SpectrumResult)
    assert str(ev.gauge) == "invariant"
    assert np.max(np.abs(np.asarray(ev.value.eigenvalues) - eig[:8])) < 1e-2
    assert ev.value.tail["lambda_max"] == pytest.approx(eig[0], abs=1e-2)
    assert ev.value.dim == d


def test_participation_ratio_closed_form():
    """participation_ratio matches the closed form (n equal eigenvalues -> n)."""
    assert participation_ratio([1.0, 1.0, 1.0, 1.0]) == pytest.approx(4.0)
    assert participation_ratio([10.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert participation_ratio([2.0, 2.0]) == pytest.approx(2.0)
    # (4+1+1+1+1)^2 / (16+1+1+1+1) = 64 / 20 = 3.2
    assert participation_ratio([4.0, 1.0, 1.0, 1.0, 1.0]) == pytest.approx(3.2)
    # k ones followed by zeros -> k
    k = 7
    spectrum = np.array([1.0] * k + [0.0] * 13)
    assert participation_ratio(spectrum) == pytest.approx(float(k))
    # magnitudes: mixed sign counted by magnitude
    assert participation_ratio([3.0, -3.0]) == pytest.approx(2.0)


def test_flat_subspace_finds_planted_flat_directions():
    """flat_subspace recovers the planted near-zero-curvature subspace."""
    d, n_flat = 48, 18
    eig = np.concatenate(
        [np.random.default_rng(4).uniform(5.0, 50.0, size=d - n_flat), np.zeros(n_flat)]
    )
    m, q = _random_spd(d, eig, seed=5)
    planted_null = q[:, d - n_flat :]  # eigenvectors with zero eigenvalue

    ev = flat_subspace(lambda v: m @ v, dim=d, tol=1e-3)
    res = ev.value
    assert res.n_flat == n_flat, f"expected {n_flat} flat directions, got {res.n_flat}"
    assert np.max(np.abs(res.curvatures)) < 1e-3
    # The recovered basis spans the planted null subspace (mean squared principal-angle cosine ~ 1).
    qb, _ = np.linalg.qr(np.asarray(res.basis))
    overlap = np.mean(np.linalg.svd(qb.T @ planted_null, compute_uv=False) ** 2)
    assert overlap > 0.99, f"recovered flat subspace does not match planted null: {overlap}"


def test_slq_density_integrates_to_trace():
    """The SLQ spectral density integrates to the dimension (mass conservation)."""
    d = 60
    eig = np.random.default_rng(6).uniform(1.0, 20.0, size=d)
    m, _ = _random_spd(d, eig, seed=7)
    grid, density = slq_density(lambda v: m @ v, dim=d, n_probes=24, seed=0)
    mass = np.trapezoid(density, grid) if hasattr(np, "trapezoid") else np.trapz(density, grid)
    assert mass == pytest.approx(d, rel=0.1)


# ---------------------------------------------------------------------------
# torch-gated: hessian_spectrum on a tiny model through Runtime.hvp
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


class _TinyRuntime:
    """A minimal object satisfying the Runtime.hvp contract for a tiny nonlinear reward head.

    The reward is ``r(h) = v . tanh(A h)`` for a fixed activation point ``h0``; its Hessian in ``h``
    is input-dependent, so this exercises real double-backprop, not a constant matrix. ``hvp`` matches
    the frozen protocol signature ``hvp(batch, scalar_fn, at, vecs)``.
    """

    def __init__(self, d: int, m: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.A = torch.randn(m, d, generator=g, dtype=torch.float64)
        self.v = torch.randn(m, generator=g, dtype=torch.float64)
        self.h0 = torch.randn(d, generator=g, dtype=torch.float64)
        self.d = d

    def _reward(self, h):
        from reward_lens.runtime.backend import RawOutput

        return RawOutput(reward=(self.v * torch.tanh(self.A @ h)).sum())

    def hvp(self, batch, scalar_fn, at, vecs):
        h = self.h0.clone().requires_grad_(True)
        scalar = scalar_fn(self._reward(h))
        (grad,) = torch.autograd.grad(scalar, h, create_graph=True)
        vt = torch.as_tensor(vecs, dtype=torch.float64)
        (hv,) = torch.autograd.grad(grad @ vt, h, retain_graph=True)
        return hv

    def dense_hessian(self):
        from torch.autograd.functional import hessian

        def f(h):
            return (self.v * torch.tanh(self.A @ h)).sum()

        return hessian(f, self.h0).detach().numpy()


def test_hessian_spectrum_on_tiny_model_via_runtime_hvp():
    """hessian_spectrum runs on a tiny model through runtime.hvp and matches a dense reference."""
    from reward_lens.core.types import Site
    from reward_lens.runtime.backend import RawOutput  # noqa: F401  (real payload type used above)

    d = 24
    rt = _TinyRuntime(d=d, m=32, seed=1)
    site = Site(layer=0, point="resid_post")

    ev = hessian_spectrum(
        rt,
        site=site,
        k=5,
        dim=d,
        scalar_fn=lambda out: out.reward,
        batch=object(),  # the toy runtime ignores the batch; a real backend would use it
    )
    top = float(ev.value.eigenvalues[0])
    assert np.isfinite(top)

    dense = rt.dense_hessian()
    dense_eig = np.sort(np.linalg.eigvalsh(0.5 * (dense + dense.T)))[::-1]
    # The Lanczos top eigenvalue (largest magnitude) matches the dense reference's extreme.
    assert abs(top) == pytest.approx(max(abs(dense_eig[0]), abs(dense_eig[-1])), abs=1e-3)

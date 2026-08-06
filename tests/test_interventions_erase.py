"""LEACE erasure and post-hoc certificates (``erase.py`` / ``certify.py``).

The proofs split cleanly along the grain of what is algebra and what is statistics. The defining
properties of the closed form (perfect linear erasure, idempotence, minimal edit) are algebraic
identities, so they are proven at the *moment* level against analytically specified covariances to
machine precision, with the minimal-edit form pinned to a hand-solved 2x2 optimum so the correct
whitening order cannot silently regress. The LEACE *guarantee* (a fresh probe cannot recover the
concept) is a statistical claim about generalization to unseen data, so it is proven at the *data*
level: fit on a train split, drop a held-out probe to chance. The certificate is then shown to
genuinely discriminate a real erasure from a sham, and the sensitive robustness arm is exercised
both when its dual-use attack generator is present and when it is honestly gated absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.evidence import evidence_from_envelope
from reward_lens.core.types import Site, TrustLevel
from reward_lens.interventions import certify as certify_mod
from reward_lens.interventions.certify import (
    ErasureCertificate,
    certify_erasure,
    certify_robustness,
    eraser_evidence,
    probe_recovery_auc,
)
from reward_lens.interventions.erase import Eraser, LeaceErasure, fit_leace, leace_matrix

EPS = 0.05  # the chance-band half-width for the LEACE guarantee (DESIGN: probe AUC ~ 0.5 +/- eps)


# ---------------------------------------------------------------------------
# Synthetic data with a planted linear concept
# ---------------------------------------------------------------------------


def _planted(seed: int, n: int, d: int, strength: float, k: int = 1):
    """Anisotropic Gaussian features with ``k`` planted binary concepts.

    The base features have a random (anisotropic) covariance ``A A^T``, so the whitening in LEACE is
    genuinely exercised rather than degenerating to a plain projection. Each concept ``j`` shifts the
    class mean by ``+/- strength`` along an orthonormal direction ``U[:, j]``; the ``k`` binary
    labels are independent. Returns ``(X, Z, U)`` with ``Z`` shape ``(n,)`` when ``k == 1`` else
    ``(n, k)``, and ``U`` the ``(d, k)`` concept directions.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d)) / np.sqrt(d)
    base = rng.standard_normal((n, d)) @ A.T
    U, _ = np.linalg.qr(rng.standard_normal((d, k)))  # d x k orthonormal concept directions
    labels = (rng.random((n, k)) < 0.5).astype(np.float64)
    X = base + (strength * (2.0 * labels - 1.0)) @ U.T
    Z = labels[:, 0] if k == 1 else labels
    return X, Z, U


# ---------------------------------------------------------------------------
# 1. The closed form: known-answer and algebraic identities (moment level)
# ---------------------------------------------------------------------------


def test_leace_matrix_matches_hand_solved_minimum():
    """The LEACE projection equals the hand-solved least-squares-minimal eraser on a 2x2 case.

    With ``Sigma = diag(4, 1)`` and a concept correlating equally with both raw coordinates
    (``Sigma_XZ = [1, 1]^T``), the minimal-edit eraser that satisfies ``P Sigma_XZ = Sigma_XZ`` while
    minimizing ``tr(P Sigma P^T)`` is, by a one-line Lagrange solve, ``P = [[0.2, 0.8], [0.2, 0.8]]``
    (edit cost 1.6). The transpose-ordered whitening ``W^+ Pi W`` also erases the concept but costs
    5.2, so pinning this matrix is what keeps the correct, minimal form from regressing to the wrong
    one.
    """
    sigma = np.diag([4.0, 1.0])
    sigma_xz = np.array([[1.0], [1.0]])
    p, rank = leace_matrix(sigma, sigma_xz)
    assert rank == 1
    np.testing.assert_allclose(p, [[0.2, 0.8], [0.2, 0.8]], atol=1e-12)
    # It genuinely satisfies the erasure constraint P Sigma_XZ = Sigma_XZ.
    np.testing.assert_allclose(p @ sigma_xz, sigma_xz, atol=1e-12)


def test_leace_perfect_erasure_and_idempotent_at_moment_level():
    """``(I - P) Sigma_XZ = 0`` and ``P^2 = P`` exactly, on an anisotropic analytic covariance.

    These are algebraic identities of the construction, not statistical claims, so they hold to
    machine precision when ``P`` is built from exact moments. ``(I - P) Sigma_XZ = 0`` is the LEACE
    guarantee at the population level: the erased feature has exactly zero covariance with the
    concept.
    """
    rng = np.random.default_rng(1)
    d, k = 8, 2
    B = rng.standard_normal((d, d))
    sigma = B @ B.T + 0.1 * np.eye(d)  # SPD, anisotropic
    sigma_xz = rng.standard_normal((d, k))
    p, rank = leace_matrix(sigma, sigma_xz)
    assert rank == k
    np.testing.assert_allclose((np.eye(d) - p) @ sigma_xz, 0.0, atol=1e-9)
    np.testing.assert_allclose(p @ p, p, atol=1e-9)


def test_leace_preserves_orthogonal_direction_minimal_edit():
    """A direction orthogonal to the concept is left untouched by the erasure (minimal edit).

    Constructed so it is exact: with ``Sigma = U diag(lambda) U^T`` and the concept covariance along
    a single eigenvector ``U[:, 0]``, the eraser removes only that direction. For any ``w`` in the
    span of the other eigenvectors, ``P^T w = 0``, so ``(r(x) - x) . w = -(P (x - mu)) . w = 0`` for
    every ``x``: the orthogonal component's contribution is preserved exactly. Meanwhile the concept
    eigendirection is fully projected (``P U[:,0] = U[:,0]``).
    """
    rng = np.random.default_rng(2)
    d = 6
    U, _ = np.linalg.qr(rng.standard_normal((d, d)))
    lam = np.array([5.0, 4.0, 3.0, 2.0, 1.5, 1.0])
    sigma = (U * lam) @ U.T
    concept = U[:, 0]
    sigma_xz = (2.3 * concept)[:, None]  # concept covariance along one eigendirection
    p, rank = leace_matrix(sigma, sigma_xz)
    assert rank == 1
    # The concept direction is fully captured; every orthogonal eigendirection is untouched.
    np.testing.assert_allclose(p @ concept, concept, atol=1e-9)
    for j in range(1, d):
        w_perp = U[:, j]
        np.testing.assert_allclose(p.T @ w_perp, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 2. The LEACE guarantee: held-out probe drops to chance (data level)
# ---------------------------------------------------------------------------


def test_leace_guarantee_heldout_probe_drops_to_chance(capsys):
    """A fresh held-out probe for the concept is >> 0.5 before erasure and ~0.5 after.

    This is the whole point of LEACE. The eraser is fit on the train split only; the probe is then
    trained and evaluated on a disjoint held-out split, both before and after applying the
    train-fitted eraser. Before erasure the concept is strongly decodable; after, it collapses into
    the chance band, which no train/held-out overfitting can fake because the erasure was never
    shown the held-out data.
    """
    X, z, _ = _planted(seed=10, n=12000, d=20, strength=1.5)
    n_tr = 8000
    X_tr, z_tr = X[:n_tr], z[:n_tr]
    X_ho, z_ho = X[n_tr:], z[n_tr:]
    n_ho = X_ho.shape[0]
    half = n_ho // 2

    auc_pre = probe_recovery_auc(X_ho[:half], z_ho[:half], X_ho[half:], z_ho[half:])

    eraser = fit_leace(X_tr, z_tr, fit_data_id="ds:train10", concept_id="concept:planted")
    E_ho = eraser.apply(X_ho)
    auc_post = probe_recovery_auc(E_ho[:half], z_ho[:half], E_ho[half:], z_ho[half:])

    print(
        f"[LEACE guarantee] held-out probe AUC pre-erasure={auc_pre:.4f} post-erasure={auc_post:.4f}"
    )
    assert auc_pre > 0.90, f"concept should be strongly decodable pre-erasure; got {auc_pre:.4f}"
    assert abs(auc_post - 0.5) <= EPS, f"post-erasure AUC {auc_post:.4f} not in chance band"


def test_leace_idempotent_and_affine_on_data():
    """The eraser is affine and idempotent: ``r(r(X)) == r(X)`` on data, to ~1e-9."""
    X, z, _ = _planted(seed=11, n=4000, d=12, strength=1.5)
    eraser = fit_leace(X, z)
    once = eraser.apply(X)
    twice = eraser.apply(once)
    np.testing.assert_allclose(twice, once, atol=1e-9)
    # Affine: r acts as x - (x - mu) P^T, so the difference r(X) - X is linear in X with no constant
    # beyond the mu term; check two samples' erasure matches the explicit affine map.
    explicit = X - (X - eraser.mu) @ eraser.P.T
    np.testing.assert_allclose(once, explicit, atol=1e-12)


def test_leace_multidim_subspace_all_concepts_drop(capsys):
    """Erasing a k-dim concept subspace drops a probe for every concept in the span to chance.

    Three independent binary concepts are planted along orthonormal directions and erased together
    as an ``(n, 3)`` subspace. On held-out data a fresh probe for each individual concept, and for a
    random linear combination that also lives in the erased span, collapses into the chance band,
    while every one of them was decodable before erasure.
    """
    k = 3
    X, Z, U = _planted(seed=12, n=26000, d=24, strength=1.6, k=k)
    n_tr = 20000
    X_tr, Z_tr = X[:n_tr], Z[:n_tr]
    X_ho, Z_ho = X[n_tr:], Z[n_tr:]
    half = X_ho.shape[0] // 2

    eraser = fit_leace(X_tr, Z_tr, concept_id="concept:3d")
    assert eraser.rank == k
    E_ho = eraser.apply(X_ho)

    for j in range(k):
        zj = Z_ho[:, j]
        pre = probe_recovery_auc(X_ho[:half], zj[:half], X_ho[half:], zj[half:])
        post = probe_recovery_auc(E_ho[:half], zj[:half], E_ho[half:], zj[half:])
        print(f"[LEACE k=3] concept {j}: pre={pre:.4f} post={post:.4f}")
        assert pre > 0.90
        assert abs(post - 0.5) <= EPS

    # A random combination concept, defined purely from the planted labels (so its signal lies in
    # the erased span), also drops to chance after erasure.
    rng = np.random.default_rng(99)
    w = rng.standard_normal(k)
    combo = (Z_ho @ w > np.median(Z_ho @ w)).astype(np.float64)
    post_combo = probe_recovery_auc(E_ho[:half], combo[:half], E_ho[half:], combo[half:])
    print(f"[LEACE k=3] combination concept: post={post_combo:.4f}")
    assert abs(post_combo - 0.5) <= EPS


# ---------------------------------------------------------------------------
# 3. Surfaces: head-only readout projection and the Intervention hook
# ---------------------------------------------------------------------------


def test_head_only_readout_surface_matches_feature_erasure():
    """Projecting ``w_r`` against the concept subspace reproduces erased-feature reward up to a shift.

    The head-only surface (DESIGN line 601) edits the reward direction once instead of mounting a
    hook. The reward read from erased features and the reward read with the projected ``w_r'`` must
    agree up to a single per-sample-constant offset, which the reward gauge quotients out; so their
    difference is constant across samples.
    """
    X, z, _ = _planted(seed=13, n=3000, d=10, strength=1.5)
    eraser = fit_leace(X, z)
    rng = np.random.default_rng(7)
    w_r = rng.standard_normal(10)
    w_r_proj = eraser.apply_to_readout(w_r)

    reward_erased = eraser.apply(X) @ w_r
    reward_headonly = X @ w_r_proj
    diff = reward_erased - reward_headonly
    assert float(np.std(diff)) < 1e-6, (
        f"head-only reward should match up to a constant; std={np.std(diff)}"
    )


def test_leace_intervention_hook_matches_numpy():
    """The Intervention mount hook applies the same affine map as ``Eraser.apply`` (torch surface)."""
    torch = pytest.importorskip("torch")
    X, z, _ = _planted(seed=14, n=2000, d=8, strength=1.5)
    eraser = fit_leace(X, z, fit_data_id="ds:hook", concept_id="concept:hook")
    site = Site(3, "resid_post")
    iv = LeaceErasure(eraser, sites=(site,))
    compiled = iv.compile(signal=None)

    assert compiled.fingerprint == iv.fingerprint()
    assert str(site) in [str(s) for s in [site]]  # site key present
    assert site in compiled.mounts
    assert compiled.meta["fit_data_id"] == "ds:hook"
    assert compiled.meta["rank"] == eraser.rank

    hook = compiled.mounts[site]
    hidden = torch.as_tensor(
        np.random.default_rng(0).standard_normal((2, 4, 8)), dtype=torch.float64
    )
    out = hook(hidden, {"site": site}).numpy()
    expected = eraser.apply(hidden.numpy().reshape(-1, 8)).reshape(2, 4, 8)
    np.testing.assert_allclose(out, expected, atol=1e-9)


def test_eraser_fingerprint_binds_fit_data_provenance():
    """The eraser fingerprint folds in its fit-data id (R8): different provenance, different id."""
    X, z, _ = _planted(seed=15, n=1500, d=6, strength=1.5)
    e1 = fit_leace(X, z, fit_data_id="ds:A", concept_id="c")
    e2 = fit_leace(X, z, fit_data_id="ds:B", concept_id="c")
    assert e1.fit_data_id == "ds:A" and e2.fit_data_id == "ds:B"
    assert e1.fingerprint() != e2.fingerprint()
    # Same everything reproduces the same id.
    e1b = fit_leace(X, z, fit_data_id="ds:A", concept_id="c")
    assert e1.fingerprint() == e1b.fingerprint()


# ---------------------------------------------------------------------------
# 4. The erasure certificate: real vs sham discrimination, and gate 1 trust
# ---------------------------------------------------------------------------


def _sham_eraser(X: np.ndarray, rank: int, seed: int) -> Eraser:
    """A same-rank random affine map: the operation of an eraser aimed at the wrong subspace."""
    d = X.shape[1]
    V, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, rank)))
    p_sham = V @ V.T
    return Eraser(
        P=p_sham,
        mu=X.mean(axis=0),
        rank=rank,
        dim=d,
        method="sham",
        fit_data_id="ds:sham",
        concept_id="concept:sham",
    )


def test_certificate_discriminates_real_erasure_from_sham(capsys):
    """The certificate reports ~0.5 recovery for a real LEACE erase and >0.5 for a sham erase.

    Both erasers are the same shape (rank-1 affine maps); only the real one targets the concept
    subspace. Certified on identical held-out data, the real erase yields a recovery AUC in the
    chance band and a CALIBRATED certificate; the sham leaves the concept linearly present, so its
    recovery AUC is high and the certificate refuses calibration, staying EXPLORATORY. This is the
    certificate genuinely discriminating a real erasure from a fake one (DESIGN line 603).
    """
    X, z, _ = _planted(seed=20, n=16000, d=20, strength=1.5)
    n_tr = 10000
    X_tr, z_tr = X[:n_tr], z[:n_tr]
    X_ho, z_ho = X[n_tr:], z[n_tr:]

    real = fit_leace(X_tr, z_tr, fit_data_id="ds:train20", concept_id="concept:planted")
    sham = _sham_eraser(X_tr, rank=real.rank, seed=123)

    cert_real = certify_erasure(real, X_ho, z_ho, eps=EPS, concept_id="concept:planted")
    cert_sham = certify_erasure(sham, X_ho, z_ho, eps=EPS, concept_id="concept:planted")

    print(
        f"[certificate] real recovery AUC={cert_real.value.recovery_auc:.4f} "
        f"(passed={cert_real.value.passed}, trust={cert_real.trust}); "
        f"sham recovery AUC={cert_sham.value.recovery_auc:.4f} "
        f"(passed={cert_sham.value.passed}, trust={cert_sham.trust})"
    )

    # Real erase: certificate passes, sits in the chance band, and is CALIBRATED by gate 1.
    assert cert_real.value.passed is True
    assert abs(cert_real.value.recovery_auc - 0.5) <= EPS
    assert cert_real.trust == TrustLevel.CALIBRATED
    assert cert_real.is_calibrated

    # Sham erase: certificate fails, recovery stays well above the band, and it is EXPLORATORY.
    assert cert_sham.value.passed is False
    assert cert_sham.value.recovery_auc > 0.5 + EPS
    assert cert_sham.trust == TrustLevel.EXPLORATORY
    assert not cert_sham.is_calibrated
    # The certificate is a stored Evidence object carrying provenance: it round-trips through the
    # store envelope exactly.
    restored = evidence_from_envelope(cert_real.envelope())
    assert isinstance(restored.value, ErasureCertificate)
    assert restored.value.recovery_auc == pytest.approx(cert_real.value.recovery_auc)
    assert restored.trust == TrustLevel.CALIBRATED


def test_eraser_without_certificate_is_exploratory():
    """An uncertified eraser is EXPLORATORY (gate 1); a passing certificate lifts it to CALIBRATED."""
    X, z, _ = _planted(seed=21, n=12000, d=16, strength=1.5)
    n_tr = 8000
    eraser = fit_leace(X[:n_tr], z[:n_tr], fit_data_id="ds:t21", concept_id="c")

    bare = eraser_evidence(eraser)
    assert bare.trust == TrustLevel.EXPLORATORY
    assert not bare.is_calibrated

    cert = certify_erasure(eraser, X[n_tr:], z[n_tr:], eps=EPS)
    assert cert.value.passed is True
    certified = eraser_evidence(eraser, cert)
    assert certified.trust == TrustLevel.CALIBRATED
    assert certified.is_calibrated


# ---------------------------------------------------------------------------
# 5. The robustness certificate (SENSITIVE arm): real path and honest skip
# ---------------------------------------------------------------------------


def test_robustness_certificate_real_attack_finds_budget(capsys):
    """With the dual-use attack generator present, the certificate finds a finite rebreak budget.

    On a real LEACE erase, an attacker who may perturb each erased feature by up to budget ``B``
    toward its concept class restores separability as ``B`` grows. Recovery AUC rises monotonically
    from the chance band, and ``budget_to_rebreak`` is the smallest tested budget reaching the
    rebreak threshold. This genuinely calls ``geometry.hessian.gradient_ascent_probe`` (lazy).
    """
    X, z, U = _planted(seed=30, n=1600, d=12, strength=1.5)
    n_tr = 1000
    eraser = fit_leace(X[:n_tr], z[:n_tr], concept_id="c")
    X_ho, z_ho = X[n_tr:], z[n_tr:]

    ev = certify_robustness(
        eraser, X_ho, z_ho, U[:, 0], budgets=(0.0, 1.0, 2.0, 4.0, 8.0), rebreak_auc=0.6, seed=1
    )
    cert = ev.value
    print(
        f"[robustness] budgets={cert.budgets} recovered={['%.3f' % a for a in cert.recovered_auc]} "
        f"budget_to_rebreak={cert.budget_to_rebreak}"
    )
    assert cert.skipped is False
    assert cert.recovered_auc[0] < 0.6  # budget 0 is still erased (chance band)
    assert cert.recovered_auc[-1] >= 0.6  # a large budget rebreaks
    assert cert.recovered_auc[-1] >= cert.recovered_auc[0]  # recovery grows with budget
    assert cert.budget_to_rebreak is not None and cert.budget_to_rebreak > 0.0
    assert cert.sensitivity == "sensitive:dual-use"


def test_robustness_certificate_skips_honestly_when_attack_absent():
    """With the attack generator unavailable the arm skips and never fabricates a budget (RK8)."""
    X, z, U = _planted(seed=31, n=600, d=8, strength=1.5)
    eraser = fit_leace(X[:400], z[:400], concept_id="c")

    ev = certify_robustness(eraser, X[400:], z[400:], U[:, 0], attack_loader=lambda: None)
    cert = ev.value
    assert cert.skipped is True
    assert cert.budget_to_rebreak is None  # no fabricated budget
    assert cert.recovered_auc == []
    assert "unavailable" in cert.reason
    assert ev.trust == TrustLevel.EXPLORATORY


def test_attack_generator_is_not_reexported():
    """The dual-use generator is neither in certify's public exports nor imported at module scope."""
    assert "gradient_ascent_probe" not in certify_mod.__all__
    # Lazy import: certify does not hold the sensitive symbol as a module attribute.
    assert not hasattr(certify_mod, "gradient_ascent_probe")

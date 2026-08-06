"""The index library computes its catalogue definitions correctly on synthetic data.

Each index is proved two ways. Its pure function is exercised on synthetic data where the answer is
known in closed form: chi recovers a planted covariance, tail recovers a planted exponential scale and a
planted polynomial index, participation ratio and the Welch floor match their closed forms, KUI is high
for a represented-but-unpriced property and low for a priced one. Then the Observable is run through the
gated runner on a deterministic ``FakeSignal`` (activations and scores set so the known answer survives
to the Evidence) so the production path, the gates, and ``ctx.emit`` are covered without a model, plus a
single real-substrate smoke test on the tiny model.

The ``FakeSignal`` exposes exactly the surface the indices touch through the frozen protocol: ``meta``,
``caps``, ``readouts``, ``score``, and ``capture``. Its capture returns the same activation matrix for
any requested site (the index math, not per-site fidelity, is what these tests assert), and its score
returns the preset reward, so a chi computed on ``d_0 = w_r`` equals the teacher-variance exactly.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from reward_lens.core.errors import CapabilityError
from reward_lens.core.types import Capability, GaugeStatus, Site, TrustLevel
from reward_lens.measure import base as mb
from reward_lens.measure import indices as idx
from reward_lens.measure.indices import Property
from reward_lens.signals.base import PositionSpec, Readout

# ---------------------------------------------------------------------------
# A deterministic fake signal exposing only the protocol surface the indices touch
# ---------------------------------------------------------------------------


class _AnySite(dict):
    """A tensor map that returns the same activation matrix for any requested Site."""

    def __init__(self, acts: np.ndarray) -> None:
        super().__init__()
        self._acts = acts

    def __getitem__(self, key: object) -> np.ndarray:
        return self._acts

    def __contains__(self, key: object) -> bool:
        return True


class FakeSignal:
    """Minimal ``RewardSignal`` stand-in with settable activations, scores, and readouts."""

    def __init__(
        self,
        *,
        caps: Capability,
        w_r: np.ndarray | None = None,
        activations: np.ndarray | None = None,
        scores: np.ndarray | None = None,
        n_layers: int | None = None,
        criterion_dirs: np.ndarray | None = None,
    ) -> None:
        self.caps = caps
        d = 0
        if w_r is not None:
            d = int(np.asarray(w_r).size)
        elif activations is not None:
            d = int(np.asarray(activations).shape[1])
        self.meta = SimpleNamespace(fingerprint="mfp:fake", n_layers=n_layers, d_model=d)
        self._acts = None if activations is None else np.asarray(activations, dtype=np.float64)
        self._scores = None if scores is None else np.asarray(scores, dtype=np.float64)
        site = Site(max((n_layers or 1) - 1, 0), "resid_post")
        pos = PositionSpec("final")
        ro: list[Readout] = []
        if w_r is not None:
            ro.append(Readout("reward", "linear", site, pos, vector=np.asarray(w_r)))
        if criterion_dirs is not None:
            for i, v in enumerate(np.asarray(criterion_dirs)):
                ro.append(Readout(f"criterion:{i}", "linear", Site(0, "resid_post"), pos, vector=v))
        self._readouts = ro

    def readouts(self) -> list[Readout]:
        return list(self._readouts)

    def score(self, view: object, readout: str = "reward") -> object:
        return SimpleNamespace(value=SimpleNamespace(values=self._scores))

    def capture(self, view: object, spec: object) -> list[object]:
        return [SimpleNamespace(tensors=_AnySite(self._acts))]


def _run(observable: mb.BaseObservable, signal: FakeSignal, **ctx_kw: object) -> object:
    return mb.run(observable, mb.Context(signal=signal, **ctx_kw))


# ---------------------------------------------------------------------------
# A3 TeacherCompatibility
# ---------------------------------------------------------------------------


def test_teacher_compatibility_equals_quadratic_form():
    rng = np.random.default_rng(0)
    d, n = 5, 4000
    lmat = rng.standard_normal((d, d))
    z = rng.standard_normal((n, d))
    acts = z @ lmat.T  # covariance L L^T
    w = rng.standard_normal(d)
    tc = idx.teacher_compatibility(w, acts)
    sigma = lmat @ lmat.T
    assert tc == pytest.approx(float(w @ sigma @ w), rel=0.05)


def test_teacher_compatibility_spectral_sums_to_total_and_diagonal_closed_form():
    rng = np.random.default_rng(1)
    d, n = 4, 6000
    variances = np.array([4.0, 1.0, 0.25, 9.0])
    acts = rng.standard_normal((n, d)) * np.sqrt(variances)
    w = np.array([1.0, -2.0, 0.5, 0.3])
    total, contribs, _evals = idx.teacher_compatibility_spectral(w, acts)
    assert total == pytest.approx(idx.teacher_compatibility(w, acts), rel=1e-9)
    assert float(np.sum(contribs)) == pytest.approx(total, rel=1e-9)
    # diagonal covariance: the total is sum_i var_i * w_i^2
    assert total == pytest.approx(float(np.sum(variances * w**2)), rel=0.05)


def test_teacher_compatibility_observable_emits_gated_evidence():
    rng = np.random.default_rng(2)
    d, n = 6, 1500
    acts = rng.standard_normal((n, d))
    w = np.array([1.0, 0.5, 0.0, 0.0, -0.3, 0.0])
    sig = FakeSignal(
        caps=Capability.ACTIVATIONS | Capability.LINEAR_READOUT,
        w_r=w,
        activations=acts,
        n_layers=2,
    )
    ev = _run(idx.TeacherCompatibility(), sig)
    assert ev.observable == "TeacherCompatibility"
    assert ev.trust is TrustLevel.EXPLORATORY
    assert ev.gauge is GaugeStatus.INVARIANT
    assert ev.value["teacher_compatibility"] == pytest.approx(idx.teacher_compatibility(w, acts))
    assert len(ev.value["by_layer"]) == 2


def test_teacher_compatibility_capability_gate():
    sig = FakeSignal(caps=Capability.SCORES, w_r=np.ones(3), activations=np.ones((4, 3)))
    with pytest.raises(CapabilityError):
        _run(idx.TeacherCompatibility(), sig)


# ---------------------------------------------------------------------------
# A12 chi
# ---------------------------------------------------------------------------


def test_chi_recovers_planted_covariance():
    rng = np.random.default_rng(3)
    n = 8000
    f0 = rng.standard_normal(n)
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    features = np.column_stack([f0, f1, f2])
    reward = 3.0 * f0 + 0.0 * f1 + rng.standard_normal(n) * 0.1  # only f0 drives r
    chi = idx.susceptibility(features, reward)
    assert chi[0] == pytest.approx(3.0, abs=0.15)  # Cov(f0, 3 f0) = 3 Var(f0) = 3
    assert chi[1] == pytest.approx(0.0, abs=0.1)
    assert chi[2] == pytest.approx(0.0, abs=0.1)


def test_chi_predicted_hack_modes():
    chi = np.array([0.5, 0.2, -0.1, 0.3])  # reward pushes 0, 1, 3 up
    chi_gold = np.array([-0.2, 0.4, -0.5, -0.05])  # gold prices only 1; 0 and 3 are unpriced
    hacks = idx.predicted_hack_modes(chi, chi_gold)
    assert hacks.tolist() == [True, False, False, True]


def test_chi_observable_matches_teacher_variance_on_w_r_feature():
    rng = np.random.default_rng(4)
    d, n = 6, 3000
    acts = rng.standard_normal((n, d))
    w = np.array([1.0, 0.5, -0.2, 0.0, 0.0, 0.1])
    reward = acts @ w  # the exact linear reward, so Cov(w.h, r) = Var(r) = TC
    directions = np.vstack([w, rng.standard_normal((3, d))])
    bank = idx.LinearFeatureBank(directions, names=("wr", "r1", "r2", "r3"))
    sig = FakeSignal(
        caps=Capability.ACTIVATIONS | Capability.SCORES | Capability.LINEAR_READOUT,
        w_r=w,
        activations=acts,
        scores=reward,
    )
    ev = _run(idx.Chi(feature_bank=bank, null_draws=200), sig)
    tc = idx.teacher_compatibility(w, acts)
    assert ev.value["chi"][0] == pytest.approx(tc, rel=1e-6)
    assert len(ev.value["null_p_values"]) == 4
    assert ev.value["null_p_values"][0] < 0.05  # the w_r feature is far from the shuffle null


def test_chi_observable_flags_hacks_with_gold():
    rng = np.random.default_rng(5)
    d, n = 4, 4000
    acts = rng.standard_normal((n, d))
    w = np.array([1.0, 0.0, 0.0, 0.0])  # reward loves direction 0
    gold_w = np.array([-0.5, 1.0, 0.0, 0.0])  # gold prices direction 1, penalizes direction 0
    reward = acts @ w
    gold_reward = acts @ gold_w
    directions = np.eye(d)
    bank = idx.LinearFeatureBank(directions)
    caps = Capability.ACTIVATIONS | Capability.SCORES | Capability.LINEAR_READOUT
    sig = FakeSignal(caps=caps, w_r=w, activations=acts, scores=reward)
    gold = FakeSignal(caps=caps, w_r=gold_w, activations=acts, scores=gold_reward)
    ev = _run(idx.Chi(feature_bank=bank, gold=gold, null_draws=100), sig)
    # feature 0 is pushed up by reward but not by gold: a predicted hack mode
    assert ev.value["predicted_hack_modes"][0] is True
    assert ev.value["predicted_hack_modes"][1] is False


def test_chi_observable_degrades_without_bank():
    rng = np.random.default_rng(6)
    acts = rng.standard_normal((50, 4))
    sig = FakeSignal(
        caps=Capability.ACTIVATIONS | Capability.SCORES | Capability.LINEAR_READOUT,
        w_r=np.ones(4),
        activations=acts,
        scores=rng.standard_normal(50),
    )
    ev = _run(idx.Chi(feature_bank=None, null_draws=10), sig)
    assert ev.value["chi"] == []
    assert "no feature bank" in ev.value["note"]


# ---------------------------------------------------------------------------
# A4 tail
# ---------------------------------------------------------------------------


def test_tail_recovers_exponential_scale():
    rng = np.random.default_rng(7)
    tau = 2.5
    r = rng.exponential(scale=tau, size=40000)
    est = idx.tail_estimate(r, quantile=0.9)
    assert est["regime"] == "exponential"
    assert est["tau"] == pytest.approx(tau, rel=0.2)
    assert est["lambda_c"] == pytest.approx(1.0 / tau, rel=0.25)


def test_tail_recovers_polynomial_index():
    rng = np.random.default_rng(8)
    alpha = 3.0
    r = rng.pareto(alpha, size=40000) + 1.0  # standard Pareto, tail index alpha
    _gamma, hill_alpha = idx.hill_tail_index(r)
    assert hill_alpha == pytest.approx(alpha, rel=0.35)
    est = idx.tail_estimate(r, quantile=0.9)
    assert est["regime"] == "polynomial"
    assert est["lambda_c"] == 0.0  # catastrophic: MGF diverges for all lambda > 0


def test_tail_observable_emits_with_ci():
    rng = np.random.default_rng(9)
    r = rng.exponential(scale=1.5, size=5000)
    sig = FakeSignal(caps=Capability.SCORES, scores=r)
    ev = _run(idx.TailIndex(bootstrap=100), sig)
    assert ev.value["aggregate"]["regime"] == "exponential"
    assert ev.uncertainty.ci_low is not None and ev.uncertainty.ci_high is not None


# ---------------------------------------------------------------------------
# A1 KUI
# ---------------------------------------------------------------------------


def test_kui_percentiles_are_within_unit_interval():
    pct = idx.percentile_within_battery(np.array([10.0, 20.0, 20.0, 5.0, 100.0]))
    assert np.all((pct >= 0.0) & (pct <= 1.0))
    assert pct[3] == pytest.approx(pct.min())  # the smallest value gets the smallest percentile


def test_kui_high_for_represented_but_unpriced_low_for_priced():
    # battery: index 0 = represented-but-unpriced, index 1 = represented-and-priced
    decode = [0.95, 0.95, 0.6, 0.55, 0.7]
    mediate = [0.01, 0.90, 0.30, 0.35, 0.40]
    plane = idx.kui_plane(decode, mediate)
    kui = plane["kui"]
    assert int(np.argmax(kui)) == 0  # the unpriced property has the highest KUI
    assert kui[0] > kui[1]  # unpriced beats priced
    assert kui[1] < 0  # the priced property sits below the diagonal


def test_kui_from_properties_uses_linear_proxy():
    rng = np.random.default_rng(10)
    d = 8
    w_r = rng.standard_normal(d)
    parallel = w_r.copy()  # priced: parallel to w_r -> high mediation
    orthogonal = np.linalg.svd(w_r[None, :])[2][-1]  # a direction orthogonal to w_r
    props = [
        Property("unpriced", decodability=0.95, direction=orthogonal),
        Property("priced", decodability=0.95, direction=parallel),
        Property("mid", decodability=0.6, direction=rng.standard_normal(d)),
    ]
    result = idx.kui_from_properties(props, w_r)
    assert result["mediate_raw"][1] == pytest.approx(1.0, abs=1e-6)  # parallel cos = 1
    assert result["mediate_raw"][0] == pytest.approx(0.0, abs=1e-6)  # orthogonal cos = 0
    assert result["kui"][0] > result["kui"][1]


def test_kui_observable_emits_plane():
    rng = np.random.default_rng(11)
    d = 8
    w_r = rng.standard_normal(d)
    orthogonal = np.linalg.svd(w_r[None, :])[2][-1]
    props = [
        Property("unpriced", 0.95, direction=orthogonal),
        Property("priced", 0.95, direction=w_r.copy()),
        Property("mid", 0.5, direction=rng.standard_normal(d)),
    ]
    sig = FakeSignal(caps=Capability.ACTIVATIONS | Capability.LINEAR_READOUT, w_r=w_r)
    ev = _run(idx.KUI(properties=props, null_draws=500), sig)
    assert ev.value["names"][int(np.argmax(ev.value["kui"]))] == "unpriced"
    assert "mediation_null_p95" in ev.value


# ---------------------------------------------------------------------------
# A9 coherence
# ---------------------------------------------------------------------------


def test_coherence_orthonormal_criteria():
    directions = np.eye(5)  # K = d = 5 orthonormal criteria
    report = idx.coherence_report(directions)
    assert report["max_offdiagonal_coherence"] == pytest.approx(0.0, abs=1e-9)
    assert report["d_eff"] == pytest.approx(5.0, rel=1e-6)
    assert report["welch_bound"] == 0.0  # K <= d, no real floor


def test_coherence_identical_criteria_collapse_d_eff():
    base = np.array([1.0, 2.0, -1.0, 0.5])
    directions = np.vstack([base, base, base])  # three identical criteria
    report = idx.coherence_report(directions)
    assert report["d_eff"] == pytest.approx(1.0, rel=1e-6)
    assert report["max_offdiagonal_coherence"] == pytest.approx(1.0, rel=1e-9)


def test_welch_bound_closed_form():
    # sqrt((K - d) / (d (K - 1)))
    assert idx.welch_bound(5, 3) == pytest.approx(np.sqrt(2 / 12))
    assert idx.welch_bound(3, 5) == 0.0  # K <= d


def test_coherence_matrix_matches_dot_products():
    rng = np.random.default_rng(12)
    v = rng.standard_normal((4, 6))
    mu = idx.coherence_matrix(v)
    unit = v / np.linalg.norm(v, axis=1, keepdims=True)
    assert np.allclose(mu, unit @ unit.T)
    assert np.allclose(np.diag(mu), 1.0)


def test_coherence_observable_emits():
    rng = np.random.default_rng(13)
    directions = rng.standard_normal((6, 10))
    sig = FakeSignal(caps=Capability.MULTI_READOUT)
    ev = _run(idx.Coherence(directions=directions, null_draws=500), sig)
    assert ev.value["n_criteria"] == 6
    assert "coherence_null_p95" in ev.value
    assert ev.gauge is GaugeStatus.INVARIANT


# ---------------------------------------------------------------------------
# A2 distortion + coverage disparity
# ---------------------------------------------------------------------------


def test_distortion_weights_by_uncovered():
    dist = idx.distortion_per_dimension([1.0, 1.0, 2.0], [0.0, 1.0, 0.5])
    assert dist.tolist() == pytest.approx([1.0, 0.0, 1.0])


def test_linear_sensitivity_closed_form():
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    w_r = np.array([2.0, 0.0, 0.0])
    sens = idx.linear_sensitivity(directions, w_r)
    assert sens.tolist() == pytest.approx([2.0, 0.0])


def test_distortion_observable_with_injected_curvature():
    sig = FakeSignal(caps=Capability.ACTIVATIONS | Capability.LINEAR_READOUT, w_r=np.ones(3))
    ev = _run(idx.Distortion(curvature=[2.0, 1.0], coverage=[0.0, 1.0]), sig)
    assert ev.value["distortion"] == pytest.approx([2.0, 0.0])


def test_coverage_disparity_uniform_is_zero():
    assert idx.coverage_disparity([0.5, 0.5, 0.5])["range"] == 0.0
    rep = idx.coverage_disparity([0.0, 1.0])
    assert rep["range"] == 1.0
    assert rep["mean_coverage"] == 0.5


def test_coverage_disparity_observable_is_not_faithful_to_distortion():
    assert idx.CoverageDisparity.faithful_to is None
    sig = FakeSignal(caps=Capability.SCORES)
    ev = _run(idx.CoverageDisparity(coverage=[0.2, 0.8, 0.5]), sig)
    assert ev.value["range"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# A6 verification score + style share
# ---------------------------------------------------------------------------


def test_verification_score_ratio():
    assert idx.verification_score(2.0, 1.5) == 0.75
    assert np.isnan(idx.verification_score(0.0, 1.0))


def test_style_share_recovers_planted_fraction():
    w_r = np.array([1.0, 1.0, 0.0])
    style_basis = np.array([[1.0, 0.0, 0.0]])  # style is the e0 axis
    delta_h = np.array([3.0, 4.0, 0.0])  # 3 along style, 4 along content
    # w_r . P_style dh = 3 ; w_r . dh = 7 ; share = 3/7
    assert idx.style_share(delta_h, style_basis, w_r) == pytest.approx(3.0 / 7.0)


def test_verification_and_style_observables_emit():
    sig = FakeSignal(caps=Capability.STEP_SCORES)
    ev = _run(idx.VerificationScore(dr_total=2.0, dr_error_span=1.0), sig)
    assert ev.value["verification_score"] == 0.5

    sig2 = FakeSignal(caps=Capability.ACTIVATIONS | Capability.LINEAR_READOUT, w_r=np.ones(3))
    ev2 = _run(
        idx.StyleShare(delta_h=np.array([1.0, 0.0, 0.0]), style_basis=np.array([[1.0, 0.0, 0.0]])),
        sig2,
    )
    assert ev2.value["style_share"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# A10 dark reward + A11 interp coverage
# ---------------------------------------------------------------------------


def test_dark_reward_recovers_noise_fraction():
    rng = np.random.default_rng(14)
    n = 8000
    channels = rng.standard_normal((n, 3))
    beta = np.array([1.0, -2.0, 0.5])
    signal_part = channels @ beta
    noise = rng.standard_normal(n) * float(np.std(signal_part)) * 0.5  # ~20% variance
    reward = signal_part + noise
    dark = idx.dark_reward(reward, channels)
    expected = float(np.var(noise) / np.var(reward))
    assert dark == pytest.approx(expected, abs=0.03)


def test_interp_coverage_split():
    assert idx.interp_coverage([3.0, -1.0], [1.0, -1.0]) == pytest.approx(4.0 / 6.0)
    assert np.isnan(idx.interp_coverage([0.0], [0.0]))


def test_dark_and_interp_observables_emit():
    rng = np.random.default_rng(15)
    channels = rng.standard_normal((200, 2))
    reward = channels @ np.array([1.0, 1.0])
    sig = FakeSignal(caps=Capability.SCORES, scores=reward)
    ev = _run(idx.DarkReward(named_contributions=channels), sig)
    assert ev.value["dark_reward"] == pytest.approx(0.0, abs=1e-6)

    sig2 = FakeSignal(caps=Capability.ACTIVATIONS)
    ev2 = _run(
        idx.InterpCoverage(feature_contributions=[4.0], error_contributions=[1.0]),
        sig2,
    )
    assert ev2.value["interp_coverage"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# A7 receipt reliance + A8 skepticism
# ---------------------------------------------------------------------------


def test_receipt_reliance_ratio():
    assert idx.receipt_reliance(1.2, 3.0) == pytest.approx(0.4)


def test_skepticism_sign():
    assert idx.skepticism(0.8, 0.1) == pytest.approx(0.7)
    batch = idx.skepticism_batch(np.array([1.0, 2.0]), np.array([0.5, 0.5]))
    assert batch["skepticism"] == pytest.approx(1.0)


def test_honesty_observables_emit():
    sig = FakeSignal(caps=Capability.SPAN_TYPES)
    ev = _run(idx.ReceiptReliance(dr_receipt=1.0, dr_total=4.0), sig)
    assert ev.value["receipt_reliance"] == 0.25
    ev2 = _run(
        idx.Skepticism(r_receipt_absent=np.array([1.0]), r_receipt_failure=np.array([0.2])),
        sig,
    )
    assert ev2.value["credulous"] is True


# ---------------------------------------------------------------------------
# A17 eval-awareness
# ---------------------------------------------------------------------------


def test_eval_awareness_separable_vs_random():
    rng = np.random.default_rng(16)
    d, n = 16, 400
    direction = rng.standard_normal(d)
    direction /= np.linalg.norm(direction)
    labels = np.array([0, 1] * (n // 2))
    acts = rng.standard_normal((n, d)) + 5.0 * labels[:, None] * direction[None, :]
    probe = idx.eval_awareness_probe(acts, labels, seed=0)
    assert probe["balanced_accuracy"] > 0.9  # benchmark vs organic is recoverable

    random_labels = rng.integers(0, 2, size=n)
    random_acts = rng.standard_normal((n, d))
    rand_probe = idx.eval_awareness_probe(random_acts, random_labels, seed=0)
    assert rand_probe["balanced_accuracy"] < 0.7  # no real signal


def test_eval_awareness_observable_emits_with_null():
    rng = np.random.default_rng(17)
    d, n = 12, 200
    direction = rng.standard_normal(d)
    labels = np.array([0, 1] * (n // 2))
    acts = rng.standard_normal((n, d)) + 5.0 * labels[:, None] * direction[None, :]
    sig = FakeSignal(caps=Capability.ACTIVATIONS, w_r=np.ones(d), activations=acts)
    ev = _run(idx.EvalAwareness(is_benchmark=labels, null_draws=50, seed=0), sig)
    assert ev.value["balanced_accuracy"] > 0.8
    assert "null_p_value" in ev.value


# ---------------------------------------------------------------------------
# robustness SNR + contested direction
# ---------------------------------------------------------------------------


def test_robustness_snr_high_when_between_dominates():
    # three clusters far apart, tiny within-cluster noise
    rng = np.random.default_rng(18)
    groups = np.repeat([0, 1, 2], 20)
    means = np.repeat([0.0, 10.0, 20.0], 20)
    values = means + rng.standard_normal(60) * 0.1
    report = idx.robustness_snr(values, groups)
    assert report["snr"] > 100


def test_contested_direction_recovers_planted_axis():
    rng = np.random.default_rng(19)
    d, n = 10, 500
    axis = rng.standard_normal(d)
    axis /= np.linalg.norm(axis)
    delta_h = rng.standard_normal((n, d))
    disagreement = delta_h @ axis + rng.standard_normal(n) * 0.1
    result = idx.contested_direction(delta_h, disagreement)
    cos = abs(float(np.dot(result["direction"], axis)))
    assert cos > 0.9
    assert result["correlation"] > 0.8


def test_snr_and_contested_observables_emit():
    rng = np.random.default_rng(20)
    groups = np.repeat([0, 1], 10)
    values = np.repeat([0.0, 5.0], 10) + rng.standard_normal(20) * 0.1
    sig = FakeSignal(caps=Capability.SCORES, scores=values)
    ev = _run(idx.RobustnessSNR(groups=groups), sig)
    assert ev.value["snr"] > 10

    d, n = 6, 100
    axis = rng.standard_normal(d)
    delta_h = rng.standard_normal((n, d))
    disagreement = delta_h @ axis
    sig2 = FakeSignal(caps=Capability.ACTIVATIONS)
    ev2 = _run(idx.Contested(delta_h=delta_h, disagreement=disagreement), sig2)
    assert ev2.gauge is GaugeStatus.COVARIANT
    assert ev2.value["correlation"] > 0.8


# ---------------------------------------------------------------------------
# A14 VCE
# ---------------------------------------------------------------------------


def test_mean_cos2_alignment_closed_form():
    rng = np.random.default_rng(21)
    q, _ = np.linalg.qr(rng.standard_normal((12, 3)))
    assert idx.mean_cos2_alignment(q, q) == pytest.approx(1.0)  # identical subspaces
    q2, _ = np.linalg.qr(rng.standard_normal((12, 3)))
    # two random 3-subspaces of R^12 align near k/d = 0.25
    assert idx.mean_cos2_alignment(q, q2) < 0.6


def test_vce_excess_and_null():
    res = idx.value_convergence_excess(0.85, 0.40, d=16, k=3, null_draws=500, seed=0)
    assert res["vce"] == pytest.approx(0.45)
    assert res["null_mean"] == pytest.approx(3 / 16, abs=0.05)
    assert res["exceeds_identifiability_null"] == 1.0  # reward alignment beats the RUM floor


def test_vce_observable_emits():
    sig = FakeSignal(caps=Capability.SCORES)
    ev = _run(idx.VCE(reward_alignment=0.7, capability_alignment=0.3, d=32, k=4), sig)
    assert ev.value["vce"] == pytest.approx(0.4)
    assert ev.gauge is GaugeStatus.COVARIANT


# ---------------------------------------------------------------------------
# A15 legibility
# ---------------------------------------------------------------------------


def test_legibility_frontier_rises_and_finds_knee():
    rng = np.random.default_rng(22)
    n = 600
    preds = rng.standard_normal((n, 4))
    costs = [1.0, 1.0, 5.0, 5.0]  # the two cheap predicates carry the reward
    reward = 2.0 * preds[:, 0] + 3.0 * preds[:, 1] + rng.standard_normal(n) * 0.01
    report = idx.legibility_frontier(preds, reward, costs)
    assert report["max_fidelity"] > 0.95
    assert report["k_star"] <= 2.0  # the cheap pair suffices
    assert report["tacit_variance_fraction"] < 0.05  # little illegible residual


def test_legibility_observable_emits():
    rng = np.random.default_rng(23)
    n = 300
    preds = rng.standard_normal((n, 3))
    reward = preds @ np.array([1.0, 1.0, 1.0])
    sig = FakeSignal(caps=Capability.SCORES, scores=reward)
    ev = _run(idx.Legibility(predicates=preds, costs=[1.0, 1.0, 1.0]), sig)
    assert ev.value["max_fidelity"] > 0.95


# ---------------------------------------------------------------------------
# Metadata: every Observable declares its catalogue anchor and gauge
# ---------------------------------------------------------------------------


def test_every_observable_declares_faithful_to_and_gauge():
    expected = {
        idx.TeacherCompatibility: "A3",
        idx.Chi: "A12",
        idx.TailIndex: "A4",
        idx.KUI: "A1",
        idx.Coherence: "A9",
        idx.Distortion: "A2",
        idx.CoverageDisparity: None,
        idx.VerificationScore: "A6",
        idx.StyleShare: "A6",
        idx.DarkReward: "A10",
        idx.InterpCoverage: "A11",
        idx.ReceiptReliance: "A7",
        idx.Skepticism: "A8",
        idx.EvalAwareness: "A17",
        idx.RobustnessSNR: None,
        idx.Contested: None,
        idx.VCE: "A14",
        idx.Legibility: "A15",
    }
    for cls, anchor in expected.items():
        assert cls.faithful_to == anchor
        assert isinstance(cls.gauge_status, GaugeStatus)
        assert isinstance(cls.version, str)
        assert isinstance(cls.deviations, tuple)


# ---------------------------------------------------------------------------
# Real-substrate smoke test on the tiny model (the production path)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_signal():
    try:
        from reward_lens.signals.loaders import from_tiny

        return from_tiny(seed=5, conformance_quickcheck=False)
    except Exception as exc:  # pragma: no cover - environment without the tiny model
        pytest.skip(f"tiny model unavailable: {exc}")


def test_teacher_compatibility_runs_on_tiny_model(tiny_signal):
    items = [
        ("What is 2+2?", "The answer is four."),
        ("Name a color.", "Blue is a color."),
        ("Say hi.", "Hello there, friend."),
        ("Define pi.", "Pi is about 3.14159."),
        ("A greeting?", "Good morning to you."),
        ("Capital of France?", "Paris is the capital."),
    ]
    ev = mb.run(idx.TeacherCompatibility(), mb.Context(signal=tiny_signal, view=items))
    assert ev.observable == "TeacherCompatibility"
    assert ev.value["teacher_compatibility"] >= 0.0
    assert ev.value["n_items"] == len(items)
    assert np.isfinite(ev.value["teacher_compatibility"])


def test_chi_runs_on_tiny_model(tiny_signal):
    items = [
        ("Q1", "A short answer."),
        ("Q2", "Another response here."),
        ("Q3", "Yet more text to score."),
        ("Q4", "The fourth completion."),
        ("Q5", "Fifth and final reply."),
    ]
    from reward_lens.measure.indices._support import reward_vector

    w_r = reward_vector(tiny_signal)
    rng = np.random.default_rng(0)
    directions = np.vstack([w_r, rng.standard_normal((3, w_r.size))])
    bank = idx.LinearFeatureBank(directions, names=("wr", "a", "b", "c"))
    ev = mb.run(
        idx.Chi(feature_bank=bank, null_draws=50),
        mb.Context(signal=tiny_signal, view=items),
    )
    assert len(ev.value["chi"]) == 4
    assert all(np.isfinite(c) for c in ev.value["chi"])
    assert ev.value["chi"][0] > 0.0  # the w_r-aligned feature covaries positively with reward

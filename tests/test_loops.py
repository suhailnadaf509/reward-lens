"""M10: the ``loops`` subsystem, proven on CPU.

Optimization, serving, and recording. Every acceptance property is pinned here
and runs on the tiny synthetic model or on pure-numpy synthetic organisms, so the whole file is
CPU-provable; the GPU-scale pieces (real RL rollouts, the vLLM-backed BoN draw, the live training
callbacks) are coded and marked in the modules, not exercised here.

The six load-bearing claims:

- the exact best-of-n KL identity ``KL(bo_n || base) = log(n) - (n-1)/n`` is reproduced exactly;
- the best-of-n expected reward is monotone nondecreasing in ``n``;
- the tilt emulator refuses beyond ``lambda_c / 2`` with a clear error, and its susceptibility
  ``chi_i`` equals the base-policy covariance ``Cov_0(f_i, r)`` on a synthetic feature bank;
- the recorder names a planted drift direction and reports a positive lead time before the gold
  reward diverges;
- the anneal protocol produces a nonzero hysteresis loop area on a synthetic bistable system.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.evidence import Evidence, evidence_from_envelope
from reward_lens.core.types import GaugeStatus
from reward_lens.loops import (
    BoNLadder,
    DriftReport,
    HysteresisLoop,
    RolloutRecorder,
    SusceptibilitySpectrum,
    bon_kl,
    bon_ladder,
    cosine_schedule,
    critical_lambda_from_tail,
    double_well_responder,
    expected_bon_reward,
    flag_hack_modes,
    linear_schedule,
    run_hysteresis,
    susceptibility,
    synthetic_hack_rollout,
    tilt_sweep,
    up_down_schedule,
)
from reward_lens.loops.tilt import ESSGuardError

# ===========================================================================
# bon.py
# ===========================================================================


def test_bon_kl_identity_exact():
    """The acceptance test: ``KL(bo_n || base) = log(n) - (n-1)/n`` reproduced exactly.

    Bit-for-bit, not within a tolerance: the identity is a closed form of ``n`` alone (Beirami et
    al. 2401.01879), and reproducing it exactly is what makes the BoN sweep a calibration-free KL
    axis for the optimization frontier.
    """
    ns = np.array([1, 2, 3, 4, 8, 16, 100, 1000, 10000])
    expected = np.log(ns) - (ns - 1.0) / ns
    got = bon_kl(ns)
    assert np.array_equal(got, expected)
    # best-of-1 is the base policy, so its KL is exactly zero.
    assert bon_kl(1)[()] == 0.0 or float(bon_kl(1)) == 0.0
    # a couple of hand values, so a regression in the formula cannot hide behind the vectorized form.
    assert float(bon_kl(2)) == pytest.approx(np.log(2) - 0.5)
    assert float(bon_kl(4)) == pytest.approx(np.log(4) - 0.75)


def test_bon_kl_rejects_n_below_one():
    with pytest.raises(ValueError):
        bon_kl(0)


def test_bon_ladder_monotone_nondecreasing():
    """The best-of-n expected reward is monotone nondecreasing in ``n`` (the ladder acceptance).

    Keeping the best of more draws cannot lower the expected best, so the plug-in estimator must be
    nondecreasing across the whole ladder, for every prompt and for the across-prompt mean.
    """
    rng = np.random.default_rng(1)
    scores = rng.standard_normal((25, 256))
    ev = bon_ladder(scores, ns=(1, 2, 4, 8, 16, 64, 256, 1024, 4096, 10000))
    reward = ev.value.expected_reward
    assert np.all(np.diff(reward) >= -1e-12)
    # monotone per prompt too, not just in the mean.
    for row in scores:
        per = np.array([expected_bon_reward(row, n) for n in (1, 2, 8, 64, 1000, 10000)])
        assert np.all(np.diff(per) >= -1e-12)


def test_bon_ladder_returns_evidence_and_frontier():
    """The ladder returns ``Evidence[BoNLadder]`` (INVARIANT), with the exact KL and the base mean."""
    rng = np.random.default_rng(2)
    scores = rng.standard_normal((8, 128))
    ev = bon_ladder(scores, ns=(1, 2, 4, 16, 256, 10000))
    assert isinstance(ev, Evidence)
    assert isinstance(ev.value, BoNLadder)
    assert ev.gauge is GaugeStatus.INVARIANT
    ladder = ev.value
    # the KL column is the exact identity, not a fit.
    assert np.allclose(ladder.kl, np.log(ladder.ns) - (ladder.ns - 1.0) / ladder.ns, atol=0)
    # baseline reward at n=1 is the mean base-policy reward.
    assert ladder.baseline_reward == pytest.approx(scores.mean(), abs=1e-6)
    kl, reward = ladder.frontier()
    assert kl.shape == reward.shape == ladder.ns.shape
    # the frontier climbs: more KL buys more reward.
    assert reward[-1] >= reward[0]


def test_bon_expected_reward_matches_monte_carlo():
    """The plug-in expected best-of-n matches a brute-force Monte Carlo of the empirical max.

    A sanity check on the order-statistic formula: for a small bank and modest ``n``, drawing ``n``
    samples with replacement and averaging the max over many trials converges to the closed form.
    """
    rng = np.random.default_rng(3)
    bank = rng.standard_normal(40)
    n = 5
    trials = rng.choice(bank, size=(40000, n), replace=True)
    mc = trials.max(axis=1).mean()
    assert expected_bon_reward(bank, n) == pytest.approx(mc, abs=0.02)


# ===========================================================================
# tilt.py
# ===========================================================================


def _feature_bank(seed: int = 0, n: int = 4000):
    """A synthetic bank with known covariance structure: f0 = r, f1 = -r, f2 = 0.5 r + noise."""
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(n)
    feats = np.stack([r, -r, 0.5 * r + rng.standard_normal(n)], axis=1)
    return r, feats


def test_tilt_susceptibility_matches_base_covariance():
    """``chi_i = Cov_0(f_i, r)`` matches the base-policy covariance exactly (Appendix A12).

    The susceptibility is the population covariance of feature and reward under the base policy, and
    the ``f = r`` diagonal is the teacher variance ``Var_0(r)`` (Appendix A3). Both must equal the
    hand-computed covariance to machine precision, because they are that covariance.
    """
    r, feats = _feature_bank(seed=10)
    ev = susceptibility(r, feats, ["pos", "neg", "mix"])
    assert isinstance(ev.value, SusceptibilitySpectrum)
    assert ev.gauge is GaugeStatus.INVARIANT
    r_c = r - r.mean()
    cov = np.array([((feats[:, i] - feats[:, i].mean()) * r_c).mean() for i in range(3)])
    assert np.allclose(ev.value.chi, cov, atol=1e-12)
    assert ev.value.teacher_variance == pytest.approx(float((r_c * r_c).mean()), abs=1e-12)
    # f0 = r, so its susceptibility is exactly the teacher variance.
    assert ev.value.chi[0] == pytest.approx(ev.value.teacher_variance, abs=1e-12)


def test_tilt_refuses_beyond_half_lambda_c():
    """The tilt refuses ``lambda > lambda_c / 2`` with a clear error naming the ceiling.

    Beyond half the critical pressure the exponential tilt stops emulating practical optimization,
    and past ``lambda_c`` there is no tilted optimum for a heavy tail (Appendix A4/A5). The emulator
    must refuse, not extrapolate.
    """
    r, feats = _feature_bank(seed=11)
    lambda_c = 2.0
    with pytest.raises(ESSGuardError) as excinfo:
        tilt_sweep(r, feats, [1.5], lambda_c=lambda_c)  # 1.5 > lambda_c/2 = 1.0
    msg = str(excinfo.value)
    assert "lambda_c/2" in msg and "1" in msg
    # within the valid range it returns a prediction whose chi is the base covariance.
    ev = tilt_sweep(r, feats, [0.0, 0.25, 0.5], lambda_c=lambda_c)
    r_c = r - r.mean()
    cov = np.array([((feats[:, i] - feats[:, i].mean()) * r_c).mean() for i in range(3)])
    assert np.allclose(ev.value.chi, cov, atol=1e-12)
    assert ev.value.lambda_c == pytest.approx(lambda_c)


def test_tilt_initial_slope_tracks_susceptibility():
    """The SNIS tilted-mean slope at ``lambda -> 0`` recovers ``chi`` (fluctuation-dissipation, A12).

    ``d/dlambda E_lambda[f_i]|_0 = Cov_0(f_i, r)``. A central difference of the emulated tilted means
    across a small symmetric ``lambda`` window recovers the susceptibility the closed form predicts,
    which is the emulator agreeing with its own zeroth-order law.
    """
    r, feats = _feature_bank(seed=12)
    h = 0.05
    ev = tilt_sweep(r, feats, [-h, 0.0, h], lambda_c=10.0)
    fm = ev.value.feature_means
    slope = (fm[2] - fm[0]) / (2 * h)
    assert np.allclose(slope, ev.value.chi, rtol=0.1, atol=1e-2)


def test_tilt_ess_collapse_guard():
    """The second guard fires when the SNIS weights collapse even within ``lambda_c / 2``.

    A single dominating sample gives an effective sample size of one, so a confident tilted estimate
    from it would be a lie. The guard refuses when the ESS fraction falls below the floor.
    """
    scores = np.concatenate([np.zeros(50), [20.0]])
    feats = np.stack([scores, np.random.default_rng(0).standard_normal(51)], axis=1)
    with pytest.raises(ESSGuardError) as excinfo:
        tilt_sweep(scores, feats, [1.0], lambda_c=100.0, min_ess_frac=0.5)
    assert "effective sample size" in str(excinfo.value)


def test_flag_hack_modes_and_critical_lambda():
    """Predicted hack modes are ``chi_i > 0`` with ``Cov_0(f_i, gold) <= 0``; lambda_c from the tail."""
    r, feats = _feature_bank(seed=13)
    spectrum = susceptibility(r, feats, ["hack", "anti", "mix"]).value
    # hack (chi>0) anti-correlated with gold is flagged; anti (chi<0) is not.
    modes = flag_hack_modes(spectrum, gold_covariance=[-0.3, 0.4, 0.1])
    assert modes == ["hack"]
    # an exponential reward tail (rate 1) has critical pressure lambda_c ~ 1.
    lam_c = critical_lambda_from_tail(np.random.default_rng(0).standard_exponential(4000))
    assert lam_c == pytest.approx(1.0, abs=0.25)


# ===========================================================================
# recorder.py
# ===========================================================================


@pytest.mark.parametrize("seed", [0, 2, 4])
def test_recorder_names_direction_and_reports_lead_time(seed):
    """The crown jewel: the recorder names the planted hack direction and leads the gold divergence.

    On a synthetic rollout that drifts along a planted hack feature (proxy loves it, gold does not),
    the recorder must name that feature as exploited and detect its drift onset before the gold
    reward diverges. A positive lead time is the reward-feature signal preceding the behavioral one
    (science S13).
    """
    roll = synthetic_hack_rollout(seed=seed)
    rec = RolloutRecorder(roll.feature_bank, roll.w_r, roll.baseline)
    for t in range(len(roll.activations)):
        rec.observe(roll.activations[t], roll.proxy[t], roll.gold[t])
    report = rec.report(n_perm=400, seed=0)

    # names the exploited direction, and it is the planted one.
    assert report.exploited_direction == roll.planted_direction
    # the exploited feature has the largest CUSUM magnitude of the bank.
    assert report.exploited_index == int(np.argmax(report.dose_cusum))
    # a positive lead time: the dose onset precedes the gold divergence.
    assert report.feature_onset is not None and report.gold_onset is not None
    assert report.lead_time is not None and report.lead_time > 0
    assert report.feature_onset < report.gold_onset
    # both onsets sit in a sane place relative to the plant (dose starts at 6, gold near 14).
    assert report.feature_onset >= roll.dose_onset
    # the alarms carry both the concept-dose onset and the gold divergence.
    kinds = {a.kind for a in report.onset_alarms}
    assert "concept-dose" in kinds and "gold-divergence" in kinds


def test_recorder_tracks_drift_decomposition_and_outliers():
    """The recorder tracks crystallization, the effective-vs-null split, and Mahalanobis outliers.

    As the policy is paid to excite ``w_r`` the crystallization along ``w_r`` grows, the effective
    (reward-subspace) drift dominates the null-space drift because the hack lives in ``w_r`` here,
    and the off-distribution movement shows up as a rising Mahalanobis outlier rate.
    """
    roll = synthetic_hack_rollout(seed=1)
    rec = RolloutRecorder(roll.feature_bank, roll.w_r, roll.baseline)
    for t in range(len(roll.activations)):
        rec.observe(roll.activations[t], roll.proxy[t], roll.gold[t])
    report = rec.report(n_perm=200, seed=0)

    # crystallization along w_r grows from the start of the rollout to the end.
    assert report.crystallization[-1] > report.crystallization[roll.dose_onset]
    # the hack drift is inside the reward subspace, so effective drift dominates the null space.
    assert report.drift_effective[-1] > report.drift_nullspace[-1]
    # the Mahalanobis outlier rate climbs as the policy moves off-distribution.
    assert report.mahalanobis_outlier_rate[-1] > report.mahalanobis_outlier_rate[0]


def test_recorder_evidence_is_raw_only():
    """``RolloutRecorder.evidence`` returns ``Evidence[DriftReport]`` typed RAW_ONLY and round-trips.

    Concept doses and drift magnitudes are projections in one model's activation basis, so they are
    raw coordinates (gate 2). The Evidence must carry that gauge, and its payload must survive a
    store round-trip so a card can render it later.
    """
    roll = synthetic_hack_rollout(seed=5)
    rec = RolloutRecorder(roll.feature_bank, roll.w_r, roll.baseline)
    for t in range(len(roll.activations)):
        rec.observe(roll.activations[t], roll.proxy[t], roll.gold[t])
    ev = rec.evidence(n_perm=100)
    assert isinstance(ev.value, DriftReport)
    assert ev.gauge is GaugeStatus.RAW_ONLY
    back = evidence_from_envelope(ev.envelope())
    assert back.id == ev.id
    assert back.value.exploited_direction == ev.value.exploited_direction


def test_recorder_no_gold_gives_no_lead_time():
    """Without a gold signal the report still names the drift but carries no lead time (honest)."""
    roll = synthetic_hack_rollout(seed=0)
    rec = RolloutRecorder(roll.feature_bank, roll.w_r, roll.baseline)
    for t in range(len(roll.activations)):
        rec.observe(roll.activations[t], roll.proxy[t], gold_reward=None)
    report = rec.report(n_perm=200, seed=0)
    assert report.exploited_direction == roll.planted_direction
    assert report.gold_onset is None
    assert report.lead_time is None


# ===========================================================================
# anneal.py
# ===========================================================================


def test_anneal_bistable_has_nonzero_hysteresis_area():
    """The acceptance test: a synthetic bistable reward produces a nonzero hysteresis loop (S14).

    Sweeping optimization pressure up through onset and back down, the tilted double-well follows its
    local optimum, so the aligned and hacked branches do not coincide and the loop encloses a nonzero
    area. That area is the irreversibility signature: a hacked policy cannot be annealed back
    (science S14).
    """
    up, down = up_down_schedule(-2.5, 2.5, 60)
    responder = double_well_responder(reward_weight=1.0)
    ev = run_hysteresis(responder, up, down, init_state=-1.0)
    assert isinstance(ev.value, HysteresisLoop)
    assert ev.gauge is GaugeStatus.INVARIANT
    assert ev.value.loop_area > 1.0
    assert ev.value.irreversible is True
    # the branches genuinely differ at the same betas (the loop is open in the middle).
    loop = ev.value
    mid = len(loop.beta_up) // 2
    down_at_mid = loop.order_down[len(loop.beta_down) - 1 - mid]
    assert abs(loop.order_up[mid] - down_at_mid) > 0.5


def test_anneal_monostable_loop_closes():
    """A single-valued (crossover) responder retraces its path, so the loop area is ~0 (the control).

    The contrast that makes the bistable result meaningful: a smooth, history-free response gives the
    same order parameter up and down, and a closed loop has no area. A crossover is not a first-order
    transition, and the runner reports it as such.
    """
    up, down = up_down_schedule(-2.5, 2.5, 60)
    responder = lambda beta, m: float(np.tanh(0.5 * beta))  # noqa: E731 - single-valued crossover
    ev = run_hysteresis(responder, up, down, init_state=0.0)
    assert ev.value.loop_area < 1e-6
    assert ev.value.irreversible is False


def test_anneal_schedules():
    """The beta schedules have the right endpoints and shapes."""
    lin = linear_schedule(0.0, 2.0, 11)
    assert lin[0] == 0.0 and lin[-1] == 2.0 and lin.size == 11
    cos = cosine_schedule(0.0, 2.0, 11)
    assert cos[0] == pytest.approx(0.0) and cos[-1] == pytest.approx(2.0)
    up, down = up_down_schedule(-1.0, 1.0, 5)
    assert np.array_equal(down, up[::-1])


# ===========================================================================
# integrations/
# ===========================================================================


@pytest.fixture(scope="module")
def tiny_signal():
    from reward_lens.signals.loaders import from_tiny

    return from_tiny(seed=7, conformance_quickcheck=False)


def test_integration_reward_fn_shape(tiny_signal):
    """``make_reward_fn`` (and the framework aliases) return floats matching ``signal.score``.

    The reward-function shape is the real, framework-agnostic path PPO/GRPO call; it must produce
    exactly the signal's scores as plain floats, regardless of which trainer wraps it.
    """
    from reward_lens.loops.integrations import make_reward_fn
    from reward_lens.loops.integrations.openrlhf import openrlhf_reward_fn
    from reward_lens.loops.integrations.trl import trl_reward_fn
    from reward_lens.loops.integrations.verl import verl_reward_score

    prompts = ["what is 2+2?", "name a fruit", "capital of France?"]
    responses = ["4", "an apple", "Paris"]
    direct = tiny_signal.score(list(zip(prompts, responses))).value.values

    for fn in (make_reward_fn(tiny_signal), trl_reward_fn(tiny_signal)):
        rewards = fn(prompts, responses)
        assert all(isinstance(x, float) for x in rewards)
        assert np.allclose(rewards, direct, atol=1e-5)
    # the OpenRLHF/veRL argument names differ but the values are the same shape.
    assert np.allclose(openrlhf_reward_fn(tiny_signal)(prompts, responses), direct, atol=1e-5)
    assert np.allclose(verl_reward_score(tiny_signal)(prompts, responses), direct, atol=1e-5)


def test_integration_geometry_logger_every_k(tiny_signal):
    """The geometry logger logs the RM's geometry on fixed probes exactly every ``k`` steps.

    Held-fixed probes scored while the policy moves is how the RM's geometry is tracked inside a run.
    The logger must fire on steps 0, k, 2k, ... and each log must carry the probe scores, the
    reward-direction norm, and the concept doses against the feature bank.
    """
    from reward_lens.loops.integrations import GeometryLogger, GeometryProbe
    from reward_lens.loops.recorder import DirectionBank

    d = tiny_signal.meta.d_model
    rng = np.random.default_rng(0)
    bank = DirectionBank(["fa", "fb"], rng.standard_normal((2, d)))
    probe = GeometryProbe(prompts=["p1", "p2"], responses=["r1", "r2"], feature_bank=bank, k=5)
    logger = GeometryLogger(probe)

    logged = [logger.maybe_log(tiny_signal, step) for step in range(11)]
    fired = [i for i, log in enumerate(logged) if log is not None]
    assert fired == [0, 5, 10]
    assert len(logger.logs) == 3
    log = logger.logs[0]
    assert log.probe_scores.shape == (2,)
    assert np.isfinite(log.w_r_norm)
    assert log.concept_dose is not None and log.concept_dose.shape == (2,)


def test_integration_framework_hooks_require_extra(tiny_signal):
    """The framework-specific hooks refuse with ``IntegrationUnavailableError`` when the extra is absent.

    The reward-fn shape runs without any framework, but binding a callback/worker into a live
    trainer needs the framework, and that entrypoint must say so clearly rather than fail
    obscurely (R14).

    Availability is resolved per framework rather than assumed. The first version of this test
    asserted all three hooks raise, which was true only because none of the three extras happened
    to be installed, so it pinned the environment rather than the contract. Installing ``trl``
    behind its declared extra broke it while the code was correct, which is the tell. What the
    contract actually says is a biconditional: absent framework, typed error naming the extra;
    present framework, no such error. Both directions are asserted here, and the message is
    checked rather than only the type, because an ``IntegrationUnavailableError`` that does not
    name the extra leaves the reader in the loop the error exists to break.
    """
    import importlib.util

    from reward_lens.loops.integrations import GeometryProbe, IntegrationUnavailableError
    from reward_lens.loops.integrations.openrlhf import openrlhf_geometry_hook
    from reward_lens.loops.integrations.trl import trl_geometry_callback
    from reward_lens.loops.integrations.verl import verl_geometry_worker

    probe = GeometryProbe(prompts=["p"], responses=["r"], k=10)
    cases = (
        (trl_geometry_callback, "trl", "trl"),
        (openrlhf_geometry_hook, "openrlhf", "openrlhf"),
        (verl_geometry_worker, "verl", "verl"),
    )
    checked = 0
    for hook, module, extra in cases:
        if importlib.util.find_spec(module) is None:
            with pytest.raises(IntegrationUnavailableError) as caught:
                hook(tiny_signal, probe)
            assert extra in str(caught.value), (
                f"{hook.__name__} refused without naming the {extra!r} extra, so the remedy is "
                f"not in the message: {caught.value}"
            )
            checked += 1
        else:
            # Present: it may fail for its own reasons, and it must not claim the extra is missing.
            try:
                hook(tiny_signal, probe)
            except IntegrationUnavailableError as exc:  # pragma: no cover - a real regression
                pytest.fail(
                    f"{module!r} is importable and {hook.__name__} still reported it unavailable: "
                    f"{exc}"
                )
            except Exception:
                pass
            checked += 1
    assert checked == len(cases)

"""``reward_lens.loops`` — optimization, serving, and recording.

This subsystem makes reward-lens a dependency of training pipelines rather than a post-hoc tool. It
is what unblocks the sciences that live in the optimization loop: thermodynamics (S3), hackability
forecasting (S12), the recorder and coupling (S13), and thermodynamic phase structure (S14).

Four analysis modules plus the training-framework integrations:

- ``bon``: best-of-n ladders with the exact ``KL(n) = log(n) - (n-1)/n`` identity, so a BoN sweep is
  the quasi-static equilibrium curve of reward against KL with no RL, the reference arm for
  forecasting and thermodynamics.
- ``tilt``: an SNIS emulator of the tilted family ``pi_lambda ~ pi_0 exp(lambda r)`` with ESS
  guards, and the susceptibility spectrum ``chi_i = Cov_0(f_i, r)`` (the predicted initial drift)
  computed with zero gradient updates.
- ``recorder``: the ``RolloutRecorder`` that watches reward-feature drift, its ``DriftReport``, and
  the changepoint-based ``OnsetAlarm``, proven on a synthetic planted-hack rollout where it names the
  exploited direction and reports a lead time before the gold reward diverges.
- ``anneal``: beta schedules and the hysteresis protocol runner, where a nonzero loop area on a
  bistable system is the irreversibility signature of a first-order hacking transition.

The GPU-scale pieces (real RL rollouts, the vLLM-backed BoN draw, the live training callbacks) are
coded and marked, never faked; the synthetic organisms here are the CPU-provable stand-ins. Import
is torch-free: these modules pull numpy and ``reward_lens.core`` only, and torch enters at call time
through the signal.
"""

from __future__ import annotations

from reward_lens.loops.anneal import (
    HysteresisLoop,
    cosine_schedule,
    double_well_responder,
    linear_schedule,
    run_hysteresis,
    up_down_schedule,
)
from reward_lens.loops.bon import DEFAULT_NS, BoNLadder, bon_kl, bon_ladder, expected_bon_reward
from reward_lens.loops.recorder import (
    Changepoint,
    DirectionBank,
    DriftReport,
    OnsetAlarm,
    RolloutRecorder,
    SyntheticRollout,
    cusum_changepoint,
    synthetic_hack_rollout,
)
from reward_lens.loops.tilt import (
    ESSGuardError,
    SusceptibilitySpectrum,
    TiltPrediction,
    critical_lambda_from_tail,
    flag_hack_modes,
    susceptibility,
    tilt_sweep,
)

__all__ = [
    # bon
    "bon_kl",
    "expected_bon_reward",
    "bon_ladder",
    "BoNLadder",
    "DEFAULT_NS",
    # tilt
    "susceptibility",
    "SusceptibilitySpectrum",
    "flag_hack_modes",
    "critical_lambda_from_tail",
    "tilt_sweep",
    "TiltPrediction",
    "ESSGuardError",
    # recorder
    "RolloutRecorder",
    "DriftReport",
    "OnsetAlarm",
    "DirectionBank",
    "Changepoint",
    "cusum_changepoint",
    "SyntheticRollout",
    "synthetic_hack_rollout",
    # anneal
    "linear_schedule",
    "cosine_schedule",
    "up_down_schedule",
    "double_well_responder",
    "run_hysteresis",
    "HysteresisLoop",
]

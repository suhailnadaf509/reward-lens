"""``reward_lens.loops.integrations`` — training-framework entrypoints.

Callbacks, reward functions, and worker entries that log the reward model's own geometry every ``k``
steps on fixed probes and live rollouts, so reward-lens rides inside a training run rather than
analyzing it afterward. The reward-function shape and the geometry-logging core are framework-
agnostic and run now (on the tiny model in tests); the framework-specific bindings (TRL callback,
OpenRLHF worker hook, veRL worker) need their frameworks, which are optional extras not installed
here, so they raise ``IntegrationUnavailableError`` naming the extra (R14).

Import is light: this package imports numpy and ``reward_lens.core`` only. Torch is pulled at call
time through the signal, and each framework only when its binding is actually built.
"""

from __future__ import annotations

from reward_lens.loops.integrations.base import (
    GeometryLogger,
    GeometryProbe,
    IntegrationUnavailableError,
    ProbeLog,
    RewardFn,
    every_k,
    make_reward_fn,
    probe_geometry,
    require_framework,
)
from reward_lens.loops.integrations.openrlhf import openrlhf_geometry_hook, openrlhf_reward_fn
from reward_lens.loops.integrations.trl import trl_geometry_callback, trl_reward_fn
from reward_lens.loops.integrations.verl import verl_geometry_worker, verl_reward_score

__all__ = [
    # framework-agnostic core
    "IntegrationUnavailableError",
    "require_framework",
    "every_k",
    "make_reward_fn",
    "GeometryProbe",
    "ProbeLog",
    "probe_geometry",
    "GeometryLogger",
    "RewardFn",
    # TRL
    "trl_reward_fn",
    "trl_geometry_callback",
    # OpenRLHF
    "openrlhf_reward_fn",
    "openrlhf_geometry_hook",
    # veRL
    "verl_reward_score",
    "verl_geometry_worker",
]

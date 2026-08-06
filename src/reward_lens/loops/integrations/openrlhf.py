"""OpenRLHF integration: a remote reward function and a geometry-logging worker hook.

OpenRLHF drives PPO with a reward model served as a remote reward function (a callable over a batch
of query/response strings) and runs the actor, critic, reward, and reference models as separate Ray
workers. This module gives the reward-function shape (real, framework-agnostic) and the
geometry-logging core (real, framework-agnostic), and structures the worker entry that binds the
geometry logger into an OpenRLHF PPO trainer, which needs the framework.

The reward function and ``GeometryLogger`` run on the tiny model now. Registering the logger as an
OpenRLHF training callback needs ``openrlhf`` installed, so ``openrlhf_geometry_hook`` raises
``IntegrationUnavailableError`` naming the extra rather than faking the worker wiring (R14).
"""

from __future__ import annotations

from typing import Any, Sequence

from reward_lens.loops.integrations.base import (
    GeometryLogger,
    GeometryProbe,
    make_reward_fn,
    require_framework,
)


def openrlhf_reward_fn(
    signal: Any, readout: str = "reward", *, batch_size: int | None = None
) -> Any:
    """The remote reward function OpenRLHF calls with a batch of query/response strings.

    OpenRLHF passes queries and responses as sequences of strings and expects a list of scalar
    rewards back. Returns ``reward_fn(queries, responses) -> list[float]`` built on the signal; real
    and framework-free. Wrapping is provided so the argument names match OpenRLHF's ``queries`` /
    ``responses`` convention while delegating to the shared shape.
    """
    base = make_reward_fn(signal, readout, batch_size=batch_size)

    def reward_fn(queries: Sequence[str], responses: Sequence[str]) -> list[float]:
        return base(queries, responses)

    return reward_fn


def openrlhf_geometry_hook(signal: Any, probe: GeometryProbe, readout: str = "reward") -> Any:
    """Build the OpenRLHF training hook that logs the RM's geometry every ``probe.k`` steps.

    Wraps a ``GeometryLogger`` and is meant to be registered on the OpenRLHF PPO trainer so it fires
    per optimization step, the framework's place to observe the run without changing training. The
    logging is proven through ``GeometryLogger`` on the tiny model; this function adds the OpenRLHF
    binding and so requires the extra, raising ``IntegrationUnavailableError`` when it is absent. The
    structure below is what runs once ``reward-lens[openrlhf]`` is installed.
    """
    openrlhf = require_framework("openrlhf", "openrlhf")  # raises if absent
    logger = GeometryLogger(probe, readout=readout)

    def on_train_step(step: int, **_: Any) -> Any:  # pragma: no cover - needs openrlhf installed
        return logger.maybe_log(signal, int(step))

    on_train_step.geometry_logger = logger  # type: ignore[attr-defined]
    _ = openrlhf  # the PPOTrainer that registers this hook lives in openrlhf
    return on_train_step


__all__ = ["openrlhf_reward_fn", "openrlhf_geometry_hook"]

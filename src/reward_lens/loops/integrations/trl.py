"""TRL integration: a reward function and a geometry-logging callback.

TRL runs PPO and GRPO with a reward function ``(prompts, responses) -> rewards`` and a list of
``transformers`` ``TrainerCallback`` hooks. This module gives both: the reward function is the real,
framework-agnostic ``make_reward_fn`` shape, and the geometry logging is the framework-agnostic
``GeometryLogger``, so the science of the run (what the RM's geometry does every k steps) is proven
on the tiny model with no TRL installed.

The one piece that genuinely needs TRL is binding the callback into a live ``PPOTrainer`` /
``GRPOTrainer``, because that requires the framework's callback base class and trainer loop. That
entrypoint is structured here and raises ``IntegrationUnavailableError`` naming the extra, rather
than pretending to run. TRL is an optional extra and is not a dependency of the library (R14).
"""

from __future__ import annotations

from typing import Any

from reward_lens.loops.integrations.base import (
    GeometryLogger,
    GeometryProbe,
    RewardFn,
    make_reward_fn,
    require_framework,
)


def trl_reward_fn(
    signal: Any, readout: str = "reward", *, batch_size: int | None = None
) -> RewardFn:
    """The reward function to hand a TRL ``PPOTrainer`` / ``GRPOTrainer``.

    Returns ``reward_fn(prompts, responses) -> list[float]``, the exact shape TRL's reward hook
    calls. This is real and needs only the signal (torch), not TRL: the geometry science is
    independent of which trainer drives the loop.
    """
    return make_reward_fn(signal, readout, batch_size=batch_size)


def trl_geometry_callback(signal: Any, probe: GeometryProbe, readout: str = "reward") -> Any:
    """Build a TRL ``TrainerCallback`` that logs the RM's geometry every ``probe.k`` steps.

    The callback wraps a ``GeometryLogger`` and calls it from ``on_step_end`` with the trainer's
    global step, which is the TRL-native place to observe the run without changing training. The
    logging logic is proven on the tiny model through ``GeometryLogger`` directly; this function only
    adds the framework binding, so it requires TRL and raises ``IntegrationUnavailableError`` when the
    extra is absent. The structure below is what runs once ``reward-lens[trl]`` is installed.
    """
    trl = require_framework("trl", "trl")  # raises IntegrationUnavailableError if absent
    from transformers import TrainerCallback  # available with trl; imported lazily

    logger = GeometryLogger(probe, readout=readout)

    class _GeometryLoggingCallback(TrainerCallback):  # pragma: no cover - needs trl installed
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            log = logger.maybe_log(signal, int(state.global_step))
            if log is not None and getattr(args, "report_to", None):
                # Surface the probe scores / crystallization proxy to the trainer's loggers.
                self._report(state, log)
            return control

        @staticmethod
        def _report(state: Any, log: Any) -> None:
            metrics = {
                "rm_geometry/probe_score_mean": float(log.probe_scores.mean()),
                "rm_geometry/w_r_norm": float(log.w_r_norm),
            }
            if getattr(state, "log_history", None) is not None:
                state.log_history.append({"step": log.step, **metrics})

    callback = _GeometryLoggingCallback()
    callback.geometry_logger = logger  # type: ignore[attr-defined]
    _ = trl  # the trainer that consumes this callback lives in trl
    return callback


__all__ = ["trl_reward_fn", "trl_geometry_callback"]

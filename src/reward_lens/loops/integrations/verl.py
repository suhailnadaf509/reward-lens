"""veRL integration: a reward manager and a geometry-logging worker entry.

veRL computes rewards through a reward manager that consumes a batch (a ``DataProto`` of token ids
and metadata) and returns a reward tensor, and runs rollout/actor/critic as workers under its
single-controller design. This module gives the reward-scoring shape (real, framework-agnostic, over
decoded prompt/response strings) and the geometry-logging core (real), and structures the worker
entry that logs the RM's geometry every k steps inside a veRL run, which needs the framework.

The scoring shape and ``GeometryLogger`` run on the tiny model now. Binding a reward manager and a
worker into veRL needs ``verl`` installed, so ``verl_geometry_worker`` raises
``IntegrationUnavailableError`` naming the extra rather than faking the worker registration (R14).
"""

from __future__ import annotations

from typing import Any, Sequence

from reward_lens.loops.integrations.base import (
    GeometryLogger,
    GeometryProbe,
    make_reward_fn,
    require_framework,
)


def verl_reward_score(
    signal: Any, readout: str = "reward", *, batch_size: int | None = None
) -> Any:
    """The scoring callable a veRL reward manager wraps.

    veRL's reward manager ultimately needs a scalar per (prompt, response). Returns
    ``score(prompts, responses) -> list[float]`` built on the signal; real and framework-free. The
    veRL-specific manager tokenizes/decodes the ``DataProto`` around this call, which is why the
    string-level scoring is the reusable core and the ``DataProto`` adapter is the framework binding.
    """
    base = make_reward_fn(signal, readout, batch_size=batch_size)

    def score(prompts: Sequence[str], responses: Sequence[str]) -> list[float]:
        return base(prompts, responses)

    return score


def verl_geometry_worker(signal: Any, probe: GeometryProbe, readout: str = "reward") -> Any:
    """Build the veRL worker entry that logs the RM's geometry every ``probe.k`` steps.

    Wraps a ``GeometryLogger`` to be driven from veRL's training loop (the single-controller step),
    the framework's place to observe the run without changing training. The logging is proven through
    ``GeometryLogger`` on the tiny model; this function adds the veRL binding and so requires the
    extra, raising ``IntegrationUnavailableError`` when it is absent. The structure below is what runs
    once ``reward-lens[verl]`` is installed.
    """
    verl = require_framework("verl", "verl")  # raises if absent
    logger = GeometryLogger(probe, readout=readout)

    def on_step(step: int, batch: Any = None, **_: Any) -> Any:  # pragma: no cover - needs verl
        return logger.maybe_log(signal, int(step))

    on_step.geometry_logger = logger  # type: ignore[attr-defined]
    _ = verl  # the single-controller trainer that drives this worker lives in verl
    return on_step


__all__ = ["verl_reward_score", "verl_geometry_worker"]

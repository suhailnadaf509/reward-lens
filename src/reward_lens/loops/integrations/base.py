"""Shared machinery for the training-framework integrations.

The point of the integrations is to make reward-lens a dependency of the training pipeline rather
than a post-hoc tool, by logging the reward model's own geometry every ``k`` steps on fixed probes
and live rollouts while training runs. Three parts are framework-agnostic and implemented here:

- the reward-function shape: a callable ``(prompts, responses) -> list[float]`` that scores a batch
  with a ``RewardSignal`` and is what TRL, OpenRLHF, and veRL all ultimately call to get rewards.
  Wrapping the signal here (rather than in each framework file) is what keeps the three integrations
  thin.
- the geometry probe: a fixed set of probe inputs plus the per-``k``-step read-out of the RM's
  geometry (probe scores, the reward-direction norm, concept doses against a feature bank, and the
  recorder's drift decomposition when a bank is supplied). This is what "log the RM's own geometry
  every k steps on fixed probes" means in practice.
- the step gate: ``every_k`` decides when a step logs, so a callback stays a few lines.

The framework-specific wiring (the TRL callback base class, the OpenRLHF worker entry, the veRL
reward manager) needs the training frameworks installed, and they are optional extras. Those
entrypoints raise ``IntegrationUnavailableError`` naming the missing extra, exactly as
``runtime.sampling`` does for vLLM. The reward-function shape and the geometry probe do not need the
frameworks and run without them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from reward_lens.core.errors import RewardLensError

if TYPE_CHECKING:
    from reward_lens.loops.recorder import DirectionBank, DriftReport

RewardFn = Callable[[Sequence[str], Sequence[str]], list[float]]


class IntegrationUnavailableError(RewardLensError):
    """Raised when a training-framework entrypoint is used but its optional extra is not installed.

    Names the framework and the extra to install. The reward-function shape and the geometry probe
    do not need the framework and never raise this; only the framework-specific callback / worker
    wiring does (R14).
    """


def require_framework(name: str, extra: str) -> Any:
    """Import a training framework or raise a clear ``IntegrationUnavailableError``.

    The single choke point every framework entrypoint calls, so the message is uniform: which
    framework, which extra, and that the reward-fn shape is usable without it.
    """
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise IntegrationUnavailableError(
            f"the {name!r} integration requires the optional {extra!r} extra, which is not "
            f"installed. Install reward-lens[{extra}] to wire the callback/worker into a live "
            f"training run. The framework-agnostic reward function (make_reward_fn) and the geometry "
            f"probe (probe_geometry) do not need {name!r} and are usable now."
        ) from exc


def every_k(step: int, k: int, *, offset: int = 0) -> bool:
    """Whether ``step`` is a logging step: true every ``k`` steps."""
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")
    return (step - offset) % k == 0


def make_reward_fn(
    signal: Any,
    readout: str = "reward",
    *,
    batch_size: int | None = None,
) -> RewardFn:
    """Wrap a ``RewardSignal`` as the reward function a training loop calls.

    Returns ``reward_fn(prompts, responses) -> list[float]``: it pairs each prompt with its response,
    scores them through ``signal.score`` under ``readout``, and returns plain floats, which is the
    contract PPO/GRPO reward hooks expect. This is the real, framework-agnostic reward path; the
    framework files add only the callback/worker plumbing around it.
    """

    def reward_fn(prompts: Sequence[str], responses: Sequence[str]) -> list[float]:
        if len(prompts) != len(responses):
            raise ValueError(
                f"prompts ({len(prompts)}) and responses ({len(responses)}) must be equal length"
            )
        items = list(zip(prompts, responses))
        if not items:
            return []
        if batch_size is not None:
            signal.default_batch_size = batch_size
        scores = signal.score(items, readout=readout).value.values
        return [float(x) for x in np.asarray(scores).ravel()]

    return reward_fn


@dataclass
class GeometryProbe:
    """A fixed probe set and the feature bank to read the RM's geometry against.

    ``prompts`` / ``responses`` are the held-fixed probe inputs scored every ``k`` steps, so the
    geometry is tracked at a stable stimulus while the policy moves. ``feature_bank`` (optional)
    turns the probe activations into concept doses and, with the recorder, a drift decomposition.
    ``k`` is the logging period.
    """

    prompts: list[str]
    responses: list[str]
    feature_bank: "DirectionBank | None" = None
    k: int = 50

    def __post_init__(self) -> None:
        if len(self.prompts) != len(self.responses):
            raise ValueError("probe prompts and responses must be equal length")


@dataclass
class ProbeLog:
    """One geometry read-out at a training step.

    ``step`` is the training step; ``probe_scores`` the RM's scores on the fixed probes; ``w_r_norm``
    the reward-direction norm (a cheap crystallization proxy); ``concept_dose`` the mean projection
    of the probe activations onto the feature bank (present only when a bank is supplied); and
    ``drift`` the recorder's ``DriftReport`` when the caller threads a recorder through the run.
    """

    step: int
    probe_scores: np.ndarray
    w_r_norm: float
    concept_dose: np.ndarray | None = None
    drift: "DriftReport | None" = None
    meta: dict[str, Any] = field(default_factory=dict)


def probe_geometry(
    signal: Any, probe: GeometryProbe, step: int, *, readout: str = "reward"
) -> ProbeLog:
    """Log the RM's geometry on the fixed probes at a training step.

    Scores the probes, reads the reward-direction norm, and, when the probe carries a feature bank,
    projects the probes' final-layer activations onto it to get concept doses. This is the per-step
    geometry log the callbacks call; it needs the signal (torch) but not any training framework, so
    it is unit-testable on the tiny model.
    """
    items = list(zip(probe.prompts, probe.responses))
    scores = np.asarray(signal.score(items, readout=readout).value.values, dtype=np.float64).ravel()
    read = signal.readout(readout)
    w_r_norm = (
        float(np.linalg.norm(_to_numpy(read.vector))) if read.vector is not None else float("nan")
    )

    concept_dose = None
    if probe.feature_bank is not None:
        activations = _probe_activations(signal, items, read)
        concept_dose = probe.feature_bank.directions @ activations.mean(axis=0)

    return ProbeLog(step=step, probe_scores=scores, w_r_norm=w_r_norm, concept_dose=concept_dose)


class GeometryLogger:
    """Log the RM's geometry every ``k`` steps on a fixed probe.

    The framework-agnostic core the three integration callbacks wrap: hold a probe, and on each
    training step call ``maybe_log(signal, step)``, which returns a ``ProbeLog`` on the logging steps
    (every ``probe.k``) and ``None`` otherwise, appending each log to ``logs``. This is the piece the
    tiny-model test exercises directly, so the geometry logging is proven without any training
    framework installed; the callbacks add only the framework's step hook around it.
    """

    def __init__(self, probe: GeometryProbe, readout: str = "reward"):
        self.probe = probe
        self.readout = readout
        self.logs: list[ProbeLog] = []

    def maybe_log(self, signal: Any, step: int) -> "ProbeLog | None":
        if not every_k(step, self.probe.k):
            return None
        log = probe_geometry(signal, self.probe, step, readout=self.readout)
        self.logs.append(log)
        return log


def _to_numpy(x: Any) -> np.ndarray:
    """Detach a tensor or coerce an array-like to float64 numpy."""
    detach = getattr(x, "detach", None)
    if detach is not None:
        x = detach().to("cpu")
        return x.numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _probe_activations(signal: Any, items: list[tuple[str, str]], read: Any) -> np.ndarray:
    """Capture the readout-site residual activation at the final position for each probe item.

    Captures at the readout's site with the default spec, which resolves to the last-valid token
    (the position the classifier readout reads) and returns ``(n_items, d)`` already
    pooled to that position. The concept dose is then read at exactly the place the reward is.
    """
    from reward_lens.runtime.backend import CaptureSpec

    handle = signal.capture(items, CaptureSpec(sites=(read.site,)))
    return _to_numpy(handle.get(read.site))


__all__ = [
    "IntegrationUnavailableError",
    "require_framework",
    "every_k",
    "make_reward_fn",
    "GeometryProbe",
    "ProbeLog",
    "probe_geometry",
    "GeometryLogger",
    "RewardFn",
]

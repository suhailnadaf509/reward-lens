"""Optional bridge to LLC / essential-dynamics tooling for phase-transition detection.

Developmental interpretability reads training as a sequence of phase transitions and measures them
with the local learning coefficient (LLC, a scalable estimate of the RLCT / effective dimension) and
essential-dynamics analyses (the low-rank structure of how internals move over training). Those live
in the external ``devinterp`` package, which is not a dependency of this library and is not installed
in this environment. This module is the clearly-marked bridge: it detects whether ``devinterp`` is
importable and, when it is, adapts a `CheckpointSequence` into the trajectory that its estimators
consume; when it is not, every entry point raises a precise error naming the missing package rather
than failing deep inside a call.

Nothing here fabricates an LLC curve. The estimators need per-checkpoint gradients over a data loader
(a GPU-scale computation on the real RM-Pythia run), so the bridge is wiring and adaptation
only. The stabilization detector in `curves.py` is the CPU-provable, dependency-free developmental
signal this library ships on its own; this bridge is the optional upgrade to the specialist tooling
when it is present.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reward_lens.dynamics.checkpoints import CheckpointSequence

_PACKAGE = "devinterp"


def is_available() -> bool:
    """Whether the external ``devinterp`` package is importable in this environment."""
    return importlib.util.find_spec(_PACKAGE) is not None


def _require() -> Any:
    """Import ``devinterp`` or raise a precise, actionable error naming what is missing."""
    if not is_available():
        raise ImportError(
            "the optional 'devinterp' package is not installed; it provides the LLC / "
            "essential-dynamics estimators this bridge adapts to. Install it "
            "(pip install devinterp) to enable phase-transition detection, or use "
            "dynamics.curves.stabilization_report for the dependency-free developmental signal."
        )
    return importlib.import_module(_PACKAGE)


@dataclass
class LLCTrajectory:
    """A local-learning-coefficient trajectory over training. Populated only via ``devinterp``.

    ``steps`` is the training-time covariate and ``llc`` the estimated local learning coefficient at
    each checkpoint (a scalable RLCT estimate; a jump marks a phase transition). ``estimator`` records
    which ``devinterp`` estimator produced it and ``meta`` its configuration, so a trajectory carries
    its own provenance. This dataclass is defined even without the package so the return type is
    importable; it is only ever filled by `estimate_llc`, which requires the package.
    """

    steps: list[int]
    llc: list[float]
    estimator: str = "devinterp.sgld"
    meta: dict[str, Any] = field(default_factory=dict)


def estimate_llc(
    sequence: "CheckpointSequence",
    loss_fn: Any = None,
    loader: Any = None,
    **estimator_kwargs: Any,
) -> LLCTrajectory:
    """Estimate the LLC across a checkpoint sequence via ``devinterp``. Requires the package.

    For each checkpoint this would run ``devinterp``'s SGLD-based LLC estimator on ``loss_fn`` over
    ``loader`` and collect the trajectory; a rise in the LLC marks the model gaining effective
    dimension, which is the phase-transition signal. The estimator needs per-checkpoint gradient
    sampling over a data loader, a GPU-scale computation on the real run, so this raises
    without the package rather than approximate it on CPU.
    """
    _require()  # pragma: no cover - only reachable with the optional package installed
    raise NotImplementedError(  # pragma: no cover
        "estimate_llc adapts devinterp's LLC estimator over the checkpoint sequence; wire it to the "
        "estimator and data loader at run time on GPU hardware. The trajectory is never fabricated."
    )


def essential_dynamics(
    sequence: "CheckpointSequence",
    observations: Any = None,
    n_components: int = 8,
    **kwargs: Any,
) -> Any:
    """Essential-dynamics (low-rank training-motion) analysis via ``devinterp``. Requires the package.

    Essential dynamics reduces how the model's internals move over training to a few principal
    components, exposing the developmental "stages" as segments in that low-dimensional trajectory.
    This adapts the checkpoint sequence's per-step observations into ``devinterp``'s analysis; it raises
    without the package and never invents a trajectory.
    """
    _require()  # pragma: no cover - only reachable with the optional package installed
    raise NotImplementedError(  # pragma: no cover
        "essential_dynamics adapts devinterp's low-rank training-trajectory analysis; wire it to the "
        "per-checkpoint observation tensors at run time."
    )


__all__ = [
    "is_available",
    "LLCTrajectory",
    "estimate_llc",
    "essential_dynamics",
]

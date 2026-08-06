"""``reward_lens.signals`` — the ``RewardSignal`` protocol and its adapters.

This subsystem is the answer to substrate lock-in: it generalizes v1's single-scalar
``RewardModel`` into a protocol every reward substrate implements, parameterized by first-class
``Readout`` and ``PositionSpec`` objects. The first adapter is ``ClassifierRM`` (the rebuilt
sequence-classification reward model), alongside the loaders that construct it and the conformance
suite every adapter must pass.

``base`` is the frozen protocol surface and imports torch only under ``TYPE_CHECKING``, so importing
the types is cheap. The concrete adapters and loaders require torch; they are re-exported here for
convenience and imported lazily so ``import reward_lens.signals`` stays light until a signal is
actually built.

Beyond ``ClassifierRM``, the seven remaining adapters live here too: the generative judge, the
process (step-level) RM, the implicit (DPO log-ratio) RM, the rubric grader,
the trajectory RM, the gated dense-reward extractor, and the ensemble / distributional composites.
Each implements the same ``RewardSignal`` protocol and clears the per-adapter conformance suite
(``run_adapter_conformance``) before it is trusted.
"""

from __future__ import annotations

from reward_lens.core.extras import require_extra

require_extra("white-box", subsystem="reward_lens.signals")

from reward_lens.signals.base import (
    PositionSpec,
    Readout,
    RewardSignal,
    Scores,
    SignalMeta,
    TokenCurves,
    TokenizedInput,
)

__all__ = [
    # frozen protocol surface
    "RewardSignal",
    "Readout",
    "PositionSpec",
    "SignalMeta",
    "Scores",
    "TokenCurves",
    "TokenizedInput",
    # adapters + loaders (lazy, torch-backed)
    "ClassifierRM",
    "load_signal",
    "wrap_hf_model",
    "from_tiny",
    "SignalSpec",
    "run_conformance",
    "ConformanceReport",
    # the seven remaining adapters
    "GenerativeJudge",
    "ProcessRM",
    "StepScores",
    "ImplicitRM",
    "RubricRM",
    "RubricSpec",
    "TrajectoryRM",
    "DenseRewardExtractor",
    "SignalEnsemble",
    "DistributionalSignal",
    "run_adapter_conformance",
]

# Lazy dispatch table: attribute name -> submodule that defines it. PEP 562 keeps
# ``import reward_lens.signals`` torch-free until one of these is actually touched.
_LAZY: dict[str, str] = {
    "ClassifierRM": "classifier",
    "build_readouts": "classifier",
    "load_signal": "loaders",
    "wrap_hf_model": "loaders",
    "from_tiny": "loaders",
    "SignalSpec": "loaders",
    "run_conformance": "conformance",
    "ConformanceReport": "conformance",
    "ConformanceCheck": "conformance",
    "run_adapter_conformance": "conformance_adapters",
    "GenerativeJudge": "judge",
    "ProcessRM": "process",
    "StepScores": "process",
    "ImplicitRM": "implicit",
    "RubricRM": "rubric",
    "RubricSpec": "rubric",
    "TrajectoryRM": "trajectory",
    "DenseRewardExtractor": "dense",
    "SignalEnsemble": "ensemble",
    "DistributionalSignal": "ensemble",
}


def __getattr__(name: str):  # PEP 562: lazy access to the torch-backed pieces
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module 'reward_lens.signals' has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"reward_lens.signals.{module_name}")
    return getattr(module, name)

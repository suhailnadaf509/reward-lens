"""``reward_lens.organisms`` — the ground-truth foundry and the calibration gate.

An organism is a reward model with a decision rule planted *by construction*, so its ground truth is
known exactly: the training data was generated from the rule, and calibrating an instrument means
measuring whether it recovers the planted structure. This subsystem is the epistemological floor of
the whole design (I2, gate 1) and the true go/no-go for the scientific ambition:
without organisms that separate methods, every instrument is merely suggestive and nothing is
provable.

The layout:

- `spec` (`Predicate`, `RuleSpec`, `PlantedChannel`, `AnswerKey`): the declarative planted rule.
- `foundry`: a generator for every planted structure, each emitting lineage-complete data plus an
  `AnswerKey`. Pure (no torch), fully buildable and testable now.
- `scorecard` (`MethodScorecard`): the answer-key ROC of an instrument recovering a planted structure,
  monotone in the dose. Pure-testable with a synthetic detector.
- `game`: the minimal auditing-game harness (a blind operator must name the planted rule).
- `train` / `verify` / `micro`: the torch-gated path that plants a rule in a tiny CPU trunk, verifies
  the rule governs behaviour OOD, and recovers it with a built-in linear detector (the CI hook).
- `zoo`: HF release tooling for RewardBench-GT (a marked stub, deferred).

**Import is torch-free.** The pure objects (`spec`, `foundry`, `scorecard`, `game`) import eagerly.
The torch-gated objects (`train`, `verify`, `micro`, the tiny trunk) are exposed through a lazy
``__getattr__`` so ``import reward_lens.organisms`` never imports torch; torch is loaded only when a
training, verification, or micro-calibration symbol is first accessed.
"""

from __future__ import annotations

from reward_lens.core.extras import require_extra

require_extra("white-box", subsystem="reward_lens.organisms")

from typing import TYPE_CHECKING, Any

# -- Pure (torch-free) surface, imported eagerly -----------------------------
from reward_lens.organisms.foundry import (
    annotator_mixture_organism,
    compositional_rule_organism,
    curl_harmonic_organism,
    empirical_annotator_entropy,
    epistemic_error_organism,
    gate_organism,
    hack_direction_organism,
    has_cycle,
    hidden_objective_organism,
    intransitivity_organism,
    kinship_organism,
    make_rubric_directions,
    measure_hack_signature,
    measure_spurious_correlation,
    mixture_entropy_bits,
    rubric_organism,
    spurious_correlation_organism,
    value_error_organism,
)
from reward_lens.organisms.game import GameResult, feature_gap_operator, run_auditing_game
from reward_lens.organisms.scorecard import (
    DetectorReadout,
    MethodScorecard,
    ScorecardEntry,
    ScorecardSummary,
    synthetic_dose_detector,
)
from reward_lens.organisms.spec import AnswerKey, ChannelKind, PlantedChannel, Predicate, RuleSpec
from reward_lens.organisms.zoo import export_rewardbench_gt

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps the torch-gated names known to checkers
    from reward_lens.organisms.micro import (
        DetectorResult,
        MicroCalibrationResult,
        mean_difference_detector,
        micro_organism_calibration,
    )
    from reward_lens.organisms.train import TrainedOrganism, TrainRecipe, train_organism
    from reward_lens.organisms.verify import VerifyResult, verify_organism

# Lazy torch-gated symbols: name -> submodule that defines it. Accessing any of these triggers the
# import of its submodule (and thus torch) on first use only (torch-free import).
_LAZY: dict[str, str] = {
    "train_organism": "reward_lens.organisms.train",
    "TrainRecipe": "reward_lens.organisms.train",
    "TrainedOrganism": "reward_lens.organisms.train",
    "verify_organism": "reward_lens.organisms.verify",
    "VerifyResult": "reward_lens.organisms.verify",
    "micro_organism_calibration": "reward_lens.organisms.micro",
    "mean_difference_detector": "reward_lens.organisms.micro",
    "DetectorResult": "reward_lens.organisms.micro",
    "MicroCalibrationResult": "reward_lens.organisms.micro",
    "make_micro_trunk": "reward_lens.organisms._tiny",
}


def __getattr__(name: str) -> Any:
    """Lazily import a torch-gated symbol so the package import stays torch-free."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module 'reward_lens.organisms' has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, name)


__all__ = [
    # spec
    "Predicate",
    "RuleSpec",
    "PlantedChannel",
    "ChannelKind",
    "AnswerKey",
    # foundry
    "compositional_rule_organism",
    "spurious_correlation_organism",
    "measure_spurious_correlation",
    "hidden_objective_organism",
    "gate_organism",
    "intransitivity_organism",
    "has_cycle",
    "annotator_mixture_organism",
    "mixture_entropy_bits",
    "empirical_annotator_entropy",
    "rubric_organism",
    "make_rubric_directions",
    "hack_direction_organism",
    "measure_hack_signature",
    "epistemic_error_organism",
    "value_error_organism",
    "curl_harmonic_organism",
    "kinship_organism",
    # scorecard
    "MethodScorecard",
    "ScorecardEntry",
    "ScorecardSummary",
    "DetectorReadout",
    "synthetic_dose_detector",
    # game
    "GameResult",
    "feature_gap_operator",
    "run_auditing_game",
    # zoo (stub)
    "export_rewardbench_gt",
    # torch-gated (lazy)
    "train_organism",
    "TrainRecipe",
    "TrainedOrganism",
    "verify_organism",
    "VerifyResult",
    "micro_organism_calibration",
    "mean_difference_detector",
    "DetectorResult",
    "MicroCalibrationResult",
    "make_micro_trunk",
]

"""W6.3 / K1, the distillation gap: what survives the step that produces the model that ships.

Kill risk number one for the in-run layer of this library. Multi-teacher on-policy distillation is
the default final step at seven or more labs, the shipped model is never the RL policy, and not one
of them publishes a behavioural audit of that step. If group-relative geometry does not survive it,
everything measured during a run is measured on a model nobody deploys.

Nothing here has been run against a real pair of checkpoints and none of it ever will be by the
people who wrote it. What the package ships is the measurement, frozen predictions made before the
subject exists, an acceptance test on a planted organism, a runbook and a price:

- `instrument.DistillationGap` is the `Instrument`, `artifact.distillation_delta` in percentage
  points.
- `survival` and `fit` are its arithmetic, including the errors-in-variables correction without
  which the reported survival depends on how many completions per prompt somebody drew.
- `regions` splits a response into its turn-entry window and its body, which is the cheap half of
  the localisation question the one published audit of this step leaves open.
- `planted` builds the subject the acceptance test recovers a known survival fraction from.
- `analysis` is the frozen study: five registered predictions and three kill criteria.
- `price` is the bill with its assumptions written out, and `runbook` prints what to type.
"""

from studies.w6_distillation.fit import (
    ShiftDesign,
    SurvivalContrast,
    SurvivalFit,
    corrected_slope,
    per_feature_survival,
    raw_slope,
    reliability,
    survival_contrast,
    survival_fit,
)
from studies.w6_distillation.instrument import DISTILLATION_ENVELOPE, DistillationGap
from studies.w6_distillation.reading import DistillationSurvival, distillation_survival
from studies.w6_distillation.regions import RegionFeatures, region_of
from studies.w6_distillation.survival import Arm, ArmSummary, DetectionFloor, shared_prompts

__all__ = [
    "DISTILLATION_ENVELOPE",
    "Arm",
    "ArmSummary",
    "DetectionFloor",
    "DistillationGap",
    "DistillationSurvival",
    "RegionFeatures",
    "ShiftDesign",
    "SurvivalContrast",
    "SurvivalFit",
    "corrected_slope",
    "distillation_survival",
    "per_feature_survival",
    "raw_slope",
    "region_of",
    "reliability",
    "shared_prompts",
    "survival_contrast",
    "survival_fit",
]

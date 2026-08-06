"""``reward_lens.concepts`` — directions, probes, dictionaries, and beliefs.

The concept subsystem turns activations into named, persisted directions and the calibrated probes
that read them. Five layers sit here:

  - ``vectors``: the canonical, substrate-free concept-direction estimator (mean-difference) that the
    v3 battery shares with the v1 primitive.
  - ``probes``: the probe factory ``fit_probe``, with seed-level cross-validation and automatic
    scorecard binding, producing persisted ``Direction`` artifacts. A direction without a calibration
    can be used but taints downstream Evidence to EXPLORATORY.
  - ``beliefs``: answer-keyed belief probes over externally verifiable latents, held to the strictest
    calibration standard and refusing a self-labeled target.
  - ``diff_dict``: preference-difference dictionaries on ``delta_h = h_chosen - h_rejected``, whose
    Bradley-Terry margin decomposes exactly and whose decomposition is verified and stored as
    Evidence.
  - ``banks``: the standard concept batteries (style, safety, quality, belief targets).

The v1 ``ConceptExtractor`` primitive that used to sit here as the reference retired with the
rest of the v1 corpus; ``vectors`` has been the canonical direction estimator since 2.0.
"""

from __future__ import annotations

from reward_lens.core.extras import require_extra

require_extra("white-box", subsystem="reward_lens.concepts")

from reward_lens.concepts.banks import (
    BELIEF_BANK,
    QUALITY_BANK,
    SAFETY_BANK,
    STANDARD_BANKS,
    STYLE_BANK,
    BankManifest,
    BuiltBank,
    ConceptBank,
    ConceptSides,
    ConceptSpec,
    all_specs,
    bank,
    build_bank,
    capture_concept_sides,
    default_feature_bank,
)
from reward_lens.concepts.beliefs import (
    BeliefProbe,
    BeliefTarget,
    SelfLabeledBeliefError,
    UncalibratedBeliefError,
    answer_key_target,
    fit_belief_probe,
    meta_key_target,
    self_labeled_target,
)
from reward_lens.concepts.diff_dict import (
    DiffDictArtifact,
    DiffDictionary,
    DiffDictResult,
    DiffDictVerification,
    activations,
    decompose_margin,
    delta_h_from_pairs,
    diff_dict_evidence,
    fit_and_verify,
    reconstruct,
    train_diff_dict,
    true_margin,
    verification_evidence,
    verify_decomposition,
)
from reward_lens.concepts.probes import (
    Direction,
    DirectionArtifact,
    ProbeFit,
    SiteCaptures,
    SiteProbe,
    capture_probe_inputs,
    direction_evidence,
    feature_target,
    fit_probe,
    group_kfold_indices,
    make_direction,
    probe_scores,
)
from reward_lens.concepts.vectors import (
    concept_direction,
    dose_response_slope,
    reward_alignment,
)

__all__ = [
    # canonical direction estimator (vectors)
    "concept_direction",
    "reward_alignment",
    "dose_response_slope",
    # probe factory (probes)
    "fit_probe",
    "probe_scores",
    "Direction",
    "DirectionArtifact",
    "ProbeFit",
    "SiteProbe",
    "SiteCaptures",
    "make_direction",
    "direction_evidence",
    "group_kfold_indices",
    "capture_probe_inputs",
    "feature_target",
    # belief probes (beliefs)
    "fit_belief_probe",
    "BeliefProbe",
    "BeliefTarget",
    "answer_key_target",
    "self_labeled_target",
    "meta_key_target",
    "SelfLabeledBeliefError",
    "UncalibratedBeliefError",
    # preference-difference dictionaries (diff_dict)
    "train_diff_dict",
    "fit_and_verify",
    "verify_decomposition",
    "decompose_margin",
    "DiffDictionary",
    "DiffDictArtifact",
    "DiffDictVerification",
    "DiffDictResult",
    "delta_h_from_pairs",
    "true_margin",
    "activations",
    "reconstruct",
    "diff_dict_evidence",
    "verification_evidence",
    # standard banks (banks)
    "STANDARD_BANKS",
    "STYLE_BANK",
    "SAFETY_BANK",
    "QUALITY_BANK",
    "BELIEF_BANK",
    "ConceptSpec",
    "ConceptBank",
    "BankManifest",
    "ConceptSides",
    "BuiltBank",
    "build_bank",
    "bank",
    "all_specs",
    "capture_concept_sides",
    "default_feature_bank",
    # v1 primitive (reference)
]

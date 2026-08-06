"""``reward_lens.geometry`` — frames, gauge, curvature, and the scalar bottleneck.

The mathematical heart of cross-model validity. This subsystem turns three of the corpus's deepest
programs into shared machinery: field theory (the reward Hessian spectrum), gauge (frames and
canonicalization), and capacity (the effective dimension). Its job is to make invariant I3
executable, so that a cross-signal reward comparison measures a functional change rather than a
coordinate change.

Five modules:

  - ``frame``: the `Frame` whitening artifact and `fit_frame` (Ledoit-Wolf shrinkage covariance,
    symmetric roots, null-subspace estimation), the per-(site, corpus) object that fixes the gauge.
  - ``canonical``: `canonicalize` (``Sigma^{1/2} w`` normalized) and `effective_angle` (the canonical
    cosine with a bootstrap CI and a STARC-style behavioural regret bound), COVARIANT and
    frame-gated.
  - ``hessian``: `hessian_spectrum` and `flat_subspace` through HVPs, `participation_ratio` for the
    effective dimension, and the Lanczos / SLQ numerics that make them provable on CPU.
  - ``skew``: `PreferenceRankTest`, the rank-``k`` skew operator that recovers cyclic preferences the
    scalar head cannot express (T8).
  - ``subspace``: `cka`, `procrustes`, and null-anchored subspace / feature alignment, every number
    COVARIANT and frame-gated.

Importing this package pulls no torch. torch is needed only when `hessian_spectrum` or
`flat_subspace` actually run on a real model through ``Runtime.hvp``, and even then it is imported
lazily at call time. The dual-use ``gradient_ascent_probe`` (RK8) lives in ``hessian`` but is
deliberately not re-exported here; import it explicitly if you need it.
"""

from __future__ import annotations

from reward_lens.core.extras import require_extra

require_extra("white-box", subsystem="reward_lens.geometry")

from reward_lens.geometry.canonical import AngleResult, canonicalize, effective_angle
from reward_lens.geometry.frame import (
    Frame,
    FrameArtifact,
    fit_frame,
    frame_artifact,
    frame_evidence,
)
from reward_lens.geometry.hessian import (
    SpectrumResult,
    SubspaceResult,
    SymmetricOperator,
    flat_subspace,
    hessian_spectrum,
    lanczos,
    participation_ratio,
    slq_density,
)
from reward_lens.geometry.skew import PreferenceRankResult, PreferenceRankTest
from reward_lens.geometry.subspace import (
    AlignmentResult,
    ProcrustesResult,
    cka,
    hungarian_feature_alignment,
    procrustes,
    subspace_alignment,
)

__all__ = [
    # frame
    "Frame",
    "FrameArtifact",
    "fit_frame",
    "frame_artifact",
    "frame_evidence",
    # canonical / effective angle
    "canonicalize",
    "effective_angle",
    "AngleResult",
    # hessian spectroscopy
    "hessian_spectrum",
    "flat_subspace",
    "participation_ratio",
    "lanczos",
    "slq_density",
    "SymmetricOperator",
    "SpectrumResult",
    "SubspaceResult",
    # skew operator
    "PreferenceRankTest",
    "PreferenceRankResult",
    # subspace comparison
    "cka",
    "procrustes",
    "ProcrustesResult",
    "subspace_alignment",
    "hungarian_feature_alignment",
    "AlignmentResult",
]

"""``reward_lens.measure.indices`` — the index library (the A-series theory objects).

One module per index, each an `Observable` with a formal definition, a ``faithful_to`` anchor naming
the theory object it instantiates, and a ``gauge_status``. The indices are the scalar diagnostics the
cards and the scoreboard speak in; centralizing them here is what keeps the fifty programs of the
corpus speaking one language, and giving each a single definition with an explicit ``deviations`` list
is the structural fix for the operationalization drift that let v1 compute a coverage statistic while
citing a distortion index.

Every module exposes two things: a pure function that is the mathematics of the index, provable on
synthetic data where the answer is known, and an `Observable` subclass whose ``measure`` reads the
substrate and calls that function. The pure functions are the tested contract; the Observables are the
production path, degrading gracefully (a note in the Evidence, never a fabricated number) when a
dependency the corpus is still building — a concept-feature bank, a probe battery, an interventions
patch — is not yet available.

Theory-object coverage:

  - A1  KUI                  ``kui``                  (fully computed)
  - A2  Distortion v2        ``distortion``           (definition + synthetic test)
  -     coverage disparity   ``coverage_disparity``   (v1 statistic, honestly named)
  - A3  TeacherCompatibility ``teacher_compatibility``(fully computed)
  - A4  Tail index           ``tail``                 (fully computed)
  - A6  Verification / Style ``verification_score`` / ``style_share``  (definition + synthetic test)
  - A7  Receipt reliance     ``receipt_reliance``     (definition + synthetic test)
  - A8  Skepticism           ``skepticism``           (definition + synthetic test)
  - A9  Coherence            ``coherence``            (fully computed)
  - A10 Dark reward          ``dark_reward``          (definition + synthetic test)
  - A11 Interp coverage      ``interp_coverage``      (definition + synthetic test)
  - A12 χ susceptibility     ``chi``                  (fully computed)
  - A14 VCE                  ``vce``                  (definition + synthetic test)
  - A15 Legibility           ``legibility``           (definition + synthetic test)
  - A17 Eval-awareness       ``eval_awareness``       (definition + synthetic test)
  -     Robustness SNR       ``snr``                  (definition + synthetic test)
  -     Contested direction  ``contested``            (definition + synthetic test)

Importing this package pulls no torch: the pure functions are numpy, and the substrate readers import
torch lazily only when an Observable actually runs on a model.
"""

from __future__ import annotations

from reward_lens.measure.indices._support import (
    FeatureBank,
    LinearFeatureBank,
    percentile_within_battery,
)
from reward_lens.measure.indices.chi import Chi, predicted_hack_modes, susceptibility
from reward_lens.measure.indices.coherence import (
    Coherence,
    coherence_matrix,
    coherence_report,
    effective_dimension,
    max_offdiagonal_coherence,
    welch_bound,
)
from reward_lens.measure.indices.contested import Contested, contested_direction
from reward_lens.measure.indices.coverage_disparity import CoverageDisparity, coverage_disparity
from reward_lens.measure.indices.dark_reward import DarkReward, dark_reward
from reward_lens.measure.indices.distortion import (
    Distortion,
    distortion_per_dimension,
    linear_sensitivity,
)
from reward_lens.measure.indices.eval_awareness import EvalAwareness, eval_awareness_probe
from reward_lens.measure.indices.interp_coverage import InterpCoverage, interp_coverage
from reward_lens.measure.indices.kui import (
    KUI,
    Property,
    kui_from_properties,
    kui_plane,
    linear_mediation_proxy,
)
from reward_lens.measure.indices.legibility import Legibility, legibility_frontier
from reward_lens.measure.indices.receipt_reliance import ReceiptReliance, receipt_reliance
from reward_lens.measure.indices.skepticism import Skepticism, skepticism, skepticism_batch
from reward_lens.measure.indices.snr import RobustnessSNR, robustness_snr
from reward_lens.measure.indices.style_share import StyleShare, style_share
from reward_lens.measure.indices.tail import TailIndex, hill_tail_index, tail_estimate
from reward_lens.measure.indices.teacher_compatibility import (
    TeacherCompatibility,
    teacher_compatibility,
    teacher_compatibility_by_layer,
    teacher_compatibility_spectral,
)
from reward_lens.measure.indices.vce import (
    VCE,
    mean_cos2_alignment,
    value_convergence_excess,
)
from reward_lens.measure.indices.verification_score import (
    VerificationScore,
    verification_score,
)

__all__ = [
    # support / feature-bank interface
    "FeatureBank",
    "LinearFeatureBank",
    "percentile_within_battery",
    # A1 KUI
    "KUI",
    "Property",
    "kui_plane",
    "kui_from_properties",
    "linear_mediation_proxy",
    # A2 distortion + v1 coverage statistic
    "Distortion",
    "distortion_per_dimension",
    "linear_sensitivity",
    "CoverageDisparity",
    "coverage_disparity",
    # A3 teacher compatibility
    "TeacherCompatibility",
    "teacher_compatibility",
    "teacher_compatibility_spectral",
    "teacher_compatibility_by_layer",
    # A4 tail
    "TailIndex",
    "tail_estimate",
    "hill_tail_index",
    # A6 verification / style
    "VerificationScore",
    "verification_score",
    "StyleShare",
    "style_share",
    # A7 / A8 honesty law
    "ReceiptReliance",
    "receipt_reliance",
    "Skepticism",
    "skepticism",
    "skepticism_batch",
    # A9 coherence
    "Coherence",
    "coherence_matrix",
    "welch_bound",
    "max_offdiagonal_coherence",
    "effective_dimension",
    "coherence_report",
    # A10 / A11
    "DarkReward",
    "dark_reward",
    "InterpCoverage",
    "interp_coverage",
    # A12 chi
    "Chi",
    "susceptibility",
    "predicted_hack_modes",
    # A14 VCE
    "VCE",
    "value_convergence_excess",
    "mean_cos2_alignment",
    # A15 legibility
    "Legibility",
    "legibility_frontier",
    # A17 eval-awareness
    "EvalAwareness",
    "eval_awareness_probe",
    # robustness SNR + contested direction
    "RobustnessSNR",
    "robustness_snr",
    "Contested",
    "contested_direction",
]

"""``reward_lens.stats`` — the epistemics engine.

This is the layer that turns measurements into defensible claims: it holds the
numerical machinery the three gates rely on, and it enforces invariant I2 (a
number is worth no more than the sample behind it). It ports v1's excellent
``statistics.py`` and grows it into the place where effective sample size,
multiplicity, null models, and answer-key ROC live.

Five modules, one job each:

  - ``effects``: Cohen's d, the correlation effect size r, percentile and BCa
    bootstrap CIs, paired permutation, and Spearman-with-CI. The bootstrap
    result type ``BootstrapResult`` (with its ``method`` tag) lives here and is
    shared across the package.
  - ``ess``: design rule R7. Effective sample size from stimulus lineage, clone
    detection, and bootstrap/permutation that cluster-resample at the seed level
    by default. This is the module that makes v1's fake-n failure class
    impossible to hide.
  - ``multiplicity``: Benjamini-Hochberg FDR and its hierarchical (two-level)
    form for the Atlas's battery-of-batteries.
  - ``nulls``: random-direction, label-shuffle, and RUM-identifiability nulls,
    so every "aligns / predicts / exceeds" claim carries its own noise floor.
  - ``roc``: ROC/PR curves, calibration diagrams, and operating points, the
    compute engine behind the organism scorecard.

The package is deliberately light: it imports nothing heavier than numpy and
scipy, and scikit-learn is an optional accelerator, never required. ``import
reward_lens.stats`` is torch-free, so the whole epistemics layer is usable
without a model. Every routine returns plain values or plain dataclasses; the
Observable layer wraps them in Evidence later.
"""

from __future__ import annotations

from reward_lens.stats.changepoint import ChangePoint, bocpd, cusum, onset_lead_time
from reward_lens.stats.effects import (
    BootstrapResult,
    bca_bootstrap,
    bootstrap_ci,
    bootstrap_cohens_d,
    cohens_d,
    effect_size_r,
    paired_permutation_test,
    spearman_with_ci,
)
from reward_lens.stats.ess import (
    cluster_bootstrap,
    cluster_permutation,
    detect_clones,
    effective_sample_size,
)
from reward_lens.stats.meta import (
    EggerTest,
    FixedEffect,
    Heterogeneity,
    MetaAnalysis,
    PredictionRule,
    ProportionMeta,
    VoteCount,
    cochran_q,
    eggers_test,
    fixed_effect,
    heterogeneity,
    power_for_pooled_effect,
    power_to_detect_heterogeneity,
    prediction_interval,
    proportion_effects,
    proportion_meta,
    random_effects,
    tau2_dersimonian_laird,
    tau2_paule_mandel,
    tau2_q_profile_ci,
    tau2_reml,
    vote_count,
)
from reward_lens.stats.mi import (
    MICalibration,
    calibrate_gaussian,
    draw_correlated_gaussian,
    entropy_discrete_bits,
    gaussian_mi_bits,
    mi_binned,
    mi_discrete_continuous,
    mi_ksg,
)
from reward_lens.stats.multiplicity import bh_fdr, hierarchical_fdr
from reward_lens.stats.nulls import (
    random_direction_cosines,
    random_direction_null,
    rum_identifiability_null,
    shuffle_null,
)
from reward_lens.stats.roc import (
    ROCResult,
    calibration_curve,
    operating_point,
    roc_pr,
)

__all__ = [
    # meta-analysis (X8)
    "MetaAnalysis",
    "ProportionMeta",
    "Heterogeneity",
    "FixedEffect",
    "PredictionRule",
    "EggerTest",
    "VoteCount",
    "random_effects",
    "proportion_meta",
    "proportion_effects",
    "fixed_effect",
    "heterogeneity",
    "prediction_interval",
    "cochran_q",
    "tau2_dersimonian_laird",
    "tau2_paule_mandel",
    "tau2_reml",
    "tau2_q_profile_ci",
    "eggers_test",
    "vote_count",
    "power_for_pooled_effect",
    "power_to_detect_heterogeneity",
    # effects
    "BootstrapResult",
    "cohens_d",
    "effect_size_r",
    "bootstrap_ci",
    "bootstrap_cohens_d",
    "bca_bootstrap",
    "paired_permutation_test",
    "spearman_with_ci",
    # ess (R7)
    "effective_sample_size",
    "detect_clones",
    "cluster_bootstrap",
    "cluster_permutation",
    # multiplicity
    "bh_fdr",
    "hierarchical_fdr",
    # nulls
    "random_direction_cosines",
    "random_direction_null",
    "shuffle_null",
    "rum_identifiability_null",
    # roc
    "ROCResult",
    "roc_pr",
    "calibration_curve",
    "operating_point",
    # changepoint
    "ChangePoint",
    "cusum",
    "bocpd",
    "onset_lead_time",
    # mutual information (organism-calibrated)
    "gaussian_mi_bits",
    "entropy_discrete_bits",
    "mi_ksg",
    "mi_discrete_continuous",
    "mi_binned",
    "MICalibration",
    "draw_correlated_gaussian",
    "calibrate_gaussian",
]

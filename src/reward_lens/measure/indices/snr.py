"""Robustness SNR: reward signal against perturbation noise (a robustness diagnostic).

The robustness signal-to-noise ratio asks whether a reward's differences between conditions survive the
noise it shows under meaning-preserving perturbation. It is v1's ``PromptSNR`` as an index: cluster the
samples so that within a cluster the inputs are paraphrases (the perturbation the reward should ignore)
and across clusters they are genuinely different (the signal the reward should track). Then

    ``SNR = Var(cluster means) / mean(within-cluster variance)``,

the between-group reward variance the reward means carry over the within-group variance a paraphrase
induces. A high SNR means the reward's between-condition ordering is stable under perturbation; an SNR
near or below one means paraphrase noise swamps the signal, and any ranking read off the reward is
fragile.

This module has no single theory-object letter; it is the robustness statistic the cards and the
adversarial-robustness science (S13) consume. Deviation: the pure function is the variance-ratio
arithmetic on supplied grouped scores; the production path scores paraphrase clusters through the
signal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.types import Access, AccessMatrix, Capability, Component, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context, PreflightResult
from reward_lens.measure.indices._support import (
    ANY_SUBSTRATE,
    GRADER_STUDY_PHASES,
    MEASURED_BY,
    measured_without_input,
    missing_injection,
    reward_scores,
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def robustness_snr(values: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """Between-group over within-group reward variance (the robustness SNR).

    ``values`` are the per-sample rewards; ``groups`` labels each sample's paraphrase cluster. Returns
    the between-group variance (of the cluster means), the mean within-group variance (the paraphrase
    noise), and their ratio ``snr``. Clusters of size one contribute no within-group variance and are
    excluded from the noise estimate. A zero noise floor yields ``inf`` (perfectly stable), which is the
    honest reading, not a divide-by-zero error.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    g = np.asarray(groups).ravel()
    labels = np.unique(g)
    means = np.array([v[g == lab].mean() for lab in labels], dtype=np.float64)
    within = [np.var(v[g == lab], ddof=0) for lab in labels if np.sum(g == lab) > 1]
    between_var = float(np.var(means, ddof=0))
    within_var = float(np.mean(within)) if within else 0.0
    if within_var == 0:
        snr = float("inf") if between_var > 0 else float("nan")
    else:
        snr = between_var / within_var
    return {
        "snr": snr,
        "between_var": between_var,
        "within_var": within_var,
        "n_groups": labels.size,
    }


class RobustnessSNR(BaseObservable):
    """Robustness SNR: between-condition reward signal over within-condition paraphrase noise.

    Requires scores. The paraphrase-cluster labels are injected (or read from the view's group ids);
    the production path scores paraphrase clusters through the signal. Gauge is INVARIANT: a
    variance ratio is scale-free.

    What it cannot do. The whole reading rests on the claim that within a cluster the inputs are
    paraphrases and across clusters they are genuinely different. Nothing here checks that, and it
    is the assumption most likely to be wrong: a cluster whose members differ in content inflates
    the denominator and reports a fragile reward, while clusters that are near-duplicates of each
    other deflate the numerator and report the same thing for the opposite reason. An SNR near one
    is the boundary where a ranking read off the reward stops being reproducible, not a threshold
    anything here enforces.
    """

    name = "RobustnessSNR"
    version = "1.0"
    capabilities = Capability.SCORES
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = None
    deviations = (
        "no single theory-object letter; this is v1's PromptSNR as an index (robustness "
        "diagnostic for S13). The pure function is the variance-ratio arithmetic on grouped scores.",
    )

    # -- the observable declarations ---------------------------------------
    quantity = "grader.robustness_snr"
    requires: AccessMatrix = {Component.GRADER: Access.QUERY}
    substrates = ANY_SUBSTRATE
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset(
            {RegimeCondition.STATIONARY_GRADER, RegimeCondition.GROUP_NONDEGENERATE}
        ),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: Both variances scale by the same square of the reward's scale factor and neither sees the
    #: shift, so the ratio is unchanged under an affine rescaling.
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = ("baseline.shuffled_cluster_labels", "baseline.length")
    rung = 0

    def __init__(self, groups: Any = None) -> None:
        self.groups = groups

    def preflight(self, ctx: Context) -> PreflightResult:
        """The cluster labels or a refusal. Without them there is no within-group variance to divide by.

        The injected input is absent, which makes this a `Refusal` rather than an Evidence
        carrying a note. Nothing has to be computed to know it, so the question belongs
        here: `estimate` returns this refusal before `measure` is reached, and the
        capability report gets it with no work at all.
        """
        if self.groups is None:
            return missing_injection(
                self,
                needs={
                    "groups": "a length-n cluster label per sample, one cluster per set of paraphrases",
                },
                have="no cluster labels were injected",
                remedy=(
                    "Construct `RobustnessSNR(groups=...)` with one label per sample in the view, giving "
                    "samples that paraphrase each other the same label. The reading is between-cluster "
                    "variance over within-cluster variance, and a cluster of size one contributes no noise "
                    "estimate, so at least two clusters with two members each are needed for the ratio to "
                    "mean anything."
                ),
            )
        return super().preflight(ctx)

    def measure(self, ctx: Context) -> "Evidence":
        if self.groups is None:
            raise measured_without_input(self)
        reward = reward_scores(ctx.signal, ctx.view, ctx.readout)
        report = robustness_snr(reward, np.asarray(self.groups))
        return ctx.emit(report, uncertainty=Uncertainty(n=int(reward.size), method="none"))


__all__ = ["robustness_snr", "RobustnessSNR"]

"""``PromptSNR`` (E16): the reward's discriminative signal against its noise floor.

A reward model is only useful where its preference signal rises above the noise in its own scores. E16
frames this as a signal-to-noise ratio: the signal is how strongly the reward separates chosen from
rejected on an axis, and the noise is the spread of that separation across stimuli. A high ratio means
a reliable, reproducible preference; a ratio near the floor means the axis is within noise, and any
single-pair read of it is not trustworthy.

This is a working port. The signal power is the squared mean of the chosen-minus-rejected reward delta
on an axis and the noise power is its variance, so the ratio is ``mean^2 / var``, a genuine power SNR
(distinct from the standardized effect size E06 reports). The full E16 study drives the noise term
with the meaning-preserving prompt perturbations in ``data.corruptions`` (the robustness-SNR stimulus
generator); here the noise is the across-stimulus spread of the same axis, which is the substrate
available without the corruption pipeline. The ratio is dimensionless, so the gauge is INVARIANT.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.types import Access, AccessMatrix, Capability, Component, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.battery._common import (
    ANY_SUBSTRATE,
    GRADER_STUDY_PHASES,
    MEASURED_BY,
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def power_snr(deltas: np.ndarray) -> float:
    """Power signal-to-noise ratio ``mean(deltas)^2 / var(deltas)`` (0.0 when the noise is zero).

    A degenerate axis with no spread returns 0.0 rather than an infinite ratio, keeping the summary
    finite. This is a power ratio, not a standardized effect size, so it differs from Cohen's d.
    """
    arr = np.asarray(deltas, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    var = float(np.var(arr, ddof=1))
    if var < 1e-12:
        return 0.0
    return float(np.mean(arr) ** 2 / var)


class PromptSNR(BaseObservable):
    """Per-axis reward signal-to-noise ratio (E16).

    Requires only scores. The view is grouped by each pair's ``axis``. INVARIANT (a dimensionless
    power ratio).

    What it cannot do. The noise term is the spread of the reward delta across the stimuli of one
    axis, which is not the same object E16's ratio is about: E16 drives noise with meaning-preserving
    paraphrases of a single stimulus, so its denominator is the grader disagreeing with itself,
    while this denominator also contains real variation between stimuli. That makes this SNR a lower
    bound on the robustness SNR, not an estimate of it, and the two must not be compared. An axis
    with no spread returns 0.0 rather than infinity, so a zero here means "degenerate", not "no
    signal".
    """

    name = "PromptSNR"
    version = "1.0"
    capabilities = Capability.SCORES
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "E16 prompt robustness SNR"
    deviations = (
        "signal is mean(delta)^2, noise is var(delta) across the axis's stimuli; the full E16 drives "
        "noise with data.corruptions robustness-SNR paraphrases, deferred here to a working port",
    )

    # -- the declarations --------------------------------------------------
    quantity = "grader.prompt_snr"
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
    #: Numerator and denominator are both quadratic in the same deltas, so an affine rescaling of
    #: the reward cancels and the ratio does not move.
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = ("baseline.shuffled_axis_labels", "baseline.length")
    rung = 0

    def measure(self, ctx: Context) -> "Evidence":
        signal = ctx.signal
        readout = ctx.readout

        by_axis: dict[str, list] = defaultdict(list)
        for pair in ctx.view:
            axis = getattr(pair, "axis", "default")
            by_axis[axis].append(pair)

        per_axis: dict[str, dict] = {}
        for axis in sorted(by_axis):
            pairs = by_axis[axis]
            chosen = [(p.prompt_text, p.chosen.text) for p in pairs]
            rejected = [(p.prompt_text, p.rejected.text) for p in pairs]
            sc = signal.score(chosen, readout).value.values
            sr = signal.score(rejected, readout).value.values
            deltas = np.asarray(sc, dtype=np.float64) - np.asarray(sr, dtype=np.float64)
            per_axis[axis] = {
                "snr": power_snr(deltas),
                "mean_delta": float(np.mean(deltas)),
                "var_delta": float(np.var(deltas, ddof=1)) if deltas.size > 1 else 0.0,
                "n_pairs": len(pairs),
            }

        snrs = {a: v["snr"] for a, v in per_axis.items()}
        best = max(snrs, key=lambda a: snrs[a]) if snrs else None
        payload = {
            "per_axis": per_axis,
            "highest_snr_axis": best,
            "max_snr": max(snrs.values(), default=0.0),
            "n_axes": len(per_axis),
        }
        return ctx.emit(payload)


__all__ = ["PromptSNR", "power_snr"]

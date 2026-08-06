"""``LensCrystallization`` (E02): where preference forms across depth.

The reward lens projects the residual stream at each layer onto the reward direction to read the
reward the model would assign if it stopped there. For a preference pair the differential lens is the
chosen-minus-rejected projection at each layer; it traces when the model starts distinguishing the
two completions. The crystallization layer is the first layer at which that differential reaches half
its final value, and the crystallization fraction is that layer over the layer count, a depth in
``[~0, 1]`` that is comparable across models because it is a fraction of depth, not a reward-scale
quantity.

This Observable is a faithful port of v1's ``RewardLens``: same lens points (the post-embedding
residual as layer -1 and each block's ``resid_post``), same differential, same first-crossing rule
with the same degenerate-final fallback (when a soft cap collapses the final differential to
numerical zero, the reference becomes the largest-magnitude finite differential). The E-parity suite
checks it reproduces v1's per-pair crystallization layer and differential to 1e-6 on the tiny model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.types import Access, AccessMatrix, Capability, Component, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.battery._common import (
    GRADER_STUDY_PHASES,
    MEASURED_BY,
    NEURAL_SUBSTRATES,
    capture_sites,
    pair_sides,
    resid_sites,
    reward_direction,
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def crystallization_layer(differential: np.ndarray, layers: np.ndarray) -> int:
    """The first layer whose differential reaches half the final (reference) differential.

    This is v1's ``RewardLens.trace`` rule, reproduced exactly so the port is faithful. The reference
    is the final-layer differential; when that is non-finite or numerically zero (a soft-capped late
    layer), the reference falls back to the largest-magnitude finite differential. The crossing test
    respects the reference's sign, and a degenerate differential returns the last layer.
    """
    final_diff = differential[-1]
    ref = final_diff
    if not np.isfinite(ref) or abs(ref) < 1e-8:
        finite = differential[np.isfinite(differential)]
        if finite.size > 0:
            cand = float(finite[int(np.argmax(np.abs(finite)))])
            if abs(cand) >= 1e-8:
                ref = cand
    if np.isfinite(ref) and abs(ref) > 1e-8:
        threshold = 0.5 * ref
        crystal_idx = int(layers[-1])
        for i, d in enumerate(differential):
            if not np.isfinite(d):
                continue
            if (ref > 0 and d >= threshold) or (ref < 0 and d <= threshold):
                crystal_idx = int(layers[i])
                break
        return crystal_idx
    return int(layers[-1])


class LensCrystallization(BaseObservable):
    """Layer-by-layer preference formation and its crystallization depth (E02).

    Requires activation capture and a linear readout. The crystallization fraction is a depth in
    ``[~0, 1]`` and is gauge-invariant (comparable across signals), which is why E02 headlines the
    per-model mean fraction directly.

    What it cannot do. The crossing rule is a threshold on one differential curve, so it reports a
    depth even when the curve never separates: on a signal whose chosen-minus-rejected differential
    is inside its own noise, the returned layer is an argmax over noise and means nothing. The
    envelope requires a stationary grader because the whole curve is read off one set of weights;
    re-running it across a window in which the head changed compares two objects under one number.
    """

    name = "LensCrystallization"
    version = "1.0"
    capabilities = Capability.ACTIVATIONS | Capability.LINEAR_READOUT
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "E02 crystallization depth"
    deviations = (
        "crystallization fraction is layer / n_layers with the layer index running from -1 "
        "(post-embedding) to n_layers-1, matching v1's RewardLens exactly",
    )

    # -- the declarations --------------------------------------------------
    quantity = "grader.crystallization_depth"
    requires: AccessMatrix = {Component.GRADER: Access.FORWARD}
    substrates = NEURAL_SUBSTRATES
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: Two groups act on this reading and it is invariant under both, for two different reasons.
    #:
    #: Under `repr.basis`, every lens point is an inner product between a residual and the readout,
    #: so a shared orthogonal change of representation basis leaves the whole curve, and therefore
    #: the crossing layer, exactly where it was.
    #:
    #: Under `reward.affine` (`r -> a*r + b`) the offset cancels in the chosen-minus-rejected
    #: differential and the scale multiplies the curve and its half-of-final threshold together, so
    #: the crossing index does not move. That is true, and it is **measured**: over 2,000 random
    #: depth curves crossed with scales drawn log-uniformly on [0.1, 10] at both signs,
    #: `crystallization_layer(a * d)` equals `crystallization_layer(d)` on 2,000 of 2,000.
    #:
    #: It is still not declared, and the reason is worth recording rather than working around. The
    #: mapping form of `invariance_relation` now makes a second group declarable, but the
    #: `reward.affine` group action replaces `InvariancePayload.scores` and nothing else, and this
    #: instrument's reading is built from `activations @ readouts[0]`. Declaring the group therefore
    #: generates a test that passes because the action never reached the reading, not because the
    #: reading is invariant: checked directly, `r -> 4.885*r + 0.052` leaves `readouts` untouched and
    #: the crossing layer at 1.0 on both sides. A test that passes for the wrong reason reads as
    #: evidence and is not, which is worse than the comment it would replace. What would close this
    #: is an affine action that also scales the readout, since `w_r` is what carries the reward
    #: scale for an instrument that never looks at a recorded score.
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = ("baseline.final_layer_only", "baseline.random_direction")
    rung = 0
    #: An `IncrementalValidity` is required on every white-box reading and this
    #: instrument cannot produce one. The id is checkable and the prose is the argument.
    incremental_exemption = (
        "NO_SUBJECT_WITH_SIGNAL",
        "the differential curve is per-pair and the preference label is real, and a projection read "
        "at an intermediate layer is something no black-box method can produce, so the comparison is "
        "the right one to want. It is not available here: the only offline grader in this build is "
        "`signals.from_tiny`, a randomly initialised two-layer LlamaForSequenceClassification. "
        "Measured on the 84-pair diagnostic set unfolded to 168 items, it scores AUROC 0.5568 against "
        "0.6892 for the TF-IDF baseline, so the white-box side of the comparison has no signal to "
        "contribute and the four numbers would describe the fixture rather than the instrument. A "
        "trained grader is the missing input, and the crossing layer has to be fixed out of fold "
        "before the own-score is honest, because each pair's crystallisation layer is read off that "
        "pair's own curve.",
    )

    def measure(self, ctx: Context) -> "Evidence":
        signal = ctx.signal
        n_layers = int(signal.meta.n_layers)
        w_r = reward_direction(signal, ctx.readout)
        sites = resid_sites(n_layers)
        layers = np.array([-1] + list(range(n_layers)))

        chosen, rejected = pair_sides(ctx.view)
        cap_c = capture_sites(signal, chosen, sites)
        cap_r = capture_sites(signal, rejected, sites)

        import torch

        w = w_r.to(torch.float32)
        # Per-site differential projection: (chosen - rejected) . w_r at the final token, per pair.
        cols = []
        for site in sites:
            hc = cap_c[site].to(torch.float32)
            hr = cap_r[site].to(torch.float32)
            cols.append(((hc - hr) @ w).cpu().numpy())
        differential = np.stack(cols, axis=1)  # (n_pairs, n_lens_points)

        crystal_layers = np.array(
            [crystallization_layer(differential[i], layers) for i in range(differential.shape[0])]
        )
        crystal_frac = crystal_layers / n_layers
        final_diff = differential[:, -1]

        payload = {
            "mean_crystal_frac": float(np.mean(crystal_frac)),
            "per_pair_crystal_frac": crystal_frac.tolist(),
            "per_pair_crystal_layer": crystal_layers.tolist(),
            "per_pair_final_differential": final_diff.tolist(),
            "differential": differential.tolist(),
            "layers": layers.tolist(),
            "n_layers": n_layers,
            "n_pairs": int(differential.shape[0]),
        }
        return ctx.emit(payload)


__all__ = ["LensCrystallization", "crystallization_layer"]

"""``ConflictMatrix`` (E09): the geometry of competing reward terms.

Different quality axes (helpfulness, verbosity, formatting, ...) each define a direction in activation
space, estimated as the mean chosen-minus-rejected difference for that axis. Their pairwise cosines
say whether two axes cooperate (aligned), are independent (orthogonal), or pull against each other
(conflict). Conflicting reward terms are where monitorability degrades, because optimizing one term
degrades another the grader also cares about.

This ports v1's ``RewardConflictAnalyzer``: learn a direction per axis by mean difference, then read
the cosine matrix and classify each pair. The cosines are raw-coordinate (RAW_ONLY): they depend on
the residual-stream basis and are a single-model internal geometry. The view is grouped by each pair's
``axis``, so a diagnostic set with several axes yields the full inter-axis conflict matrix.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.errors import CapabilityError
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.types import Access, AccessMatrix, Capability, Component, GaugeStatus, Site
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.battery._common import (
    GRADER_STUDY_PHASES,
    MEASURED_BY,
    NEURAL_SUBSTRATES,
    capture_sites,
)
from reward_lens.measure.battery.geometry import cosine_matrix

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


class ConflictMatrix(BaseObservable):
    """Inter-axis reward-term conflict geometry (E09).

    Requires activation capture. The view must span at least two axes. Marked RAW_ONLY because the
    term cosines are in raw residual-stream coordinates.

    What it cannot do. A term direction here is the mean activation difference on an axis, which is
    a correlational summary of the diagnostic set and not the direction the grader prices: two axes
    whose stimuli happen to co-vary in the data produce an aligned pair of directions whether or not
    the reward treats them as one term. The conflict threshold of -0.3 is a reporting convention,
    not a measurement, and the count beside it inherits that. Nothing here is causal; steering along
    one direction and reading the other axis is what would make it so.

    This instrument does not declare ``LINEAR_READOUT``. It never reaches a readout vector: the
    directions come from activation differences at a hardcoded final residual site, and the cosines
    are computed between those differences. The declaration was dropped after an audit of all
    fourteen sites in ``measure/``.
    """

    name = "ConflictMatrix"
    version = "1.0"
    capabilities = Capability.ACTIVATIONS
    gauge_status = GaugeStatus.RAW_ONLY
    faithful_to = "E09 reward-term conflict geometry"
    deviations = (
        "term directions are the mean chosen-minus-rejected difference per axis (unnormalized, as "
        "v1 learned them); cosines are RAW_ONLY (basis-dependent), meaningful within one model",
    )

    # -- the declarations --------------------------------------------------
    quantity = "grader.term_conflict"
    requires: AccessMatrix = {Component.GRADER: Access.FORWARD}
    substrates = NEURAL_SUBSTRATES
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: A cosine between two vectors is unchanged when the same orthogonal map acts on both, and
    #: `repr.basis` is exactly that map. ``GaugeStatus.RAW_ONLY`` above is a different claim and it
    #: stands: it says two *different* models' bases have no correspondence, which is a frame
    #: question gate 2 enforces, not a statement about one model under a change of coordinates.
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = ("baseline.random_direction_pair", "baseline.shuffled_axis_labels")
    rung = 0
    #: An `IncrementalValidity` is required on every white-box reading and this
    #: instrument cannot produce one. The id is checkable and the prose is the argument.
    incremental_exemption = (
        "NO_PER_ITEM_VERDICT",
        "the reading is a cosine matrix over axes: one number per ordered pair of axes, each computed "
        "from a mean activation difference over all the pairs on that axis. The per-item structure is "
        "consumed by the averaging, and no label says which axes ought to conflict, so neither side "
        "of the comparison has an error vector.",
    )

    def measure(self, ctx: Context) -> "Evidence":
        import torch

        signal = ctx.signal
        n_layers = int(signal.meta.n_layers)
        site = Site(n_layers - 1, "resid_post")

        by_axis: dict[str, list] = defaultdict(list)
        for pair in ctx.view:
            axis = getattr(pair, "axis", "default")
            by_axis[axis].append(pair)
        axes = sorted(by_axis)
        if len(axes) < 2:
            raise CapabilityError(
                f"ConflictMatrix needs pairs spanning >=2 axes; the view has {len(axes)}. "
                f"Pass a multi-axis diagnostic view."
            )

        directions = []
        for axis in axes:
            pairs = by_axis[axis]
            chosen = [(p.prompt_text, p.chosen.text) for p in pairs]
            rejected = [(p.prompt_text, p.rejected.text) for p in pairs]
            hc = capture_sites(signal, chosen, (site,))[site].to(torch.float32)
            hr = capture_sites(signal, rejected, (site,))[site].to(torch.float32)
            directions.append((hc - hr).mean(dim=0).cpu().numpy())
        vectors = np.stack(directions, axis=0)
        cosines = cosine_matrix(vectors)
        off = cosines[~np.eye(len(axes), dtype=bool)]

        payload = {
            "axes": axes,
            "cosine_matrix": cosines.tolist(),
            "mean_offdiagonal_cosine": float(np.mean(off)),
            "min_cosine": float(np.min(off)),
            "n_conflicting_pairs": int(np.sum(off < -0.3) // 2),
            "n_axes": len(axes),
        }
        return ctx.emit(payload)


__all__ = ["ConflictMatrix"]

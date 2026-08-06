"""``MultiObjectiveGeometry`` (E18): the geometry of a multi-objective reward head.

A multi-objective reward model such as ArmoRM carries one reward direction per objective (nineteen
for ArmoRM). The geometry of those directions, the pairwise cosines between objectives, is what tells
you whether the objectives cooperate, are orthogonal, or conflict. v1 could only see this after
collapsing the head to a row mean, which throws the geometry away by construction; the whole point of
the first-class readout (R4) is that each objective is its own readout and the geometry is directly
computable.

This Observable reads every objective readout off the head (never the row-mean composite) and returns
the full cosine matrix. The cosines are raw-coordinate quantities: they depend on the residual-stream
basis and carry no shared frame, so the gauge is RAW_ONLY. They are honest and scientifically
interesting within one model (this is E18's whole result), but a cosine here must not be compared
across models without a frame (gate 2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.errors import CapabilityError
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    GaugeStatus,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.battery._common import GRADER_STUDY_PHASES, MEASURED_BY

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def _objective_readouts(signal) -> list:
    """The per-objective readouts (the ``criterion:*`` rows), never the row-mean ``reward``.

    A multi-objective head exposes one ``criterion:k`` readout per row plus a legacy ``reward``
    composite whose vector is the row mean. E18 is precisely the geometry the row mean destroys, so
    this selects the criterion readouts and refuses to fall back to the composite.
    """
    criteria = [
        r for r in signal.readouts() if r.name.startswith("criterion:") and r.vector is not None
    ]
    return criteria


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """The pairwise cosine-similarity matrix of a set of row vectors ``(K, d)``.

    Rows are normalized to unit length (a zero row is left as zero, yielding zero cosines) and the
    Gram matrix of the normalized rows is returned, an ``(K, K)`` matrix with ones on the diagonal.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms < 1e-12, 1.0, norms)
    unit = vectors / safe
    return unit @ unit.T


class MultiObjectiveGeometry(BaseObservable):
    """The cosine geometry of a multi-objective reward head (E18).

    Requires a linear readout with more than one objective row. Marked RAW_ONLY because the cosines
    are in raw residual-stream coordinates; they are a single-model internal geometry and are not
    cross-model comparable without a frame.

    What it cannot do. This is the geometry of the head's weights and nothing else. Two objectives
    whose directions are nearly orthogonal can still be perfectly correlated in practice if the
    activations the model produces occupy a subspace where both load the same way, and this
    instrument never looks at an activation, so it cannot see that. The conflict threshold of -0.3
    is a reporting convention and the count beside it inherits that.
    """

    name = "MultiObjectiveGeometry"
    version = "1.0"
    capabilities = Capability.LINEAR_READOUT
    gauge_status = GaugeStatus.RAW_ONLY
    faithful_to = "E18 ArmoRM 19x19 objective geometry"
    deviations = (
        "reads the per-objective (criterion) readouts directly and never the row-mean composite; "
        "cosines are raw-coordinate (RAW_ONLY), meaningful within one model only",
    )

    # -- the declarations --------------------------------------------------
    quantity = "grader.objective_geometry"
    requires: AccessMatrix = {Component.GRADER: Access.FORWARD}
    #: NEURAL_SCALAR only. A multi-objective head is a matrix of criterion rows; a GenRM has a
    #: verdict direction and no per-criterion weights, and a PROCEDURAL rubric ensemble has criteria
    #: but no vectors at all, so neither has an objective geometry to read.
    substrates = frozenset({Substrate.NEURAL_SCALAR})
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: A cosine between two readouts is unchanged when the same orthogonal map acts on both.
    #: ``GaugeStatus.RAW_ONLY`` above is the separate and still-true claim that two different
    #: models' bases have no correspondence, which gate 2 enforces with a frame.
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = ("baseline.random_direction_pair", "baseline.row_mean_composite")
    rung = 0
    #: An `IncrementalValidity` is required on every white-box reading and this
    #: instrument cannot produce one. The id is checkable and the prose is the argument.
    incremental_exemption = (
        "NO_PER_ITEM_VERDICT",
        "the reading is the cosine matrix of the head's own objective rows. It never looks at an "
        "activation and never looks at an item, so there is nothing per-item to score and no "
        "black-box rival to a statement about weight geometry. This instrument declares "
        "LINEAR_READOUT only, so the shipped WHITE_BOX mask does not currently reach it; the "
        "exemption is declared anyway so that widening the mask does not silently open a hole.",
    )

    def measure(self, ctx: Context) -> "Evidence":
        import torch

        signal = ctx.signal
        readouts = _objective_readouts(signal)
        if len(readouts) < 2:
            raise CapabilityError(
                f"MultiObjectiveGeometry needs a multi-objective head (>=2 objective readouts); "
                f"signal {signal.meta.fingerprint} exposes {len(readouts)}. A scalar reward model "
                f"has no objective geometry to read."
            )
        names = [r.name for r in readouts]
        vectors = np.stack(
            [r.vector.to(torch.float32).detach().cpu().numpy() for r in readouts], axis=0
        )
        cosines = cosine_matrix(vectors)
        off = cosines[~np.eye(len(names), dtype=bool)]

        payload = {
            "objectives": names,
            "n_objectives": len(names),
            "cosine_matrix": cosines.tolist(),
            "mean_offdiagonal_cosine": float(np.mean(off)),
            "min_cosine": float(np.min(off)),
            "max_offdiagonal_cosine": float(np.max(off)),
            "n_conflicting_pairs": int(np.sum(off < -0.3) // 2),
        }
        return ctx.emit(payload)


__all__ = ["MultiObjectiveGeometry", "cosine_matrix"]

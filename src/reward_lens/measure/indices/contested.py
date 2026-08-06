"""Contested direction: the axis annotators disagree along (S11 machine psychology).

Where preferences are contested, a single scalar reward cannot represent everyone (T7); the useful
object is the direction in representation space along which the disagreement lives. Given, per pair, the
activation difference ``Δh`` (chosen minus rejected) and a disagreement signal (annotator vote entropy,
label variance, or a diverging-preferences score), the contested direction is the axis of ``Δh`` whose
projection best tracks disagreement:

    ``c ∝ Σ_i (disagreement_i − mean) · Δh_i``,

the covariance direction between the representation change and the disagreement. Pairs the annotators
split on pull ``c`` toward the representation change that distinguishes them; unanimous pairs contribute
nothing. The magnitude of the alignment says how much of the disagreement is linearly organized along a
single axis versus scattered.

This module has no single theory-object letter; it is the contested-direction diagnostic S11
consumes. Deviation: the pure function is the covariance-direction recovery on supplied ``Δh`` and
disagreement; the production path reads ``Δh`` from the signal and disagreement from the data
plane's annotator records. The direction is COVARIANT, so a cross-signal comparison of contested
directions needs a shared frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def contested_direction(delta_h: np.ndarray, disagreement: np.ndarray) -> dict[str, object]:
    """The disagreement-covariance direction and how much disagreement it organizes.

    ``delta_h`` is ``(n, d)`` (per-pair chosen-minus-rejected activations); ``disagreement`` is ``(n,)``.
    Returns the unit contested direction ``c`` (the normalized covariance ``Δhᵀ (disagreement − mean)``),
    the correlation between ``Δh @ c`` and the disagreement (how well a single axis explains it), and the
    raw covariance norm. When the covariance is degenerate the direction is returned as zeros and the
    correlation as ``nan``.
    """
    dh = np.asarray(delta_h, dtype=np.float64)
    dis = np.asarray(disagreement, dtype=np.float64).ravel()
    dis_c = dis - dis.mean()
    cov = dh.T @ dis_c / dh.shape[0]  # (d,)
    norm = float(np.linalg.norm(cov))
    if norm == 0:
        return {"direction": np.zeros(dh.shape[1]), "correlation": float("nan"), "cov_norm": 0.0}
    direction = cov / norm
    proj = dh @ direction
    proj_c = proj - proj.mean()
    denom = np.linalg.norm(proj_c) * np.linalg.norm(dis_c)
    corr = float(proj_c @ dis_c / denom) if denom > 0 else float("nan")
    return {"direction": direction, "correlation": corr, "cov_norm": norm}


class Contested(BaseObservable):
    """Contested direction: the representation axis annotators disagree along.

    Requires activations. The per-pair ``Δh`` and the disagreement signal are injected (the data plane's
    annotator records supply disagreement in production). The direction is COVARIANT: comparing contested
    directions across signals requires a shared frame (gate 2). Within one signal the recovered axis and
    its correlation are reported directly.

    What it cannot do. There is always a direction: the covariance of ``Δh`` with any disagreement
    vector is non-zero for essentially any input, so this instrument returns an axis whether or not
    the disagreement is organised at all. The correlation beside it is the only thing that says
    otherwise, and it has no null attached here, so it must be read against the shuffled-disagreement
    baseline rather than on its own. One axis is fitted, so disagreement that lives on two competing
    axes is reported as their resultant, which is a direction nobody disagrees along.
    """

    name = "Contested"
    version = "1.0"
    capabilities = Capability.ACTIVATIONS
    gauge_status = GaugeStatus.COVARIANT
    faithful_to = None
    deviations = (
        "no single theory-object letter; the contested-direction diagnostic for S11 (annotator "
        "disagreement, T7). Direction is COVARIANT and frame-gated for cross-signal comparison.",
    )

    # -- the observable declarations ---------------------------------------
    quantity = "grader.contested_axis"
    #: Both inputs are recorded: the per-pair activation difference from an earlier capture and the
    #: annotator disagreement from the data plane. Nothing here calls the grader.
    requires: AccessMatrix = {Component.RECORD: Access.RECORD}
    substrates = ANY_SUBSTRATE
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.GROUP_NONDEGENERATE}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: The reported scalar is a correlation between two projections, which a shared orthogonal map
    #: leaves unchanged. The direction itself moves with the basis, which is what
    #: ``GaugeStatus.COVARIANT`` records and what gate 2 makes a caller supply a frame for.
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = ("baseline.shuffled_disagreement", "baseline.random_direction")
    rung = 0
    #: A white-box reading owes an `IncrementalValidity` and this instrument cannot produce
    #: one. The id is checkable and the prose is the argument.
    incremental_exemption = (
        "NO_BLACK_BOX_ON_THESE_ITEMS",
        "the instrument consumes an (n, d) activation-difference matrix and an (n,) disagreement "
        "vector and never touches the view, so its items carry no text, no logged series and no "
        "judge, and all six baselines refuse for want of a field they name. The white-box side is "
        "ready: the projection onto the fitted axis is a per-item score and the disagreement is a "
        "per-item target. A record becomes measurable the moment the pairs arrive with their "
        "transcripts attached.",
    )

    def __init__(
        self,
        delta_h: np.ndarray | None = None,
        disagreement: np.ndarray | None = None,
    ) -> None:
        self.delta_h = delta_h
        self.disagreement = disagreement

    def preflight(self, ctx: Context) -> PreflightResult:
        """Both arrays or a refusal. A contested direction is a covariance and needs two vectors.

        The injected input is absent, which makes this a `Refusal` rather than an Evidence
        carrying a note. Nothing has to be computed to know it, so the question belongs
        here: `estimate` returns this refusal before `measure` is reached, and the
        capability report gets it with no work at all.
        """
        if self.delta_h is None or self.disagreement is None:
            return missing_injection(
                self,
                needs={
                    "delta_h": "an (n, d) array of per-pair chosen-minus-rejected activations",
                    "disagreement": "an (n,) array of per-pair grader disagreement",
                },
                have="neither was injected",
                remedy=(
                    "Construct `Contested(delta_h=..., disagreement=...)`. `delta_h` is the (n, d) matrix of "
                    "chosen-minus-rejected final-token activations, one row per comparison; `disagreement` "
                    "is the matching (n,) vector of how far the graders disagreed on that comparison. They "
                    "index the same pairs, so their first dimension has to match."
                ),
            )
        return super().preflight(ctx)

    def measure(self, ctx: Context) -> "Evidence":
        if self.delta_h is None or self.disagreement is None:
            raise measured_without_input(self)
        result = contested_direction(self.delta_h, self.disagreement)
        payload = {
            "direction": np.asarray(result["direction"], dtype=np.float64),
            "correlation": result["correlation"],
            "cov_norm": result["cov_norm"],
            "n_pairs": int(np.asarray(self.delta_h).shape[0]),
        }
        return ctx.emit(payload, uncertainty=Uncertainty(method="none"))


__all__ = ["contested_direction", "Contested"]

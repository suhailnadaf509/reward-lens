"""A2 Distortion v2: per-dimension reward distortion.

Formal definition, A2. ``D(P) = (curvature/sensitivity of r along the P direction) × (1 −
coverage(P))``, with ``coverage(P)`` supplied by KUI's mediation term (A1). This is the Wang-Huang
per-dimension distortion object (faithful_to Wang-Huang 2603.28063): the reward is distorted along a
property when it is sensitive to that property yet the property is not covered by the intended
criteria, so the sensitivity is spurious. High sensitivity that is fully covered is intended pricing,
not distortion; low sensitivity is no distortion regardless of coverage.

This is the v2 that A2 distinguishes from v1's coverage statistic. v1 (E10) computed a
coverage disparity while citing Wang-Huang's distortion index; that statistic survives under its honest
name in ``coverage_disparity.py`` and is not this object. Deviation from A2: sensitivity defaults to the
linear proxy ``|w_r · v_P|`` for a unit direction; passing a reward-Hessian quadratic form ``v_Pᵀ H
v_P`` gives the curvature reading A2 names first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.types import Access, AccessMatrix, Capability, Component, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.indices._support import (
    GRADER_STUDY_PHASES,
    MEASURED_BY,
    NEURAL_SUBSTRATES,
    reward_vector,
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def linear_sensitivity(directions: np.ndarray, w_r: np.ndarray) -> np.ndarray:
    """Per-dimension linear sensitivity ``|w_r · v_P|`` for unit-normalized directions (A2).

    The first-order change in reward per unit step along each property direction. ``directions`` is
    ``(K, d)``; returns the ``(K,)`` vector of sensitivities. This is the cheap proxy for A2's
    curvature/sensitivity term; the Hessian quadratic form is the production reading.
    """
    v = np.asarray(directions, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError(f"directions must be (K, d); got shape {v.shape}")
    w = np.asarray(w_r, dtype=np.float64).ravel()
    norms = np.linalg.norm(v, axis=1)
    norms[norms == 0] = 1.0
    return np.abs(v @ w) / norms


def distortion_per_dimension(
    sensitivity: Sequence[float],
    coverage: Sequence[float],
) -> np.ndarray:
    """Per-dimension distortion ``D(P) = sensitivity(P) × (1 − coverage(P))`` (A2).

    ``coverage`` is KUI's mediation term in ``[0, 1]`` (how priced/covered the property is). A property
    that is sensitive but uncovered (``coverage → 0``) carries its full sensitivity as distortion; a
    fully covered property (``coverage → 1``) carries none. Both inputs are length ``K``.
    """
    s = np.asarray(sensitivity, dtype=np.float64).ravel()
    c = np.clip(np.asarray(coverage, dtype=np.float64).ravel(), 0.0, 1.0)
    if s.size != c.size:
        raise ValueError(f"sensitivity ({s.size}) and coverage ({c.size}) must match in length")
    return s * (1.0 - c)


class Distortion(BaseObservable):
    """A2 per-dimension distortion v2, with KUI supplying the coverage term.

    Requires activations and a linear readout. Takes a battery of property directions and their KUI
    mediation-percentile coverage (injected; the concept layer and the KUI index supply both in
    production), computes the linear sensitivity ``|w_r · v_P|`` (or uses an injected Hessian curvature),
    and reports the per-dimension distortion. Gauge is INVARIANT: distortion is a within-signal
    per-direction magnitude.

    What it cannot do. The default sensitivity is the linear proxy ``|w_r . v_P|``, which is the
    first-order response along a direction and not the curvature A2 names first; on a reward with
    real curvature along a property the proxy can be near zero where the Hessian quadratic form is
    large, and the product with ``(1 - coverage)`` then reports no distortion at a dimension that is
    badly distorted. Coverage is consumed, never computed here, so the whole reading is conditional
    on KUI's mediation term and on the property battery being the right one: a property nobody
    supplied a direction for contributes zero distortion and is indistinguishable from one that is
    fully covered.
    """

    name = "Distortion"
    version = "1.0"
    capabilities = Capability.ACTIVATIONS | Capability.LINEAR_READOUT
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A2"
    deviations = (
        "sensitivity defaults to the linear proxy |w_r . v_P|; a reward-Hessian quadratic form gives "
        "the curvature reading",
        "coverage is KUI's mediation term (A1); this index consumes it rather than recomputing it",
    )

    # -- the observable declarations ---------------------------------------
    quantity = "grader.distortion_index"
    #: The reward direction is read off the head; the property directions and their coverage come
    #: from earlier measurements by the concept layer and by KUI.
    requires: AccessMatrix = {
        Component.GRADER: Access.FORWARD,
        Component.RECORD: Access.RECORD,
    }
    substrates = NEURAL_SUBSTRATES
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.STATIONARY_GRADER, RegimeCondition.ABOVE_LOD}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: The sensitivity is an inner product between a property direction and the readout, and both
    #: rotate together under `repr.basis`, so the per-dimension distortion does not move.
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = ("baseline.random_direction", "baseline.zero_coverage")
    rung = 0
    #: A white-box reading owes an `IncrementalValidity` and this instrument cannot produce
    #: one. The id is checkable and the prose is the argument.
    incremental_exemption = (
        "NO_PER_ITEM_VERDICT",
        "the reading is one distortion per property direction, K numbers over a battery the caller "
        "supplies. A property direction is not an item any black-box method can read, and no label "
        "says which properties are distorted, so neither method has an error vector. The coverage "
        "term is consumed from KUI rather than measured here, which is a second reason the reading is "
        "not this instrument's own verdict about anything.",
    )

    def __init__(
        self,
        directions: np.ndarray | None = None,
        *,
        coverage: Sequence[float] | None = None,
        curvature: Sequence[float] | None = None,
        names: Sequence[str] | None = None,
    ) -> None:
        self.directions = directions
        self.coverage = coverage
        self.curvature = curvature
        self.names = names

    def measure(self, ctx: Context) -> "Evidence":
        if self.coverage is None or (self.directions is None and self.curvature is None):
            return ctx.emit(
                {
                    "distortion": [],
                    "note": "distortion needs property directions/curvature and KUI coverage; "
                    "none injected",
                },
                uncertainty=Uncertainty(method="none"),
            )
        if self.curvature is not None:
            sensitivity = np.abs(np.asarray(self.curvature, dtype=np.float64).ravel())
            k = sensitivity.size
        else:
            w_r = reward_vector(ctx.signal, ctx.readout)
            sensitivity = linear_sensitivity(self.directions, w_r)
            k = sensitivity.size
        distortion = distortion_per_dimension(sensitivity, self.coverage)
        names = list(self.names) if self.names is not None else [f"P{i}" for i in range(k)]
        payload = {
            "names": names,
            "distortion": distortion.tolist(),
            "sensitivity": sensitivity.tolist(),
            "coverage": list(np.asarray(self.coverage, dtype=float)),
            "total_distortion": float(np.sum(distortion)),
        }
        return ctx.emit(payload, uncertainty=Uncertainty(n=k, method="none"))


__all__ = ["linear_sensitivity", "distortion_per_dimension", "Distortion"]

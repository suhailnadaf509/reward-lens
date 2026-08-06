"""A1 KUI: the Knowledge-Utilization Index.

Formal definition, A1. For a property ``P`` in a battery of properties:

  - ``decode(P)`` = percentile-within-battery of a calibrated probe's balanced accuracy for ``P`` from
    the activations ``h`` (how legibly the reward model represents ``P``).
  - ``mediate(P)`` = percentile-within-battery of the causal ``Δr`` from steering or ablating the ``P``
    direction, with ``cos(w_P, w_r)`` as the cheap linear proxy (how much the reward is actually priced
    on ``P``).
  - ``KUI(P)`` = signed perpendicular distance of ``(decode, mediate)`` above the diagonal.

The diagonal ``decode = mediate`` is the locus where a property is priced exactly in proportion to how
legibly it is represented. ``KUI(P) > 0`` means ``P`` is more represented than it is priced:
represented-but-unpriced, the predicted hack dimension, because optimization can move a property the
reward can see but does not currently charge for. ``KUI(P) < 0`` means priced-beyond-representation.
This is Orgad-style knowledge/behaviour dissociation made grader-side (faithful_to Orgad-style
dissociation).

This module fixes the v1 unit bug named in A1: v1 computed ``decodability − mediation`` on raw
incommensurable scales (a balanced accuracy minus a cosine), a subtraction with no meaning. Here both
axes are pushed to their percentile-within-battery first, so the difference is between two ranks in
``[0, 1]`` and the plane is the honest object. Deviation from A1: mediation uses the cheap linear proxy
``cos(w_P, w_r)`` by default; the production path substitutes the measured causal ``Δr`` from steering,
and the reported plane is unchanged when it does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

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
    percentile_within_battery,
    reward_vector,
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence

_SQRT2 = float(np.sqrt(2.0))


@dataclass
class Property:
    """One property in the KUI battery: its name, decodability, and pricing evidence.

    ``decodability`` is the probe's balanced accuracy for the property (the production path fits the
    probe; the test supplies it directly). ``mediation`` is the measured causal ``Δr`` when available;
    when it is None and a ``direction`` ``w_P`` is given, the cheap linear proxy ``|cos(w_P, w_r)|`` is
    used instead. A property needs at least one of ``mediation`` or ``direction``.
    """

    name: str
    decodability: float
    mediation: float | None = None
    direction: np.ndarray | None = None


def linear_mediation_proxy(direction: np.ndarray, w_r: np.ndarray) -> float:
    """The cheap linear mediation proxy ``|cos(w_P, w_r)|`` (A1).

    A property whose direction is nearly parallel to the reward direction is one the reward prices
    heavily; a direction orthogonal to ``w_r`` moves the reward not at all under a linear head. The
    magnitude is taken because pricing is about coupling strength, not sign.
    """
    p = np.asarray(direction, dtype=np.float64).ravel()
    w = np.asarray(w_r, dtype=np.float64).ravel()
    denom = np.linalg.norm(p) * np.linalg.norm(w)
    if denom == 0:
        return 0.0
    return float(abs(np.dot(p, w)) / denom)


def kui_plane(
    decodability: Sequence[float],
    mediation: Sequence[float],
) -> dict[str, np.ndarray]:
    """The KUI plane: standardize both axes to percentile-within-battery, then signed diagonal distance.

    ``KUI = (decode_pct − mediate_pct) / √2`` is the signed perpendicular distance from the diagonal
    ``decode = mediate``, positive for represented-but-unpriced properties. Standardizing both axes to
    ranks in ``[0, 1]`` before subtracting is the whole point (A1's unit-bug fix); the raw
    balanced accuracy and raw cosine never meet on the same scale. Returns the two percentile axes and
    the KUI vector, all length ``m`` (the battery size).
    """
    decode_pct = percentile_within_battery(np.asarray(decodability, dtype=np.float64))
    mediate_pct = percentile_within_battery(np.asarray(mediation, dtype=np.float64))
    kui = (decode_pct - mediate_pct) / _SQRT2
    return {"kui": kui, "decode_pct": decode_pct, "mediate_pct": mediate_pct}


def kui_from_properties(
    properties: Sequence[Property],
    w_r: np.ndarray,
) -> dict[str, Any]:
    """Assemble the KUI plane from a battery of ``Property`` objects (the observable's core).

    Fills each property's mediation from its measured ``Δr`` when present, else from the linear proxy
    ``|cos(w_P, w_r)|``, then builds the plane. Returns the names, both raw axes, the two percentile
    axes, and the per-property KUI, ready for the payload.
    """
    names = [p.name for p in properties]
    decode_raw = np.asarray([p.decodability for p in properties], dtype=np.float64)
    mediate_raw = np.empty(len(properties), dtype=np.float64)
    for i, p in enumerate(properties):
        if p.mediation is not None:
            mediate_raw[i] = float(p.mediation)
        elif p.direction is not None:
            mediate_raw[i] = linear_mediation_proxy(p.direction, w_r)
        else:
            raise ValueError(f"property {p.name!r} has neither a mediation nor a direction")
    plane = kui_plane(decode_raw, mediate_raw)
    return {
        "names": names,
        "decode_raw": decode_raw,
        "mediate_raw": mediate_raw,
        "decode_pct": plane["decode_pct"],
        "mediate_pct": plane["mediate_pct"],
        "kui": plane["kui"],
    }


class KUI(BaseObservable):
    """A1 Knowledge-Utilization Index: the ``(decode, mediate)`` plane and the diagonal distance.

    Requires activations and a linear readout on the production path (a probe reads decodability from
    ``h`` and a direction gives the mediation proxy). The battery of properties is injected (each a
    ``Property`` with a decodability and either a measured ``Δr`` or a direction ``w_P``); the concept
    layer supplies probes and directions in production. When the mediation proxy is used, the cosine is
    read against a random-direction null so a "represented-but-unpriced" flag beats the high-dimensional
    cosine noise floor. Gauge is INVARIANT: the plane is a within-battery, within-signal object.

    What it cannot do. Mediation defaults to ``|cos(w_P, w_r)|``, a geometric proxy for a causal
    quantity: a property whose direction is nearly orthogonal to the reward can still move the
    reward through a non-linear route, and the proxy reports it as unpriced. Both axes are
    percentiles within the battery, so every reading is relative to which properties the caller
    chose and no KUI value is comparable across two different batteries. A battery of one is
    undefined and says so rather than returning 0.5 as though it had ranked something.
    """

    name = "KUI"
    version = "1.0"
    capabilities = Capability.ACTIVATIONS | Capability.LINEAR_READOUT
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A1"
    deviations = (
        "mediation defaults to the cheap linear proxy |cos(w_P, w_r)|; the production path uses the "
        "measured causal delta-r from steering",
        "both axes are standardized to percentile-within-battery before subtraction (the A1 unit-bug "
        "fix); a singleton battery is undefined and reported as such",
    )

    # -- the observable declarations ---------------------------------------
    quantity = "grader.knowledge_utilization"
    #: The reward direction is read off the head; the property battery, with its decodabilities and
    #: directions, is a recorded measurement from the concept layer.
    requires: AccessMatrix = {
        Component.GRADER: Access.FORWARD,
        Component.RECORD: Access.RECORD,
    }
    substrates = NEURAL_SUBSTRATES
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: The mediation proxy is a cosine between a property direction and the readout, which a shared
    #: orthogonal change of basis leaves unchanged, and the percentile step is monotone in it.
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = ("baseline.random_direction", "baseline.decodability_only")
    rung = 0
    #: A white-box reading owes an `IncrementalValidity` and this instrument cannot produce
    #: one. The id is checkable and the prose is the argument.
    incremental_exemption = (
        "NO_PER_ITEM_VERDICT",
        "the reading is a plane over a battery of properties, one point per property, and both of its "
        "axes are percentiles within that battery. There is no per-item verdict and no ground truth "
        "saying which properties should be represented-but-unpriced, so neither the index nor a "
        "baseline has an error vector. The battery is also injected, so the reading is relative to "
        "which properties the caller chose.",
    )

    def __init__(
        self,
        properties: Sequence[Property] | None = None,
        *,
        null_draws: int = 10000,
        seed: int = 0,
    ) -> None:
        self.properties = list(properties) if properties is not None else None
        self.null_draws = int(null_draws)
        self.seed = int(seed)

    def measure(self, ctx: Context) -> "Evidence":
        signal = ctx.signal
        w_r = reward_vector(signal, ctx.readout)

        if not self.properties:
            return ctx.emit(
                {
                    "kui": [],
                    "names": [],
                    "note": "no property battery available (none injected; concept probes pending)",
                },
                uncertainty=Uncertainty(method="none"),
            )
        if len(self.properties) < 2:
            return ctx.emit(
                {
                    "kui": [float("nan")],
                    "names": [self.properties[0].name],
                    "note": "KUI needs a battery of at least two properties to rank within",
                },
                uncertainty=Uncertainty(method="none"),
            )

        result = kui_from_properties(self.properties, w_r)

        # Read the mediation cosine against the random-direction null (the noise floor A1 must beat).
        from reward_lens.stats.nulls import random_direction_null

        null = random_direction_null(
            float(np.max(result["mediate_raw"])), d=int(w_r.size), n=self.null_draws, seed=self.seed
        )

        payload = {
            "names": result["names"],
            "kui": result["kui"].tolist(),
            "decode_pct": result["decode_pct"].tolist(),
            "mediate_pct": result["mediate_pct"].tolist(),
            "decode_raw": result["decode_raw"].tolist(),
            "mediate_raw": result["mediate_raw"].tolist(),
            "mediation_null_p95": null["null_p95"],
            "n_properties": len(self.properties),
        }
        return ctx.emit(payload, uncertainty=Uncertainty(n=len(self.properties), method="none"))


__all__ = [
    "Property",
    "linear_mediation_proxy",
    "kui_plane",
    "kui_from_properties",
    "KUI",
]

"""N8, noise and angle: the two numbers every reward component needs and no tooling separates.

    Congruity = cos(angle(M' alpha, B')),   Distortion = 1 - Congruity

The agent's first-order condition is `M' alpha = C'(t)`, so `M' alpha` is the direction the contract
pushes the agent in effort space, and `B'` is the direction that actually creates value. The cosine
between them is how much of the push is aimed at the thing you want.

**Why one number is not enough, and this is the whole content of the instrument.** A noisy but
congruent component is pointed the right way and is merely imprecise, so the fix is a lower weight
and the shrinkage factor says exactly how much lower. A precise but distorted component is pointed
the wrong way, and **no weight fixes an angle**: lowering it buys less of the wrong thing rather than
more of the right one, and the fix is a different measure. Those two failures call for opposite
actions and a single scalar cannot distinguish them, so every reward component needs a noise and an
angle. Nothing in current tooling reports the second.

**What a per-component angle is measured against, which the source does not say and which decides
whether the number means anything.** The contract-level congruity has a natural reference:
`cos(angle(M' alpha, B'))` can reach 1, so `1 - congruity` is a distortion. A single component's
cosine cannot. With `m` components each covering its own task, a clean component's direction is a
coordinate axis and its cosine to `B'` is `B_i / ||B'||`, which for `m` equally valued tasks is
`1/sqrt(m)` and never approaches 1 however good the component is. Judging a component against a
reference of 1 would call every component of every three-part composite distorted.

So the per-component reference is that clean-measure cosine, and

    reference_i = B_i / ||B'||,   distortion_i = reference_i - cos(angle(m_i, B'))

**A diagonal sensitivity matrix has a distortion of exactly zero for every component, by
construction.** That is a sharper statement of the degenerate case than a flag: distortion is real
exactly when the sensitivity matrix has off-diagonal structure, when working on one thing moves
another thing's score. A judge that rewards length while length is not what you value is the
canonical case, and it is an off-diagonal entry. On a diagonal `M` the reading still produces the
noise half of the pair, which is a measurement, and says plainly that the angle half is zero because
nothing was there to measure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BaselineID
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.decision._base import NOISE_ACCESS, DecisionInstrument
from reward_lens.measure.decision.assumptions import render_assumptions
from reward_lens.measure.decision.parameters import ContractParameters
from reward_lens.measure.decision.sorting import noise_to_signal
from reward_lens.measure.decision.weights import optimal_weights, shrinkage

N8_BASELINES: tuple[BaselineID, ...] = (
    "baseline.marginal_correlation",
    "baseline.raw_variance",
)

#: Below this shrinkage factor the noise term has more than halved the component's weight, which is
#: the boundary at which "this is noisy" stops being a description and starts being a decision. Not
#: a catalogue threshold: the catalogue carries no N8 record. Stated here, and a constructor
#: argument.
DEFAULT_SHRINKAGE_FLOOR = 0.5

#: Above this distortion a component's cross-response has cost it more than five cosine points
#: against what a clean own-task measure of the same thing would achieve, which is the boundary at
#: which "this measures something else as well" stops being a description and starts being a
#: decision. Chosen rather than derived, for the same reason and with the same escape hatch.
DEFAULT_DISTORTION_CEILING = 0.05


# ---------------------------------------------------------------------------
# The angle
# ---------------------------------------------------------------------------


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return math.nan
    return float(np.dot(u, v) / (nu * nv))


def congruity(
    sensitivity: np.ndarray,
    weights: Sequence[float] | np.ndarray,
    benefit: Sequence[float] | np.ndarray,
) -> float:
    """`cos(angle(M' alpha, B'))`: how much of what the contract pushes for is what you want.

    Returns NaN when either vector is zero, which is the honest answer: an angle to a zero vector is
    undefined rather than zero, and returning 1.0 there would report perfect alignment for a
    contract that incentivises nothing.
    """
    m_mat = np.asarray(sensitivity, dtype=np.float64)
    alpha = np.asarray(weights, dtype=np.float64).ravel()
    b = np.asarray(benefit, dtype=np.float64).ravel()
    return _cosine(m_mat.T @ alpha, b)


def component_congruity(
    sensitivity: np.ndarray, benefit: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Per component, `cos(angle(m_i, B'))` with `m_i` the i-th row of `M`.

    `m_i` is the gradient of component i's score in effort space, so it is the direction the agent
    moves in when it is paid for that component alone. Its angle to `B'` is that component's own
    congruity, independent of the weight it is given, which is what makes it the second number
    rather than a restatement of the first.
    """
    m_mat = np.asarray(sensitivity, dtype=np.float64)
    b = np.asarray(benefit, dtype=np.float64).ravel()
    return np.array([_cosine(m_mat[i, :], b) for i in range(m_mat.shape[0])], dtype=np.float64)


def clean_measure_congruity(benefit: Sequence[float] | np.ndarray) -> np.ndarray:
    """`B_i / ||B'||`: what component `i` would score if it responded only to its own task.

    The reference a per-component cosine has to be read against. It is not 1 and it cannot be: with
    `m` components covering `m` equally valued tasks the best any single one can do is `1/sqrt(m)`,
    so judging against 1 would call every component of every composite distorted.
    """
    b = np.asarray(benefit, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(b))
    if norm == 0.0:
        return np.full(b.size, math.nan)
    return b / norm


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------

#: The four cells of the noise-by-angle square, and what each one tells you to do.
VERDICTS: Mapping[tuple[bool, bool], str] = {
    (True, True): "keep: precise and congruent, and its weight is doing what it looks like",
    (False, True): "lower the weight: noisy but congruent, so the fix is intensity and the "
    "shrinkage factor says how much",
    (True, False): "replace the measure: precise and distorted, and no weight fixes an angle",
    (False, False): "drop it: noisy and distorted, so it buys imprecision aimed elsewhere",
}

#: What a row says when its angle is zero by construction, which is every row of a diagonal
#: sensitivity matrix. Half the pair is a measurement and half of it is arithmetic, and reporting a
#: verdict about an angle that could not have been anything else would be the instrument finding
#: something on every composite it is shown.
NOISE_ONLY_VERDICTS: Mapping[bool, str] = {
    True: "noise half only: precise, and the distortion is zero because M is diagonal",
    False: "noise half only: lower the weight, and the distortion is zero because M is diagonal",
}


@dataclass(frozen=True)
class NoiseAngleRow:
    """One component's pair: a noise and an angle, with the action each combination implies."""

    component: str
    noise: float
    shrinkage: float
    #: `cos(angle(m_i, B'))`, the source's object.
    congruity: float
    #: `B_i / ||B'||`, what this component would score if it responded only to its own task. The
    #: reference the cosine has to be read against, because a single component's cosine cannot
    #: reach 1 when the value direction spreads over several tasks.
    reference: float
    weight: float
    precise: bool
    congruent: bool
    #: False when the sensitivity matrix is diagonal, so `congruity` equals `reference` exactly and
    #: the distortion is zero by construction rather than by measurement.
    angle_is_informative: bool = True

    @property
    def distortion(self) -> float:
        """`reference - congruity`: what this component's cross-response costs it.

        Zero for a measure that responds only to its own task, positive when the cross-response
        points away from value, and negative in the case worth knowing about, where a component's
        response to other tasks happens to point at value better than its own task does.
        """
        return self.reference - self.congruity

    @property
    def verdict(self) -> str:
        if not self.angle_is_informative:
            return NOISE_ONLY_VERDICTS[self.precise]
        return VERDICTS[(self.precise, self.congruent)]

    def render(self) -> str:
        return (
            f"  {self.component:<16} n = {self.noise:>10.4g}  shrink = {self.shrinkage:>6.3f}  "
            f"cos = {self.congruity:>7.4f}  dist = {self.distortion:>7.4f}  {self.verdict}"
        )


@dataclass(frozen=True)
class NoiseAngleReading:
    """N8's reading: the pair per component, the contract's own angle, and the degeneracy check."""

    rows: tuple[NoiseAngleRow, ...]
    contract_congruity: float
    weights_are_optimal: bool
    diagonal_sensitivity: bool
    shrinkage_floor: float
    distortion_ceiling: float
    baselines: Mapping[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(r.component for r in self.rows)

    @property
    def contract_distortion(self) -> float:
        return 1.0 - self.contract_congruity

    @property
    def wants_lower_weight(self) -> tuple[str, ...]:
        """Noisy, and either congruent or on a reading where the angle says nothing either way."""
        return tuple(
            r.component
            for r in self.rows
            if not r.precise and (r.congruent or not r.angle_is_informative)
        )

    @property
    def wants_a_different_measure(self) -> tuple[str, ...]:
        """Distorted, and only where the angle is a measurement rather than a restatement of `B'`."""
        return tuple(r.component for r in self.rows if r.angle_is_informative and not r.congruent)

    def says(self) -> str:
        lower = self.wants_lower_weight
        different = self.wants_a_different_measure
        head = (
            f"The contract points {self.contract_congruity:.4f} of the way toward what you value, "
            f"so {self.contract_distortion:.1%} of the push is aimed elsewhere."
        )
        if self.diagonal_sensitivity:
            tail = (
                f" Lower the weight on {', '.join(lower)}: the noise term has more than halved it."
                if lower
                else " Every component clears the shrinkage floor."
            )
            return (
                f"{head} The sensitivity matrix is diagonal, so every component's distortion is "
                f"exactly zero by construction rather than by measurement: its cosine to B' is the "
                f"clean-measure reference B_i / ||B'||. The noise half of the pair is a "
                f"measurement; the angle half needs a sensitivity matrix with off-diagonal "
                f"structure to say anything.{tail}"
            )
        parts = [head]
        if lower:
            parts.append(f"Lower the weight on {', '.join(lower)}: noisy but congruent.")
        if different:
            parts.append(
                f"Replace {', '.join(different)}: no weight fixes an angle, so a lower weight buys "
                f"less of the wrong thing rather than more of the right one."
            )
        if not lower and not different:
            parts.append("Every component is both precise and congruent at the stated floors.")
        return " ".join(parts)

    def render(self) -> str:
        lines = [self.says(), ""]
        lines.extend(r.render() for r in self.rows)
        lines.append(
            f"  contract congruity cos(angle(M' alpha, B')) = {self.contract_congruity:.6f}, "
            f"computed at the "
            + ("optimal" if self.weights_are_optimal else "supplied")
            + " weights"
        )
        lines.append(
            f"  floors: shrinkage at or above {self.shrinkage_floor:.2g} is precise, distortion at "
            f"or below {self.distortion_ceiling:.2g} is congruent. Both are this module's own and "
            f"both are arguments; the numbers above are floors applied to measurements, not "
            f"measurements."
        )
        lines.extend(f"  {n}" for n in self.notes)
        lines.append("")
        lines.append(render_assumptions())
        return "\n".join(lines)


def noise_and_angle(
    params: ContractParameters,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    shrinkage_floor: float = DEFAULT_SHRINKAGE_FLOOR,
    distortion_ceiling: float = DEFAULT_DISTORTION_CEILING,
) -> NoiseAngleReading:
    """The pair per component, from a settled parameter set."""
    if params.sensitivity is None:
        raise ValueError(
            "noise_and_angle needs the sensitivity matrix. The instrument refuses before reaching "
            "here; this path is for a direct caller who skipped it."
        )
    m_mat = params.sensitivity
    mu = np.diag(m_mat).astype(np.float64)
    n = noise_to_signal(np.diag(params.noise).astype(np.float64), mu)
    cos = component_congruity(m_mat, params.benefit)
    reference = clean_measure_congruity(params.benefit)
    if weights is None:
        alpha = optimal_weights(params)
        optimal = True
    else:
        alpha = np.asarray(weights, dtype=np.float64).ravel()
        optimal = False
        if alpha.size != params.m:
            raise ValueError(f"{alpha.size} weight(s) for {params.m} component(s)")

    diagonal = bool(np.allclose(m_mat, np.diag(mu), atol=1e-12))
    rows = []
    for i in range(params.m):
        shrink = shrinkage(
            params.risk_aversion,
            float(params.cost_curvature[i, i]),
            float(params.noise[i, i]),
        )
        rows.append(
            NoiseAngleRow(
                component=params.components[i],
                noise=float(n[i]),
                shrinkage=float(shrink),
                congruity=float(cos[i]),
                reference=float(reference[i]),
                weight=float(alpha[i]),
                precise=bool(shrink >= shrinkage_floor),
                congruent=bool(float(reference[i]) - float(cos[i]) <= distortion_ceiling),
                angle_is_informative=not diagonal,
            )
        )

    notes: list[str] = []
    if diagonal:
        notes.append(
            "the sensitivity matrix is diagonal, so every per-component cosine above is exactly "
            "the clean-measure reference B_i / ||B'|| and every distortion is exactly zero. That "
            "is arithmetic rather than a measurement. Distortion becomes measurable when working "
            "on one thing moves another thing's score, which is an off-diagonal entry of M."
        )
    if np.any(mu < 0.0):
        flipped = [params.components[i] for i in np.flatnonzero(mu < 0.0)]
        notes.append(
            f"{', '.join(flipped)} have a negative own-sensitivity, so more effort lowers their "
            f"score. Their congruity is negative by construction and the sign is the finding."
        )

    b = params.benefit
    if float(np.linalg.norm(b)) == 0.0:
        notes.append(
            "B' is the zero vector, so every angle to it is undefined and every congruity below "
            "is NaN. That is the honest value: an angle to a direction that is not a direction."
        )
    return NoiseAngleReading(
        rows=tuple(rows),
        contract_congruity=congruity(m_mat, alpha, b),
        weights_are_optimal=optimal,
        diagonal_sensitivity=diagonal,
        shrinkage_floor=float(shrinkage_floor),
        distortion_ceiling=float(distortion_ceiling),
        baselines={
            "baseline.marginal_correlation": float(np.mean(np.abs(cos))),
            "baseline.raw_variance": float(np.mean(np.diag(params.noise))),
        },
        notes=tuple(notes),
    )


class NoiseAndAngle(DecisionInstrument):
    """N8. A noise and an angle for every reward component, because one number cannot say both.

    Kill condition: if every real composite tested has a diagonal sensitivity matrix, congruity
    carries no information beyond `B'` and only the noise half of the pair ships.
    """

    name = "NoiseAndAngle"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "N8"
    deviations = (
        "the source gives the whole-contract congruity cos(angle(M' alpha, B')) and asserts "
        "that every component needs two numbers, without saying what the per-component angle is. "
        "It is taken here as the cosine between the i-th row of M, the gradient of that "
        "component's score in effort space, and B'. That makes it independent of the weight, which "
        "is what a second number has to be to add anything to the first",
        "a single component's cosine to B' cannot reach 1 when the value direction spreads over "
        "several tasks: a clean own-task measure reaches B_i / ||B'||, which is 1/sqrt(m) for m "
        "equally valued tasks. So distortion is measured against that reference rather than "
        "against 1, and the consequence is exact: a diagonal sensitivity matrix has a distortion "
        "of zero for every component, by construction. The contract-level distortion stays the "
        "source's 1 - congruity, which does have 1 as its reference",
        "the precise and congruent verdicts are floors applied to measurements, not measurements. "
        "Both floors are this module's own, both are constructor arguments, and both numbers "
        "travel on every reading so a reader can apply different ones without recomputing",
        "the catalogue carries no N8 record and no registered quantity rows. "
        "`quantities.as_catalogue_rows()` emits the proposed record",
    )

    quantity = "reward.component_congruity"
    #: The other quantity this instrument reports. `Instrument.quantity` is singular and the whole
    #: point of N8 is that a component needs two numbers, so the second is declared here and the
    #: payload carries both.
    also_reports: tuple[str, ...] = ("reward.component_noise",)
    requires = NOISE_ACCESS
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = N8_BASELINES
    rung = 1

    def __init__(
        self,
        parameters: ContractParameters | None = None,
        *,
        weights: Sequence[float] | np.ndarray | None = None,
        shrinkage_floor: float = DEFAULT_SHRINKAGE_FLOOR,
        distortion_ceiling: float = DEFAULT_DISTORTION_CEILING,
    ) -> None:
        self.parameters = parameters
        self.weights = weights
        self.shrinkage_floor = float(shrinkage_floor)
        self.distortion_ceiling = float(distortion_ceiling)

    def compute(self) -> Any:
        if self.parameters is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    "no contract parameters were supplied, so there is no component to pair a "
                    "noise with an angle for"
                ),
                remedy=(
                    "build a `ContractParameters` for the composite. The noise half needs A2's "
                    "variance components; the angle half needs the sensitivity matrix, and it only "
                    "says anything when that matrix has off-diagonal structure."
                ),
            )
        bad = self.parameters.refuse_unstated(
            self.name, ["benefit", "cost_curvature", "noise", "risk_aversion", "sensitivity"]
        )
        if bad is not None:
            return bad
        return noise_and_angle(
            self.parameters,
            weights=self.weights,
            shrinkage_floor=self.shrinkage_floor,
            distortion_ceiling=self.distortion_ceiling,
        )

    def payload(self, computed: NoiseAngleReading) -> dict[str, Any]:
        return {
            "components": list(computed.components),
            "rows": [
                {
                    "component": r.component,
                    "noise": r.noise,
                    "shrinkage": r.shrinkage,
                    "congruity": r.congruity,
                    "reference": r.reference,
                    "distortion": r.distortion,
                    "weight": r.weight,
                    "precise": r.precise,
                    "congruent": r.congruent,
                    "angle_is_informative": r.angle_is_informative,
                    "verdict": r.verdict,
                }
                for r in computed.rows
            ],
            "contract_congruity": computed.contract_congruity,
            "contract_distortion": computed.contract_distortion,
            "diagonal_sensitivity": computed.diagonal_sensitivity,
            "weights_are_optimal": computed.weights_are_optimal,
            "wants_lower_weight": list(computed.wants_lower_weight),
            "wants_a_different_measure": list(computed.wants_a_different_measure),
            "shrinkage_floor": computed.shrinkage_floor,
            "distortion_ceiling": computed.distortion_ceiling,
            "notes": list(computed.notes),
            "baselines": dict(computed.baselines),
            "says": computed.says(),
        }


__all__ = [
    "DEFAULT_DISTORTION_CEILING",
    "DEFAULT_SHRINKAGE_FLOOR",
    "N8_BASELINES",
    "NOISE_ONLY_VERDICTS",
    "VERDICTS",
    "NoiseAndAngle",
    "NoiseAngleReading",
    "NoiseAngleRow",
    "clean_measure_congruity",
    "component_congruity",
    "congruity",
    "noise_and_angle",
]

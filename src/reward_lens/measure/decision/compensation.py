"""N6, the equal-compensation table: which component of your composite reward is being starved.

The principle is Milgrom and Roberts (1992), not Holmstrom and Milgrom (1991). The phrase "equal
compensation principle" does not appear in the 1991 paper and citing it there is a citation error
this module declines to repeat.

**The derivation, because the source's statement of it is inverted.** Let the agent hold a pool of
capacity it can spend on any task, so its cost depends on the total: `c(sum_j t_j)`. Its
payoff under a composite `sum_i alpha_i x_i` is `sum_i alpha_i mu_i(t_i) - c(T)`, and the first-order
condition on task `i` is

    alpha_i mu'_i = c'(T)

The right-hand side is one scalar shared by every task. So every task the agent touches must offer
the same marginal return per unit of capacity, and the quantity that must be equal across components
is the **commission** `kappa_i = alpha_i mu'_i`. The weights themselves must therefore satisfy
`alpha_i` proportional to `1/mu'_i`, and the component that gets starved is the one with the
**lowest** `alpha_i mu'_i`, because capacity flows to wherever the marginal return is highest and a
task offering less than the others receives none of it once effort is fungible.

The source states the first half correctly, that after dividing each signal by its own sensitivity
the commissions must be equal, and then draws the reciprocal conclusion: it prints
`alpha_i` proportional to `mu'_i` and names the starved component as the one with the lowest
`alpha_i / mu'_i`. Dividing signal `x_i` by `mu'_i` produces a signal with unit sensitivity whose
commission is `alpha_i mu'_i`, so the two halves of that sentence contradict each other and the
second half is the one that is wrong. This module implements the derivation and `deviations` records
the disagreement, because a table that named the wrong component as starved would be exactly the
confident wrong number the library exists to refuse.

**One number every practitioner has, and one they do not.** The weights are in the config file. The
sensitivity `mu'_i`, the amount a component's score moves per unit of capacity spent on it, is a
dose-response slope and it takes a weight sweep to measure: perturb one component's weight, let the
policy re-equilibrate, and record how far that component's score moved. A static record of scores
does not carry it and no amount of re-reading the record produces it, which is why the refusal when
it is missing is `RECORD_INCOMPLETE` rather than `ACCESS_INSUFFICIENT`: the fix is upstream in
whatever produced the run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BaselineID
from reward_lens.core.reading import Refusal, RefusalReason, refuse_incomplete
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.decision._base import WEIGHTS_ACCESS, DecisionInstrument
from reward_lens.measure.decision.assumptions import render_assumptions
from reward_lens.measure.decision.parameters import ContractParameters, ParameterSource

N6_BASELINES: tuple[BaselineID, ...] = (
    "baseline.weights_as_commissions",
    "baseline.equal_weights",
)

#: How far the commissions may spread before the composite counts as mis-weighted. Not a catalogue
#: threshold: the catalogue carries no N6 record. It is this module's declared floor and it is an
#: argument so a caller can state a different one. A ratio of 1.0 is exact equal compensation; 0.9
#: says the worst-paid component pays at least ninety percent of what the best-paid one does, which
#: is inside the resolution of any sensitivity anybody will actually measure.
DEFAULT_STARVATION_FLOOR = 0.9


@dataclass(frozen=True)
class CompensationRow:
    """One component's line in the table."""

    component: str
    weight: float
    sensitivity: float
    commission: float
    ratio: float
    equalising_weight: float
    starved: bool

    def render(self) -> str:
        mark = "  STARVED" if self.starved else ""
        return (
            f"  {self.component:<16} {self.weight:>8.4g} {self.sensitivity:>10.4g} "
            f"{self.commission:>12.4g} {self.ratio:>7.3f} {self.equalising_weight:>10.4g}{mark}"
        )


@dataclass(frozen=True)
class CompensationTable:
    """N6's reading: the commissions, the starved component, and the reweighting that levels them."""

    rows: tuple[CompensationRow, ...]
    #: `M' alpha`: the agent's marginal return per unit of effort on each *task*, which is what the
    #: theorem is actually about. Equal to the per-signal commissions when M is diagonal.
    task_returns: np.ndarray
    sensitivity_source: ParameterSource
    sensitivity_note: str
    starvation_floor: float
    diagonal_sensitivity: bool
    baselines: Mapping[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(r.component for r in self.rows)

    @property
    def commissions(self) -> np.ndarray:
        return np.array([r.commission for r in self.rows], dtype=np.float64)

    @property
    def ratios(self) -> np.ndarray:
        return np.array([r.ratio for r in self.rows], dtype=np.float64)

    @property
    def starved(self) -> CompensationRow:
        """The component with the lowest commission. Always defined; `holds` says whether it matters."""
        return min(self.rows, key=lambda r: r.commission)

    @property
    def spread(self) -> float:
        """`min(kappa)/max(kappa)`: 1.0 is exact equal compensation, 0 is total starvation."""
        return float(np.min(self.ratios))

    @property
    def holds(self) -> bool:
        return self.spread >= self.starvation_floor

    @property
    def rung(self) -> int:
        """1 when the sensitivity was measured, 0 when it was stated or assumed."""
        return 1 if self.sensitivity_source is ParameterSource.MEASURED else 0

    def says(self) -> str:
        s = self.starved
        if self.holds:
            return (
                f"The commissions are equal to within {1.0 - self.spread:.1%}, so no component is "
                f"being starved: the lowest-paying component, {s.component}, pays "
                f"{s.ratio:.1%} of what the best-paying one does per unit of capacity."
            )
        return (
            f"{s.component} is the starved component. It pays the policy {s.ratio:.1%} of what the "
            f"best-paying component pays for the same unit of capacity, so its stated weight of "
            f"{s.weight:.4g} is not the incentive it delivers. Equal compensation would put it at "
            f"{s.equalising_weight:.4g}."
        )

    def render(self) -> str:
        lines = [
            self.says(),
            "",
            "  component          weight      mu'     commission   ratio   equalising",
        ]
        lines.extend(r.render() for r in self.rows)
        lines.append(
            f"  sensitivity: {self.sensitivity_source.name.lower()}"
            + (f". {self.sensitivity_note}" if self.sensitivity_note else "")
        )
        if not self.diagonal_sensitivity:
            lines.append(
                "  the sensitivity matrix is not diagonal, so a component's score responds to "
                "effort on tasks other than its own. The per-signal commissions above are then a "
                "summary and the task returns below are what the principle is about:"
            )
            lines.append(
                "  task returns (M' alpha): "
                + ", ".join(f"{c}={v:.4g}" for c, v in zip(self.components, self.task_returns))
            )
        lines.extend(f"  {n}" for n in self.notes)
        lines.append("")
        lines.append(render_assumptions())
        return "\n".join(lines)


def equal_compensation(
    weights: Sequence[float] | np.ndarray,
    params: ContractParameters,
    *,
    starvation_floor: float = DEFAULT_STARVATION_FLOOR,
) -> CompensationTable:
    """The table, from a weight vector and a sensitivity matrix whose provenance is settled."""
    alpha = np.asarray(weights, dtype=np.float64).ravel()
    if alpha.size != params.m:
        raise ValueError(
            f"{alpha.size} weight(s) for {params.m} component(s). The table is keyed by component "
            f"name and a length mismatch would silently pair the wrong ones."
        )
    if params.sensitivity is None:
        raise ValueError(
            "equal_compensation needs the sensitivity matrix. The instrument refuses before "
            "reaching here; this path is for a direct caller who skipped it."
        )
    m_mat = params.sensitivity
    mu = np.diag(m_mat).astype(np.float64)
    zero = [params.components[i] for i in np.flatnonzero(mu == 0.0)]
    if zero:
        raise ValueError(
            f"component(s) {', '.join(zero)} have a dose-response slope of exactly zero, so no "
            f"weight on them produces any incentive at all and the equalising weight is infinite. "
            f"A component the policy cannot move is a finding rather than a row in this table."
        )
    kappa = alpha * mu
    peak = float(np.max(kappa))
    if peak == 0.0:
        raise ValueError(
            "every commission is zero, so the composite offers no incentive on any component and "
            "there is nothing to equalise."
        )
    ratios = kappa / peak
    total = float(np.sum(alpha))
    inv = 1.0 / mu
    equalising = total * inv / float(np.sum(inv)) if float(np.sum(inv)) != 0.0 else inv * math.nan
    starved_idx = int(np.argmin(kappa))

    diagonal = bool(np.allclose(m_mat, np.diag(mu), atol=1e-12))
    notes: list[str] = []
    if params.source.get("sensitivity") is not ParameterSource.MEASURED:
        notes.append(
            "the sensitivity was not measured, so this table is a conditional statement: it says "
            "which component is starved if the slopes are what was stated. Measuring them takes a "
            "weight sweep and changes the rung rather than the arithmetic."
        )
    return CompensationTable(
        rows=tuple(
            CompensationRow(
                component=params.components[i],
                weight=float(alpha[i]),
                sensitivity=float(mu[i]),
                commission=float(kappa[i]),
                ratio=float(ratios[i]),
                equalising_weight=float(equalising[i]),
                starved=(i == starved_idx),
            )
            for i in range(params.m)
        ),
        task_returns=m_mat.T @ alpha,
        sensitivity_source=params.source.get("sensitivity", ParameterSource.UNKNOWN),
        sensitivity_note=params.note.get("sensitivity", ""),
        starvation_floor=float(starvation_floor),
        diagonal_sensitivity=diagonal,
        baselines={
            "baseline.weights_as_commissions": float(np.min(alpha) / np.max(alpha))
            if float(np.max(alpha)) != 0.0
            else math.nan,
            "baseline.equal_weights": 1.0,
        },
        notes=tuple(notes),
    )


class EqualCompensation(DecisionInstrument):
    """N6. Which component of a composite reward the policy is being paid least to work on.

    Kill condition: if the commission ratio is within a few percent of 1 for every component of
    every real composite tested, composites are already equal-compensating and the table is a
    formality.
    """

    name = "EqualCompensation"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "N6"
    deviations = (
        "the source states the conclusion of the equal-compensation principle as alpha_i "
        "proportional to mu'_i, and names the starved component as the one with the lowest "
        "alpha_i / mu'_i. Both are the reciprocal of what its own preceding clause implies and of "
        "what the agent's first-order condition gives, which is that alpha_i mu'_i must be equal "
        "across components. This module implements the derivation, so its starved component is the "
        "one with the lowest alpha_i mu'_i and its equalising weights are proportional to 1/mu'_i",
        "the per-signal commission alpha_i mu'_i is exact only when the sensitivity matrix is "
        "diagonal. When a component's score responds to effort on other tasks, the quantity the "
        "principle constrains is the task-level marginal return (M' alpha)_j, which is reported "
        "beside the table and named as the one to read",
        "the catalogue carries no N6 record and no registered quantity row. "
        "`quantities.as_catalogue_rows()` emits the proposed record",
        "the starvation floor of 0.9 is this module's own, not a catalogue threshold. It exists so "
        "that `holds` is a verdict rather than a float comparison against zero, and every reading "
        "carries the spread so a reader can apply a different floor without recomputing",
    )

    quantity = "reward.equal_compensation_ratio"
    requires = WEIGHTS_ACCESS
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = N6_BASELINES
    rung = 0

    def __init__(
        self,
        weights: Sequence[float] | np.ndarray | None = None,
        parameters: ContractParameters | None = None,
        *,
        record: str = "",
        starvation_floor: float = DEFAULT_STARVATION_FLOOR,
    ) -> None:
        self.weights = weights
        self.parameters = parameters
        #: What the weights were read off, when they came from a record. Naming it turns the missing
        #: sensitivity from "you need more access" into "this run did not record a weight sweep",
        #: which is a different remedy and a different refusal reason.
        self.record = record
        self.starvation_floor = float(starvation_floor)
        if parameters is not None and parameters.source.get("sensitivity") is (
            ParameterSource.MEASURED
        ):
            self.rung = 1

    def compute(self) -> Any:
        if self.weights is None or self.parameters is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    "no composite weights or no parameter set were supplied, so there is no "
                    "composite to check for equal compensation"
                ),
                remedy=(
                    "pass the composite's weights and a `ContractParameters` naming its "
                    "components. The weights are whatever is in the reward config; nothing has to "
                    "be measured to get them."
                ),
            )
        if self.parameters.source.get("sensitivity") is ParameterSource.UNKNOWN:
            if self.record:
                return refuse_incomplete(
                    self.name,
                    field="a weight sweep, from which the dose-response slope mu' is read",
                    subject=self.record,
                    remedy=(
                        "re-run with a weight sweep: perturb one component's weight, let the "
                        "policy re-equilibrate, and record how far that component's score moved. "
                        "Nothing in this record recovers it, because a slope is a response to a "
                        "change that was never made. If you have a slope from elsewhere, state it "
                        "on the parameter set and the table becomes a conditional reading at "
                        "rung 0."
                    ),
                    components=list(self.parameters.components),
                )
            bad = self.parameters.refuse_unstated(self.name, ["sensitivity"])
            if bad is not None:
                return bad
        return equal_compensation(
            self.weights, self.parameters, starvation_floor=self.starvation_floor
        )

    def payload(self, computed: CompensationTable) -> dict[str, Any]:
        return {
            "components": list(computed.components),
            "rows": [
                {
                    "component": r.component,
                    "weight": r.weight,
                    "sensitivity": r.sensitivity,
                    "commission": r.commission,
                    "ratio": r.ratio,
                    "equalising_weight": r.equalising_weight,
                    "starved": r.starved,
                }
                for r in computed.rows
            ],
            "starved": computed.starved.component,
            "spread": computed.spread,
            "holds": computed.holds,
            "starvation_floor": computed.starvation_floor,
            "task_returns": [float(v) for v in computed.task_returns],
            "diagonal_sensitivity": computed.diagonal_sensitivity,
            "sensitivity_source": computed.sensitivity_source.name,
            "rung": computed.rung,
            "notes": list(computed.notes),
            "baselines": dict(computed.baselines),
            "says": computed.says(),
        }


__all__ = [
    "DEFAULT_STARVATION_FLOOR",
    "N6_BASELINES",
    "CompensationRow",
    "CompensationTable",
    "EqualCompensation",
    "equal_compensation",
]

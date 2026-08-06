"""N5, the optimal component weights: how hard to optimise, derived rather than tuned.

Holmstrom and Milgrom (1991) solve the multi-task principal-agent problem for linear contracts under
CARA utility and Brownian noise, and the solution is

    alpha* = (I + r C'' Sigma)^-1 B'

whose diagonal case is `alpha_i = B_i / (1 + r C_ii sigma_i^2)`. **The optimal weight on a reward
component is its true marginal value divided by one plus (risk aversion times effort-cost curvature
times that component's noise variance).** The noisier the grader component, the lower its optimal
weight, with a functional form rather than a hand wave. Nobody in reinforcement learning from human
feedback sets a component's weight from that component's measured noise, and this is the instrument
that does.

Note the ordering. `C''` comes before `Sigma`. Matrix products do not commute and at least one
transcription in this project's own corpus has them the other way round; `test_measure_decision.py`
computes a worked example by hand where transposing the product triples the recommended weight on one
of two components.

Note also the structural identity nobody has remarked on: the shrinkage factor
`1/(1 + r C'' sigma^2)` is the same object as the regressional-Goodhart factor
`Var(X)/(Var(X) + Var(Z))`. Contract theory derives it as what the principal should do; the Goodhart
scaling literature derives it as what happens if the principal does not. One object, two directions
of use.

**Two results ship on top of the base formula and they are the reason this module exists.**

*The unmeasurable-task correction.* With one task that carries no signal at all,

    alpha*_1 = [B_1 - B_2 (C_12/C_22)] / [1 + r sigma_1^2 (C_11 - C_12^2/C_22)],  alpha*_2 = 0

The numerator subtracts the value you destroy on the unmeasured task by pulling effort off it, so
**the presence of something you cannot measure should make you turn down the gain on the thing you
can.** Nothing in current practice does this.

*The zero-weight theorem.* When effort is perfectly substitutable across the two tasks,
`C_11 = C_12 = C_22`, the numerator becomes `B_1 - B_2` and the denominator becomes exactly 1,
because the Schur complement `C_11 - C_12^2/C_22` is exactly zero. Two equally valuable tasks, one
measurable and one not, with fungible capacity, means the optimal weight on the measurable one is
**exactly zero**. Not small. Zero. And it is worse than that: the principal's surplus at any nonzero
weight diverges downward as the tasks approach perfect substitutability, so any positive power on the
clever signal is not merely suboptimal but unboundedly so. That is the formal core of every smart
reward signal that led to policy collapse.

Everything here is numpy and closed-form linear algebra. No torch, no model, no I/O, no training run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from reward_lens.core.invariance import Relation
from reward_lens.core.quantity import BaselineID
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.decision._base import NOISE_ACCESS, DecisionInstrument
from reward_lens.measure.decision.assumptions import render_assumptions
from reward_lens.measure.decision.parameters import (
    ContractParameters,
    ParameterSource,
    Sweep,
)

#: A weight is a price per unit of signal, so rescaling the signal by `a` must divide the weight by
#: `a`. That is a covariant relation with weight -1, and it is the only reading in this package that
#: is not exactly invariant under `reward.affine`.
COVARIANT_INVERSE = Relation("covariant", weight=-1.0)

#: Above this condition number of `C''` the tasks are close enough to perfect substitutes that the
#: agent's effort response is numerically unbounded, and a weight quoted without saying so is
#: hiding the fact that a tiny change in the contract moves the agent arbitrarily far. Not a
#: catalogue threshold: the catalogue carries no N5 record. It is this module's declared floor, and
#: it is a constructor argument so a caller can state a different one.
DEFAULT_CONDITION_LIMIT = 1e12

N5_BASELINES: tuple[BaselineID, ...] = (
    "baseline.equal_weights",
    "baseline.value_weights",
)


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


def shrinkage(risk_aversion: float, cost_curvature: float, noise_variance: float) -> float:
    """`1/(1 + r C sigma^2)`, the scalar the whole layer turns on.

    The same object as the regressional-Goodhart factor `Var(X)/(Var(X) + Var(Z))`, which is worth
    checking rather than asserting: put `Var(X) = 1/(r C)` and `Var(Z) = sigma^2` and the two
    expressions are identical. The identity is exercised in the unit tests.
    """
    denom = 1.0 + float(risk_aversion) * float(cost_curvature) * float(noise_variance)
    if denom <= 0.0:
        raise ValueError(
            f"the shrinkage denominator is {denom:.6g}, which is not positive. That needs a "
            f"negative risk aversion, a non-convex cost or a negative variance, and none of the "
            f"three is inside this model."
        )
    return 1.0 / denom


def optimal_weights_diagonal(
    benefit: Sequence[float] | np.ndarray,
    cost_diagonal: Sequence[float] | np.ndarray,
    noise_diagonal: Sequence[float] | np.ndarray,
    risk_aversion: float,
) -> np.ndarray:
    """`alpha_i = B_i / (1 + r C_ii sigma_i^2)`, the diagonal case, written out.

    Kept as its own function rather than folded into the general solve, because it is the form the
    source states and the form a reader recognises, and having both lets a test assert they
    agree wherever both apply. They do, to machine precision.
    """
    b = np.asarray(benefit, dtype=np.float64).ravel()
    c = np.asarray(cost_diagonal, dtype=np.float64).ravel()
    s = np.asarray(noise_diagonal, dtype=np.float64).ravel()
    if not (b.size == c.size == s.size):
        raise ValueError(f"shapes disagree: B' {b.size}, C'' {c.size}, Sigma {s.size}")
    return np.array(
        [bi * shrinkage(risk_aversion, ci, si) for bi, ci, si in zip(b, c, s)], dtype=np.float64
    )


def _quadratic_form(params: ContractParameters) -> tuple[np.ndarray, np.ndarray]:
    """`(g, H)` for the principal's surplus `S(alpha) = g'alpha - alpha'H alpha / 2`.

    `g = M C''^-1 B'` and `H = M C''^-1 M' + r Sigma`. The optimum is `H^-1 g`, which is the general
    form of the formula and reduces to `(I + r C'' Sigma)^-1 B'` at `M = I`.
    """
    if params.sensitivity is None:
        raise ValueError(
            "the general form needs the sensitivity matrix M. Call `assume_unit_sensitivity()` to "
            "set M = I with the normalisation recorded on the reading, or supply a measured M."
        )
    m_mat = params.sensitivity
    inv_cb = np.linalg.solve(params.cost_curvature, params.benefit)
    inv_cm = np.linalg.solve(params.cost_curvature, m_mat.T)
    g = m_mat @ inv_cb
    h = m_mat @ inv_cm + params.risk_aversion * params.noise
    # H is symmetric by construction; symmetrise to kill the last bit of round-off so the solve and
    # the surplus evaluation cannot disagree in the last place.
    return g, 0.5 * (h + h.T)


def optimal_weights(params: ContractParameters) -> np.ndarray:
    """`alpha* = [M C''^-1 M' + r Sigma]^-1 M C''^-1 B'`, the general form.

    Unmeasurable tasks are held at exactly zero rather than solved for, because a task with no
    signal has no weight to set: the zero is a fact about the contract space and not the output of
    an optimisation, and computing it would invite reading it as one.
    """
    g, h = _quadratic_form(params)
    free = list(params.measurable)
    alpha = np.zeros(params.m, dtype=np.float64)
    if not free:
        return alpha
    sub_h = h[np.ix_(free, free)]
    sub_g = g[free]
    alpha[free] = np.linalg.solve(sub_h, sub_g)
    return alpha


def noiseless_weights(params: ContractParameters) -> np.ndarray:
    """`alpha*` at `Sigma = 0`, which works out to `M^-T B'`. The recommendation with the noise
    term deleted.

    This is the baseline the whole layer argues with, because it is what weighting a composite by
    how much you care about each part produces. At `M = I` it is `B'` itself, which is what the
    diagonal formula's numerator makes obvious; at a general `M` it is `B'` converted into the units
    each signal happens to be reported in, which is the part practice gets wrong before it gets to
    the noise.
    """
    m_mat = params.sensitivity
    if m_mat is None:
        raise ValueError("the noiseless optimum needs the sensitivity matrix")
    cond = float(np.linalg.cond(m_mat))
    if not math.isfinite(cond) or cond > 1.0 / np.finfo(np.float64).eps:
        return np.full(params.m, math.nan)
    out = np.linalg.solve(m_mat.T, params.benefit)
    for i in params.unmeasurable:
        out[i] = 0.0
    return out


def principal_surplus(alpha: Sequence[float] | np.ndarray, params: ContractParameters) -> float:
    """The principal's certainty equivalent at a given weight vector.

    `S(alpha) = B' t - C(t) - (r/2) alpha' Sigma alpha` with the agent's best response
    `t = C''^-1 M' alpha` substituted in. It is what `optimal_weights` maximises, so evaluating it
    at the recommendation and at each baseline is how the recommendation earns the word optimal
    instead of asserting it.
    """
    a = np.asarray(alpha, dtype=np.float64).ravel()
    g, h = _quadratic_form(params)
    return float(g @ a - 0.5 * a @ h @ a)


# ---------------------------------------------------------------------------
# The unmeasurable task, and the zero-weight theorem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnmeasurableCorrection:
    """The two-task reading: what one thing you cannot measure does to the one you can."""

    measured: str
    unmeasured: str
    benefit_measured: float
    benefit_unmeasured: float
    c11: float
    c12: float
    c22: float
    noise_variance: float
    risk_aversion: float
    #: `C_12/C_22`: how much effort leaves the unmeasured task per unit added to the measured one.
    substitution: float
    #: `C_11 - C_12^2/C_22`, the Schur complement. Zero at perfect substitutability.
    schur: float
    numerator: float
    denominator: float
    weight: float
    #: `B_1/(1 + r C_11 sigma_1^2)`: the answer you get by pretending the unmeasured task is not
    #: there. This is what current practice computes when it computes anything.
    weight_ignoring: float
    #: True when the numerator is exactly zero, so the optimum is zero rather than small.
    zero_weight: bool
    #: True when the Schur complement is zero and the numerator is not, which is a knife edge with
    #: unbounded surplus rather than an optimum.
    degenerate: bool

    @property
    def discount(self) -> float:
        """How much of the naive weight the unmeasured task takes away, as a fraction."""
        if self.weight_ignoring == 0.0:
            return 0.0
        return 1.0 - self.weight / self.weight_ignoring

    def says(self) -> str:
        if self.degenerate:
            return (
                f"The two tasks are perfect substitutes and their values differ, so the agent's "
                f"effort response is unbounded and there is no interior optimum. "
                f"alpha*({self.measured}) = {self.weight:.6g} is the knife edge where the marginal "
                f"social value of shifting effort is zero, not a maximum of a bounded surplus."
            )
        if self.zero_weight:
            return (
                f"The optimal weight on {self.measured} is exactly zero, not small. "
                f"{self.unmeasured} is unmeasurable, effort substitutes between the two at a rate "
                f"of {self.substitution:.6g}, and B_1 = B_2 * (C_12/C_22) exactly, so every unit of "
                f"incentive on {self.measured} destroys as much value on {self.unmeasured} as it "
                f"creates. Any positive weight is worse than none."
            )
        return (
            f"Weight {self.measured} at {self.weight:.4g}, not {self.weight_ignoring:.4g}. "
            f"{self.unmeasured} cannot be measured and effort substitutes between them, so the "
            f"presence of the thing you cannot measure takes {self.discount:.1%} off the gain on "
            f"the thing you can. alpha*({self.unmeasured}) = 0, by construction rather than by "
            f"arithmetic: there is no signal to weight."
        )

    def render(self) -> str:
        return "\n".join(
            [
                self.says(),
                f"  B' = ({self.benefit_measured:.6g}, {self.benefit_unmeasured:.6g}), "
                f"C'' = [[{self.c11:.6g}, {self.c12:.6g}], [{self.c12:.6g}, {self.c22:.6g}]], "
                f"sigma_1^2 = {self.noise_variance:.6g}, r = {self.risk_aversion:.6g}",
                f"  substitution C_12/C_22 = {self.substitution:.6g}; "
                f"Schur complement C_11 - C_12^2/C_22 = {self.schur:.6g}",
                f"  numerator = {self.numerator:.6g}, denominator = {self.denominator:.6g}",
            ]
        )


def unmeasurable_correction(
    *,
    benefit: Sequence[float],
    cost_curvature: Sequence[Sequence[float]] | np.ndarray,
    noise_variance: float,
    risk_aversion: float,
    names: tuple[str, str] = ("measured", "unmeasured"),
    tol: float = 0.0,
) -> UnmeasurableCorrection:
    """The two-task solution with task 2 carrying no signal.

    Derived from the same model as the matrix formula rather than quoted. The agent's two
    first-order conditions are `alpha_1 = C_1(t)` and `0 = C_2(t)`, so differentiating the pair
    gives `dt_1/dalpha_1 = 1/(C_11 - C_12^2/C_22)` and
    `dt_2/dalpha_1 = -(C_12/C_22) / (C_11 - C_12^2/C_22)`. Substituting those into the principal's
    first-order condition and cancelling the Schur complement produces the stated form.

    ``tol`` is the absolute tolerance at which the numerator counts as exactly zero. It defaults to
    zero, which means the zero-weight verdict is a bit-exact fact about the arithmetic rather than a
    judgement call; pass a small positive value to ask whether a real composite is near the
    boundary instead of on it.
    """
    b = np.asarray(benefit, dtype=np.float64).ravel()
    c = np.asarray(cost_curvature, dtype=np.float64)
    if b.size != 2 or c.shape != (2, 2):
        raise ValueError(
            f"the unmeasurable-task correction is the two-task case; got B' of size {b.size} and "
            f"C'' of shape {c.shape}. For more tasks, mark them in "
            f"`ContractParameters.unmeasurable` and use the matrix form."
        )
    c11, c12, c22 = float(c[0, 0]), float(c[0, 1]), float(c[1, 1])
    if c22 <= 0.0:
        raise ValueError(
            f"C_22 = {c22:.6g}. The unmeasured task has no effort cost curvature, so its "
            f"first-order condition does not pin its effort and the correction is undefined."
        )
    substitution = c12 / c22
    schur = c11 - (c12 * c12) / c22
    numerator = float(b[0]) - float(b[1]) * substitution
    denominator = 1.0 + float(risk_aversion) * float(noise_variance) * schur
    if denominator <= 0.0:
        raise ValueError(
            f"the denominator 1 + r sigma^2 (C_11 - C_12^2/C_22) is {denominator:.6g}, which is "
            f"not positive. C'' is not positive semidefinite if the Schur complement is negative."
        )
    weight = numerator / denominator
    naive_denominator = 1.0 + float(risk_aversion) * c11 * float(noise_variance)
    return UnmeasurableCorrection(
        measured=names[0],
        unmeasured=names[1],
        benefit_measured=float(b[0]),
        benefit_unmeasured=float(b[1]),
        c11=c11,
        c12=c12,
        c22=c22,
        noise_variance=float(noise_variance),
        risk_aversion=float(risk_aversion),
        substitution=substitution,
        schur=schur,
        numerator=numerator,
        denominator=denominator,
        weight=weight,
        weight_ignoring=float(b[0]) / naive_denominator,
        zero_weight=abs(numerator) <= tol,
        degenerate=schur == 0.0 and numerator != 0.0,
    )


def two_task_surplus(alpha_1: float, corr: UnmeasurableCorrection) -> float:
    """The principal's certainty equivalent at a weight, for the two-task unmeasurable case.

    `S(a) = a N / det - a^2 C_22 / (2 det) - r sigma^2 a^2 / 2` with `N = B_1 C_22 - B_2 C_12` and
    `det = C_11 C_22 - C_12^2`. It is a concave quadratic whose maximiser is the stated formula, so
    evaluating it is how "any positive power is worse than none" becomes a computation rather than
    a claim.

    At exactly perfect substitutability `det` is zero and this raises. That is not a gap: with
    fungible capacity the agent's effort response to any nonzero incentive is unbounded, so the
    surplus has no finite value and returning one would be the confident wrong number this library
    exists to refuse. Approach the boundary through the family and read the limit, which is what the
    zero-weight test does.
    """
    det = corr.c11 * corr.c22 - corr.c12 * corr.c12
    if det <= 0.0:
        raise ValueError(
            f"det(C'') = {det:.6g}. At perfect substitutability the agent can move effort between "
            f"the tasks at no net cost, so its response to any nonzero weight is unbounded and the "
            f"surplus has no finite value. Evaluate along a family approaching the boundary "
            f"instead: the limit of the optimum is what the closed form reports."
        )
    n = corr.benefit_measured * corr.c22 - corr.benefit_unmeasured * corr.c12
    a = float(alpha_1)
    return (
        a * n / det
        - 0.5 * a * a * corr.c22 / det
        - 0.5 * corr.risk_aversion * corr.noise_variance * a * a
    )


# ---------------------------------------------------------------------------
# The recommendation as a function of what nobody measured
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Crossing:
    """Two components whose recommended weights swap order somewhere in the swept range."""

    first: str
    second: str
    at: float
    parameter: str

    def says(self) -> str:
        return (
            f"{self.first} and {self.second} swap order at {self.parameter} = {self.at:.4g}; "
            f"below that the recommendation prefers one and above it the other."
        )


@dataclass(frozen=True)
class SensitivityCurve:
    """The recommendation as a function of the parameter nobody measured.

    The point of this object is the one claim that survives not knowing `r`: over how much of a
    plausible range does the recommended *ordering* of the components hold. That question has an
    answer where the weights themselves do not, and it is usually the answer a reader can act on.
    """

    sweep: Sweep
    components: tuple[str, ...]
    #: `(n_values, m)`. One recommended weight vector per swept value.
    weights: np.ndarray
    orderings: tuple[tuple[int, ...], ...]
    dominant_ordering: tuple[int, ...]
    dominant_span: float
    distinct_orderings: int
    crossings: tuple[Crossing, ...]

    @property
    def ordering_is_stable(self) -> bool:
        return self.distinct_orderings == 1

    def named_ordering(self, ordering: tuple[int, ...]) -> str:
        return " > ".join(self.components[i] for i in ordering)

    def says(self) -> str:
        if self.ordering_is_stable:
            return (
                f"The recommended weights depend on {self.sweep.parameter}, which nobody has "
                f"measured, but the ordering does not: {self.named_ordering(self.dominant_ordering)}"
                f" holds across all {self.sweep.values.size} swept values from "
                f"{self.sweep.values.min():.4g} to {self.sweep.values.max():.4g}."
            )
        return (
            f"The recommended ordering is not stable over {self.sweep.parameter}: "
            f"{self.distinct_orderings} distinct orderings appear across the swept range, and "
            f"{self.named_ordering(self.dominant_ordering)} holds over {self.dominant_span:.0%} of "
            f"it. {len(self.crossings)} pair(s) cross inside the range."
        )

    def render(self) -> str:
        lines = [self.says(), f"  {self.sweep.render()}"]
        lines.extend(f"  {c.says()}" for c in self.crossings)
        return "\n".join(lines)


def _with_parameter(params: ContractParameters, name: str, value: float) -> ContractParameters:
    if name == "risk_aversion":
        return ContractParameters(
            components=params.components,
            benefit=params.benefit,
            cost_curvature=params.cost_curvature,
            noise=params.noise,
            risk_aversion=float(value),
            sensitivity=params.sensitivity,
            effort=params.effort,
            unmeasurable=params.unmeasurable,
            source={**params.source, "risk_aversion": ParameterSource.SWEPT},
            note=params.note,
        )
    if name == "cost_scale":
        return ContractParameters(
            components=params.components,
            benefit=params.benefit,
            cost_curvature=float(value) * params.cost_curvature,
            noise=params.noise,
            risk_aversion=params.risk_aversion,
            sensitivity=params.sensitivity,
            effort=params.effort,
            unmeasurable=params.unmeasurable,
            source={**params.source, "cost_curvature": ParameterSource.SWEPT},
            note=params.note,
        )
    raise ValueError(
        f"this layer sweeps 'risk_aversion' or 'cost_scale'; got {name!r}. Both enter only through "
        f"the product r * C'' * Sigma, so a sweep over either is a sweep over that product."
    )


def sweep_weights(params: ContractParameters, sweep: Sweep) -> SensitivityCurve:
    """The recommendation at every value of a swept parameter, with the ordering summarised."""
    rows = []
    orderings = []
    for value in sweep.values:
        alpha = optimal_weights(_with_parameter(params, sweep.parameter, float(value)))
        rows.append(alpha)
        orderings.append(tuple(int(i) for i in np.argsort(-alpha, kind="stable")))
    weights = np.vstack(rows)
    counts: dict[tuple[int, ...], int] = {}
    for o in orderings:
        counts[o] = counts.get(o, 0) + 1
    dominant = max(counts, key=lambda k: counts[k])

    crossings: list[Crossing] = []
    m = params.m
    for i in range(m):
        for j in range(i + 1, m):
            diff = weights[:, i] - weights[:, j]
            sign = np.sign(diff)
            flips = np.flatnonzero(sign[:-1] * sign[1:] < 0)
            for k in flips:
                lo, hi = float(sweep.values[k]), float(sweep.values[k + 1])
                for _ in range(60):
                    mid = math.sqrt(lo * hi) if lo > 0 and hi > 0 else 0.5 * (lo + hi)
                    a = optimal_weights(_with_parameter(params, sweep.parameter, mid))
                    if (a[i] - a[j]) * diff[k] > 0:
                        lo = mid
                    else:
                        hi = mid
                crossings.append(
                    Crossing(
                        first=params.components[i],
                        second=params.components[j],
                        at=math.sqrt(lo * hi) if lo > 0 and hi > 0 else 0.5 * (lo + hi),
                        parameter=sweep.parameter,
                    )
                )
    return SensitivityCurve(
        sweep=sweep,
        components=params.components,
        weights=weights,
        orderings=tuple(orderings),
        dominant_ordering=dominant,
        dominant_span=counts[dominant] / len(orderings),
        distinct_orderings=len(counts),
        crossings=tuple(crossings),
    )


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightRecommendation:
    """N5's reading: the weights, what noise took off them, the two baselines, and the sweep."""

    components: tuple[str, ...]
    weights: np.ndarray
    #: `alpha_i` divided by the noiseless optimum `(M^-T B')_i`: the factor by which noise cut each
    #: component's weight below what its value alone would buy. At `M = I` with diagonal `C''` and
    #: `Sigma` this is exactly `1/(1 + r C_ii sigma_i^2)`.
    shrinkage: np.ndarray
    surplus: float
    baseline_equal: np.ndarray
    baseline_value: np.ndarray
    surplus_equal: float
    surplus_value: float
    parameters: ContractParameters
    cost_condition_number: float
    correction: UnmeasurableCorrection | None = None
    curve: SensitivityCurve | None = None
    notes: tuple[str, ...] = ()
    baselines: Mapping[str, float] = field(default_factory=dict)

    @property
    def most_shrunk(self) -> tuple[str, float]:
        """The component the noise term cut hardest, and its shrinkage factor."""
        idx = int(np.argmin(self.shrinkage))
        return self.components[idx], float(self.shrinkage[idx])

    def says(self) -> str:
        name, factor = self.most_shrunk
        gain = self.surplus - self.surplus_value
        head = (
            f"Weight {', '.join(f'{c} at {w:.4g}' for c, w in zip(self.components, self.weights))}. "
            f"Noise cuts {name} hardest, to {factor:.1%} of what its value alone would buy, so "
            f"ignoring the noise overweights it by {1.0 / factor if factor > 0 else math.inf:.2g}x."
        )
        if gain > 0 and self.surplus_value < 0.0 <= self.surplus:
            head += (
                f" Weighting by value alone scores {self.surplus_value:.4g}, which is below doing "
                f"nothing at all: on this composite the noise term is not a refinement, it is the "
                f"difference between a contract that pays and one that costs."
            )
        elif gain > 0:
            head += (
                f" Against weighting by value alone, this is worth {gain:.4g} of surplus "
                f"({gain / abs(self.surplus_value) if self.surplus_value else math.inf:.1%} more)."
            )
        return head

    def render(self) -> str:
        lines = [self.says(), "", "  component        weight   noiseless   shrinkage"]
        for i, name in enumerate(self.components):
            mark = "  (unmeasurable)" if i in set(self.parameters.unmeasurable) else ""
            lines.append(
                f"  {name:<16} {self.weights[i]:>8.4g} {self.baseline_value[i]:>11.4g} "
                f"{self.shrinkage[i]:>11.4f}{mark}"
            )
        lines.append(
            f"  surplus: recommendation {self.surplus:.6g}, equal weights "
            f"{self.surplus_equal:.6g}, value weights {self.surplus_value:.6g}"
        )
        lines.append(
            f"  condition number of C'': {self.cost_condition_number:.4g}. The closer this is to "
            f"infinity the more nearly the tasks are perfect substitutes and the less bounded the "
            f"agent's response to any of these weights."
        )
        lines.extend(f"  {n}" for n in self.notes)
        if self.correction is not None:
            lines.append("")
            lines.append(self.correction.render())
        if self.curve is not None:
            lines.append("")
            lines.append(self.curve.render())
        lines.append("")
        lines.append(self.parameters.render_provenance())
        lines.append("")
        lines.append(render_assumptions())
        return "\n".join(lines)


def recommend_weights(
    params: ContractParameters,
    *,
    sweep: Sweep | None = None,
    correction_names: tuple[str, str] | None = None,
) -> WeightRecommendation:
    """The reading, from a parameter set whose provenance has already been settled."""
    alpha = optimal_weights(params)
    b = params.benefit
    # The two baselines. `value` is the recommendation with the noise term deleted, which is what
    # weighting a composite by how much you care about each part produces; `equal` is every
    # component at the same weight, which is what a composite with no analysis at all uses. Both are
    # expressed in the same units as the recommendation, so the surplus comparison is meaningful.
    value = noiseless_weights(params)
    equal = np.full(params.m, float(np.nanmean(np.abs(value))), dtype=np.float64)
    for i in params.unmeasurable:
        equal[i] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        shrink = np.where(value != 0.0, alpha / np.where(value == 0.0, 1.0, value), np.nan)

    notes: list[str] = []
    cond = params.cost_condition_number
    if cond > DEFAULT_CONDITION_LIMIT:
        notes.append(
            f"C'' has condition number {cond:.4g}, so the tasks are numerically indistinguishable "
            f"from perfect substitutes. The weights below are finite and the effort they induce is "
            f"not; read them as the boundary of a family rather than as an operating point."
        )
    if params.unmeasurable:
        named = ", ".join(params.components[i] for i in params.unmeasurable)
        notes.append(
            f"{named} carries no signal, so its weight is zero by construction. Its value still "
            f"enters every other component's weight through the off-diagonal of C''."
        )

    correction = None
    if params.m == 2 and len(params.unmeasurable) == 1:
        measured_idx = params.measurable[0]
        unmeasured_idx = params.unmeasurable[0]
        order = [measured_idx, unmeasured_idx]
        correction = unmeasurable_correction(
            benefit=[float(b[measured_idx]), float(b[unmeasured_idx])],
            cost_curvature=params.cost_curvature[np.ix_(order, order)],
            noise_variance=float(params.noise[measured_idx, measured_idx]),
            risk_aversion=params.risk_aversion,
            names=correction_names
            or (params.components[measured_idx], params.components[unmeasured_idx]),
        )

    return WeightRecommendation(
        components=params.components,
        weights=alpha,
        shrinkage=shrink,
        surplus=principal_surplus(alpha, params),
        baseline_equal=equal,
        baseline_value=value,
        surplus_equal=principal_surplus(equal, params),
        surplus_value=principal_surplus(value, params),
        parameters=params,
        cost_condition_number=cond,
        correction=correction,
        curve=None if sweep is None else sweep_weights(params, sweep),
        notes=tuple(notes),
        baselines={
            "baseline.equal_weights": principal_surplus(equal, params),
            "baseline.value_weights": principal_surplus(value, params),
        },
    )


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


class OptimalWeights(DecisionInstrument):
    """N5. The weight a reward component should carry, given its measured noise.

    Kill condition: if the recommended weights are within measurement error of the
    value-proportional baseline on every real composite tested, the noise term is doing no work and
    the instrument collapses into a restatement of `B'`.
    """

    name = "OptimalWeights"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.COVARIANT
    faithful_to = "N5"
    deviations = (
        "the source states the formula at M = I, which is not scale free: rescaling a "
        "component's score sends sigma^2 to a^2 sigma^2 while the 1 in the denominator does not "
        "move, so the diagonal form applied to raw reward model scores is a units error rather "
        "than an approximation. The general form alpha* = [M C''^-1 M' + r Sigma]^-1 M C''^-1 B' "
        "is what is computed, it reduces to the stated form at M = I, and it is exactly covariant "
        "under a joint rescaling of Sigma and M",
        "no interval is reported on the weights. The sampling error of Sigma is not what dominates "
        "the uncertainty of a recommendation whose other four parameters were stated rather than "
        "measured, and an interval carrying only the measured part would read as the uncertainty "
        "of the whole. The sweep is the interval this instrument has",
        "the catalogue carries no N5 record and no registered quantity for it. The record and "
        "the quantity row proposed here fill that gap, and `quantities.as_catalogue_rows()` "
        "emits them",
        "the source's own reduction of the perfectly-substitutable case is incomplete. "
        "C_11 = C_12 = C_22 gives a Schur complement of exactly zero and a numerator of B_1 - B_2, "
        "so the optimum is exactly zero when the two tasks are equally valuable and is a knife "
        "edge with unbounded surplus when they are not. Both cases are computed and flagged rather "
        "than collapsed into one sentence",
    )

    quantity = "reward.optimal_weights"
    requires = NOISE_ACCESS
    invariance = "reward.affine"
    invariance_relation = COVARIANT_INVERSE
    baselines = N5_BASELINES
    rung = 1

    def __init__(
        self,
        parameters: ContractParameters | None = None,
        *,
        sweep: Sweep | None = None,
        condition_limit: float = DEFAULT_CONDITION_LIMIT,
    ) -> None:
        self.parameters = parameters
        self.sweep = sweep
        self.condition_limit = float(condition_limit)

    def compute(self) -> Any:
        if self.parameters is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    "no contract parameters were supplied, so there is nothing to recommend a "
                    "weight for"
                ),
                remedy=(
                    "build a `ContractParameters` naming the components, the marginal value of "
                    "each, the effort-cost curvature, the noise covariance and the risk aversion. "
                    "`noise_from_gauge_studies` turns A2's readings into the noise covariance; the "
                    "rest are stated or swept and the reading records which."
                ),
            )
        needed = ["benefit", "cost_curvature", "noise", "risk_aversion", "sensitivity"]
        if self.sweep is not None and self.sweep.parameter == "risk_aversion":
            needed.remove("risk_aversion")
        bad = self.parameters.refuse_unstated(self.name, needed)
        if bad is not None:
            return bad
        return recommend_weights(self.parameters, sweep=self.sweep)

    def payload(self, computed: WeightRecommendation) -> dict[str, Any]:
        body: dict[str, Any] = {
            "components": list(computed.components),
            "weights": [float(w) for w in computed.weights],
            "noiseless_weights": [float(w) for w in computed.baseline_value],
            "equal_weights": [float(w) for w in computed.baseline_equal],
            "shrinkage": [float(s) for s in computed.shrinkage],
            "surplus": computed.surplus,
            "surplus_equal_weights": computed.surplus_equal,
            "surplus_value_weights": computed.surplus_value,
            "cost_condition_number": computed.cost_condition_number,
            "unmeasurable": [computed.components[i] for i in computed.parameters.unmeasurable],
            "parameter_provenance": computed.parameters.provenance_rows(),
            "notes": list(computed.notes),
            "baselines": dict(computed.baselines),
            "says": computed.says(),
        }
        if computed.correction is not None:
            c = computed.correction
            body["unmeasurable_correction"] = {
                "measured": c.measured,
                "unmeasured": c.unmeasured,
                "substitution": c.substitution,
                "schur_complement": c.schur,
                "numerator": c.numerator,
                "denominator": c.denominator,
                "weight": c.weight,
                "weight_ignoring_the_unmeasured_task": c.weight_ignoring,
                "discount": c.discount,
                "zero_weight": c.zero_weight,
                "degenerate": c.degenerate,
                "says": c.says(),
            }
        if computed.curve is not None:
            curve = computed.curve
            body["sensitivity"] = {
                "parameter": curve.sweep.parameter,
                "reason": curve.sweep.reason,
                "values": [float(v) for v in curve.sweep.values],
                "weights": [[float(x) for x in row] for row in curve.weights],
                "distinct_orderings": curve.distinct_orderings,
                "dominant_ordering": [curve.components[i] for i in curve.dominant_ordering],
                "dominant_span": curve.dominant_span,
                "ordering_is_stable": curve.ordering_is_stable,
                "crossings": [
                    {"first": c.first, "second": c.second, "at": c.at} for c in curve.crossings
                ],
                "says": curve.says(),
            }
        return body


__all__ = [
    "COVARIANT_INVERSE",
    "DEFAULT_CONDITION_LIMIT",
    "N5_BASELINES",
    "Crossing",
    "OptimalWeights",
    "SensitivityCurve",
    "UnmeasurableCorrection",
    "WeightRecommendation",
    "optimal_weights",
    "optimal_weights_diagonal",
    "noiseless_weights",
    "principal_surplus",
    "recommend_weights",
    "shrinkage",
    "sweep_weights",
    "two_task_surplus",
    "unmeasurable_correction",
]

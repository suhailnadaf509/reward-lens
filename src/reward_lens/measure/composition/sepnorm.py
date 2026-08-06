"""B5. The scalar-representability bound, reported for the theorem it carries and never as the estimate.

``‖U‖_⊕ = inf_{g,f} ‖U − g ⊕ f‖_∞`` is the distance from a comparison kernel to the nearest
additively separable one, in the sup norm. It exists in this library for exactly one reason: it
carries a convergence theorem. The contraction factor of the comparison kernel is

    c = min(1 − e^{−2‖U‖_⊕}, ‖U‖_⊕)

so a perfectly scalar-representable grader has ``c = 0`` and converges in one step, recovering RLHF
exactly, and a grader at ``‖U‖_⊕ = 0.31`` contracts at 0.31 per step instead. That is a real and
useful statement and nothing in the L² decomposition replaces it.

**It is also the statistically worse of the two numbers, and it is not the primary estimate.**
``‖U‖_⊕`` and B1's curl mass are norms of the same interaction term: ``Δ(U)`` below is exactly
``‖curl U‖_∞`` where the curl mass is ``‖curl U‖²₂ / ‖U‖²₂``. A sup norm over a noisy field is a
maximum of a noisy field, so it is upward biased with bias growing like ``√(log n)``, it fluctuates
Gumbel rather than Gaussian, and **collecting more comparisons makes the estimate worse**. The L²
version is root-n estimable and its noise term is analytically subtractable. So this module always
reports B1's curl mass beside its own number, and a card that prints one without the other is
printing the wrong one.

**The linear program is not solved.** Proposition 2.5 sandwiches the exact value,

    (1/3)·Δ(U)  ≤  ‖U‖_⊕  ≤  Δ(U),      Δ(U) = esssup |U(x,y) + U(y,z) + U(z,x)|

and a factor of three from an ``O(n³)`` scan is worth more than an exact answer from an LP at the
group sizes that matter. Two rungs: rung 0 is the ``O(n²)`` bound from the column-mean potential,
rung 1 adds the triple scan and closes the sandwich from both sides.

**What is computed is the observed sub-kernel.** ``Δ`` is defined as an essential supremum over all
triples and only the filled triangles are visible, so the scanned value is a lower bound on the true
``Δ``. That keeps ``Δ/3`` a valid lower bound on ``‖U‖_⊕`` and it makes ``Δ`` an upper bound only for
the kernel restricted to the pairs actually compared. The reading is that restriction, which is the
object the data supports, and it lower-bounds the full-kernel value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import Evidence
from reward_lens.core.gates import require_frame_for_comparison
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
)
from reward_lens.measure.base import BaseObservable, Context, run
from reward_lens.measure.composition.hodge import (
    ComparisonFlow,
    Triangle,
    split_flow,
    triangles,
)


def contraction_factor(sep_distance: float) -> float:
    """``c = min(1 − e^{−2‖U‖_⊕}, ‖U‖_⊕)``, the theorem this whole module exists to carry.

    Monotone increasing in its argument, so a sandwich on ``‖U‖_⊕`` maps to a sandwich on ``c`` and
    the upper end of the sandwich is the honest worst case to quote.
    """
    d = max(float(sep_distance), 0.0)
    return float(min(1.0 - np.exp(-2.0 * d), d))


@dataclass(frozen=True)
class SepDistance:
    """The sandwich on ``‖U‖_⊕``, both contraction factors, and the width that decides usefulness.

    ``lower`` and ``upper`` bracket the true value. ``upper`` is the tighter of the two upper bounds
    available: the ``O(n³)`` ``Δ(U)`` and the ``O(n²)`` sup-norm residual after removing the
    column-mean potential. Taking the minimum is free and is strictly better than either.

    ``width`` and ``ratio`` are what B5's kill condition is stated in. A sandwich whose ratio is
    three at every group size has told you the factor-of-three theorem and nothing about this
    grader.
    """

    n_items: int
    n_pairs: int
    n_triangles: int
    rung: int
    delta: float
    column_mean_bound: float
    lower: float
    upper: float
    contraction_lower: float
    contraction_upper: float
    curl_mass: float
    note: str = ""

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def ratio(self) -> float:
        return float("inf") if self.lower <= 0.0 else self.upper / self.lower

    def to_dict(self) -> dict[str, Any]:
        return {
            "sep_distance_lower": self.lower,
            "sep_distance_upper": self.upper,
            "sandwich_width": self.width,
            "sandwich_ratio": self.ratio,
            "contraction_lower": self.contraction_lower,
            "contraction_upper": self.contraction_upper,
            "delta_triple_scan": self.delta,
            "column_mean_bound": self.column_mean_bound,
            "curl_mass": self.curl_mass,
            "n_items": self.n_items,
            "n_pairs": self.n_pairs,
            "n_triangles": self.n_triangles,
            "rung": self.rung,
            "note": self.note,
        }


def triple_scan(
    flow: ComparisonFlow, tris: Sequence[Triangle] | None = None
) -> tuple[float, Triangle | None]:
    """``Δ(U) = max |U(x,y) + U(y,z) + U(z,x)|`` over filled triangles, with the worst triple named.

    ``O(n³)`` on a complete graph and proportional to the number of filled triangles otherwise. The
    triple sum is the circulation around the triangle, so this is ``‖curl U‖_∞`` under a different
    name, which is the sense in which B5 and B1 are two norms of one object.

    Returning the worst triple is not decoration. ``Δ`` is a maximum, so it is decided by exactly one
    triple, and a reader who cannot see which one has been handed a number they cannot check.
    """
    if flow.n_edges == 0:
        return 0.0, None
    value = {}
    for (a, b), y in zip(flow.edges, flow.flow):
        value[(a, b)] = float(y)
        value[(b, a)] = -float(y)
    tris = list(tris) if tris is not None else triangles(flow.edges)
    best, worst = 0.0, None
    for i, j, k in tris:
        circulation = abs(value[(i, j)] + value[(j, k)] + value[(k, i)])
        if circulation > best:
            best, worst = circulation, (i, j, k)
    return best, worst


def column_mean_bound(flow: ComparisonFlow) -> float:
    """An ``O(n²)`` upper bound: ``‖U − ∂h‖_∞`` with ``h`` the mean over each item's comparisons.

    The additively separable kernels ``g ⊕ f`` include every discrete gradient (take ``g = −h`` and
    ``f = h``), so the sup-norm residual after removing *any* potential is an upper bound on
    ``‖U‖_⊕``. The column-mean potential is the Borda score, and on a complete graph it is exactly
    the least-squares Hodge potential, so on the design where the bound matters most it is the best
    gradient any ``L²`` method would have found.

    Cost is one pass over the pairs, which is where the ``O(n²)`` in B5's rung-0 entry comes from.
    Note that it delivers one side of the sandwich rather than both: the lower bound needs the triple
    scan, because a lower bound has to certify that no separable kernel comes closer and a single
    residual cannot.
    """
    if flow.n_edges == 0:
        return 0.0
    total = np.zeros(flow.n_items)
    count = np.zeros(flow.n_items)
    for (a, b), y in zip(flow.edges, flow.flow):
        total[b] += float(y)
        total[a] -= float(y)
        count[a] += 1.0
        count[b] += 1.0
    potential = np.divide(total, np.maximum(count, 1.0))
    residual = np.array(
        [abs(float(y) - (potential[b] - potential[a])) for (a, b), y in zip(flow.edges, flow.flow)]
    )
    return float(residual.max())


def sep_distance(flow: ComparisonFlow, *, rung: int = 1) -> SepDistance | Refusal:
    """The sandwich on ``‖U‖_⊕``, with B1's curl mass carried alongside it.

    ``rung=0`` computes the ``O(n²)`` bound only and returns an upper bound with a lower bound of
    zero, which is honest rather than useless: zero is a correct lower bound and the width says how
    much the triple scan would buy. ``rung=1`` runs the scan and closes the sandwich.
    """
    if flow.n_edges == 0:
        return Refusal(
            instrument="SepDistance",
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail="the comparison flow has no edges, so there is no kernel to measure",
            remedy=(
                "supply at least three items compared to each other. The triple scan needs a filled "
                "triangle, and with none of them the sandwich degenerates to [0, 0], which is not a "
                "claim that the grader is scalar-representable."
            ),
        )
    tris = triangles(flow.edges)
    if not tris:
        return Refusal(
            instrument="SepDistance",
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=(
                f"the comparison graph has {flow.n_edges} pairs and no filled triangle, so no triple "
                f"is fully observed and Δ(U) has nothing to take a maximum over"
            ),
            remedy=(
                "compare all three pairs of at least one triple. A design with no closed triangle "
                "cannot see a cycle at all, so it cannot bound the distance from a scalar."
            ),
            statistics={"n_pairs": flow.n_edges, "n_triangles": 0},
        )

    column = column_mean_bound(flow)
    split = split_flow(flow, with_betti=False)
    if rung <= 0:
        return SepDistance(
            n_items=flow.n_items,
            n_pairs=flow.n_edges,
            n_triangles=len(tris),
            rung=0,
            delta=float("nan"),
            column_mean_bound=column,
            lower=0.0,
            upper=column,
            contraction_lower=0.0,
            contraction_upper=contraction_factor(column),
            curl_mass=split.curl_mass,
            note=(
                "rung 0: the O(n²) column-mean bound only. The lower end is 0 because a single "
                "residual certifies no separable kernel is further away, not that none is closer. "
                "Rung 1's triple scan is what raises it."
            ),
        )

    delta, worst = triple_scan(flow, tris)
    upper = min(delta, column)
    lower = delta / 3.0
    if lower > upper:
        # Δ is scanned over filled triangles only, so on an incomplete design it can exceed the
        # column-mean residual without either being wrong. Widening rather than reporting an
        # inverted interval keeps the bracket valid; the note says which side moved.
        upper = delta
    return SepDistance(
        n_items=flow.n_items,
        n_pairs=flow.n_edges,
        n_triangles=len(tris),
        rung=1,
        delta=delta,
        column_mean_bound=column,
        lower=lower,
        upper=upper,
        contraction_lower=contraction_factor(lower),
        contraction_upper=contraction_factor(upper),
        curl_mass=split.curl_mass,
        note=(
            f"rung 1: (1/3)Δ ≤ ‖U‖_⊕ ≤ min(Δ, column-mean bound), with Δ = {delta:.4g} attained on "
            f"triple {worst}. Computed on the {flow.n_edges} pairs actually compared, so it "
            f"lower-bounds the full-kernel value. Report beside curl mass "
            f"{split.curl_mass:.4g}, which is the estimate; this is the bound that carries the "
            f"convergence theorem."
        ),
    )


#: B5's envelope is unconditional, and the catalogue line is the reason it needed correcting: the
#: source reads "Env `SCALAR_REPRESENTABLE` under test". B5's own quantity *is* the distance from
#: scalar representability, so requiring the condition would make the instrument refuse in exactly
#: the case it exists to report on.
SEP_DISTANCE_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "the quantity is the distance from scalar representability, so `SCALAR_REPRESENTABLE` is "
        "this instrument's output rather than its precondition. The sandwich is an inequality "
        "between two functionals of the observed kernel and holds pointwise, so no regime can make "
        "it false."
    ),
)


class SepDistanceBound(BaseObservable):
    """B5. The worst-case number that carries the convergence theorem.

    Always reported beside B1's curl mass, which this instrument carries in its own payload so the
    pair cannot be separated by a card that reads only one field.
    """

    name = "SepDistanceBound"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to: str | None = (
        "the L^∞ scalar-representability distance ‖U‖_⊕ and its Proposition 2.5 sandwich"
    )
    deviations: tuple[str, ...] = (
        "the linear program is not solved, by instruction: the factor-of-three sandwich is O(n²) "
        "and O(n³) against the LP's cost and is free at the group sizes that matter",
        "Δ(U) is scanned over filled triangles only, because an unobserved pair makes a triple's "
        "circulation undefined. The scanned Δ therefore lower-bounds the true Δ and the reading is "
        "the distance for the kernel restricted to the pairs actually compared",
        "the upper end of the sandwich is the smaller of Δ and the column-mean sup-norm residual "
        "rather than Δ alone, which is a strict improvement on the printed bound and costs nothing",
    )

    quantity = "grader.sep_distance"
    requires = {Component.GRADER: Access.QUERY}
    substrates = frozenset()
    phases = frozenset()
    envelope = SEP_DISTANCE_ENVELOPE
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = ("B1's curl mass, always reported beside it",)
    rung = 1

    def __init__(self, flow: ComparisonFlow | None = None, *, rung: int = 1) -> None:
        self.flow = flow
        self.rung = int(rung)
        self._computed: Any = None

    def compute(self) -> Any:
        if self.flow is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no comparison flow was supplied, so there is no kernel to bound",
                remedy=(
                    "pass `flow=edge_flow(pairs, n_items)` built from the grader's observed "
                    "comparisons, the same object B1 reads."
                ),
            )
        return sep_distance(self.flow, rung=self.rung)

    def payload(self, computed: SepDistance) -> dict[str, Any]:
        return computed.to_dict()

    def estimate(self, ctx: Context) -> Reading:
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        out = self.compute()
        if isinstance(out, Refusal):
            return out
        self._computed = out
        try:
            if ctx.signal is not None:
                return run(self, ctx)
            if ctx.is_comparison:
                require_frame_for_comparison(self.gauge_status, ctx.frame)
            return self.measure(ctx)
        finally:
            self._computed = None

    def measure(self, ctx: Context) -> Evidence:
        out = self._computed if self._computed is not None else self.compute()
        if isinstance(out, Refusal):
            raise ValueError(
                f"{self.name}.measure was called on a measurement that declines to produce "
                f"Evidence: {out.reason.name}. Call `estimate`, which returns the refusal as a "
                f"value carrying its remedy."
            )
        return ctx.emit(self.payload(out))


def sep_distance_of_payload(payload: Any) -> float:
    """The upper end of the sandwich for the comparisons implied by an invariance payload.

    The adapter the generated invariance test runs through, built on B1's so the two instruments are
    checked against the same construction. Under `reward.affine` the scores are rescaled by a
    positive factor and shifted, which cannot change which item wins a comparison, so the kernel and
    every norm of it are unchanged and the declared relation is invariance.
    """
    from reward_lens.measure.composition.hodge import PairCount, disjoint_union, edge_flow

    payload.require("scores", "group_ids")
    scores = np.asarray(payload.scores, dtype=np.float64)
    gids = np.asarray(payload.group_ids)
    flows = []
    for g in np.unique(gids):
        idx = np.flatnonzero(gids == g)
        if idx.size < 3:
            continue
        local = scores[idx]
        flows.append(
            edge_flow(
                [
                    PairCount(
                        a,
                        b,
                        1.0 if local[a] > local[b] else 0.0,
                        1.0 if local[b] > local[a] else 0.0,
                    )
                    for a in range(idx.size)
                    for b in range(a + 1, idx.size)
                ],
                idx.size,
            )
        )
    if not flows:
        return 0.0
    out = sep_distance(disjoint_union(flows), rung=1)
    return 0.0 if isinstance(out, Refusal) else out.upper


def report_line(bound: SepDistance) -> str:
    """One line for a card, with the theorem spelled out and the estimate named as the estimate."""
    return (
        f"‖U‖_⊕ ∈ [{bound.lower:.3g}, {bound.upper:.3g}], so the comparison kernel contracts at "
        f"c ≤ {bound.contraction_upper:.3g} rather than in one step. Curl mass "
        f"{bound.curl_mass:.3g} is the estimate; this is the worst case."
    )


__all__ = [
    "SEP_DISTANCE_ENVELOPE",
    "SepDistance",
    "SepDistanceBound",
    "column_mean_bound",
    "contraction_factor",
    "report_line",
    "sep_distance",
    "sep_distance_of_payload",
    "triple_scan",
]

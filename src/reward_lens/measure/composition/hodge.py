"""B1. The combinatorial Hodge decomposition of a grader's comparison flow.

Given items as vertices and observed comparisons as edges, the edge flow ``Y`` is antisymmetric and
the space it lives in splits three ways::

    C¹ = im(grad) ⊕ ker(Δ₁) ⊕ im(curl*)

The three pieces are not interchangeable and reporting their sum is what this module exists to stop.

    GRADIENT   the flow is the difference of one scalar potential over the items. This is the whole
               of what a scalar reward can carry.
    CURL       the flow circulates around filled triangles. The grader is locally cyclic: it prefers
               A to B, B to C and C to A on three items it compared to each other. **More data does
               not remove this.** It is a statement about the grader.
    HARMONIC   the flow is divergence-free and curl-free and still not a gradient, because the
               comparison graph has a hole the flow wraps around. The dimension of that space is the
               first Betti number of the clique complex, which is a topological invariant of *which
               pairs you chose to compare*. **Comparing different pairs does remove this.** It is a
               statement about the experimental design.

So the two numbers answer different questions and only one of them is actionable. A single
``intransitive_mass`` throws away the half that says what to do, which is why ``grader.curl_mass``
and ``grader.harmonic_mass`` are separate registered quantities.

**Why this is a model-class problem rather than a data problem.** Any model whose fitted comparison
kernel has the form ``f(s_j - s_i)`` for a scalar score ``s`` is confined to ``im(grad)`` by
construction: its stationary condition sets the divergence of the residual to zero and leaves the
residual in the cycle space untouched. For the weighted-least-squares member of that family the
statement is an identity rather than an analogy, since Bradley-Terry fitting on the log-odds edge
flow *is* Hodge gradient projection. Either way the fit provably discards ``‖curl‖² + ‖harm‖²`` of
the flow's energy no matter how many comparisons you collect, so a nonzero cyclic mass is a
statement about the model class and not about the sample size.

**The sup-norm twin is in `sepnorm.py` and is not the primary estimate.** ``‖U‖_⊕`` carries a
convergence theorem, which is the only reason it is computed at all. It is a maximum of a noisy
field, so it is upward biased with bias growing like ``√(log n)``, it fluctuates Gumbel rather than
Gaussian, and collecting more comparisons makes it worse. The L² curl mass here is root-n estimable
and its noise term is analytically subtractable, so this is the number that gets reported and
``grader.sep_distance`` travels beside it as the worst case.

**The estimator is not the quantity, and on ±1 data the gap is the whole result.** The flow this
module decomposes is built from win counts, and a comparison recorded as a bare win or loss is a
sign rather than a margin. A sign flow is not a gradient even when the underlying preference is a
perfect total order: on the complete graph ``K_n`` a transitive ±1 tournament has curl mass exactly
``(n-2)/(3n)``, rising from 0.111 at three items to 0.25 at eight and approaching one third. That is
an encoding artifact with no intransitivity in it at all, and it is why `transitive_baseline` in
`nulls.py` is mandatory rather than optional. Any curl mass reported without it is uninterpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import Evidence
from reward_lens.core.gates import require_frame_for_comparison
from reward_lens.core.invariance import INVARIANT, InvariancePayload
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context, run

Edge = tuple[int, int]
Triangle = tuple[int, int, int]

#: Replications per pair the nulls need: at least 11 per pair, which is also A6's measured
#: requirement that eleven repeats let a majority vote match a fifty-trial reference at 95%. Below
#: this the split-half and the parametric bootstrap refuse and name it. It is a sample-size
#: requirement rather than a regime condition, so it is a refusal with a stated n and not an
#: envelope entry.
MIN_REPLICATIONS_FOR_NULLS = 11

#: The Haldane-Anscombe continuity correction added to each side of a pair before the logit. Without
#: it the log-odds flow is infinite on any unanimous pair, and unanimity is not rare: at eleven
#: replications and a true rate of 0.9 it is the majority case, and on the comparison data this
#: library actually holds every single pair is unanimous because each was decided once.
HALDANE_ANSCOMBE = 0.5

#: Relative tolerance for the two structural self-checks. The decomposition is exact linear algebra,
#: so a correct implementation lands at float64 rounding; this is loose enough never to false-alarm
#: and tight enough to catch a sign or indexing error in the operators. When the sparse solve misses
#: it, `hodge_split` escalates to a dense solve rather than returning a decomposition it cannot
#: certify.
STRUCTURAL_TOL = 1e-8

#: Above this many (edge, triangle) entries the first Betti number is not computed by a dense rank.
#: Real comparison graphs are per prompt and small: sixteen items give 120 edges and 560 triangles,
#: which is 67,200 entries. A graph large enough to trip this is one connected component, not a
#: corpus, since β₁ is additive over components and is computed per component.
_BETTI_DENSE_CAP = 4_000_000

FlowScale = Literal["margin", "logit"]


# ---------------------------------------------------------------------------
# The edge flow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairCount:
    """One unordered pair and how the comparisons of it came out.

    ``i`` and ``j`` are item indices and ``wins_i``/``wins_j`` count outcomes, so
    ``wins_i + wins_j`` is the number of times that pair was actually compared. That sum is the
    replication count the nulls are gated on, and it is per pair rather than per tournament because
    a design that compares one pair thirty times and another once has one replication for the
    purpose of a split half.
    """

    i: int
    j: int
    wins_i: float
    wins_j: float

    @property
    def replications(self) -> float:
        return self.wins_i + self.wins_j


@dataclass(frozen=True)
class ComparisonFlow:
    """An antisymmetric edge flow, with everything needed to say how it was built.

    ``edges`` are canonically oriented ``(a, b)`` with ``a < b`` and ``flow[k]`` is positive when the
    higher-indexed item of edge ``k`` is preferred. ``replications`` is the comparison count per
    edge, carried so that a null can gate on it and a weighted decomposition can use it.

    ``scale`` and ``correction`` are on the object rather than in a comment because they change the
    number. The margin scale is bounded in [-1, 1] and finite on a shutout; the logit scale is the
    canonical HodgeRank flow and is infinite on a shutout without the continuity correction, so the
    correction that was applied travels with the flow and onto the instrument's ``deviations``.

    ``correction`` is the sentence a reader wants and ``correction_value`` is the number the code
    needs. Both, because the nulls have to *invert* the logit to recover win counts and a
    human-readable string cannot be inverted: with only the string, `nulls._signed` assumed the
    Haldane-Anscombe 0.5 whatever had actually been applied, and a flow built at h = 0.1 came back
    with win fractions wrong by 0.045 and at h = 2.0 by 0.200, which feeds straight into the
    Bradley-Terry fit that Null A resimulates from.
    """

    n_items: int
    edges: tuple[Edge, ...]
    flow: np.ndarray
    replications: np.ndarray
    scale: FlowScale
    correction: str
    #: The continuity correction actually applied, per side, before the logit. Zero on the margin
    #: scale, where none is needed.
    correction_value: float = HALDANE_ANSCOMBE

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def min_replications(self) -> float:
        return float(self.replications.min()) if self.n_edges else 0.0

    @property
    def unanimous_fraction(self) -> float:
        """Share of pairs on which every comparison agreed, which is where the logit diverges."""
        if not self.n_edges:
            return 0.0
        return float(np.mean(np.abs(self.flow) >= 1.0 - 1e-12)) if self.scale == "margin" else 0.0

    @property
    def energy(self) -> float:
        return float(self.flow @ self.flow)


def edge_flow(
    pairs: Iterable[PairCount | tuple[int, int, float, float]],
    n_items: int | None = None,
    *,
    scale: FlowScale = "margin",
    correction: float = HALDANE_ANSCOMBE,
) -> ComparisonFlow:
    """Aggregate observed comparisons into one antisymmetric edge flow.

    Repeated observations of the same unordered pair accumulate, and a pair with no observation at
    all is left out, which is what leaves the holes the harmonic component lives in.

    Two scales, and the choice is reported rather than assumed. ``margin`` is
    ``(w_b - w_a) / (w_a + w_b)``, bounded in [-1, 1] and finite everywhere, which is what the
    library's existing topology study uses and therefore what any comparison against its numbers
    has to use. ``logit`` is ``log((w_b + h) / (w_a + h))``, the canonical HodgeRank flow, whose
    only problem is that it is infinite at ``h = 0`` on a unanimous pair. The continuity correction
    ``h`` defaults to the Haldane-Anscombe 0.5 and the applied value is recorded on the result, so a
    reading never leaves it to the reader to guess whether a divergence was handled or hit.
    """
    if scale not in ("margin", "logit"):
        raise ValueError(f"scale must be 'margin' or 'logit'; got {scale!r}")

    accumulated: dict[Edge, list[float]] = {}
    for p in pairs:
        pc = p if isinstance(p, PairCount) else PairCount(*p)
        if pc.i == pc.j:
            raise ValueError(f"a pair compares two distinct items; got i = j = {pc.i}")
        if pc.i < pc.j:
            key, wa, wb = (pc.i, pc.j), float(pc.wins_i), float(pc.wins_j)
        else:
            key, wa, wb = (pc.j, pc.i), float(pc.wins_j), float(pc.wins_i)
        bucket = accumulated.setdefault(key, [0.0, 0.0])
        bucket[0] += wa
        bucket[1] += wb

    edges: list[Edge] = []
    values: list[float] = []
    reps: list[float] = []
    for edge in sorted(accumulated):
        wa, wb = accumulated[edge]
        total = wa + wb
        if total <= 0.0:
            continue
        edges.append(edge)
        reps.append(total)
        if scale == "margin":
            values.append((wb - wa) / total)
        else:
            values.append(float(np.log((wb + correction) / (wa + correction))))

    inferred = 0 if not edges else max(max(e) for e in edges) + 1
    if n_items is None:
        n_items = inferred
    elif n_items < inferred:
        raise ValueError(f"n_items={n_items} is smaller than the largest item index {inferred - 1}")

    note = (
        "none: the margin scale is finite on a unanimous pair"
        if scale == "margin"
        else f"Haldane-Anscombe, h = {correction:g}, added to each side before the logit"
    )
    return ComparisonFlow(
        n_items=n_items,
        edges=tuple(edges),
        flow=np.asarray(values, dtype=np.float64),
        replications=np.asarray(reps, dtype=np.float64),
        scale=scale,
        correction=note,
        correction_value=0.0 if scale == "margin" else float(correction),
    )


def disjoint_union(flows: Sequence[ComparisonFlow]) -> ComparisonFlow:
    """One flow over the disjoint union of several comparison graphs.

    A corpus of independent tournaments is the disjoint union of their complexes, and the Hodge
    decomposition of a disjoint union is the direct sum of the pieces, so decomposing the union once
    gives exactly the energy-summed corpus estimate and keeps the per-edge components in one vector.
    Keeping the components is what the split-half and the spectral nulls need: they correlate and
    eigendecompose the curl component itself, not its norm.
    """
    kept = [f for f in flows if f.n_edges]
    if not kept:
        return ComparisonFlow(0, (), np.zeros(0), np.zeros(0), "margin", "none: empty corpus", 0.0)
    scales = {f.scale for f in kept}
    if len(scales) > 1:
        raise ValueError(f"cannot union flows built on different scales: {sorted(scales)}")
    corrections = {f.correction_value for f in kept}
    if len(corrections) > 1:
        raise ValueError(
            f"cannot union flows built with different continuity corrections: "
            f"{sorted(corrections)}. The nulls invert the logit to recover win counts and one "
            f"union carries one correction, so build the parts with the same `correction=`."
        )
    edges: list[Edge] = []
    offset = 0
    for f in kept:
        edges.extend((a + offset, b + offset) for a, b in f.edges)
        offset += f.n_items
    return ComparisonFlow(
        n_items=offset,
        edges=tuple(edges),
        flow=np.concatenate([f.flow for f in kept]),
        replications=np.concatenate([f.replications for f in kept]),
        scale=kept[0].scale,
        correction=kept[0].correction,
        correction_value=kept[0].correction_value,
    )


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def incidence(n_items: int, edges: Sequence[Edge]) -> csr_matrix:
    """``A``, the vertex-edge incidence, shape ``(n_items, n_edges)``, sparse.

    Each oriented edge ``(a, b)`` with ``a < b`` carries ``-1`` at its tail and ``+1`` at its head,
    so ``A.T @ s`` is the discrete gradient ``s[b] - s[a]`` and the gradient subspace is ``im(A.T)``.
    """
    n = len(edges)
    if n == 0:
        return csr_matrix((n_items, 0), dtype=np.float64)
    cols = np.repeat(np.arange(n), 2)
    rows = np.asarray(edges, dtype=np.int64).ravel()
    data = np.tile(np.array([-1.0, 1.0]), n)
    return coo_matrix((data, (rows, cols)), shape=(n_items, n)).tocsr()


def triangles(edges: Sequence[Edge]) -> list[Triangle]:
    """Every filled triangle of the clique complex: a triple whose three pairs were all compared.

    Enumerated by intersecting adjacency sets over the edges rather than by scanning all vertex
    triples, so the cost follows the graph's density instead of ``n³``. A triple with a missing pair
    is not filled, and that is not a detail: an unfilled triple is exactly the hole that lets a
    harmonic component exist, so which triangles are filled decides the curl-harmonic split.
    """
    adjacency: dict[int, set[int]] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    out: list[Triangle] = []
    for i in sorted(adjacency):
        higher = {v for v in adjacency[i] if v > i}
        for j in sorted(higher):
            for k in sorted(higher & {v for v in adjacency[j] if v > j}):
                out.append((i, j, k))
    return out


def curl_operator(edges: Sequence[Edge], tris: Sequence[Triangle]) -> csr_matrix:
    """``C``, the triangle-edge incidence, shape ``(n_edges, n_triangles)``, sparse.

    Triangle ``(i, j, k)`` with ``i < j < k`` has oriented boundary ``[j,k] - [i,k] + [i,j]``, so its
    column carries ``+1`` on ``(i, j)``, ``-1`` on ``(i, k)`` and ``+1`` on ``(j, k)``. Under this
    convention ``A @ C = 0`` exactly, which is what puts the curl subspace inside the cycle space and
    makes the three components mutually orthogonal rather than merely a sequence of residuals.
    """
    n_e, n_t = len(edges), len(tris)
    if n_t == 0 or n_e == 0:
        return csr_matrix((n_e, n_t), dtype=np.float64)
    index = {e: c for c, e in enumerate(edges)}
    rows = np.empty(3 * n_t, dtype=np.int64)
    cols = np.repeat(np.arange(n_t), 3)
    data = np.tile(np.array([1.0, -1.0, 1.0]), n_t)
    for c, (i, j, k) in enumerate(tris):
        rows[3 * c] = index[(i, j)]
        rows[3 * c + 1] = index[(i, k)]
        rows[3 * c + 2] = index[(j, k)]
    return coo_matrix((data, (rows, cols)), shape=(n_e, n_t)).tocsr()


def betti1(n_items: int, edges: Sequence[Edge], tris: Sequence[Triangle] | None = None) -> int:
    """The first Betti number of the clique complex: the dimension of the harmonic space.

    ``β₁ = |E| - rank(A) - rank(C)``, with ``rank(A) = V - components`` computed exactly from the
    connectivity and ``rank(C)`` from a dense rank per connected component. Splitting by component
    is what keeps it cheap: β₁ is additive over components and a corpus is one graph with thousands
    of small ones.

    This is a property of the comparison *design* and of nothing else. Two graders measured on the
    same set of pairs have the same β₁, and a grader whose harmonic mass is large is telling you to
    compare different pairs.
    """
    if not edges:
        return 0
    tris = list(tris) if tris is not None else triangles(edges)
    graph = incidence(n_items, edges)
    adjacency = (graph @ graph.T).tocsr()
    n_comp, labels = connected_components(adjacency, directed=False)
    total = 0
    by_component: dict[int, list[Edge]] = {}
    for a, b in edges:
        by_component.setdefault(int(labels[a]), []).append((a, b))
    tri_by_component: dict[int, list[Triangle]] = {}
    for t in tris:
        tri_by_component.setdefault(int(labels[t[0]]), []).append(t)
    for comp, comp_edges in by_component.items():
        comp_tris = tri_by_component.get(comp, [])
        vertices = {v for e in comp_edges for v in e}
        rank_a = len(vertices) - 1
        rank_c = 0
        if comp_tris:
            if len(comp_edges) * len(comp_tris) > _BETTI_DENSE_CAP:
                raise ValueError(
                    f"one connected component has {len(comp_edges)} edges and {len(comp_tris)} "
                    f"triangles, so its curl operator is too large for a dense rank. Split the "
                    f"comparison graph into the per-prompt tournaments it came from and take the "
                    f"sum, because β₁ is additive over connected components."
                )
            dense = curl_operator(comp_edges, comp_tris).toarray()
            rank_c = int(np.linalg.matrix_rank(dense))
        total += len(comp_edges) - rank_a - rank_c
    return total


# ---------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HodgeSplit:
    """The three components, their energies, their masses, and the checks that certify them.

    The components are kept as vectors, not only as norms. Null C correlates the curl component
    across edges between two halves of the replications and Null D eigendecomposes it, so a result
    type that returned three floats would make both of them impossible.

    ``orthogonality_residual`` and ``reconstruction_residual`` are the structural self-checks. Both
    sit at float64 zero for a correct decomposition, and `hodge_split` will not return a result that
    fails them: it escalates the solve instead.
    """

    n_items: int
    n_edges: int
    n_triangles: int
    betti1: int
    total_energy: float
    gradient_energy: float
    curl_energy: float
    harmonic_energy: float
    gradient_mass: float
    curl_mass: float
    harmonic_mass: float
    gradient: np.ndarray
    curl: np.ndarray
    harmonic: np.ndarray
    potential: np.ndarray
    solver: str
    orthogonality_residual: float
    reconstruction_residual: float

    @property
    def intransitive_mass(self) -> float:
        """``curl + harmonic``. Reported only beside its two parts, never instead of them."""
        return self.curl_mass + self.harmonic_mass

    def to_dict(self) -> dict[str, Any]:
        """A flat view for an Evidence payload. The split is first because it is the point."""
        return {
            "curl_mass": self.curl_mass,
            "harmonic_mass": self.harmonic_mass,
            "gradient_mass": self.gradient_mass,
            "intransitive_mass": self.intransitive_mass,
            "betti1": self.betti1,
            "n_items": self.n_items,
            "n_edges": self.n_edges,
            "n_triangles": self.n_triangles,
            "total_energy": self.total_energy,
            "curl_energy": self.curl_energy,
            "harmonic_energy": self.harmonic_energy,
            "gradient_energy": self.gradient_energy,
            "solver": self.solver,
            "orthogonality_residual": self.orthogonality_residual,
            "reconstruction_residual": self.reconstruction_residual,
        }


def _empty_split(n_items: int) -> HodgeSplit:
    z = np.zeros(0)
    return HodgeSplit(
        n_items=n_items,
        n_edges=0,
        n_triangles=0,
        betti1=0,
        total_energy=0.0,
        gradient_energy=0.0,
        curl_energy=0.0,
        harmonic_energy=0.0,
        gradient_mass=0.0,
        curl_mass=0.0,
        harmonic_mass=0.0,
        gradient=z,
        curl=z,
        harmonic=z,
        potential=np.zeros(n_items),
        solver="none: no edges",
        orthogonality_residual=0.0,
        reconstruction_residual=0.0,
    )


def _project(operator: csr_matrix, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Least-squares projection of ``target`` onto ``im(operator)``, sparse first, dense if needed.

    Both operators here are rank-deficient by construction: the gradient operator has the constants
    in its kernel, one per connected component, and the curl operator has every triangle relation.
    That is harmless for the projection, which is unique even where the coefficient vector is not,
    and LSQR started from zero converges to the minimum-norm solution, so the coefficients are
    well-defined too.

    What is not harmless is a solve that has not converged, because an under-converged projection
    silently moves energy from one component into another and every mass is then wrong in a way no
    reader can see. So the sparse result is checked against the defining condition of a least-squares
    projection, that the residual is orthogonal to the operator's range, and a failure escalates to a
    dense solve rather than being returned. The escalation is recorded on the result.
    """
    if operator.shape[1] == 0:
        return np.zeros(operator.shape[0]), np.zeros(0), "none: empty operator"
    scale = float(np.linalg.norm(target))
    if scale == 0.0:
        return np.zeros(operator.shape[0]), np.zeros(operator.shape[1]), "trivial: zero target"
    result = lsqr(operator, target, atol=1e-14, btol=1e-14, conlim=1e14, iter_lim=20000)
    coefficients = np.asarray(result[0], dtype=np.float64)
    projected = operator @ coefficients
    residual = target - projected
    gap = float(np.linalg.norm(operator.T @ residual))
    if gap <= STRUCTURAL_TOL * scale:
        return projected, coefficients, "scipy.sparse.linalg.lsqr"
    dense, *_ = np.linalg.lstsq(operator.toarray(), target, rcond=None)
    return operator @ dense, np.asarray(dense, dtype=np.float64), "numpy.linalg.lstsq (escalated)"


def hodge_split(
    n_items: int,
    edges: Sequence[Edge],
    flow: np.ndarray,
    *,
    with_betti: bool = True,
) -> HodgeSplit:
    """Split an edge flow into gradient, curl and harmonic, orthogonally.

    Three steps, in order. Project onto ``im(A.T)`` for the potential ``s``; the residual
    ``Y_perp`` is what no scalar can carry. Project ``Y_perp`` onto ``im(C)`` for the curl. What
    remains is harmonic: divergence-free, curl-free, and not a gradient.

    The two structural invariants are asserted before returning. Orthogonality of the components
    and exact reconstruction both follow from ``A @ C = 0``, so a violation is a bug in the
    operators and not a property of the data, and returning masses that look plausible while the
    operators are wrong is the single worst thing this function could do.
    """
    flow = np.asarray(flow, dtype=np.float64)
    if flow.shape != (len(edges),):
        raise ValueError(f"flow has shape {flow.shape}, expected ({len(edges)},) to match edges")
    if not len(edges):
        return _empty_split(n_items)

    tris = triangles(edges)
    grad_op = incidence(n_items, edges).T.tocsr()
    curl_op = curl_operator(edges, tris)

    y_grad, potential, solver_g = _project(grad_op, flow)
    perpendicular = flow - y_grad
    y_curl, _, solver_c = _project(curl_op, perpendicular)
    y_harm = perpendicular - y_curl

    e_total = float(flow @ flow)
    e_grad = float(y_grad @ y_grad)
    e_curl = float(y_curl @ y_curl)
    e_harm = float(y_harm @ y_harm)

    ortho = float(abs(y_grad @ y_curl) + abs(y_grad @ y_harm) + abs(y_curl @ y_harm))
    recon = float(np.linalg.norm(flow - (y_grad + y_curl + y_harm)))

    if e_total > 0.0:
        assert ortho <= STRUCTURAL_TOL * e_total, f"components not orthogonal: {ortho}"
        assert recon <= STRUCTURAL_TOL * (1.0 + e_total**0.5), f"reconstruction failed: {recon}"
        masses = (e_grad / e_total, e_curl / e_total, e_harm / e_total)
        assert abs(sum(masses) - 1.0) <= STRUCTURAL_TOL, f"masses sum to {sum(masses)}"
    else:
        masses = (0.0, 0.0, 0.0)

    return HodgeSplit(
        n_items=n_items,
        n_edges=len(edges),
        n_triangles=len(tris),
        betti1=betti1(n_items, edges, tris) if with_betti else -1,
        total_energy=e_total,
        gradient_energy=e_grad,
        curl_energy=e_curl,
        harmonic_energy=e_harm,
        gradient_mass=masses[0],
        curl_mass=masses[1],
        harmonic_mass=masses[2],
        gradient=y_grad,
        curl=y_curl,
        harmonic=y_harm,
        potential=potential,
        solver=f"gradient: {solver_g}; curl: {solver_c}",
        orthogonality_residual=ortho,
        reconstruction_residual=recon,
    )


def split_flow(flow: ComparisonFlow, *, with_betti: bool = True) -> HodgeSplit:
    """`hodge_split` on a `ComparisonFlow`, which is the call every caller actually wants."""
    return hodge_split(flow.n_items, flow.edges, flow.flow, with_betti=with_betti)


# ---------------------------------------------------------------------------
# B1, as an instrument
# ---------------------------------------------------------------------------

#: B1's envelope is unconditional, and the reason is worth stating rather than assuming. The
#: catalogue entry prints "Env at least 11 replications per pair for the nulls", which is a
#: sample-size requirement: it makes the *nulls* unavailable and it does not make the decomposition
#: wrong. So it is a refusal with a stated n, which the nulls return, rather than a regime
#: condition, which would withhold the reading.
CURL_MASS_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "the decomposition is exact linear algebra on an observed edge flow: the three components "
        "are orthogonal and their energies sum to the flow's energy in every regime, so no property "
        "of the run that produced the comparisons can make the split wrong. The eleven-replications "
        "requirement in the catalogue entry gates the nulls and is returned as a refusal naming n, "
        "which is a sample size and not a regime condition."
    ),
)


class CurlMass(BaseObservable):
    """B1. How much preference structure no scalar can carry, and how much of that is fixable.

    Construct with a `ComparisonFlow` and call `estimate`. The reading carries both masses, the
    first Betti number that bounds the harmonic one, and every null that could be run on the data
    supplied, each either with its result or with the refusal that says what it needed.

    Refusing is a normal outcome here and is not a failure of the instrument. A grader measured once
    per pair cannot support a split half, and the honest reading says so with the number eleven in
    it rather than reporting a curl mass as though a null had cleared it.
    """

    name = "CurlMass"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to: str | None = (
        "HodgeRank (Jiang, Lim, Yao, Ye 2011), combinatorial Hodge decomposition"
    )
    deviations: tuple[str, ...] = (
        "the default edge flow is the normalised win margin (w_b - w_a)/(w_a + w_b) rather than "
        "HodgeRank's log-odds, because the log-odds is infinite on a unanimous pair and unanimity "
        "is the common case rather than the exception. Pass scale='logit' for the canonical flow, "
        "which then carries a Haldane-Anscombe continuity correction of 0.5 per side and says so",
        "the decomposition is unweighted: HodgeRank weights each edge by its comparison count, "
        "which matters when replications are unequal across pairs. Equal weights reproduce the "
        "existing library result exactly and unequal-replication data should be read with that in "
        "mind",
        "the curl and harmonic masses are energy shares of the *observed* flow, so a comparison "
        "recorded as a bare win or loss carries an encoding curl that has no intransitivity in it. "
        "`nulls.transitive_baseline` measures that floor and it is reported beside the mass",
    )

    quantity = "grader.curl_mass"
    requires = {Component.GRADER: Access.QUERY}
    substrates = frozenset({Substrate.NEURAL_GEN, Substrate.PROCEDURAL})
    phases = frozenset({Phase.PRE_RUN})
    envelope = CURL_MASS_ENVELOPE
    invariance = "reward.affine, group.permutation"
    invariance_relation = INVARIANT
    baselines = (
        "a Bradley-Terry fit's residual, which is the gradient projection and therefore the "
        "complement of this reading rather than an independent check",
        "the random-profile null: uniformly random ordinal profiles through the identical pipeline",
        "a transitive-plus-measured-flip-rate simulation on the same comparison graph",
    )
    rung = 0

    #: Also produced by this instrument, and the reason the record lists two quantities.
    secondary_quantity = "grader.harmonic_mass"

    def __init__(
        self,
        flow: ComparisonFlow | None = None,
        *,
        nulls: Sequence[str] = ("C", "A", "D", "E"),
        replicates: Mapping[Edge, Sequence[int]] | None = None,
        seed: int = 0,
        n_draws: int = 200,
    ) -> None:
        self.flow = flow
        self.nulls = tuple(nulls)
        self.replicates = replicates
        self.seed = int(seed)
        self.n_draws = int(n_draws)
        self._computed: Any = None

    # -- the estimate path ----------------------------------------------------

    def compute(self) -> Any:
        """The split and every null the data can support. Returns a payload or a `Refusal`."""
        from reward_lens.measure.composition import nulls as null_bank

        if self.flow is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no comparison flow was supplied, so there is nothing to decompose",
                remedy=(
                    "pass `flow=edge_flow(pairs, n_items)` built from the grader's observed "
                    "comparisons. Each pair needs the two win counts; a k-wise tournament supplies "
                    "them directly."
                ),
            )
        if self.flow.n_edges == 0:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="the comparison flow has no edges, so no pair was ever compared",
                remedy=(
                    "supply at least one compared pair. A curl mass needs a filled triangle to be "
                    "nonzero at all, so three items compared to each other is the smallest design "
                    "that can report anything."
                ),
            )

        split = split_flow(self.flow)
        report: dict[str, Any] = {
            **split.to_dict(),
            "scale": self.flow.scale,
            "continuity_correction": self.flow.correction,
            "min_replications_per_pair": self.flow.min_replications,
            "unanimous_pair_fraction": self.flow.unanimous_fraction,
        }

        results: dict[str, Any] = {}
        if "C" in self.nulls:
            results["C_split_half"] = null_bank.split_half_reliability(
                self.flow, self.replicates, seed=self.seed
            )
        if "A" in self.nulls:
            results["A_parametric_bootstrap"] = null_bank.bradley_terry_bootstrap(
                self.flow, n_draws=self.n_draws, seed=self.seed
            )
        if "D" in self.nulls:
            results["D_curl_anisotropy"] = null_bank.curl_anisotropy(
                self.flow, split, n_draws=self.n_draws, seed=self.seed
            )
        if "E" in self.nulls:
            results["E_random_profile"] = null_bank.random_profile_baseline(
                self.flow, n_draws=self.n_draws, seed=self.seed
            )
            results["E_transitive_baseline"] = null_bank.transitive_baseline(
                self.flow, n_draws=self.n_draws, seed=self.seed
            )

        report["nulls"] = {
            key: (value.render() if isinstance(value, Refusal) else value.to_dict())
            for key, value in results.items()
        }
        report["nulls_refused"] = sorted(k for k, v in results.items() if isinstance(v, Refusal))
        report["rung"] = self._rung_reached(results)
        report["split_component"] = split
        return report

    def _rung_reached(self, results: Mapping[str, Any]) -> int:
        """Which rung of B1's ladder the data actually supported.

        r0 is the mass from a single pass, r1 adds Null C, r2 adds A, D and E. A null that refused
        does not count towards a rung, so a reading that says rung 0 is saying that the number has
        no null behind it, which is the fact a reader most needs and the one most often lost.
        """
        ran = {k for k, v in results.items() if not isinstance(v, Refusal)}
        if {"A_parametric_bootstrap", "D_curl_anisotropy", "E_random_profile"} <= ran:
            return 2
        if "C_split_half" in ran:
            return 1
        return 0

    def payload(self, computed: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in computed.items() if k != "split_component"}

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


def curl_mass_of_payload(payload: InvariancePayload) -> float:
    """The curl mass of the comparisons implied by an invariance payload's scores.

    The adapter the generated invariance test runs through. Scores within a group are read as an
    item's quality and every within-group pair becomes one comparison decided by which score is
    larger, which is what a tournament grader does. Two group actions then have honest and different
    meanings: `reward.affine` rescales the scores, which cannot change who wins a comparison because
    the scale factor is positive, so the curl mass is invariant; `group.permutation` relabels the
    items within a group, and the curl mass is a graph invariant, so it is invariant too and a
    failure would mean the instrument was reading rollout order.
    """
    payload.require("scores", "group_ids")
    scores = np.asarray(payload.scores, dtype=np.float64)
    gids = np.asarray(payload.group_ids)
    per_group: list[ComparisonFlow] = []
    for g in np.unique(gids):
        idx = np.flatnonzero(gids == g)
        if idx.size < 3:
            continue
        local = scores[idx]
        pairs = [
            PairCount(
                a, b, 1.0 if local[a] > local[b] else 0.0, 1.0 if local[b] > local[a] else 0.0
            )
            for a in range(idx.size)
            for b in range(a + 1, idx.size)
        ]
        per_group.append(edge_flow(pairs, idx.size))
    if not per_group:
        return 0.0
    return split_flow(disjoint_union(per_group), with_betti=False).curl_mass


__all__ = [
    "CURL_MASS_ENVELOPE",
    "HALDANE_ANSCOMBE",
    "MIN_REPLICATIONS_FOR_NULLS",
    "STRUCTURAL_TOL",
    "ComparisonFlow",
    "CurlMass",
    "Edge",
    "FlowScale",
    "HodgeSplit",
    "PairCount",
    "Triangle",
    "betti1",
    "curl_mass_of_payload",
    "curl_operator",
    "disjoint_union",
    "edge_flow",
    "hodge_split",
    "incidence",
    "split_flow",
    "triangles",
]

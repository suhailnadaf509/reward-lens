"""Combinatorial Hodge decomposition of pairwise preferences (S6, HORIZON H1, scoreboard T12).

This is the mathematical core of the preference-topology science. Given a set of pairwise
comparisons over ``n`` items (the responses of a tournament), we treat the comparisons as an edge
flow on a graph and split that flow into three orthogonal pieces with the combinatorial Hodge
decomposition of Jiang, Lim, Yao, and Ye ("Statistical ranking and combinatorial Hodge theory",
Mathematical Programming 127(1), 2011). The three pieces are the whole content of the result:

    GRADIENT   the transitive part, a flow that is the difference of a single scalar potential
               ``s`` over the items. This is exactly what a Bradley-Terry (scalar) reward model can
               represent, so its mass is the fraction of preference a scalar head can fit.
    CURL       the locally cyclic part, the piece supported on filled triangles. A rock-paper-
               scissors cycle among three mutually compared items lives here.
    HARMONIC   the globally cyclic part, a flow that is locally consistent (zero divergence and zero
               curl on every triangle) yet still not the gradient of any potential, because the
               comparison graph has a hole the flow wraps around.

The gradient is the reachable target of a scalar model; curl and harmonic together are the
``intransitive_mass``, the fraction of the observed preference that no scalar reward can express. A
nonzero intransitive mass is therefore a computable, coordinate-free lower bound on the error of any
scalar reward model on that preference data, which is the headline the science exists to report.

Operators and sign conventions. We orient every edge from its smaller to its larger item index.
``B1`` is the vertex-edge incidence (the boundary map from edges to vertices), of shape
``(n_items, n_edges)``, with a ``-1`` at the tail and ``+1`` at the head of each edge. Its transpose
is the discrete gradient: ``(B1.T @ s)[i, j] = s[j] - s[i]``, so the gradient subspace is
``im(B1.T)``. ``B2`` is the edge-triangle incidence (the boundary map from triangles to edges), of
shape ``(n_edges, n_triangles)``, built from the oriented boundary ``[j, k] - [i, k] + [i, j]`` of
each filled triangle ``(i, j, k)``. Its transpose is the discrete curl. The curl subspace is
``im(B2)`` and the harmonic subspace is ``ker(B1) ∩ ker(B2.T)``, the kernel of the 1-Laplacian
``B1.T @ B1 + B2 @ B2.T``. Because ``B1 @ B2 = 0`` exactly under these conventions, the three
subspaces are mutually orthogonal and span the whole edge space, so the squared norms of the three
components sum to the squared norm of the flow and the mass fractions sum to one.

**Where the number of record now comes from.** This module was written before B1 existed. B1
(``measure/composition/hodge.py``) is a superset of it: the same decomposition with a sparse solve
that escalates to a dense one when the projection's defining orthogonality condition is not met,
the first Betti number, the two flow scales, and the nulls. Two implementations of one piece of
mathematics is one too many to cite, so ``decompose_tournament`` now builds its flow with B1's
``edge_flow`` and splits it with B1's ``split_flow``, and everything downstream is unchanged.

The local solver below is kept, and kept exercised, because two independent implementations that
agree are worth more than one that is merely trusted. Measured on this study's own three corpora,
local against B1, at the corpus level:

    planted three-cycles (72 items, 72 edges)     curl mass differs by 2.2e-16
    planted chordless rings (22 items, 22 edges)  curl and harmonic mass differ by 0.0 exactly
    the synthetic judge corpus (360, 683, 501)    curl 8.3e-17, harmonic 3.5e-18, gradient 1.1e-16

Per tournament, which is the call that changed, the worst disagreement across all three corpora is
2.7e-15, on the small sparse judge tournaments where a single graph carries little energy. The
corpus headline moves from 0.257032716855589 to 0.25703271685558904, so no reported digit moves and
the repoint is a change of authority rather than of arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

from reward_lens.data.schema import Tournament
from reward_lens.measure.composition.hodge import PairCount, edge_flow, split_flow

Edge = tuple[int, int]
Triangle = tuple[int, int, int]

# Relative tolerance for the two structural self-checks (orthogonality and reconstruction). The
# decomposition is exact linear algebra, so the observed residuals are at the level of float64
# rounding; this bound is loose enough never to be a false alarm and tight enough to catch a genuine
# sign or indexing error in the operators.
_STRUCTURAL_TOL = 1e-8


@dataclass(frozen=True)
class HodgeDecomposition:
    """The result of a Hodge decomposition: three orthogonal mass fractions summing to one.

    ``gradient_mass``, ``curl_mass``, and ``harmonic_mass`` are the fractions of the flow's energy
    (its squared L2 norm) carried by the transitive, locally cyclic, and globally cyclic components.
    They sum to one whenever the flow carries any energy at all. ``intransitive_mass`` is the sum of
    the two cyclic fractions, the part no scalar reward can express. The ``*_energy`` fields are the
    raw squared norms the fractions are computed from, kept so a corpus can be aggregated by summing
    energies. ``orthogonality_residual`` and ``reconstruction_residual`` are the two structural
    self-checks: the first is the summed absolute cross-inner-product of the three components, the
    second is the norm of the flow minus the sum of its components. Both are near float64 zero for a
    correct decomposition.
    """

    n_items: int
    n_edges: int
    n_triangles: int
    total_energy: float
    gradient_energy: float
    curl_energy: float
    harmonic_energy: float
    gradient_mass: float
    curl_mass: float
    harmonic_mass: float
    orthogonality_residual: float
    reconstruction_residual: float

    @property
    def intransitive_mass(self) -> float:
        """The cyclic fraction ``curl + harmonic``: the part no scalar reward can express."""
        return self.curl_mass + self.harmonic_mass

    @property
    def transitive_mass(self) -> float:
        """The gradient fraction, an alias a scalar-reward reader will expect by that name."""
        return self.gradient_mass

    def to_dict(self) -> dict[str, float]:
        """A flat, JSON-friendly view for an Evidence value payload."""
        return {
            "gradient_mass": self.gradient_mass,
            "curl_mass": self.curl_mass,
            "harmonic_mass": self.harmonic_mass,
            "intransitive_mass": self.intransitive_mass,
            "total_energy": self.total_energy,
            "n_items": float(self.n_items),
            "n_edges": float(self.n_edges),
            "n_triangles": float(self.n_triangles),
            "orthogonality_residual": self.orthogonality_residual,
            "reconstruction_residual": self.reconstruction_residual,
        }


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def incidence_b1(n_items: int, edges: Sequence[Edge]) -> np.ndarray:
    """The vertex-edge incidence ``B1`` of shape ``(n_items, n_edges)`` (the boundary of edges).

    Each oriented edge ``(a, b)`` with ``a < b`` contributes ``-1`` at its tail ``a`` and ``+1`` at
    its head ``b``, so ``B1.T @ s`` is the discrete gradient ``s[b] - s[a]`` of a potential ``s``.
    """
    b1 = np.zeros((n_items, len(edges)), dtype=np.float64)
    for col, (a, b) in enumerate(edges):
        b1[a, col] = -1.0
        b1[b, col] = 1.0
    return b1


def enumerate_triangles(edges: Iterable[Edge]) -> list[Triangle]:
    """Every filled triangle: an ordered triple whose three edges are all present in ``edges``.

    A triangle is filled only when all three of its pairs were actually compared. Missing a pair
    leaves a hole in the complex, which is precisely what lets a globally cyclic (harmonic) flow
    exist, so the choice of which triangles are filled is not cosmetic; it decides the split between
    curl and harmonic.
    """
    edge_set = set(edges)
    vertices = sorted({v for e in edge_set for v in e})
    triangles: list[Triangle] = []
    for i, j, k in combinations(vertices, 3):
        if (i, j) in edge_set and (i, k) in edge_set and (j, k) in edge_set:
            triangles.append((i, j, k))
    return triangles


def triangle_b2(edges: Sequence[Edge], triangles: Sequence[Triangle]) -> np.ndarray:
    """The edge-triangle incidence ``B2`` of shape ``(n_edges, n_triangles)`` (boundary of triangles).

    Each triangle ``(i, j, k)`` with ``i < j < k`` has oriented boundary ``[j, k] - [i, k] + [i, j]``,
    so the column carries ``+1`` on edge ``(i, j)``, ``-1`` on ``(i, k)``, and ``+1`` on ``(j, k)``.
    With this convention ``B1 @ B2 = 0`` exactly, which is what makes the curl subspace ``im(B2)`` a
    subspace of the cycle space ``ker(B1)`` and the whole decomposition orthogonal.
    """
    index = {e: c for c, e in enumerate(edges)}
    b2 = np.zeros((len(edges), len(triangles)), dtype=np.float64)
    for col, (i, j, k) in enumerate(triangles):
        b2[index[(i, j)], col] += 1.0
        b2[index[(i, k)], col] -= 1.0
        b2[index[(j, k)], col] += 1.0
    return b2


# ---------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------


def hodge_decomposition(
    n_items: int, edges: Sequence[Edge], flow: np.ndarray
) -> HodgeDecomposition:
    """Split an edge flow into gradient, curl, and harmonic components and report their masses.

    The three projections are computed in sequence. The gradient component is the orthogonal
    projection of the flow onto ``im(B1.T)``, found by the least-squares potential ``s`` that best
    reproduces the flow. The residual after removing the gradient lies in the cycle space ``ker(B1)``.
    Its projection onto ``im(B2)`` is the curl, found by the least-squares triangle field. Whatever
    remains is the harmonic component, which is divergence-free and curl-free yet not a gradient. The
    function asserts the two structural invariants (orthogonality of the components and exact
    reconstruction) before returning, so a wrong operator can never silently produce plausible masses.
    """
    flow = np.asarray(flow, dtype=np.float64)
    n_edges = len(edges)
    if flow.shape != (n_edges,):
        raise ValueError(f"flow has shape {flow.shape}, expected ({n_edges},) to match edges")

    triangles = enumerate_triangles(edges)
    n_triangles = len(triangles)

    if n_edges == 0:
        return HodgeDecomposition(
            n_items=n_items,
            n_edges=0,
            n_triangles=0,
            total_energy=0.0,
            gradient_energy=0.0,
            curl_energy=0.0,
            harmonic_energy=0.0,
            gradient_mass=0.0,
            curl_mass=0.0,
            harmonic_mass=0.0,
            orthogonality_residual=0.0,
            reconstruction_residual=0.0,
        )

    b1 = incidence_b1(n_items, edges)
    grad_op = b1.T  # shape (n_edges, n_items): the discrete gradient
    potential, *_ = np.linalg.lstsq(grad_op, flow, rcond=None)
    y_grad = grad_op @ potential
    residual = flow - y_grad

    if n_triangles > 0:
        b2 = triangle_b2(edges, triangles)
        triangle_field, *_ = np.linalg.lstsq(b2, residual, rcond=None)
        y_curl = b2 @ triangle_field
    else:
        y_curl = np.zeros_like(flow)
    y_harm = residual - y_curl

    e_total = float(flow @ flow)
    e_grad = float(y_grad @ y_grad)
    e_curl = float(y_curl @ y_curl)
    e_harm = float(y_harm @ y_harm)

    ortho = float(abs(y_grad @ y_curl) + abs(y_grad @ y_harm) + abs(y_curl @ y_harm))
    recon = float(np.linalg.norm(flow - (y_grad + y_curl + y_harm)))

    if e_total > 0.0:
        gradient_mass = e_grad / e_total
        curl_mass = e_curl / e_total
        harmonic_mass = e_harm / e_total
        # Structural self-checks. The components must be mutually orthogonal and must reconstruct the
        # flow, both up to float64 rounding. A failure here is a bug in the operators, not the data.
        assert ortho <= _STRUCTURAL_TOL * e_total, f"components not orthogonal: {ortho}"
        assert recon <= _STRUCTURAL_TOL * (1.0 + e_total**0.5), f"reconstruction failed: {recon}"
        assert abs(gradient_mass + curl_mass + harmonic_mass - 1.0) <= _STRUCTURAL_TOL
    else:
        gradient_mass = curl_mass = harmonic_mass = 0.0

    return HodgeDecomposition(
        n_items=n_items,
        n_edges=n_edges,
        n_triangles=n_triangles,
        total_energy=e_total,
        gradient_energy=e_grad,
        curl_energy=e_curl,
        harmonic_energy=e_harm,
        gradient_mass=gradient_mass,
        curl_mass=curl_mass,
        harmonic_mass=harmonic_mass,
        orthogonality_residual=ortho,
        reconstruction_residual=recon,
    )


# ---------------------------------------------------------------------------
# Tournament adapter
# ---------------------------------------------------------------------------


def tournament_flow(tournament: Tournament) -> tuple[int, list[Edge], np.ndarray]:
    """Read a tournament's win counts as an oriented edge flow (the input to the decomposition).

    Every observed comparison is oriented from the smaller to the larger item index and its flow is
    the normalized win margin ``(wins_head - wins_tail) / (wins_head + wins_tail)``, which lives in
    ``[-1, 1]`` and is positive when the higher-indexed item is preferred. This is the standard
    HodgeRank edge flow for win/loss data: bounded, antisymmetric by construction, and defined
    without a logarithm so a shutout (five to zero) is a finite ``+/-1`` rather than an infinity.
    Repeated comparisons of the same pair accumulate, and a pair with no recorded wins is dropped.
    """
    accumulated: dict[Edge, list[float]] = {}
    for e in tournament.edges:
        if e.i < e.j:
            a, b, wins_a, wins_b = e.i, e.j, e.wins_i, e.wins_j
        else:
            a, b, wins_a, wins_b = e.j, e.i, e.wins_j, e.wins_i
        bucket = accumulated.setdefault((a, b), [0.0, 0.0])
        bucket[0] += float(wins_a)
        bucket[1] += float(wins_b)

    edges: list[Edge] = []
    flow: list[float] = []
    for edge in sorted(accumulated):
        wins_a, wins_b = accumulated[edge]
        total = wins_a + wins_b
        if total <= 0.0:
            continue
        edges.append(edge)
        flow.append((wins_b - wins_a) / total)
    return len(tournament.responses), edges, np.asarray(flow, dtype=np.float64)


def decompose_tournament(tournament: Tournament) -> HodgeDecomposition:
    """Decompose a single tournament's preference flow into its three Hodge components.

    Computed by B1, ``measure.composition.hodge``, and returned in this module's result type so no
    caller changes. The margin scale is B1's default and is the same edge flow ``tournament_flow``
    builds: ``(w_head - w_tail) / (w_head + w_tail)``, which is what any comparison against this
    study's published numbers has to use.

    ``with_betti=False`` because the local result type has never carried the first Betti number and
    computing one to discard it would make the repoint cost something. B1 reports it to callers that
    ask, and `read` does.
    """
    pairs = [
        PairCount(e.i, e.j, float(e.wins_i), float(e.wins_j))
        for e in tournament.edges
        if e.wins_i + e.wins_j > 0
    ]
    split = split_flow(edge_flow(pairs, len(tournament.responses)), with_betti=False)
    return HodgeDecomposition(
        n_items=split.n_items,
        n_edges=split.n_edges,
        n_triangles=split.n_triangles,
        total_energy=split.total_energy,
        gradient_energy=split.gradient_energy,
        curl_energy=split.curl_energy,
        harmonic_energy=split.harmonic_energy,
        gradient_mass=split.gradient_mass,
        curl_mass=split.curl_mass,
        harmonic_mass=split.harmonic_mass,
        orthogonality_residual=split.orthogonality_residual,
        reconstruction_residual=split.reconstruction_residual,
    )


def decompose_corpus(tournaments: Iterable[Tournament]) -> HodgeDecomposition:
    """Aggregate the decomposition over a corpus by summing component energies across tournaments.

    A corpus of independent tournaments is the disjoint union of their comparison complexes, and the
    Hodge decomposition of a disjoint union is the direct sum of the pieces, so the corpus-level mass
    fractions are exactly the summed component energies over the summed total energy. Summing energy
    rather than averaging per-tournament fractions gives larger and more-decisive tournaments their
    due weight, which is the honest corpus estimate.
    """
    e_total = e_grad = e_curl = e_harm = 0.0
    n_items = n_edges = n_triangles = 0
    ortho = recon = 0.0
    for tournament in tournaments:
        d = decompose_tournament(tournament)
        e_total += d.total_energy
        e_grad += d.gradient_energy
        e_curl += d.curl_energy
        e_harm += d.harmonic_energy
        n_items += d.n_items
        n_edges += d.n_edges
        n_triangles += d.n_triangles
        ortho = max(ortho, d.orthogonality_residual)
        recon = max(recon, d.reconstruction_residual)

    if e_total > 0.0:
        gradient_mass = e_grad / e_total
        curl_mass = e_curl / e_total
        harmonic_mass = e_harm / e_total
    else:
        gradient_mass = curl_mass = harmonic_mass = 0.0

    return HodgeDecomposition(
        n_items=n_items,
        n_edges=n_edges,
        n_triangles=n_triangles,
        total_energy=e_total,
        gradient_energy=e_grad,
        curl_energy=e_curl,
        harmonic_energy=e_harm,
        gradient_mass=gradient_mass,
        curl_mass=curl_mass,
        harmonic_mass=harmonic_mass,
        orthogonality_residual=ortho,
        reconstruction_residual=recon,
    )


__all__ = [
    "Edge",
    "Triangle",
    "HodgeDecomposition",
    "incidence_b1",
    "enumerate_triangles",
    "triangle_b2",
    "hodge_decomposition",
    "tournament_flow",
    "decompose_tournament",
    "decompose_corpus",
]

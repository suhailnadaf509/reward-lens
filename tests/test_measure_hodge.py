"""Unit tests: the Hodge decomposition, the four nulls, and the sup-norm bound.

Three constructions decide whether a Hodge implementation is correct, and nothing else does. A flow
that is literally the gradient of a potential must come back gradient mass one. A three-cycle with
its triangle filled must come back curl mass one. A chordless ring must come back harmonic mass one.
Get any of the three wrong and the masses still sum to one and still look plausible, which is why
they are here rather than a spot check on random data.

Everything after that is the property that makes the split meaningful (the three components are
mutually orthogonal and their masses sum to one, on arbitrary flows), the refusal that a design with
one comparison per pair has to produce, and the cross-check against the implementation this library
already shipped: the new sparse path and the existing dense one in `studies/s06_topology/hodge.py`
must agree, because the campaign's published number came out of the second one and a re-adjudication
that quietly changed the estimator would be adjudicating a different quantity.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.composition.hodge import (
    MIN_REPLICATIONS_FOR_NULLS,
    ComparisonFlow,
    CurlMass,
    PairCount,
    betti1,
    curl_mass_of_payload,
    curl_operator,
    disjoint_union,
    edge_flow,
    hodge_split,
    incidence,
    split_flow,
    triangles,
)
from reward_lens.measure.composition.nulls import (
    bradley_terry_bootstrap,
    bradley_terry_scores,
    curl_anisotropy,
    random_profile_baseline,
    split_half_reliability,
    transitive_baseline,
    transitive_curl_mass_complete,
)
from reward_lens.measure.composition.sepnorm import (
    SepDistance,
    column_mean_bound,
    contraction_factor,
    sep_distance,
    triple_scan,
)


def complete(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def sign_pairs(n: int, order: np.ndarray) -> list[PairCount]:
    """Every pair of ``n`` items decided once by a total order, as ±1 win counts."""
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    return [
        PairCount(a, b, 1.0 if rank[a] > rank[b] else 0.0, 1.0 if rank[b] > rank[a] else 0.0)
        for a in range(n)
        for b in range(a + 1, n)
    ]


def replicated(n_items: int, kernel: np.ndarray, reps: int, seed: int) -> ComparisonFlow:
    """A flow from ``reps`` binomial replications of every pair at a stated preference kernel."""
    rng = np.random.default_rng(seed)
    pairs = []
    for a in range(n_items):
        for b in range(a + 1, n_items):
            wins_b = int(rng.binomial(reps, 1.0 / (1.0 + np.exp(-kernel[a, b]))))
            pairs.append(PairCount(a, b, reps - wins_b, wins_b))
    return edge_flow(pairs, n_items)


def scalar_kernel(rng: np.random.Generator, n: int, spread: float = 1.2) -> np.ndarray:
    """A Bradley-Terry world: the kernel is a difference of scalars, so its curl is only noise."""
    s = rng.standard_normal(n) * spread
    return s[None, :] - s[:, None]


def cyclic_kernel(rng: np.random.Generator, n: int, strength: float = 2.5) -> np.ndarray:
    """A genuinely non-scalar grader: a scalar part plus a skew-symmetric interaction term."""
    s = rng.standard_normal(n) * 1.2
    raw = rng.standard_normal((n, n))
    skew = raw - raw.T
    skew = skew / np.linalg.norm(skew) * strength * n**0.5
    return (s[None, :] - s[:, None]) + skew


# ---------------------------------------------------------------------------
# The three constructions whose answer is known exactly
# ---------------------------------------------------------------------------


def test_a_pure_gradient_flow_is_all_gradient():
    """A flow that is the gradient of a potential has curl mass and harmonic mass exactly zero."""
    n = 6
    edges = complete(n)
    potential = np.array([0.0, 1.0, 3.0, 4.0, 9.0, -2.0])
    flow = np.array([potential[j] - potential[i] for i, j in edges])

    split = hodge_split(n, edges, flow)

    assert split.gradient_mass == pytest.approx(1.0, abs=1e-12)
    assert split.curl_mass == pytest.approx(0.0, abs=1e-12)
    assert split.harmonic_mass == pytest.approx(0.0, abs=1e-12)
    # The recovered potential reproduces the flow, up to the constant that is in the kernel.
    recovered = split.potential
    assert np.allclose(recovered - recovered[0], potential - potential[0], atol=1e-8)


def test_a_pure_three_cycle_is_all_curl():
    """A > B > C > A on three items compared to each other is curl mass one, and no harmonic."""
    edges = [(0, 1), (1, 2), (0, 2)]
    flow = np.array([1.0, 1.0, -1.0])  # 0<1, 1<2, 2<0

    split = hodge_split(3, edges, flow)

    assert split.curl_mass == pytest.approx(1.0, abs=1e-12)
    assert split.gradient_mass == pytest.approx(0.0, abs=1e-12)
    assert split.harmonic_mass == pytest.approx(0.0, abs=1e-12)
    assert split.n_triangles == 1
    assert split.betti1 == 0  # the triangle is filled, so there is no hole


@pytest.mark.parametrize("length", [4, 5, 6, 7])
def test_a_chordless_ring_is_all_harmonic(length: int):
    """A ring with no interior pair compared has a hole, and a flow around it is pure harmonic.

    This is the construction the curl-harmonic split lives or dies on. The flow is locally
    consistent (every filled triangle, and there are none, has zero circulation) and still not the
    gradient of any potential, because the comparison design left a hole.
    """
    edges = [(step, step + 1) for step in range(length - 1)] + [(0, length - 1)]
    flow = np.array([1.0] * (length - 1) + [-1.0])

    split = hodge_split(length, edges, flow)

    assert split.harmonic_mass == pytest.approx(1.0, abs=1e-12)
    assert split.curl_mass == pytest.approx(0.0, abs=1e-12)
    assert split.gradient_mass == pytest.approx(0.0, abs=1e-12)
    assert split.n_triangles == 0
    assert split.betti1 == 1


def test_filling_the_ring_moves_the_mass_from_harmonic_to_curl():
    """The same cyclic preference is harmonic on a ring and curl once the pairs are compared.

    The point of the split in one test. Nothing about the grader changed between these two lines;
    the experimenter compared two more pairs. Harmonic mass is a property of the design.
    """
    ring = [(0, 1), (1, 2), (2, 3), (0, 3)]
    ring_flow = np.array([1.0, 1.0, 1.0, -1.0])
    open_split = hodge_split(4, ring, ring_flow)

    filled = ring + [(0, 2), (1, 3)]
    filled_flow = np.concatenate([ring_flow, [0.0, 0.0]])
    closed_split = hodge_split(4, filled, filled_flow)

    assert open_split.harmonic_mass == pytest.approx(1.0, abs=1e-12)
    assert open_split.betti1 == 1
    assert closed_split.betti1 == 0
    assert closed_split.harmonic_mass == pytest.approx(0.0, abs=1e-12)
    assert closed_split.curl_mass > 0.4


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=40)
@given(
    n_items=st.integers(min_value=3, max_value=9),
    seed=st.integers(min_value=0, max_value=10_000),
    drop=st.floats(min_value=0.0, max_value=0.5),
)
def test_the_three_masses_sum_to_one_and_the_components_are_orthogonal(
    n_items: int, seed: int, drop: float
):
    """On any flow on any comparison graph: masses sum to one, components are mutually orthogonal."""
    rng = np.random.default_rng(seed)
    edges = [e for e in complete(n_items) if rng.random() >= drop]
    if not edges:
        return
    flow = rng.standard_normal(len(edges))
    split = hodge_split(n_items, edges, flow)

    total = split.gradient_mass + split.curl_mass + split.harmonic_mass
    assert total == pytest.approx(1.0, abs=1e-9)
    assert split.orthogonality_residual <= 1e-8 * max(split.total_energy, 1.0)
    assert split.reconstruction_residual <= 1e-8 * (1.0 + split.total_energy**0.5)
    assert abs(float(split.gradient @ split.curl)) <= 1e-8 * max(split.total_energy, 1.0)
    assert abs(float(split.gradient @ split.harmonic)) <= 1e-8 * max(split.total_energy, 1.0)
    assert abs(float(split.curl @ split.harmonic)) <= 1e-8 * max(split.total_energy, 1.0)


@settings(deadline=None, max_examples=30)
@given(
    n_items=st.integers(min_value=3, max_value=8),
    seed=st.integers(min_value=0, max_value=10_000),
    drop=st.floats(min_value=0.0, max_value=0.5),
)
def test_the_operators_compose_to_zero_and_betti_counts_the_harmonic_dimension(
    n_items: int, seed: int, drop: float
):
    """``A @ C = 0`` is what makes the split orthogonal, and β₁ is the harmonic dimension."""
    rng = np.random.default_rng(seed)
    edges = [e for e in complete(n_items) if rng.random() >= drop]
    if not edges:
        return
    tris = triangles(edges)
    a = incidence(n_items, edges)
    c = curl_operator(edges, tris)
    assert np.allclose((a @ c).toarray(), 0.0, atol=1e-12)

    # β₁ is the dimension of ker(Δ₁), so a harmonic component can be nonzero only when β₁ is.
    dimension = betti1(n_items, edges, tris)
    if dimension == 0:
        flow = rng.standard_normal(len(edges))
        assert hodge_split(n_items, edges, flow).harmonic_energy <= 1e-16 * max(
            float(flow @ flow), 1.0
        )


@settings(deadline=None, max_examples=25)
@given(n_items=st.integers(min_value=3, max_value=10), seed=st.integers(0, 10_000))
def test_a_transitive_sign_tournament_hits_the_closed_form(n_items: int, seed: int):
    """``(n-2)/(3n)``, exactly, for every total order. This is the encoding floor, not intransitivity.

    The number the campaign's 0.214 has to be read against. A perfectly transitive grader recorded
    as wins and losses on a complete graph already carries this much curl, because a sign is not a
    margin and a ±1 flow is not the gradient of anything.
    """
    order = np.random.default_rng(seed).permutation(n_items)
    flow = edge_flow(sign_pairs(n_items, order), n_items)
    split = split_flow(flow)
    assert split.curl_mass == pytest.approx(transitive_curl_mass_complete(n_items), abs=1e-12)
    assert split.harmonic_mass == pytest.approx(0.0, abs=1e-12)


def test_the_sparse_path_agrees_with_the_shipped_dense_implementation():
    """The new sparse decomposition and `studies/s06_topology/hodge.py` return the same masses.

    The campaign's published `intransitive_mass` came out of the dense implementation, so a
    re-adjudication has to be running the same estimator on the same data rather than a new one that
    happens to be nearby.
    """
    from studies.s06_topology.hodge import hodge_decomposition

    rng = np.random.default_rng(7)
    for n_items in (4, 6, 9):
        for drop in (0.0, 0.2, 0.4):
            edges = [e for e in complete(n_items) if rng.random() >= drop]
            if not edges:
                continue
            flow = rng.standard_normal(len(edges))
            mine = hodge_split(n_items, edges, flow)
            theirs = hodge_decomposition(n_items, edges, flow)
            assert mine.gradient_mass == pytest.approx(theirs.gradient_mass, abs=1e-10)
            assert mine.curl_mass == pytest.approx(theirs.curl_mass, abs=1e-10)
            assert mine.harmonic_mass == pytest.approx(theirs.harmonic_mass, abs=1e-10)


def test_the_disjoint_union_is_the_direct_sum():
    """Decomposing a corpus at once equals summing its tournaments' energies, which is why it is done once."""
    rng = np.random.default_rng(3)
    flows = []
    for k in range(6):
        n = 4 + k % 3
        pairs = [
            PairCount(a, b, float(rng.integers(0, 6)), float(rng.integers(0, 6)))
            for a in range(n)
            for b in range(a + 1, n)
        ]
        flows.append(edge_flow(pairs, n))
    pieces = [split_flow(f) for f in flows]
    union = split_flow(disjoint_union(flows))
    assert union.curl_energy == pytest.approx(sum(p.curl_energy for p in pieces), rel=1e-9)
    assert union.harmonic_energy == pytest.approx(sum(p.harmonic_energy for p in pieces), abs=1e-9)
    assert union.betti1 == sum(p.betti1 for p in pieces)


# ---------------------------------------------------------------------------
# The edge flow, and the logit's divergence
# ---------------------------------------------------------------------------


def test_the_logit_flow_is_finite_on_a_unanimous_pair_and_says_which_correction_did_it():
    """The whole reason the correction exists, and the reason it is recorded on the flow.

    Without a continuity correction the log-odds flow is infinite on any pair where one side won
    every comparison. On the comparison data this library holds that is every pair, so an
    uncorrected logit would return an array of infinities and the decomposition would return NaN.
    """
    pairs = [PairCount(0, 1, 11.0, 0.0), PairCount(1, 2, 0.0, 11.0), PairCount(0, 2, 11.0, 0.0)]
    flow = edge_flow(pairs, 3, scale="logit")

    assert np.all(np.isfinite(flow.flow))
    assert "Haldane-Anscombe" in flow.correction
    assert "0.5" in flow.correction
    # The margin scale is finite too, and says so rather than staying silent about a non-issue.
    margin = edge_flow(pairs, 3)
    assert margin.correction.startswith("none")
    assert margin.unanimous_fraction == pytest.approx(1.0)


def test_repeated_comparisons_of_a_pair_accumulate_and_uncompared_pairs_are_left_out():
    flow = edge_flow(
        [PairCount(0, 1, 3.0, 1.0), PairCount(1, 0, 2.0, 0.0), PairCount(0, 2, 0.0, 0.0)], 3
    )
    assert flow.edges == ((0, 1),)
    assert flow.replications[0] == pytest.approx(6.0)
    assert flow.flow[0] == pytest.approx((1.0 + 2.0 - 3.0) / 6.0)


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


def test_null_c_refuses_below_eleven_replications_and_names_eleven_and_the_remedy():
    """Standing rule: fewer than eleven replications per pair is a refusal with the n in it."""
    order = np.arange(4)
    flow = edge_flow(sign_pairs(4, order), 4)  # one comparison per pair
    result = split_half_reliability(flow)

    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert str(MIN_REPLICATIONS_FOR_NULLS) in result.detail
    assert "GRADER:REPLICATE" in result.remedy
    assert result.statistics["required_replications"] == MIN_REPLICATIONS_FOR_NULLS
    assert result.statistics["observed_min_replications"] == pytest.approx(1.0)


def test_null_a_refuses_below_eleven_replications_too():
    flow = edge_flow(sign_pairs(5, np.arange(5)), 5)
    result = bradley_terry_bootstrap(flow)
    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert str(MIN_REPLICATIONS_FOR_NULLS) in result.detail


def test_null_c_runs_at_exactly_eleven_and_not_at_ten():
    rng = np.random.default_rng(11)
    kernel = cyclic_kernel(rng, 6)
    assert isinstance(split_half_reliability(replicated(6, kernel, 10, seed=1)), Refusal)
    assert not isinstance(split_half_reliability(replicated(6, kernel, 11, seed=1)), Refusal)


def test_the_instrument_refuses_with_no_flow_and_with_an_empty_one():
    empty = CurlMass().compute()
    assert isinstance(empty, Refusal)
    assert "no comparison flow" in empty.detail
    assert empty.remedy.strip()

    nothing = CurlMass(edge_flow([], 4)).compute()
    assert isinstance(nothing, Refusal)
    assert "no edges" in nothing.detail


def test_null_d_refuses_when_there_is_no_curl_to_have_a_spectrum():
    ring = edge_flow(
        [
            PairCount(0, 1, 0, 1),
            PairCount(1, 2, 0, 1),
            PairCount(2, 3, 0, 1),
            PairCount(0, 3, 1, 0),
        ],
        4,
    )
    result = curl_anisotropy(ring)
    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.BELOW_LOD
    assert "harmonic" in result.remedy


def test_sep_distance_refuses_when_no_triple_is_fully_observed():
    path = edge_flow([PairCount(0, 1, 0, 1), PairCount(1, 2, 0, 1)], 3)
    result = sep_distance(path)
    assert isinstance(result, Refusal)
    assert "no filled triangle" in result.detail
    assert result.remedy.strip()


# ---------------------------------------------------------------------------
# The nulls, on data whose answer is known by construction
# ---------------------------------------------------------------------------


def test_null_c_separates_a_scalar_grader_from_a_cyclic_one():
    """The decisive test of the decisive null, with its positive control checked in both arms.

    Under a Bradley-Terry truth at eleven replications the observed curl does not replicate; under a
    kernel with a real skew-symmetric part it does. The separation is what this test is for and it
    is unchanged: 0.19 against 0.72.

    **What changed is the verdict on the scalar arm, and the old assertion was wrong.** It read
    ``assert noise.verdict == "does_not_replicate"``, whose note says in as many words that the
    cyclic mass is measurement error. The gradient control does not license that: ``im(grad)`` has
    dimension 150 on this design where ``im(C)`` has 300, so it replicates better at equal signal
    energy for reasons that have nothing to do with the grader. The matched control settles it: a
    curl planted at exactly the observed energy, in the same subspace, through the same sampling
    noise, replicates at only 0.4395 here. A design that cannot see a curl of that size has not
    shown that this one is absent, so the verdict is ``underpowered`` and it names which control
    failed.
    """
    rng = np.random.default_rng(1)
    scalar = disjoint_union(
        [replicated(6, scalar_kernel(rng, 6), 11, seed=100 + k) for k in range(30)]
    )
    cyclic = disjoint_union(
        [replicated(6, cyclic_kernel(rng, 6), 11, seed=200 + k) for k in range(30)]
    )

    noise = split_half_reliability(scalar, n_splits=30, seed=0)
    real = split_half_reliability(cyclic, n_splits=30, seed=0)

    assert noise.gradient_reliability > 0.7, "the gradient control holds in both arms"
    assert real.gradient_reliability > 0.7
    assert real.curl_reliability > noise.curl_reliability + 0.3
    assert real.verdict == "replicates"

    assert noise.curl_subspace_dim == 300
    assert noise.gradient_subspace_dim == 150
    assert noise.planted_curl_reliability < 0.7 < noise.gradient_reliability, (
        "the asymmetry the gradient control hides: a planted curl of the observed size does not "
        "replicate on this design even though the gradient does"
    )
    assert noise.verdict == "underpowered"
    assert "matched control" in noise.note
    assert real.planted_curl_reliability >= 0.7, (
        "the cyclic arm's control passes, so its verdict counts"
    )


def test_the_does_not_replicate_branch_is_reachable_once_the_design_can_see_a_curl():
    """The matched control is a power statement, not a blanket veto, and this is its turn-on point.

    The same scalar grader at rising replication: the branch that says "this cyclic mass is
    measurement error" opens exactly when a planted curl of the observed size starts to replicate.
    Measured at 0.4395 planted at eleven replications, 0.5397 at 31, 0.7230 at 101. At 301 the
    observed curl itself replicates at 0.8306, which is correct and is not a failure of the null: a
    scalar grader on the margin scale carries a deterministic encoding curl, so once the sampling
    noise is small enough that curl reproduces exactly. It is `transitive_baseline` and not Null C
    that says such a curl carries no intransitivity.
    """
    rng = np.random.default_rng(1)
    at_101 = disjoint_union(
        [replicated(6, scalar_kernel(rng, 6), 101, seed=100 + k) for k in range(30)]
    )
    result = split_half_reliability(at_101, n_splits=20, seed=0)
    assert result.planted_curl_reliability > 0.7
    assert result.curl_reliability < 0.7
    assert result.verdict == "does_not_replicate"
    assert "planted at the same energy" in result.note


def test_null_c_reports_underpowered_rather_than_a_verdict_when_the_control_fails():
    """A null claim with no working positive control is not evidence, and the verdict says so."""
    rng = np.random.default_rng(5)
    # A grader with almost no signal at all: neither component replicates, so neither arm is
    # informative and the honest report is that the design cannot tell.
    flat = disjoint_union(
        [replicated(6, scalar_kernel(rng, 6, spread=0.02), 11, seed=300 + k) for k in range(8)]
    )
    result = split_half_reliability(flat, n_splits=20, seed=0)
    assert result.verdict == "underpowered"
    assert "positive control failed" in result.note


def test_null_a_reports_an_excess_near_zero_under_its_own_null():
    """A Bradley-Terry world resimulated through the identical pipeline has no excess curl."""
    rng = np.random.default_rng(2)
    flow = disjoint_union(
        [replicated(6, scalar_kernel(rng, 6), 11, seed=400 + k) for k in range(20)]
    )
    result = bradley_terry_bootstrap(flow, n_draws=80, seed=0)
    assert abs(result.curl_excess) < 0.05
    assert result.curl_p_value > 0.05
    assert result.curl_null_ci[0] <= result.curl_null_mean <= result.curl_null_ci[1]


def test_null_a_finds_the_excess_when_the_grader_is_genuinely_cyclic():
    rng = np.random.default_rng(2)
    flow = disjoint_union(
        [replicated(6, cyclic_kernel(rng, 6), 11, seed=500 + k) for k in range(20)]
    )
    result = bradley_terry_bootstrap(flow, n_draws=80, seed=0)
    assert result.curl_excess > 0.05
    assert result.curl_p_value < 0.05


def test_the_bradley_terry_fit_is_finite_under_complete_separation():
    """One unanimous comparison per pair is where an unpenalised MLE runs off to infinity."""
    flow = edge_flow(sign_pairs(6, np.arange(6)), 6)
    scores, iterations = bradley_terry_scores(flow)
    assert np.all(np.isfinite(scores))
    assert iterations >= 1
    # A total order is recovered in the right order, which is the fit doing its job.
    assert list(np.argsort(scores)) == list(range(6))


def test_null_d_finds_a_direction_only_when_the_curl_has_one():
    """Isotropic noise spreads over the curl subspace; two competing criteria concentrate in a plane."""
    rng = np.random.default_rng(3)
    n = 16
    u, v = rng.standard_normal(n), rng.standard_normal(n)
    plane = np.outer(u, v) - np.outer(v, u)
    plane = plane / np.linalg.norm(plane) * 3.0 * n**0.5
    two_criteria = replicated(n, (np.zeros((n, n)) + plane), 11, seed=9)
    isotropic = replicated(n, cyclic_kernel(rng, n, strength=3.0), 11, seed=9)

    directed = curl_anisotropy(two_criteria, n_draws=150, seed=0)
    spread = curl_anisotropy(isotropic, n_draws=150, seed=0)

    assert directed.p_value < 0.05
    assert spread.p_value > 0.05
    assert directed.leading_plane_fraction > spread.leading_plane_fraction
    assert directed.top_plane.shape == (n, 2)


def test_null_e_transitive_baseline_recovers_the_closed_form_and_needs_no_replications():
    """The one null that runs on singly-observed data, checked against the exact value."""
    flow = edge_flow(sign_pairs(7, np.arange(7)), 7)
    result = transitive_baseline(flow, n_draws=40, seed=0)
    assert not isinstance(result, Refusal)
    assert result.curl_null_mean == pytest.approx(transitive_curl_mass_complete(7), abs=1e-12)
    assert result.curl_excess == pytest.approx(0.0, abs=1e-12)


def test_null_e_random_profile_matches_the_transitive_baseline_at_one_replication():
    """Stated in the module docstring and worth a test: at R = 1 the classic null separates nothing.

    A single uniformly random total order is transitive, so the incoherent-preference null and the
    perfectly-coherent-grader null are the same distribution when every pair was decided once.
    """
    flow = edge_flow(sign_pairs(6, np.arange(6)), 6)
    coherent = transitive_baseline(flow, n_draws=40, seed=0)
    incoherent = random_profile_baseline(flow, n_draws=40, seed=0)
    assert coherent.curl_null_mean == pytest.approx(incoherent.curl_null_mean, abs=1e-12)


def test_the_random_profile_null_only_adds_spread_to_the_transitive_baseline():
    """Even at eleven replications the classic null centres on the encoding floor and is weaker.

    Averaging a profile of random total orders does not make the aggregate less curly, because every
    member of the profile is itself a transitive tournament that carries the encoding curl. So the
    random profile shares the transitive baseline's centre and adds variance, which makes it the
    less powerful of the two wherever both apply. Recorded here because it is the opposite of what
    the classic null is usually assumed to do.
    """
    rng = np.random.default_rng(4)
    flow = disjoint_union(
        [replicated(6, scalar_kernel(rng, 6), 11, seed=600 + k) for k in range(10)]
    )
    coherent = transitive_baseline(flow, n_draws=40, seed=0)
    incoherent = random_profile_baseline(flow, n_draws=40, seed=0)

    # The transitive baseline at a zero flip rate is a point mass at the closed form, at every R.
    assert coherent.curl_null_mean == pytest.approx(transitive_curl_mass_complete(6), abs=1e-12)
    assert coherent.curl_null_ci[1] - coherent.curl_null_ci[0] < 1e-9
    # The random profile has the same centre and real spread.
    assert abs(incoherent.curl_null_mean - coherent.curl_null_mean) < 0.02
    assert incoherent.curl_null_ci[1] - incoherent.curl_null_ci[0] > 0.02


# ---------------------------------------------------------------------------
# B5
# ---------------------------------------------------------------------------


def test_the_sandwich_brackets_the_true_value_on_a_kernel_whose_answer_is_known():
    """A pure three-cycle at unit flow: Δ = 3, so the sandwich is [1, 1] and it is exact."""
    flow = edge_flow([PairCount(0, 1, 0, 1), PairCount(1, 2, 0, 1), PairCount(0, 2, 1, 0)], 3)
    bound = sep_distance(flow)
    assert isinstance(bound, SepDistance)
    delta, worst = triple_scan(flow)
    assert delta == pytest.approx(3.0)
    assert worst == (0, 1, 2)
    assert bound.lower == pytest.approx(1.0)
    assert bound.upper == pytest.approx(1.0)
    assert bound.lower <= bound.upper


def test_a_scalar_representable_kernel_converges_in_one_step():
    """The theorem, at its own boundary: ``‖U‖_⊕ = 0`` gives ``c = 0``."""
    n = 5
    potential = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    edges = tuple((a, b) for a in range(n) for b in range(a + 1, n))
    # Built directly rather than through `edge_flow`, because a kernel that is exactly a difference
    # of scalars is not something win counts can express: the margin scale is bounded at ±1.
    flow = ComparisonFlow(
        n_items=n,
        edges=edges,
        flow=np.array([potential[b] - potential[a] for a, b in edges]),
        replications=np.ones(len(edges)),
        scale="margin",
        correction="constructed for this test, not aggregated from counts",
    )
    bound = sep_distance(flow)
    assert bound.delta == pytest.approx(0.0, abs=1e-12)
    assert bound.upper == pytest.approx(0.0, abs=1e-12)
    assert bound.contraction_upper == pytest.approx(0.0, abs=1e-12)
    assert contraction_factor(0.0) == 0.0


@settings(deadline=None, max_examples=30)
@given(n_items=st.integers(min_value=3, max_value=8), seed=st.integers(0, 10_000))
def test_the_sandwich_is_ordered_and_the_column_mean_bound_is_an_upper_bound(
    n_items: int, seed: int
):
    """``lower ≤ upper`` always, and the O(n²) bound never falls below Δ/3."""
    rng = np.random.default_rng(seed)
    pairs = [
        PairCount(a, b, float(rng.integers(0, 12)), float(rng.integers(0, 12)))
        for a in range(n_items)
        for b in range(a + 1, n_items)
    ]
    flow = edge_flow(pairs, n_items)
    if flow.n_edges == 0:
        return
    bound = sep_distance(flow)
    if isinstance(bound, Refusal):
        return
    assert bound.lower <= bound.upper + 1e-12
    assert column_mean_bound(flow) >= bound.delta / 3.0 - 1e-12
    assert bound.contraction_lower <= bound.contraction_upper + 1e-12


def test_b5_carries_b1s_curl_mass_in_its_own_payload():
    """The pair cannot be separated by a card that reads one field: the bound holds both."""
    rng = np.random.default_rng(6)
    flow = replicated(7, cyclic_kernel(rng, 7), 11, seed=1)
    bound = sep_distance(flow)
    assert bound.curl_mass == pytest.approx(split_flow(flow).curl_mass)
    assert "curl mass" in bound.note


def test_contraction_factor_is_the_stated_minimum():
    for d in (0.0, 0.05, 0.31, 0.7, 1.0, 3.0):
        assert contraction_factor(d) == pytest.approx(min(1.0 - np.exp(-2.0 * d), d))


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


def test_the_instrument_reports_the_rung_the_data_supported():
    """Rung 0 with no null behind it is what singly-observed data earns, and the reading says so."""
    singly = CurlMass(edge_flow(sign_pairs(7, np.arange(7)), 7), n_draws=20)
    report = singly.compute()
    assert report["rung"] == 0
    assert "C_split_half" in report["nulls_refused"]
    assert "A_parametric_bootstrap" in report["nulls_refused"]
    assert "E_transitive_baseline" not in report["nulls_refused"]

    rng = np.random.default_rng(8)
    replicated_flow = disjoint_union(
        [replicated(6, cyclic_kernel(rng, 6), 11, seed=700 + k) for k in range(8)]
    )
    full = CurlMass(replicated_flow, n_draws=20).compute()
    assert full["rung"] == 2
    assert full["nulls_refused"] == []


def test_the_payload_reports_both_masses_separately_and_the_sum_only_beside_them():
    rng = np.random.default_rng(9)
    flow = disjoint_union(
        [replicated(6, cyclic_kernel(rng, 6), 11, seed=800 + k) for k in range(4)]
    )
    report = CurlMass(flow, nulls=(), n_draws=5).compute()
    assert "curl_mass" in report
    assert "harmonic_mass" in report
    assert report["intransitive_mass"] == pytest.approx(
        report["curl_mass"] + report["harmonic_mass"]
    )
    assert report["betti1"] >= 0
    assert report["min_replications_per_pair"] == pytest.approx(11.0)


def test_curl_mass_of_a_payload_reads_scores_as_comparisons():
    """The adapter the generated invariance test runs through, checked against a direct call."""
    from reward_lens.core.invariance import InvariancePayload

    rng = np.random.default_rng(10)
    scores = rng.standard_normal(24)
    groups = np.repeat(np.arange(4), 6)
    direct = curl_mass_of_payload(InvariancePayload(scores=scores, group_ids=groups))
    assert 0.0 <= direct <= 1.0
    # Four disjoint transitive six-item tournaments: the closed form, again.
    assert direct == pytest.approx(transitive_curl_mass_complete(6), abs=1e-10)

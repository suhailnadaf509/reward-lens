"""Acceptance: B1 and B5, and the re-adjudication of this project's own 0.214.

The clause this file discharges: *the split-half null runs and its verdict is recorded; the existing
0.214 is re-adjudicated against it and split into curl and harmonic.*

All three parts are here and the verdict is not the one the card records.

**The split-half null runs, and it refuses.** Every pair in the campaign's stored preference data was
compared exactly once: 130,129 edges across 10,000 tournaments, and ``wins_i + wins_j == 1`` on every
one of them. A split half needs at least eleven replications per pair and there is one, so Null C
returns a refusal naming eleven and naming the access level that would supply them. That is the
verdict and it is recorded below with its numbers. It is also not a defect in the instrument: the
same null, on the same code path, separates a genuinely cyclic grader from a Bradley-Terry one at
eleven replications, and that separation is tested here so the refusal is visibly a property of the
data rather than of the implementation.

**The 0.214 is reproduced and split.** ``curl_mass = 0.2139761313773251`` against the stored
``0.2139761313773256``, and ``harmonic_mass = 1.8e-30``, which is float zero. So the number is
entirely curl and none of it is the half that could be fixed by comparing different pairs. The
reason is visible in the design: Nectar compares all 21 pairs of 7 items, so its clique complex is
simply connected and there is no harmonic space for a component to live in, and of UltraFeedback's
5,000 tournaments 338 do have a hole and the observed flow has zero circulation around every one.

**And the curl does not clear its baseline.** A *perfectly transitive* grader on the identical
comparison design produces a curl mass of 0.2172, against the observed 0.2140. The observed value is
below the null. On the complete graph the baseline has a closed form, ``(n-2)/(3n)``, which is
0.2380952 at seven items, and Nectar's measured curl mass is 0.2380952 to sixteen digits. The direct
check agrees: B5's triple scan finds a maximum triangle circulation of 1 over all 186,378 filled
triangles, and a cyclic triangle on ±1 data has circulation 3, so not one triangle in the corpus is
cyclic. The 0.214 is what a ±1 sign encoding costs, and it contains no intransitivity.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from reward_lens.core.invariance import INVARIANT, InvariancePayload, check_invariance
from reward_lens.core.quantity import QUANTITIES
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.composition.hodge import (
    MIN_REPLICATIONS_FOR_NULLS,
    ComparisonFlow,
    CurlMass,
    PairCount,
    curl_mass_of_payload,
    disjoint_union,
    edge_flow,
    split_flow,
)
from reward_lens.measure.composition.nulls import (
    bradley_terry_bootstrap,
    split_half_reliability,
    transitive_baseline,
    transitive_curl_mass_complete,
)
from reward_lens.measure.composition.sepnorm import (
    SepDistanceBound,
    sep_distance,
    sep_distance_of_payload,
    triple_scan,
)

#: The campaign evidence store, which is not in this repository. There is no default: point
#: ``REWARD_LENS_CAMPAIGN_STORE`` at the store directory or the tests that need it skip.
_CAMPAIGN_ENV = os.environ.get("REWARD_LENS_CAMPAIGN_STORE")
CAMPAIGN = Path(_CAMPAIGN_ENV) if _CAMPAIGN_ENV else None

#: What the campaign recorded, from `campaign.adjudication.TOPO-HODGE`
#: (``ev:844ad06a5aeb1d89c21c13c19a28cbd2``). Pinned here so the re-adjudication is against the
#: stored number rather than against a remembered one.
RECORDED_INTRANSITIVE_MASS = 0.2139761313773256
RECORDED_CURL_MASS = 0.2139761313773256
RECORDED_HARMONIC_MASS = 2.425774958668703e-31
RECORDED_GRADIENT_MASS = 0.7860238686226744

needs_campaign = pytest.mark.skipif(
    CAMPAIGN is None or not (CAMPAIGN / "evidence.jsonl").exists(),
    reason="no campaign evidence store; set REWARD_LENS_CAMPAIGN_STORE",
)


def load_campaign_tournaments() -> dict[str, list[ComparisonFlow]]:
    """The two stored tournament slices as per-tournament comparison flows, read-only.

    The store is the only copy of a $17.73 experiment, so it is opened through `CampaignStore`,
    which constructs its `EvidenceStore` with ``readonly=True`` and never writes. Nothing here
    mutates a file, a sidecar, or an in-memory row.
    """
    from reward_lens.record.convert.store import CampaignStore

    store = CampaignStore(CAMPAIGN)
    out: dict[str, list[ComparisonFlow]] = {}
    for row in store.by_observable("campaign.tournaments"):
        matrix = np.asarray(store.value(row)["matrix"])
        ids = matrix[:, 0].astype(int)
        order = np.argsort(ids, kind="stable")
        matrix, ids = matrix[order], ids[order]
        flows = []
        for block in np.split(np.arange(len(ids)), np.flatnonzero(np.diff(ids)) + 1):
            rows = matrix[block]
            n_items = int(max(rows[:, 1].max(), rows[:, 2].max())) + 1
            flows.append(
                edge_flow(
                    [PairCount(int(r[1]), int(r[2]), float(r[3]), float(r[4])) for r in rows],
                    n_items,
                )
            )
        out[str(row.slice_name)] = flows
    return out


# ---------------------------------------------------------------------------
# The clause, part one: the split-half null runs and its verdict is recorded
# ---------------------------------------------------------------------------


@needs_campaign
def test_the_split_half_null_runs_on_the_campaign_data_and_its_verdict_is_a_refusal():
    """Null C, on the stored TOPO-HODGE data. The verdict: one replication per pair, eleven needed.

    This is the clause's first half and the answer is a refusal, which is a result. The
    campaign's preference data carries one comparison per pair, so there is no second half of the
    replications to correlate the curl component against, and no amount of care in the analysis
    creates one. The refusal names the number, names the access level that would supply it, and
    carries the observed count so the claim is auditable rather than asserted.
    """
    slices = load_campaign_tournaments()
    flow = disjoint_union([f for group in slices.values() for f in group])

    assert flow.n_edges == 130_129
    assert flow.min_replications == 1.0, "every stored pair was compared exactly once"
    assert flow.unanimous_fraction == 1.0, "so every pair is unanimous, and the logit would diverge"

    verdict = split_half_reliability(flow)

    assert isinstance(verdict, Refusal), "the recorded verdict for the campaign data"
    assert verdict.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert verdict.statistics["observed_min_replications"] == 1.0
    assert verdict.statistics["required_replications"] == MIN_REPLICATIONS_FOR_NULLS
    assert str(MIN_REPLICATIONS_FOR_NULLS) in verdict.detail
    assert "GRADER:REPLICATE" in verdict.remedy
    # The parametric bootstrap is gated on the same n and refuses the same way, so the reading the
    # instrument produces on this data is rung 0: a curl mass with no null behind it.
    assert isinstance(bradley_terry_bootstrap(flow), Refusal)


@needs_campaign
def test_the_instrument_reports_rung_zero_on_the_campaign_data_and_names_the_refused_nulls():
    """A reading with no null behind it says so, in the field a card reads."""
    slices = load_campaign_tournaments()
    flow = disjoint_union([f for group in slices.values() for f in group])
    report = CurlMass(flow, nulls=("C", "A"), n_draws=1).compute()

    assert report["rung"] == 0
    assert report["nulls_refused"] == ["A_parametric_bootstrap", "C_split_half"]
    assert report["min_replications_per_pair"] == 1.0
    assert "none:" in report["continuity_correction"]


def test_the_split_half_null_is_not_merely_a_refusal_emitter():
    """The same null on the same code path, at eleven replications, separates noise from structure.

    Without this the refusal above would be uninterpretable: a null that always refuses proves
    nothing about the data. Under a Bradley-Terry truth the observed curl is sampling noise and does
    not replicate; under a kernel with a real skew-symmetric part it does. The gradient component
    replicates in both arms, which is the positive control that makes the first result a finding
    rather than a report that the simulation was small.
    """
    rng = np.random.default_rng(1)

    def replicated(kernel, seed, n=6, reps=11):
        gen = np.random.default_rng(seed)
        pairs = []
        for a in range(n):
            for b in range(a + 1, n):
                wins_b = int(gen.binomial(reps, 1.0 / (1.0 + np.exp(-kernel[a, b]))))
                pairs.append(PairCount(a, b, reps - wins_b, wins_b))
        return edge_flow(pairs, n)

    def scalar(n=6):
        s = rng.standard_normal(n) * 1.2
        return s[None, :] - s[:, None]

    def cyclic(n=6):
        s = rng.standard_normal(n) * 1.2
        raw = rng.standard_normal((n, n))
        skew = raw - raw.T
        return (s[None, :] - s[:, None]) + skew / np.linalg.norm(skew) * 2.5 * n**0.5

    noise = split_half_reliability(
        disjoint_union([replicated(scalar(), 100 + k) for k in range(30)]), n_splits=30, seed=0
    )
    structure = split_half_reliability(
        disjoint_union([replicated(cyclic(), 200 + k) for k in range(30)]), n_splits=30, seed=0
    )

    assert noise.gradient_reliability > 0.7 and structure.gradient_reliability > 0.7
    assert structure.verdict == "replicates"
    assert structure.curl_reliability - noise.curl_reliability > 0.3

    # The scalar arm's verdict was `does_not_replicate` and the assertion has been changed, because
    # the gradient control cannot support that branch's note. `im(grad)` has dimension 150 on this
    # design where `im(C)` has 300, so the gradient replicates better at equal signal energy for a
    # reason that is about the geometry rather than about the grader. The matched control settles
    # it: a curl planted at the observed energy, in the same subspace, through the same sampling
    # noise, replicates at only 0.44 at eleven replications, so this design cannot see a curl of
    # that size and its absence has not been shown. At 101 replications the same arm reaches
    # `does_not_replicate` with the matched control at 0.72, which is where the branch is earned.
    assert noise.curl_subspace_dim == 300 and noise.gradient_subspace_dim == 150
    assert noise.planted_curl_reliability < 0.7
    assert noise.verdict == "underpowered"
    assert "matched control" in noise.note
    assert structure.planted_curl_reliability >= 0.7


# ---------------------------------------------------------------------------
# The clause, part two: 0.214 re-adjudicated, split into curl and harmonic
# ---------------------------------------------------------------------------


@needs_campaign
def test_the_recorded_0214_is_reproduced_and_split_into_curl_and_harmonic():
    """The re-adjudication. The number holds; the split says all of it is the unfixable half.

    Reproducing the recorded value first is what makes the rest a re-adjudication rather than a
    different measurement. It agrees to fifteen significant figures through a sparse solver the
    original did not use.
    """
    slices = load_campaign_tournaments()
    flow = disjoint_union([f for group in slices.values() for f in group])
    split = split_flow(flow)

    assert split.intransitive_mass == pytest.approx(RECORDED_INTRANSITIVE_MASS, abs=1e-14)
    assert split.gradient_mass == pytest.approx(RECORDED_GRADIENT_MASS, abs=1e-14)

    # The split. All curl, no harmonic.
    assert split.curl_mass == pytest.approx(RECORDED_CURL_MASS, abs=1e-14)
    assert split.harmonic_mass < 1e-20
    assert RECORDED_HARMONIC_MASS < 1e-20
    assert split.curl_mass + split.harmonic_mass == pytest.approx(split.intransitive_mass)

    # And why the harmonic half is empty: 338 tournaments have a hole and no flow circulates one.
    assert split.betti1 == 338
    assert split.n_edges == 130_129
    assert split.n_triangles == 186_378


@needs_campaign
def test_the_harmonic_half_is_empty_for_a_reason_the_design_decides():
    """Nectar compared every pair, so it has no hole to hold a harmonic component at all.

    β₁ is a property of which pairs were compared and of nothing else. A complete comparison graph
    has β₁ = 0, so a harmonic mass of zero there is a fact about the design rather than a
    measurement of the grader, and reporting it without that context would be a fact reported as a
    finding.
    """
    slices = load_campaign_tournaments()
    nectar = split_flow(disjoint_union(slices["nectar-tournaments"]))
    ultra = split_flow(disjoint_union(slices["ultrafeedback-tournaments"]))

    assert nectar.betti1 == 0, "7 items and 21 pairs is the complete graph, which has no hole"
    assert nectar.n_triangles == 175_000
    assert ultra.betti1 == 338, "338 of 5,000 UltraFeedback tournaments have a chordless 4-cycle"
    assert nectar.harmonic_mass < 1e-20
    assert ultra.harmonic_mass < 1e-20, "holes exist and the flow circulates none of them"


@needs_campaign
def test_the_campaign_curl_mass_does_not_clear_a_perfectly_transitive_baseline():
    """B1's own declared baseline, run on the identical comparison design. The curl does not survive.

    A transitive grader recorded as wins and losses already carries curl, because a ±1 sign is not
    the gradient of anything. On Nectar's complete seven-item graph the closed form is
    ``(7-2)/(3·7) = 0.2380952``, and the measured value is that number. The excess is zero.
    """
    slices = load_campaign_tournaments()
    nectar = disjoint_union(slices["nectar-tournaments"][:400])
    observed = split_flow(nectar)
    baseline = transitive_baseline(nectar, n_draws=8, seed=0)

    assert observed.curl_mass == pytest.approx(transitive_curl_mass_complete(7), abs=1e-12)
    assert baseline.curl_null_mean == pytest.approx(observed.curl_mass, abs=1e-12)
    assert baseline.curl_excess == pytest.approx(0.0, abs=1e-12)
    assert baseline.curl_p_value > 0.5, "the observation sits on top of the null, not above it"

    ultra = disjoint_union(slices["ultrafeedback-tournaments"][:1500])
    ultra_baseline = transitive_baseline(ultra, n_draws=8, seed=0)
    assert ultra_baseline.curl_excess < 0.0, "below what a perfectly transitive grader would give"


@needs_campaign
def test_not_one_filled_triangle_in_the_campaign_data_is_cyclic():
    """The direct check, and B5 supplies it for free.

    On ±1 comparison data the circulation around a filled triangle is 3 when it is cyclic and 1 when
    it is transitive, so the maximum circulation over the whole corpus is the question "is any
    triangle cyclic?" answered in one scan. Δ = 1 over 186,378 filled triangles is a rock-paper-
    scissors count of zero, which is what the 0.214 has to be read against.
    """
    slices = load_campaign_tournaments()
    flow = disjoint_union([f for group in slices.values() for f in group])
    delta, worst = triple_scan(flow)

    assert delta == pytest.approx(1.0), "3 would mean at least one cyclic triangle; 1 means none"
    assert worst is not None


@needs_campaign
def test_b5_reports_its_bound_beside_b1s_curl_mass_on_the_campaign_data():
    """The pair, on real data, with the theorem attached and the sandwich width visible."""
    slices = load_campaign_tournaments()
    flow = disjoint_union(slices["nectar-tournaments"][:500])
    bound = sep_distance(flow)

    assert bound.delta == pytest.approx(1.0)
    assert bound.lower == pytest.approx(1.0 / 3.0)
    assert bound.lower <= bound.upper
    assert bound.curl_mass == pytest.approx(transitive_curl_mass_complete(7), abs=1e-12)
    assert 0.0 < bound.contraction_upper <= 1.0
    assert bound.n_triangles == 500 * 35


# ---------------------------------------------------------------------------
# The generated invariance tests, and lint
# ---------------------------------------------------------------------------


def _payload(seed: int = 0, groups: int = 8, size: int = 6) -> InvariancePayload:
    rng = np.random.default_rng(seed)
    return InvariancePayload(
        scores=rng.standard_normal(groups * size),
        group_ids=np.repeat(np.arange(groups), size),
    )


def test_b1_is_invariant_under_an_affine_rescaling_of_the_reward():
    """A positive rescaling and a shift cannot change which of two responses wins a comparison."""
    report = check_invariance(
        CurlMass(),
        "reward.affine",
        _payload(1),
        n=16,
        relation=INVARIANT,
        run=lambda _inst, payload: curl_mass_of_payload(payload),
    )
    assert report.passed, report.render()
    assert report.n == 16


def test_b1_is_invariant_under_permuting_rollout_order_within_a_group():
    """The curl mass is a graph invariant, so relabelling the items cannot move it.

    A failure here would be informative rather than fatal: it would mean the instrument was reading
    rollout order, which for a judge is position bias. It does not fail, and the check is the reason
    anyone can say so.
    """
    report = check_invariance(
        CurlMass(),
        "group.permutation",
        _payload(2),
        n=16,
        relation=INVARIANT,
        run=lambda _inst, payload: curl_mass_of_payload(payload),
    )
    assert report.passed, report.render()


def test_b5_is_invariant_under_an_affine_rescaling_of_the_reward():
    report = check_invariance(
        SepDistanceBound(),
        "reward.affine",
        _payload(3),
        n=16,
        relation=INVARIANT,
        run=lambda _inst, payload: sep_distance_of_payload(payload),
    )
    assert report.passed, report.render()


def test_both_instruments_pass_the_section_4_2_lint():
    """No instrument merges with an undeclared quantity, baseline, envelope or invariance group."""
    for instrument in (CurlMass(), SepDistanceBound()):
        assert lint_instrument(instrument) == [], [f.render() for f in lint_instrument(instrument)]


def test_the_quantities_are_the_registered_ones():
    assert CurlMass.quantity in QUANTITIES
    assert CurlMass.secondary_quantity in QUANTITIES
    assert SepDistanceBound.quantity in QUANTITIES
    assert QUANTITIES.get("grader.curl_mass").unit.dimension == "1"
    assert QUANTITIES.get("grader.harmonic_mass").unit.dimension == "1"
    assert QUANTITIES.get("grader.sep_distance").unit.dimension == "1"


def test_both_instruments_return_a_reading_and_never_a_bare_number():
    """`estimate` returns Evidence or a Refusal, on both the happy path and the empty one."""
    rng = np.random.default_rng(4)
    pairs = [
        PairCount(a, b, float(rng.integers(0, 12)), float(rng.integers(0, 12)))
        for a in range(6)
        for b in range(a + 1, 6)
    ]
    flow = edge_flow(pairs, 6)

    evidence = CurlMass(flow, nulls=("D", "E"), n_draws=10).estimate(Context())
    assert not isinstance(evidence, Refusal)
    assert "curl_mass" in evidence.value
    assert "harmonic_mass" in evidence.value

    bound = SepDistanceBound(flow).estimate(Context())
    assert not isinstance(bound, Refusal)
    assert "sep_distance_upper" in bound.value

    assert isinstance(CurlMass().estimate(Context()), Refusal)
    assert isinstance(SepDistanceBound().estimate(Context()), Refusal)

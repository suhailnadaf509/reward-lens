"""Acceptance: B2 and B6 on real comparison data, with the readings each one licenses.

The clause this file discharges: *on real comparison data, an Afriat index with its GARP violation
count beside it, and a Condorcet rate, a Smith set size and a λ₂ with the annotation-budget reading
that λ₂ licenses.*

Real means the campaign store, which is the only comparison data this project holds that nobody
constructed for these instruments. Three subjects come out of it and they answer different halves of
the clause.

The **12-model panel on RewardBench 2** is the one that carries the whole clause. Twelve open reward
models scored the same 1,763 prompts at four candidates each, and a majority vote over the twelve is
a tournament reward: a composite grader whose comparison relation is nobody's scalar. It cycles, it
loses its Condorcet winner, it has a disconnected comparison graph on one prompt, and every number
B2 and B6 report is non-degenerate on it.

The **UltraFeedback and Nectar annotator tournaments** are the recorded preference data the
TOPO-HODGE card measured. They are rating-derived, so they cannot cycle, and saying that plainly is
better than presenting an index of 1.0 as though the instrument had discriminated. What they are
good for is the B1 comparison: Nectar is a strict total order on every one of 5,000 prompts, its
Afriat index is exactly 1.0, and the campaign's own recorded curl mass on the same edges is 0.2381.
Two numbers on one dataset, disagreeing as far as they can, which is the catalogue's kill condition
failing to fire.

A **single reward model's own tournaments** are the scope limit, stated rather than discovered: a
scalar head's within-prompt comparisons are a total order by construction, so it cannot violate GARP
and a violation there would be a bug in the pipeline rather than a finding about the grader.

The store lives outside the repository because it is a 313 MB archive of a $17.73 experiment and is
read here through `CampaignStore`, which opens it read-only. ``REWARD_LENS_CAMPAIGN_STORE`` is the
only way to point at it and there is no default, so the module skips naming that variable.
Which store matters: ``campaign-results/runs/campaign``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import Context
from reward_lens.measure.composition.revealed import (
    AfriatIndex,
    ComparisonSet,
    afriat_index,
    bank_from_score_panels,
    bank_from_scores,
    garp_violations,
    houtman_maks,
    money_pump,
)
from reward_lens.measure.composition.tournament import (
    ComparisonConnectivity,
    CondorcetRate,
    SmithSetSize,
    algebraic_connectivity,
    condorcet,
    summarise,
)

#: The campaign evidence store, which is not in this repository. There is no default: point
#: ``REWARD_LENS_CAMPAIGN_STORE`` at the store directory or the tests that need it skip.
_CAMPAIGN_ENV = os.environ.get("REWARD_LENS_CAMPAIGN_STORE")
CAMPAIGN_STORE = Path(_CAMPAIGN_ENV) if _CAMPAIGN_ENV else None

pytestmark = pytest.mark.skipif(
    CAMPAIGN_STORE is None or not (CAMPAIGN_STORE / "evidence.jsonl").exists(),
    reason=(
        "no campaign evidence store. It is the archive the 2.0 campaign produced and it is not "
        "in the repository; set REWARD_LENS_CAMPAIGN_STORE to its directory to run this file."
    ),
)

#: The curl mass the campaign measured on the Nectar tournament edges, read back from its own result
#: row rather than restated here. B1 owns the quantity; this file only puts it beside B2's.
NECTAR_CURL_MASS = 5.0 / 21.0


def _store():
    from reward_lens.record.convert import CampaignStore

    return CampaignStore(CAMPAIGN_STORE)


def _tournament_bank(store, slice_name: str, limit: int | None = None):
    """The recorded (tournament, i, j, wins_i, wins_j) edges of one bank, as comparison sets."""
    for row in store.by_observable("campaign.tournaments"):
        if row.slice_name != slice_name:
            continue
        value = store.value(row)
        matrix = value["matrix"]
        n_items = list(value["meta"]["n_items"])
        order = np.argsort(matrix[:, 0], kind="stable")
        sorted_edges = matrix[order]
        bounds = np.searchsorted(sorted_edges[:, 0], np.arange(len(n_items) + 1))
        upto = len(n_items) if limit is None else min(limit, len(n_items))
        return [
            ComparisonSet.from_edges(sorted_edges[bounds[t] : bounds[t + 1], 1:5], int(n_items[t]))
            for t in range(upto)
        ]
    raise AssertionError(f"the store holds no campaign.tournaments row for {slice_name!r}")


@pytest.fixture(scope="module")
def store():
    return _store()


@pytest.fixture(scope="module")
def rb2_scores(store):
    """Every reward model's scores on the 1,763 RewardBench 2 prompts, four candidates each."""
    banks = {}
    for row in store.by_observable("campaign.scores"):
        if row.slice_name == "rb2-full":
            banks[row.roster_key] = store.value(row)["scores"]
    assert len(banks) >= 12, f"expected the full roster on rb2-full; got {sorted(banks)}"
    names = sorted(banks)
    stacked = np.stack([banks[k] for k in names])
    assert stacked.shape[1:] == (1763, 4), stacked.shape
    return names, stacked


@pytest.fixture(scope="module")
def panel(rb2_scores):
    """The majority-vote tournament of the whole roster: a grader whose relation is nobody's scalar.

    A tie in the vote records nothing, for the same reason a tie in a score records nothing. That is
    what makes 20% of these comparison graphs incomplete and one of them disconnected, and it is the
    honest encoding: twelve models split six to six expressed no preference.
    """
    _names, stacked = rb2_scores
    _m, n_prompts, k = stacked.shape
    wins = np.zeros((n_prompts, k, k))
    for a in range(k):
        for b in range(k):
            if a != b:
                wins[:, a, b] = (stacked[:, :, a] > stacked[:, :, b]).sum(axis=0)
    return [
        ComparisonSet.from_edges(
            [(i, j, wins[n, i, j], wins[n, j, i]) for i in range(k) for j in range(i + 1, k)], k
        )
        for n in range(n_prompts)
    ]


# ---------------------------------------------------------------------------
# The clause, one part at a time
# ---------------------------------------------------------------------------


def test_an_afriat_index_on_real_comparison_data_with_its_violation_count_beside_it(panel):
    """B2's half of the clause. Both numbers on one reading, and neither one alone.

    Measured on the twelve-model panel: an index of 0.8333 pooled and 0.99959 mean over 1,763
    prompts, with 15 GARP violations. The pooled figure is Afriat's own definition applied to the
    bank, so it is a minimum over prompts and one cyclic group sets it; the mean is what a reader
    wants and it travels on the same reading rather than replacing it.
    """
    reading = AfriatIndex(panel, curl_mass=NECTAR_CURL_MASS, baseline_draws=8).estimate(Context())
    assert not isinstance(reading, Refusal), getattr(reading, "render", str)()
    value = reading.value

    assert value["afriat_index"] == pytest.approx(5.0 / 6.0, abs=1e-9)
    assert value["garp_violations"] == 15
    assert value["afriat_index_mean"] == pytest.approx(0.99959, abs=1e-4)
    assert value["n_groups"] == 1763
    assert value["n_comparisons"] == 10165
    assert not value["ordinal_only"], "a twelve-model vote gives a real win fraction per pair"
    assert not value["rationalizable"]

    assert value["houtman_maks_exact"], "K = 4 is enumerable, so the largest subset is exact"
    assert value["houtman_maks_fraction"] == pytest.approx(0.99953, abs=1e-4)
    assert value["money_pump_cycles"] == 5
    assert value["money_pump_mean"] == pytest.approx(0.2202, abs=1e-3)

    # The headline is a minimum over 1,763 groups, so its baseline is the chance minimum over the
    # same groups and not the chance mean. A random grader's minimum here is exactly 0.0: over 1,763
    # four-item tournaments at least one draw always cycles at unit margins. The chance mean is
    # 0.6945 and comparing 0.8333 against that would have been comparing a min with a mean.
    baseline = value["baselines"]["baseline.random_tournament"]
    assert baseline == pytest.approx(0.0, abs=1e-9)
    assert value["baseline_random_tournament_mean"] == pytest.approx(0.6945, abs=1e-3)
    assert value["afriat_index"] > baseline
    assert value["afriat_index_mean"] > value["baseline_random_tournament_mean"]


def test_a_condorcet_rate_and_the_reason_each_failure_happened(panel):
    """B6's first quantity, with its failures split by cause.

    Measured: a Condorcet winner on 94.55% of the 1,763 prompts. Of the 5.45% without one, the rest
    is missing comparisons, which at K = 4 means the twelve models split evenly on a pair.

    **The cycle rate moved from 0.284% to 0.113% and the earlier figure was the wrong quantity.**
    Five of the 1,763 comparison graphs contain a cycle; on three of them a fourth candidate beats
    all three members of the cycle, so a Condorcet winner exists anyway. The rate that decomposes
    ``no_condorcet_rate`` counts only the two where the cycle is the reason there is no winner, and
    the unconditional count travels beside it as ``any_cycle_rate``. Reported as one number the two
    could not be added up, and the reported cycle rate could exceed the failure rate it was
    supposedly a part of.

    The catalogue's kill condition is *if Condorcet cycles are under 1% at real group sizes*. Both
    figures are under 1%, so the kill condition fires on this grader at this group size on either
    reading, and it is recorded as such rather than argued away.
    """
    reading = CondorcetRate(panel, baseline_draws=8).estimate(Context())
    assert not isinstance(reading, Refusal), getattr(reading, "render", str)()
    value = reading.value

    assert value["condorcet_rate"] == pytest.approx(0.94555, abs=1e-4)
    assert value["no_condorcet_rate"] == pytest.approx(1.0 - value["condorcet_rate"], abs=1e-12)
    assert value["cycle_rate"] == pytest.approx(0.00113, abs=1e-4)
    assert value["any_cycle_rate"] == pytest.approx(0.00284, abs=1e-4)
    assert value["cycle_rate"] <= value["no_condorcet_rate"], "a part cannot exceed its whole"
    assert value["incomplete_rate"] == pytest.approx(0.20420, abs=1e-4)
    assert value["mean_group_size"] == 4.0
    assert value["slater_exact"], "K = 4 is enumerable, so Slater is the optimum and not a bound"

    # No scalar Copeland-versus-Slater rate: the old 0.00737 was 13 groups picked out by `argmax`
    # against the first enumerated Slater ordering, and 85 of the 1,763 groups have a tie in one
    # rule or the other. The range those 85 span is what is reported.
    assert "copeland_slater_disagreement_rate" not in value
    assert value["copeland_slater_forced_disagreement_rate"] == 0.0
    assert value["copeland_slater_forced_agreement_rate"] == pytest.approx(1678 / 1763, abs=1e-6)
    assert value["copeland_slater_undetermined_rate"] == pytest.approx(85 / 1763, abs=1e-6)
    low, high = value["copeland_slater_disagreement_range"]
    assert (low, high) == (pytest.approx(0.0), pytest.approx(0.04821, abs=1e-4))
    assert low <= 0.00737 <= high, (
        "the scalar that used to be printed sits inside the range, which is the point: it is one "
        "tie-break's answer and the range is every tie-break's"
    )

    assert value["any_cycle_rate"] < 0.01, (
        "recorded finding, not a passing assertion: at K = 4 with a twelve-model panel the cycle "
        "rate is below the catalogue's 1% kill threshold for B6 on either reading"
    )
    chance = value["baselines"]["baseline.random_tournament"]
    assert chance < 0.6 < value["condorcet_rate"], (
        f"a random grader on the same comparison graphs finds a Condorcet winner {chance:.4f} of "
        f"the time, so the measured 0.9456 is a property of the panel and not of K = 4"
    )


def test_a_smith_set_size_which_is_the_number_the_reward_chooses_among_arbitrarily(panel):
    """B6's second quantity. Mean 1.0822, singleton on 94.55% of prompts, up to 4 on the worst."""
    reading = SmithSetSize(panel, baseline_draws=8).estimate(Context())
    assert not isinstance(reading, Refusal), getattr(reading, "render", str)()
    value = reading.value

    assert value["smith_set_size"] == pytest.approx(1.08225, abs=1e-4)
    assert value["modal_smith_set_size"] == 1
    assert value["max_smith_set_size"] == 4
    assert value["singleton_rate"] == pytest.approx(0.94555, abs=1e-4)

    # A correction to the record. B2 reported the agreement of these two rates as a corroborating
    # measurement on real data; it is a theorem, immediate from the definitions, and it cannot fail
    # for n >= 2. The Smith set is {x} exactly when x beats every other alternative, which is what a
    # Condorcet winner is. Kept as a code-correctness invariant, which is what it always was, and no
    # longer offered as evidence about the panel.
    condorcet_reading = CondorcetRate(panel, baseline_draws=2).estimate(Context())
    assert value["singleton_rate"] == pytest.approx(
        condorcet_reading.value["condorcet_rate"], abs=1e-12
    ), "an identity the implementation has to preserve, not a finding about this grader"

    chance = value["baselines"]["baseline.random_tournament"]
    assert chance > 2.0 > value["smith_set_size"]


def test_lambda2_refuses_on_the_disconnected_design_and_names_the_pair_that_fixes_it(panel):
    """B6's third quantity, and the refusal it is supposed to produce.

    One of the 1,763 comparison graphs is disconnected: on prompt 516 the twelve models split six to
    six against candidate 0 on all three of its pairs, so candidate 0 is isolated and λ₂ is exactly
    0. The error bound is then infinite and no annotation budget identifies the ranking across the
    split, so a reading of 0 would be a number standing in for a quantity that does not exist. The
    refusal carries λ₂ over the groups that are connected and a remedy naming the pair to compare.
    """
    reading = ComparisonConnectivity(panel, baseline_draws=4).estimate(Context())
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ABOVE_LOD_BELOW_LOQ
    assert reading.statistics["disconnected_groups"] == 1
    assert "compare alternatives" in reading.remedy
    assert reading.is_bounded
    assert reading.partial.value["graph_lambda2"] == 0.0
    assert reading.partial.value["n_groups"] == 1763

    disconnected = [n for n, cs in enumerate(panel) if not algebraic_connectivity(cs).connected]
    assert disconnected == [516], f"expected exactly one disconnected design; got {disconnected}"


def test_lambda2_with_the_annotation_budget_reading_it_licenses(panel):
    """The clause's last part, on the 1,762 prompts whose comparison graph is connected.

    Measured: λ₂ = 3.5693 against the 4.0 a complete four-way comparison would give, with a worst
    prompt at 0.5858, which is the path graph.

    **Two numbers in the earlier version of this clause were wrong and both are corrected here.**

    The error bound was evaluated at the mean group size and the mean λ₂, giving 1.2464, and read as
    a bound. It is not one: 359 of the 1,762 connected prompts need a larger bound than that, and the
    worst needs 3.0767. The bound is now evaluated per group and the worst group's is the headline,
    with the count above the mean figure beside it.

    And "pushing that to 0.1 wants about 155 comparisons per pair" is a budget the bound does not
    license. ``√(n log n / (L λ₂))`` is Shah et al., JMLR 17(58), 2016, Theorem 2 equation (8b),
    where the constant absorbs ``ζ²/γ²`` and ``σ²``; 155.4 is the formula with that constant silently
    set to 1. What survives division by an unknown constant is the ratio between two designs and the
    cost of a stated improvement, and those are what the reading prints now: this design costs 1.12
    times what the complete four-way comparison would for the same accuracy, and halving the error
    costs 4 comparisons per pair whatever the constant is.
    """
    connected = [cs for cs in panel if algebraic_connectivity(cs).connected]
    assert len(connected) == 1762
    reading = ComparisonConnectivity(connected, baseline_draws=4).estimate(Context())
    assert not isinstance(reading, Refusal), getattr(reading, "render", str)()
    value = reading.value

    assert value["graph_lambda2"] == pytest.approx(3.5693, abs=1e-3)
    assert value["graph_lambda2_min"] == pytest.approx(2.0 - np.sqrt(2.0), abs=1e-6)
    assert value["baselines"]["baseline.assume_total_order"] == 4.0

    assert "error_bound_L1" not in value
    assert "replications_for_error_0p1" not in value
    assert value["error_bound_at_the_mean_L1"] == pytest.approx(1.2464, abs=1e-3)
    assert value["error_bound_worst_group_L1"] == pytest.approx(3.0767, abs=1e-3)
    assert value["groups_above_the_mean_bound"] == 359, (
        "the figure that used to be printed as the bound fails to bound 359 of 1,762 prompts"
    )
    assert value["error_bound_worst_group_L11"] == pytest.approx(
        value["error_bound_worst_group_L1"] / np.sqrt(11), rel=1e-9
    )
    assert value["budget_ratio_against_complete_design"] == pytest.approx(1.1207, abs=1e-3)
    assert value["replications_to_halve_the_error"] == pytest.approx(4.0)
    assert "155" not in value["annotation_budget_reading"]
    assert "prices one design against another" in value["annotation_budget_reading"]

    design = value["baselines"]["baseline.random_comparison_design"]
    assert design == pytest.approx(value["graph_lambda2"], abs=0.2), (
        "spending the same number of comparisons on randomly chosen pairs gives about the same λ₂ "
        "at K = 4, because there are only six pairs to choose from. The baseline earns its place at "
        "the group sizes where it does not, and reporting that it is flat here is the point of "
        "running it"
    )


# ---------------------------------------------------------------------------
# What the same instruments say about the data that cannot move them
# ---------------------------------------------------------------------------


def test_the_nectar_bank_separates_the_afriat_index_from_the_curl_mass(store):
    """B2's kill condition, tested where it would bite, and not firing.

    Nectar's recorded tournaments are strict rankings of seven responses per prompt, so every one of
    the 5,000 is a total order: no cycle, a Condorcet winner every time, an Afriat index of exactly
    1.0. The campaign's own Hodge decomposition of the same 105,000 edges puts the curl mass at
    0.2381, because a unit-margin total order is not the gradient of any potential. The kill
    condition is *if Afriat and curl mass are monotonically related across twenty graders, in which
    case report one*, and one dataset where they take the extreme opposite values is not twenty
    graders, but it does show the two are measuring different things rather than restating one.
    """
    bank = _tournament_bank(store, "nectar-tournaments")
    assert len(bank) == 5000
    assert {cs.n_items for cs in bank} == {7}
    assert sum(cs.n_comparisons for cs in bank) == 105000

    result = afriat_index(bank)
    assert result.pooled == 1.0
    assert result.n_violations == 0
    assert houtman_maks(bank).fraction == 1.0
    assert money_pump(bank).n_cycles == 0

    tournament = summarise(bank)
    assert tournament.condorcet_rate == 1.0
    assert tournament.mean_smith_size == 1.0
    assert tournament.mean_lambda2 == pytest.approx(7.0), "the seven-way comparison is complete"

    recorded = _recorded_curl_mass(store, "nectar")
    assert recorded == pytest.approx(NECTAR_CURL_MASS, abs=1e-12), (
        "the campaign's stored curl mass for this bank is 5/21 exactly, which is the least-squares "
        "residual of a unit-margin total order on seven items"
    )
    assert result.pooled - recorded > 0.7, (
        f"one dataset, rationalizability fraction {result.pooled:.4f} and energy share "
        f"{recorded:.4f}. An instrument reporting only one of these is discarding the other"
    )


def test_the_same_twelve_models_read_as_a_panel_rather_than_as_a_vote(rb2_scores):
    """The comparison record B2 is actually defined over, on the same real subject.

    The `panel` fixture reduces twelve reward models to a majority vote, which throws away every
    margin and leaves one verdict per pair. `from_score_panel` keeps all twelve as twelve
    observations of the same four candidates, which is Afriat's setting: several observations, one
    menu. The two readings are consistent and they are not the same reading.

    Consistent, because collapsing the panel to its majority reproduces the vote exactly: identical
    Condorcet rate, cycle rate, Smith size and incompleteness on all 1,763 prompts. Not the same,
    because the vote hides the disagreement it is a vote over. As a vote the panel has 15 GARP
    violations over 10,165 comparisons and a mean index of 0.9996; as a panel it has 19,198
    violations over 126,772 and a mean index of 0.1311, with 18.2% of the record needing to go
    before one utility orders the rest.

    Neither number is wrong and the second is the one the catalogue's sentence is about. A composite
    grader assembled from twelve models is not close to being a scalar, and reading it through its
    own majority vote is what made it look like one.
    """
    _names, stacked = rb2_scores
    panel = bank_from_score_panels(np.transpose(stacked, (1, 0, 2)))
    assert len(panel) == 1763
    assert panel[0].n_comparisons == 72, "twelve observations of six pairs, ties dropped"
    assert panel[0].has_repeated_pairs

    result = afriat_index(panel)
    assert result.n_comparisons == 126772
    assert result.n_violations == 19198
    assert result.mean == pytest.approx(0.1311, abs=1e-3)
    assert not result.ordinal
    assert int(np.sum(result.per_group < 1.0)) == 1762

    assert houtman_maks(panel).fraction == pytest.approx(0.8183, abs=1e-3)
    pump = money_pump(panel)
    assert pump.n_cycles == 28083
    assert pump.mean == pytest.approx(0.4898, abs=1e-3)

    collapsed = summarise(panel)
    voted = summarise([cs for cs in _panel_from_votes(stacked)])
    for field in ("condorcet_rate", "cycle_rate", "any_cycle_rate", "incomplete_rate"):
        assert getattr(collapsed, field) == pytest.approx(getattr(voted, field), abs=1e-12), field
    assert collapsed.mean_smith_size == pytest.approx(voted.mean_smith_size, abs=1e-12)


def _panel_from_votes(stacked: np.ndarray) -> list[ComparisonSet]:
    """The majority-vote bank, rebuilt here so the panel can be checked against it."""
    _m, n_prompts, k = stacked.shape
    wins = np.zeros((n_prompts, k, k))
    for a in range(k):
        for b in range(k):
            if a != b:
                wins[:, a, b] = (stacked[:, :, a] > stacked[:, :, b]).sum(axis=0)
    return [
        ComparisonSet.from_edges(
            [(i, j, wins[n, i, j], wins[n, j, i]) for i in range(k) for j in range(i + 1, k)], k
        )
        for n in range(n_prompts)
    ]


def _recorded_curl_mass(store, dataset: str) -> float:
    for row in store.by_observable("campaign.result.TOPO-HODGE"):
        per_dataset = store.value(row)["meta"]["per_dataset"]
        return float(per_dataset[dataset]["curl_mass"])
    raise AssertionError("the store holds no campaign.result.TOPO-HODGE row")


def test_every_condorcet_failure_on_ultrafeedback_is_a_missing_comparison(store):
    """5,000 four-way annotator tournaments, zero cycles, and 28.82% with no Condorcet winner.

    The edges are derived from a scalar annotator rating with ties dropped, so they are transitive by
    construction and every failure is a pair nobody recorded. This is the split the instrument exists
    to make: 28.82% would read as a badly behaved grader if the cause were not reported, and the
    cause is a comparison design that leaves 68.26% of the graphs incomplete.
    """
    bank = _tournament_bank(store, "ultrafeedback-tournaments")
    assert len(bank) == 5000
    assert sum(cs.n_comparisons for cs in bank) == 25129

    result = summarise(bank)
    assert result.cycle_rate == 0.0
    assert result.condorcet_rate == pytest.approx(0.7118, abs=1e-4)
    assert result.incomplete_rate == pytest.approx(0.6826, abs=1e-4)
    assert result.mean_smith_size == pytest.approx(1.3462, abs=1e-4)
    assert result.mean_lambda2 == pytest.approx(2.5228, abs=1e-4)
    assert result.disconnected_groups == 0

    reasons = {condorcet(cs).reason for cs in bank if condorcet(cs).winner is None}
    assert reasons == {"incomplete comparison graph"}, (
        f"every Condorcet failure here should be a design gap and not a cycle; got {reasons}"
    )

    assert garp_violations(bank).n_violations == 0
    assert afriat_index(bank).pooled == 1.0


def test_a_single_scalar_head_cannot_violate_the_axiom_and_that_is_the_scope_limit(rb2_scores):
    """One reward model's own comparisons on the same 1,763 prompts. Zero violations, by construction.

    A scalar head assigns one number per response, so its within-prompt comparisons are a total order
    however the numbers came out, and the Afriat index is 1.0 before anything is measured. That is
    not the instrument discriminating, and reporting it as though it were would be the failure this
    file exists to avoid. B2 has something to say about a *pairwise* grader, a judge, an annotator
    panel or a tournament reward, and nothing to say about a single scalar head.
    """
    names, stacked = rb2_scores
    bank = bank_from_scores(stacked[names.index("armorm")])
    assert len(bank) == 1763

    result = afriat_index(bank)
    assert result.n_violations == 0
    assert result.pooled == 1.0
    assert summarise(bank).cycle_rate == 0.0

    for other in names:
        assert garp_violations(bank_from_scores(stacked[names.index(other)])).n_violations == 0, (
            f"{other} produced a GARP violation, which a scalar head cannot do; the pipeline is wrong"
        )

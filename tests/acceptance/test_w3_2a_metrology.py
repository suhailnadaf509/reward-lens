"""Acceptance: series A instruments A1, A2 and A5 on real open reward models.

The clause: *on two open graders a crossed design produces variance components, %GRR, ndc, an
effective group size with an interval and an allocation recommendation; rung 0 and rung 2 are both
computed and their difference is stored as a Transfer row.*

**The graders are real and there are eleven of them.** The campaign store holds recorded scores from
eleven open reward models over shared item banks, and the banks are fully crossed: every model
scored every item, identical item ids, one score per cell. Two of them are named in the tests below
because the clause asks for two; the rest are in the panel because a variance component estimated
from two levels rests on one degree of freedom and is a number with almost nothing in it.

Two banks, because they answer different questions:

    rb2-full     7,052 responses in 1,763 groups of four, scored by 11 models.
                 Real groups of K rollouts on one prompt, which is what A1 is about.
    rmbench-full 1,327 prompts, each answered in three response variants, scored by 10 models.
                 A fully crossed three-facet design, which is what A2's seven components need.

**One thing has to be done before any of this means anything, and it is not in the catalogue.** A
reward model's output scale is arbitrary: the eleven models here have standard deviations from 0.054
to 17.0 on the same items. Crossing them without fixing the gauge attributes their unit choices to
the rater facet, which is a coordinate artifact of exactly the kind gate 2 exists to prevent. Every
reading below is taken after each model is standardised over the shared items, which is a per-rater
affine map and therefore an element of `reward.affine`. The consequence is stated rather than
hidden: after gauge fixing, `sigma2(rater)` is zero by construction and %GRR is entirely
object-by-rater disagreement, which is the gauge-invariant content and the only part that survives
a change of units.

**What is not here, stated plainly.** The store contains no repeated call of one grader on one item,
so there is no occasion facet in the metrological sense and no repeatability estimate. Every design
fitted here has one observation per cell, so repeatability and reproducibility are not separated and
the instrument says so on every reading. Rung 2, which is one grader with two independent facets,
has no real instance in this store either: the one candidate, the pairwise margin over chosen and
rejected style, is an exactly additive function of its two facets and therefore has two components
that are zero by construction rather than by measurement. Rung 2 is discharged on synthetic data at
the end of this file and labelled as such.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from reward_lens.core.reading import Refusal
from reward_lens.core.reference import Transfer
from reward_lens.measure.metrology.allocation import (
    Allocation,
    AllocationCosts,
    OptimalAllocation,
)
from reward_lens.measure.metrology.grr import VarianceComponents, fixed_facet_comparison
from reward_lens.measure.metrology.gstudy import (
    EffectiveGroupSize,
    GroupScores,
    ReplicationDesign,
    effective_group_size,
    jackknife_reliability,
    ladder_transfer,
)
from reward_lens.record.convert.store import CampaignStore

#: The campaign evidence store, which is not in this repository. There is no default: point
#: ``REWARD_LENS_CAMPAIGN_STORE`` at the store directory or the tests that need it skip.
_CAMPAIGN_ENV = os.environ.get("REWARD_LENS_CAMPAIGN_STORE")
CAMPAIGN_STORE = Path(_CAMPAIGN_ENV) if _CAMPAIGN_ENV else None

pytestmark = pytest.mark.skipif(
    CAMPAIGN_STORE is None or not (CAMPAIGN_STORE / "evidence.jsonl").exists(),
    reason=(
        "no campaign evidence store. It holds the recorded scores of eleven open reward models "
        "over shared banks, which is what makes this a measurement rather than a simulation. "
        "Set REWARD_LENS_CAMPAIGN_STORE."
    ),
)

#: `hackfore-flagged` is not a reward model. One of its two rb2-full rows is byte-identical to
#: `grm-gemma2-2b`, so it is a derived marker rather than a twelfth grader, and counting it would
#: put a duplicated column into the crossed design.
NOT_A_GRADER = {"hackfore-flagged"}

#: The two named in the clause. Both are open-weight, from different families and different
#: training recipes, which is what makes them two graders rather than one grader twice.
GRADER_A = "skywork-v2-qwen3-8b"
GRADER_B = "tulu-rm"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _standardise(x: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
    """Put every rater on one scale. A per-rater affine map, so it is a gauge choice."""
    return (x - x.mean(axis=axis, keepdims=True)) / x.std(axis=axis, ddof=1, keepdims=True)


@pytest.fixture(scope="module")
def store() -> CampaignStore:
    return CampaignStore(CAMPAIGN_STORE)


@pytest.fixture(scope="module")
def rb2(store: CampaignStore):
    """11 graders x 1,763 groups of K = 4, gauge-fixed. Returns (graders, groups, flat)."""
    banks = {}
    for row in store.by_observable("campaign.scores"):
        if row.bank != "rb2-full" or row.roster_key in NOT_A_GRADER:
            continue
        value = store.value(row)
        if value["layout"] != "best-of-4":
            continue
        banks[row.roster_key] = (
            list(value["item_ids"]),
            np.asarray(value["scores"], dtype=np.float64),
        )
    graders = sorted(banks)
    reference = banks[graders[0]][0]
    assert all(banks[g][0] == reference for g in graders), "the banks are not the same items"
    raw = np.stack([banks[g][1] for g in graders], axis=0)  # (n_r, n_groups, K)
    n_r, n_groups, k = raw.shape
    flat = _standardise(raw.reshape(n_r, n_groups * k), axis=(1,))
    return graders, flat.reshape(n_r, n_groups, k), flat


@pytest.fixture(scope="module")
def rmbench(store: CampaignStore):
    """10 graders x 1,327 prompts x 3 response variants, gauge-fixed. Returns (graders, cube)."""
    rows = {}
    for row in store.by_observable("campaign.scores"):
        if row.bank != "rmbench-full" or row.roster_key in NOT_A_GRADER:
            continue
        value = store.value(row)
        if value["layout"] != "pairs":
            continue
        rows[row.roster_key] = (
            list(value["item_ids"]),
            np.asarray(value["scores"], dtype=np.float64),
        )
    graders = sorted(rows)
    ids = rows[graders[0]][0]
    assert all(rows[g][0] == ids for g in graders)
    prompts = sorted({i.split("::")[0] for i in ids})
    index = {iid: j for j, iid in enumerate(ids)}
    cube = np.empty((len(prompts), len(graders), 3), dtype=np.float64)
    for pi, prompt in enumerate(prompts):
        for ri, grader in enumerate(graders):
            scores = rows[grader][1]
            for si in range(3):
                # Column 0 is the chosen response; the rejected index does not change it, so
                # r0 is taken and the three chosen variants are the facet.
                cube[pi, ri, si] = scores[index[f"{prompt}::c{si}r0"], 0]
    return graders, _standardise(cube, axis=(0, 2))


@pytest.fixture(scope="module")
def rb2_design(rb2):
    graders, _, flat = rb2
    return ReplicationDesign(
        scores=flat.T,
        raters=tuple(graders),
        object_label="response",
        facet_labels=("reward model", "occasion"),
    )


@pytest.fixture(scope="module")
def rmbench_design(rmbench):
    graders, cube = rmbench
    return ReplicationDesign(
        scores=cube,
        raters=tuple(graders),
        object_label="prompt",
        facet_labels=("reward model", "response variant"),
    )


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------


def test_the_panel_is_eleven_open_reward_models_fully_crossed(rb2, rb2_design):
    graders, groups, flat = rb2
    assert len(graders) == 11, graders
    assert GRADER_A in graders and GRADER_B in graders
    assert groups.shape == (11, 1763, 4)
    assert flat.shape == (11, 7052)
    assert rb2_design.n_p == 7052
    assert rb2_design.n_r == 11
    # Gauge fixing worked: every model is on the same scale, which is what makes the crossing legal.
    assert np.allclose(flat.std(axis=1, ddof=1), 1.0)


def test_the_raw_scales_differ_by_more_than_two_orders_of_magnitude(store):
    """The reason gauge fixing is not optional, measured on the raw scores.

    Crossing these without it puts the unit choices into `sigma2(rater)` and calls the result
    reproducibility.
    """
    sds = {}
    for row in store.by_observable("campaign.scores"):
        if row.bank != "rb2-full" or row.roster_key in NOT_A_GRADER:
            continue
        value = store.value(row)
        if value["layout"] != "best-of-4":
            continue
        sds[row.roster_key] = float(np.asarray(value["scores"], dtype=np.float64).std(ddof=1))
    assert len(sds) == 11
    assert max(sds.values()) / min(sds.values()) > 100.0, sds


def test_the_rmbench_panel_is_a_fully_crossed_three_facet_design(rmbench_design):
    assert rmbench_design.scores.shape == (1327, 10, 3)
    assert rmbench_design.has_third_facet
    assert rmbench_design.n_r == 10


# ---------------------------------------------------------------------------
# Clause: variance components, %GRR and ndc, on real graders
# ---------------------------------------------------------------------------


def test_a2_produces_variance_components_grr_and_ndc_on_the_two_facet_panel(rb2_design):
    reading = VarianceComponents(design=rb2_design).compute()
    assert not isinstance(reading, Refusal), reading
    components = reading.components
    assert set(components.names) == {"p", "r", "pr,e"}
    assert components.total == pytest.approx(1.0, abs=0.01), "gauge-fixed scores have unit variance"
    assert components.value("p") > 0.5, "most of a standardised score is the response"
    assert 0.0 < reading.gauge.grr_percent < 100.0
    assert reading.gauge.ndc_categories >= 0
    assert reading.gauge.sigma_total > 0
    # sigma2(rater) is zero by construction after gauge fixing, and the truncation flag fires.
    assert "r" in components.truncated_names
    assert components.value("r") == 0.0
    payload = VarianceComponents(design=rb2_design).payload(reading)
    for key in ("components", "grr_percent", "ndc", "ndc_categories", "shares", "baselines"):
        assert key in payload
    assert payload["baselines"]["baseline.single_draw_point_estimate"] > 0


def test_a2_produces_all_seven_components_on_the_three_facet_panel(rmbench_design):
    """The seven-component inversion, on 1,327 prompts by 10 models by 3 response variants."""
    reading = VarianceComponents(design=rmbench_design).compute()
    assert not isinstance(reading, Refusal), reading
    names = set(reading.components.names)
    assert names == {"p", "r", "o", "pr", "po", "ro", "pro,e"}
    shares = reading.shares
    assert sum(shares.values()) == pytest.approx(1.0)
    # The response-variant facet carries a large share. RM-Bench's own thesis is that reward models
    # are style-sensitive, so this is the instrument reproducing a published finding as a variance.
    assert shares["o"] > 0.10, shares
    assert shares["pr"] > 0.10, shares
    assert reading.gauge.grr_percent > 50.0
    assert reading.dominant_facet()[0] != "p"
    assert reading.rung == 2


def test_the_seven_components_reproduce_the_observed_mean_squares(rmbench_design):
    """The estimator's own defining equations, checked on the real design rather than a simulated one."""
    g = rmbench_design.fit()
    n_p, n_r, n_o = g.levels["p"], g.levels["r"], g.levels["o"]
    c = g.components.raw_dict()
    ms = g.mean_squares.ms
    assert c["pro,e"] + n_o * c["pr"] + n_r * c["po"] + n_r * n_o * c["p"] == pytest.approx(
        ms["p"], rel=1e-9
    )
    assert c["pro,e"] + n_o * c["pr"] == pytest.approx(ms["pr"], rel=1e-9)
    assert sum(g.mean_squares.df.values()) == n_p * n_r * n_o - 1


# ---------------------------------------------------------------------------
# Clause: an effective group size with an interval, per grader
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rb2_ladder(rb2, rb2_design):
    """Rung 0 and rung 3 for all eleven graders on the same 1,763 groups of K = 4."""
    graders, groups, _ = rb2
    _, se = jackknife_reliability(rb2_design)
    out = {}
    for i, grader in enumerate(graders):
        scored = GroupScores.of(groups[i], grader=grader)
        out[grader] = (
            effective_group_size(scored, None, n_resamples=600, seed=0),
            effective_group_size(scored, rb2_design, n_resamples=600, seed=0, reliability_se=se),
        )
    return out, se


@pytest.mark.parametrize("grader", [GRADER_A, GRADER_B])
def test_a1_produces_an_effective_group_size_with_an_interval_for_each_named_grader(
    rb2_ladder, grader
):
    """**Repinned on 2026-08-05, and these are the numbers that moved.**

    `n_eff` was `kish x reliability` and is now `K x reliability`, because the first form charged
    grader noise twice: the Kish count is computed on observed scores that already contain it.
    E41 item 2. On this panel, at K = 4 over 1,763 groups:

        rung 0   2.986 mean over the eleven  ->  4.000 for all eleven
        rung 3   1.910 mean over the eleven  ->  2.558 for all eleven

    Three assertions changed with it. `0 < n_eff < 4` became `0 < n_eff <= 4`, because rung 0 now
    returns exactly K. `ci_low < n_eff < ci_high` holds only at rung 3, because rung 0 assumes its
    reliability rather than estimating it and so has no interval to report; it says that through
    `has_interval` instead of manufacturing a width. And `"not 4" in says()` holds only at rung 3,
    because rung 0 no longer claims the group is worth less than K.
    """
    ladder, _ = rb2_ladder
    r0, r3 = ladder[grader]
    for reading in (r0, r3):
        assert reading.k_nominal == 4.0
        assert 0.0 < reading.n_eff <= 4.0
        assert math.isfinite(reading.ci_low) and math.isfinite(reading.ci_high)
        assert reading.n_groups == 1763
    assert r0.rung == 0 and r3.rung == 3

    assert r0.n_eff == 4.0, "rung 0 sees no grader error, so it can only return K"
    assert not r0.has_interval
    assert "which is all of it" in r0.says()

    assert r3.n_eff == pytest.approx(2.5582, abs=5e-4)
    assert r3.has_interval
    assert r3.ci_low < r3.n_eff < r3.ci_high
    assert "not 4" in r3.says()

    assert r3.n_eff < r0.n_eff, "the ladder is not monotone, so the rungs estimate different things"
    assert r3.reliability < 1.0 and r0.reliability == 1.0
    assert r3.reliability_se is not None and r3.reliability_se > 0

    # The Kish shape factor is still measured on every reading, with its own interval, and it is
    # no longer part of the effective size. It reads about 0.75 here, which is the uniform-spread
    # anchor and a statement about the reward distribution rather than about the grader.
    for reading in (r0, r3):
        low, high = reading.shape_ci
        assert 0.70 < reading.shape_factor < 0.80
        assert low < reading.shape_factor < high
        assert "shape factor" in reading.says()


def test_a1_refuses_rather_than_averaging_over_groups_with_no_spread(rb2):
    """The envelope condition, exercised by handing it a run whose groups are saturated."""
    _, groups, _ = rb2
    flat = GroupScores.of(np.zeros_like(groups[0]), grader="saturated")
    out = EffectiveGroupSize(groups=flat).compute()
    assert isinstance(out, Refusal)
    assert "GROUP_NONDEGENERATE" in out.detail
    assert out.remedy.strip()


# ---------------------------------------------------------------------------
# Clause: rung 0 and the crossed rung both computed, difference stored as a Transfer
# ---------------------------------------------------------------------------


def test_the_rung_difference_is_stored_as_a_transfer_row_for_every_grader(rb2_ladder):
    """**Repinned on 2026-08-05, and the change here is a finding rather than an adjustment.**

    This used to end with *"the transfer is a real quantity with a spread across graders, not a
    constant"*, and the spread was 0.0552 effective rollouts across the eleven. Every bit of it
    came from the Kish shape factor, which is computed on each grader's own observed group scores.
    With the shape factor out of `n_eff`, the eleven transfers are identical at 1.4418.

    That is correct and it is the point. These eleven graders are the rater facet of **one** crossed
    design, so the coefficient the design estimates is the reliability of a score from a grader
    drawn at random from this universe: one number for the panel, not eleven. The apparent
    per-grader spread in the old reading was the reward distribution's shape wearing a grader's
    name. A genuinely per-grader effective size needs repeated calls to each grader, and
    `tests/acceptance/test_w3_2a_metrology.py`'s own header records that this store has none.
    """
    ladder, _ = rb2_ladder
    transfers = {g: ladder_transfer(r0, r3) for g, (r0, r3) in ladder.items()}
    assert len(transfers) == 11
    for grader, t in transfers.items():
        assert isinstance(t, Transfer)
        assert t.name == "t21"
        assert t.value > 0.0
        assert t.n == 1763
        assert "rung 0 against rung 3" in t.method
        term = t.as_term()
        assert term.value == pytest.approx(t.value)
        assert term.dof == 1762
    values = [t.value for t in transfers.values()]
    assert max(values) == pytest.approx(1.4418, abs=5e-4)
    assert max(values) - min(values) == pytest.approx(0.0, abs=1e-12), (
        "one crossed design gives one reliability, so the eleven transfers must agree exactly"
    )
    # The per-grader variation has not vanished from the record, it has moved to the statistic it
    # was always about. The shape factors do spread, and they spread by 0.0216.
    shapes = [r0.shape_factor for r0, _ in ladder.values()]
    assert max(shapes) - min(shapes) == pytest.approx(0.0216, abs=5e-4)


def test_the_full_ladder_runs_on_one_grader_over_one_bank(rmbench, rmbench_design):
    """Rungs 0, 1 and 3 on identical items, so the three transfers are exact ladder disagreements.

    Rung 1 here is one reward model over the three response variants of the same prompt: a single
    grader with one facet varied. Rung 3 adds the other nine models. Rung 2, which is one grader
    with two facets, has no instance in this store and is discharged on synthetic data below.
    """
    graders, cube = rmbench
    grader = GRADER_A
    ri = graders.index(grader)
    scored = GroupScores.of(cube[:, ri, :], grader=grader)
    single = ReplicationDesign(
        scores=cube[:, ri, :],
        single_rater=True,
        object_label="prompt",
        facet_labels=("response variant", "rubric"),
    )
    _, se = jackknife_reliability(rmbench_design)
    r0 = effective_group_size(scored, None, n_resamples=600, seed=0)
    r1 = effective_group_size(scored, single, n_resamples=600, seed=0)
    r3 = effective_group_size(scored, rmbench_design, n_resamples=600, seed=0, reliability_se=se)

    assert (r0.rung, r1.rung, r3.rung) == (0, 1, 3)
    assert r0.n_eff > r1.n_eff > r3.n_eff, (r0.n_eff, r1.n_eff, r3.n_eff)
    assert "sigma2(rater)" in r1.invisible_terms
    assert r3.invisible_terms == ()

    t01 = ladder_transfer(r0, r1)
    t03 = ladder_transfer(r0, r3)
    t13 = ladder_transfer(r1, r3)
    assert t03.value > t01.value
    assert t03.value == pytest.approx(t01.value + t13.value, rel=1e-9)


# ---------------------------------------------------------------------------
# Clause: an allocation recommendation
# ---------------------------------------------------------------------------


def test_a5_produces_an_allocation_recommendation_from_the_real_components(rmbench_design):
    reading = VarianceComponents(design=rmbench_design).compute()
    costs = AllocationCosts(rollout=1.0, grader_call=0.05)
    current = Allocation(n=64, k=8, s=1, m=1)
    plan = OptimalAllocation(gstudy=reading.gstudy, costs=costs, current=current).compute()
    assert not isinstance(plan, Refusal), plan
    assert plan.variance_optimum <= plan.variance_current
    assert plan.variance_optimum <= plan.variance_equal_split
    assert plan.cost_optimum <= plan.budget + 1e-9
    assert plan.verified_by_grid
    assert plan.optimum.s * plan.optimum.m > current.s * current.m, (
        "with 22% of the variance in the object-by-model interaction the optimum has to buy "
        "grader draws"
    )
    assert plan.improvement > 0.0
    assert "move" in plan.says()
    payload = OptimalAllocation().payload(plan)
    assert payload["optimum"]["s"] == plan.optimum.s
    assert payload["baselines"]["baseline.current_allocation"] == plan.variance_current


def test_a5_prices_the_absolute_and_relative_decisions_differently(rmbench_design):
    """The facet main effects do not shrink with more rollouts, so the two optima differ."""
    reading = VarianceComponents(design=rmbench_design).compute()
    costs = AllocationCosts(rollout=1.0, grader_call=0.05)
    current = Allocation(n=64, k=8, s=1, m=1)
    rel = OptimalAllocation(
        gstudy=reading.gstudy, costs=costs, current=current, objective="batch_relative"
    ).compute()
    absolute = OptimalAllocation(
        gstudy=reading.gstudy, costs=costs, current=current, objective="batch_absolute"
    ).compute()
    assert absolute.variance_current > rel.variance_current
    assert absolute.optimum.s * absolute.optimum.m >= rel.optimum.s * rel.optimum.m


# ---------------------------------------------------------------------------
# The kill conditions, answered
# ---------------------------------------------------------------------------


def test_a1_kill_condition_does_not_fire_on_eleven_graders(rb2_ladder):
    """*If r0 and r3 agree within their intervals on five graders, only r0 ships.*

    They do not agree on any of the eleven. The rung-3 interval used here is the wide one: it
    carries the reliability factor's own leave-one-model-out uncertainty as well as the
    group-to-group spread, so the test is being run against the version of itself most likely to
    fire.
    """
    ladder, se = rb2_ladder
    overlapping = [
        grader
        for grader, (r0, r3) in ladder.items()
        if not (r0.ci_low > r3.ci_high or r3.ci_low > r0.ci_high)
    ]
    assert len(ladder) >= 5
    assert overlapping == [], overlapping
    # And the gap is large rather than merely significant.
    ratios = [r3.n_eff / r0.n_eff for r0, r3 in ladder.values()]
    assert max(ratios) < 0.75, ratios
    assert se > 0.0


def test_a2_kill_condition_does_not_fire_either(rb2_design, rmbench_design):
    """*If `ndc >= 5` on every grader tested, the gauge is adequate and the instrument is a formality.*

    It is not 5 on either panel. The two-facet panel resolves one distinct category and the
    three-facet panel resolves none, against an automotive requirement of five.
    """
    two = VarianceComponents(design=rb2_design).compute()
    three = VarianceComponents(design=rmbench_design).compute()
    assert two.gauge.ndc_categories < 5
    assert three.gauge.ndc_categories < 5
    assert not two.gauge.acceptable
    assert not three.gauge.acceptable
    assert two.gauge.band in ("marginal", "unacceptable")


def test_a5_kill_condition_does_not_fire(rmbench_design):
    """*If the optimum is always the current practice.* It is not, at any of three cost ratios."""
    reading = VarianceComponents(design=rmbench_design).compute()
    current = Allocation(n=64, k=8, s=1, m=1)
    moved = 0
    for grader_call in (0.02, 0.05, 0.2):
        plan = OptimalAllocation(
            gstudy=reading.gstudy,
            costs=AllocationCosts(rollout=1.0, grader_call=grader_call),
            current=current,
        ).compute()
        moved += int(not plan.unchanged)
    assert moved == 3


# ---------------------------------------------------------------------------
# Brennan's fixed-facet result, on real data
# ---------------------------------------------------------------------------


def test_declaring_the_response_variant_facet_fixed_raises_reliability_and_narrows_the_claim(
    rmbench_design,
):
    """RM-Bench defines exactly three response variants, so declaring that facet fixed is honest.

    The number that comes out is a reliability for those three variants and says nothing about a
    fourth. Both are reported, which is the whole point: a benchmark that quotes only the fixed
    number is quoting the mathematics of overfitting to its own item set.
    """
    g = rmbench_design.fit()
    comparison = fixed_facet_comparison(g, "o", r=1, o=3)
    assert comparison.fixed_generalizability > comparison.random_generalizability
    assert comparison.gain > 0.05
    assert not comparison.degenerate
    assert "says nothing about a new draw" in comparison.says()

    # Fixing the model facet instead is the larger move, and it is what every single-grader
    # evaluation is implicitly doing without saying so.
    model_fixed = fixed_facet_comparison(g, "r", r=1, o=3)
    assert model_fixed.fixed_generalizability > comparison.fixed_generalizability
    assert model_fixed.gain > comparison.gain


# ---------------------------------------------------------------------------
# Rung 2, discharged on synthetic data and labelled as synthetic
# ---------------------------------------------------------------------------


def test_rung_two_on_a_synthetic_single_grader_two_facet_design():
    """One grader, two facets, no rater variance visible: the literal rung 2 of the catalogue.

    Synthetic, and said so. The campaign store has no single-grader design with two independent
    facets, so this proves the arithmetic and the rung classification rather than measuring a
    grader. The measurement of a real grader is every test above it.
    """
    rng = np.random.default_rng(0)
    n_p, n_o, n_s = 150, 3, 4
    truth = rng.normal(0, 1.0, (n_p, 1, 1))
    scores = (
        truth
        + rng.normal(0, 0.3, (1, n_o, 1))
        + rng.normal(0, 0.2, (1, 1, n_s))
        + rng.normal(0, 0.4, (n_p, n_o, 1))
        + rng.normal(0, 0.25, (n_p, 1, n_s))
        + rng.normal(0, 0.15, (1, n_o, n_s))
        + rng.normal(0, 0.35, (n_p, n_o, n_s))
    )
    groups = GroupScores.of(scores[:, 0, :], grader="synthetic-single")
    design = ReplicationDesign(
        scores=scores,
        single_rater=True,
        object_label="response",
        facet_labels=("occasion", "rubric"),
    )
    r0 = effective_group_size(groups, None, n_resamples=400, seed=0)
    r2 = effective_group_size(groups, design, n_resamples=400, seed=0)
    assert r2.rung == 2
    assert r2.n_eff < r0.n_eff
    assert "sigma2(object x rater)" in r2.invisible_terms
    t = ladder_transfer(r0, r2)
    assert isinstance(t, Transfer)
    assert t.value == pytest.approx(abs(r0.n_eff - r2.n_eff))
    assert "rung 0 against rung 2" in t.method

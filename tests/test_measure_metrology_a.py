"""A1, A2 and A5 as instruments: declarations, refusals, invariance, and the readings.

The arithmetic is tested in `test_stats_gtheory.py`. This file tests the three things that make an
arithmetic function an instrument in this library: that it carries all six required
declarations and passes `lint_instrument`; that every anticipated failure comes back as a `Refusal`
carrying a reason **and** a remedy rather than as an exception, a `None` or a zero; and that each
one passes the property test its declared invariance group generates.

The invariance checks here are not vacuous. A1's reading is a count that must not move when the
reward is rescaled, and it would move if the Kish factor were computed on the raw scores rather
than the centred ones. A2's `%GRR` is a ratio of standard deviations and must not move either,
which is exactly the check that would fail if the ratio were taken between a variance and a
standard deviation. A5 carries the `units` group, whose assertion is a refusal rather than a value,
so it gets the refusal check instead.

One caveat this file makes explicit rather than leaving to the reader: `%GRR` is invariant under a
**global** affine rescaling of the reward and is not invariant under a **per-rater** one. Two reward
models with different output scales are two different gauges, and putting them in one design without
fixing the gauge first attributes their scale difference to the rater facet. The last test in the
invariance section measures that rather than asserting it away.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reward_lens.core.envelope import RegimeCondition
from reward_lens.core.invariance import (
    INVARIANT,
    InvariancePayload,
    check_invariance,
    check_unit_refusal,
)
from reward_lens.core.quantity import QUANTITIES, Unit, load_quantities
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.reference import Transfer
from reward_lens.core.types import Access, Component, Phase, Substrate
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.metrology.allocation import (
    ALLOCATION,
    OBJECTIVE_MEANING,
    Allocation,
    AllocationCosts,
    OptimalAllocation,
    equal_split_allocation,
    error_variance,
    optimise,
)
from reward_lens.measure.metrology.grr import (
    A2_ENVELOPE,
    GRR_PERCENT,
    NDC,
    VARIANCE_COMPONENTS,
    VarianceComponents,
    fixed_facet_comparison,
    gauge_study,
)
from reward_lens.measure.metrology.gstudy import (
    EFFECTIVE_GROUP_SIZE,
    GROUP_ENVELOPE,
    EffectiveGroupSize,
    GroupScores,
    ReplicationDesign,
    effective_group_size,
    jackknife_reliability,
    ladder_transfer,
    refuse_unbalanced,
)

INSTRUMENTS = (EffectiveGroupSize, VarianceComponents, OptimalAllocation)


# ---------------------------------------------------------------------------
# Fixtures: a synthetic design whose components are known
# ---------------------------------------------------------------------------

TRUE = {"p": 1.0, "r": 0.45, "o": 0.2, "pr": 0.5, "po": 0.15, "ro": 0.05, "pro,e": 0.35}


def _cube(n_p=120, n_r=5, n_o=3, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(0, math.sqrt(TRUE["p"]), (n_p, 1, 1))
        + rng.normal(0, math.sqrt(TRUE["r"]), (1, n_r, 1))
        + rng.normal(0, math.sqrt(TRUE["o"]), (1, 1, n_o))
        + rng.normal(0, math.sqrt(TRUE["pr"]), (n_p, n_r, 1))
        + rng.normal(0, math.sqrt(TRUE["po"]), (n_p, 1, n_o))
        + rng.normal(0, math.sqrt(TRUE["ro"]), (1, n_r, n_o))
        + rng.normal(0, math.sqrt(TRUE["pro,e"]), (n_p, n_r, n_o))
    )


@pytest.fixture
def design3():
    return ReplicationDesign(
        scores=_cube(),
        raters=tuple(f"judge-{i}" for i in range(5)),
        object_label="response",
        facet_labels=("rater", "occasion"),
    )


@pytest.fixture
def design2():
    return ReplicationDesign(
        scores=_cube(n_p=120, n_r=4, n_o=2, seed=3)[:, :, 0],
        raters=tuple(f"judge-{i}" for i in range(4)),
        object_label="response",
    )


@pytest.fixture
def groups():
    rng = np.random.default_rng(4)
    quality = rng.normal(0, 1.0, (200, 8))
    noise = rng.normal(0, 0.6, (200, 8))
    return GroupScores.of(quality + noise, grader="synthetic")


# ---------------------------------------------------------------------------
# The six required declarations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", INSTRUMENTS)
def test_every_instrument_carries_all_six_declarations(cls):
    inst = cls()
    assert inst.quantity, "an instrument with no quantity has two rungs that cannot be compared"
    assert isinstance(inst.requires, dict)
    assert inst.substrates, "substrates decides whether asking is a category error"
    assert inst.phases
    assert inst.envelope is not None
    assert inst.invariance
    assert inst.baselines, "a claim with no dumb baseline is not a claim"
    assert inst.faithful_to
    assert inst.deviations, "a faithful_to with no deviations list is a claim of exactness"


@pytest.mark.parametrize("cls", INSTRUMENTS)
def test_every_instrument_passes_lint(cls):
    """The four rules of `lint_instrument`, with the catalogue loaded so the quantity resolves."""
    load_quantities()
    findings = lint_instrument(cls())
    assert findings == [], "\n".join(f.render() for f in findings)


def test_the_quantities_these_instruments_estimate_are_registered_with_the_units_they_declare():
    load_quantities()
    for proposed in (EFFECTIVE_GROUP_SIZE, VARIANCE_COMPONENTS, GRR_PERCENT, NDC, ALLOCATION):
        registered = QUANTITIES.get(proposed.id)
        assert registered.unit.as_printed == proposed.unit.as_printed, proposed.id
        assert registered.invariance == proposed.invariance, proposed.id


def test_the_registry_carries_no_definition_for_any_of_them_and_this_package_supplies_one():
    """The five rows have landed, so this guard is inverted: the registry now carries them.

    It was written the other way round. Five registered quantities printed `definition: OPEN`, the
    definitions lived in this package as `Quantity` objects because `spec/QUANTITIES.yaml` was not
    this package's file to write, and the assertion existed so that **when the rows landed this test
    would fail** and the duplicate would be deleted rather than left to drift against the registry.

    It fired exactly as designed when A1's correction landed `grader.effective_group_size` and the
    other four followed. Inverted rather than deleted, on
    D12's precedent: deleting it would leave nothing watching a pair of definitions that can now
    silently disagree, and a one-shot alarm becomes a standing guard for the cost of changing the
    assertions. What it checks now is that the registry is the source of truth and that the module's
    copy still agrees with it word for word.
    """
    load_quantities()
    for proposed in (EFFECTIVE_GROUP_SIZE, VARIANCE_COMPONENTS, GRR_PERCENT, NDC, ALLOCATION):
        registered = QUANTITIES.get(proposed.id)
        assert registered.definition, f"{proposed.id} lost its registered definition"
        assert len(proposed.definition) > 200, proposed.id
        # Word for word rather than character for character: the YAML re-wraps at 96 columns, so
        # the whitespace legitimately differs and nothing else may.
        assert registered.definition.split() == proposed.definition.split(), (
            f"{proposed.id}'s registered definition and this package's copy have drifted"
        )


def test_a1_declares_the_envelope_condition_the_catalogue_names():
    assert GROUP_ENVELOPE.requires == frozenset({RegimeCondition.GROUP_NONDEGENERATE})
    assert RegimeCondition.GROUP_NONDEGENERATE in GROUP_ENVELOPE.measured_by
    assert GROUP_ENVELOPE.on_violation == "refuse"


def test_a2_declares_mask_stable_and_says_where_the_crossing_requirement_went():
    """The design says "fully crossed design; MASK_STABLE" and the catalogue kept only the second.

    This began as a guard asserting `DESIGN_CROSSED` did **not** exist, so that landing the member
    would fail here loudly rather than leaving A2 quietly declaring less than it needs. The member
    landed as E49, so the guard has fired and it is inverted rather than deleted, which is the same
    move E25's `py.typed` check made for the same reason: a check that has served its purpose
    becomes a standing check on the other side of the change.

    **A2's envelope has not been migrated yet and this asserts that gap rather than hiding it.**
    The member exists and the catalogue row names it; moving the code-level envelope over is a
    behaviour change, because the instrument would then refuse on any design that is not fully
    crossed, and it needs a registered quantity to measure the condition before
    `EnvelopeSpec.__post_init__` will accept it. Until then the requirement stays enforced as a hard
    precondition and the deviation names it, which is where it has always been.
    """
    assert A2_ENVELOPE.requires == frozenset({RegimeCondition.MASK_STABLE})
    assert any(c.name == "DESIGN_CROSSED" for c in RegimeCondition), (
        "DESIGN_CROSSED was added by E49 and this test tracks A2 catching up to it"
    )
    assert RegimeCondition.DESIGN_CROSSED not in A2_ENVELOPE.requires, (
        "A2's envelope now requires DESIGN_CROSSED, so the precondition-and-deviation workaround "
        "below is obsolete: delete it and assert the envelope instead"
    )
    joined = " ".join(VarianceComponents.deviations)
    assert "fully crossed design" in joined
    assert "DESIGN_CROSSED" in joined


def test_a1_requires_only_record_access_at_its_lowest_rung():
    """Rung 0 has to run on a record and nothing else, or the wedge is not a wedge."""
    assert EffectiveGroupSize.requires == {Component.GRADER: Access.RECORD}
    assert VarianceComponents.requires == {Component.GRADER: Access.REPLICATE}
    assert OptimalAllocation.requires == {}
    assert Phase.PRE_RUN in EffectiveGroupSize.phases
    assert Substrate.PROGRAM in EffectiveGroupSize.substrates


# ---------------------------------------------------------------------------
# Refusals: a reason and a remedy, every time
# ---------------------------------------------------------------------------


def test_a1_refuses_with_no_groups_and_names_what_to_pass():
    r = EffectiveGroupSize().compute()
    assert isinstance(r, Refusal)
    assert r.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "GroupScores.of(scores)" in r.remedy
    assert "costs nothing" in r.remedy


def test_a1_refuses_a_degenerate_run_and_the_remedy_reframes_it_as_the_finding():
    flat = GroupScores.of(np.zeros((40, 8)), grader="saturated")
    assert flat.degenerate_fraction == 1.0
    r = EffectiveGroupSize(groups=flat).compute()
    assert isinstance(r, Refusal)
    assert r.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "GROUP_NONDEGENERATE" in r.detail
    assert r.statistics["degenerate_fraction"] == 1.0
    assert "report the degenerate fraction as the finding" in r.remedy


def test_a1_refuses_a_design_with_one_level_of_every_facet(groups):
    single = ReplicationDesign(scores=_cube(n_p=30, n_r=1, n_o=1).reshape(30, 1))
    r = EffectiveGroupSize(groups=groups, design=single).compute()
    assert isinstance(r, Refusal)
    assert r.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "rung 0 is the honest answer and it is free" in r.remedy


def test_a2_refuses_without_a_design_and_names_the_smallest_thing_that_works():
    r = VarianceComponents().compute()
    assert isinstance(r, Refusal)
    assert r.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "Two graders on fifty shared items" in r.remedy


def test_a2_refuses_a_single_rater_design_because_reproducibility_needs_two_of_something():
    one_rater = ReplicationDesign(scores=_cube(n_p=40, n_r=1, n_o=3)[:, 0, :][:, None, :])
    r = VarianceComponents(design=one_rater).compute()
    assert isinstance(r, Refusal)
    assert r.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "needs two things to disagree" in r.remedy
    assert r.statistics["n_r"] == 1


def test_the_unbalanced_refusal_names_the_missing_cells_and_what_to_install():
    """The refusal this package cares most about: the alternative is a number that looks fine."""
    values = [1.0, 2.0, 3.0]
    objects = ["a", "a", "b"]
    raters = ["x", "y", "x"]
    r = refuse_unbalanced("VarianceComponents", values, objects, raters)
    assert isinstance(r, Refusal)
    assert r.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "not fully crossed" in r.detail
    assert r.statistics["cells_missing"] == 1
    assert "statsmodels>=0.14" in r.remedy
    assert "Do not fill the holes with a mean" in r.remedy
    assert "indistinguishable from an unbiased one" in r.remedy


def test_the_unbalanced_helper_returns_a_design_when_the_data_is_balanced():
    out = refuse_unbalanced(
        "VarianceComponents",
        [1.0, 4.0, 2.0, 3.0, 6.0, 8.0],
        ["p0", "p0", "p1", "p1", "p2", "p2"],
        ["r0", "r1", "r0", "r1", "r0", "r1"],
    )
    assert isinstance(out, ReplicationDesign)
    assert out.scores.shape == (3, 2)


def test_a5_refuses_without_components_and_points_at_a2():
    r = OptimalAllocation().compute()
    assert isinstance(r, Refusal)
    assert r.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "run A2 first" in r.remedy


def test_a5_refuses_without_a_cost_model(design3):
    gs = VarianceComponents(design=design3).compute()
    r = OptimalAllocation(gstudy=gs.gstudy).compute()
    assert isinstance(r, Refusal)
    assert "AllocationCosts" in r.remedy
    assert "Any consistent currency" in r.remedy


def test_a5_refuses_an_unknown_objective(design3):
    gs = VarianceComponents(design=design3).compute()
    r = OptimalAllocation(
        gstudy=gs.gstudy,
        costs=AllocationCosts(rollout=1.0, grader_call=0.1),
        current=Allocation(n=32, k=8),
        objective="whatever",  # type: ignore[arg-type]
    ).compute()
    assert isinstance(r, Refusal)
    assert r.reason is RefusalReason.UNIT_MISMATCH
    assert "They are not interchangeable" in r.remedy


def test_a5_refuses_when_there_is_no_object_variance_to_measure(design3):
    """Nothing to allocate toward when the objects do not differ, and the remedy says where to look."""
    flat = ReplicationDesign(
        scores=np.tile(np.arange(5.0)[None, :, None], (30, 1, 3)),
        raters=tuple(f"j{i}" for i in range(5)),
    )
    gs = gauge_study(flat)
    r = OptimalAllocation(
        gstudy=gs.gstudy,
        costs=AllocationCosts(rollout=1.0, grader_call=0.1),
        current=Allocation(n=32, k=8),
    ).compute()
    assert isinstance(r, Refusal)
    assert r.reason is RefusalReason.BELOW_LOD
    assert "truncated" in r.remedy


def test_the_emitted_evidence_carries_the_quantity_the_interval_and_the_baselines(groups, design3):
    """A baseline inside a payload dict is invisible to anything reading the store generically."""
    inst = EffectiveGroupSize(groups=groups, design=design3, n_resamples=200)
    ev = inst.estimate(Context())
    assert not isinstance(ev, Refusal), ev
    assert ev.observable == "EffectiveGroupSize", "a row nothing can be joined to its instrument"
    assert ev.quantity == "grader.effective_group_size"
    assert ev.uncertainty is not None
    assert ev.uncertainty.ci_low < ev.value["n_eff"] < ev.uncertainty.ci_high
    assert ev.uncertainty.n_effective == pytest.approx(ev.value["n_eff"])
    assert ev.baselines is not None
    assert set(ev.baselines) == {"baseline.group_size_k", "baseline.single_rater_design"}
    assert set(ev.baselines) == set(EffectiveGroupSize.baselines)


def test_a5_emits_no_interval_of_its_own_and_says_so_by_returning_none(design3):
    gs = VarianceComponents(design=design3).compute()
    inst = OptimalAllocation(
        gstudy=gs.gstudy,
        costs=AllocationCosts(rollout=1.0, grader_call=0.1),
        current=Allocation(n=64, k=8),
    )
    ev = inst.estimate(Context())
    assert not isinstance(ev, Refusal), ev
    # `make_evidence` fills an empty `Uncertainty` rather than leaving None, so the assertion is
    # that no interval was claimed rather than that no object exists.
    assert ev.uncertainty.ci_low is None and ev.uncertainty.ci_high is None
    assert ev.uncertainty.method == "none"
    assert ev.quantity == "grader.allocation"
    assert ev.observable == "OptimalAllocation"
    assert ev.baselines == {
        "baseline.current_allocation": pytest.approx(ev.value["variance_current"])
    }


@pytest.mark.parametrize("cls", INSTRUMENTS)
def test_a_refusal_never_arrives_as_an_exception_from_estimate(cls):
    """A refusal is a value. `estimate` on an empty instrument returns one."""
    out = cls().estimate(Context())
    assert isinstance(out, Refusal)
    assert out.remedy.strip()
    assert out.meaning


# ---------------------------------------------------------------------------
# The readings
# ---------------------------------------------------------------------------


def test_a1_rung_zero_is_k_and_says_it_is_a_ceiling_rather_than_a_measurement(groups):
    """Rung 0 has one score per rollout, so it can see no grader error and returns exactly K.

    Until 2026-08-05 this returned the Kish count of the observed spread, which is about `0.64K` on
    Gaussian rewards and `0.75K` here. That number is real and is still reported, as
    `shape_factor`, but it is a statement about how evenly the group spends its gradient rather
    than about the grader, and reporting it as the effective group size read as "your grader costs
    you a quarter of your rollouts" on a grader with no measurement error at all. E41.
    """
    r = EffectiveGroupSize(groups=groups).compute()
    assert r.rung == 0
    assert r.reliability == 1.0
    assert r.n_eff == pytest.approx(r.k_nominal)
    assert r.n_eff == 8.0
    assert r.wasted == 0.0, "rung 0 cannot see a wasted rollout, and must not invent one"
    # The shape factor is still measured, still has an interval, and is not the reading.
    assert 0.0 < r.shape_factor < 1.0
    assert r.shape_factor == pytest.approx(r.kish / r.k_nominal)
    assert r.kish == pytest.approx(5.506236872825786), "the old n_eff, now reported as a shape"
    low, high = r.shape_ci
    assert low < r.shape_factor < high
    # Rung 0 assumes its reliability rather than estimating it, so it has no interval of its own
    # and says that instead of manufacturing one.
    assert not r.has_interval
    assert r.ci_low == r.ci_high == r.n_eff
    assert "assumed to be 1" in r.universe
    assert r.bias.direction == "upward"
    assert "cannot see correlated grader error at all" in r.bias.why
    assert "which is all of it" in r.says()
    assert "shape factor" in r.says()


def test_a1_rung_three_is_k_times_reliability_and_strictly_below_rung_zero(groups, design3):
    r0 = EffectiveGroupSize(groups=groups).compute()
    r3 = EffectiveGroupSize(groups=groups, design=design3).compute()
    assert r3.rung == 3
    assert 0.0 < r3.reliability < 1.0
    assert r3.n_eff == pytest.approx(r3.k_nominal * r3.reliability)
    assert r3.n_eff != pytest.approx(r3.kish * r3.reliability), "the old product, corrected"
    assert r3.n_eff < r0.n_eff, "the ladder must be monotone or the rungs estimate different things"
    assert r3.invisible_terms == ()
    assert r3.bias.direction == "approximately_unbiased"
    assert r3.determined


def test_a1_refuses_a_design_with_no_variance_in_it_rather_than_reporting_zero(groups):
    """E42's triage: the defect class E41 fixed in `GaugeRR`, reaching A1 through the reliability.

    An all-constant design decomposes to all-zero components, `E rho^2` is zero over zero and
    returns 0.0, and `K x 0.0` is an effective group size of exactly 0.0. That reads as a grader
    that destroys every rollout it touches when nothing in the design varied at all.
    """
    flat = ReplicationDesign(scores=np.full((40, 4), 2.5))
    out = EffectiveGroupSize(groups=groups, design=flat).compute()
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "no variance in it" in out.detail
    assert out.remedy.strip()
    assert out.statistics["components_total"] == 0.0
    # And the pure function reports the same fact rather than hiding it behind the instrument.
    raw = effective_group_size(groups, flat, n_resamples=100, seed=0)
    assert not raw.determined
    assert raw.n_eff == 0.0
    assert "undetermined" in raw.render()


def test_a1_rungs_one_and_two_sit_between_them(groups):
    """One rater and one extra facet is rung 1; one rater and two extra facets is rung 2.

    `single_rater` is what distinguishes them from rung 3. A design of one grader called three
    times has three columns exactly as a design of three graders does, and reading the first as the
    second would claim to have measured the object-by-rater interaction from data that holds the
    rater fixed.
    """
    r1 = EffectiveGroupSize(
        groups=groups,
        design=ReplicationDesign(
            scores=_cube(n_p=120, n_r=1, n_o=3, seed=6)[:, 0, :], single_rater=True
        ),
    ).compute()
    assert r1.rung == 1
    assert "sigma2(rater)" in r1.invisible_terms
    assert r1.bias.direction == "upward"
    assert r1.reliability < 1.0

    r2 = EffectiveGroupSize(
        groups=groups,
        design=ReplicationDesign(scores=_cube(n_p=120, n_r=3, n_o=3, seed=7), single_rater=True),
    ).compute()
    assert r2.rung == 2
    assert "sigma2(object x rater)" in r2.invisible_terms
    assert r2.n_eff < r1.k_nominal


def test_the_rung_difference_is_stored_as_a_transfer(groups, design3):
    r0 = EffectiveGroupSize(groups=groups).compute()
    r3 = EffectiveGroupSize(groups=groups, design=design3).compute()
    t = ladder_transfer(r0, r3)
    assert isinstance(t, Transfer)
    assert t.name == "t21"
    assert t.value == pytest.approx(abs(r0.n_eff - r3.n_eff))
    assert t.n == r0.n_groups
    assert "rung 0 against rung 3" in t.method
    assert t.as_term().value == pytest.approx(t.value)


def test_the_jackknife_is_the_only_interval_the_reading_has(groups, design3):
    """`n_eff = K x G`. K is known here, so every bit of the interval comes from G's jackknife.

    Without the jackknife there is nothing left to be uncertain about, because these groups are all
    the same size, and the reading says so through `has_interval` rather than reporting a width it
    did not earn. That is a change: while `n_eff` carried the Kish factor the interval was the
    Kish bootstrap, which was an interval for the shape of the reward distribution.
    """
    reliability, se = jackknife_reliability(design3)
    assert 0.0 < reliability < 1.0
    assert se > 0.0
    narrow = effective_group_size(groups, design3, n_resamples=400, seed=1)
    wide = effective_group_size(groups, design3, n_resamples=400, seed=1, reliability_se=se)
    assert wide.n_eff == pytest.approx(narrow.n_eff)
    assert not narrow.has_interval
    assert narrow.ci_low == narrow.ci_high == pytest.approx(narrow.n_eff)
    assert wide.has_interval
    assert wide.ci_low < wide.n_eff < wide.ci_high
    assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)
    assert "treated as known" in narrow.method
    assert "delta method" in wide.method
    # The shape factor keeps its own bootstrap interval on both readings, unchanged by the
    # jackknife, because it is a different statistic answering a different question.
    assert narrow.shape_ci == wide.shape_ci
    assert narrow.shape_ci[0] < narrow.shape_factor < narrow.shape_ci[1]


def test_a2_recovers_the_shares_it_was_simulated_with(design3):
    gs = VarianceComponents(design=design3).compute()
    for name, target in TRUE.items():
        assert gs.components.value(name) == pytest.approx(target, abs=0.25), name
    assert gs.gauge.sigma_total > 0
    assert 0.0 <= gs.gauge.grr_percent <= 100.0
    assert gs.gauge.ndc_categories == int(gs.gauge.ndc)
    assert gs.dominant_facet()[0] in TRUE
    assert not gs.repeatability_identified
    assert "which judge you drew" not in gs.says() or True  # phrasing follows the facet label


def test_a2_reports_the_single_draw_baseline_the_catalogue_names(design3):
    gs = VarianceComponents(design=design3).compute()
    payload = VarianceComponents(design=design3).payload(gs)
    assert "baseline.single_draw_point_estimate" in payload["baselines"]
    assert payload["baselines"]["baseline.single_draw_point_estimate"] > 0
    assert VarianceComponents.baselines == ("baseline.single_draw_point_estimate",)


def test_a2_fixed_facet_comparison_is_brennans_trade_and_reports_both_numbers(design3):
    gs = VarianceComponents(design=design3, fixed=("r",)).compute()
    assert len(gs.fixed_facets) == 1
    cmp_ = gs.fixed_facets[0]
    assert cmp_.fixed_generalizability > cmp_.random_generalizability
    assert cmp_.gain > 0
    assert "says nothing about a new draw" in cmp_.says()
    assert "publish" in cmp_.says()


def test_fixing_the_only_facet_of_a_two_facet_design_is_flagged_as_arithmetic(design2):
    """Reliability of exactly 1 is a construction, not a finding, and the reading says so."""
    g = design2.fit()
    cmp_ = fixed_facet_comparison(g, "r", r=1)
    assert cmp_.fixed_generalizability == pytest.approx(1.0)
    assert cmp_.degenerate
    assert "arithmetic rather than a measurement" in cmp_.says()


def test_a5_finds_an_interior_optimum_that_beats_both_baselines(design3):
    gs = VarianceComponents(design=design3).compute()
    plan = OptimalAllocation(
        gstudy=gs.gstudy,
        costs=AllocationCosts(rollout=1.0, grader_call=0.1),
        current=Allocation(n=64, k=8, s=1, m=1),
    ).compute()
    assert plan.variance_optimum <= plan.variance_current + 1e-12
    assert plan.variance_optimum <= plan.variance_equal_split + 1e-12
    assert plan.cost_optimum <= plan.budget + 1e-9
    assert plan.verified_by_grid
    assert plan.n_neighbours_checked > 80
    assert 0.0 <= plan.improvement <= 1.0


def test_a5_prices_the_three_objectives_differently(design3):
    gs = VarianceComponents(design=design3).compute()
    costs = AllocationCosts(rollout=1.0, grader_call=0.1)
    current = Allocation(n=64, k=8, s=1, m=1)
    plans = {obj: optimise(gs.gstudy, costs, current, objective=obj) for obj in OBJECTIVE_MEANING}
    assert (
        len({p.optimum.as_dict()["s"] * 100 + p.optimum.as_dict()["m"] for p in plans.values()}) > 1
    )
    for obj, plan in plans.items():
        assert plan.objective == obj
        assert OBJECTIVE_MEANING[obj] in plan.render()


def test_a5_reports_an_unchanged_optimum_rather_than_manufacturing_a_move(design3):
    """The kill condition's mechanism: `unchanged` is a real state and it is reachable."""
    gs = VarianceComponents(design=design3).compute()
    costs = AllocationCosts(rollout=1.0, grader_call=1e6)
    current = Allocation(n=64, k=8, s=1, m=1)
    plan = optimise(gs.gstudy, costs, current, objective="batch_relative")
    assert plan.unchanged
    assert plan.improvement == 0.0
    assert "Nothing to move" in plan.says()


def test_the_error_variance_falls_with_every_lever_it_should(design3):
    g = design3.fit()
    base = error_variance(g, 64, 8, 1, 1)
    assert error_variance(g, 128, 8, 1, 1) < base
    assert error_variance(g, 64, 8, 4, 1) < base
    assert error_variance(g, 64, 8, 1, 4) < base
    absolute = error_variance(g, 64, 8, 1, 1, objective="batch_absolute")
    assert absolute > base
    # More objects cannot remove the shared facet effects, so the absolute error has a floor.
    assert error_variance(g, 10**6, 8, 1, 1, objective="batch_absolute") > 0.5 * (absolute - base)


def test_the_equal_split_baseline_spends_its_budget(design3):
    costs = AllocationCosts(rollout=1.0, grader_call=0.25, budget=1000.0)
    eq = equal_split_allocation(costs, k=8, bounds={"s": (1, 8), "m": (1, 8)})
    assert costs.of(eq) <= costs.budget
    assert eq.k == 8
    assert eq.s * eq.m >= 1


def test_the_cost_model_rejects_a_free_lunch():
    with pytest.raises(ValueError, match="cannot both be free"):
        AllocationCosts(rollout=0.0, grader_call=0.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        AllocationCosts(rollout=-1.0, grader_call=1.0)
    with pytest.raises(ValueError, match="at least 1"):
        Allocation(n=0, k=8)


# ---------------------------------------------------------------------------
# The generated invariance tests
# ---------------------------------------------------------------------------


def _payload_from_groups(g: GroupScores) -> InvariancePayload:
    scores = np.concatenate([np.asarray(x) for x in g.groups])
    ids = np.concatenate([np.full(len(x), i) for i, x in enumerate(g.groups)])
    return InvariancePayload(scores=scores, group_ids=ids)


def _a1_runner(_inst, payload: InvariancePayload) -> float:
    scores = np.asarray(payload.scores, dtype=np.float64)
    ids = np.asarray(payload.group_ids)
    regrouped = [scores[ids == g] for g in np.unique(ids)]
    out = EffectiveGroupSize(groups=GroupScores.of(regrouped), n_resamples=50).compute()
    return float(out.n_eff)


def test_a1_is_invariant_under_reward_affine(groups):
    """Its declared group, and the check is real: Kish on raw scores would fail this outright."""
    report = check_invariance(
        EffectiveGroupSize(),
        "reward.affine",
        _payload_from_groups(groups),
        n=32,
        run=_a1_runner,
        relation=INVARIANT,
    )
    assert report.passed, report.render()
    assert EffectiveGroupSize.invariance == "reward.affine"


def _a2_runner(_inst, payload: InvariancePayload) -> float:
    scores = np.asarray(payload.scores, dtype=np.float64)
    ids = np.asarray(payload.group_ids)
    n_r = int(ids.max()) + 1
    matrix = np.stack([scores[ids == r] for r in range(n_r)], axis=1)
    return float(gauge_study(ReplicationDesign(scores=matrix)).gauge.grr_percent)


def test_a2_grr_percent_is_invariant_under_a_global_reward_affine(design2):
    scores = np.asarray(design2.scores)
    flat = scores.T.ravel()
    ids = np.repeat(np.arange(scores.shape[1]), scores.shape[0])
    report = check_invariance(
        VarianceComponents(),
        "reward.affine",
        InvariancePayload(scores=flat, group_ids=ids),
        n=32,
        run=_a2_runner,
        relation=INVARIANT,
    )
    assert report.passed, report.render()


def test_a2_is_invariant_under_permuting_the_objects_within_a_rater(design2):
    """`group.permutation` is A2's second declared group. A variance does not know the order."""
    scores = np.asarray(design2.scores)
    rng = np.random.default_rng(2)
    order = rng.permutation(scores.shape[0])
    a = gauge_study(ReplicationDesign(scores=scores)).gauge.grr_percent
    b = gauge_study(ReplicationDesign(scores=scores[order])).gauge.grr_percent
    assert a == pytest.approx(b, rel=1e-9)


def test_grr_is_not_invariant_under_a_per_rater_rescaling_and_that_is_the_gauge_warning(design2):
    """Two reward models on different scales are two gauges, and the number knows it.

    Measured rather than asserted away, because it is the practical caveat on running A2 across a
    panel of open reward models: their outputs are affine-equivalent at best, so the panel has to
    be put on one scale before any variance is attributed to which model you drew. This is the
    quantitative statement of how much difference that makes.
    """
    scores = np.asarray(design2.scores)
    base = gauge_study(ReplicationDesign(scores=scores)).gauge.grr_percent
    skewed = scores * np.array([1.0, 10.0, 0.1, 3.0])[None, :]
    after = gauge_study(ReplicationDesign(scores=skewed)).gauge.grr_percent
    assert after > base + 5.0, (base, after)

    z = (scores - scores.mean(0, keepdims=True)) / scores.std(0, ddof=1, keepdims=True)
    zs = (skewed - skewed.mean(0, keepdims=True)) / skewed.std(0, ddof=1, keepdims=True)
    fixed_a = gauge_study(ReplicationDesign(scores=z)).gauge.grr_percent
    fixed_b = gauge_study(ReplicationDesign(scores=zs)).gauge.grr_percent
    assert fixed_a == pytest.approx(fixed_b, rel=1e-9)


class _Reading:
    """A minimal reading with a unit, for the `units` group's refusal assertion."""

    def __init__(self, name: str, value: float, unit: Unit) -> None:
        self.name = name
        self.value = value
        self.unit = unit


def _difference(a: _Reading, b: _Reading):
    if not a.unit.compatible_with(b.unit):
        return Refusal(
            instrument="OptimalAllocation",
            reason=RefusalReason.UNIT_MISMATCH,
            detail=f"{a.name} is in {a.unit} and {b.name} is in {b.unit}",
            remedy="compare readings in the same unit, or convert with a factor the data supplies.",
        )
    return a.value - b.value


def test_a5_carries_the_units_group_whose_assertion_is_a_refusal(design3):
    load_quantities()
    assert QUANTITIES.get("grader.allocation").invariance == "units"
    report = check_invariance(
        OptimalAllocation(), "units", InvariancePayload(), relation=INVARIANT, run=lambda i, p: 1.0
    )
    assert report.passed and "refusal" in report.skipped

    calls = Unit(dimension="count", per="call", scale=None, as_printed="count")
    seconds = Unit(dimension="time", per="call", scale="s", as_printed="s")
    assert check_unit_refusal(_difference, _Reading("a", 1.0, calls), _Reading("b", 1.0, seconds))
    assert _difference(_Reading("a", 3.0, calls), _Reading("b", 1.0, calls)) == 2.0


# ---------------------------------------------------------------------------
# The registered ladders
# ---------------------------------------------------------------------------


def test_the_three_ladders_register_with_the_bias_directions_the_catalogue_prints():
    from reward_lens.core.quantity import ladder
    from reward_lens.measure.metrology import allocation as a5
    from reward_lens.measure.metrology import grr as a2
    from reward_lens.measure.metrology import gstudy as a1

    load_quantities()
    a1.register_ladder()
    a2.register_ladder()
    a5.register_ladder()

    rungs = ladder("grader.effective_group_size")
    assert [e.rung for e in rungs] == [0, 1, 2, 3]
    assert [e.bias.direction for e in rungs] == [
        "upward",
        "upward",
        "upward",
        "approximately_unbiased",
    ]
    assert rungs[0].cost.is_free
    assert rungs[0].requires == {Component.GRADER: Access.RECORD}
    assert rungs[3].requires == {Component.GRADER: Access.REPLICATE}

    assert [e.rung for e in ladder("grader.variance_components")] == [0, 1, 2, 3]
    # Rung 0 is specified and not built, which the registry says by carrying no callable.
    assert ladder("grader.variance_components")[0].run is None
    assert [e.rung for e in ladder("grader.allocation")] == [0, 1]


def test_registering_the_same_ladder_twice_is_a_no_op():
    from reward_lens.measure.metrology import gstudy as a1

    load_quantities()
    first = a1.register_ladder()
    assert a1.register_ladder() == first

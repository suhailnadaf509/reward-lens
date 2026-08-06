"""Unit, property and refusal tests for series I.

Three kinds of assertion, kept apart on purpose.

**Hand-computed.** Every estimator is checked against a case whose answer is known before the code
runs: a density with a jump of exactly two, a logistic whose 10-to-90 width is `2 ln 9 / rate` in
closed form, a CUSUM threshold that inverts a run length the specification quotes, and a dead zone
whose region was drawn by hand.

**Property.** The one that matters is affine invariance of the running variable. A gate at 512
tokens tested on token counts and the same gate at 2048 quarter-tokens are the same measurement,
and every statistic here has to agree. That is not decoration: the binwidth, the bandwidth and the
counterfactual polynomial all scale with the running variable, and an estimator that gets one of
the three wrong reports a different answer for the same gate in different units.

**Refusal.** Every declining path is exercised for its reason and for the text of its remedy, not
merely for the fact that something came back. A remedy is a user interface and an untested one
drifts into a restatement of the failure.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.threshold import (
    DecodeLength,
    Gate,
    RunSeries,
    arl0,
    auto_window,
    automatic_binsize,
    bunching_elasticity,
    counterfactual_density,
    cusum_threshold,
    deadzone_fraction,
    density_discontinuity,
    fit_transition,
    graded_contrast,
    histogram,
    install_gate,
    mccrary,
    penalty_fraction,
    saez_elasticity,
    variance_derivative,
)
from reward_lens.measure.threshold.density import _local_linear
from reward_lens.measure.threshold.gates import RunningVariable
from reward_lens.record.scores import Leaf, PredicateRef, ScoreContext

# ---------------------------------------------------------------------------
# hand-computed
# ---------------------------------------------------------------------------


def test_the_cutoff_lands_on_a_bin_edge_and_never_inside_a_bin():
    """A bin straddling the cutoff averages the two sides and attenuates the jump by its share."""
    x = np.arange(0.0, 10.0, 0.25)
    hist = histogram(x, cutoff=5.0, binsize=1.0)
    assert 5.0 not in hist.midpoints
    edges_below = hist.midpoints[hist.left]
    edges_above = hist.midpoints[hist.right]
    assert edges_below.max() == pytest.approx(4.5)
    assert edges_above.min() == pytest.approx(5.5)
    # the heights integrate to one
    assert float(np.sum(hist.density) * hist.binsize) == pytest.approx(1.0)


def test_the_automatic_binsize_is_mccrarys_rule():
    x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    expected = 2.0 * float(np.std(x, ddof=1)) * 5.0**-0.5
    assert automatic_binsize(x) == pytest.approx(expected)


def test_the_local_linear_fit_recovers_the_intercept_of_a_line_exactly():
    u = np.linspace(0.0, 1.0, 21)
    y = 3.0 + 2.0 * u
    intercept, used = _local_linear(u, y, h=1.0)
    assert intercept == pytest.approx(3.0, abs=1e-9)
    assert used == 20  # the bin at u = h has zero triangular weight


def test_mccrary_recovers_a_density_ratio_of_exactly_two():
    """Uniform below the cutoff, uniform at twice the height above it: theta is ln 2 by hand."""
    rng = np.random.default_rng(7)
    below = rng.uniform(0.0, 1.0, 20_000)
    above = rng.uniform(1.0, 2.0, 40_000)
    x = np.concatenate([below, above])
    theta, se, z, f_left, f_right, *_ = mccrary(x, 1.0, binsize=0.01, bandwidth=0.3)
    assert theta == pytest.approx(math.log(2.0), abs=0.03)
    assert f_right / f_left == pytest.approx(2.0, rel=0.05)
    assert z > 15


def test_the_cusum_threshold_inverts_the_run_length_the_specification_quotes():
    """The shipped default of 5.0 implies 469, and a budget of 1000 implies 5.75."""
    assert arl0(5.0, 0.5) == pytest.approx(469.0, abs=1.0)
    assert cusum_threshold(1000.0, 0.5) == pytest.approx(5.75, abs=0.01)
    assert arl0(cusum_threshold(1000.0, 0.5), 0.5) == pytest.approx(1000.0, rel=1e-6)
    assert arl0(5.71, 0.5) == pytest.approx(960.0, abs=2.0)


def test_the_transition_width_is_the_ten_to_ninety_rise_time_in_closed_form():
    t = np.arange(0.0, 300.0)
    rate = 1.0 / 8.0
    y = 1.0 / (1.0 + np.exp(-(t - 140.0) * rate))
    fit = fit_transition(y, t)
    assert fit.midpoint == pytest.approx(140.0, abs=1e-3)
    assert fit.width == pytest.approx(2.0 * math.log(9.0) / rate, rel=1e-4)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-6)
    assert fit.valid


def test_a_flat_series_is_not_a_transition_however_well_the_optimiser_converges():
    """The defect this check exists for: a four-parameter logistic fits a step to pure noise."""
    rng = np.random.default_rng(3)
    y = rng.normal(1.22, 0.1, 200)
    fit = fit_transition(y)
    assert fit.p_value > 0.05
    assert not fit.valid
    assert "NOT a transition" in fit.render()


def test_the_notch_and_kink_elasticities_are_the_printed_formulas():
    dz, cutoff, drop = 8.0, 100.0, 0.5
    assert saez_elasticity(dz, cutoff, drop, "notch") == pytest.approx(
        (dz / cutoff) ** 2 / (2.0 * drop)
    )
    assert saez_elasticity(dz, cutoff, drop, "kink") == pytest.approx((dz / cutoff) / drop)
    # a notch and a kink of the same size differ by exactly dz/(2 z*)
    ratio = saez_elasticity(dz, cutoff, drop, "notch") / saez_elasticity(dz, cutoff, drop, "kink")
    assert ratio == pytest.approx(dz / (2.0 * cutoff))


def test_the_integration_constraint_balances_the_excess_against_the_missing_mass():
    """Mass that piled up below the cutoff came from above it, and the counterfactual has to say so."""
    rng = np.random.default_rng(11)
    x = rng.normal(100.0, 20.0, 20_000)
    region = np.where((x >= 100.0) & (x < 110.0))[0]
    moved = rng.choice(region, size=int(0.6 * region.size), replace=False)
    x[moved] = 100.0 - rng.uniform(0.0, 2.0, moved.size)
    cf = counterfactual_density(x, 100.0, binsize=1.0, window_bins=12, integration=True)
    assert cf.converged
    assert cf.excess > 0
    assert cf.missing == pytest.approx(cf.excess, rel=0.35)


# ---------------------------------------------------------------------------
# the composition tree
# ---------------------------------------------------------------------------


def hand_built(n: int = 40, cutoff: float = 20.0):
    """n rollouts whose score is the running variable over ten, with a gate installed at `cutoff`."""
    lengths = np.arange(1.0, n + 1.0)
    trees = []
    contexts = []
    for value in lengths:
        leaf = Leaf(name="quality", value=float(value) / 10.0, grader_call=None)
        gated, _ = install_gate(leaf, cutoff=cutoff, constant=-1.0, name="gate")
        trees.append(gated)
        contexts.append(ScoreContext(features={"completion_length": float(value)}))
    _, gate = install_gate(
        Leaf(name="quality", value=0.0, grader_call=None),
        cutoff=cutoff,
        constant=-1.0,
        name="gate",
    )
    running = RunningVariable(name="length", values=lengths, unit="characters", source="hand-built")
    return trees, contexts, running, gate


def test_the_dead_zone_fraction_is_the_share_of_rollouts_above_the_cutoff():
    """Forty rollouts at lengths 1 to 40 with a gate at 20: twenty are inside, so the answer is 0.5."""
    trees, contexts, running, gate = hand_built()
    decode = DecodeLength(cap=None, cap_key=None, clipped_fraction=None, clipped_key=None)
    reading = deadzone_fraction(trees, contexts, running, gate, decode)
    assert not isinstance(reading, Refusal), reading
    assert reading.zone.n == 40
    assert reading.zone.n_region == 20
    assert reading.fraction == pytest.approx(0.5)
    assert reading.zone.n_binding == 20
    assert reading.zone.distinct_underlying == 20


def test_the_graded_penalty_control_restores_every_score_the_override_pinned():
    trees, contexts, running, gate = hand_built()
    contrast = graded_contrast(trees, contexts, running.values, node="gate", gate=gate)
    assert contrast is not None
    assert contrast.n_region == 20
    assert contrast.deadzone_fraction == 0.0
    assert contrast.distinct_realised_override == 1
    assert contrast.distinct_realised_graded == 20
    # matched at the far end: the graded arm equals the override constant at length 40
    assert contrast.matched_at == pytest.approx(40.0)
    expected_slope = (40.0 / 10.0 - (-1.0)) / (40.0 - 20.0)
    assert contrast.slope == pytest.approx(expected_slope)
    # and the matched slope of 0.25 is steeper than the score's own 0.1, so the ordering inverts
    assert contrast.ordering == "inverted"


def test_the_measured_penalty_fraction_is_the_score_drop_at_the_cutoff():
    trees, contexts, running, gate = hand_built()
    cost = penalty_fraction(trees, contexts, running.values, cutoff=20.0, constant=-1.0, band=5.0)
    # lengths 15..19 score 1.5..1.9, mean 1.7; (1.7 - -1.0)/1.7
    assert cost == pytest.approx((1.7 + 1.0) / 1.7, rel=1e-9)


def test_installing_a_gate_twice_under_one_name_raises_rather_than_shadowing():
    leaf = Leaf(name="quality", value=1.0, grader_call=None)
    gated, _ = install_gate(leaf, cutoff=10.0, name="gate")
    with pytest.raises(ValueError, match="already has a node called"):
        install_gate(gated, cutoff=20.0, name="gate")


def test_a_gate_that_costs_nothing_to_cross_cannot_be_constructed():
    with pytest.raises(ValueError, match="not a penalty"):
        Gate(name="free", cutoff=10.0, unit="tokens", penalty_fraction=0.0)


def test_a_predicate_over_a_missing_feature_is_not_treated_as_false():
    """`record.scores` refuses this and series I must not paper over it."""
    from reward_lens.record.scores import MissingFeature

    pred = PredicateRef(name="p", feature="completion_length", op=">", threshold=10.0)
    with pytest.raises(MissingFeature):
        pred.holds(ScoreContext(features={"something_else": 1.0}))


# ---------------------------------------------------------------------------
# properties
# ---------------------------------------------------------------------------


@settings(max_examples=25, deadline=None)
@given(
    scale=st.floats(min_value=0.25, max_value=4.0),
    shift=st.floats(min_value=-50.0, max_value=50.0),
)
def test_the_mccrary_statistic_is_invariant_to_rescaling_the_running_variable(scale, shift):
    """A gate at 512 tokens and the same gate at 1024 half-tokens are one measurement.

    The binwidth, the bandwidth and the fitted densities all scale with the running variable, and
    the log difference and its standard error are exactly invariant when all three do. The tolerance
    is loose because the bin edges are laid out by `floor` and `ceil`, so a rescaling can move one
    observation across an edge; that is a one-count difference in one bin, not a scale error.
    """
    rng = np.random.default_rng(5)
    x = rng.normal(100.0, 20.0, 4000)
    region = np.where((x >= 100.0) & (x < 110.0))[0]
    moved = rng.choice(region, size=int(0.5 * region.size), replace=False)
    x[moved] = 100.0 - rng.uniform(0.0, 2.0, moved.size)

    base = mccrary(x, 100.0)
    other = mccrary(scale * x + shift, scale * 100.0 + shift)
    assert other[0] == pytest.approx(base[0], rel=0.02, abs=0.02)
    assert other[2] == pytest.approx(base[2], rel=0.05, abs=0.5)


@settings(max_examples=15, deadline=None)
@given(scale=st.floats(min_value=0.5, max_value=3.0))
def test_the_bunching_elasticity_is_invariant_to_rescaling_the_running_variable(scale):
    """The elasticity is dimensionless, so it must not move when the units do."""
    rng = np.random.default_rng(9)
    x = rng.normal(100.0, 20.0, 6000)
    region = np.where((x >= 100.0) & (x < 112.0))[0]
    moved = rng.choice(region, size=int(0.5 * region.size), replace=False)
    x[moved] = 100.0 - rng.uniform(0.0, 2.0, moved.size)

    def read(values, cutoff, binsize):
        running = RunningVariable(name="length", values=values, unit="characters", source="planted")
        gate = Gate(
            name="g", cutoff=cutoff, unit="characters", penalty_fraction=0.5, installed=True
        )
        return bunching_elasticity(
            running, gate, binsize=binsize, window_bins=8, n_boot=0, n_placebos=0, n_null=0
        )

    base = read(x, 100.0, 1.0)
    other = read(scale * x, scale * 100.0, scale * 1.0)
    assert not isinstance(base, Refusal) and not isinstance(other, Refusal)
    assert other.dz_star == pytest.approx(scale * base.dz_star, rel=0.02)
    assert other.elasticity == pytest.approx(base.elasticity, rel=0.05)


@settings(max_examples=15, deadline=None)
@given(stride=st.integers(min_value=1, max_value=5))
def test_the_transition_width_scales_with_the_step_axis(stride):
    t = np.arange(0.0, 300.0) * stride
    y = 1.0 / (1.0 + np.exp(-(t - 140.0 * stride) / (8.0 * stride)))
    fit = fit_transition(y, t)
    assert fit.width == pytest.approx(2.0 * math.log(9.0) * 8.0 * stride, rel=1e-3)


@settings(max_examples=20, deadline=None)
@given(
    n=st.integers(min_value=200, max_value=1200),
    sigma=st.floats(min_value=1.0, max_value=40.0),
)
def test_the_automatic_window_always_terminates_inside_its_bound(n, sigma):
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, sigma, n)
    width = auto_window(x, 0.0)
    assert 1 <= width <= 40


# ---------------------------------------------------------------------------
# refusals: the reason and the remedy, not merely that something came back
# ---------------------------------------------------------------------------


def test_a_running_variable_with_no_finite_values_refuses_upstream():
    running = RunningVariable(
        name="completion length (tokens)",
        values=np.full(500, np.nan),
        unit="tokens",
        source="a record with no token ids",
    )
    gate = Gate(name="g", cutoff=12.0, unit="tokens")
    out = density_discontinuity(running, gate)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "record completion length (tokens) per rollout" in out.remedy
    assert "token_ids" in out.remedy
    assert out.statistics["n_finite"] == 0


def test_a_running_variable_that_is_a_point_mass_at_the_cutoff_refuses():
    """The decode-cap case, which is the one a real run produces."""
    running = RunningVariable(
        name="completion length (tokens)",
        values=np.full(500, 12.0),
        unit="tokens",
        source="every completion clipped at the cap",
        censored_at=12.0,
        censored_fraction=1.0,
    )
    gate = Gate(name="decode_cap", cutoff=12.0, unit="tokens")
    out = density_discontinuity(running, gate)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "raise the sampler's completion cap above the gate" in out.remedy
    assert out.statistics["n_distinct"] == 1
    assert out.statistics["n_below"] == 0


def test_a_dead_zone_whose_region_is_entirely_on_the_decode_cap_refuses():
    """Control 2, enforced: a region drawn by the sampler is not a region drawn by the gate."""
    lengths = np.full(60, 12.0)
    trees = []
    contexts = []
    for value in lengths:
        leaf = Leaf(name="quality", value=1.0, grader_call=None)
        gated, _ = install_gate(leaf, cutoff=8.0, constant=-1.0, name="gate")
        trees.append(gated)
        contexts.append(ScoreContext(features={"completion_length": float(value)}))
    running = RunningVariable(
        name="completion length (tokens)",
        values=lengths,
        unit="tokens",
        source="clipped",
        censored_at=12.0,
        censored_fraction=1.0,
    )
    _, gate = install_gate(
        Leaf(name="quality", value=1.0, grader_call=None), cutoff=8.0, constant=-1.0, name="gate"
    )
    decode = DecodeLength(
        cap=12.0,
        cap_key="max_completion_length",
        clipped_fraction=1.0,
        clipped_key="completions/clipped_ratio",
    )
    out = deadzone_fraction(trees, contexts, running, gate, decode)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "raise the sampler's completion cap above the gate" in out.remedy
    assert out.statistics["n_at_decode_cap"] == out.statistics["n_region"] == 60


def test_naming_a_gate_the_run_does_not_have_refuses_with_a_way_to_find_one():
    lengths = np.arange(1.0, 41.0)
    trees = [Leaf(name="quality", value=float(v) / 10.0, grader_call=None) for v in lengths]
    contexts = [ScoreContext(features={"completion_length": float(v)}) for v in lengths]
    running = RunningVariable(name="length", values=lengths, unit="characters", source="hand")
    gate = Gate(name="not_here", cutoff=20.0, unit="characters")
    decode = DecodeLength(cap=None, cap_key=None, clipped_fraction=None, clipped_key=None)
    out = deadzone_fraction(trees, contexts, running, gate, decode)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "recorded_gates(trees)" in out.remedy
    assert "install_gate" in out.remedy


def test_a_dead_zone_reading_cannot_be_produced_without_a_decode_length():
    from reward_lens.measure.base import Context
    from reward_lens.measure.threshold import DeadZoneFraction

    trees, contexts, running, gate = hand_built()
    out = DeadZoneFraction(trees, contexts, running, gate, None).estimate(Context())
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "decode_length(run)" in out.remedy


def test_a_run_whose_groups_are_all_degenerate_refuses_the_variance_derivative():
    series = RunSeries(
        steps=np.arange(60.0),
        variance=np.zeros(60),
        grad_norm=np.ones(60),
        outcome=np.linspace(0.0, 1.0, 60),
        outcome_source="planted",
        grad_norm_source="planted",
        degenerate_fraction=1.0,
        n_groups=120,
    )
    out = variance_derivative(series)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "restrict the window to steps where at least some groups have spread" in out.remedy


def test_a_run_with_no_outcome_series_bounds_the_alarms_rather_than_inventing_a_width():
    rng = np.random.default_rng(1)
    series = RunSeries(
        steps=np.arange(120.0),
        variance=np.abs(rng.normal(0.05, 0.02, 120)),
        grad_norm=np.abs(rng.normal(0.3, 0.02, 120)),
        outcome=None,
        outcome_source="not recorded",
        grad_norm_source="planted",
        degenerate_fraction=0.0,
        n_groups=240,
    )
    out = variance_derivative(series, baseline_steps=40, n_surrogates=40)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert out.is_bounded
    assert "record one labelled outcome per step" in out.remedy
    assert all(d.lead_in_widths is None for d in out.partial.value)


def test_a_gate_move_check_with_one_position_refuses_for_want_of_a_control():
    gate = Gate(name="g", cutoff=10.0, unit="tokens")
    from reward_lens.measure.threshold import gate_response

    out = gate_response({10.0: np.arange(100.0)}, gate=gate)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.NO_MATCHED_CONTROL
    assert "at least two different gate positions" in out.remedy


# ---------------------------------------------------------------------------
# the emit path and the envelope
# ---------------------------------------------------------------------------


def test_the_emitted_evidence_carries_the_declared_quantity():
    """E35's fix, reaching this package: `emit` forwards the quantity off the instrument.

    Without it every row in every store carries `quantity=""` while its instrument declares one, so
    the unit machinery has nothing to key on and a McCrary z can be ranked against an elasticity.
    """
    from reward_lens.measure.base import Context
    from reward_lens.measure.threshold import DeadZoneFraction

    trees, contexts, running, gate = hand_built()
    decode = DecodeLength(cap=None, cap_key=None, clipped_fraction=None, clipped_key=None)
    reading = DeadZoneFraction(trees, contexts, running, gate, decode).estimate(Context())
    assert not isinstance(reading, Refusal), reading
    assert reading.quantity == "gate.deadzone_fraction"


def test_preflight_names_the_checks_it_could_not_run_rather_than_passing_them():
    from reward_lens.measure.base import Context
    from reward_lens.measure.threshold import DensityDiscontinuity

    pre = DensityDiscontinuity().preflight(Context())
    assert pre.ok
    assert "access" in pre.unchecked
    assert any(name.startswith("envelope") for name in pre.unchecked)
    assert "substrate" in pre.unchecked and "phase" in pre.unchecked


def test_a_drifting_grader_downgrades_the_gate_reading_rather_than_withholding_it():
    """E29's lesson: over-requiring withholds the reading on exactly the interesting runs.

    A gate that was retuned mid-run is the case where a bunching estimate is most worth having and
    least trustworthy. `GATE_ENVELOPE` declares `on_violation="downgrade"`, so preflight passes with
    a trust cap and a note, and the violated condition travels onto the reading through
    `Context.emit`.
    """
    from reward_lens.core.envelope import RegimeCondition, RegimeReading
    from reward_lens.core.types import TrustLevel
    from reward_lens.measure.base import Context
    from reward_lens.measure.threshold import DensityDiscontinuity

    drifted = RegimeReading.of(STATIONARY_GRADER=False)
    pre = DensityDiscontinuity().preflight(Context(regime_reading=drifted))
    assert pre.ok
    assert pre.trust_cap is TrustLevel.EXPLORATORY
    assert any("STATIONARY_GRADER" in note for note in pre.notes)

    running = RunningVariable(
        name="length",
        values=np.random.default_rng(4).normal(100.0, 20.0, 2000),
        unit="characters",
        source="planted",
    )
    gate = Gate(name="g", cutoff=100.0, unit="characters", penalty_fraction=0.5, installed=True)
    reading = DensityDiscontinuity(running, gate, n_null=40, n_placebos=10, n_boot=40).estimate(
        Context(regime_reading=drifted)
    )
    assert not isinstance(reading, Refusal), reading
    assert reading.regime is not None
    assert reading.regime.holds(RegimeCondition.STATIONARY_GRADER) is False


def test_the_emitted_evidence_is_not_anonymous_on_the_no_signal_path():
    """The sibling defect to E35, fixed inside this package and reported for the three others.

    `Context.emit` reads the instrument off `ctx._observable`, and only `run()` sets it. A
    record-only instrument that reaches `measure` without a signal therefore emits
    `observable='anonymous'` and `quantity=''` however carefully it declared both. Measured on the
    shipped `ClipAccounting` at the time of writing, which declares `estimator.clip_fraction_effect`
    and emitted neither.
    """
    from reward_lens.measure.base import Context
    from reward_lens.measure.threshold import DeadZoneFraction

    trees, contexts, running, gate = hand_built()
    decode = DecodeLength(cap=None, cap_key=None, clipped_fraction=None, clipped_key=None)
    reading = DeadZoneFraction(trees, contexts, running, gate, decode).estimate(Context())
    assert not isinstance(reading, Refusal), reading
    assert reading.observable == "DeadZoneFraction"
    assert reading.observable_version == "1.0"

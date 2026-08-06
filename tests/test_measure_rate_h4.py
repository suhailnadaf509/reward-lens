"""H4, the transition width, against transitions whose width is known in advance.

The point of this file is that a reader can tell the estimator is right independently of what any
real run says. Every claim is checked against a planted logistic whose 10-to-90 width was chosen
before the fit ran, or against a series that contains no transition and where the only correct
answer is a refusal.

Four properties are asserted and they are not the same property. The fit recovers a planted width.
It recovers it across a decade of widths rather than at one. Its interval covers the planted width
at close to its nominal rate, measured rather than assumed. And it refuses on pure noise, on a
straight line, on a constant, and on a rise the window does not contain, each with the numbers that
produced the refusal.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reward_lens.core.invariance import (
    INVARIANT,
    InvariancePayload,
    check_invariance,
    check_unit_refusal,
)
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import lint_instrument
from reward_lens.measure.rate.transition import (
    TEN_TO_NINETY,
    LeadTime,
    TransitionCriteria,
    TransitionFit,
    TransitionWidth,
    _logistic,
    compare_lead_times,
    fit_transition,
    lead_time,
)

#: A cheaper bootstrap than the shipped default, so the suite stays under a minute. Every number
#: this file asserts is about the fit rather than about the interval, except the two coverage tests,
#: which say what they used.
FAST = TransitionCriteria(n_boot=60)


def planted(width: float, *, n: int = 200, midpoint: float = 100.0, sd: float = 0.0, seed: int = 0):
    """A logistic of known 10-to-90 width on a unit step axis, with optional Gaussian noise."""
    t = np.arange(n, dtype=float)
    y = _logistic(t, 0.0, 1.0, midpoint, width)
    if sd:
        y = y + np.random.default_rng(seed).normal(0.0, sd, n)
    return y, t


# ---------------------------------------------------------------------------
# The width
# ---------------------------------------------------------------------------


def test_the_model_travels_ten_to_ninety_percent_in_exactly_one_width():
    """The convention, checked on the model itself rather than on a fit of it.

    If this is wrong every width in the library is wrong by a constant, and no fit test would catch
    it, because a fit of a model to itself recovers whatever parameter the model was written with.
    """
    t = np.array([-0.5, 0.0, 0.5])
    y = _logistic(t, 0.0, 1.0, 0.0, 1.0)
    assert y[0] == pytest.approx(0.1, abs=1e-12)
    assert y[1] == pytest.approx(0.5, abs=1e-12)
    assert y[2] == pytest.approx(0.9, abs=1e-12)
    assert TEN_TO_NINETY == pytest.approx(2.0 * math.log(9.0))


def test_a_noiseless_planted_width_is_recovered_to_a_tenth_of_a_percent():
    y, t = planted(40.0)
    fit = fit_transition(y, t, series="planted", criteria=FAST)
    assert not isinstance(fit, Refusal)
    assert fit.width == pytest.approx(40.0, rel=1e-3)
    assert fit.midpoint == pytest.approx(100.0, abs=1e-2)
    assert fit.onset_10 == pytest.approx(80.0, abs=1e-2)
    assert fit.onset_90 == pytest.approx(120.0, abs=1e-2)
    assert fit.direction == 1
    assert fit.usable


@pytest.mark.parametrize("width", [10.0, 25.0, 58.0, 120.0])
def test_planted_widths_are_recovered_across_a_decade(width):
    """One width recovered proves the optimiser found a minimum; four prove it found the right one."""
    y, t = planted(width, n=400, midpoint=200.0, sd=0.02, seed=7)
    fit = fit_transition(y, t, series="planted", criteria=FAST)
    assert not isinstance(fit, Refusal)
    assert fit.width == pytest.approx(width, rel=0.05)


def test_a_falling_transition_has_a_positive_width_and_a_negative_direction():
    y, t = planted(30.0)
    fit = fit_transition(1.0 - y, t, series="falling", criteria=FAST)
    assert not isinstance(fit, Refusal)
    assert fit.width == pytest.approx(30.0, rel=1e-3)
    assert fit.direction == -1
    assert fit.amplitude < 0


def test_the_width_estimator_is_close_to_unbiased_at_two_noise_levels():
    """Measured over 40 replicates at two noise levels, not asserted from theory.

    Forty replicates cannot resolve the *direction* of a one-percent drift, because the estimator's
    own spread at sd = 0.2 is 6.9 steps and the standard error of a forty-replicate mean is
    therefore 1.1 steps. So this asserts the magnitude only. Over 200 replicates the drift is
    upward at every noise level from 0.02 to 0.2, by +0.17 percent to +2.2 percent of the planted
    width, and upward is the safe direction: a width biased up makes every lead measured against it
    smaller, so the error is toward claiming less margin than there was. That measurement was made
    at a replicate count this test cannot carry, so it is reported rather than asserted here.
    """
    for sd, tolerance in ((0.02, 0.02), (0.2, 0.06)):
        widths = [
            fit.width
            for s in range(40)
            for fit in [
                fit_transition(
                    *planted(40.0, sd=sd, seed=1000 + s),
                    series="planted",
                    criteria=TransitionCriteria(n_boot=0),
                )
            ]
            if not isinstance(fit, Refusal)
        ]
        assert len(widths) == 40, "no replicate at this noise level should have refused"
        assert float(np.mean(widths)) == pytest.approx(40.0, rel=tolerance)


def test_the_width_interval_covers_the_planted_width_at_close_to_its_nominal_rate():
    """20 replicates at n = 200 and sd = 0.1. Nominal 0.95; this asserts at least 0.80.

    The assertion is loose because 20 replicates cannot resolve 0.95 from 0.85, and the number
    worth having is the one printed in the failure rather than a tight bound that flakes. A wider
    measurement is in the report.
    """
    hits = 0
    for s in range(20):
        y, t = planted(40.0, sd=0.1, seed=2000 + s)
        fit = fit_transition(y, t, series="planted", criteria=FAST, seed=s)
        assert not isinstance(fit, Refusal)
        lo, hi = fit.quality.width_ci
        hits += int(lo <= 40.0 <= hi)
    assert hits >= 16, f"only {hits}/20 intervals covered the planted width"


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


def test_pure_noise_refuses_below_lod_with_the_information_criterion_in_the_refusal():
    y = np.random.default_rng(0).normal(0.0, 1.0, 200)
    out = fit_transition(y, series="noise", criteria=FAST)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.BELOW_LOD
    assert out.statistics["delta_aicc_line"] < out.statistics["threshold"]
    assert "width of the noise" in out.detail
    assert "not comparable across runs" in out.remedy


def test_a_straight_line_refuses_because_a_line_is_not_a_transition():
    t = np.arange(200, dtype=float)
    y = 0.01 * t + np.random.default_rng(1).normal(0.0, 0.05, 200)
    out = fit_transition(y, t, series="trend", criteria=FAST)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.BELOW_LOD


def test_a_constant_series_refuses_record_incomplete_and_names_the_value():
    out = fit_transition([12.0] * 40, series="completions/mean_length", criteria=FAST)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert "every value is 12" in out.detail
    assert "series that moves" in out.remedy


def test_a_series_shorter_than_the_floor_refuses_rather_than_interpolating():
    y, t = planted(4.0, n=8, midpoint=4.0)
    out = fit_transition(y, t, series="short", criteria=FAST)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.RECORD_INCOMPLETE
    assert out.statistics["n"] == 8
    assert "unidentified" in out.remedy


def test_a_transition_wider_than_the_window_comes_back_as_a_bound():
    y, t = planted(250.0, n=200, midpoint=100.0, sd=0.01, seed=3)
    out = fit_transition(y, t, series="too_wide", criteria=FAST)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ABOVE_LOD_BELOW_LOQ
    assert out.is_bounded
    assert out.partial is not None
    assert out.statistics["width"] >= out.statistics["span"]
    assert "extrapolated rather than observed" in out.detail


def test_a_midpoint_outside_the_recorded_steps_comes_back_as_a_bound():
    """The rise starts at step 180 and its centre is 10 steps past the end of the record.

    Pushed further out the series stops looking like a rise at all and the information criterion
    refuses first, which is the right order: a transition nobody saw the start of is not a
    transition whose centre was extrapolated, it is a transition nobody saw.
    """
    y, t = planted(60.0, n=200, midpoint=210.0, sd=0.01, seed=4)
    out = fit_transition(y, t, series="off_the_end", criteria=FAST)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.ABOVE_LOD_BELOW_LOQ
    assert "outside the recorded range" in out.detail


def test_a_mismatched_step_axis_raises_rather_than_refusing():
    """A length mismatch is a programming error, not an anticipated data condition."""
    with pytest.raises(ValueError, match="same length"):
        fit_transition([1.0] * 20, list(range(19)))


# ---------------------------------------------------------------------------
# The unit
# ---------------------------------------------------------------------------


def _usable_fit() -> TransitionFit:
    y, t = planted(58.0, sd=0.02, seed=5)
    fit = fit_transition(y, t, series="planted", criteria=FAST)
    assert not isinstance(fit, Refusal)
    return fit


def test_a_lead_is_reported_in_widths_with_the_step_count_labelled():
    fit = _usable_fit()
    lead = lead_time(60.0, fit)
    assert lead.widths == pytest.approx((fit.midpoint - 60.0) / fit.width)
    assert lead.steps == pytest.approx(fit.midpoint - 60.0)
    # The two conventions differ by exactly half a width, which is the ambiguity H4 exists to kill.
    assert lead.widths - lead.widths_from_onset == pytest.approx(0.5)
    assert "not comparable across runs" in lead.render()


def test_the_specification_worked_example_reproduces():
    """The worked example prints "a fitted width of 58 steps. A 40-step lead is 0.69 of a window."

    The width is planted at 58 and the lead is planted at 40 steps before the fitted midpoint, so
    what this checks is the arithmetic of the unit and the fit's ability to land on 58.
    """
    fit = _usable_fit()
    assert fit.width == pytest.approx(58.0, rel=0.02)
    lead = lead_time(fit.midpoint - 40.0, fit)
    assert lead.widths == pytest.approx(0.69, abs=0.02)


def test_a_lead_below_the_sampling_resolution_says_so_rather_than_printing_a_fraction():
    fit = _usable_fit()
    fine = lead_time(fit.midpoint - 40.0, fit)
    assert fine.resolved
    coarse = lead_time(fit.midpoint - 0.2, fit)
    assert not coarse.resolved
    assert "sampling" in coarse.render()


def test_comparing_a_lead_in_widths_against_a_bare_step_count_refuses():
    fit = _usable_fit()
    a = lead_time(60.0, fit)
    out = compare_lead_times(a, 40.0)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.UNIT_MISMATCH
    assert "not the same width" in out.detail
    assert compare_lead_times(a, a) == pytest.approx(0.0)


def test_the_generated_invariance_check_for_the_units_group():
    """The `units` group's assertion is a refusal, so `check_invariance` routes it and says so."""
    report = check_invariance(
        TransitionWidth(None),
        "units",
        InvariancePayload(),
        relation=INVARIANT,
        run=lambda i, p: 1.0,
    )
    assert report.passed
    assert "refusal" in report.skipped
    fit = _usable_fit()
    assert check_unit_refusal(compare_lead_times, lead_time(60.0, fit), 40.0)


def test_a_fit_with_no_interval_is_still_usable_and_says_so_everywhere_it_prints():
    """Turning the bootstrap off costs the precision of the denominator, not its existence.

    The note has to survive into every rendering, because a width quoted without its interval is
    the failure this instrument exists to prevent and a note nobody prints is not a note.
    """
    y, t = planted(58.0, sd=0.02, seed=5)
    fit = fit_transition(y, t, series="planted", criteria=TransitionCriteria(n_boot=0))
    assert not isinstance(fit, Refusal)
    assert fit.usable
    assert all(math.isnan(v) for v in fit.quality.width_ci)
    assert "carries no interval" in fit.quality.render()
    assert "carries no interval" in fit.render()
    assert "carries no interval" in lead_time(60.0, fit).render()


def test_lead_time_refuses_to_divide_by_a_fit_that_was_never_usable():
    fit = _usable_fit()
    broken = TransitionFit(
        width=float("nan"),
        midpoint=fit.midpoint,
        amplitude=fit.amplitude,
        baseline=fit.baseline,
        direction=fit.direction,
        onset_10=fit.onset_10,
        onset_90=fit.onset_90,
        span=fit.span,
        cadence=fit.cadence,
        quality=fit.quality,
        series="hand-built",
        method="hand-built",
    )
    with pytest.raises(ValueError, match="not usable as a denominator"):
        lead_time(60.0, broken)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=15)
@given(
    scale=st.floats(min_value=0.2, max_value=5.0),
    shift=st.floats(min_value=-10.0, max_value=10.0),
)
def test_the_width_is_invariant_under_an_affine_rescaling_of_the_outcome(scale, shift):
    """A width is a property of the step axis, so rescaling what is being measured cannot move it.

    This is the strongest property available here and it is not the `units` group's assertion: that
    one is about comparing across a unit boundary and this one is about the reading itself. A width
    that moved when the outcome was rescaled would be reporting the outcome's units.
    """
    y, t = planted(45.0, sd=0.01, seed=9)
    a = fit_transition(y, t, criteria=TransitionCriteria(n_boot=0))
    b = fit_transition(scale * y + shift, t, criteria=TransitionCriteria(n_boot=0))
    assert not isinstance(a, Refusal) and not isinstance(b, Refusal)
    assert b.width == pytest.approx(a.width, rel=1e-3)
    assert b.midpoint == pytest.approx(a.midpoint, abs=1e-2)


@settings(deadline=None, max_examples=10)
@given(offset=st.integers(min_value=-500, max_value=500))
def test_shifting_the_step_axis_moves_the_midpoint_and_leaves_the_width_alone(offset):
    y, t = planted(45.0, sd=0.01, seed=9)
    a = fit_transition(y, t, criteria=TransitionCriteria(n_boot=0))
    b = fit_transition(y, t + offset, criteria=TransitionCriteria(n_boot=0))
    assert not isinstance(a, Refusal) and not isinstance(b, Refusal)
    assert b.width == pytest.approx(a.width, rel=1e-3)
    assert b.midpoint == pytest.approx(a.midpoint + offset, abs=1e-2)


# ---------------------------------------------------------------------------
# The instrument's own declarations
# ---------------------------------------------------------------------------


def test_the_instrument_passes_the_section_four_two_lint():
    assert lint_instrument(TransitionWidth(None)) == []


def test_the_envelope_downgrades_rather_than_refusing():
    """Both conditions are confounders for a fitted width and neither makes the fit unreal.

    An envelope that refused here would withhold the fit on exactly the records where a reader most
    needs to see it and then decide, which is E29's failure in the other direction.
    """
    envelope = TransitionWidth.envelope
    assert envelope.on_violation == "downgrade"
    assert {c.name for c in envelope.requires} == {"STATIONARY_GRADER", "EXOGENOUS_CURRICULUM"}


def test_a_lead_time_round_trips_through_the_evidence_codec():
    from reward_lens.core.evidence import ValueCodec

    lead = lead_time(60.0, _usable_fit())
    codec = ValueCodec()
    back = codec.decode(codec.encode(lead))
    assert isinstance(back, LeadTime)
    assert back.widths == pytest.approx(lead.widths)
    assert back.fit.width == pytest.approx(lead.fit.width)

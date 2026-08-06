"""Unit tests for `organisms/dose.py`: the Hill fit's intervals and the therapeutic index.

The point estimate is M1's (`measure/meta/floor.fit_hill`) and is tested there. What is tested here
is what this module adds: bootstrap intervals on all three parameters, the two-curve therapeutic
index, and the refusals that stop a curve being reported where there is not one.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.organisms.dose import (
    MIN_DOSES_FOR_HILL,
    DosePoint,
    dose_response,
    fit_curve,
    sweep,
)


def _hill(c: float, e_max: float, ec50: float, n: float) -> float:
    return 0.0 if c <= 0 else e_max * c**n / (ec50**n + c**n)


def _points(e_max=1.6, ec50=0.7, n=2.0, cost_ec50=1.8, cost_n=1.0, doses=None):
    doses = doses or (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    return [
        DosePoint(dose=c, effect=_hill(c, e_max, ec50, n), cost=_hill(c, 0.9, cost_ec50, cost_n))
        for c in doses
    ]


class TestTheFitRecoversAPlantedCurve:
    def test_all_three_parameters_come_back(self) -> None:
        curve = fit_curve(_points(), n_boot=100, seed=0)
        assert not isinstance(curve, Refusal)
        assert curve.e_max == pytest.approx(1.6, abs=1e-4)
        assert curve.ec50 == pytest.approx(0.7, abs=1e-4)
        assert curve.hill_n == pytest.approx(2.0, abs=1e-4)
        assert curve.residual_rms < 1e-9

    def test_the_intervals_bracket_the_truth(self) -> None:
        curve = fit_curve(_points(), n_boot=200, seed=1)
        assert curve.ec50_low <= 0.7 <= curve.ec50_high
        assert curve.hill_n_low <= 2.0 <= curve.hill_n_high

    def test_cooperativity_above_one_is_called_switch_like_only_on_the_interval(self) -> None:
        """A point estimate of n > 1 is not the claim; the lower endpoint is."""
        switch = fit_curve(_points(n=4.0), n_boot=200, seed=2)
        assert switch.is_switch_like
        ramp = fit_curve(_points(n=1.0), n_boot=200, seed=2)
        assert not ramp.is_switch_like

    def test_dose_for_inverts_the_curve_and_ed50_is_ec50(self) -> None:
        curve = fit_curve(_points(), n_boot=50, seed=0)
        assert curve.dose_for(0.5) == pytest.approx(curve.ec50, rel=1e-9)
        assert curve.dose_for(1.0) == np.inf
        assert curve.dose_for(0.0) == 0.0
        # 90% of E_max on an n=2 curve at EC50 0.7 is 0.7*sqrt(9) = 2.1
        assert curve.dose_for(0.9) == pytest.approx(2.1, rel=1e-3)

    def test_the_kernel_calibration_curve_is_what_comes_out(self) -> None:
        curve = fit_curve(_points(), n_boot=50, seed=0)
        kernel = curve.curve()
        assert kernel.response(0.7) == pytest.approx(0.8, abs=1e-4)  # half of E_max 1.6
        assert kernel.slope(0.7) > 0


class TestTheTherapeuticIndex:
    def test_ti_is_the_ratio_of_the_two_half_maximal_doses(self) -> None:
        out = dose_response(_points(ec50=0.4, cost_ec50=0.9), n_boot=100, seed=0)
        assert not isinstance(out, Refusal)
        assert out.therapeutic_index == pytest.approx(0.9 / 0.4, rel=1e-3)
        assert out.has_a_window

    def test_no_window_when_the_cost_arrives_first(self) -> None:
        out = dose_response(_points(ec50=1.5, cost_ec50=0.3), n_boot=100, seed=0)
        assert out.therapeutic_index is not None and out.therapeutic_index < 1.0
        assert not out.has_a_window
        assert "no usable dose window" in out.render()

    def test_a_sweep_with_no_cost_column_has_no_index_and_says_so(self) -> None:
        points = [DosePoint(dose=p.dose, effect=p.effect) for p in _points()]
        out = dose_response(points, n_boot=50, seed=0)
        assert out.therapeutic_index is None
        assert out.cost is None
        assert "cannot give a therapeutic index" in out.cost_refusal

    def test_the_index_agrees_with_the_kernel_at_the_default_levels(self) -> None:
        """`CalibrationCurve.therapeutic_index` is the one place the ratio lives."""
        out = dose_response(_points(ec50=0.4, cost_ec50=0.9), n_boot=50, seed=0)
        assert out.therapeutic_index == pytest.approx(
            out.effect.curve().therapeutic_index(out.td50), rel=1e-12
        )


class TestItRefusesRatherThanFitting:
    def test_too_few_dose_levels(self) -> None:
        points = [DosePoint(dose=c, effect=_hill(c, 1.0, 0.5, 1.0)) for c in (0.0, 0.5, 1.0)]
        out = fit_curve(points, n_boot=10)
        assert isinstance(out, Refusal)
        assert out.reason is RefusalReason.RECORD_INCOMPLETE
        assert str(MIN_DOSES_FOR_HILL) in out.remedy

    def test_a_flat_response_refuses_rather_than_fitting_a_vanishing_curve(self) -> None:
        """An identically-zero column fits `E_max = 1.4e-9` and converges, so it needs its own guard.

        M1's `e_max <= 0` check does not catch it because the optimiser lands on a tiny *positive*
        maximum. That is adequate for M1, which divides by the slope and gets an enormous floor;
        here the curve is the reading and three parameters describing nothing is the confident
        wrong number a refusal exists to prevent.
        """
        points = [DosePoint(dose=c, effect=0.0) for c in (0.0, 0.3, 0.6, 1.0, 1.5)]
        out = fit_curve(points, n_boot=10)
        assert isinstance(out, Refusal)
        assert out.reason is RefusalReason.BELOW_LOD
        assert out.statistics["dynamic_range"] == 0.0

    def test_a_response_that_falls_refuses_and_names_the_sign_convention(self) -> None:
        points = [
            DosePoint(dose=c, effect=-_hill(c, 1.0, 0.5, 1.0)) for c in (0.0, 0.3, 0.6, 1.0, 1.5)
        ]
        out = fit_curve(points, n_boot=10)
        assert isinstance(out, Refusal)
        assert out.reason is RefusalReason.BELOW_LOD
        assert out.statistics["monotone"] is False
        # The remedy names the two ways a falling column happens: a signed delta passed where a
        # magnitude was wanted, and a response that turns over once the host is damaged.
        assert "magnitudes that increase" in out.remedy
        assert "turning point" in out.remedy

    def test_an_empty_sweep(self) -> None:
        out = dose_response([])
        assert isinstance(out, Refusal)
        assert out.reason is RefusalReason.RECORD_INCOMPLETE

    def test_a_negative_dose_is_a_programming_error_not_a_refusal(self) -> None:
        with pytest.raises(ValueError, match="magnitude"):
            DosePoint(dose=-0.1, effect=0.5)


class TestSweep:
    def test_the_loop_records_the_dose_each_point_was_taken_at(self) -> None:
        seen: list[float] = []

        def run(c: float) -> tuple[float, float | None]:
            seen.append(c)
            return _hill(c, 1.0, 0.5, 1.0), 0.1 * c

        points = sweep([0.0, 0.25, 0.5, 1.0], run)
        assert seen == [0.0, 0.25, 0.5, 1.0]
        assert [p.dose for p in points] == seen
        assert all(p.cost is not None for p in points)

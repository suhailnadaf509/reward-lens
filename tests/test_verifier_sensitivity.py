"""Unit tests for D4, `verifier.sobol_ST`.

Four things that are easy to get wrong and expensive to notice late. The sample count is
`N·(2D+2)` and `N` is the base size rather than the budget, so passing a budget overshoots by a
factor of eight on three inputs. `N` must be a power of two, and SALib warns and then returns
numbers anyway. The bootstrap interval SALib returns is `Z·s` rather than `s`, so composing it into
a GUM budget without dividing by `Z` overstates the standard uncertainty by about two. And the
dose-response slope this instrument hands to the contract layer has to be the real slope, which is
checkable against a rubric whose coefficients are known.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from reward_lens.core.invariance import INVARIANT, InvariancePayload, check_invariance
from reward_lens.core.reading import Refusal
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.verifier.sensitivity import (
    RubricInput,
    SobolSensitivity,
    dose_response_slopes,
    one_at_a_time,
    sobol_indices,
    sobol_problem,
    sobol_sample,
    total_effect,
)

INPUTS = [
    RubricInput("a", 0.0, 1.0),
    RubricInput("b", 0.0, 1.0),
    RubricInput("inert", 0.0, 1.0),
]


def linear(criteria) -> float:
    """`3a + 2b`, with `inert` multiplied by exactly 0.0. Analytic S1: 9/13 and 4/13."""
    return 3.0 * criteria["a"] + 2.0 * criteria["b"] + 0.0 * criteria["inert"]


# ---------------------------------------------------------------------------
# The dependency, pinned by file rather than by name
# ---------------------------------------------------------------------------


def test_salib_is_the_package_we_think_it_is_and_reproduces_its_reference_values():
    """The dependency check, plus a re-verification of the two claims D4 leans on.

    `SALib` has no `__version__`, so the version comes from the distribution metadata. The
    reference values are analytic: for `3a + 2b` the first-order indices are `9/13 = 0.6923` and
    `4/13 = 0.3077`, and the third input's are bit-exact zero.
    """
    import importlib.metadata

    import SALib
    import SALib.analyze.sobol
    import SALib.sample.sobol

    assert SALib.__file__.endswith("SALib/__init__.py")
    assert importlib.metadata.version("SALib") >= "1.5"

    problem = {"num_vars": 3, "names": ["x1", "x2", "x3"], "bounds": [[0, 1]] * 3}
    x = SALib.sample.sobol.sample(problem, 1024, calc_second_order=True, seed=0)
    assert x.shape == (8192, 3), "N·(2D+2) = 1024 · 8"
    y = 3.0 * x[:, 0] + 2.0 * x[:, 1] + 0.0 * x[:, 2]
    si = SALib.analyze.sobol.analyze(problem, y, seed=0)

    assert si["S1"][0] == pytest.approx(9 / 13, abs=1e-3)
    assert si["S1"][1] == pytest.approx(4 / 13, abs=1e-3)
    assert si["ST"][2] == 0.0
    assert si["S1"][2] == 0.0


def test_a_power_of_two_base_sample_emits_no_warning_and_a_smaller_one_never_reaches_salib():
    """SALib's `UserWarning` is easy to lose under a filter, so this refuses before the call."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sobol_sample(INPUTS, 128, seed=0)
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []

    with pytest.raises(ValueError, match=r"N=1000 .*power of two"):
        sobol_sample(INPUTS, 1000, seed=0)
    with pytest.raises(ValueError, match="1024"):
        sobol_sample(INPUTS, 1000, seed=0)


def test_the_sample_count_is_the_base_size_and_not_the_budget():
    assert sobol_sample(INPUTS, 64, calc_second_order=True).shape == (64 * 8, 3)
    assert sobol_sample(INPUTS, 64, calc_second_order=False).shape == (64 * 5, 3)
    assert sobol_problem(INPUTS)["names"] == ["a", "b", "inert"]


# ---------------------------------------------------------------------------
# The three rungs
# ---------------------------------------------------------------------------


def test_rung_zero_is_one_at_a_time_and_is_exact_on_a_linear_rubric():
    """With no interaction there is nothing for one-at-a-time to miss, so it agrees with rung 2.

    Which is the honest statement of what rung 0 costs: it is not always wrong, it is wrong exactly
    when interaction exists, and it cannot tell you which case you are in. That is why it is
    reported next to rung 2 rather than instead of it.
    """
    effect, share = one_at_a_time(linear, INPUTS)
    assert effect.tolist() == [3.0, 2.0, 0.0]
    assert share == pytest.approx([3 / 5, 2 / 5, 0.0])

    profile = SobolSensitivity(linear, INPUTS, n_base=256, seed=0).estimate().value
    assert profile.rank_inversions == 0
    assert abs(profile.interaction_mass) < 0.01


def test_the_dose_response_slope_recovers_the_rubric_coefficients():
    """`μ'_i` for the contract layer. On a linear rubric the answer is known exactly."""
    x = sobol_sample(INPUTS, 128, seed=0)
    y = 3.0 * x[:, 0] + 2.0 * x[:, 1]
    slope, stderr = dose_response_slopes(x, y)
    assert slope == pytest.approx([3.0, 2.0, 0.0], abs=1e-8)
    assert np.all(stderr < 1e-6)


def test_the_profile_hands_the_contract_layer_a_structure_rather_than_a_verdict():
    profile = SobolSensitivity(linear, INPUTS, n_base=256, seed=0).estimate().value
    contract = profile.contract_inputs()
    assert [c.name for c in contract] == ["a", "b", "inert"]
    assert profile.mu_prime() == {
        c.name: pytest.approx(m, abs=1e-6) for c, m in zip(contract, [3.0, 2.0, 0.0])
    }
    assert contract[0].total_effect > contract[1].total_effect > 0.0
    assert contract[2].total_effect == 0.0
    assert contract[0].input_range == (0.0, 1.0)
    # No ratio, no ranking, no verdict: the equal-compensation check is N6's and is not built here.
    assert not hasattr(profile, "equal_compensation")


def test_the_bootstrap_interval_composes_as_a_gum_budget_after_dividing_out_z():
    """SALib returns `ST_conf = Z·s`. A budget wants `s`, and `Z ≈ 1.96` at the 95% level."""
    from scipy.stats import norm

    profile = (
        SobolSensitivity(linear, INPUTS, n_base=256, num_resamples=100, conf_level=0.95, seed=0)
        .estimate()
        .value
    )
    budget = profile.budget()
    term = {t.name: t for t in budget.terms}["sobol_bootstrap"]

    z = float(norm.ppf(0.5 + 0.95 / 2))
    assert term.value == pytest.approx(profile.dominant.st_conf / z)
    assert term.kind == "A"
    assert term.dof == 99.0
    assert budget.expanded == pytest.approx(2.0 * budget.combined)


def test_a_second_budget_term_appears_only_when_a_negative_index_makes_it_real():
    """A term of zero and an unestimated term look the same on a table and are not the same thing."""
    profile = SobolSensitivity(linear, INPUTS, n_base=256, seed=0).estimate().value
    names = {t.name for t in profile.budget().terms}
    if min(i.st for i in profile.indices) < 0:
        assert "estimator_noise_floor" in names
    else:
        assert names == {"sobol_bootstrap"}


# ---------------------------------------------------------------------------
# The generated invariance test, and it is not vacuous
# ---------------------------------------------------------------------------


def test_the_indices_do_not_move_under_an_affine_rescaling_of_the_reward():
    """`verifier.sobol_ST` is declared under `reward.affine` and the check has real content.

    A variance ratio is invariant under `r -> a·r + b` because numerator and denominator both scale
    by `a²` and the shift cancels in every difference the estimators take. SALib also centres and
    scales the output before estimating, which makes it exact rather than merely asymptotic.
    """
    x = sobol_sample(INPUTS, 256, seed=0)
    y = np.array([linear(dict(zip(["a", "b", "inert"], row))) for row in x])

    report = check_invariance(
        SobolSensitivity(linear, INPUTS, n_base=256),
        "reward.affine",
        InvariancePayload(scores=y),
        n=24,
        relation=INVARIANT,
        run=lambda inst, payload: float(total_effect(INPUTS, np.asarray(payload.scores))[0]),
    )
    assert report.passed, report.render()
    assert report.n == 24
    assert report.draws, "not vacuous: 24 group elements were drawn and applied"
    assert report.max_deviation < report.tol


# ---------------------------------------------------------------------------
# Refusals and declarations
# ---------------------------------------------------------------------------


def test_a_constant_grader_is_refused_rather_than_producing_nan_indices():
    reading = SobolSensitivity(lambda c: 0.7, INPUTS, n_base=32).estimate()
    assert isinstance(reading, Refusal)
    assert reading.reason.name == "BELOW_LOD"
    assert reading.statistics["constant_score"] == 0.7
    assert "silent_zero_rate" in reading.remedy


def test_a_degenerate_input_range_is_refused_at_construction():
    with pytest.raises(ValueError, match="empty or inverted"):
        RubricInput("bad", 1.0, 1.0)


def test_no_inputs_is_a_construction_error_rather_than_an_empty_profile():
    with pytest.raises(ValueError, match="at least one input"):
        SobolSensitivity(linear, [], n_base=64)


def test_the_instrument_declares_the_twelve_and_passes_lint():
    inst = SobolSensitivity(linear, INPUTS, n_base=64)
    assert lint_instrument(inst) == []
    assert inst.quantity == "verifier.sobol_ST"
    assert inst.invariance == "reward.affine"
    assert inst.rung == 2
    assert set(inst.baselines) == {"equal_weighting", "one_at_a_time"}
    assert inst.envelope is not None and inst.envelope.requires


def test_preflight_costs_the_run_without_evaluating_the_rubric():
    calls = {"n": 0}

    def counted(criteria):
        calls["n"] += 1
        return criteria["a"]

    inst = SobolSensitivity(counted, INPUTS, n_base=1024)
    plan = inst.preflight(Context(signal=inst.subject))
    assert plan.ok
    assert calls["n"] == 0
    assert inst.evaluations == 1024 * 8
    assert plan.cost.calls == 1024 * 8 + 6


def test_second_order_indices_are_returned_and_keyed_by_pair():
    def interacting(criteria):
        return criteria["a"] + 4.0 * criteria["a"] * criteria["b"] + 0.0 * criteria["inert"]

    profile = SobolSensitivity(interacting, INPUTS, n_base=256, seed=0).estimate().value
    assert "a|b" in profile.second_order
    assert profile.second_order["a|b"] > 0.05
    assert profile.interaction_mass > 0.05


def test_the_analyzer_and_the_sampler_agree_on_the_second_order_flag():
    x = sobol_sample(INPUTS, 64, calc_second_order=False, seed=0)
    y = 3.0 * x[:, 0] + 2.0 * x[:, 1]
    si = sobol_indices(INPUTS, y, calc_second_order=False, num_resamples=10, seed=0)
    assert "S2" not in si
    assert si["ST"][2] == 0.0

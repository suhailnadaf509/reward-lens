"""Acceptance: the F5 variance spike, and what its verdict made of the instrument.

The clause: *"F5, spike first. Measure the score-function estimator's variance at
`K in {4, 8, 16, 64}` before building anything on it. Accept: a variance-versus-K curve with the
variance-reduction techniques ablated, and an explicit go or no-go recorded as a design note. If
no-go, F5 reduces to the differentiable-surrogate case and the plan changes here."*

**The spike returned no-go**, so this file asserts the surrogate path and the recorded decision
rather than the covector, which is what the clause asks for in that branch. Specifically: that the
curve exists at all four group sizes with the reduction ladder ablated, that the decision is
recorded against thresholds fixed before the numbers were produced, that P8's resolution is
registered through `studies.freeze`, that the score-function rungs refuse below the measured floor
with the floor in the remedy, and that rung 3 runs on the model that wrote the 200-step record and
returns Evidence carrying the `IncrementalValidity` lint rule four makes mandatory.

The real-subject half needs `transformers` and a CPU forward on
`trl-internal-testing/tiny-Qwen3ForCausalLM`, and skips without them. The recorded-decision half
needs neither and always runs, because a spike result that could be skipped is not a record.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import is_white_box, lint_instrument, lint_reading
from reward_lens.measure.frontier.covector import (
    F5,
    P8_COSINE_THRESHOLD,
    P8_RESOLUTION,
    P8_RSE_THRESHOLD,
    POOLED_N_FLOOR,
    SelectionCovector,
    SelectionDimensionality,
    SelectionSpectrum,
    SelectionStableRank,
    p8_resolution,
    p8_study,
)

MODEL_ID = "trl-internal-testing/tiny-Qwen3ForCausalLM"
K_GRID = ("4", "8", "16", "64")


# ---------------------------------------------------------------------------
# the clause's first half: the curve, the ablations, and the recorded decision
# ---------------------------------------------------------------------------


def test_the_curve_exists_at_every_registered_group_size():
    """`K in {4, 8, 16, 64}`, for every arm of the ablation ladder."""
    for arm, curve in P8_RESOLUTION["rse_by_k"].items():
        assert tuple(sorted(curve, key=int)) == K_GRID, f"{arm} does not cover the grid"
        assert all(np.isfinite(curve[k]) and curve[k] > 0 for k in K_GRID), arm


def test_the_three_named_reduction_techniques_were_ablated():
    """Three techniques are named, and all three are separately on the curve.

    Group-mean baselines, antithetic sampling, and control variates. The ladder runs no-baseline,
    group mean, leave-one-out, leave-one-out plus control variate, and antithetic plus everything,
    so each technique can be read on and off.
    """
    arms = P8_RESOLUTION["rse_by_k"]
    assert "no_baseline" in arms
    assert "group_mean_baseline" in arms
    assert "leave_one_out_baseline" in arms
    assert "loo_plus_control_variate" in arms
    assert "antithetic_plus_all" in arms
    assert "differentiable_surrogate" in arms


def test_the_decision_is_explicit_and_it_is_a_no_go():
    assert P8_RESOLUTION["verdict"] == "no-go"
    assert P8_RESOLUTION["go"] is False
    assert P8_RESOLUTION["h1_confirmed"] is False
    assert P8_RESOLUTION["h2_confirmed"] is False


def test_the_decision_is_against_thresholds_and_not_against_the_numbers():
    """Both registered thresholds were fixed before the spike ran, and both are missed."""
    assert P8_RESOLUTION["rse_k64"] >= P8_RSE_THRESHOLD
    assert P8_RESOLUTION["cosine_k64"] <= P8_COSINE_THRESHOLD


def test_the_no_go_beat_its_registered_comparator_in_the_direction_that_matters():
    """P8's must-beat column is the differentiable surrogate, and the surrogate wins.

    That is the resolution: the comparator the prediction named is less noisy by a factor of 51 at
    the same group size on the same subject, and its direction reproduces at cosine 0.9998 against
    0.145. A prediction that named a comparator and lost to it is refuted.
    """
    assert P8_RESOLUTION["surrogate_rse_k64"] < P8_RESOLUTION["rse_k64"]
    assert P8_RESOLUTION["surrogate_cosine_k64"] > P8_RESOLUTION["cosine_k64"]
    assert P8_RESOLUTION["noise_ratio_score_function_over_surrogate"] > 50.0


def test_p8_is_registered_through_studies_freeze():
    """The prediction was frozen at c746e9f before this instrument existed; this is the record."""
    frozen = p8_study()
    assert frozen.study_id.startswith("study:w52-f5-variance-spike@v")
    assert frozen.spec_hash
    assert frozen.git_sha and frozen.git_sha != "unknown"
    assert {h.id for h in frozen.spec.hypotheses} == {"H1", "H2"}
    thresholds = {h.prediction.metric: h.prediction.threshold for h in frozen.spec.hypotheses}
    assert thresholds["covector_relative_standard_error_k64"] == P8_RSE_THRESHOLD
    assert thresholds["covector_split_half_cosine_k64"] == P8_COSINE_THRESHOLD


def test_the_registered_analysis_scores_the_registered_predictions():
    result = p8_resolution()
    spec = p8_study().spec
    assert result.killed is True
    assert result.kill_outcomes["K1"] == "fired"
    for hypothesis in spec.hypotheses:
        assert result.outcomes[hypothesis.id] == "refuted"
        assert hypothesis.prediction.check(result.metrics[hypothesis.prediction.metric]) is False


def test_the_plan_change_is_in_the_instrument_and_not_only_in_the_note():
    """ "F5 reduces to the differentiable-surrogate case" as a default, not as a sentence."""
    for cls in F5:
        assert cls.rung == 3


# ---------------------------------------------------------------------------
# declarations, and the four rules lint enforces
# ---------------------------------------------------------------------------


def test_every_instrument_passes_lint():
    for cls in F5:
        assert lint_instrument(cls(rewards=[1.0, 2.0])) == []


def test_the_instruments_are_white_box_so_incremental_validity_is_mandatory():
    for cls in F5:
        assert is_white_box(cls(rewards=[1.0, 2.0]))


def test_stable_rank_and_participation_ratio_are_reported_and_numerical_rank_is_not():
    """Numerical rank is the wrong statistic and nobody should have reported it."""
    from reward_lens.measure.frontier.covector import SelectionGeometry

    fields = set(SelectionGeometry.__dataclass_fields__)
    assert "stable_rank" in fields
    assert "participation_ratio" in fields
    assert "pr_convention" in fields
    assert not any("numerical_rank" in name or name == "rank" for name in fields)


def test_the_four_quantities_are_the_catalogue_record_for_f5():
    assert SelectionDimensionality.quantity == "selection.dimensionality"
    assert SelectionSpectrum.quantity == "selection.spectrum"
    assert SelectionStableRank.quantity == "selection.stable_rank"
    assert SelectionCovector.quantity == "selection.covector"


# ---------------------------------------------------------------------------
# the real subject
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def policy():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from reward_lens.policy.hf import wrap_hf_policy

    torch.set_num_threads(2)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as exc:  # no network and no cache
        pytest.skip(f"{MODEL_ID} is not available: {exc}")
    return wrap_hf_policy(model, tokenizer, contrast=(" yes", " no"), device="cpu")


@pytest.fixture(scope="module")
def rollouts(policy):
    """Real completions from the model that wrote the 200-step record, scored by its own grader.

    The grader is the fixture's `len(text) / 50`, which is the one thing about this subject that
    makes it a real optimisation target rather than a shaped input.
    """
    from reward_lens.policy.base import SampleSpec

    prompts = [f"question {i}: count upward from {i}" for i in range(4)]
    drawn = policy.sample(
        prompts, SampleSpec(max_new_tokens=8, temperature=1.0, group_size=6, seed=3)
    ).value
    items, rewards, groups, texts = [], [], [], []
    for index, prompt in enumerate(prompts):
        for completion in drawn.texts[index]:
            items.append((prompt, completion))
            rewards.append(len(completion) / 50.0)
            groups.append(index)
            texts.append(completion)
    return items, np.asarray(rewards), np.asarray(groups), texts


def _context(policy, items):
    from reward_lens.core.envelope import ConditionReading, RegimeCondition, RegimeReading
    from reward_lens.core.types import Access, Component, Phase, Substrate
    from reward_lens.measure.base import Context

    regime = RegimeReading(
        conditions={
            RegimeCondition.LINEAR_RESPONSE: ConditionReading(
                RegimeCondition.LINEAR_RESPONSE, True, 0.62, 0.30
            ),
            RegimeCondition.ABOVE_LOD: ConditionReading(
                RegimeCondition.ABOVE_LOD, True, 0.90, 0.10
            ),
        }
    )
    return Context(
        signal=policy,
        view=items,
        readout="decision",
        access={Component.POLICY: Access.BACKWARD, Component.GRADER: Access.QUERY},
        substrate=Substrate.NEURAL_GEN,
        phase=Phase.POST_RUN,
        regime_reading=regime,
    )


def test_the_surrogate_rung_runs_on_the_model_that_wrote_the_record(policy, rollouts):
    """Rung 3 end to end: Evidence, with the spectrum and both dimensionality statistics on it."""
    items, rewards, groups, texts = rollouts
    instrument = SelectionStableRank(rewards=rewards, groups=groups, texts=texts, rung=3)
    reading = instrument.estimate(_context(policy, items))
    assert not isinstance(reading, Refusal), getattr(reading, "render", lambda: reading)()

    value = reading.value
    assert value.rung == 3
    assert "surrogate" in value.estimator
    assert value.d_model == policy.meta.d_model
    assert value.n_items == len(items)
    assert 1.0 <= value.stable_rank <= value.d_model
    assert 1.0 <= value.participation_ratio <= value.d_model
    assert value.pr_convention.startswith("moment_ratio")
    assert len(value.eigenvalues) == value.d_model
    assert value.eigenvalues == sorted(value.eigenvalues, reverse=True)
    assert 0.0 <= value.top_share <= 1.0
    assert len(value.whitened_eigenvalues) == value.d_model
    assert value.whitened_ridge > 0.0


def test_the_reading_carries_the_incremental_validity_section_6_4_requires(policy, rollouts):
    """Lint rule four, on a real reading: a white-box reading with no record fails."""
    items, rewards, groups, texts = rollouts
    instrument = SelectionStableRank(rewards=rewards, groups=groups, texts=texts, rung=3)
    reading = instrument.estimate(_context(policy, items))
    assert reading.incremental is not None
    assert lint_reading(reading, instrument) == []
    assert reading.quantity == "selection.stable_rank"
    assert np.isfinite(reading.incremental.error_correlation)


def test_all_four_mandatory_baselines_are_run_or_named_as_refused(policy, rollouts):
    """A control that could not run is reported by name, never quietly dropped."""
    items, rewards, groups, texts = rollouts
    instrument = SelectionStableRank(rewards=rewards, groups=groups, texts=texts, rung=3)
    value = instrument.estimate(_context(policy, items)).value
    accounted = set(value.baselines) | set(value.baseline_refusals)
    for baseline in instrument.baselines:
        assert baseline in accounted, f"{baseline} neither scored nor refused"
    assert "selection.top_direction" in value.baselines


def test_the_apparatus_is_recorded_on_the_reading(policy, rollouts):
    """`nnsight` patches `torch.Tensor.backward` at import and no name-based check sees it."""
    items, rewards, groups, texts = rollouts
    instrument = SelectionStableRank(rewards=rewards, groups=groups, texts=texts, rung=3)
    value = instrument.estimate(_context(policy, items)).value
    assert "nnsight_imported" in value.apparatus
    assert value.apparatus["backward_patched_by_name"] is False
    assert "sys.modules" in value.apparatus["note"]


def test_the_score_function_rung_refuses_below_the_measured_floor(policy, rollouts):
    """The no-go, enforced: rung 0 on 24 pooled rollouts is a refusal with the number in it."""
    items, rewards, groups, texts = rollouts
    instrument = SelectionCovector(rewards=rewards, groups=groups, texts=texts, rung=0)
    reading = instrument.estimate(_context(policy, items))
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.BELOW_LOD
    assert str(POOLED_N_FLOOR) in reading.remedy
    assert "differentiable surrogate" in reading.remedy
    assert reading.statistics["n_pooled"] == len(items)
    assert reading.statistics["floor"] == POOLED_N_FLOOR


def test_the_refusal_carries_a_remedy_that_is_an_instruction(policy, rollouts):
    """A remedy has to be actionable: "envelope violated" is not, and "pool at least 600 rollouts" is."""
    items, rewards, groups, texts = rollouts
    reading = SelectionCovector(rewards=rewards, groups=groups, texts=texts, rung=0).estimate(
        _context(policy, items)
    )
    assert isinstance(reading, Refusal)
    assert "pool at least" in reading.remedy
    # and it says what pooling costs, rather than only that it is available
    assert "assumes" in reading.remedy


def test_a_misaligned_reward_array_refuses_rather_than_measuring_something(policy, rollouts):
    items, rewards, _groups, texts = rollouts
    reading = SelectionStableRank(rewards=rewards[:-1], texts=texts, rung=3).estimate(
        _context(policy, items)
    )
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.RECORD_INCOMPLETE


def test_the_says_string_carries_this_reading_s_own_numbers(policy, rollouts):
    """The catalogue's `says` cell, filled from the reading rather than from the document."""
    items, rewards, groups, texts = rollouts
    value = (
        SelectionStableRank(rewards=rewards, groups=groups, texts=texts, rung=3)
        .estimate(_context(policy, items))
        .value
    )
    assert f"{value.stable_rank:.2f}" in value.says
    assert "stable rank" in value.says
    assert "2.69 at K = 4" in value.says  # the small-sample bias travels with the number

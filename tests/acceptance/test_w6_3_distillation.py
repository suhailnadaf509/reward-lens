"""K1: the distillation gap, on a planted subject where the answer is known.

The clause for a compute-gated package is not "the phenomenon is real"; nobody here can
buy the checkpoints. It is that the arithmetic recovers a survival fraction that was planted, that
the instrument lints, that its generated invariance test passes, that every refusal it can produce
is reachable and carries a remedy, and that the frozen study, the price and the runbook exist and
agree with each other.

The lint calls are the point of the file rather than a formality. E56: four instruments shipped
failing lint rule 1 while their package read `done`, because their acceptance test rendered
readings and never linted. Every instrument this package ships is linted here, and so
is every reading it emits.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.invariance import InvariancePayload, check_invariance
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import TrustLevel
from reward_lens.measure.base import lint_instrument, lint_reading
from reward_lens.measure.ledger.features import RecordedFeatures, SurfaceFeatures
from reward_lens.record.schema import make_trajectory
from reward_lens.record.turns import Turn
from reward_lens.stats.baselines import ALL_SIX, BaselineBank, lint_claim
from reward_lens.studies import render_report, run_study
from studies.w6_distillation.analysis import build_spec, frozen_study
from studies.w6_distillation.fit import ShiftDesign, corrected_slope, raw_slope, reliability
from studies.w6_distillation.instrument import DistillationGap
from studies.w6_distillation.planted import (
    HACK_MARKER,
    PLANTED_HACK_FEATURES,
    PLANTED_NAMES,
    PlantSpec,
    expected_contrast_pp,
    expected_pooled_survival,
    expected_region_contrast_pp,
    plant,
)
from studies.w6_distillation.price import price, reference_multi_seed_gpu_hours
from studies.w6_distillation.reading import distillation_survival
from studies.w6_distillation.regions import RegionFeatures, region_of, split_region
from studies.w6_distillation.runbook import render_runbook
from studies.w6_distillation.survival import Arm, shared_prompts


def _featuriser() -> RecordedFeatures:
    return RecordedFeatures(names=PLANTED_NAMES)


def _read(spec: PlantSpec, *, seed: int = 7, n_blanks: int = 3):
    arms = plant(spec, n_blanks=n_blanks)
    return arms, distillation_survival(
        arms["base"],
        arms["expert"],
        arms["student"],
        blanks=[arms[f"blank{i}"] for i in range(n_blanks)],
        featuriser=_featuriser(),
        hack_features=PLANTED_HACK_FEATURES,
        markers=(HACK_MARKER,),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# The clause: the arithmetic recovers what was planted
# ---------------------------------------------------------------------------


def test_recovers_the_planted_survival_fraction() -> None:
    """The corrected fit's interval covers the survival fraction the plant put there.

    Against the plant's own arithmetic rather than against a number written into the test:
    `expected_pooled_survival` computes `sum(d^2 s) / sum(d^2)`, which is what a fit that weights
    each feature by how much there was to lose recovers in expectation.
    """
    _, reading = _read(PlantSpec(group_size=8))
    assert not isinstance(reading, Refusal), getattr(reading, "detail", "")
    expected = 100.0 * expected_pooled_survival()
    assert reading.survival_ci_low_pp <= expected <= reading.survival_ci_high_pp, (
        f"planted {expected:.2f} outside [{reading.survival_ci_low_pp:.2f}, "
        f"{reading.survival_ci_high_pp:.2f}]"
    )
    assert reading.delta_pp == pytest.approx(100.0 - reading.survival_pp)
    assert reading.n_features_fitted == len(PLANTED_NAMES)


def test_recovers_the_planted_hack_versus_capability_contrast() -> None:
    """K1's headline: the two feature families survive at different, recoverable rates."""
    _, reading = _read(PlantSpec(group_size=8))
    expected = expected_contrast_pp()
    assert expected > 0.0, "the plant is supposed to make hacking survive better"
    assert reading.contrast_ci_low_pp <= expected <= reading.contrast_ci_high_pp
    assert reading.hack_survival_pp > reading.capability_survival_pp
    assert sorted(reading.hack_features) == sorted(PLANTED_HACK_FEATURES)


def test_recovers_the_planted_localisation() -> None:
    """The entry window survives worse than the body, which is the published direction."""
    _, reading = _read(PlantSpec(group_size=8))
    expected = expected_region_contrast_pp()
    assert expected < 0.0
    assert reading.region_ci_low_pp <= expected <= reading.region_ci_high_pp


# ---------------------------------------------------------------------------
# The correction, which is the finding this package had to make before it could measure
# ---------------------------------------------------------------------------


def test_the_uncorrected_slope_depends_on_completions_per_prompt_and_the_corrected_one_does_not() -> (
    None
):
    """The reason the errors-in-variables correction is not optional.

    The raw through-origin slope is attenuated by sampling error in its own regressor, and the
    attenuation shrinks as the completions per prompt rise. So the raw estimator is not a shared
    quantity: two labs auditing one pair of checkpoints at K = 4 and K = 16 would publish different
    survival fractions for it. The corrected one is stable across the same change.
    """
    expected = 100.0 * expected_pooled_survival()
    _, low = _read(PlantSpec(group_size=4))
    _, high = _read(PlantSpec(group_size=16))

    raw_drift = abs(high.raw_survival_pp - low.raw_survival_pp)
    corrected_drift = abs(high.survival_pp - low.survival_pp)
    assert raw_drift > 3.0, f"expected the raw slope to move with K; it moved {raw_drift:.2f} pp"
    assert corrected_drift < raw_drift

    # And the raw slope is biased toward zero at both K, while the corrected one brackets the plant.
    assert low.raw_survival_pp < low.survival_pp
    assert high.raw_survival_pp < high.survival_pp
    assert low.survival_ci_low_pp <= expected <= low.survival_ci_high_pp
    assert high.survival_ci_low_pp <= expected <= high.survival_ci_high_pp
    assert low.reliability < high.reliability < 1.0


def test_the_correction_is_the_identity_when_there_is_no_sampling_error() -> None:
    """With zero measured within-prompt variance the corrected slope is the raw slope.

    The property that says this is a correction rather than a second estimator.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 5))
    y = 0.6 * x + rng.normal(scale=0.05, size=(40, 5))
    zero = np.zeros_like(x)
    design = ShiftDesign(x=x, y=y, var_base=zero, var_expert=zero)
    assert corrected_slope(design) == pytest.approx(raw_slope(design))
    assert reliability(design) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Lint. E56: an acceptance test that renders a reading and never lints proves half of it.
# ---------------------------------------------------------------------------


def test_the_instrument_lints() -> None:
    arms, _ = _read(PlantSpec(group_size=4))
    instrument = DistillationGap(
        base=arms["base"],
        expert=arms["expert"],
        student=arms["student"],
        featuriser=_featuriser(),
    )
    assert lint_instrument(instrument) == []
    assert instrument.quantity == "artifact.distillation_delta"
    assert instrument.baselines, "lint rule 2: a claim with no dumb baseline is not a claim"
    assert instrument.envelope is not None
    assert instrument.invariance == "group.permutation"


def _bank_from(payload) -> BaselineBank:
    """Rebuild the bank's refusal set from what the reading recorded.

    A downstream consumer holding only the stored row has exactly this: a mapping of the baselines
    that scored and a mapping of the ones that refused with the reason. `lint_claim`'s third rule
    reads the refusals, so reconstructing them is how the shipped lint gets applied to a stored
    reading rather than to a live bank.
    """
    return BaselineBank(
        task_name="K1:expert-vs-student",
        readings={
            bid: Refusal(
                instrument=bid,
                reason=RefusalReason.RECORD_INCOMPLETE,
                detail=detail,
                remedy="supply the input this baseline reads on the same items",
            )
            for bid, detail in payload.baseline_refusals.items()
        },
    )


def test_the_reading_lints_and_carries_its_baselines() -> None:
    """Lint rule 4 on the reading, and the claim-level baseline rule on its payload."""
    arms, _ = _read(PlantSpec(group_size=8))
    instrument = DistillationGap(
        base=arms["base"],
        expert=arms["expert"],
        student=arms["student"],
        blanks=[arms[f"blank{i}"] for i in range(3)],
        featuriser=_featuriser(),
        hack_features=PLANTED_HACK_FEATURES,
        markers=(HACK_MARKER,),
    )
    from reward_lens.measure.base import Context

    reading = instrument.estimate(Context())
    assert not isinstance(reading, Refusal), getattr(reading, "detail", "")
    assert lint_reading(reading, instrument) == []
    assert reading.baselines, "Evidence.baselines was not populated by the emit path"

    payload = reading.value
    # Every one of the six either scored or is recorded as refused with what it needed. The two
    # that refuse here refuse for a structural reason: `gradnorm_peak` reads a logged per-item
    # scalar series that a rollout comparison does not have, and `scaffolded_prompt` needs a judge
    # callable. Recording the refusal is the rule; dropping them silently is what it forbids.
    assert set(payload.baselines) | set(payload.baseline_refusals) >= set(ALL_SIX)
    assert lint_claim(payload, _bank_from(payload)) == []
    assert set(payload.baseline_refusals) >= {
        "baseline.gradnorm_peak",
        "baseline.scaffolded_prompt",
    }


def test_the_generated_invariance_test_passes() -> None:
    """Permuting rollouts within a prompt leaves the survival slope exactly where it was.

    The instrument is adapted to `InvariancePayload`'s vocabulary the way a scores-only
    instrument is: `scores` carries every rollout's value for every (arm, prompt, feature) cell and
    `group_ids` labels each such cell, so a within-group permutation is a reordering of the rollouts
    inside one cell. The statistic re-forms the three arms, takes the per-prompt means and their
    sampling variances, and returns the corrected slope. A permutation inside a cell cannot move a
    mean or a variance, so the relation is exact rather than approximate, and the test's tolerance
    of 5e-08 is what catches an implementation that reads rollout order anywhere.
    """
    rng = np.random.default_rng(11)
    n_arms, n_prompts, k, n_rollouts = 3, 20, 5, 6
    cells = n_arms * n_prompts * k
    base = rng.normal(size=(n_prompts, k, n_rollouts))
    expert = base + 1.0 + rng.normal(scale=0.5, size=base.shape)
    student = base + 0.62 + rng.normal(scale=0.5, size=base.shape)
    scores = np.stack([base, expert, student]).ravel()
    group_ids = np.repeat(np.arange(cells), n_rollouts)

    def survival_of(payload: InvariancePayload) -> float:
        block = np.asarray(payload.scores, dtype=np.float64).reshape(
            n_arms, n_prompts, k, n_rollouts
        )
        mu = block.mean(axis=3)
        var = block.var(axis=3, ddof=1) / n_rollouts
        return corrected_slope(
            ShiftDesign(
                x=mu[1] - mu[0],
                y=mu[2] - mu[0],
                var_base=var[0],
                var_expert=var[1],
            )
        )

    report = check_invariance(
        survival_of,
        "group.permutation",
        InvariancePayload(scores=scores, group_ids=group_ids),
        n=40,
    )
    assert report.passed, report.render()


# ---------------------------------------------------------------------------
# Every refusal is reachable, and every one carries an instruction
# ---------------------------------------------------------------------------


def _one_rollout_arm(name: str, prompts: int = 8) -> Arm:
    return Arm(
        name=name,
        rollouts={
            f"prompt-{i:04d}": (
                make_trajectory(
                    id=f"{name}/{i}",
                    task_ref=f"prompt-{i:04d}",
                    turns=(Turn(index=0, role="assistant", text="a b c d"),),
                    features={n: 1.0 for n in PLANTED_NAMES},
                ),
            )
            for i in range(prompts)
        },
    )


def test_refuses_when_the_arms_do_not_share_a_prompt_set() -> None:
    arms = plant(PlantSpec(n_prompts=8, group_size=2))
    shifted = Arm(
        name="student",
        rollouts={f"other-{k}": v for k, v in arms["student"].rollouts.items()},
    )
    reading = distillation_survival(arms["base"], arms["expert"], shifted, featuriser=_featuriser())
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.RECORD_INCOMPLETE
    assert "same prompt file" in reading.remedy


def test_refuses_a_single_completion_per_prompt() -> None:
    """The correction needs a within-prompt variance and one completion gives none."""
    arms = plant(PlantSpec(n_prompts=8, group_size=2))
    reading = distillation_survival(
        arms["base"], arms["expert"], _one_rollout_arm("student"), featuriser=_featuriser()
    )
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.RECORD_INCOMPLETE
    assert "two completions per prompt" in reading.detail or "two completions" in reading.remedy


def test_refuses_below_five_shared_prompts() -> None:
    arms = plant(PlantSpec(n_prompts=4, group_size=4))
    reading = distillation_survival(
        arms["base"], arms["expert"], arms["student"], featuriser=_featuriser()
    )
    assert isinstance(reading, Refusal)
    assert "shared prompts" in reading.detail


def test_refuses_when_nothing_was_installed() -> None:
    """A student identical to a base identical to an expert has no denominator.

    `BELOW_LOD` rather than a survival of anything: the reading says the RL run left no trace this
    basis can see, which is a finding about the run and not a failed measurement.
    """
    arms = plant(PlantSpec(group_size=4))
    reading = distillation_survival(
        arms["base"],
        arms["blank0"],  # a re-draw of the base standing in for an expert that installed nothing
        arms["blank1"],
        blanks=[arms["blank2"]],
        featuriser=_featuriser(),
    )
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.BELOW_LOD
    assert reading.remedy


def test_refuses_arms_drawn_under_different_decoding() -> None:
    """A temperature difference lands in the same number as a weight difference."""
    from reward_lens.measure.base import Context

    arms = plant(PlantSpec(n_prompts=8, group_size=2))
    instrument = DistillationGap(
        base=arms["base"],
        expert=arms["expert"],
        student=arms["student"],
        featuriser=_featuriser(),
        sampling={
            "base": {"temperature": 1.0, "top_p": 1.0},
            "expert": {"temperature": 1.0, "top_p": 1.0},
            "student": {"temperature": 0.0, "top_p": 1.0},
        },
    )
    reading = instrument.estimate(Context())
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ENVELOPE_VIOLATED
    assert "temperature" in reading.detail
    assert reading.statistics["differing_keys"] == ["temperature"]


def test_matched_decoding_does_not_refuse() -> None:
    from reward_lens.measure.base import Context

    arms = plant(PlantSpec(group_size=4))
    same = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 512}
    instrument = DistillationGap(
        base=arms["base"],
        expert=arms["expert"],
        student=arms["student"],
        blanks=[arms["blank0"]],
        featuriser=_featuriser(),
        sampling={"base": same, "expert": same, "student": same},
    )
    assert not isinstance(instrument.estimate(Context()), Refusal)


# ---------------------------------------------------------------------------
# The region featuriser, on real text rather than on planted feature values
# ---------------------------------------------------------------------------


def test_the_region_featuriser_splits_at_a_word_boundary() -> None:
    entry, body = split_region("one two three four five", entry_words=2)
    assert entry == "one two"
    assert body == "three four five"
    assert region_of("entry:response_chars") == "entry"
    assert region_of("n_turns") == "whole"


def test_the_region_featuriser_runs_on_generated_text() -> None:
    """The `SurfaceFeatures`/`RegionFeatures` path over real text, end to end."""
    arms = plant(PlantSpec(group_size=8))
    reading = distillation_survival(
        arms["base"],
        arms["expert"],
        arms["student"],
        blanks=[arms[f"blank{i}"] for i in range(3)],
        featuriser=RegionFeatures(entry_words=6),
        hack_features=("entry:type_token_ratio", "body:type_token_ratio"),
        markers=(HACK_MARKER,),
        seed=3,
    )
    assert not isinstance(reading, Refusal), getattr(reading, "detail", "")
    assert reading.featuriser == "RegionFeatures"
    assert any(n.startswith("entry:") for n in reading.feature_names)
    assert any(n.startswith("body:") for n in reading.feature_names)


def test_surface_features_drops_the_constant_feature_by_name() -> None:
    """Every planted trajectory has two turns, so `n_turns` cannot carry a shift and is named."""
    arms = plant(PlantSpec(group_size=4))
    reading = distillation_survival(
        arms["base"],
        arms["expert"],
        arms["student"],
        blanks=[arms["blank0"]],
        featuriser=SurfaceFeatures(),
    )
    assert not isinstance(reading, Refusal), getattr(reading, "detail", "")
    assert "n_turns" in reading.constant_features
    assert "n_turns" not in reading.feature_names


# ---------------------------------------------------------------------------
# The frozen study, the price and the runbook
# ---------------------------------------------------------------------------


def test_the_study_is_frozen_and_its_id_depends_only_on_the_spec() -> None:
    a = frozen_study(frozen_at="1970-01-01T00:00:00+00:00")
    b = frozen_study(frozen_at="2030-01-01T00:00:00+00:00")
    assert a.study_id == b.study_id == "study:w6-3-distillation-gap@v1#5d34c1df"
    spec = build_spec()
    assert len(spec.hypotheses) == 5
    assert len(spec.kill_criteria) == 3
    assert {k.id for k in spec.kill_criteria} == {"K1-DEAD", "K1-DUMB", "K1-NO-DENOMINATOR"}
    for h in spec.hypotheses:
        assert h.prediction.rationale, f"{h.id} registers no rationale"


def test_the_study_runs_on_the_plant_and_voids_the_arm_that_needs_a_grader(tmp_path) -> None:
    """Four arms confirm on the plant; the aggregate-score arm produces no metric and voids.

    A planted invisibility would prove only that the planter chose it, so `aggregate_score_z` is
    never written and the runner turns its absence into a named void with a remedy. The study is
    therefore VOID as a whole, which is the correct state for a compute-gated row: a void is a work
    item and not a result.
    """
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    assert result.outcomes["H1-something-survives"] == "confirmed"
    assert result.outcomes["H2-hack-survives-better"] == "confirmed"
    assert result.outcomes["H3-entry-survives-worse"] == "confirmed"
    assert result.outcomes["H5-beats-the-dumb-bank"] == "confirmed"
    assert result.outcomes["H4-invisible-in-aggregate"] == "void"
    assert "aggregate_score_z" not in result.metrics
    assert result.outcome.value == "void"

    # No kill criterion could be silently un-evaluated: each is `passed` or `fired`, never absent.
    assert set(result.kill_outcomes) == {"K1-DEAD", "K1-DUMB", "K1-NO-DENOMINATOR"}
    assert all(v in {"passed", "fired"} for v in result.kill_outcomes.values())
    assert not result.killed

    gate = store.find(observable="K1.AggregateScoreGate")
    assert gate and gate[0].trust is TrustLevel.REGISTERED
    assert gate[0].value["blocks_metric"] == "aggregate_score_z"
    assert gate[0].value["price_gpu_hours"] > 0
    assert "shared base" in gate[0].value["missing_subject"]

    reading = store.find(observable="K1.DistillationGap")
    assert reading and reading[0].trust is TrustLevel.REGISTERED
    assert reading[0].quantity == "artifact.distillation_delta"

    report = render_report(frozen, result, store)
    assert "VOID" in report


def test_without_a_blank_arm_the_loq_kill_voids_rather_than_firing(tmp_path) -> None:
    """A kill criterion must never fire because a check did not happen.

    With no blank arm nothing measures the detection floor, every feature's verdict is `unmeasured`,
    and a naive count of quantifiable features would be zero, which would fire `K1-NO-DENOMINATOR`
    and report "there is almost nothing to divide by" about a study where nobody looked. The metric
    is withheld instead, so the criterion voids with the remedy attached.
    """
    arms = plant(PlantSpec(group_size=4), n_blanks=0)
    subjects = dict(arms)
    subjects["featuriser"] = _featuriser()
    subjects["hack_features"] = PLANTED_HACK_FEATURES
    _, result = run_study(build_spec(), subjects=subjects, store=EvidenceStore(tmp_path))

    assert "n_features_above_loq" not in result.metrics
    assert result.kill_outcomes["K1-NO-DENOMINATOR"] == "void"
    assert not result.killed
    assert "K1-NO-DENOMINATOR" in result.voids
    assert result.voids["K1-NO-DENOMINATOR"].remedy
    # And the measurement itself still ran: withholding the verdicts is not withholding the reading.
    assert result.metrics["survival_pp"] > 0


def test_the_power_calculation_runs_at_the_realised_n(tmp_path) -> None:
    """M10 on the design it fits, and a stated MDE on the one it does not."""
    _, result = run_study(build_spec(), store=EvidenceStore(tmp_path))
    assert result.metrics["contrast_mde_pp"] > 0
    assert result.metrics["contrast_prompts_for_target"] > result.metrics["n_prompts"]
    assert 0.0 <= result.metrics["detector_power"] <= 1.0
    assert result.metrics["detector_n_star"] > 0


def test_the_price_is_computed_and_the_measurement_is_not_what_costs_money() -> None:
    bill = price()
    assert bill.gpu_hours == pytest.approx(1323.14, abs=0.05)
    assert bill.dollars[0] == pytest.approx(1984.71, abs=0.5)
    assert bill.dollars[1] == pytest.approx(2659.51, abs=0.5)
    # The finding: producing the checkpoints is the bill and the measurement is a rounding error.
    assert bill.inference_gpu_hours / bill.gpu_hours < 0.01
    assert bill.sensitivity(4.0) < 0.05
    # And the specification's own scale marker reproduces from the sourced RL-run unit.
    assert reference_multi_seed_gpu_hours() == pytest.approx(11_520.0)


def test_the_runbook_renders_and_matches_the_checked_in_copy() -> None:
    from pathlib import Path

    rendered = render_runbook()
    for arm in ("A0", "A1", "A2", "A4"):
        assert f"### {arm}" in rendered, f"the runbook does not name arm {arm}"
    assert "A failed A0 looks like" in rendered
    assert "A failed A1 looks like" in rendered
    assert "A failed A2 looks like" in rendered
    assert "study:w6-3-distillation-gap@v1#5d34c1df" in rendered

    checked_in = Path(__file__).resolve().parents[2] / "studies/w6_distillation/RUNBOOK.md"
    assert checked_in.exists()
    assert checked_in.read_text() == rendered, (
        "RUNBOOK.md has drifted from its renderer. Regenerate it with "
        "`python -m studies.w6_distillation.runbook > studies/w6_distillation/RUNBOOK.md`."
    )


def test_shared_prompts_is_an_intersection() -> None:
    a = Arm("a", {"p0": (1,), "p1": (1,)})  # type: ignore[arg-type]
    b = Arm("b", {"p1": (1,), "p2": (1,)})  # type: ignore[arg-type]
    assert shared_prompts(a, b) == ("p1",)
    assert shared_prompts() == ()

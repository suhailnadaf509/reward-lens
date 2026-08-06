"""Acceptance: amplifier safety on every component of one real composite reward.

The clause this file discharges, verbatim: *amplifier safety is computed for every component of one
real composite reward and the ranking is recorded before any collapse is observed, as a frozen
prediction.*

**The reward is a real composite and the run is a real GRPO run.** Three reward functions at weights
1.0, 0.3 and 0.1 go through `reward_lens.tap.adapters.trl`, so the record carries a three-leaf
`WeightedSum` per rollout, an `EstimatorSpec` read off `GRPOConfig`, and 24 optimizer steps of
`OptimizerTelemetry`. The task component is a binary verifier the policy sometimes satisfies, which
is what gives the window both all-fail groups and mixed ones; the other two are auxiliaries whose
variance does not depend on whether the task was solved, which is the dangerous shape.

**How to run it.** TRL is not installed in the shared venv; the TRL tap installs it to a
scratch `--target` directory and ran against that. Two routes, and both are here:

    pip install --no-deps --target /tmp/trlpkgs "trl>=1.9.2,<2.0"
    PYTHONPATH=/tmp/trlpkgs pytest tests/acceptance/test_w3_5_estimator.py

or, against a record a previous run already wrote:

    REWARD_LENS_GRPO_RECORD=/path/to/record pytest tests/acceptance/test_w3_5_estimator.py

Without either, every test here skips with the reason named. That is the same shape as
`test_w4_1_trl_tap.py`, which is the acceptance test that produced this record's format.

**The freeze, and what is honestly registered by it.** `PREDICTION` below is a module constant. It
is frozen with a fixed timestamp so its content hash is stable, and the hash is pinned in a test, so
editing any registered field after seeing a number produces a visibly different study version. Two
of its three hypotheses are derivations from the mechanism rather than guesses, and the file says so
where they are declared: a binary task component **cannot** vary inside a group where every rollout
failed, so its amplifier safety is exactly zero and it must rank safest. The third is a genuine
forecast about runs that do not exist yet.

**The kill criterion is registered and left unresolved, deliberately.** E4's catalogue kill
condition is that amplifier safety does not predict collapse on three held-out runs. One run exists.
Resolving that criterion here would mean adjudicating a prediction against a sample of one, so the
criterion is frozen with its threshold and the test asserts it is **unresolved**, which is what
`StudyOutcome.VOID` is for. A void study is a work item rather than a result.

**The git stamp.** Six agents are writing in this tree while this runs, so `freeze()` records a
dirty git sha and the study is not reproducible from that sha alone. The content hash does not
depend on it. Re-freezing at commit time is what makes the stamp meaningful, and it is a work item
rather than a defect in the freeze.

**The frozen numbers moved once, and this is where that is written down.** E4's denominator pooled
all-pass groups with mixed ones; correcting it holds 2 of this record's 48 groups out of the
denominator. `length_reward` moved from 1.9323 to 1.8640, `format_reward` from 0.9057 to 0.9293,
`task_reward` stayed at exactly 0. Every registered prediction keeps its verdict: H1's gap goes
from -0.9057 to -0.9293 and stays below zero, H2's maximum goes from 1.9323 to 1.8640 and stays at
or above one, the ranking is `length_reward > format_reward > task_reward` both ways, and the three
verdicts are unchanged. `EXPECTED_SPEC_HASH` therefore does not move: no registered field of
`PREDICTION` was edited. `test_the_correction_to_the_denominator_did_not_move_the_frozen_verdicts`
pins both sets of numbers so the re-basing is a record rather than a silent edit.
"""

from __future__ import annotations

import math
import os
import pathlib

import numpy as np
import pytest

from reward_lens.core.quantity import load_quantities
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import Context, lint_instrument
from reward_lens.measure.estimator import (
    ESTIMATOR,
    AllFailGroups,
    AmplifierSafety,
    ClipAccounting,
    DegenerateGroups,
    FailureFloor,
    LogprobMismatch,
    NoiseShare,
    RecordedEstimator,
    check_replay,
    measure_amplifier_safety,
    register_all,
)
from reward_lens.studies.freeze import freeze
from reward_lens.studies.spec import (
    Hypothesis,
    KillCriterion,
    Prediction,
    StudySpec,
    SubjectQuery,
)

load_quantities()
register_all()

MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
SEED = 1234
STEPS = 24
BATCH = 8
K = 4

#: The task component, and the score that counts as a failure on it. Zero, because the component is
#: a binary verifier. The composite total will not do: a rollout that fails the task and collects a
#: format bonus scores above zero and has passed nothing.
TASK_COMPONENT = "task_reward"
FLOOR = FailureFloor(at=0.0, component=TASK_COMPONENT)
COMPONENTS = ("task_reward", "format_reward", "length_reward")


# ===========================================================================
# The frozen prediction
# ===========================================================================

#: Fixed so the study's identity does not move with the clock. `freeze` does not hash the
#: timestamp, so this only affects what `FrozenStudy.predates` reports.
FROZEN_AT = "2026-08-01T00:00:00+00:00"

PREDICTION = StudySpec(
    id="w3-5-amplifier-safety",
    title="Amplifier safety ranks the components of a composite reward by their danger",
    science="S05-estimator",
    hypotheses=(
        Hypothesis(
            id="H1",
            statement=(
                "The binary task component ranks strictly safer than every auxiliary. This is a "
                "derivation rather than a guess: a component that is identically zero inside an "
                "all-fail group has zero within-group variance there by construction, so its "
                "amplifier safety is exactly zero, and an auxiliary whose variance does not depend "
                "on task success keeps some. Registering it makes the derivation falsifiable: an "
                "auxiliary that also vanishes in the all-fail phase would tie."
            ),
            prediction=Prediction(
                metric="task_safety_minus_min_auxiliary_safety",
                comparator="<",
                threshold=0.0,
                rationale="strictly safer, not merely no worse",
            ),
        ),
        Hypothesis(
            id="H2",
            statement=(
                "At least one auxiliary is a live amplifier, meaning its variance in the all-fail "
                "groups is at least as large as in the mixed ones. This is the condition section "
                "3.2 states and it is not implied by H1: every auxiliary could decay together with "
                "the task signal, in which case the composite is amplifier-safe throughout."
            ),
            prediction=Prediction(
                metric="max_auxiliary_safety",
                comparator=">=",
                threshold=1.0,
                rationale="order one or above is a live amplifier",
            ),
        ),
        Hypothesis(
            id="H3",
            statement=(
                "The ranking is a property of the components rather than of their scales. Rescaling "
                "any component by a positive factor and shifting it by a constant leaves its "
                "amplifier safety unchanged, because both variances are variances of that same "
                "component. This is what makes the ranking comparable across reward terms whose "
                "magnitudes differ by orders of magnitude, and it is the reason magnitude is the "
                "wrong diagnostic."
            ),
            prediction=Prediction(
                metric="max_rescaling_deviation",
                comparator="<",
                threshold=1e-9,
                rationale="exact in float64: a**2 cancels in a ratio of two variances",
            ),
        ),
    ),
    analysis="reward_lens.measure.estimator.measure_amplifier_safety",
    subjects=SubjectQuery(
        signals=("trl-internal-testing/tiny-Qwen3ForCausalLM",),
        datasets=("w3-5-count-upward",),
        extra={
            "estimator": "grpo/dapo, group z-score, std_epsilon=1e-4",
            "components": list(COMPONENTS),
            "failure_floor": {"component": TASK_COMPONENT, "at": 0.0},
            "steps": STEPS,
            "group_size": K,
        },
    ),
    kill_criteria=(
        KillCriterion(
            id="K1",
            metric="n_heldout_runs_where_safety_predicted_collapse",
            comparator="<",
            threshold=3.0,
            description=(
                "E4's catalogue kill condition. If amplifier safety does not predict collapse on "
                "three held-out runs, it is a description rather than a forecast. One run exists, "
                "so this criterion is registered and left unresolved: adjudicating it against a "
                "sample of one would be the failure this whole discipline exists to prevent."
            ),
        ),
    ),
    version=1,
    notes=(
        "Frozen before any collapse is observed. The record this is measured on runs 24 steps and "
        "no collapse happens in it, which is the point: the ranking is a forecast about what would "
        "amplify if the all-fail fraction rose, not a description of something that already did."
    ),
)

FROZEN = freeze(PREDICTION, frozen_at=FROZEN_AT)

#: Pinned, so that editing any registered field of `PREDICTION` after seeing a number fails here
#: rather than passing quietly under a new hash. `freeze` hashes the hypotheses, their predictions,
#: the analysis path, the subjects and the kill criteria; it does not hash `notes`, the timestamp or
#: the git sha.
EXPECTED_SPEC_HASH = "spec:2c11a4775ab061d7eb0532aa8ab51315"


# ===========================================================================
# The record
# ===========================================================================


def task_reward(prompts, completions, completion_ids, **kwargs):
    """A binary verifier the tiny policy satisfies about a third of the time."""
    return [1.0 if c.count(" ") >= 6 else 0.0 for c in completions]


def format_reward(prompts, completions, completion_ids, **kwargs):
    """An auxiliary whose variance does not depend on whether the task was solved."""
    return [(sum(1 for ch in c if ch.isupper()) % 5) / 4.0 for c in completions]


def length_reward(prompts, completions, completion_ids, **kwargs):
    """A second auxiliary, with one abstention per seven rollouts so that path is exercised."""
    return [None if (i % 7 == 3) else len(c) / 50.0 for i, c in enumerate(completions)]


def _train() -> object:
    """Run 24 real GRPO steps and return the `Run` the tap built."""
    pytest.importorskip("torch")
    pytest.importorskip("datasets")
    pytest.importorskip(
        "trl",
        reason=(
            "TRL is not in the shared venv. Install it to a scratch target and put that on "
            "PYTHONPATH: pip install --no-deps --target /tmp/trlpkgs 'trl>=1.9.2,<2.0'"
        ),
    )
    import tempfile

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from reward_lens.tap.adapters.trl import TRLTap
    from reward_lens.tap.contract import TapBudget

    budget = TapBudget(
        max_added_latency_ms_p99=1000.0,
        max_resident_bytes=64 * 1024 * 1024,
        max_added_alloc_bytes_per_step=64 * 1024 * 1024,
    )
    tap = TRLTap(run_id="w3-5-estimator", budget=budget, emit_extra=True)
    prompts = [f"question {i}: count upward from {i}" for i in range(BATCH * 4)]
    with tempfile.TemporaryDirectory() as out:
        args = GRPOConfig(
            output_dir=out,
            per_device_train_batch_size=BATCH,
            num_generations=K,
            max_completion_length=12,
            max_steps=STEPS,
            logging_steps=1,
            report_to=[],
            seed=SEED,
            use_cpu=True,
            save_strategy="no",
            disable_tqdm=True,
            log_completions=True,
            reward_weights=[1.0, 0.3, 0.1],
        )
        trainer = GRPOTrainer(
            model=MODEL,
            reward_funcs=tap.wrap([task_reward, format_reward, length_reward]),
            args=args,
            train_dataset=Dataset.from_dict({"prompt": prompts}),
        )
        tap.attach(trainer)
        trainer.train()
    return tap.finish()


@pytest.fixture(scope="module")
def run():
    """The real record: read from disk when one was written, otherwise trained here."""
    from reward_lens.record.reader import open_run

    recorded = os.environ.get("REWARD_LENS_GRPO_RECORD")
    if recorded:
        root = pathlib.Path(recorded)
        runs = sorted((root / "runs").glob("run_*")) if (root / "runs").is_dir() else []
        if not runs:
            pytest.skip(f"no record under {root}/runs")
        return open_run(root, runs[0].name.replace("run_", "run:"))
    return _train()


@pytest.fixture(scope="module")
def reading(run):
    """Through the instrument rather than the bare function, so the envelope is measured."""
    out = AmplifierSafety(run, floor=FLOOR, trajectory=True).estimate(Context())
    assert not isinstance(out, Refusal), out.render()
    return out.value


def _auxiliaries(safety: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in safety.items() if k != TASK_COMPONENT}


# ===========================================================================
# Clause: computed for every component of one real composite reward
# ===========================================================================


def test_the_reward_is_a_real_three_leaf_composite_and_the_estimator_is_the_one_that_ran(run):
    """E1 first, because everything below is conditioned on it."""
    from reward_lens.record.scores import WeightedSum, leaves

    spec_reading = RecordedEstimator(run).estimate(Context())
    assert not isinstance(spec_reading, Refusal), spec_reading
    value = spec_reading.value
    assert value.family.startswith("grpo/")
    assert value.z_scored, "amplifier safety is a statement about the z-score"
    assert value.spec_fields["std_epsilon"] == 1e-4
    assert value.stable, "one transform across the window, or nothing pools across it"
    assert value.n_groups == 2 * STEPS
    assert value.default_table == "trl/grpo"

    trees = [t.scores for s in run.steps for g in s.groups for t in g.trajectories]
    assert all(isinstance(t, WeightedSum) for t in trees)
    assert {leaf.name for leaf in leaves(trees[0])} == set(COMPONENTS)
    assert trees[0].weights == (1.0, 0.3, 0.1)


def test_amplifier_safety_is_computed_for_every_component(reading):
    assert set(reading.safety) == set(COMPONENTS)
    for name in COMPONENTS:
        assert math.isfinite(reading.safety[name]), name
        detail = reading.detail[name]
        assert detail["n_allfail_groups"] > 0 and detail["n_mixed_groups"] > 0
        assert detail["dof_allfail"] > 0 and detail["dof_mixed"] > 0
        assert reading.verdicts[name] in {"live amplifier", "watch", "amplifier-safe"}

    assert reading.n_allfail_groups > 0 and reading.n_mixed_groups > 0
    # A three-way partition, and it was a two-way one. This assertion read
    # `n_allfail + n_mixed == n_groups`, which is the statement that every group that is not
    # all-fail is mixed, and that is what put the all-pass groups in the denominator. On this
    # record 2 of the 48 groups are all-pass, and holding them out moves `length_reward` from
    # 1.9323 to 1.8640 and `format_reward` from 0.9057 to 0.9293.
    assert (
        reading.n_allfail_groups
        + reading.n_mixed_groups
        + reading.n_allpass_groups
        + reading.n_unscored_groups
        == reading.n_groups
        == 2 * STEPS
    )
    assert (reading.n_allfail_groups, reading.n_mixed_groups) == (14, 32)
    assert (reading.n_allpass_groups, reading.n_unscored_groups) == (2, 0)
    for name in COMPONENTS:
        detail = reading.detail[name]
        assert detail["n_allpass_groups"] == 2.0, name
        assert detail["n_mixed_groups"] == 32.0, name
    assert "all-pass held out of the denominator" in reading.render()

    assert reading.envelope_measured, "GROUP_NONDEGENERATE measured, not assumed"
    assert reading.envelope_statistic <= 0.20


def test_the_mandatory_baseline_is_rendered_beside_the_ratio(reading):
    """The component's absolute magnitude, which is what people look at and which the z-score
    removes. Both rankings are on the reading and both are in the sentence."""
    assert set(reading.magnitude_ranking) == set(COMPONENTS)
    assert set(reading.ranking) == set(COMPONENTS)
    for name in COMPONENTS:
        assert math.isfinite(reading.baselines[f"magnitude/{name}"]), name
        assert math.isfinite(reading.baselines[f"sd/{name}"]), name
    assert "baseline.component_magnitude" in reading.baselines
    rendered = reading.render()
    assert "AmplifierSafety" in rendered and "mean |r|" in rendered


# ===========================================================================
# Clause: the ranking, against the frozen prediction
# ===========================================================================


def test_the_frozen_prediction_is_hashed_and_its_content_has_not_moved():
    assert FROZEN.spec_hash == EXPECTED_SPEC_HASH, (
        "a registered field of PREDICTION changed. That is allowed and it produces a new study "
        f"version; update EXPECTED_SPEC_HASH to {FROZEN.spec_hash!r} and bump PREDICTION.version "
        "so the edit is visible rather than silent."
    )
    assert FROZEN.study_id.startswith("study:w3-5-amplifier-safety@v1#")
    assert FROZEN.predates == FROZEN_AT
    assert [h.id for h in PREDICTION.hypotheses] == ["H1", "H2", "H3"]


def test_the_ranking_confirms_the_registered_prediction(reading):
    safety = reading.safety
    aux = _auxiliaries(safety)

    h1 = PREDICTION.hypotheses[0].prediction
    gap = safety[TASK_COMPONENT] - min(aux.values())
    assert h1.check(gap), (
        f"H1 refuted: task safety {safety[TASK_COMPONENT]:.4g} against the safest auxiliary "
        f"{min(aux.values()):.4g}"
    )
    assert reading.ranking[-1] == TASK_COMPONENT
    assert safety[TASK_COMPONENT] == 0.0, "identically zero inside an all-fail group, by derivation"

    h2 = PREDICTION.hypotheses[1].prediction
    assert h2.check(max(aux.values())), (
        f"H2 refuted: no auxiliary reaches {h2.threshold}; the largest is {max(aux.values()):.4g}"
    )
    assert any(reading.verdicts[name] == "live amplifier" for name in aux)


def test_the_ranking_does_not_move_when_a_component_is_rescaled(run, reading):
    """H3. The ratio is a ratio of variances of the same component, so `a**2` cancels exactly."""
    from dataclasses import replace as dc_replace

    from reward_lens.record.scores import Leaf, WeightedSum

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(8):
        a = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
        b = float(rng.normal(0.0, 1.0))
        target = "format_reward"

        def rescale(tree):
            return dc_replace(
                tree,
                children=tuple(
                    dc_replace(c, value=a * c.value + b)
                    if isinstance(c, Leaf) and c.name == target and c.value is not None
                    else c
                    for c in tree.children
                ),
            )

        scaled_groups = [
            dc_replace(
                g,
                trajectories=tuple(dc_replace(t, scores=rescale(t.scores)) for t in g.trajectories),
            )
            for s in run.steps
            for g in s.groups
        ]
        assert all(isinstance(g.trajectories[0].scores, WeightedSum) for g in scaled_groups)
        moved = measure_amplifier_safety(scaled_groups, floor=FLOOR)
        assert not isinstance(moved, Refusal), moved
        worst = max(worst, abs(moved.safety[target] - reading.safety[target]))
        assert moved.ranking == reading.ranking

    h3 = PREDICTION.hypotheses[2].prediction
    assert h3.check(worst), f"H3 refuted: the ratio moved by {worst:.3g} under a rescaling"


def test_the_correction_to_the_denominator_did_not_move_the_frozen_verdicts(run, reading):
    """The one number the frozen study was re-based on, with both values in the file.

    E4's denominator was `Var(r_c | not all-fail)`, which pools the groups where nothing failed in
    with the mixed ones. Two of this record's 48 groups are all-pass. Holding them out moves both
    auxiliaries and leaves every registered verdict where it was; a re-basing that changed a verdict
    would be a different study and would need a new hash.
    """
    from reward_lens.measure.estimator import component_scores, partition_by_floor

    groups = [g for s in run.steps for g in s.groups]
    phases = partition_by_floor(groups, FLOOR)
    assert phases.count("all_fail") == 14
    assert phases.count("mixed") == 32
    assert phases.count("all_pass") == 2
    assert phases.count("unscored") == 0

    # The denominator as it used to be: every group that is not all-fail.
    from reward_lens.measure.estimator import pooled_within_variance

    before: dict[str, float] = {}
    for name in COMPONENTS:
        af = [component_scores(g, name) or [] for g, p in zip(groups, phases) if p == "all_fail"]
        not_af = [
            component_scores(g, name) or [] for g, p in zip(groups, phases) if p != "all_fail"
        ]
        v_af, _, _ = pooled_within_variance(af)
        v_pooled, _, _ = pooled_within_variance(not_af)
        before[name] = v_af / v_pooled if v_pooled > 0 else math.inf

    assert before["task_reward"] == pytest.approx(0.0)
    assert before["format_reward"] == pytest.approx(0.905693, rel=1e-5)
    assert before["length_reward"] == pytest.approx(1.932347, rel=1e-5)

    after = reading.safety
    assert after["task_reward"] == pytest.approx(0.0)
    assert after["format_reward"] == pytest.approx(0.929274, rel=1e-5)
    assert after["length_reward"] == pytest.approx(1.864041, rel=1e-5)

    # Both directions occur on one record, which is worth pinning: the natural expectation is that
    # all-pass groups always suppress the denominator, and `format_reward`'s do the opposite.
    assert before["format_reward"] < after["format_reward"]
    assert before["length_reward"] > after["length_reward"]

    # Every frozen verdict survives the move.
    for safety in (before, after):
        aux = _auxiliaries(safety)
        assert PREDICTION.hypotheses[0].prediction.check(safety[TASK_COMPONENT] - min(aux.values()))
        assert PREDICTION.hypotheses[1].prediction.check(max(aux.values()))
        assert sorted(safety, key=lambda n: -safety[n]) == [
            "length_reward",
            "format_reward",
            "task_reward",
        ]
    assert reading.verdicts == {
        "task_reward": "amplifier-safe",
        "format_reward": "watch",
        "length_reward": "live amplifier",
    }
    assert FROZEN.spec_hash == EXPECTED_SPEC_HASH


def test_the_shipped_replay_divisor_is_not_the_one_trl_wrote_these_advantages_with(run):
    """The ddof divergence, isolated on the record TRL itself produced.

    `check_replay` reports 0 of 48 on this record and three separate divergences contribute. Two of
    them are stripped here so the third is visible on its own: the ratio clip written into
    `EstimatorSpec.clip_low`/`clip_high`, and `record.scores.evaluate` returning NaN for a total one
    of whose leaves abstained where TRL's `nansum` contributes zero for it. The second is deliberate
    and recorded in E7: a total missing a term is not a smaller total, and the record
    keeps the distinction TRL discards. What is left is the variance divisor, and TRL's `nanstd`
    applies Bessel's correction (`trl/trainer/utils.py:877-879`) while `replay_advantages` calls
    numpy's `std()`.

    Measured over all 48 groups: `ddof=1` reproduces TRL's own advantages to 9.2e-07 and `ddof=0`
    misses by 0.232, against a `REPLAY_TOL` of 1e-4. Both numbers are computed here rather than
    read off `check_replay`, so the size of the whole disagreement can move as the other two
    divergences are addressed without this measurement moving with it.
    """
    from reward_lens.measure.estimator import REPLAY_TOL, check_replay
    from reward_lens.record.scores import leaves

    check = check_replay(run)
    assert check.checked and not check.agrees
    assert check.max_abs_error > 1000 * REPLAY_TOL

    groups = [g for s in run.steps for g in s.groups]
    eps = groups[0].estimator.std_epsilon
    assert eps == 1e-4
    weights = (1.0, 0.3, 0.1)

    worst0 = worst1 = 0.0
    n = 0
    for g in groups:
        recorded = [t.advantage for t in g.trajectories]
        if any(a is None for a in recorded):
            continue
        totals = []
        for t in g.trajectories:
            values = {
                leaf.name: (None if leaf.abstained else leaf.value) for leaf in leaves(t.scores)
            }
            totals.append(sum(w * (values[c] or 0.0) for w, c in zip(weights, COMPONENTS)))
        arr = np.asarray(totals, dtype=float)
        centred = arr - arr.mean()
        a0 = centred / (arr.std(ddof=0) + eps)
        a1 = centred / (arr.std(ddof=1) + eps)
        b = np.asarray(recorded, dtype=float)
        worst0 = max(worst0, float(np.max(np.abs(a0 - b))))
        worst1 = max(worst1, float(np.max(np.abs(a1 - b))))
        n += 1

    assert n == 2 * STEPS
    assert worst1 < REPLAY_TOL, (
        f"Bessel's correction reproduces TRL's advantages to {worst1:.3g}, so the divisor is "
        f"settled by the record itself and not only by reading TRL's source"
    )
    assert worst1 == pytest.approx(9.16e-07, rel=0.1)
    assert worst0 > 1000 * REPLAY_TOL
    assert worst0 == pytest.approx(0.2317, rel=1e-3)


def test_no_collapse_is_observed_in_the_window_the_ranking_was_taken_on(reading, run):
    """ "Before any collapse is observed" is a claim about the record, so it is checked.

    Collapse in the relevant sense is the all-fail-dominated phase: the task signal vanishing and
    staying gone while an auxiliary keeps moving. Two things establish that it has not happened
    here. The all-fail fraction over the window the ranking is pooled from is below the dominance
    level, and the task signal in the last third of the run is no weaker than in the first. So the
    ranking is a forecast about a phase this record has not entered rather than a description of one
    it has.

    The per-step fraction is deliberately not used for this. Each step of this run holds two groups,
    so a per-step all-fail fraction takes three values and hits 1.0 whenever both groups happen to
    be all-fail. That is the statistic's resolution rather than a collapse, and reading it as one is
    the kind of confident wrong number the window-level statement avoids.
    """
    from reward_lens.measure.estimator import DOMINANCE_AT

    assert len(reading.trajectory["allfail_fraction"]) == STEPS
    groups_per_step = {len(s.groups) for s in run.steps}
    assert groups_per_step == {2}, groups_per_step

    window_fraction = reading.n_allfail_groups / reading.n_groups
    assert window_fraction < DOMINANCE_AT, (
        f"the all-fail fraction over the window is {window_fraction:.3g}, at or past the dominance "
        f"level, so this window is not 'before any collapse'"
    )

    task_by_step = [
        float(np.mean([v for g in s.groups for v in _task_values(g)])) for s in run.steps
    ]
    third = STEPS // 3
    assert max(task_by_step) > 0.0, (
        "the policy solves the task sometimes, so there is a signal to lose"
    )
    early, late = float(np.mean(task_by_step[:third])), float(np.mean(task_by_step[-third:]))
    assert late >= 0.5 * early, (
        f"the task signal fell from {early:.3g} to {late:.3g} across the run, which is the start "
        f"of the phase the ranking is meant to forecast rather than describe"
    )

    # And the projection agrees: the step at which the all-fail fraction would reach dominance sits
    # beyond the end of this record.
    assert reading.predicted_collapse_step is None or reading.predicted_collapse_step > STEPS


def _task_values(group):
    from reward_lens.record.scores import leaves

    return [
        leaf.value
        for t in group.trajectories
        for leaf in leaves(t.scores)
        if leaf.name == TASK_COMPONENT and leaf.value is not None
    ]


def test_the_kill_criterion_is_registered_and_left_unresolved():
    """One run is not three. The criterion is frozen with its threshold and not adjudicated."""
    k1 = PREDICTION.kill_criteria[0]
    assert k1.metric == "n_heldout_runs_where_safety_predicted_collapse"
    assert (k1.comparator, k1.threshold) == ("<", 3.0)
    n_heldout_runs_available = 0
    assert n_heldout_runs_available < 3
    # `fired` would be True on zero held-out runs, and firing a kill criterion on the absence of
    # data rather than on a measurement is the void condition, not a result.
    assert k1.fired(float(n_heldout_runs_available))
    assert "registered and left unresolved" in k1.description


# ===========================================================================
# What the rest of the series does on the same record, stated per instrument
# ===========================================================================


def test_e1_finds_that_the_recorded_spec_does_not_reproduce_the_recorded_advantages(run):
    """Real defects, found on the first real record, and reported by the instrument itself.

    Three of them, enumerated in `ReplayCheck`'s own docstring. Two are now closed and one is not,
    and this test asserts the exact population each of them explains rather than a bare count.

    TRL's `epsilon` and `epsilon_high` are the PPO **ratio** clip, and the tap writes them into
    `EstimatorSpec.clip_low` and `clip_high`. While `record.scores.replay_advantages` applied those
    as bounds on the **advantage**, every replayed advantage came back as exactly 0.2 against
    recorded advantages spanning more than two units: 0 of 48 groups agreed, worst 1.68. The ratio
    clip belongs to the loss and is no longer applied to the advantage (E50).

    `replay_advantages` divided by `std(ddof=0)` while TRL's `nanstd` applies Bessel's correction,
    which is 15.47% at this record's K of 4 against a tolerance of 1e-4. The divisor is now
    `EstimatorSpec.std_ddof` and the tap writes TRL's, which is 1.

    What is left is the abstention convention, and it is deliberate. `record.scores.evaluate`
    returns NaN for a composite total one of whose leaves abstained; TRL aggregates with `nansum`
    and contributes zero for that leaf instead, so the two sides are totalling different things on
    any group holding an abstention. `length_reward` abstains on one rollout in seven, so
    ``1 - (6/7)**4 = 46%`` of four-rollout groups hold one, and the split is exactly that: the
    groups with no abstention reproduce TRL to float32 logging precision and the groups with one do
    not.
    """
    from reward_lens.measure.estimator import REPLAY_TOL
    from reward_lens.record.scores import leaves

    check = check_replay(run)
    assert check.checked and not check.agrees
    assert check.n_comparable == 2 * STEPS
    assert check.max_abs_error > 1000 * REPLAY_TOL

    groups = [g for s in run.steps for g in s.groups]
    clean = [
        g
        for g in groups
        if not any(leaf.abstained for t in g.trajectories for leaf in leaves(t.scores))
    ]
    dirty = [g for g in groups if g not in clean]
    assert len(clean) + len(dirty) == 2 * STEPS
    assert 0 < len(dirty) < len(groups), "the abstaining path is exercised and is not everything"

    # Every clean group replays exactly, and the count is the whole of `n_agree`.
    clean_check = check_replay(clean)
    assert clean_check.n_agree == clean_check.n_comparable == len(clean)
    assert clean_check.max_abs_error < REPLAY_TOL
    assert check.n_agree == len(clean)

    # And no group holding an abstention does, for the one reason left.
    dirty_check = check_replay(dirty)
    assert dirty_check.n_agree == 0
    assert dirty_check.max_abs_error > 1.0
    assert "does not describe the transform that ran" in dirty_check.render()

    spec = next(iter(run.steps)).groups[0].estimator
    assert spec.clip_low == spec.clip_high == 0.2
    assert spec.std_ddof == 1, "TRL's nanstd applies Bessel's correction; the tap records it"
    assert spec.extra["loss_type"] == "dapo"


def test_e2_counts_the_groups_and_refuses_the_all_fail_fraction_without_a_stated_floor(run):
    census = DegenerateGroups(run, floor=FLOOR, attribute=True).estimate(Context())
    assert not isinstance(census, Refusal), census
    value = census.value
    assert value.n_groups == 2 * STEPS
    assert 0.0 <= value.degenerate_fraction <= 1.0
    assert 0.0 < value.allfail_fraction < 1.0
    assert value.baselines["baseline.nominal_group_size"] == float(K)
    # TRL logs the same quantity every step, so the comparator is a real one rather than a stand-in.
    assert math.isfinite(value.baselines["baseline.framework_logged_fraction"])
    assert value.framework_agrees is True

    # And with no floor, the all-fail fraction is refused rather than read off a flag whose
    # threshold the schema does not record. Every flag on this record is False and every group is
    # all-fail on the task component, so a fraction of zero would have been a confident wrong number.
    assert all(not g.group_stats.all_fail for s in run.steps for g in s.groups)
    refused = AllFailGroups(run).estimate(Context())
    assert isinstance(refused, Refusal)
    assert refused.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "records the all-fail verdict without recording the threshold" in refused.detail
    assert refused.is_bounded


def test_e3_refuses_because_the_run_is_trl_and_the_proxies_are_a_verl_feature(run):
    out = NoiseShare(run).estimate(Context())
    assert isinstance(out, Refusal)
    assert "variance-proxy metrics" in out.detail
    assert "actor.calculate_sum_pi_squared: true" in out.remedy


def test_e5_hands_back_the_clip_fraction_as_a_bound_because_only_one_norm_is_recorded(run):
    out = ClipAccounting(run).estimate(Context())
    assert isinstance(out, Refusal)
    assert "none recorded both gradient norms" in out.detail
    assert "grad_norm_unclipped" in out.remedy
    assert out.is_bounded
    bound = out.partial.value
    assert math.isfinite(bound.clip_fraction)
    assert "clip_ratio/" in bound.clip_fraction_source
    assert all(
        s.optimizer.grad_norm_clipped is not None and s.optimizer.grad_norm_unclipped is None
        for s in run.steps
    )

    # The norm that is present is in the wrong field, and this is where that is written down.
    # `transformers/trainer.py:2523-2524` documents `_clip_grad_norm` as returning "the pre-clip
    # gradient norm" and `trainer.py:2087` logs that value under `grad_norm`, which the TRL tap
    # writes into `grad_norm_clipped`. The refusal is still correct, because one norm is one norm
    # whichever field it sits in, but the value is the unclipped one and the label says otherwise.
    # `max_grad_norm` is 1.0 here and every logged norm is below it, so nothing was clipped on this
    # run and the two norms would have been equal anyway.
    norms = [s.optimizer.grad_norm_clipped for s in run.steps]
    assert all(0.0 < n < 1.0 for n in norms), (
        f"logged norms span {min(norms):.4g} to {max(norms):.4g} against max_grad_norm 1.0"
    )


def test_e6_refuses_because_the_tap_records_neither_logprob_stream(run):
    out = LogprobMismatch(run).estimate(Context())
    assert isinstance(out, Refusal)
    assert "no turn in this window carries both logprob streams" in out.detail
    assert "logprobs_sampling" in out.remedy and "logprobs_train" in out.remedy
    assert out.statistics["n_with_sampling"] == 0
    assert out.statistics["n_with_train"] == 0


def test_every_instrument_in_the_series_returns_evidence_or_a_refusal_on_this_record(run):
    """The instrument contract, over a real record: never an exception, never a None, never a zero."""
    instruments = [
        RecordedEstimator(run),
        DegenerateGroups(run, floor=FLOOR, attribute=True),
        AllFailGroups(run, floor=FLOOR),
        NoiseShare(run),
        AmplifierSafety(run, floor=FLOOR),
        ClipAccounting(run),
        LogprobMismatch(run),
    ]
    for instrument in instruments:
        out = instrument.estimate(Context())
        assert out is not None, instrument.name
        if isinstance(out, Refusal):
            assert out.remedy.strip(), instrument.name
        else:
            assert out.value is not None, instrument.name
            assert out.value.render().strip(), instrument.name


def test_lint_instrument_is_empty_for_all_eight_instruments():
    for cls in ESTIMATOR:
        assert lint_instrument(cls()) == [], cls.__name__

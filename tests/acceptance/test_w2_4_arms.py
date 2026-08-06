"""Acceptance: two arms with a deliberate configuration diff produce a void, not a comparison.

The clause this file discharges: *`record/arms.py`. Arms, coupling, common random
numbers, the arm-divergence void gate. Accept: two synthetic arms with a deliberate configuration
diff produce a void rather than a comparison.*

"Rather than a comparison" is the load-bearing half and it is asserted structurally rather than by
inspection: the comparison function records every call it receives, and the test asserts it was
never called on the diverged arms. A gate that returns a warning beside a number is not this.

Four more assertions are here because each is a place the gate can look right and be worthless.
The void has to name the diverging key and print both values, or it is `inconclusive` with a new
spelling. It has to be condition 4 of the seven already declared rather than a new
condition invented in this module. It has to work on two records read back off disk, because that
is where arms actually diverge, long after the plan that built them. And the gate has to be able
to say yes, or it is not a gate.
"""

from __future__ import annotations

import numpy as np

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Access, Component, Substrate
from reward_lens.record import (
    ComponentRef,
    InMemoryStepStream,
    RecordReader,
    RecordSamplingPolicy,
    RecordWriter,
    RegimeDeclaration,
    Run,
    RunID,
    RunLineage,
)
from reward_lens.record.arms import (
    Arm,
    ArmSet,
    arms,
    check_divergence,
    check_runs,
    guard_comparison,
    independent_seeds,
    shared_seeds,
)
from reward_lens.studies.void import StudyOutcome, VoidReason

#: A training configuration with the shape a real one has: nested sections, a mix of types, and a
#: knob in it that nobody thinks about until two arms disagree on it.
BASE_CONFIG = {
    "beta_kl": 0.0,
    "optim": {"lr": 1e-6, "warmup": 100, "clip_ratio": 0.2},
    "rollout": {"k": 8, "temperature": 1.0, "max_tokens": 1024},
    "data": {"dataset": "gsm8k", "split": "train", "decontaminated": True},
    "grader": {"name": "rm-7b", "revision": "v3"},
}

#: One coupling for the whole comparison: shared seeds, common random numbers, the same prompt
#: order, and one engine identity both arms have to be served by.
COUPLING = shared_seeds(2026, engine="vllm@0.11.2+flash_attn_2")


def _calls() -> list[tuple[str, str]]:
    return []


def _paired_difference_factory(log: list[tuple[str, str]]):
    """A comparison that only makes sense on matched arms, wrapped in the gate.

    The reward difference between two arms is meaningful only if the arms differ in the declared
    contrast and in nothing else. `guard_comparison` is what makes that structural: the wrapped
    function is not called at all when the arms diverge.
    """

    @guard_comparison
    def paired_difference(left: Arm, right: Arm, scores_left, scores_right) -> float:
        log.append((left.label, right.label))
        return float(np.mean(scores_left) - np.mean(scores_right))

    return paired_difference


# ---------------------------------------------------------------------------
# The clause
# ---------------------------------------------------------------------------


def test_two_arms_with_a_deliberate_diff_produce_a_void_rather_than_a_comparison():
    """The clause, in one test.

    The two arms are built correctly by `arms()`, and then one of them is relaunched with an
    override the plan never declared, which is how this happens in the field: a stale flag in a
    launch script, a config file edited between the two submissions.
    """
    log = _calls()
    paired_difference = _paired_difference_factory(log)

    plan = arms(BASE_CONFIG, contrast={"beta_kl": [0.0, 0.02]}, coupling=COUPLING)
    control, treated = plan

    # What the launcher actually ran: the treated arm picked up a second change.
    relaunched = treated.overridden({"optim.lr": 2e-6}, label="beta_kl=0.02 (relaunched)")

    scores_a = np.array([0.61, 0.58, 0.66, 0.62])
    scores_b = np.array([0.70, 0.69, 0.72, 0.68])
    out = paired_difference(control, relaunched, scores_a, scores_b)

    # Not a comparison: the number was never computed and the function was never entered.
    assert isinstance(out, Refusal)
    assert log == [], "the comparison must not run on arms that diverged outside the contrast"

    # A void, and one that says what to do about it.
    assert out.reason is RefusalReason.VOID
    report = check_divergence(control, relaunched, contrast=plan.contrast_keys)
    void = report.void
    assert void is not None
    assert void.reason is VoidReason.ARM_DIVERGENCE


def test_the_void_names_the_diverging_key_and_both_values():
    """A void that says "the arms diverged" is worth nothing. This is the whole package."""
    plan = arms(BASE_CONFIG, contrast={"beta_kl": [0.0, 0.02]}, coupling=COUPLING)
    control, treated = plan
    relaunched = treated.overridden({"optim.lr": 2e-6, "rollout.temperature": 0.9})

    report = check_divergence(control, relaunched, contrast=plan.contrast_keys)
    void = report.void
    assert void is not None

    assert report.keys == ("optim.lr", "rollout.temperature")
    assert "optim.lr 1e-06 against 2e-06" in void.detail
    assert "rollout.temperature 1.0 against 0.9" in void.detail
    # The contrast is named too, so the reader can see what was allowed to differ.
    assert "declared to differ in beta_kl" in void.detail

    # The remedy is an instruction the user can act on, not a diagnosis of what went wrong.
    assert void.remedy.startswith("Set optim.lr, rollout.temperature to the same value in both")
    assert "re-run" in void.remedy


def test_the_void_is_condition_four_of_the_seven_and_not_a_new_one():
    """The vocabulary is `studies/void.py`. This module registers into it and does not extend it."""
    plan = arms(BASE_CONFIG, contrast={"beta_kl": [0.0, 0.02]}, coupling=COUPLING)
    control, treated = plan
    void = check_divergence(
        control, treated.overridden({"optim.lr": 2e-6}), contrast=plan.contrast_keys
    ).void

    assert void is not None
    assert void.reason is VoidReason.ARM_DIVERGENCE
    assert void.reason in set(VoidReason)
    # A study carrying this void is VOID, which is a third outcome and not a negative result.
    assert StudyOutcome.VOID is not StudyOutcome.NULL


def test_the_gate_can_say_yes_or_it_is_not_a_gate():
    """Matched arms produce the comparison, and the report says what was compared."""
    log = _calls()
    paired_difference = _paired_difference_factory(log)

    control, treated = arms(BASE_CONFIG, contrast={"beta_kl": [0.0, 0.02]}, coupling=COUPLING)
    scores_a = np.array([0.61, 0.58, 0.66, 0.62])
    scores_b = np.array([0.70, 0.69, 0.72, 0.68])

    out = paired_difference(control, treated, scores_a, scores_b)
    assert isinstance(out, float)
    assert out == np.mean(scores_a) - np.mean(scores_b)
    assert log == [("beta_kl=0.0", "beta_kl=0.02")]

    report = check_divergence(control, treated)
    assert report.ok
    assert report.void is None
    assert [d.key for d in report.contrast_realised] == ["beta_kl"]
    assert report.n_keys == 18


# ---------------------------------------------------------------------------
# The same gate where divergence actually happens: two records, after the fact
# ---------------------------------------------------------------------------


def _synthetic_run(run_id: str, *, sha: str, framework_version: str, config_hash: str) -> Run:
    return Run(
        id=RunID(run_id),
        kind="train",
        components={
            Component.GRADER: ComponentRef(
                name="rm-7b", kind="neural", substrate=Substrate.NEURAL_SCALAR, version="v3"
            ),
            Component.POLICY: ComponentRef(
                name="qwen3-4b", kind="neural", substrate=Substrate.NEURAL_GEN
            ),
        },
        access={Component.GRADER: Access.RECORD, Component.POLICY: Access.RECORD},
        regime=RegimeDeclaration(),
        steps=InMemoryStepStream([]),
        coupling=COUPLING,
        lineage=RunLineage(
            git_sha=sha,
            config_hash=config_hash,
            framework="trl",
            framework_version=framework_version,
            created_at=f"2026-08-01T09:0{len(run_id)}:00",
        ),
        sampling_policy=RecordSamplingPolicy(),
    )


def test_two_records_that_ran_on_different_framework_versions_are_void(tmp_path):
    """The failure that survives a correct plan: the second arm was submitted after an upgrade."""
    control = _synthetic_run(
        "run:control", sha="a1b2c3", framework_version="1.9.2", config_hash="cfg:beta0"
    )
    treated = _synthetic_run(
        "run:treated", sha="a1b2c3", framework_version="1.9.3", config_hash="cfg:beta002"
    )
    writer = RecordWriter(tmp_path)
    writer.write(control)
    writer.write(treated)

    back_control = RecordReader(tmp_path, "run:control").run()
    back_treated = RecordReader(tmp_path, "run:treated").run()

    report = check_runs(
        back_control,
        back_treated,
        contrast=("lineage.config_hash",),
        labels=("control", "treated"),
    )

    assert not report.ok
    assert report.keys == ("lineage.framework_version",)
    assert "'1.9.2' against '1.9.3'" in report.detail
    assert report.void is not None
    assert report.void.reason is VoidReason.ARM_DIVERGENCE
    assert "lineage.framework_version" in report.void.remedy


def test_two_records_that_differ_only_in_the_contrast_are_readable(tmp_path):
    control = _synthetic_run(
        "run:control", sha="a1b2c3", framework_version="1.9.2", config_hash="cfg:beta0"
    )
    treated = _synthetic_run(
        "run:treated", sha="a1b2c3", framework_version="1.9.2", config_hash="cfg:beta002"
    )
    writer = RecordWriter(tmp_path)
    writer.write(control)
    writer.write(treated)

    report = check_runs(
        RecordReader(tmp_path, "run:control").run(),
        RecordReader(tmp_path, "run:treated").run(),
        contrast=("lineage.config_hash",),
    )
    assert report.ok
    assert report.void is None
    assert [d.key for d in report.contrast_realised] == ["lineage.config_hash"]


# ---------------------------------------------------------------------------
# Common random numbers, which are the other half of the package
# ---------------------------------------------------------------------------


def test_the_two_arms_see_the_same_prompts_in_the_same_order():
    control, treated = arms(BASE_CONFIG, contrast={"beta_kl": [0.0, 0.02]}, coupling=COUPLING)

    assert control.prompt_order(64) == treated.prompt_order(64)
    assert control.seed_for("sampling", index=17) == treated.seed_for("sampling", index=17)
    # And the coupling is on the record, so a reader can see the comparison was paired.
    assert control.coupling.common_random_numbers is True


def test_declining_the_coupling_is_a_declaration_rather_than_a_default():
    control, treated = arms(
        BASE_CONFIG, contrast={"beta_kl": [0.0, 0.02]}, coupling=independent_seeds(2026)
    )
    assert control.prompt_order(64) != treated.prompt_order(64)
    assert control.coupling.common_random_numbers is False
    # Both arms still declare the same coupling, so this is not itself a divergence.
    assert check_divergence(control, treated).ok


def test_the_whole_arm_set_is_gated_and_its_voids_are_keyed_by_pair():
    """A three-arm sweep with one bad arm voids the pairs it is in and no others."""
    plan = arms(BASE_CONFIG, contrast={"beta_kl": [0.0, 0.02, 0.05]}, coupling=COUPLING)
    dirty = ArmSet(
        arms=(plan[0], plan[1], plan[2].overridden({"rollout.k": 16})),
        contrast_keys=plan.contrast_keys,
        coupling=plan.coupling,
    )

    voids = dirty.voids()
    assert set(voids) == {"beta_kl=0.0 vs beta_kl=0.05", "beta_kl=0.02 vs beta_kl=0.05"}
    assert all("rollout.k 8 against 16" in v.detail for v in voids.values())
    assert not dirty.ok

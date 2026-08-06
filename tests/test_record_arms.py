"""Arms, coupling, common random numbers, and the arm-divergence gate.

Three things are being pinned here and they fail in different ways.

The constructor: `arms()` has to produce configurations that differ in the declared contrast and in
nothing else, including through nesting, and it has to refuse the ways of asking for that which do
not mean it. Most of those refusals are for typos, because a mistyped contrast key is the failure
that produces a plausible number rather than an error.

The coupling: common random numbers are a claim about bit-identical draws across arms, so the tests
assert the seeds are equal rather than that a flag is set, and one test measures the variance
reduction the whole design is justified by.

The gate: a void has to name the diverging key and print both values. Every assertion about the
void checks the strings, because a void whose detail is "the arms diverged" is `inconclusive`
spelled differently and would pass a test that only checked the reason enum.
"""

from __future__ import annotations

import math
import string

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

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
    ABSENT,
    Arm,
    ArmSet,
    ArmSpecError,
    CouplingSpec,
    apply_overrides,
    arm_from_run,
    arms,
    check_divergence,
    check_runs,
    flatten_config,
    guard_comparison,
    independent_seeds,
    run_material,
    shared_seeds,
)
from reward_lens.record.schema import decode_foreign, encode_foreign
from reward_lens.studies.void import VoidReason

BASE = {
    "beta_kl": 0.0,
    "optim": {"lr": 1e-6, "warmup": 100, "clip": 0.2},
    "data": {"dataset": "gsm8k", "k": 8, "shuffle": True},
    "verifiers": ["math_verify"],
}


# ---------------------------------------------------------------------------
# Flattening, which is what makes a remedy nameable
# ---------------------------------------------------------------------------


def test_flatten_gives_the_dotted_paths_a_remedy_can_name():
    flat = flatten_config(BASE)
    assert flat["optim.lr"] == 1e-6
    assert flat["data.dataset"] == "gsm8k"
    # A list is a leaf. Indexing into it would name `verifiers[0]`, which is not a key any
    # configuration file has, so the whole list is the unit of difference.
    assert flat["verifiers"] == ["math_verify"]
    assert len(flat) == 8


def test_an_empty_section_survives_flattening_as_a_leaf():
    """Dropping every key under a section is a difference, so the section cannot vanish.

    Both halves are named: the empty section on one side and the key that only the other side
    has. That is two lines about one edit and it is the right two lines, because "the section is
    empty here" and "this key exists there" are the two things a reader has to reconcile.
    """
    assert flatten_config({"lora": {}, "lr": 1.0}) == {"lora": {}, "lr": 1.0}
    reported = check_divergence(
        Arm(label="a", config={"lora": {}}),
        Arm(label="b", config={"lora": {"r": 8}}),
    )
    assert not reported.ok
    assert reported.keys == ("lora", "lora.r")
    assert "lora {} against <absent>" in reported.detail
    assert "lora.r <absent> against 8" in reported.detail


def test_a_key_reachable_two_ways_raises_rather_than_letting_one_win():
    with pytest.raises(ArmSpecError, match="two ways"):
        flatten_config({"a.b": 1, "a": {"b": 2}})


def test_overriding_through_a_scalar_raises_instead_of_overwriting_it():
    with pytest.raises(ArmSpecError, match="holds int rather than a section"):
        apply_overrides({"optim": 3}, {"optim.lr": 1e-6})


def test_an_override_does_not_reach_into_the_configuration_it_copied():
    out = apply_overrides(BASE, {"optim.lr": 5e-6})
    assert out["optim"]["lr"] == 5e-6
    assert BASE["optim"]["lr"] == 1e-6


# ---------------------------------------------------------------------------
# arms(): the constructor, and the six ways of asking for something it will not build
# ---------------------------------------------------------------------------


def test_two_arms_differ_in_the_declared_contrast_and_in_nothing_else():
    control, treated = arms(BASE, contrast={"beta_kl": [0.0, 0.02]}, coupling=shared_seeds())

    assert control.config["beta_kl"] == 0.0
    assert treated.config["beta_kl"] == 0.02
    assert control.config["optim"] == treated.config["optim"] == BASE["optim"]
    assert control.label == "beta_kl=0.0"
    assert treated.label == "beta_kl=0.02"

    report = check_divergence(control, treated)
    assert report.ok
    # 8 configuration keys plus the six coupling fields that are not free-text and not `shared`.
    assert report.n_keys == 14
    assert [d.key for d in report.contrast_realised] == ["beta_kl"]


def test_the_contrast_reaches_into_a_nested_section():
    a, b = arms(BASE, contrast={"optim.lr": [1e-6, 2e-6]})
    assert a.config["optim"]["lr"] == 1e-6
    assert b.config["optim"]["lr"] == 2e-6
    assert a.config["optim"]["warmup"] == b.config["optim"]["warmup"] == 100
    assert check_divergence(a, b).ok


def test_common_random_numbers_are_what_you_get_when_you_say_nothing():
    """The default is the point of the API. Free variance reduction should not need an argument."""
    a, b = arms(BASE, contrast={"beta_kl": [0.0, 0.02]})
    assert a.coupling.common_random_numbers is True
    assert a.coupling.shared_prompt_order is True
    assert a.seed_for("sampling") == b.seed_for("sampling")


def test_a_mistyped_contrast_key_raises_and_points_at_the_key_it_meant():
    with pytest.raises(ArmSpecError) as exc:
        arms(BASE, contrast={"lr": [1e-6, 2e-6]})
    assert "'optim.lr'" in str(exc.value)
    assert "allow_new_keys" in str(exc.value)


def test_a_genuinely_new_key_is_allowed_when_it_is_asked_for():
    a, b = arms(BASE, contrast={"reward.shaping": [0.0, 0.5]}, allow_new_keys=True)
    assert a.config["reward"]["shaping"] == 0.0
    assert b.config["reward"]["shaping"] == 0.5
    assert check_divergence(a, b).ok


def test_a_contrast_with_one_value_is_not_a_comparison():
    with pytest.raises(ArmSpecError, match="at least two"):
        arms(BASE, contrast={"beta_kl": [0.02]})


def test_a_bare_value_is_a_change_to_the_base_configuration_not_a_contrast():
    with pytest.raises(ArmSpecError, match="one value per arm"):
        arms(BASE, contrast={"beta_kl": 0.02})


def test_no_contrast_at_all_raises():
    with pytest.raises(ArmSpecError, match="declared contrast"):
        arms(BASE, contrast={})


def test_two_arms_with_the_same_contrast_value_would_measure_noise_and_raise():
    with pytest.raises(ArmSpecError, match="identical configurations"):
        arms(BASE, contrast={"beta_kl": [0.02, 0.02]})


def test_unequal_paired_lists_raise_and_name_the_crossed_design():
    with pytest.raises(ArmSpecError) as exc:
        arms(BASE, contrast={"beta_kl": [0.0, 0.02], "optim.lr": [1e-6, 2e-6, 4e-6]})
    assert "crossed" in str(exc.value)
    assert "beta_kl has 2" in str(exc.value)


def test_a_paired_design_reads_the_contrast_down_the_columns():
    got = arms(
        BASE,
        contrast={"beta_kl": [0.0, 0.02], "optim.lr": [1e-6, 2e-6]},
        labels=("control", "treated"),
    )
    assert len(got) == 2
    assert got.by_label("treated").config["beta_kl"] == 0.02
    assert got.by_label("treated").config["optim"]["lr"] == 2e-6
    assert got.ok


def test_a_crossed_design_gives_the_two_by_two():
    got = arms(
        BASE,
        contrast={"beta_kl": [0.0, 0.02], "optim.lr": [1e-6, 2e-6]},
        design="crossed",
    )
    assert len(got) == 4
    assert got.labels == (
        "beta_kl=0.0,optim.lr=1e-06",
        "beta_kl=0.0,optim.lr=2e-06",
        "beta_kl=0.02,optim.lr=1e-06",
        "beta_kl=0.02,optim.lr=2e-06",
    )
    assert got.ok  # every pair differs only in the two declared keys


def test_the_wrong_number_of_labels_raises():
    with pytest.raises(ArmSpecError, match="1 label"):
        arms(BASE, contrast={"beta_kl": [0.0, 0.02]}, labels=("only-one",))


def test_a_set_of_one_arm_is_not_a_comparison():
    with pytest.raises(ArmSpecError, match="at least two arms"):
        ArmSet(arms=(Arm(label="solo", config=BASE),), contrast_keys=())


def test_an_arm_without_a_label_raises_because_the_void_has_to_name_it():
    with pytest.raises(ArmSpecError, match="needs a label"):
        Arm(label="  ", config=BASE)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_the_arm_id_is_content_derived_and_recomputed_rather_than_stored():
    a, b = arms(BASE, contrast={"beta_kl": [0.0, 0.02]})
    assert a.id != b.id
    assert a.id == Arm(label="other-name", config=a.config, coupling=a.coupling).id

    before = a.id
    a.config["optim"]["lr"] = 9e-9  # a frozen dataclass does not freeze the dict inside it
    assert a.id != before, "a stored hash would have gone stale here, which is worse than none"


def test_the_config_hash_is_the_one_that_goes_on_the_lineage():
    a, b = arms(BASE, contrast={"beta_kl": [0.0, 0.02]})
    assert a.config_hash.startswith("cfg:")
    assert a.config_hash != b.config_hash
    # The coupling is not in it: the lineage's config hash is about the configuration.
    assert (
        a.config_hash == Arm(label="x", config=a.config, coupling=independent_seeds()).config_hash
    )


def test_a_configuration_holding_something_unstorable_says_which_key():
    """A live object in a configuration should fail by name, not from inside the hasher.

    `Device` stands in for what people actually put there: a device handle, a compiled module, an
    open dataset. What they have in common is that the canonical form cannot reach into them, so
    the arm has no content identity and the message has to say which key is responsible.
    """

    class Device:
        __slots__ = ()

    arm = Arm(label="a", config={"lr": 1e-6, "device": Device()})
    with pytest.raises(ArmSpecError) as exc:
        arm.config_hash
    assert "device (Device)" in str(exc.value)


# ---------------------------------------------------------------------------
# Coupling and common random numbers
# ---------------------------------------------------------------------------


def test_a_shared_stream_gives_both_arms_the_same_seed_and_an_unshared_one_does_not():
    coupled = shared_seeds(7)
    assert coupled.seed_for("sampling", arm="A") == coupled.seed_for("sampling", arm="B")

    free = independent_seeds(7)
    assert free.seed_for("sampling", arm="A") != free.seed_for("sampling", arm="B")


def test_streams_and_steps_are_separated_even_when_they_are_shared():
    coupled = shared_seeds(7)
    assert coupled.seed_for("sampling") != coupled.seed_for("init")
    assert coupled.seed_for("sampling", index=0) != coupled.seed_for("sampling", index=1)


def test_derived_seeds_fit_where_a_framework_will_take_them():
    coupled = shared_seeds(7)
    for stream in coupled.streams:
        seed = coupled.seed_for(stream)
        assert 0 <= seed < 2**63


def test_a_coupling_says_which_streams_it_claims_to_drive():
    """`covers` is the claim and `shares` is the behaviour, and a framework can break the claim."""
    partial = shared_seeds(7, streams=("prompt_order",))
    assert partial.covers("prompt_order") is True
    assert partial.covers("dropout") is False
    # A stream it does not claim still derives a seed, because refusing to would leave the caller
    # with nothing; what the declaration changes is what the record says was coupled.
    assert partial.shares("dropout") is True
    assert partial.seed_for("dropout") > 0


def test_an_unshared_stream_without_an_arm_raises_rather_than_recoupling_silently():
    """The dangerous default: no arm means no arm in the hash, which is coupling by accident."""
    with pytest.raises(ArmSpecError, match="needs the arm"):
        independent_seeds(7).seed_for("sampling")


def test_the_prompt_order_and_the_sampling_noise_are_separately_declarable():
    half = CouplingSpec(seed=5, common_random_numbers=False, shared_prompt_order=True)
    assert half.prompt_order(8, arm="A") == half.prompt_order(8, arm="B")
    assert half.seed_for("sampling", arm="A") != half.seed_for("sampling", arm="B")
    assert half.shares("prompt_order") is True
    assert half.shares("sampling") is False


def test_the_shared_prompt_order_is_a_permutation_and_it_is_the_same_one():
    coupled = shared_seeds(11)
    order_a = coupled.prompt_order(12, arm="A")
    order_b = coupled.prompt_order(12, arm="B")
    assert sorted(order_a) == list(range(12))
    assert order_a == order_b
    assert coupled.prompt_order(12, arm="A", epoch=1) != order_a


def test_common_random_numbers_reduce_the_variance_of_the_arm_difference():
    """The claim the default rests on, measured rather than asserted.

    Both arms score 200 prompts whose difficulty is drawn from the coupled stream. Under common
    random numbers the two arms see the same difficulties, so the prompt-level variance cancels in
    the paired difference and what is left is the curvature of the response. Under independent
    seeds it does not cancel.

    Measured over 400 replicates at seed 11: variance 7.11e-07 coupled against 5.02e-04
    independent, a factor of 706, with the two means agreeing to 3.3e-04 (-0.05082 against
    -0.05049). The estimate does not move; only its variance does. The assertion is at a factor of
    20 rather than 706 so it is not a tripwire on the numpy version.
    """

    def score(coupling: CouplingSpec, arm: str, effect: float, replicate: int) -> float:
        rng = coupling.rng("sampling", arm=arm, index=replicate)
        difficulty = rng.normal(0.0, 1.0, size=200)
        noise = rng.normal(0.0, 0.3, size=200)
        return float(np.mean(1.0 / (1.0 + np.exp(-(effect - difficulty + noise)))))

    def differences(coupling: CouplingSpec) -> list[float]:
        return [
            score(coupling, "control", 0.0, r) - score(coupling, "treated", 0.25, r)
            for r in range(400)
        ]

    coupled = differences(shared_seeds(11))
    free = differences(independent_seeds(11))

    assert np.var(coupled) * 20 < np.var(free)
    assert abs(np.mean(coupled) - np.mean(free)) < 5e-3


def test_a_coupling_seed_that_cannot_be_recorded_raises():
    with pytest.raises(ArmSpecError, match="non-negative int"):
        CouplingSpec(seed=-1)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def _diverged() -> tuple[Arm, Arm, tuple[str, ...]]:
    """Two arms that were built correctly and then launched with a stale override."""
    plan = arms(BASE, contrast={"beta_kl": [0.0, 0.02]}, coupling=shared_seeds(3))
    control, treated = plan
    return control, treated.overridden({"optim.lr": 2e-6}), plan.contrast_keys


def test_a_divergence_names_the_key_and_prints_both_values():
    control, treated, contrast = _diverged()
    report = check_divergence(control, treated, contrast=contrast)

    assert not report.ok
    assert report.keys == ("optim.lr",)
    assert report.diverging[0].left == 1e-6
    assert report.diverging[0].right == 2e-6
    assert "optim.lr 1e-06 against 2e-06" in report.detail


def test_the_void_is_condition_four_of_the_declared_seven_and_not_a_new_one():
    control, treated, contrast = _diverged()
    void = check_divergence(control, treated, contrast=contrast).void

    assert void is not None
    assert void.reason is VoidReason.ARM_DIVERGENCE
    assert "optim.lr" in void.detail
    assert void.remedy.startswith("Set optim.lr to the same value")


def test_a_clean_comparison_has_no_void_and_still_reports_its_numbers():
    control, treated = arms(BASE, contrast={"beta_kl": [0.0, 0.02]})
    report = check_divergence(control, treated)
    assert report.void is None
    assert report.ok
    assert report.n_keys == 14
    assert "differ only in: beta_kl 0.0 against 0.02" in report.render()


def test_the_refusal_carries_the_same_facts_for_a_measurement_path():
    control, treated, contrast = _diverged()
    refusal = check_divergence(control, treated, contrast=contrast).refusal("PairedDifference")

    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.VOID
    assert refusal.statistics["diverging"]["optim.lr"] == ["1e-06", "2e-06"]
    assert refusal.statistics["n_keys_compared"] == 14
    assert "optim.lr" in refusal.remedy


def test_asking_a_readable_comparison_for_a_refusal_is_a_caller_bug():
    control, treated = arms(BASE, contrast={"beta_kl": [0.0, 0.02]})
    with pytest.raises(ArmSpecError, match="readable"):
        check_divergence(control, treated).refusal("PairedDifference")


def test_a_key_one_arm_has_and_the_other_does_not_is_a_divergence():
    left = Arm(label="a", config={"lr": 1e-6, "lora": {"r": 8}})
    right = Arm(label="b", config={"lr": 1e-6})
    report = check_divergence(left, right, contrast=())

    assert report.keys == ("lora.r",)
    assert report.diverging[0].right is ABSENT
    assert "lora.r 8 against <absent>" in report.detail


def test_a_difference_in_adoption_step_is_reported_and_is_not_a_divergence():
    """Staggered adoption: the timing is the treatment, so it is not a configuration difference.

    The module implements none of the corrected estimators. What it does is refuse to let the
    timing disappear: the report says the two arms adopted at different steps and says what that
    means for the estimator, which is the part a two-way fixed-effects regression would swallow.
    """
    plan = arms(
        BASE,
        contrast={"beta_kl": [0.0, 0.02]},
        coupling=shared_seeds(3),
        adopted_at=(1000, 3000),
    )
    control, treated = plan
    report = check_divergence(control, treated)

    assert report.ok
    assert report.void is None
    assert report.adoption == (1000, 3000)
    assert report.staggered is True
    assert "different steps in the two arms (1000 and 3000)" in report.render()
    assert "staggered-adoption estimator" in report.render()


def test_arms_that_were_one_thing_throughout_are_not_staggered():
    control, treated = arms(BASE, contrast={"beta_kl": [0.0, 0.02]})
    report = check_divergence(control, treated)
    assert report.adoption == (None, None)
    assert report.staggered is False
    assert "staggered" not in report.render()


def test_a_declared_contrast_that_did_not_take_is_void_condition_eight():
    """The mirror image: the override never reached the trainer, so there is no contrast at all.

    This is the worse of the two failures. An arm divergence announces itself; a contrast that
    never applied produces a comparison that runs, returns a tidy null, and reads as a result.
    """
    from reward_lens.studies.void import VoidReason

    left = Arm(label="a", config={"beta_kl": 0.0, "lr": 1e-6})
    right = Arm(label="b", config={"beta_kl": 0.0, "lr": 1e-6})
    report = check_divergence(left, right, contrast=("beta_kl",))

    assert not report.ok
    assert report.inert
    assert report.contrast_inert == ("beta_kl",)
    assert report.reason is VoidReason.CONTRAST_INERT

    void = report.void
    assert void is not None
    assert void.reason is VoidReason.CONTRAST_INERT
    assert "hold the same value in both" in void.detail
    assert "reached the trainer" in void.remedy
    assert "VOID(contrast_inert)" in report.render()

    refusal = report.refusal("test.instrument")
    assert refusal.statistics["void_condition"] == "contrast_inert"
    assert refusal.statistics["contrast_inert"] == ["beta_kl"]


def test_a_crossed_pair_that_holds_one_factor_fixed_is_still_a_comparison():
    """Condition 8 is "nothing in the contrast moved", not "something in it did not".

    In a two-by-two the pair that varies the learning rate holds `beta_kl` fixed by construction.
    Voiding that pair would void half of every factorial design.
    """
    left = Arm(label="a", config={"beta_kl": 0.0, "lr": 1e-6})
    right = Arm(label="b", config={"beta_kl": 0.0, "lr": 2e-6})
    report = check_divergence(left, right, contrast=("beta_kl", "lr"))

    assert report.ok
    assert not report.inert
    assert report.contrast_inert == ("beta_kl",)
    assert report.void is None


def test_arrays_and_nans_in_a_configuration_compare_the_way_a_reader_expects():
    same = Arm(label="a", config={"stages": np.array([0, 100, 400]), "floor": math.nan})
    also = Arm(label="b", config={"stages": np.array([0, 100, 400]), "floor": math.nan})
    other = Arm(label="c", config={"stages": np.array([0, 100, 500]), "floor": math.nan})

    assert check_divergence(same, also, contrast=()).ok
    assert check_divergence(same, other, contrast=()).keys == ("stages",)


def test_a_different_coupling_is_a_divergence_like_any_other_key():
    left = Arm(label="a", config={"lr": 1e-6}, coupling=shared_seeds(1, engine="vllm@0.11.2"))
    right = Arm(label="b", config={"lr": 1e-6}, coupling=shared_seeds(1, engine="vllm@0.11.3"))
    report = check_divergence(left, right, contrast=())

    assert report.keys == ("coupling.engine",)
    assert "'vllm@0.11.2' against 'vllm@0.11.3'" in report.detail


def test_the_guard_runs_the_comparison_when_the_arms_are_matched_and_not_when_they_are_not():
    calls: list[str] = []

    @guard_comparison
    def mean_difference(left: Arm, right: Arm, a: float, b: float) -> float:
        calls.append(f"{left.label}|{right.label}")
        return a - b

    control, treated = arms(BASE, contrast={"beta_kl": [0.0, 0.02]})
    assert mean_difference(control, treated, 0.62, 0.55) == pytest.approx(0.07)
    assert len(calls) == 1

    out = mean_difference(control, treated.overridden({"optim.lr": 2e-6}), 0.62, 0.55)
    assert isinstance(out, Refusal)
    assert out.reason is RefusalReason.VOID
    assert len(calls) == 1, "the comparison must not run on arms that diverged"


def test_the_set_checks_every_pair_and_keys_its_voids_by_pair():
    plan = arms(BASE, contrast={"beta_kl": [0.0, 0.02, 0.05]})
    assert len(plan.check()) == 3
    assert plan.ok
    assert plan.voids() == {}

    dirty = ArmSet(
        arms=(plan[0], plan[1], plan[2].overridden({"optim.warmup": 200})),
        contrast_keys=plan.contrast_keys,
        coupling=plan.coupling,
    )
    voids = dirty.voids()
    assert set(voids) == {"beta_kl=0.0 vs beta_kl=0.05", "beta_kl=0.02 vs beta_kl=0.05"}
    assert all(v.reason is VoidReason.ARM_DIVERGENCE for v in voids.values())
    assert all("optim.warmup" in v.detail for v in voids.values())


def test_sibling_run_ids_are_what_goes_on_the_run():
    plan = arms(BASE, contrast={"beta_kl": [0.0, 0.02]})
    bound = ArmSet(
        arms=(plan[0].with_run_id("run:a"), plan[1].with_run_id("run:b")),
        contrast_keys=plan.contrast_keys,
        coupling=plan.coupling,
    )
    assert bound.sibling_run_ids(bound[0]) == ("run:b",)
    assert bound.sibling_run_ids(bound[1]) == ("run:a",)


def test_an_unlaunched_sibling_is_left_out_rather_than_filled_in():
    plan = arms(BASE, contrast={"beta_kl": [0.0, 0.02]})
    half = ArmSet(
        arms=(plan[0].with_run_id("run:a"), plan[1]),
        contrast_keys=plan.contrast_keys,
        coupling=plan.coupling,
    )
    assert half.sibling_run_ids(half[0]) == ()


# ---------------------------------------------------------------------------
# The gate on two records rather than two plans
# ---------------------------------------------------------------------------


def _run(rid: str, *, sha: str = "abc123", coupling: CouplingSpec | None = None) -> Run:
    return Run(
        id=RunID(rid),
        kind="train",
        components={
            Component.GRADER: ComponentRef(
                name="rm-7b", kind="neural", substrate=Substrate.NEURAL_SCALAR
            )
        },
        access={Component.GRADER: Access.RECORD},
        regime=RegimeDeclaration(),
        steps=InMemoryStepStream([]),
        coupling=coupling if coupling is not None else shared_seeds(3),
        lineage=RunLineage(
            git_sha=sha,
            config_hash=f"cfg:{rid}",
            framework="trl",
            framework_version="1.9.2",
            created_at=f"2026-08-01T00:00:0{len(rid)}",
        ),
        sampling_policy=RecordSamplingPolicy(),
    )


def test_two_records_that_were_planned_alike_and_ran_differently_are_void():
    report = check_runs(
        _run("run:a"),
        _run("run:b", sha="def456"),
        contrast=("lineage.config_hash",),
        labels=("control", "treated"),
    )
    assert not report.ok
    assert report.keys == ("lineage.git_sha",)
    assert "'abc123' against 'def456'" in report.detail


def test_the_launch_timestamp_is_not_a_divergence():
    """Two arms are launched seconds apart by construction, so `created_at` says nothing."""
    material = run_material(_run("run:a"))
    assert "lineage.created_at" not in material
    assert "lineage.git_sha" in material

    report = check_runs(_run("run:a"), _run("run:bb"), contrast=("lineage.config_hash",))
    assert report.ok


def test_a_record_with_no_coupling_declares_nothing_rather_than_assuming_it():
    run = _run("run:a")
    object.__setattr__(run, "coupling", None)
    arm = arm_from_run(run)
    assert arm.coupling == CouplingSpec()
    assert arm.run_id == "run:a"


def test_the_engines_that_actually_served_the_rollouts_are_only_walked_when_asked():
    run = _run("run:a")
    assert "engines" not in run_material(run)
    assert run_material(run, engines=True)["engines"] == ()


# ---------------------------------------------------------------------------
# Round-tripping, because a coupling that does not survive the writer is a comment
# ---------------------------------------------------------------------------


def test_the_coupling_round_trips_through_the_kernel_codec():
    spec = shared_seeds(7, engine="vllm@0.11.2", shared={"prompt_set": "ps:abc"}, note="baseline")
    back = decode_foreign(encode_foreign(spec))
    assert back == spec
    assert isinstance(back.streams, tuple)
    assert isinstance(back, CouplingSpec)


def test_an_arm_set_round_trips_through_the_kernel_codec():
    plan = arms(BASE, contrast={"beta_kl": [0.0, 0.02]}, coupling=shared_seeds(7))
    back = decode_foreign(encode_foreign(plan))
    assert isinstance(back, ArmSet)
    assert back.labels == plan.labels
    assert back[0].id == plan[0].id
    assert back.ok


def test_a_divergence_report_round_trips_so_a_void_can_be_stored_with_its_numbers():
    control, treated, contrast = _diverged()
    report = check_divergence(control, treated, contrast=contrast)
    back = decode_foreign(encode_foreign(report))

    assert back == report
    assert back.keys == ("optim.lr",)
    assert back.void is not None
    assert back.void.reason is VoidReason.ARM_DIVERGENCE


def test_an_absent_key_survives_the_codec_as_absent_rather_than_as_a_value():
    """`<absent>` is a registered type, so a stored void does not turn a missing key into a string."""
    report = check_divergence(
        Arm(label="a", config={"lr": 1e-6, "lora": {"r": 8}}),
        Arm(label="b", config={"lr": 1e-6}),
        contrast=(),
    )
    back = decode_foreign(encode_foreign(report))
    assert back.diverging[0].right == ABSENT
    assert back.diverging[0].right is not None


def test_the_coupling_round_trips_through_the_writer_and_the_reader(tmp_path):
    spec = shared_seeds(7, engine="vllm@0.11.2", shared={"prompt_set": "ps:abc"})
    run = _run("run:a", coupling=spec)
    RecordWriter(tmp_path).write(run)

    back = RecordReader(tmp_path, "run:a").run()
    assert back.coupling == spec
    assert back.coupling.seed_for("sampling") == spec.seed_for("sampling")


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

_KEYS = st.text(alphabet=string.ascii_lowercase + "_", min_size=1, max_size=6)
_LEAVES = st.one_of(
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.text(alphabet=string.printable[:62], max_size=8),
    st.none(),
)
_CONFIGS = st.dictionaries(
    _KEYS,
    st.one_of(_LEAVES, st.dictionaries(_KEYS, _LEAVES, min_size=1, max_size=3)),
    min_size=2,
    max_size=5,
)


@given(config=_CONFIGS, data=st.data())
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_any_difference_outside_the_contrast_produces_a_void_naming_that_key(config, data):
    """The property the gate exists for, over arbitrary configurations.

    Two keys are drawn from the flattened configuration: one is declared as the contrast, the
    other is changed in one arm only. Whatever the configuration looks like, the second change has
    to come back as a void naming that key, with both values on it.
    """
    flat = flatten_config(config)
    keys = sorted(k for k in flat if not isinstance(flat[k], dict))
    if len(keys) < 2:
        return
    contrast_key = data.draw(st.sampled_from(keys))
    other_key = data.draw(st.sampled_from([k for k in keys if k != contrast_key]))

    original = flat[contrast_key]
    plan = arms(
        config,
        contrast={contrast_key: [original, _distinct_from(original)]},
        coupling=shared_seeds(3),
    )
    control, treated = plan
    assert check_divergence(control, treated).ok

    drifted = treated.overridden({other_key: _distinct_from(flat[other_key])})
    report = check_divergence(control, drifted, contrast=plan.contrast_keys)

    assert not report.ok
    assert other_key in report.keys
    assert other_key in report.detail
    assert report.void is not None
    assert report.void.reason is VoidReason.ARM_DIVERGENCE
    assert other_key in report.remedy or "more" in report.remedy


@given(config=_CONFIGS, data=st.data())
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_arms_always_passes_its_own_gate(config, data):
    """Whatever `arms()` builds, it differs only in the contrast it declared. That is the claim."""
    flat = flatten_config(config)
    keys = sorted(k for k in flat if not isinstance(flat[k], dict))
    if not keys:
        return
    contrast_key = data.draw(st.sampled_from(keys))
    values = [flat[contrast_key], _distinct_from(flat[contrast_key])]

    plan = arms(config, contrast={contrast_key: values})
    assert plan.ok
    assert plan.voids() == {}
    assert all(r.contrast_realised for r in plan.check())


def _distinct_from(value):
    """A value of the same shape that is not equal to this one."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, str):
        return value + "x"
    return "not-none"

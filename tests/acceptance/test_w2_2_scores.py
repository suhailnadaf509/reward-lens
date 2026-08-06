"""Acceptance: the `ScoreTree`, and the counterfactual it makes free.

The specification's clause is one sentence: *a tree with an override node re-evaluates correctly
with the node disabled, on recorded leaves, with no grader calls.* All four of its parts are
asserted separately here, because three of them are easy to pass by accident.

*A tree with an override node* is built from a real grader wrapped with `reward_lens.tap`, so the
leaves carry the call records the tap actually produced rather than hand-written ones. *Correctly*
is checked against values computed on paper and written into the test. *On recorded leaves* means
the counterfactual is run a second time against the trees that came back off disk through
`RecordWriter` and `RecordReader`, and the two runs agree to the last bit. *With no grader calls* is
asserted on a counter inside the grader itself and on the tap's own offered-record count, both of
which are read before and after and must not have moved.

Three more things are asserted because the score tree is the substrate for three catalogued
instruments and each of them would be silently wrong without one of them: the dead-zone fraction
I3 reports, the silent-zero census B4 reports, and the refusal B3 owes a record whose per-leaf
scores were never kept.
"""

from __future__ import annotations

import math
import random

import pytest

from reward_lens.core.envelope import RegimeCondition
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import Access, Component, Substrate
from reward_lens.record import (
    ComponentRef,
    EstimatorSpec,
    Group,
    GroupID,
    GroupStats,
    InMemoryStepStream,
    OptimizerTelemetry,
    RecordReader,
    RecordSamplingPolicy,
    RecordWriter,
    RegimeDeclaration,
    Run,
    RunID,
    RunLineage,
    SamplingScheme,
    Step,
    TaskID,
    Turn,
    make_trajectory,
)
from reward_lens.record.scores import (
    GraderCallRef,
    GroupContext,
    GroupScores,
    Leaf,
    Override,
    PredicateRef,
    ScoreContext,
    WeightedSum,
    census,
    counterfactual,
    deadzone,
    evaluate,
    group_scores,
)
from reward_lens.tap import SimpleRun, instrument_grader

# The composition under test: a task reward of `correct + 0.25 * style`, with Kimi K3's override
# pinning the score to -1.0 for any rollout whose token count exceeds 1.5 times the task's budget.
OVER_BUDGET = PredicateRef(
    name="over_budget", feature="tokens", op=">", threshold_feature="b0", scale=1.5
)

CENTRED = EstimatorSpec(
    family="grpo",
    group_centred=True,
    std_normalised=False,
    degenerate_policy="skip",
    aggregation="token",
    loss_mask_policy="mask_environment",
)

# correct, style, tokens. Rollouts 1 and 3 are over the 150-token budget.
WORKED = [
    (1.0, 0.8, 100.0),
    (1.0, 0.4, 200.0),
    (0.0, 0.6, 90.0),
    (0.0, 0.2, 400.0),
]


class GraderFailed(RuntimeError):
    """What the grader raises when it cannot score. Named, so nothing has to catch broadly."""


class CountingGrader:
    """A composite grader that counts its own calls, so "no grader calls" is a measured claim."""

    def __init__(self, fail_every: int = 0) -> None:
        self.calls = 0
        self.fail_every = fail_every

    def __call__(self, *, response: str, correct: float, style: float) -> dict[str, float]:
        self.calls += 1
        if self.fail_every and self.calls % self.fail_every == 0:
            raise GraderFailed(f"malformed response at call {self.calls}")
        return {"correct": correct, "style": style}


def build_leaves(
    grader, run: SimpleRun, *, correct: float, style: float, response: str = "x"
) -> tuple[Leaf, Leaf]:
    """Score one rollout through the tap and turn what the tap recorded into leaves.

    The `except GraderFailed` is the whole point of the exercise: the exception is caught by name,
    the leaf records that the call failed, and the 0.0 the framework would have substituted is kept
    beside it rather than in place of it.
    """
    try:
        out = grader(response=response, correct=correct, style=style)
        failed = False
    except GraderFailed:
        out = {"correct": 0.0, "style": 0.0}
        failed = True
    calls = run.ring.drain()
    ref = GraderCallRef.from_call(calls[-1], cost_usd=0.0004) if calls else None
    if failed:
        return (
            Leaf("correct", out["correct"], ref, abstained=True),
            Leaf("style", out["style"], ref, abstained=True),
        )
    return Leaf("correct", out["correct"], ref), Leaf("style", out["style"], ref)


def k3_tree(correct_leaf: Leaf, style_leaf: Leaf) -> Override:
    return Override(
        name="length_override",
        condition=OVER_BUDGET,
        constant=-1.0,
        otherwise=WeightedSum("task", (correct_leaf, style_leaf), (1.0, 0.25)),
    )


def ctx_for(correct: float, tokens: float, *, accuracy: float, k: int, gid: str) -> ScoreContext:
    return ScoreContext(
        features={"tokens": tokens, "b0": 100.0, "correct": correct},
        group=GroupContext(accuracy=accuracy, k=k, id=gid),
    )


# ---------------------------------------------------------------------------
# The clause
# ---------------------------------------------------------------------------


def test_the_override_re_evaluates_correctly_with_the_node_disabled_and_calls_no_grader() -> None:
    grader = CountingGrader()
    run = SimpleRun(run_id="w22")
    tapped = instrument_grader(grader, run=run, name="composite")

    trees = []
    contexts = []
    for correct, style, tokens in WORKED:
        trees.append(k3_tree(*build_leaves(tapped, run, correct=correct, style=style)))
        contexts.append(ctx_for(correct, tokens, accuracy=0.5, k=4, gid="g0"))

    # Every leaf carries the call the tap actually recorded.
    assert grader.calls == 4
    for tree in trees:
        for leaf in (tree.otherwise.children[0], tree.otherwise.children[1]):
            assert leaf.grader_call is not None
            assert leaf.grader_call.grader == "composite"
            assert leaf.grader_call.outcome == "returned"
            assert leaf.grader_call.latency_s is not None and leaf.grader_call.latency_s >= 0.0

    calls_before = grader.calls
    offered_before = run.ring.stats().offered

    group = GroupScores(trees=tuple(trees), contexts=tuple(contexts), estimator=CENTRED, id="g0")
    got = counterfactual([group], {"length_override"})

    assert not isinstance(got, Refusal), got
    # Computed on paper: the task reward is 1.20, 1.10, 0.15, 0.05 and the override pins rollouts
    # 1 and 3 to -1.0. The group mean is -0.1625 with the override and 0.625 without it.
    assert got.scores_before == (1.2, -1.0, 0.15, -1.0)
    assert got.scores_after == pytest.approx((1.2, 1.1, 0.15, 0.05))
    assert got.advantages_before == pytest.approx((1.3625, -0.8375, 0.3125, -0.8375))
    assert got.advantages_after == pytest.approx((0.575, 0.475, -0.475, -0.575))
    assert got.n_sign_reversed == 2
    assert got.n_moved == 1
    assert got.sd_reference == pytest.approx(0.9160888330287625)

    # No grader calls. Not "few", not "cheap": none, on the grader's own counter and on the tap's.
    assert grader.calls == calls_before
    assert run.ring.stats().offered == offered_before


def test_the_same_counterfactual_off_disk_agrees_to_the_last_bit(tmp_path) -> None:
    """On recorded leaves means the leaves that came back off disk, not the ones still in memory."""
    grader = CountingGrader()
    run_handle = SimpleRun(run_id="w22-record")
    tapped = instrument_grader(grader, run=run_handle, name="composite")

    trajectories = []
    trees = []
    for i, (correct, style, tokens) in enumerate(WORKED):
        tree = k3_tree(*build_leaves(tapped, run_handle, correct=correct, style=style))
        trees.append(tree)
        trajectories.append(
            make_trajectory(
                id=f"traj:{i}",
                task_ref="task:0",
                turns=(Turn(index=0, role="assistant", text="answer"),),
                scores=tree,
                advantage=None,
                features={"tokens": tokens, "b0": 100.0, "correct": correct},
                policy_version="pv:0",
            )
        )
    group = Group(
        id=GroupID("grp:0"),
        task_ref=TaskID("task:0"),
        trajectories=tuple(trajectories),
        estimator=CENTRED,
        group_stats=GroupStats.from_scores(
            [
                evaluate(t, ctx_for(c, tk, accuracy=0.5, k=4, gid="g0"))
                for t, (c, _, tk) in zip(trees, WORKED)
            ],
            std_epsilon=1e-8,
        ),
    )
    run = Run(
        id=RunID("run:w22-acceptance"),
        kind="train",
        components={
            Component.GRADER: ComponentRef(
                name="composite", kind="rubric", substrate=Substrate.COMPOSITE
            )
        },
        access={Component.GRADER: Access.RECORD},
        regime=RegimeDeclaration(
            declared={RegimeCondition.STATIONARY_GRADER: True}, declared_by="the test"
        ),
        steps=InMemoryStepStream(
            [Step(index=0, groups=(group,), schedule={"K": 4.0}, optimizer=OptimizerTelemetry())]
        ),
        lineage=RunLineage(git_sha="0" * 40, framework="none", library_version="3.0.0a1"),
        sampling_policy=RecordSamplingPolicy(
            scheme=SamplingScheme.UNIFORM, rate=1.0, population=1, recorded=1
        ),
    )
    RecordWriter(tmp_path).write(run)

    in_memory = counterfactual(
        [
            GroupScores(
                trees=tuple(trees),
                contexts=tuple(
                    ctx_for(c, tk, accuracy=0.5, k=4, gid="grp:0") for c, _, tk in WORKED
                ),
                estimator=CENTRED,
                id="grp:0",
            )
        ],
        {"length_override"},
    )

    back = RecordReader(tmp_path, "run:w22-acceptance").run()
    (step,) = list(back.steps)
    (read_group,) = step.groups
    read_tree = read_group.trajectories[0].scores
    assert isinstance(read_tree, Override)
    assert read_tree == trees[0]
    assert read_tree.otherwise.weights == (1.0, 0.25)
    assert isinstance(read_tree.otherwise.children[0].grader_call, GraderCallRef)

    calls_before = grader.calls
    from_disk = counterfactual([group_scores(read_group)], {"length_override"})
    assert grader.calls == calls_before

    assert not isinstance(from_disk, Refusal), from_disk
    assert not isinstance(in_memory, Refusal), in_memory
    assert from_disk.advantages_before == in_memory.advantages_before
    assert from_disk.advantages_after == in_memory.advantages_after
    assert from_disk.n_sign_reversed == in_memory.n_sign_reversed
    # The group's accuracy came off the recorded `correct` feature rather than being guessed.
    assert group_scores(read_group).contexts[0].group.accuracy == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# At the scale the sentence is written at
# ---------------------------------------------------------------------------


def synthetic_run(seed: int = 20260801, n_groups: int = 64, k: int = 8):
    """512 rollouts over 64 groups, with a third of them over budget and one grader in twenty
    failing."""
    rng = random.Random(seed)
    grader = CountingGrader(fail_every=20)
    handle = SimpleRun(run_id="w22-scale")
    tapped = instrument_grader(grader, run=handle, name="composite")
    groups: list[GroupScores] = []
    for g in range(n_groups):
        trees = []
        contexts = []
        corrects = []
        for _ in range(k):
            correct = 1.0 if rng.random() < 0.55 else 0.0
            style = round(rng.random(), 3)
            tokens = round(rng.lognormvariate(4.7, 0.45), 1)
            corrects.append(correct)
            trees.append(k3_tree(*build_leaves(tapped, handle, correct=correct, style=style)))
            contexts.append(ctx_for(correct, tokens, accuracy=0.0, k=k, gid=f"g{g}"))
        accuracy = sum(corrects) / k
        contexts = [
            ScoreContext(
                features=c.features,
                predicates=c.predicates,
                group=GroupContext(accuracy=accuracy, k=k, id=f"g{g}"),
            )
            for c in contexts
        ]
        groups.append(
            GroupScores(trees=tuple(trees), contexts=tuple(contexts), estimator=CENTRED, id=f"g{g}")
        )
    return groups, grader


def test_the_sentence_the_catalogue_promises_comes_out_with_this_run_s_numbers() -> None:
    groups, grader = synthetic_run()
    calls_before = grader.calls
    got = counterfactual(groups, {"length_override"})
    assert not isinstance(got, Refusal), got
    assert grader.calls == calls_before

    assert got.n == 512
    assert got.n_groups == 64
    assert got.n_trees_with_node == 512
    # Not a target: whatever this seeded run does, the sentence has to be able to say it, and the
    # parts have to be consistent with each other.
    assert 0 < got.n_sign_reversed < got.n_comparable
    assert 0 <= got.n_moved <= got.n_comparable
    assert got.degenerate_groups_before == 0 and got.degenerate_groups_after == 0
    assert got.n_comparable == got.n - got.n_abstained_after
    sentence = got.says()
    assert sentence.startswith("removing length_override changes")
    assert f"of {got.n} rollouts" in sentence
    assert f"{got.n_sign_reversed}" in sentence

    # Every abstaining rollout is excluded from both arms rather than scored zero in either.
    for i, (before, after) in enumerate(zip(got.scores_before, got.scores_after)):
        if math.isnan(before):
            assert math.isnan(after), i
    # And the count goes up, never down, when the override is removed: a grader failure sitting
    # under a binding override is invisible while the override is there, because the constant does
    # not consult the child. Removing it is what makes those failures countable.
    assert got.n_abstained_after > got.n_abstained_before


def test_the_dead_zone_is_measured_and_it_erased_a_real_spread() -> None:
    groups, _ = synthetic_run()
    trees = [t for g in groups for t in g.trees]
    contexts = [c for g in groups for c in g.contexts]
    (zone,) = deadzone(trees, contexts)
    assert zone.node == "length_override"
    assert zone.condition == "tokens > 1.5 * b0"
    assert zone.n == 512
    assert 0 < zone.n_region < 512
    assert zone.n_binding == zone.n_region  # nothing else in this tree can pin the score to -1
    assert zone.fraction == zone.n_region / 512
    # The override collapsed a range of underlying scores onto one constant, and that range is the
    # ordering it destroyed. A graded penalty would have kept it.
    assert zone.distinct_underlying > 1
    assert zone.erased_spread is not None and zone.erased_spread > 0.0


def test_the_census_finds_the_grader_failures_and_keeps_the_zeros_they_produced() -> None:
    groups, grader = synthetic_run()
    trees = [t for g in groups for t in g.trees]
    contexts = [c for g in groups for c in g.contexts]
    got = census(trees, contexts)
    assert got.n_leaves == 1024
    # One call in twenty failed, and each failed call produced two abstaining leaves.
    assert got.n_abstained == 2 * (grader.calls // 20)
    assert got.n_silent_zero == got.n_abstained  # the framework's 0.0 is recorded, not used
    assert got.substituted_total == 0.0
    assert got.n_unattributable == 0
    assert got.abstention_rate is not None and 0.0 < got.abstention_rate < 0.15
    assert got.by_grader == {"composite": got.n_abstained}
    # Some of those failures sit under a binding override, where the pipeline never used the score
    # at all, and the census says how many rather than folding them in.
    assert got.n_shadowed > 0


# ---------------------------------------------------------------------------
# The refusal B3 owes a record that did not keep its per-leaf scores
# ---------------------------------------------------------------------------


def test_a_record_with_no_per_leaf_composition_is_refused_with_a_remedy() -> None:
    """B3's kill condition: without per-leaf scores this is a request, not an instrument."""
    grader = CountingGrader()
    handle = SimpleRun(run_id="w22-partial")
    tapped = instrument_grader(grader, run=handle, name="composite")
    trees = [k3_tree(*build_leaves(tapped, handle, correct=c, style=s)) for c, s, _ in WORKED]
    group = GroupScores(
        trees=(trees[0], None, trees[2], None),
        contexts=tuple(ctx_for(c, t, accuracy=0.5, k=4, gid="g0") for c, _, t in WORKED),
        estimator=CENTRED,
        id="g0",
    )
    got = counterfactual([group], {"length_override"})
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.RECORD_INCOMPLETE
    assert "2 of 4 trajectories carry no score tree" in got.detail
    assert "one missing rollout corrupts all K" in got.detail
    assert "instrument the grader with reward_lens.tap" in got.remedy
    assert got.statistics == {"n_missing": 2, "n_total": 4}
    assert got.partial is None
    assert "RECORD_INCOMPLETE" in got.render()

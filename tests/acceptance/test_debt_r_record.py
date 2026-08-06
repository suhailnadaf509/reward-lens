"""Debt R: five record-side wrong numbers, and one property failure.

A review of `record/` against the record format found nine differences. Four of them were fixed
already. These are the other five, each asserted here with the number it changed:

1. `AbstentionCensus.abstention_rate` returned **4.0** on a five-leaf record whose stated upper
   bound came back **1.6**, below its own point estimate. It counted a leaf as unattributable and
   as an abstention at once, so the numerator held leaves the denominator had removed.
2. `GroupStats.ranks` was computed over surviving scores only, so it was shorter than `k` and every
   entry after the first abstention described the wrong rollout. **200 of the 400 groups** in
   `tests/fixtures/grpo_run/long/`.
3. `AbsentRef.as_refusal` mapped all seven absence reasons onto `ACCESS_INSUFFICIENT`, including
   three whose own remedy is answerable only upstream.
4. `check_detector` passed a detector annotated `Any` or `object`, which its own docstring argues
   it must not, and `object` genuinely accepts a `Blind`.
5. `estimate_total` returned 0.0 on an empty sample where `estimate_mean` refuses.

The sixth item is in `tests/test_record_convert.py`: a property test that failed on a fresh
`hypothesis` seed. The classification is asserted at the bottom of this file, because the number
that settles it is small and checkable and the counterexample is otherwise only in a gitignored
directory.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.composition.abstention import AbstentionReading, read_census
from reward_lens.record.labels import (
    Blind,
    LabelLeak,
    RolloutFrame,
    blind,
    check_detector,
    detector_findings,
)
from reward_lens.record.reader import open_run
from reward_lens.record.schema import GroupStats, RecordSamplingPolicy, SamplingScheme
from reward_lens.record.scores import (
    GraderCallRef,
    Leaf,
    ScoreContext,
    census,
    evaluate,
)
from reward_lens.record.tensors import ABSENCE_REFUSAL, AbsenceReason, AbsentRef

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "grpo_run"
LONG_RUN = "run:f77bf75940ab982bbc35407af99cc094"


def _reachable_abstentions() -> list[Leaf]:
    """The five-leaf record E50 measured 4.0 on.

    Four leaves that abstained with no `GraderCallRef` beside them, and one clean call. That
    co-occurrence is what `tap.adapters.trl` writes at `_score_tree`: `refs.get(name)` is None for
    any reward function the tap saw no call from, and `abstained` is set from `row[name] is None`
    on the same rows, so a run whose tap missed one function's call produces exactly this.
    """
    trees = [Leaf(name="reward", value=None, grader_call=None, abstained=True) for _ in range(4)]
    trees.append(Leaf(name="reward", value=1.0, grader_call=GraderCallRef(grader="g")))
    return trees


# ---------------------------------------------------------------------------
# 1. The abstention rate, and the bound that used to sit under it
# ---------------------------------------------------------------------------


def test_the_abstention_rate_is_a_rate_on_the_input_that_returned_four() -> None:
    counts = census(_reachable_abstentions())

    assert counts.n_leaves == 5
    assert counts.n_abstained == 4
    # Before: n_unattributable was 4, `known` was 1, and 4 / 1 was reported as a rate.
    assert counts.n_unattributable == 0
    assert counts.n_abstained_unattributed == 4
    assert counts.n_known == 5
    assert counts.abstention_rate == pytest.approx(0.8)
    assert 0.0 <= counts.abstention_rate <= 1.0

    # Nothing was lost in the move: the leaves still carry no call record and still say so.
    assert counts.n_no_call_record == 4
    assert counts.by_grader == {"unknown": 4}
    assert "nothing saying which grader declined" in counts.render()


def test_the_stated_upper_bound_is_above_the_point_estimate_and_not_below_it() -> None:
    """`grader.abstention_rate` is a headline card field, so this is the number a reader sees."""
    before = AbstentionReading(
        n_leaves=5,
        n_abstained=4,
        n_silent_zero=0,
        n_unattributable=4,
        n_shadowed=0,
        n_boundary_failures=0,
        n_reconstructed=0,
        substituted_total=0.0,
        by_grader={"unknown": 4},
        outcomes={"unrecorded": 4, "returned": 1},
        abstention_rate=4 / 1,
        silent_zero_rate=0 / 1,
        abstention_rate_upper=(4 + 4) / 5,
        silent_zero_rate_upper=(0 + 4) / 5,
    )
    assert before.abstention_rate == 4.0
    assert before.abstention_rate_upper == 1.6
    assert before.abstention_rate_upper < before.abstention_rate
    assert "400.0% declined to score at all" in before.says()

    after = read_census(_reachable_abstentions(), None, ())
    assert isinstance(after, AbstentionReading)
    assert after.abstention_rate == pytest.approx(0.8)
    assert after.abstention_rate_upper == pytest.approx(0.8)
    assert after.abstention_rate_upper >= after.abstention_rate
    assert 0.0 <= after.abstention_rate <= after.abstention_rate_upper <= 1.0
    assert "80.0% declined to score at all" in after.says()


def test_the_bound_contains_the_point_estimate_for_every_mix_of_the_four_leaf_kinds() -> None:
    """The nesting the old arithmetic broke: the numerator's set inside the denominator's."""
    kinds = {
        "clean": Leaf(name="r", value=1.0, grader_call=GraderCallRef(grader="g")),
        "abstained_attributed": Leaf(
            name="r",
            value=None,
            grader_call=GraderCallRef(grader="g", outcome="raised"),
            abstained=True,
        ),
        "abstained_unattributed": Leaf(name="r", value=None, grader_call=None, abstained=True),
        "unknown": Leaf(name="r", value=0.0, grader_call=None),
    }
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    trees = (
                        [kinds["clean"]] * a
                        + [kinds["abstained_attributed"]] * b
                        + [kinds["abstained_unattributed"]] * c
                        + [kinds["unknown"]] * d
                    )
                    if not trees:
                        continue
                    got = read_census(trees, None, ())
                    if isinstance(got, Refusal):
                        # Every leaf was an unknown outcome. Refusing is the documented answer.
                        assert got.reason is RefusalReason.RECORD_INCOMPLETE
                        assert d == len(trees)
                        continue
                    assert 0.0 <= got.abstention_rate <= 1.0, trees
                    assert got.abstention_rate <= got.abstention_rate_upper <= 1.0, trees
                    assert got.silent_zero_rate <= got.silent_zero_rate_upper <= 1.0, trees


def test_the_real_records_abstention_rate_did_not_move() -> None:
    """The 200-step record has no unattributed abstention, so the fix must leave it alone."""
    run = open_run(FIXTURES / "long", LONG_RUN)
    trees = [t.scores for step in run.steps for g in step.groups for t in g.trajectories]
    counts = census(trees)

    assert counts.n_leaves == 1600
    assert counts.n_abstained == 200
    assert counts.n_unattributable == 0
    assert counts.n_abstained_unattributed == 0
    assert counts.abstention_rate == pytest.approx(0.125)
    assert counts.by_grader == {"length_reward": 200}


# ---------------------------------------------------------------------------
# 2. Ranks, aligned on all 400 groups of the 200-step record
# ---------------------------------------------------------------------------


def _recorded_scores(group: Any) -> list[float | None]:
    """One group's per-trajectory scores, None where the grader abstained."""
    out: list[float | None] = []
    for traj in group.trajectories:
        value = evaluate(traj.scores, ScoreContext())
        out.append(None if not math.isfinite(value) else float(value))
    return out


def test_ranks_are_positional_and_full_length_on_all_400_groups() -> None:
    run = open_run(FIXTURES / "long", LONG_RUN)

    n_groups = 0
    n_with_abstention = 0
    for step in run.steps:
        for group in step.groups:
            n_groups += 1
            scores = _recorded_scores(group)
            stats = GroupStats.from_scores(scores, std_epsilon=1e-8)

            assert stats.ranks is not None
            assert len(stats.ranks) == stats.k == len(group.trajectories)

            # A hole exactly where the score is missing, and nowhere else.
            for rank, score in zip(stats.ranks, scores):
                assert (rank is None) == (score is None)

            present = [(r, s) for r, s in zip(stats.ranks, scores) if s is not None]
            assert sorted(r for r, _ in present) == list(range(len(present)))
            ordered = sorted(present, key=lambda rs: rs[0])
            assert [s for _, s in ordered] == sorted((s for _, s in present), reverse=True), (
                "rank 0 is the highest score"
            )

            if any(s is None for s in scores):
                n_with_abstention += 1

    assert n_groups == 400
    # E50's number: half the groups carry an abstention, and every one of them used to produce a
    # ranks tuple one entry short.
    assert n_with_abstention == 200


def test_a_rank_vector_that_does_not_fit_its_group_cannot_be_built_or_read_back() -> None:
    with pytest.raises(ValueError, match="positional against Group.trajectories"):
        GroupStats(k=4, ranks=(0, 1, 2))

    # And the 200 short vectors already on disk read back as "not recorded" rather than as a
    # vector describing the wrong rollouts.
    assert GroupStats.from_canonical({"k": 4, "ranks": [0, 1, 2]}).ranks is None

    run = open_run(FIXTURES / "long", LONG_RUN)
    stale = sum(1 for step in run.steps for g in step.groups if g.group_stats.ranks is None)
    assert stale == 200


def test_the_group_with_no_abstention_ranks_exactly_as_it_did_before() -> None:
    assert GroupStats.from_scores([0.1, 0.9, 0.5], std_epsilon=1e-8).ranks == (2, 0, 1)
    # And the abstaining case leaves the gap in place rather than shifting the tail.
    assert GroupStats.from_scores([1.08, None, 0.96, 0.88], std_epsilon=1e-8).ranks == (
        0,
        None,
        1,
        2,
    )
    assert GroupStats.from_scores([None, None], std_epsilon=1e-8).ranks == (None, None)


# ---------------------------------------------------------------------------
# 3. Seven absence reasons, each mapped by its own remedy
# ---------------------------------------------------------------------------

#: The mapping this file asserts, derived from each reason's `DEFAULT_REMEDY` rather than from the
#: code, by E30's question: is the remedy answerable where the reader is standing, or only upstream
#: where the record was written?
EXPECTED_REFUSAL: Mapping[AbsenceReason, RefusalReason] = {
    # "Re-run with a CaptureSpec naming this site." Nothing here recovers it.
    AbsenceReason.NOT_CAPTURED: RefusalReason.RECORD_INCOMPLETE,
    # "Raise the tap budget ... then re-run."
    AbsenceReason.EGRESS_REFUSED: RefusalReason.RECORD_INCOMPLETE,
    # "Point the reader at the tensor store that accompanies this record."
    AbsenceReason.SHARD_MISSING: RefusalReason.ACCESS_INSUFFICIENT,
    # "Pass a Recomputer to resolve()."
    AbsenceReason.RECOMPUTE_UNSUPPORTED: RefusalReason.ACCESS_INSUFFICIENT,
    # "Make the recipe's model and engine reachable."
    AbsenceReason.RECOMPUTE_UNAVAILABLE: RefusalReason.ACCESS_INSUFFICIENT,
    # "Recompute on the engine, revision, dtype and attention implementation named in the
    # RecomputeRef." The reader has the recipe and is short of the right engine.
    AbsenceReason.NUMERICS_FLOOR_EXCEEDED: RefusalReason.ACCESS_INSUFFICIENT,
    # "There is no recipe that recovers this one."
    AbsenceReason.COMPACTED: RefusalReason.RECORD_INCOMPLETE,
}


def test_every_absence_reason_refuses_with_the_reason_its_own_remedy_implies() -> None:
    assert set(EXPECTED_REFUSAL) == set(AbsenceReason)
    assert ABSENCE_REFUSAL == dict(EXPECTED_REFUSAL)

    for reason, expected in EXPECTED_REFUSAL.items():
        refusal = AbsentRef.of(reason, detail="d").as_refusal("record.test")
        assert isinstance(refusal, Refusal)
        assert refusal.reason is expected, reason.name
        assert refusal.statistics["absence"] == reason.name
        assert refusal.remedy.strip()


def test_the_mapping_is_not_one_blanket_constant_in_either_direction() -> None:
    """What let the original collapse survive an earlier pass was that it was one decision."""
    mapped = set(ABSENCE_REFUSAL.values())
    assert mapped == {RefusalReason.ACCESS_INSUFFICIENT, RefusalReason.RECORD_INCOMPLETE}
    upstream = {r for r, v in ABSENCE_REFUSAL.items() if v is RefusalReason.RECORD_INCOMPLETE}
    assert upstream == {
        AbsenceReason.NOT_CAPTURED,
        AbsenceReason.EGRESS_REFUSED,
        AbsenceReason.COMPACTED,
    }


def test_the_three_upstream_remedies_say_re_run_or_say_there_is_no_recipe() -> None:
    """The remedy is the evidence for the mapping, so it is asserted rather than trusted."""
    from reward_lens.record.tensors import DEFAULT_REMEDY

    assert "re-run" in DEFAULT_REMEDY[AbsenceReason.NOT_CAPTURED].lower()
    assert "re-run" in DEFAULT_REMEDY[AbsenceReason.EGRESS_REFUSED].lower()
    assert "no recipe that recovers this one" in DEFAULT_REMEDY[AbsenceReason.COMPACTED]


# ---------------------------------------------------------------------------
# 4. check_detector, on the two annotations that check nothing
# ---------------------------------------------------------------------------


def _takes_any(frame: Any) -> float:
    return 0.0


def _takes_object(frame: object) -> float:
    return 0.0


def _returns_any(frame: RolloutFrame) -> Any:
    return 0.0


def _returns_object(frame: RolloutFrame) -> object:
    return 0.0


def _takes_container_of_object(frame: Mapping[str, object]) -> float:
    return 0.0


def _good(frame: RolloutFrame) -> float:
    return 0.0


def test_a_detector_annotated_any_or_object_is_rejected() -> None:
    for fn in (
        _takes_any,
        _takes_object,
        _returns_any,
        _returns_object,
        _takes_container_of_object,
    ):
        findings = detector_findings(fn)
        assert len(findings) == 1, fn.__name__
        assert "constrains nothing" in findings[0]
        with pytest.raises(LabelLeak, match="can reach the held-out labels"):
            check_detector(fn)


def test_object_is_not_a_formality_because_it_genuinely_accepts_a_blind() -> None:
    """The hole in `Blind[T]`'s runtime half: every value is an `object`, including this one."""
    label: Blind[bool] = blind(True, key="hacked")
    assert isinstance(label, Blind)
    assert _takes_object(label) == 0.0  # no error, at runtime or under mypy


def test_a_correctly_annotated_detector_still_passes() -> None:
    assert detector_findings(_good) == ()
    check_detector(_good)


def test_the_blind_types_job_still_expects_exactly_eight_rejected_lines() -> None:
    """E25's standing guard. This fix is runtime-only and must not move the static count."""
    leaks = Path(__file__).resolve().parents[1] / "w2_3_typing" / "leaks.py"
    markers = re.findall(r"#\s*EXPECT:\s*([\w-]+)", leaks.read_text(encoding="utf-8"))
    assert len(markers) == 8


# ---------------------------------------------------------------------------
# 5. A Horvitz-Thompson total over nothing
# ---------------------------------------------------------------------------


def test_the_total_refuses_on_an_empty_sample_like_its_sibling_does() -> None:
    policy = RecordSamplingPolicy(scheme=SamplingScheme.UNIFORM, rate=0.25)

    got = policy.estimate_total([])
    assert isinstance(got, Refusal)
    assert got.reason is RefusalReason.ENVELOPE_VIOLATED
    assert got.statistics["n"] == 0
    assert got.remedy.strip()
    # The sibling refused all along, which is what made this a difference rather than a design.
    assert isinstance(policy.estimate_mean([]), Refusal)

    # And a full-capture policy is no different: zero recorded units is zero recorded units.
    assert isinstance(RecordSamplingPolicy().estimate_total([]), Refusal)


def test_a_non_empty_total_is_unchanged() -> None:
    policy = RecordSamplingPolicy(scheme=SamplingScheme.UNIFORM, rate=0.25)
    got = policy.estimate_total([2.0] * 25)
    assert not isinstance(got, Refusal)
    assert got.value == pytest.approx(200.0)
    assert got.method == "horvitz_thompson"
    assert got.n == 25


# ---------------------------------------------------------------------------
# 6. The property that failed on a fresh seed: property-too-strong, not a code defect
# ---------------------------------------------------------------------------


def test_the_converter_computes_the_exact_group_mean_on_the_draw_that_failed() -> None:
    """`test_every_item_is_a_group_and_every_column_a_trajectory`, classified.

    The failing row is [1.0, 524287.96875, -492901.0]. The verdict is **property-too-strong**, and
    more precisely it is the reference that was wrong rather than the tolerance: the assertion
    compared a float64 group mean against `values[i].mean()`, which on a float32 array accumulates
    in float32.

    The arithmetic, term by term. All three values are exact in float32. In float32 the partial sum
    1.0 + 524287.96875 is 524288.96875, which needs an ulp of 0.03125 and gets 0.0625 above 2**19,
    so it rounds to 524289.0. The remaining subtraction cancels five digits, condition number 32.4,
    and turns that 0.03125 into an absolute error of 0.03125 on a sum of 31387.96875. Divided by 3
    the float32 reference lands on 10462.6669921875 against an exact 10462.65625, a relative error
    of 1.03e-06, just outside the 1e-06 the test asked for.

    The converter widens each score with `float(row[k])` and `GroupStats.from_scores` averages in
    float64, so it returns 10462.65625, which is the exactly-rounded answer. There is no defect in
    the code and the tolerance did not need loosening.
    """
    from fractions import Fraction

    row = np.asarray([1.0, 524287.96875, -492901.0], dtype=np.float32)

    exact = float(sum(Fraction(float(x)) for x in row) / 3)
    float32_reference = float(row.mean())
    float64_reference = float(row.astype(np.float64).mean())

    assert exact == 10462.65625
    assert float32_reference == 10462.6669921875
    assert float64_reference == exact, "the implementation's accumulation is exactly right here"

    relative_error = abs(float32_reference - exact) / abs(exact)
    assert relative_error == pytest.approx(1.0267e-06, rel=1e-3)
    assert relative_error > 1e-06, "which is why the old reference failed the 1e-06 assertion"

    # The cancellation that does it, so the diagnosis is checkable rather than asserted.
    assert float(np.float32(1.0) + np.float32(524287.96875)) == 524289.0
    assert 1.0 + 524287.96875 == 524288.96875
    condition = float(np.abs(row.astype(np.float64)).sum() / abs(row.astype(np.float64).sum()))
    assert condition == pytest.approx(32.407, rel=1e-3)


def test_the_failing_draw_is_pinned_in_the_test_and_not_only_in_the_hypothesis_database() -> None:
    """`.hypothesis/` is gitignored, so the counterexample has to live in the file."""
    convert_tests = (Path(__file__).resolve().parents[1] / "test_record_convert.py").read_text(
        encoding="utf-8"
    )

    assert "_CANCELLATION_DRAW" in convert_tests
    assert "524287.96875" in convert_tests
    assert "-492901.0" in convert_tests
    assert "@example(grid=_CANCELLATION_DRAW)" in convert_tests

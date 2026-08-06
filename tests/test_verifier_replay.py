"""D10: re-grading a record and comparing against the score the record holds.

Small, and the arithmetic is a fraction, so most of what is worth testing is the classification:
which tasks count in the denominator, which absence a failure maps onto, and whether the
`STATIONARY_GRADER` wire that three other instruments in this package depend on actually carries a
measurement.
"""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition, RegimeReading
from reward_lens.core.invariance import InvariancePayload, check_invariance
from reward_lens.core.quantity import ladder
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.measure.base import lint_instrument
from reward_lens.record.tensors import AbsenceReason
from reward_lens.verifier import ListCorpus, Rollout
from reward_lens.verifier.replay import (
    DEFAULT_SCORE_FLOOR,
    STATIONARY_GRADER_FLOOR,
    ReplayFidelity,
    ReplayReport,
    replay_corpus,
    stationary_grader_reading,
)


def exact_match(response: str, gold: str) -> float:
    """A pure grader, which is the case where a fidelity of 1.0 is the expected answer."""
    if response == "BOOM":
        raise ValueError("grader exploded on this input")
    return 1.0 if response == gold else 0.0


#: Four tasks with a recorded score and one without. `t3`'s record claims 1.0 and the grader says
#: 0.0, so the honest fidelity is 2/4.
CORPUS = ListCorpus.of(
    [
        Rollout(id="t1", inputs={"response": "a", "gold": "a"}, score=1.0),
        Rollout(id="t2", inputs={"response": "b", "gold": "a"}, score=0.0),
        Rollout(id="t3", inputs={"response": "c", "gold": "a"}, score=1.0),
        Rollout(id="t4", inputs={"response": "BOOM", "gold": "a"}, score=0.0),
        Rollout(id="t5", inputs={"response": "d", "gold": "d"}),
    ]
)


# ---------------------------------------------------------------------------
# The fraction
# ---------------------------------------------------------------------------


def test_the_fidelity_is_reproduced_over_recorded_and_not_over_everything() -> None:
    """A task the record never scored is a gap in the record, not a replay failure."""
    report = replay_corpus(exact_match, CORPUS)
    assert isinstance(report, ReplayReport)
    assert report.n_tasks == 5
    assert report.n_attempted == 4
    assert report.n_reproduced == 2
    assert report.replay_fidelity == pytest.approx(0.5)
    assert report.n_no_recorded_score == 1


def test_each_failure_lands_on_the_right_absence():
    report = replay_corpus(exact_match, CORPUS)
    assert isinstance(report, ReplayReport)
    by_id = {t.id: t for t in report.tasks}
    assert by_id["t1"].absence == ""
    assert by_id["t3"].absence_reason is AbsenceReason.NUMERICS_FLOOR_EXCEEDED
    assert by_id["t4"].absence_reason is AbsenceReason.RECOMPUTE_UNAVAILABLE
    assert by_id["t5"].absence_reason is AbsenceReason.NOT_CAPTURED


def test_a_failure_converts_to_the_records_own_absent_ref_with_a_remedy() -> None:
    """*How this fits `RecomputeRef`.* A missing score enters the record's vocabulary, not a new one."""
    report = replay_corpus(exact_match, CORPUS)
    assert isinstance(report, ReplayReport)
    refs = report.absent_refs()
    assert len(refs) == 3
    assert {r.reason for r in refs} == {
        AbsenceReason.NUMERICS_FLOOR_EXCEEDED,
        AbsenceReason.RECOMPUTE_UNAVAILABLE,
        AbsenceReason.NOT_CAPTURED,
    }
    assert all(r.remedy.strip() for r in refs)
    floor_exceeded = next(r for r in refs if r.reason is AbsenceReason.NUMERICS_FLOOR_EXCEEDED)
    assert floor_exceeded.statistics["recorded"] == 1.0
    assert floor_exceeded.statistics["replayed"] == 0.0


def test_a_grader_that_raises_is_counted_rather_than_skipped() -> None:
    """One unreplayable task in a thousand is a measurement of 0.999, not a failure to measure."""
    report = replay_corpus(exact_match, CORPUS)
    assert isinstance(report, ReplayReport)
    assert report.n_unreplayable == 1
    raised = next(t for t in report.tasks if t.id == "t4")
    assert "ValueError" in raised.error


def test_a_pure_grader_on_its_own_record_reproduces_everything() -> None:
    """The kill condition arriving, on the one case where it is the expected answer."""
    clean = ListCorpus.of(
        [
            Rollout(id=f"c{i}", inputs={"response": s, "gold": "a"}, score=1.0 if s == "a" else 0.0)
            for i, s in enumerate(["a", "b", "a", "c"])
        ]
    )
    report = replay_corpus(exact_match, clean, repeats=3)
    assert isinstance(report, ReplayReport)
    assert report.replay_fidelity == 1.0
    assert report.n_nondeterministic == 0
    assert report.deterministic_fraction == 1.0


# ---------------------------------------------------------------------------
# Determinism, which is the other half of the instrument's name
# ---------------------------------------------------------------------------


def test_a_grader_that_disagrees_with_itself_does_not_count_as_reproduced() -> None:
    """A score that is only sometimes right is not a score the record can be audited against."""
    counter = itertools.count()

    def alternating(response: str, gold: str) -> float:
        return float(next(counter) % 2)

    corpus = ListCorpus.of([Rollout(id="x", inputs={"response": "a", "gold": "a"}, score=1.0)])
    report = replay_corpus(alternating, corpus, repeats=4)
    assert isinstance(report, ReplayReport)
    assert report.n_nondeterministic == 1
    assert report.n_reproduced == 0
    assert report.deterministic_fraction == 0.0
    assert not report.tasks[0].deterministic
    assert report.tasks[0].replays == (0.0, 1.0, 0.0, 1.0)


def test_one_repeat_says_plainly_that_the_determinism_half_did_not_run() -> None:
    report = replay_corpus(exact_match, CORPUS, repeats=1)
    assert isinstance(report, ReplayReport)
    assert any("determinism half did not run" in n for n in report.notes)
    import math

    assert math.isnan(report.deterministic_fraction)


def test_the_spread_is_not_this_instruments_business_and_the_reading_says_so() -> None:
    """`env.flakiness` is A7's quantity, in percentage points. D10 reports a fraction of tasks."""
    report = replay_corpus(exact_match, CORPUS, repeats=2)
    assert isinstance(report, ReplayReport)
    assert "A7's env.flakiness" in report.render()
    assert report.headline == report.replay_fidelity


# ---------------------------------------------------------------------------
# The wire into the envelope machinery
# ---------------------------------------------------------------------------


def test_the_condition_reading_carries_the_statistic_and_the_threshold() -> None:
    report = replay_corpus(exact_match, CORPUS)
    assert isinstance(report, ReplayReport)
    reading = report.condition_reading()
    assert reading.condition is RegimeCondition.STATIONARY_GRADER
    assert reading.holds is False
    assert reading.statistic == pytest.approx(0.5)
    assert reading.threshold == STATIONARY_GRADER_FLOOR


def test_the_measurement_admits_an_envelope_that_three_other_instruments_declare() -> None:
    """D3, D4 and D5 all require STATIONARY_GRADER and name `env.replay_fidelity`. This is the other end."""
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
        measured_by={RegimeCondition.STATIONARY_GRADER: "env.replay_fidelity"},
    )
    clean = ListCorpus.of([Rollout(id="c", inputs={"response": "a", "gold": "a"}, score=1.0)])

    ok = stationary_grader_reading(exact_match, clean, repeats=2)
    assert envelope.admits(RegimeReading(conditions={ok.condition: ok}))

    bad = stationary_grader_reading(exact_match, CORPUS)
    assert not envelope.admits(RegimeReading(conditions={bad.condition: bad}))


def test_an_unmeasurable_condition_is_unknown_rather_than_failed() -> None:
    """*Unknown is not a pass* and it is also not a failure. `classify` splits the two on purpose."""
    empty = ListCorpus.of([])
    reading = stationary_grader_reading(exact_match, empty)
    assert reading.holds is None
    assert "could not be measured" in reading.detail


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_an_empty_corpus_refuses_rather_than_reporting_perfect_fidelity() -> None:
    """*Never a zero.* A fidelity of 1.0 over zero tasks is the worst number this could return."""
    refusal = replay_corpus(exact_match, ListCorpus.of([]))
    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "most misleading number" in refusal.remedy


def test_a_corpus_with_no_recorded_scores_refuses_and_says_which_half_is_missing() -> None:
    unscored = ListCorpus.of(
        [Rollout(id=f"u{i}", inputs={"response": "a", "gold": "a"}) for i in range(3)]
    )
    refusal = replay_corpus(exact_match, unscored)
    assert isinstance(refusal, Refusal)
    assert refusal.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert refusal.statistics["with_score"] == 0
    assert "supports the re-grade but not the comparison" in refusal.remedy


def test_the_instrument_refuses_when_only_one_end_is_supplied() -> None:
    reading = ReplayFidelity(exact_match, None).estimate()
    assert isinstance(reading, Refusal)
    assert reading.reason is RefusalReason.ACCESS_INSUFFICIENT
    assert "no corpus" in reading.detail
    assert "both ends" in reading.remedy


def test_zero_repeats_is_a_programming_error_rather_than_a_refusal() -> None:
    """A refusal is for a condition the instrument anticipated; this is a caller mistake."""
    with pytest.raises(ValueError, match="at least 1"):
        replay_corpus(exact_match, CORPUS, repeats=0)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@given(
    scores=st.lists(st.sampled_from([0.0, 0.5, 1.0]), min_size=1, max_size=25),
    lies=st.lists(st.booleans(), min_size=1, max_size=25),
)
@settings(max_examples=150, deadline=None)
def test_the_fidelity_is_exactly_the_fraction_of_honest_records(
    scores: list[float], lies: list[bool]
) -> None:
    """A planted record: flip a known number of stored scores and the fidelity must find them."""

    def constant(value: float) -> float:
        return value

    n = min(len(scores), len(lies))
    rollouts = []
    n_lies = 0
    for i in range(n):
        recorded = scores[i]
        if lies[i]:
            recorded = recorded + 1.0
            n_lies += 1
        rollouts.append(Rollout(id=f"r{i}", inputs={"value": scores[i]}, score=recorded))

    report = replay_corpus(constant, ListCorpus.of(rollouts))
    assert isinstance(report, ReplayReport)
    assert report.n_reproduced == n - n_lies
    assert report.replay_fidelity == pytest.approx((n - n_lies) / n)


@given(wobble=st.floats(min_value=0.0, max_value=1e-3))
@settings(max_examples=60, deadline=None)
def test_the_score_floor_is_the_line_and_it_is_not_widened_quietly(wobble: float) -> None:
    """The same discipline `RecomputeRef.expected_numerics_floor` carries: the floor decides."""

    def wobbly(value: float) -> float:
        return value + wobble

    corpus = ListCorpus.of([Rollout(id="w", inputs={"value": 1.0}, score=1.0)])
    report = replay_corpus(wobbly, corpus)
    assert isinstance(report, ReplayReport)
    # Against the deviation the addition actually achieved rather than the nominal wobble:
    # `1.0 + 1e-9` lands at a float whose distance from 1.0 is 1.0000000272e-09, so a test written
    # against the nominal value disagrees with the code at exactly the floor for a reason that has
    # nothing to do with either.
    achieved = abs(wobbly(1.0) - 1.0)
    assert report.n_reproduced == (1 if achieved <= DEFAULT_SCORE_FLOOR else 0)


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------


def test_the_instrument_passes_lint_and_declares_all_six() -> None:
    inst = ReplayFidelity(exact_match, CORPUS)
    assert lint_instrument(inst) == []
    assert inst.quantity == "env.replay_fidelity"
    assert inst.requires and inst.substrates and inst.phases
    assert inst.envelope is not None and inst.invariance and inst.baselines


def test_the_envelope_is_unconditional_because_requiring_its_own_output_is_circular() -> None:
    inst = ReplayFidelity(exact_match, CORPUS)
    assert inst.envelope is not None
    assert inst.envelope.unconditional
    assert "circular" in inst.envelope.justification


def test_the_registered_ladder_matches_the_registrys_single_rung() -> None:
    assert [e.rung for e in ladder("env.replay_fidelity")] == [0]


def test_the_generated_invariance_test_passes() -> None:
    """*No instrument merges without its generated invariance test passing.*

    `none`, which resolves to the trivial group: no affine rescaling of the reward acts on the
    fraction of tasks whose recorded score reproduces, because both sides of the comparison move
    together. That is an answer rather than an omission. See E11.
    """
    inst = ReplayFidelity(exact_match, CORPUS)
    group = inst.invariance if inst.invariance != "none" else "trivial"
    report = check_invariance(
        inst,
        group,
        InvariancePayload(),
        n=4,
        run=lambda i, _p: float(replay_corpus(exact_match, CORPUS).replay_fidelity),
    )
    assert report.passed
    assert "trivial group" in report.skipped


def test_the_fidelity_really_is_unmoved_by_an_affine_rescaling_of_the_reward() -> None:
    """The trivial-group declaration above, checked rather than asserted.

    The generated test for `none` passes vacuously by construction, so the claim behind the
    declaration is worth one direct check: rescale every recorded score and every grader output by
    the same affine map and the fraction that agrees cannot move.
    """
    for a, b in ((2.0, 0.0), (0.5, 3.0), (-1.0, 1.0)):

        def rescaled(response: str, gold: str, _a: float = a, _b: float = b) -> float:
            return _a * exact_match(response, gold) + _b

        moved = ListCorpus.of(
            [
                Rollout(
                    id=r.id, inputs=r.inputs, score=None if r.score is None else a * r.score + b
                )
                for r in CORPUS
            ]
        )
        report = replay_corpus(rescaled, moved)
        assert isinstance(report, ReplayReport)
        assert report.replay_fidelity == pytest.approx(0.5)

"""The grader wrapper, the ring, and the Plane A contract.

The clause: *wrapping a callable that raises, is slow, and returns a dict never changes
the return value and never propagates; the overhead is measured and recorded; a deliberate budget
breach disables the tap and records it.*

"Never propagates" needs reading carefully and this file pins the reading. A failure of the
**recorder** never propagates, because the tap's job is to be invisible. An exception from the
**wrapped callable** propagates unchanged, because swallowing it would alter the host's behaviour,
and altering the host's behaviour is the one thing Plane A may never do. The two are opposite
requirements and a wrapper that gets them the wrong way round would pass a loosely-worded test.

Beyond the clause, this file asserts the things a wrapper of this kind usually gets wrong:
identity of a mutable return value, fidelity across six return types, the third distinct recorder
exception, a full ring that drops and counts rather than blocking, and an overhead number that
composes into an uncertainty budget instead of sitting beside it as prose.
"""

from __future__ import annotations

import asyncio
import gc
import inspect
import statistics
import threading
import time

import numpy as np
import pytest

from reward_lens.core.budget import BudgetTerm, UncertaintyBudget
from reward_lens.tap import (
    BreachKind,
    CallOutcome,
    InstrumentEffect,
    SimpleRun,
    TapBudget,
    TapRing,
    instrument_grader,
    tap,
)
from reward_lens.tap.contract import RECORD_FOOTPRINT_BYTES
from reward_lens.tap.grader_wrap import (
    DISTINCT_EXCEPTION_LIMIT,
    TapBudgetExceeded,
    TapGuard,
)

GENEROUS = TapBudget(
    max_added_latency_ms_p99=1000.0,
    max_resident_bytes=256 * 1024 * 1024,
    max_added_alloc_bytes_per_step=256 * 1024 * 1024,
)


def fresh_run(capacity: int = 4096) -> SimpleRun:
    return SimpleRun(run_id="acceptance", ring=TapRing(capacity=capacity))


# ---------------------------------------------------------------------------
# The clause, part 1: raises, is slow, returns a dict
# ---------------------------------------------------------------------------


class GraderExploded(Exception):
    """The grader's own failure. It belongs to the host and must reach it."""


def test_clause_raises_slow_dict():
    """One callable that does all three things the clause names, wrapped once.

    The dict comes back as the same object; the slow call is timed rather than interfered with;
    the raise reaches the caller with its own type, message and traceback, and is recorded on the
    way past with ``CallOutcome.RAISED``.
    """
    payload = {"correct": 1.0, "format": 0.5, "length": -0.25}
    run = fresh_run()

    def grader(mode):
        if mode == "raise":
            raise GraderExploded("the sandbox died")
        if mode == "slow":
            time.sleep(0.02)
            return payload
        return payload

    wrapped = instrument_grader(grader, run=run, budget=GENEROUS)

    assert wrapped("dict") is payload

    t0 = time.perf_counter()
    assert wrapped("slow") is payload
    assert time.perf_counter() - t0 >= 0.02

    with pytest.raises(GraderExploded, match="the sandbox died"):
        wrapped("raise")

    records = run.ring.peek()
    assert len(records) == 3
    assert [r.outcome for r in records] == [
        CallOutcome.RETURNED,
        CallOutcome.RETURNED,
        CallOutcome.RAISED,
    ]

    # every component of the return value, not just the total
    assert dict(records[0].components()) == payload
    assert records[0].shape.kind == "mapping"
    assert set(records[0].shape.keys) == set(payload)

    # the slow call's latency is the grader's, not the tap's
    assert records[1].inner_ns >= 20_000_000
    assert records[1].added_ns < records[1].inner_ns

    # the raise is recorded with its own identity, and the tap did not disable itself over it
    assert records[2].error_type == "GraderExploded"
    assert records[2].error_message == "the sandbox died"
    assert records[2].value is None
    assert records[2].raised is True
    assert wrapped.guard.disabled is False
    assert wrapped.guard.recorder_exceptions == 0


def test_wrapped_exception_is_reraised_unchanged():
    """Same exception object, same traceback origin, not a re-wrap.

    A wrapper that catches and re-raises with ``raise exc`` rather than a bare ``raise`` truncates
    the traceback, and one that wraps in its own exception class breaks every ``except`` clause the
    host wrote. Both are silent until somebody's error handling stops working in production.
    """
    run = fresh_run()
    sentinel = GraderExploded("original")

    def grader():
        raise sentinel

    wrapped = instrument_grader(grader, run=run, budget=GENEROUS)
    with pytest.raises(GraderExploded) as excinfo:
        wrapped()

    assert excinfo.value is sentinel
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    # the frame the exception was raised from is still the grader's, not the wrapper's
    assert excinfo.traceback[-1].name == "grader"
    assert run.ring.peek()[0].error_type == "GraderExploded"


# ---------------------------------------------------------------------------
# Return-value fidelity
# ---------------------------------------------------------------------------


def test_mutable_return_is_identical_not_merely_equal():
    """``is``, not ``==``. Equality would pass for a copy, and a copy is the bug.

    Two directions matter. The object that comes out is the object that went in, and a mutation
    made after the call is visible through the returned reference, which proves nothing was
    snapshotted on the way past.
    """
    run = fresh_run()
    payload = {"a": [1, 2, 3], "b": {"nested": True}}

    def grader():
        return payload

    wrapped = instrument_grader(grader, run=run, budget=GENEROUS)
    out = wrapped()

    assert out is payload
    assert out["a"] is payload["a"]
    out["a"].append(4)
    assert payload["a"] == [1, 2, 3, 4]
    assert run.ring.peek()[0].value is payload


@pytest.mark.parametrize(
    "value",
    [
        None,
        0.0,
        1.0,
        -0.25,
        float("nan"),
        42,
        True,
        np.float64(0.5),
        np.float32(0.5),
        np.int64(3),
        (0.1, 0.2, 0.3),
        [0.1, None, 0.3],
        {"correct": 1.0, "format": None},
        "a string, which is not a score and must survive anyway",
        b"bytes",
        set(),
    ],
    ids=lambda v: f"{type(v).__name__}:{v!r}"[:40],
)
def test_every_return_type_comes_back_unchanged(value):
    """Type and value preserved for everything a grader in the wild is known to return.

    TRL iterates the return element by element, ``verifiers`` calls ``float()`` on it, veRL fans a
    dict's keys out. A wrapper that normalises any of these to a common shape breaks a different
    framework each time, so it normalises none of them.
    """
    run = fresh_run()

    def grader():
        return value

    wrapped = instrument_grader(grader, run=run, budget=GENEROUS)
    out = wrapped()

    assert type(out) is type(value)
    if isinstance(value, float) and value != value:
        assert out != out  # NaN, and it stayed NaN rather than becoming 0.0
    else:
        assert out == value
    if not isinstance(value, (int, float, str, bytes, bool)) or isinstance(value, np.generic):
        assert out is value

    record = run.ring.peek()[0]
    expected = CallOutcome.SENTINEL if (value is None or value != value) else CallOutcome.RETURNED
    assert record.outcome is expected


def test_none_and_nan_are_recorded_as_sentinels_not_as_scores():
    """The abstention channel, which is the whole of instrument B4.

    TRL maps a ``None`` return to ``torch.nan`` and excludes the row, which is the correct
    behaviour and the model here. A tap that recorded ``None`` as a score of zero would reproduce
    exactly the defect it exists to count.
    """
    run = fresh_run()
    values = [1.0, None, 0.0, float("nan"), 0.5]
    it = iter(values)
    wrapped = instrument_grader(lambda: next(it), run=run, budget=GENEROUS, name="abstainer")
    for _ in values:
        wrapped()

    outcomes = [r.outcome for r in run.ring.peek()]
    assert outcomes == [
        CallOutcome.RETURNED,
        CallOutcome.SENTINEL,
        CallOutcome.RETURNED,  # a genuine 0.0 is a score
        CallOutcome.SENTINEL,
        CallOutcome.RETURNED,
    ]
    silent = sum(1 for r in run.ring.peek() if r.outcome is CallOutcome.SENTINEL)
    assert silent / len(values) == pytest.approx(0.4)


def test_signature_is_preserved_so_verifiers_dispatch_is_unchanged():
    """``verifiers`` filters kwargs on ``inspect.signature(func)`` (``rubric.py:199-210``).

    A wrapper reported as ``(*args, **kwargs)`` declares VAR_KEYWORD, which flips that dispatch
    from the filtered branch to the unfiltered one and changes what the grader is called with.
    ``functools.wraps`` sets ``__wrapped__`` and ``inspect.signature`` follows it, so the observable
    signature is the grader's.
    """

    def grader(prompt, completion, answer=None):
        return 1.0

    wrapped = instrument_grader(grader, run=fresh_run(), budget=GENEROUS)
    sig = inspect.signature(wrapped)
    assert list(sig.parameters) == ["prompt", "completion", "answer"]
    assert not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    assert wrapped.__name__ == "grader"


def test_async_grader_is_awaited_and_its_own_latency_is_recorded():
    """``verifiers`` awaits reward functions through ``maybe_await``, so async is the live case.

    A synchronous wrapper around an ``async def`` would time the construction of a coroutine and
    record a latency three orders of magnitude too small, which is worse than no latency at all.
    """
    run = fresh_run()
    payload = {"score": 1.0}

    async def grader(x):
        await asyncio.sleep(0.02)
        return payload

    wrapped = instrument_grader(grader, run=run, budget=GENEROUS)
    assert inspect.iscoroutinefunction(wrapped)
    out = asyncio.run(wrapped("prompt"))

    assert out is payload
    record = run.ring.peek()[0]
    assert record.inner_ns >= 20_000_000
    assert record.outcome is CallOutcome.RETURNED


# ---------------------------------------------------------------------------
# The clause, part 2: the overhead is measured and recorded
# ---------------------------------------------------------------------------


def _measure_added_ns(reps: int = 9, n: int = 4000) -> tuple[float, InstrumentEffect]:
    """End-to-end added nanoseconds per call, wrapped against bare, on the same process.

    Measured from outside rather than taken from the tap's own report, because the tap cannot time
    its own call prologue from the inside and a self-report is exactly the number that should not
    be trusted without a second instrument. The two are compared below.
    """
    args = (["p"] * 8, ["c"] * 8)
    kwargs = {"completion_ids": [[1, 2]] * 8, "seed": 3}

    def grader(prompts, completions, completion_ids=None, **kw):
        return [0.5] * 8

    bare, wrapped_t = [], []
    effect = None
    for _ in range(reps):
        run = SimpleRun(run_id="overhead", ring=TapRing(capacity=2048))
        w = instrument_grader(grader, run=run)
        for _ in range(200):  # warm both paths before either is timed
            grader(*args, **kwargs)
            w(*args, **kwargs)
        run.ring.drain()
        for target, sink in ((grader, bare), (w, wrapped_t)):
            gc.collect()
            t0 = time.perf_counter_ns()
            for i in range(n):
                target(*args, **kwargs)
                if (i & 255) == 0:
                    run.ring.drain()
            sink.append((time.perf_counter_ns() - t0) / n)
        effect = w.effect()
        run.ring.drain()
    return statistics.median(wrapped_t) - statistics.median(bare), effect


def test_overhead_is_measured_recorded_and_plausible():
    """The number, produced by code that ran, and cross-checked against the tap's own report.

    On the reference machine (Intel i9-13900HX, CPython 3.12.3) this lands at 1.89 to 2.03 us per
    call against a bare callable of 206 to 222 ns, over three runs. About 800 ns of that is the
    recorder segment the tap reports itself; the rest is the extra Python call layer and the four
    ``perf_counter_ns`` reads that make the self-measurement possible.

    The assertion ceiling is deliberately loose, because this runs on whatever a
    continuous-integration box happens to be and a ceiling tight enough to be interesting would be
    a flaky test rather than a strict one. What it catches is a regression of the kind this file
    already found once: an unguarded windowed quantile on the hot path took the per-call cost from
    1.9 us to 6.1 us, and nothing else in the suite noticed.
    """
    added_ns, effect = _measure_added_ns()

    assert added_ns > 0, "the wrapper cannot be free; a zero here means the measurement is broken"
    assert added_ns < 50_000, f"per-call overhead {added_ns:.0f} ns is far above the ~1.9 us design"

    # recorded, not merely measured
    assert effect.calls > 0
    assert effect.added_ns_total > 0
    assert effect.inner_ns_total > 0
    assert 0.0 < effect.added_ns_p50 <= effect.added_ns_p99 <= effect.added_ns_max
    assert effect.window_n > 0
    assert effect.budget is not None
    assert "instrument effect" in effect.render()
    assert effect.__canonical__()["added_ns_p99"] == effect.added_ns_p99

    # the self-report is the recorder segment only, so it must be below the end-to-end number and
    # within one Python call layer of it. Disagreement in either direction is a real defect.
    assert effect.added_ns_p50 <= added_ns * 1.5, (
        f"the tap reports {effect.added_ns_p50:.0f} ns of its own cost but the host pays "
        f"{added_ns:.0f} ns; a self-report above the end-to-end number is double counting"
    )


def test_overhead_composes_as_a_budget_term():
    """The overhead is a term in the uncertainty budget, not a footnote.

    Type B rectangular, because an uncorrected bias is a bound rather than a standard deviation:
    the tap cannot give back the time it took. The half-width is the p99 of the per-call fraction
    of wall time the tap added, which is dimensionless and therefore composes against any reading
    whose sensitivity to a timing perturbation the caller can state.
    """
    _added, effect = _measure_added_ns(reps=3, n=2000)
    term = effect.as_term()

    assert isinstance(term, BudgetTerm)
    assert term.kind == "B"
    assert term.distribution == "rectangular"
    assert term.value == pytest.approx(effect.fraction_p99 / 3**0.5)
    assert "uncorrected instrument perturbation" in term.note

    other = BudgetTerm(name="grader_replication", value=0.04, kind="A", dof=19)
    budget = UncertaintyBudget(terms=(other, term))
    assert budget.combined == pytest.approx((other.variance + term.variance) ** 0.5)
    assert sum(budget.shares().values()) == pytest.approx(1.0)
    assert budget.dominant is not None
    assert "instrument_overhead" in budget.render()

    # sensitivity stays separate from value, so a reading with no timing dependence drops the term
    # arithmetically rather than by being left out of the table
    zero = effect.as_term(sensitivity=0.0)
    assert zero.contribution == 0.0
    assert UncertaintyBudget(terms=(other, zero)).combined == pytest.approx(other.value)


# ---------------------------------------------------------------------------
# The clause, part 3: a deliberate breach disables the tap and records it
# ---------------------------------------------------------------------------


def test_latency_breach_disables_the_tap_and_records_it():
    """A budget nothing could meet, and the host does not notice that the tap gave up.

    100 ns at the p99 is below the cost of the record construction alone, so this breaches on the
    first check by construction rather than by timing luck.
    """
    run = fresh_run()
    payload = {"score": 1.0}
    budget = TapBudget(
        max_added_latency_ms_p99=0.0001,  # 100 ns
        max_resident_bytes=64 * 1024 * 1024,
        max_added_alloc_bytes_per_step=64 * 1024 * 1024,
    )
    wrapped = instrument_grader(
        lambda: payload, run=run, budget=budget, window=64, check_stride=1, name="doomed"
    )

    for _ in range(20):
        assert wrapped() is payload  # the host keeps getting the right answer throughout

    guard = wrapped.guard
    assert guard.disabled is True
    assert len(guard.breaches) >= 1
    breach = guard.breaches[0]
    assert breach.kind is BreachKind.LATENCY_P99
    assert breach.limit == pytest.approx(0.0001)
    assert breach.observed > breach.limit
    assert breach.at_call >= 1
    assert "p99 over the last" in breach.render()

    effect = wrapped.effect()
    assert effect.enabled is False
    assert effect.breaches[0].kind is BreachKind.LATENCY_P99
    assert "BREACH" in effect.render()
    assert "DISABLED" in effect.render()
    assert effect.as_term().note.endswith("tap disabled after a breach")

    # and it stopped recording, having recorded why
    before = len(run.ring.peek())
    for _ in range(50):
        assert wrapped() is payload
    assert len(run.ring.peek()) == before


def test_resident_bytes_breach_disables_the_tap():
    """The memory ceiling, breached by a ring the tap was handed rather than one it sized."""
    run = SimpleRun(run_id="fat", ring=TapRing(capacity=4096))
    budget = TapBudget(
        max_added_latency_ms_p99=1000.0,
        max_resident_bytes=10 * RECORD_FOOTPRINT_BYTES,
        max_added_alloc_bytes_per_step=64 * 1024 * 1024,
    )
    wrapped = instrument_grader(lambda: 1.0, run=run, budget=budget, window=8)

    for _ in range(200):
        assert wrapped() == 1.0

    assert wrapped.guard.disabled is True
    kinds = {b.kind for b in wrapped.guard.breaches}
    assert BreachKind.RESIDENT_BYTES in kinds
    assert len(run.ring) < 200


def test_ring_sized_from_the_byte_budget_cannot_breach_residency():
    """``TapRing.for_bytes`` turns the byte ceiling into a capacity, enforced by construction.

    ``reserve`` is the part that is easy to leave out and expensive to leave out. A ring sized on
    the whole budget breaches it the moment it fills, because the guard's two latency windows are
    also the tap's memory. The first version of this file did leave it out, and this test is what
    found it.
    """
    reserve = TapGuard.fixed_bytes(64)
    budget = TapBudget(
        max_added_latency_ms_p99=1000.0,
        max_resident_bytes=100 * RECORD_FOOTPRINT_BYTES + reserve,
        max_added_alloc_bytes_per_step=1 << 30,
    )
    ring = TapRing.for_bytes(budget.max_resident_bytes, reserve=reserve)
    assert ring.capacity == 100

    run = SimpleRun(run_id="sized", ring=ring)
    wrapped = instrument_grader(lambda: 1.0, run=run, budget=budget, window=64)
    for _ in range(500):
        assert wrapped() == 1.0

    assert wrapped.guard.disabled is False
    stats = ring.stats()
    assert stats.resident_bytes + reserve <= budget.max_resident_bytes
    assert stats.dropped_full == 400  # the ring dropped rather than breaching, and counted it

    # and the same budget with no reserve buys a ring the guard's own arrays push over the ceiling
    naive = TapRing.for_bytes(budget.max_resident_bytes)
    assert naive.capacity > ring.capacity
    assert naive.capacity * RECORD_FOOTPRINT_BYTES + reserve > budget.max_resident_bytes


def test_allocation_budget_is_measured_when_tracing_and_named_unchecked_when_not():
    """A check that could not run is named, never counted as a pass."""
    run = fresh_run()
    quiet = instrument_grader(lambda: 1.0, run=run, budget=GENEROUS, measure_alloc="auto")
    for _ in range(10):
        quiet()
    effect = quiet.effect()
    assert effect.added_alloc_bytes_this_step is None
    assert any("added allocation" in u for u in effect.unchecked)

    run2 = fresh_run()
    loud = instrument_grader(lambda: 1.0, run=run2, budget=GENEROUS, measure_alloc="always")
    import tracemalloc

    tracemalloc.start()
    try:
        for _ in range(50):
            loud()
        effect2 = loud.effect()
    finally:
        tracemalloc.stop()

    assert effect2.added_alloc_bytes_this_step is not None
    assert effect2.added_alloc_bytes_this_step > 0
    assert not any("added allocation" in u for u in effect2.unchecked)


def test_on_breach_raise_is_available_and_is_the_only_setting_that_reaches_the_host():
    """It exists for a harness. The docstring says it is never correct in a training loop."""
    budget = TapBudget(
        max_added_latency_ms_p99=0.0001,
        max_resident_bytes=1 << 20,
        max_added_alloc_bytes_per_step=1 << 20,
        on_breach="raise",
    )
    wrapped = instrument_grader(
        lambda: 1.0, run=fresh_run(), budget=budget, window=8, check_stride=1
    )
    with pytest.raises(TapBudgetExceeded, match="latency_p99"):
        for _ in range(10):
            wrapped()


# ---------------------------------------------------------------------------
# The recorder-exception ledger
# ---------------------------------------------------------------------------


class ExplodingRing(TapRing):
    """A ring whose ``offer`` fails, which is a real seam a recorder can fail at."""

    def __init__(self, exceptions):
        super().__init__(capacity=64)
        self._queue = list(exceptions)
        self.attempts = 0

    def offer(self, record):
        self.attempts += 1
        if self._queue:
            raise self._queue.pop(0)
        return super().offer(record)


def test_three_distinct_recorder_exceptions_disable_the_tap_and_the_host_sees_none():
    """The Plane A contract's hardest clause, asserted from the host's side.

    Three different failures inside the recorder, and the wrapped callable returns its own value
    every single time. Nothing is raised, nothing is substituted, nothing is dropped from the
    return path.
    """
    payload = ["a", "b"]
    ring = ExplodingRing(
        [
            ValueError("first distinct"),
            TypeError("second distinct"),
            KeyError("third distinct"),
            RuntimeError("would be a fourth, and must never be reached"),
        ]
    )
    run = SimpleRun(run_id="exploding", ring=ring)
    wrapped = instrument_grader(lambda: payload, run=run, budget=GENEROUS, name="fragile")

    for _ in range(10):
        assert wrapped() is payload

    guard = wrapped.guard
    assert guard.recorder_exceptions == DISTINCT_EXCEPTION_LIMIT
    assert len(guard.distinct_recorder_exceptions) == DISTINCT_EXCEPTION_LIMIT
    assert guard.disabled is True
    assert [k.split(":")[0] for k in guard.distinct_recorder_exceptions] == [
        "ValueError",
        "TypeError",
        "KeyError",
    ]
    breaches = [b for b in guard.breaches if b.kind is BreachKind.RECORDER_EXCEPTIONS]
    assert len(breaches) == 1
    assert breaches[0].observed == DISTINCT_EXCEPTION_LIMIT
    assert "third distinct" in breaches[0].detail
    # the fourth exception was never reached, because the tap stopped calling the ring
    assert ring.attempts == DISTINCT_EXCEPTION_LIMIT


def test_one_recorder_exception_repeated_does_not_disable_the_tap():
    """One bug firing on every call is one bug. Disabling on it would make the tap useless."""

    class AlwaysSame(TapRing):
        def offer(self, record):
            raise ValueError("the same failure, every time")

    run = SimpleRun(run_id="repeat", ring=AlwaysSame(capacity=8))
    wrapped = instrument_grader(lambda: 7.0, run=run, budget=GENEROUS)
    for _ in range(100):
        assert wrapped() == 7.0

    assert wrapped.guard.recorder_exceptions == 100
    assert len(wrapped.guard.distinct_recorder_exceptions) == 1
    assert wrapped.guard.disabled is False


def test_tap_guard_swallows_a_plane_a_body_that_raises():
    """``tap(fn, budget)`` on its own: the body's failure never reaches the caller."""
    calls = []

    def body(x):
        calls.append(x)
        raise ZeroDivisionError("a reduction went wrong")

    guarded = tap(body, GENEROUS)
    for i in range(5):
        assert guarded(i) is None  # no exception, and no value, because the body produced none

    assert calls == [0, 1, 2, 3, 4]
    assert guarded.guard.recorder_exceptions == 5
    assert len(guarded.guard.distinct_recorder_exceptions) == 1
    assert guarded.guard.disabled is False


# ---------------------------------------------------------------------------
# The ring
# ---------------------------------------------------------------------------


def test_full_ring_drops_and_counts_rather_than_blocking_or_growing():
    """A silent drop is the failure this library exists to prevent, so it is counted."""
    run = SimpleRun(run_id="tight", ring=TapRing(capacity=4))
    wrapped = instrument_grader(lambda: 1.0, run=run, budget=GENEROUS)
    for _ in range(10):
        assert wrapped() == 1.0

    stats = run.ring.stats()
    assert stats.capacity == 4
    assert stats.size == 4
    assert stats.accepted == 4
    assert stats.dropped_full == 6
    assert stats.dropped == 6
    assert stats.offered == 10
    assert stats.drop_rate == pytest.approx(0.6)
    assert stats.high_water == 4
    assert "DROPPED 6" in stats.render()

    # the drop count reaches the instrument effect, so a reading built on this record can see it
    effect = wrapped.effect()
    assert effect.ring is not None
    assert effect.ring.dropped_full == 6
    assert effect.__canonical__()["ring"]["dropped_full"] == 6

    # and the ring never grew past its declared capacity
    assert len(run.ring) == 4


def test_contended_write_drops_immediately_rather_than_waiting():
    """The lock is never waited on. A producer that loses the race counts a drop and returns."""
    ring = TapRing(capacity=64)
    run = SimpleRun(run_id="contended", ring=ring)
    wrapped = instrument_grader(lambda: 1.0, run=run, budget=GENEROUS)

    held = threading.Event()
    release = threading.Event()

    def hold_the_lock():
        ring._lock.acquire()
        held.set()
        release.wait(5.0)
        ring._lock.release()

    t = threading.Thread(target=hold_the_lock, daemon=True)
    t.start()
    assert held.wait(5.0)

    t0 = time.perf_counter()
    for _ in range(20):
        assert wrapped() == 1.0
    elapsed = time.perf_counter() - t0
    release.set()
    t.join(5.0)

    assert elapsed < 0.5, "a contended offer waited on the lock instead of dropping"
    stats = ring.stats()
    assert stats.dropped_contended == 20
    assert stats.accepted == 0
    assert "contended" in stats.render()


def test_drain_frees_the_references_the_ring_was_pinning():
    """Retaining the host's objects is the memory cost of fidelity, and it is reported."""
    run = SimpleRun(run_id="pin", ring=TapRing(capacity=32))
    wrapped = instrument_grader(lambda payload: payload, run=run, budget=GENEROUS)
    for i in range(5):
        wrapped([i] * 100)

    before = run.ring.stats()
    assert before.retained_refs == 5 * 2  # one arg plus one return value per call
    assert before.resident_bytes == 5 * RECORD_FOOTPRINT_BYTES

    drained = run.ring.drain()
    assert len(drained) == 5
    after = run.ring.stats()
    assert after.size == 0
    assert after.retained_refs == 0
    assert after.drained == 5
    assert after.accepted == 5  # the cumulative counter survives the drain


def test_retain_args_false_drops_the_references_and_keeps_the_facets():
    """The setting for a long run whose ring would otherwise pin large completion lists."""
    run = SimpleRun(run_id="lean", ring=TapRing(capacity=32))
    wrapped = instrument_grader(
        lambda prompts, seed=None: [1.0, 2.0],
        run=run,
        budget=GENEROUS,
        retain_args=False,
        retain_value=False,
        facet_keys=("seed",),
    )
    wrapped(["a"] * 1000, seed=17)

    record = run.ring.peek()[0]
    assert record.args is None
    assert record.kwargs is None
    assert record.value is None
    assert record.facets == {"seed": 17}
    # the shape survives even though the object did not, so the component count is still known
    assert record.shape.kind == "sequence"
    assert record.shape.length == 2
    assert run.ring.stats().retained_refs == 0


# ---------------------------------------------------------------------------
# The record footprint constant, measured rather than asserted
# ---------------------------------------------------------------------------


def test_record_footprint_constant_is_not_an_underestimate():
    """``RECORD_FOOTPRINT_BYTES`` turns a byte budget into a ring capacity, so it must not lie low.

    Measured with ``tracemalloc`` around the construction of a batch of records, which counts the
    record objects and the ``ReturnShape`` beside them and not the host objects they point at.
    """
    import tracemalloc

    from reward_lens.tap.contract import GraderCall, classify_return

    shape = classify_return([0.5] * 8)
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    keep = [
        GraderCall(i, "run", "g", CallOutcome.RETURNED, 1, 1, None, None, None, shape)
        for i in range(2000)
    ]
    used = tracemalloc.get_traced_memory()[0] - base
    tracemalloc.stop()

    per_record = used / len(keep)
    assert per_record <= RECORD_FOOTPRINT_BYTES, (
        f"one record costs {per_record:.0f} bytes but RECORD_FOOTPRINT_BYTES claims "
        f"{RECORD_FOOTPRINT_BYTES}; a ring sized from a byte budget would overshoot it"
    )
    assert per_record > RECORD_FOOTPRINT_BYTES / 4, (
        f"one record costs {per_record:.0f} bytes against a claimed {RECORD_FOOTPRINT_BYTES}; the "
        f"constant is so conservative that a byte budget buys a quarter of the ring it should"
    )
    del keep

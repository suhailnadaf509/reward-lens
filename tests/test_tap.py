"""Unit tests for `reward_lens.tap`: the contract types, the ring, and the guard.

The clause lives in `tests/acceptance/test_w3_1_tap.py`. This file covers the parts
underneath it: shape classification across the return types the ecosystem actually produces, the
ring's arithmetic and its two drop causes, the guard's window and its exact constant-time gate on
the p99 check, the downsample branch, and the `RunHandle` protocol the record layer has to satisfy.
"""

from __future__ import annotations

import math
import subprocess
import sys
import threading

import numpy as np
import pytest

from reward_lens.core.provenance import Cost
from reward_lens.tap.contract import (
    DEFAULT,
    RECORD_FOOTPRINT_BYTES,
    BreachKind,
    CallOutcome,
    GraderCall,
    InstrumentEffect,
    ReturnShape,
    RunHandle,
    SimpleRun,
    TapBreach,
    TapBudget,
    classify_return,
    is_sentinel,
    quantile,
)
from reward_lens.tap.grader_wrap import TapGuard, instrument_grader, tap
from reward_lens.tap.ring import RingStats, TapRing

GENEROUS = TapBudget(
    max_added_latency_ms_p99=1000.0,
    max_resident_bytes=64 * 1024 * 1024,
    max_added_alloc_bytes_per_step=64 * 1024 * 1024,
)


# ---------------------------------------------------------------------------
# TapBudget
# ---------------------------------------------------------------------------


class TestTapBudget:
    def test_the_default_is_a_real_declaration(self):
        assert DEFAULT.max_added_latency_ms_p99 == 1.0
        assert DEFAULT.max_resident_bytes == 64 * 1024 * 1024
        assert DEFAULT.on_breach == "disable"
        assert DEFAULT.max_added_latency_ns_p99 == 1e6

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_added_latency_ms_p99": 0.0},
            {"max_added_latency_ms_p99": -1.0},
            {"max_resident_bytes": 0},
            {"max_added_alloc_bytes_per_step": -1},
            {"on_breach": "shrug"},
        ],
    )
    def test_an_unmeetable_or_meaningless_budget_is_rejected_at_construction(self, kwargs):
        base = {
            "max_added_latency_ms_p99": 1.0,
            "max_resident_bytes": 1024,
            "max_added_alloc_bytes_per_step": 1024,
        }
        with pytest.raises(ValueError):
            TapBudget(**{**base, **kwargs})

    def test_canonical_form_round_trips_the_four_declared_fields(self):
        assert set(DEFAULT.__canonical__()) == {
            "max_added_latency_ms_p99",
            "max_resident_bytes",
            "max_added_alloc_bytes_per_step",
            "on_breach",
        }


# ---------------------------------------------------------------------------
# classify_return / is_sentinel
# ---------------------------------------------------------------------------


class TestClassifyReturn:
    @pytest.mark.parametrize(
        "value,kind,length",
        [
            (None, "none", None),
            (1.0, "scalar", None),
            (3, "scalar", None),
            (True, "scalar", None),
            (np.float64(0.5), "scalar", None),
            (np.float32(0.5), "scalar", None),
            (np.int64(7), "scalar", None),
            ([1.0, 2.0], "sequence", 2),
            ((1.0, 2.0, 3.0), "sequence", 3),
            ({"a": 1.0}, "mapping", 1),
            (b"xy", "sequence", 2),
            ("text", "other", None),
            (object(), "other", None),
        ],
    )
    def test_kinds(self, value, kind, length):
        shape = classify_return(value)
        assert shape.kind == kind
        assert shape.length == length

    def test_a_string_is_not_a_sequence_of_scores(self):
        """Iterating a string yields characters, which would be a component census of nonsense."""
        assert classify_return("hello").kind == "other"

    def test_mapping_keys_are_read_and_capped(self):
        big = {f"k{i}": float(i) for i in range(400)}
        shape = classify_return(big, key_cap=16)
        assert shape.kind == "mapping"
        assert shape.length == 400
        assert len(shape.keys) == 16
        assert shape.truncated is True

    def test_a_custom_mapping_is_recognised_through_the_abc(self):
        from collections import OrderedDict

        shape = classify_return(OrderedDict(a=1.0, b=2.0))
        assert shape.kind == "mapping"
        assert shape.keys == ("a", "b")

    def test_a_coroutine_return_is_flagged_rather_than_called_a_score(self):
        """A sync wrapper around an async grader would otherwise record a meaningless latency."""

        async def f():
            return 1.0

        coro = f()
        try:
            assert classify_return(coro).kind == "awaitable"
        finally:
            coro.close()

    def test_return_shape_is_a_namedtuple_so_it_is_cheap_and_immutable(self):
        shape = classify_return([1.0])
        assert isinstance(shape, tuple)
        with pytest.raises(AttributeError):
            shape.kind = "mapping"

    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, True),
            (float("nan"), True),
            (np.float64("nan"), True),
            (0.0, False),
            (1.0, False),
            ([float("nan")], False),  # elementwise is the drain's job, not the hot path's
            ({"a": None}, False),
        ],
    )
    def test_is_sentinel(self, value, expected):
        assert is_sentinel(value, classify_return(value)) is expected

    def test_is_sentinel_survives_a_value_whose_equality_raises(self):
        class Hostile:
            def __ne__(self, other):
                raise RuntimeError("no comparisons here")

            def __float__(self):
                return 1.0

        v = Hostile()
        assert is_sentinel(v, classify_return(v)) is False


# ---------------------------------------------------------------------------
# GraderCall
# ---------------------------------------------------------------------------


class TestGraderCall:
    def _call(self, value):
        return GraderCall(
            1, "run", "g", CallOutcome.RETURNED, 1000, 50, None, None, value, classify_return(value)
        )

    def test_components_of_a_mapping_are_every_key(self):
        rec = self._call({"correct": 1.0, "format": 0.0, "length": -1.0})
        assert dict(rec.components()) == {"correct": 1.0, "format": 0.0, "length": -1.0}

    def test_components_of_a_sequence_are_indexed(self):
        rec = self._call([0.1, None, 0.3])
        assert rec.components() == (("0", 0.1), ("1", None), ("2", 0.3))

    def test_components_of_a_scalar_is_one_unnamed_component(self):
        assert self._call(0.75).components() == (("", 0.75),)

    def test_components_are_capped(self):
        rec = self._call(list(range(500)))
        assert len(rec.components(cap=10)) == 10
        assert rec.shape.length == 500  # so the truncation is visible

    def test_a_raised_call_has_no_components_and_no_value(self):
        rec = GraderCall(1, "run", "g", CallOutcome.RAISED, 10, 1, error_type="ValueError")
        assert rec.components() == ()
        assert rec.raised is True

    def test_cost_is_the_graders_own_wall_time(self):
        rec = self._call(1.0)
        assert isinstance(rec.cost, Cost)
        assert rec.cost.wall_seconds == pytest.approx(1e-6)
        assert (rec.cost + rec.cost).wall_seconds == pytest.approx(2e-6)

    def test_canonical_form_carries_the_measured_part_and_not_the_host_objects(self):
        rec = self._call({"a": 1.0})
        rec.args = (["a very large prompt list"],)
        env = rec.__canonical__()
        assert env["outcome"] == "returned"
        assert env["shape"]["kind"] == "mapping"
        assert "args" not in env and "value" not in env


# ---------------------------------------------------------------------------
# quantile
# ---------------------------------------------------------------------------


class TestQuantile:
    def test_matches_numpy_linear_interpolation(self):
        for n in (1, 2, 3, 7, 64, 1000):
            xs = sorted(np.random.default_rng(n).normal(size=n).tolist())
            for q in (0.0, 0.5, 0.9, 0.99, 1.0):
                assert quantile(xs, q) == pytest.approx(float(np.quantile(xs, q)))

    def test_empty_is_zero_rather_than_an_exception(self):
        assert quantile([], 0.99) == 0.0


# ---------------------------------------------------------------------------
# TapRing
# ---------------------------------------------------------------------------


class TestTapRing:
    def _rec(self, i=0):
        return GraderCall(i, "r", "g", CallOutcome.RETURNED, 1, 1)

    def test_a_zero_capacity_ring_is_refused(self):
        with pytest.raises(ValueError, match="at least one slot"):
            TapRing(capacity=0)

    def test_offer_returns_whether_the_record_landed(self):
        ring = TapRing(capacity=2)
        assert ring.offer(self._rec(0)) is True
        assert ring.offer(self._rec(1)) is True
        assert ring.offer(self._rec(2)) is False
        assert len(ring) == 2

    def test_it_drops_the_newest_so_what_survives_is_contiguous(self):
        """Dropping the oldest would evict a record a drain may be reading and leave a gap in the
        middle; refusing the write leaves a contiguous prefix and a counted tail."""
        ring = TapRing(capacity=3)
        for i in range(6):
            ring.offer(self._rec(i))
        assert [r.seq for r in ring.peek()] == [0, 1, 2]

    def test_drain_takes_oldest_first_and_can_be_partial(self):
        ring = TapRing(capacity=8)
        for i in range(5):
            ring.offer(self._rec(i))
        first = ring.drain(2)
        assert [r.seq for r in first] == [0, 1]
        assert [r.seq for r in ring.peek()] == [2, 3, 4]
        assert ring.stats().drained == 2
        assert [r.seq for r in ring.drain()] == [2, 3, 4]
        assert len(ring) == 0

    def test_drain_to_hands_records_to_a_sink_outside_the_lock(self):
        ring = TapRing(capacity=8)
        for i in range(3):
            ring.offer(self._rec(i))
        seen = []

        def sink(rec):
            # re-entering the ring from the sink would deadlock if the drain held its lock
            ring.offer(self._rec(100 + rec.seq))
            seen.append(rec.seq)

        assert ring.drain_to(sink) == 3
        assert seen == [0, 1, 2]
        assert [r.seq for r in ring.peek()] == [100, 101, 102]

    def test_extend_from_puts_records_back_for_a_drain_that_failed(self):
        ring = TapRing(capacity=2)
        taken = [self._rec(0), self._rec(1), self._rec(2)]
        assert ring.extend_from(taken) == 2
        assert ring.stats().dropped_full == 1

    def test_close_stops_acceptance_and_counts_the_refusals_separately(self):
        ring = TapRing(capacity=4)
        ring.offer(self._rec(0))
        ring.close()
        assert ring.offer(self._rec(1)) is False
        stats = ring.stats()
        assert stats.dropped_closed == 1
        assert stats.dropped_full == 0
        assert stats.accepted == 1

    def test_a_contended_offer_drops_instead_of_waiting(self):
        ring = TapRing(capacity=4)
        ring._lock.acquire()
        try:
            assert ring.offer(self._rec(0)) is False
        finally:
            ring._lock.release()
        assert ring.stats().dropped_contended == 1

    def test_high_water_records_how_close_the_capacity_came_to_being_a_problem(self):
        ring = TapRing(capacity=10)
        for i in range(9):
            ring.offer(self._rec(i))
        ring.drain()
        for i in range(2):
            ring.offer(self._rec(i))
        assert ring.stats().high_water == 9

    def test_for_bytes_subtracts_the_reserve(self):
        assert TapRing.for_bytes(10 * RECORD_FOOTPRINT_BYTES).capacity == 10
        assert (
            TapRing.for_bytes(
                10 * RECORD_FOOTPRINT_BYTES, reserve=RECORD_FOOTPRINT_BYTES * 3
            ).capacity
            == 7
        )
        assert TapRing.for_bytes(1).capacity == 1  # never zero, which would be a silent switch-off

    def test_stats_arithmetic(self):
        s = RingStats(capacity=8, size=2, accepted=6, dropped_full=3, dropped_contended=1)
        assert s.dropped == 4
        assert s.offered == 10
        assert s.drop_rate == pytest.approx(0.4)
        assert s.__canonical__()["dropped_contended"] == 1
        assert RingStats(capacity=1, size=0).drop_rate == 0.0

    def test_retained_refs_counts_the_host_objects_the_ring_is_pinning(self):
        ring = TapRing(capacity=8)
        ring.offer(GraderCall(1, "r", "g", CallOutcome.RETURNED, 1, 1, ("a", "b"), {"k": 1}, [1.0]))
        assert ring.stats().retained_refs == 4  # two args, one kwarg, one return value

    def test_concurrent_producers_never_exceed_capacity(self):
        ring = TapRing(capacity=50)
        errors: list[BaseException] = []

        def produce():
            try:
                for i in range(500):
                    ring.offer(self._rec(i))
            except BaseException as exc:  # a ring that raises into a producer is the bug
                errors.append(exc)

        threads = [threading.Thread(target=produce) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        assert not errors
        assert len(ring) == 50
        stats = ring.stats()
        assert stats.accepted == 50
        assert stats.accepted + stats.dropped == 4000


# ---------------------------------------------------------------------------
# TapGuard
# ---------------------------------------------------------------------------


class TestTapGuard:
    def test_rejects_a_nonsensical_configuration(self):
        with pytest.raises(ValueError):
            TapGuard("g", GENEROUS, window=0)
        with pytest.raises(ValueError):
            TapGuard("g", GENEROUS, check_stride=0)
        with pytest.raises(ValueError, match="measure_alloc"):
            TapGuard("g", GENEROUS, measure_alloc="sometimes")

    def test_window_wraps_and_reports_quantiles_over_the_live_window(self):
        g = TapGuard("g", GENEROUS, window=4)
        for v in (10, 20, 30, 40, 50):
            g.observe(v, 1000)
        p50, p99, frac = g.window_quantiles()
        assert g.calls == 5
        assert g._window_n == 4
        assert p50 == pytest.approx(35.0)  # 20,30,40,50 with the 10 overwritten
        assert p99 == pytest.approx(49.7)
        assert 0.0 < frac < 1.0

    def test_the_p99_gate_is_exact_rather_than_a_heuristic(self):
        """The constant-time gate must never say "cannot breach" when the real p99 does breach.

        Checked against the quantile it is standing in for, over every window fill and every count
        of over-limit samples. This is the assertion that lets the gate sit on the hot path.
        """
        limit_ns = 1000.0
        budget = TapBudget(
            max_added_latency_ms_p99=limit_ns / 1e6,
            max_resident_bytes=1 << 20,
            max_added_alloc_bytes_per_step=1 << 20,
        )
        for window in (1, 2, 8, 64, 101, 256):
            for over in range(0, window + 1):
                g = TapGuard("g", budget, window=window)
                for i in range(window):
                    g.observe(5000 if i < over else 10, 1000)
                _p50, p99, _f = g.window_quantiles()
                truth = p99 > limit_ns
                gate = g.latency_p99_can_breach()
                assert gate or not truth, (
                    f"window={window} over={over}: the gate said no breach was possible but the "
                    f"p99 is {p99}"
                )

    def test_over_limit_bookkeeping_survives_the_window_wrapping(self):
        budget = TapBudget(
            max_added_latency_ms_p99=0.001,  # 1000 ns
            max_resident_bytes=1 << 20,
            max_added_alloc_bytes_per_step=1 << 20,
        )
        g = TapGuard("g", budget, window=4)
        for _ in range(4):
            g.observe(5000, 100)
        assert g._over_limit == 4
        for _ in range(4):
            g.observe(10, 100)
        assert g._over_limit == 0

    def test_three_distinct_exceptions_disable_and_a_repeat_does_not(self):
        g = TapGuard("g", GENEROUS)
        assert g.note_recorder_exception(ValueError("a")) is None
        for _ in range(50):
            assert g.note_recorder_exception(ValueError("a")) is None
        assert g.note_recorder_exception(TypeError("b")) is None
        assert g.disabled is False
        breach = g.note_recorder_exception(KeyError("c"))
        assert breach is not None
        assert breach.kind is BreachKind.RECORDER_EXCEPTIONS
        assert g.disabled is True
        assert g.recorder_exceptions == 53
        assert len(g.distinct_recorder_exceptions) == 3

    def test_distinctness_is_type_plus_the_first_line_of_the_message(self):
        g = TapGuard("g", GENEROUS)
        g.note_recorder_exception(ValueError("shape mismatch\nand a long traceback tail"))
        g.note_recorder_exception(ValueError("shape mismatch\na different tail entirely"))
        assert len(g.distinct_recorder_exceptions) == 1

    def test_the_breach_list_is_bounded(self):
        g = TapGuard("g", GENEROUS)
        for i in range(100):
            g._record_breach(
                TapBreach(kind=BreachKind.LATENCY_P99, observed=1.0, limit=0.5, at_call=i)
            )
        assert len(g.breaches) == 16

    def test_downsample_halves_the_sampling_rate_then_gives_up(self):
        budget = TapBudget(
            max_added_latency_ms_p99=1e-6,  # 1 ns: unmeetable by construction
            max_resident_bytes=1 << 20,
            max_added_alloc_bytes_per_step=1 << 20,
            on_breach="downsample",
        )
        g = TapGuard("g", budget, window=8, check_stride=1)
        ring = TapRing(capacity=8)
        rates = []
        for _ in range(12):
            g.observe(5000, 1000)
            g.check(ring)
            rates.append(g._sample_every)
        assert rates[0] == 2
        assert rates == sorted(rates)
        assert (
            g.disabled is True
        )  # it stopped doubling and switched off rather than sampling 1/2048

    def test_should_sample_respects_the_rate(self):
        g = TapGuard("g", GENEROUS)
        assert all(g.should_sample() for _ in range(10))
        g._sample_every = 4
        taken = [g.should_sample() for _ in range(12)]
        assert sum(taken) == 3

    def test_fixed_bytes_is_what_for_bytes_has_to_reserve(self):
        g = TapGuard("g", GENEROUS, window=256)
        assert g.resident_bytes(TapRing(capacity=4)) == TapGuard.fixed_bytes(256)
        assert TapGuard.fixed_bytes(1024) == 2 * 1024 * 8

    def test_unchecked_names_the_allocation_check_when_it_could_not_run(self):
        off = TapGuard("g", GENEROUS, measure_alloc="off")
        assert any("measure_alloc='off'" in u for u in off.unchecked())
        auto = TapGuard("g", GENEROUS, measure_alloc="auto")
        assert any("tracemalloc is not tracing" in u for u in auto.unchecked())

    def test_per_step_allocation_resets_when_the_step_moves(self):
        g = TapGuard("g", GENEROUS)
        g.observe_alloc(100, step=0)
        g.observe_alloc(50, step=0)
        assert g._alloc_this_step == 150
        g.observe_alloc(10, step=1)
        assert g._alloc_this_step == 10
        assert g._alloc_step == 1


# ---------------------------------------------------------------------------
# InstrumentEffect
# ---------------------------------------------------------------------------


class TestInstrumentEffect:
    def test_an_effect_with_no_calls_renders_and_composes_to_nothing(self):
        eff = InstrumentEffect(tap_name="t", run_id="r", calls=0)
        assert eff.fraction_mean == 0.0
        assert eff.as_term().value == 0.0
        assert "0 calls" in eff.render()

    def test_fraction_mean_is_added_over_inner(self):
        eff = InstrumentEffect(
            tap_name="t", run_id="r", calls=10, added_ns_total=500, inner_ns_total=10_000
        )
        assert eff.fraction_mean == pytest.approx(0.05)

    def test_as_term_is_type_b_rectangular_on_the_p99_fraction(self):
        eff = InstrumentEffect(tap_name="t", run_id="r", calls=100, fraction_p99=0.12, window_n=100)
        term = eff.as_term()
        assert term.kind == "B"
        assert term.distribution == "rectangular"
        assert term.value == pytest.approx(0.12 / math.sqrt(3))
        assert "100 calls" in term.note

    def test_render_names_a_breach_and_the_disable(self):
        eff = InstrumentEffect(
            tap_name="t",
            run_id="r",
            calls=5,
            enabled=False,
            breaches=(TapBreach(kind=BreachKind.LATENCY_P99, observed=4.2, limit=1.0, at_call=3),),
            unchecked=("added allocation",),
        )
        out = eff.render()
        assert "BREACH latency_p99" in out
        assert "DISABLED" in out
        assert "not checked: added allocation" in out

    def test_as_step_record_matches_the_record_schemas_field_set(self):
        """The projection onto `record.schema.InstrumentEffect`, checked against the real one.

        Two types with one name is worth pinning rather than hoping about: this asserts the key
        set matches exactly and that the round trip through `from_canonical` reproduces the
        numbers, so a rename on either side fails here instead of at a step boundary.

        Skipped rather than failed when `record/` is absent. `tap/` does not depend on it, by
        design: a hot-path package that imports a serialisation layer has the dependency arrow
        backwards, and this test is a cross-check on a neighbour rather than a requirement of this
        one.
        """
        schema = pytest.importorskip("reward_lens.record.schema")
        StepEffect = schema.InstrumentEffect

        eff = InstrumentEffect(
            tap_name="t",
            run_id="r",
            calls=1000,
            added_ns_p50=800.0,
            added_ns_p99=1200.0,
            resident_bytes=4096,
            added_alloc_bytes_this_step=2048,
            recorder_exceptions=2,
            enabled=False,
            breaches=(
                TapBreach(kind=BreachKind.LATENCY_P99, observed=4.2, limit=1.0, at_call=900),
            ),
        )
        payload = eff.as_step_record()
        assert set(payload) == set(StepEffect().__canonical__())

        step = StepEffect.from_canonical(payload)
        assert step.added_latency_ms_p50 == pytest.approx(0.0008)
        assert step.added_latency_ms_p99 == pytest.approx(0.0012)
        assert step.added_resident_bytes == 4096
        assert step.added_alloc_bytes == 2048
        assert step.invocations == 1000
        assert step.exceptions == 2
        assert step.disabled is True
        assert "latency_p99" in step.breach

    def test_as_step_record_loses_the_ring_counters_and_the_unchecked_list(self):
        """Named in the docstring, asserted here, so the loss is a known gap and not a surprise."""
        eff = InstrumentEffect(
            tap_name="t",
            run_id="r",
            calls=10,
            ring=RingStats(capacity=4, size=4, accepted=4, dropped_full=6),
            unchecked=("added allocation",),
        )
        payload = eff.as_step_record()
        assert not any("drop" in k for k in payload)
        assert not any("unchecked" in k for k in payload)
        assert eff.ring.dropped_full == 6  # still there on the full effect, which is what to keep

    def test_canonical_form_is_json_shaped(self):
        import json

        eff = InstrumentEffect(
            tap_name="t",
            run_id="r",
            calls=2,
            ring=RingStats(capacity=4, size=1),
            budget=DEFAULT,
            breaches=(TapBreach(kind=BreachKind.RESIDENT_BYTES, observed=2, limit=1, at_call=1),),
        )
        blob = json.dumps(eff.__canonical__())
        assert '"resident_bytes"' in blob


# ---------------------------------------------------------------------------
# tap() and instrument_grader() behaviour not covered by the acceptance file
# ---------------------------------------------------------------------------


class TestTapFunction:
    def test_a_healthy_body_returns_its_value(self):
        guarded = tap(lambda x: x * 2, GENEROUS)
        assert guarded(21) == 42
        assert guarded.guard.calls == 1

    def test_a_disabled_tap_stops_calling_the_body_at_all(self):
        calls = []
        guarded = tap(lambda: calls.append(1), GENEROUS)
        guarded()
        guarded.guard.disable(
            TapBreach(kind=BreachKind.LATENCY_P99, observed=1.0, limit=0.5, at_call=1)
        )
        guarded()
        assert len(calls) == 1

    def test_keyboard_interrupt_is_not_swallowed(self):
        """Control flow is not an error, and a monitor that eats Ctrl-C is worse than one that
        crashes."""

        def body():
            raise KeyboardInterrupt

        guarded = tap(body, GENEROUS)
        with pytest.raises(KeyboardInterrupt):
            guarded()

    def test_the_name_defaults_to_the_wrapped_callable(self):
        def reduce_logits():
            return None

        assert tap(reduce_logits, GENEROUS).guard.name == "reduce_logits"
        assert tap(reduce_logits, GENEROUS, name="explicit").guard.name == "explicit"


class TestInstrumentGrader:
    def test_it_works_with_no_run_handle_at_all(self):
        wrapped = instrument_grader(lambda: 1.0, budget=GENEROUS)
        assert wrapped() == 1.0
        assert wrapped.run.run_id == "anonymous"
        assert len(wrapped.ring) == 1

    def test_the_step_is_read_from_the_run_handle(self):
        run = SimpleRun(run_id="r", ring=TapRing(capacity=8), step=0)
        wrapped = instrument_grader(lambda: 1.0, run=run, budget=GENEROUS)
        wrapped()
        run.step = 41
        wrapped()
        assert [r.step for r in run.ring.peek()] == [0, 41]

    def test_sequence_numbers_are_monotone_so_a_drain_can_see_a_gap(self):
        run = SimpleRun(run_id="r", ring=TapRing(capacity=2))
        wrapped = instrument_grader(lambda: 1.0, run=run, budget=GENEROUS)
        for _ in range(5):
            wrapped()
        assert [r.seq for r in run.ring.peek()] == [1, 2]
        assert run.ring.stats().dropped_full == 3

    def test_facets_are_taken_from_the_call_context_and_nothing_is_computed(self):
        run = SimpleRun(run_id="r", ring=TapRing(capacity=8))
        wrapped = instrument_grader(
            lambda **kw: 1.0,
            run=run,
            budget=GENEROUS,
            facet_keys=("seed", "opponent", "rubric_draw", "absent"),
        )
        wrapped(seed=11, opponent="model-b", rubric_draw=3, unrelated=99)
        assert run.ring.peek()[0].facets == {"seed": 11, "opponent": "model-b", "rubric_draw": 3}

    def test_an_open_ended_kwarg_set_is_recorded_without_assuming_a_closed_one(self):
        """TRL splats every remaining dataset column into `reward_kwargs` (`grpo_trainer.py:1618`),
        so the key set is open-ended and the tap keeps whatever arrives."""
        run = SimpleRun(run_id="r", ring=TapRing(capacity=8))
        wrapped = instrument_grader(lambda **kw: [1.0], run=run, budget=GENEROUS)
        wrapped(prompts=["p"], completions=["c"], difficulty=3, cf_rating=1500, hack_group="x")
        kwargs = run.ring.peek()[0].kwargs
        assert set(kwargs) == {"prompts", "completions", "difficulty", "cf_rating", "hack_group"}

    def test_a_timeout_is_a_distinct_outcome_from_a_raise(self):
        run = SimpleRun(run_id="r", ring=TapRing(capacity=8))

        def grader(kind):
            raise TimeoutError("the judge did not answer") if kind else ValueError("bad input")

        wrapped = instrument_grader(grader, run=run, budget=GENEROUS)
        for kind in (True, False):
            with pytest.raises((TimeoutError, ValueError)):
                wrapped(kind)
        assert [r.outcome for r in run.ring.peek()] == [
            CallOutcome.TIMED_OUT,
            CallOutcome.RAISED,
        ]

    def test_a_client_specific_timeout_class_can_be_declared(self):
        class ReadTimeout(Exception):
            pass

        run = SimpleRun(run_id="r", ring=TapRing(capacity=8))

        def grader():
            raise ReadTimeout("httpx style")

        wrapped = instrument_grader(grader, run=run, budget=GENEROUS, timeout_types=(ReadTimeout,))
        with pytest.raises(ReadTimeout):
            wrapped()
        assert run.ring.peek()[0].outcome is CallOutcome.TIMED_OUT

    def test_a_callable_object_with_no_dunder_name_is_wrappable(self):
        class Judge:
            def __call__(self, x):
                return x

        judge = Judge()
        wrapped = instrument_grader(judge, budget=GENEROUS, name="judge")
        assert wrapped(3) == 3
        assert wrapped.guard.name == "judge"

    def test_a_disabled_tap_still_returns_the_grader_value_unchanged(self):
        payload = [1.0]
        wrapped = instrument_grader(lambda: payload, budget=GENEROUS)
        wrapped.guard.disable(
            TapBreach(kind=BreachKind.LATENCY_P99, observed=1.0, limit=0.5, at_call=0)
        )
        assert wrapped() is payload
        assert len(wrapped.ring) == 0

    def test_a_disabled_async_tap_still_awaits_and_returns_unchanged(self):
        import asyncio

        payload = {"a": 1.0}

        async def grader():
            return payload

        wrapped = instrument_grader(grader, budget=GENEROUS)
        wrapped.guard.disable(
            TapBreach(kind=BreachKind.LATENCY_P99, observed=1.0, limit=0.5, at_call=0)
        )
        assert asyncio.run(wrapped()) is payload

    def test_an_async_grader_that_raises_reraises_unchanged(self):
        import asyncio

        run = SimpleRun(run_id="r", ring=TapRing(capacity=8))

        async def grader():
            raise RuntimeError("async boom")

        wrapped = instrument_grader(grader, run=run, budget=GENEROUS)
        with pytest.raises(RuntimeError, match="async boom"):
            asyncio.run(wrapped())
        assert run.ring.peek()[0].outcome is CallOutcome.RAISED


# ---------------------------------------------------------------------------
# The RunHandle protocol and the import contract
# ---------------------------------------------------------------------------


class TestRunHandle:
    def test_simple_run_satisfies_the_protocol(self):
        run = SimpleRun()
        assert isinstance(run, RunHandle)
        assert isinstance(run.ring, TapRing)
        assert run.step is None

    def test_any_object_with_the_three_attributes_satisfies_it(self):
        """The protocol is three attributes and no methods so `record/`'s Run can satisfy it
        without being rewritten around the tap."""

        class ForeignRun:
            def __init__(self):
                self.run_id = "from-elsewhere"
                self.ring = TapRing(capacity=4)
                self.step = 7

        run = ForeignRun()
        assert isinstance(run, RunHandle)
        wrapped = instrument_grader(lambda: 1.0, run=run, budget=GENEROUS)
        wrapped()
        assert run.ring.peek()[0].run_id == "from-elsewhere"
        assert run.ring.peek()[0].step == 7

    def test_a_run_handle_missing_step_still_works_and_records_none(self):
        class Minimal:
            run_id = "minimal"
            ring = TapRing(capacity=4)

        wrapped = instrument_grader(lambda: 1.0, run=Minimal(), budget=GENEROUS)
        wrapped()
        assert Minimal.ring.peek()[0].step is None


def test_importing_the_tap_pulls_no_torch():
    """`tap/` is part of the torch-free core and runs on hosts that never installed it."""
    code = "import reward_lens.tap, sys; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_shape_of_a_return_is_captured_at_call_time_so_later_mutation_is_visible():
    """The tap retains the return value by reference, which is what preserves identity. A host that
    mutates its own return value afterwards would be read wrongly by a lazy component walk, so the
    shape is recorded eagerly and the disagreement is detectable."""
    run = SimpleRun(run_id="r", ring=TapRing(capacity=4))
    payload = [1.0, 2.0]
    wrapped = instrument_grader(lambda: payload, run=run, budget=GENEROUS)
    wrapped()
    payload.append(3.0)

    record = run.ring.peek()[0]
    assert record.shape.length == 2
    assert len(record.components()) == 3
    assert record.shape.length != len(record.components())


def test_return_shape_canonical_form_stringifies_keys():
    shape = ReturnShape("mapping", "dict", 2, (1, "b"), False)
    assert shape.__canonical__()["keys"] == ["1", "b"]

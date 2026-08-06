"""The grader wrapper and the Plane A guard (sections 2.7 and 8.1).

This is the only code in the library that runs inside somebody else's training loop, so its
reliability contract is the inverse of everything else here. Plane B fails closed and loud. Plane A
**fails open**: it never raises into the host, it never blocks, it never grows without bound, and
when it cannot keep its promises it switches itself off and records that it did.

Two functions.

``tap(fn, budget)`` is the general guard. Give it any Plane A body and it comes back wrapped so
that the body cannot raise into the caller and cannot exceed its declared budget. Every invocation
is timed, a breach records a ``TapBreach`` and disables the tap, and a recorder exception is caught,
counted and recorded, with the third *distinct* one disabling the tap.

``instrument_grader(score_fn, run=..., budget=...)`` is the wedge. It wraps a scoring callable so
every call is recorded and returns exactly what the callable returned. That one wrapper enables A1
through A7, B1 through B6, D1 through D10, E1 through E4, I1 through I3, L1 through L3 and M1
through M11, and it needs no framework patch, no GPU and no white-box access.

**Return-value fidelity is the hard part and it is where wrappers like this usually leak.** A
scoring callable may return a float, a list, a dict, a tuple, ``None`` or a numpy scalar, and the
frameworks each expect a different one: TRL iterates the return element by element and maps a
``None`` element to ``torch.nan`` (``grpo_trainer.py:1663``), ``verifiers`` calls ``float()`` on it
(``rubric.py:203``), veRL fans a dict's keys out into ``reward_extra_info``. So the wrapper returns
**the original object**, not a reconstruction. Nothing is coerced, nothing is rounded, nothing goes
through JSON, and the acceptance test asserts identity with ``is`` rather than equality, because
equality would pass for a copy and a copy is the bug.

**Exceptions from the wrapped callable are the opposite case.** They are recorded and then
re-raised unchanged, with a bare ``raise`` so the traceback is the original one. Swallowing them
would change the host's behaviour, which is the one thing a tap may never do, and it is also the
defect the tap exists to detect: ``verifiers``' ``rubrics/rubric.py:204-217`` catches any exception
from a reward function and substitutes ``ans = 0.0``. Wrapping the reward function itself, rather
than the rubric, is what puts the tap upstream of that substitution.

**One signature detail that is load-bearing.** ``verifiers`` dispatches on
``inspect.signature(func)`` and branches on whether the function declares ``**kwargs``
(``rubric.py:201``). A naive ``def wrapper(*args, **kwargs)`` would therefore change which keyword
arguments the grader receives. ``functools.wraps`` sets ``__wrapped__``, which ``inspect.signature``
follows by default, so the wrapper reports the wrapped function's signature and the dispatch is
unchanged. There is a test for it.
"""

from __future__ import annotations

import functools
import inspect
import math
import time
import tracemalloc
from typing import Any, Callable, Mapping, Sequence

from reward_lens.core.errors import RewardLensError
from reward_lens.tap.contract import (
    DEFAULT,
    BreachKind,
    CallOutcome,
    GraderCall,
    InstrumentEffect,
    RunHandle,
    SimpleRun,
    TapBreach,
    TapBudget,
    classify_return,
    is_sentinel,
    quantile,
)
from reward_lens.tap.ring import TapRing

_now = time.perf_counter_ns


class TapBudgetExceeded(RewardLensError):
    """Raised only under ``on_breach="raise"``, which breaks the Plane A contract on purpose.

    A test harness sometimes wants a breach to be loud. A training loop never does, and the
    ``"disable"`` default is the only setting that keeps the contract unconditionally.
    """


#: How many distinct recorder exceptions the tap tolerates before switching itself off. Three is
#: the specification's number and the reasoning behind it is worth keeping: one distinct exception
#: firing a thousand times is one bug, and disabling on the first occurrence would make the tap
#: useless in the presence of a single edge case it can already survive. Three distinct exceptions
#: is a recorder that does not understand the shape of what it is being handed.
DISTINCT_EXCEPTION_LIMIT = 3


class TapGuard:
    """The Plane A state machine: what the tap has cost so far, and whether it is still allowed.

    One guard per wrapped callable. It owns the latency window, the allocation accounting, the
    recorder-exception ledger and the disable switch. Nothing on it is a statistic in the Plane B
    sense: a quantile over a fixed window and a running sum are reductions, which Plane A may do,
    and there is no inference, no decision about the host and no I/O anywhere in this class.

    **Breaches survive a full ring.** A breach is held in the guard's own bounded list as well as
    being offered to the ring, because the moment a tap is most likely to be dropping records is
    exactly the moment it is breaching, and losing the explanation of a disable to the same
    congestion that caused it would be the worst failure this file could have.
    """

    __slots__ = (
        "name",
        "budget",
        "_window_added",
        "_window_inner",
        "_window_idx",
        "_window_n",
        "_check_stride",
        "_since_check",
        "calls",
        "added_ns_total",
        "inner_ns_total",
        "added_ns_max",
        "_disabled",
        "_breaches",
        "_exc_keys",
        "_exc_count",
        "_alloc_mode",
        "_alloc_this_step",
        "_alloc_step",
        "_alloc_sampled",
        "_sample_every",
        "_sample_i",
        "_resident_extra",
        "_lat_limit_ns",
        "_over_limit",
    )

    def __init__(
        self,
        name: str,
        budget: TapBudget = DEFAULT,
        *,
        window: int = 1024,
        check_stride: int = 64,
        measure_alloc: str = "auto",
    ) -> None:
        if window < 1:
            raise ValueError(f"the latency window needs at least one slot; got {window}")
        if check_stride < 1:
            raise ValueError(f"check_stride must be at least 1; got {check_stride}")
        if measure_alloc not in ("off", "auto", "always"):
            raise ValueError(
                f"measure_alloc must be one of off, auto, always; got {measure_alloc!r}"
            )
        self.name = name
        self.budget = budget
        self._window_added = [0] * window
        self._window_inner = [0] * window
        self._window_idx = 0
        self._window_n = 0
        self._check_stride = check_stride
        self._since_check = 0
        self.calls = 0
        self.added_ns_total = 0
        self.inner_ns_total = 0
        self.added_ns_max = 0
        self._disabled = False
        self._breaches: list[TapBreach] = []
        self._exc_keys: list[str] = []
        self._exc_count = 0
        self._alloc_mode = measure_alloc
        self._alloc_this_step = 0
        self._alloc_step: int | None = None
        self._alloc_sampled = False
        self._sample_every = 1
        self._sample_i = 0
        self._resident_extra = TapGuard.fixed_bytes(window)
        self._lat_limit_ns = budget.max_added_latency_ns_p99
        self._over_limit = 0

    @staticmethod
    def fixed_bytes(window: int = 1024) -> int:
        """What a guard holds outside the ring: two ``window``-slot pointer arrays, eight bytes each.

        The small integers the arrays hold are interned by CPython, so the pointers are the cost.
        A ring sized from a byte budget has to subtract this or it will breach that budget the
        first time it fills, which is the defect the acceptance test for ``for_bytes`` caught.
        """
        return 2 * window * 8

    # -- state -------------------------------------------------------------

    @property
    def disabled(self) -> bool:
        return self._disabled

    @property
    def breaches(self) -> tuple[TapBreach, ...]:
        return tuple(self._breaches)

    @property
    def recorder_exceptions(self) -> int:
        return self._exc_count

    @property
    def distinct_recorder_exceptions(self) -> tuple[str, ...]:
        return tuple(self._exc_keys)

    def should_sample(self) -> bool:
        """Whether this call is recorded, under a downsample that a breach may have imposed."""
        if self._sample_every == 1:
            return True
        self._sample_i += 1
        if self._sample_i >= self._sample_every:
            self._sample_i = 0
            return True
        return False

    def disable(self, breach: TapBreach) -> None:
        """Switch the tap off and keep the reason. The host is not told and does not change."""
        self._disabled = True
        self._record_breach(breach)

    def _record_breach(self, breach: TapBreach) -> None:
        # Bounded: a tap that breached sixteen times has a problem that sixteen records describe as
        # well as sixteen thousand would, and an unbounded list inside a Plane A object would be the
        # unbounded growth this whole design exists to avoid.
        if len(self._breaches) < 16:
            self._breaches.append(breach)

    # -- the recorder-exception ledger --------------------------------------

    def note_recorder_exception(self, exc: BaseException) -> TapBreach | None:
        """Count a recorder failure, and disable on the third distinct one.

        "Distinct" is keyed on the exception class and the first line of its message, truncated.
        One bug firing on every call is one distinct exception and the tap survives it; three
        different bugs is a recorder that does not understand its input, and it stops.
        """
        message = str(exc).split("\n", 1)[0][:160]
        key = f"{type(exc).__name__}: {message}"
        self._exc_count += 1
        if key not in self._exc_keys:
            self._exc_keys.append(key)
        if len(self._exc_keys) >= DISTINCT_EXCEPTION_LIMIT and not self._disabled:
            breach = TapBreach(
                kind=BreachKind.RECORDER_EXCEPTIONS,
                observed=float(len(self._exc_keys)),
                limit=float(DISTINCT_EXCEPTION_LIMIT),
                at_call=self.calls,
                detail="; ".join(self._exc_keys),
            )
            self.disable(breach)
            return breach
        return None

    # -- accounting ---------------------------------------------------------

    def observe(self, added_ns: int, inner_ns: int) -> None:
        """Record one call's timings into the window. Two list writes and a handful of additions.

        ``_over_limit`` is maintained here rather than derived at check time, and it is what keeps
        the budget check off the sorting path. It counts how many samples in the live window exceed
        the declared latency limit; the slots start at zero and zero never exceeds a positive
        limit, so the decrement on overwrite is correct from the first call without a fill flag.
        """
        self.calls += 1
        self.added_ns_total += added_ns
        self.inner_ns_total += inner_ns
        if added_ns > self.added_ns_max:
            self.added_ns_max = added_ns
        i = self._window_idx
        limit = self._lat_limit_ns
        if self._window_added[i] > limit:
            self._over_limit -= 1
        if added_ns > limit:
            self._over_limit += 1
        self._window_added[i] = added_ns
        self._window_inner[i] = inner_ns
        i += 1
        self._window_idx = 0 if i >= len(self._window_added) else i
        if self._window_n < len(self._window_added):
            self._window_n += 1

    def observe_alloc(self, delta_bytes: int, step: int | None) -> None:
        """Accumulate allocation within a step, resetting when the step index moves."""
        if step != self._alloc_step:
            self._alloc_step = step
            self._alloc_this_step = 0
        self._alloc_this_step += delta_bytes
        self._alloc_sampled = True

    def alloc_enabled(self) -> bool:
        """Whether allocation can be measured at all right now.

        ``tracemalloc.is_tracing()`` costs about 25 ns on the reference machine, so asking every
        call is free; ``get_traced_memory()`` costs about 933 ns, which is why it is only paid when
        the host already has tracing on and has therefore already accepted a much larger slowdown.
        """
        if self._alloc_mode == "off":
            return False
        if self._alloc_mode == "always":
            return True
        return tracemalloc.is_tracing()

    # -- the budget check ---------------------------------------------------

    def window_quantiles(self) -> tuple[float, float, float]:
        """(p50 added, p99 added, p99 of the per-call added fraction), over the live window.

        Two sorts and a thousand divisions, which is about 200 us on the reference machine. That is
        far too expensive for the hot path and it does not run there: ``check`` uses the ``latency_
        p99_can_breach`` gate instead, and this runs when a card, a step boundary or an
        ``InstrumentEffect`` asks for the numbers.
        """
        n = self._window_n
        if n == 0:
            return (0.0, 0.0, 0.0)
        added = sorted(self._window_added[:n])
        fractions = sorted(
            (
                self._window_added[k] / (self._window_added[k] + self._window_inner[k])
                if (self._window_added[k] + self._window_inner[k]) > 0
                else 0.0
            )
            for k in range(n)
        )
        return (quantile(added, 0.5), quantile(added, 0.99), quantile(fractions, 0.99))

    def latency_p99_can_breach(self) -> bool:
        """Whether the windowed p99 could possibly exceed the limit, in constant time.

        Exact rather than heuristic, and that is what makes it usable as a gate. ``quantile``
        interpolates between the samples at ``lo = floor(0.99(n-1))`` and ``hi = ceil(0.99(n-1))``
        of the ascending sort, so the result can only exceed the limit if the sample at ``hi``
        does, which needs at least ``n - hi`` samples above the limit. Counting those on the write
        path costs two comparisons; recomputing the quantile to find out costs 200 us.

        The window is small enough that a single slow call breaches immediately below about a
        hundred samples, which is the behaviour a short run should have: with forty calls recorded
        there is no distribution to speak of and the largest one is the honest estimate of the p99.
        """
        n = self._window_n
        if n == 0 or self._over_limit == 0:
            return False
        hi = math.ceil(0.99 * (n - 1))
        return self._over_limit >= (n - hi)

    def resident_bytes(self, ring: TapRing) -> int:
        """What the tap's own containers hold. Not the host objects the records point at."""
        return ring.resident_bytes + self._resident_extra

    def check(self, ring: TapRing, *, force: bool = False) -> TapBreach | None:
        """Enforce the budget. Returns the breach that fired, or None.

        Latency is checked on a stride because a p99 is a claim about a distribution and a single
        sample cannot breach one. Residency is checked on every call because it is a length times a
        constant. Allocation is checked on the stride alongside latency, and named as unchecked
        rather than counted as a pass whenever the host is not tracing.
        """
        b = self.budget
        resident = self.resident_bytes(ring)
        if resident > b.max_resident_bytes:
            return self._fire(
                TapBreach(
                    kind=BreachKind.RESIDENT_BYTES,
                    observed=float(resident),
                    limit=float(b.max_resident_bytes),
                    at_call=self.calls,
                    detail=f"ring holds {len(ring)} of {ring.capacity} records",
                    step=self._alloc_step,
                )
            )

        self._since_check += 1
        if not force and self._since_check < self._check_stride:
            return None
        self._since_check = 0

        if self.latency_p99_can_breach():
            _p50, p99, _frac = self.window_quantiles()
            if p99 > b.max_added_latency_ns_p99:
                return self._fire(
                    TapBreach(
                        kind=BreachKind.LATENCY_P99,
                        observed=p99 / 1e6,
                        limit=b.max_added_latency_ms_p99,
                        at_call=self.calls,
                        detail=f"milliseconds, p99 over the last {self._window_n} calls",
                        step=self._alloc_step,
                    )
                )

        if self._alloc_sampled and self._alloc_this_step > b.max_added_alloc_bytes_per_step:
            return self._fire(
                TapBreach(
                    kind=BreachKind.ALLOC_BYTES_PER_STEP,
                    observed=float(self._alloc_this_step),
                    limit=float(b.max_added_alloc_bytes_per_step),
                    at_call=self.calls,
                    detail=f"traced allocation within step {self._alloc_step}",
                    step=self._alloc_step,
                )
            )
        return None

    def _fire(self, breach: TapBreach) -> TapBreach:
        """Apply ``on_breach``. Disable is the default and the only one that keeps the contract."""
        mode = self.budget.on_breach
        if mode == "raise":
            self._record_breach(breach)
            raise TapBudgetExceeded(breach.render())
        if mode == "downsample":
            self._record_breach(breach)
            self._sample_every = min(self._sample_every * 2, 1024)
            if self._sample_every >= 1024:
                self._disabled = True
            return breach
        self.disable(breach)
        return breach

    # -- what it cost -------------------------------------------------------

    def unchecked(self) -> tuple[str, ...]:
        """Checks that could not run, named. Never counted as passes."""
        out: list[str] = []
        if not self._alloc_sampled:
            if self._alloc_mode == "off":
                out.append("added allocation (measure_alloc='off')")
            else:
                out.append("added allocation (tracemalloc is not tracing in this process)")
        if self._alloc_step is None and self._alloc_sampled:
            out.append("per-step allocation attribution (the run handle carries no step index)")
        return tuple(out)

    def effect(self, run: RunHandle | None, ring: TapRing | None) -> InstrumentEffect:
        """The instrument effect, as of now. Off the hot path; a card or a step boundary calls it."""
        p50, p99, frac = self.window_quantiles()
        return InstrumentEffect(
            tap_name=self.name,
            run_id=getattr(run, "run_id", "anonymous"),
            calls=self.calls,
            added_ns_total=self.added_ns_total,
            inner_ns_total=self.inner_ns_total,
            added_ns_p50=p50,
            added_ns_p99=p99,
            added_ns_max=self.added_ns_max,
            fraction_p99=frac,
            resident_bytes=self.resident_bytes(ring) if ring is not None else self._resident_extra,
            added_alloc_bytes_this_step=(self._alloc_this_step if self._alloc_sampled else None),
            step=self._alloc_step,
            ring=ring.stats() if ring is not None else None,
            recorder_exceptions=self._exc_count,
            distinct_recorder_exceptions=tuple(self._exc_keys),
            enabled=not self._disabled,
            breaches=tuple(self._breaches),
            unchecked=self.unchecked(),
            budget=self.budget,
            window_n=self._window_n,
        )


# ---------------------------------------------------------------------------
# tap(): the general Plane A guard
# ---------------------------------------------------------------------------


def tap(
    fn: Callable[..., Any],
    budget: TapBudget = DEFAULT,
    *,
    name: str | None = None,
    ring: TapRing | None = None,
    window: int = 1024,
    check_stride: int = 64,
    measure_alloc: str = "auto",
) -> Callable[..., Any]:
    """Wrap a tap so it can never raise into the host and can never exceed its budget.

    ``fn`` is Plane A work whose return value the host does not need: read a tensor that already
    exists, reduce it, append the reduction to a ring. Every invocation is timed and its allocation
    delta measured where the host makes that possible. A breach records an ``InstrumentEffect``
    carrying the breach and continues with the tap switched off. Exceptions are caught, counted and
    recorded, and the third distinct one disables the tap.

    The wrapper returns whatever ``fn`` returned, or ``None`` when ``fn`` raised or the tap was
    already off. That asymmetry is deliberate and is the difference between this and
    ``instrument_grader``: here the return value belongs to the tap, so losing it is acceptable;
    there it belongs to the host, so losing it is not.

    ``BaseException`` is deliberately **not** caught. ``KeyboardInterrupt``, ``SystemExit`` and
    ``GeneratorExit`` are control flow rather than errors, and swallowing a Ctrl-C inside somebody
    else's training loop would be a worse breach of trust than any exception this could hide.

    The guard is reachable as ``wrapper.guard`` and the effect as ``wrapper.effect()``.
    """
    guard = TapGuard(
        name or getattr(fn, "__name__", repr(fn)),
        budget,
        window=window,
        check_stride=check_stride,
        measure_alloc=measure_alloc,
    )
    the_ring = (
        ring
        if ring is not None
        else TapRing.for_bytes(budget.max_resident_bytes, reserve=TapGuard.fixed_bytes(window))
    )

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if guard.disabled:
            return None
        t0 = _now()
        traced = guard.alloc_enabled()
        a0 = tracemalloc.get_traced_memory()[0] if traced else 0
        try:
            out = fn(*args, **kwargs)
        except Exception as exc:
            guard.note_recorder_exception(exc)
            out = None
        t1 = _now()
        try:
            if traced:
                guard.observe_alloc(tracemalloc.get_traced_memory()[0] - a0, None)
            guard.observe(t1 - t0, 0)
            guard.check(the_ring)
        except TapBudgetExceeded:
            raise
        except Exception as exc:  # the guard's own bookkeeping failed
            guard.note_recorder_exception(exc)
        return out

    wrapper.guard = guard  # type: ignore[attr-defined]
    wrapper.ring = the_ring  # type: ignore[attr-defined]
    wrapper.effect = lambda: guard.effect(None, the_ring)  # type: ignore[attr-defined]
    return wrapper


# ---------------------------------------------------------------------------
# instrument_grader(): the wedge
# ---------------------------------------------------------------------------


def instrument_grader(
    score_fn: Callable[..., Any],
    *,
    run: RunHandle | None = None,
    budget: TapBudget = DEFAULT,
    name: str | None = None,
    retain_args: bool = True,
    retain_value: bool = True,
    facet_keys: Sequence[str] = (),
    timeout_types: tuple[type[BaseException], ...] = (TimeoutError,),
    window: int = 1024,
    check_stride: int = 64,
    measure_alloc: str = "auto",
) -> Callable[..., Any]:
    """Wrap any scoring callable so every call is recorded, losing nothing.

    Records, per call: the arguments as received; every component of the return value, not just the
    total; the facets available from the call context; latency; cost; the raw grader output where
    retained; and whether the call raised, timed out, or returned a sentinel.

    Returns exactly what the wrapped callable returned, byte-identical. Not a copy, not a
    reconstruction, not a coercion: the same object, so ``wrapped(x) is score_fn(x)`` holds for any
    callable that returns a stable object.

    Plane A contract: an exception in the recorder is caught, counted and recorded; three distinct
    exceptions disable recording and the run continues untouched. An exception from ``score_fn`` is
    a different thing entirely: it is recorded and then re-raised unchanged.

    **Components are read from the retained return value, not extracted on the hot path.** A
    mapping's keys are read at call time, because that is what says which components exist, but the
    values are read by ``GraderCall.components()`` at drain time. Walking a 256-element reward list
    costs roughly ten times everything else the tap does, and Plane A may not spend that. The
    consequence is stated rather than hidden: a host that mutates its own return value after
    returning it would be read wrongly, and ``retain_value=False`` plus an eager drain is the
    answer for a host that does. ``ReturnShape`` is captured at call time either way, so the
    mutation is detectable.

    ``retain_args=False`` drops the references to the host's arguments, which is the setting for a
    long run whose ring would otherwise pin large prompt and completion lists until the drain
    catches up. Set ``facet_keys`` alongside it, or the facets go with the references.

    ``async def`` callables are wrapped by an ``async def`` wrapper, so the recorded latency is the
    time the grader took rather than the time it took to build a coroutine. A synchronous callable
    that *returns* an awaitable (the ``verifiers`` ``maybe_await`` shape) is recorded with
    ``ReturnShape.kind == "awaitable"``, which flags the latency as not being the grader's.

    **The tap imposes no timeout of its own.** Doing so would need a thread or a signal handler and
    would change the host's behaviour. ``TIMED_OUT`` is recorded when the callable itself raises a
    timeout, which on Python 3.11 and later means ``TimeoutError`` covers the asyncio and
    concurrent.futures cases too. An HTTP judge whose client raises its own class (``httpx``'s
    ``ReadTimeout``, say) is classified by passing ``timeout_types``; the class-name fallback
    catches Python 3.10's separate ``asyncio.TimeoutError`` without importing asyncio here.
    """
    handle: RunHandle = run if run is not None else SimpleRun()
    the_ring: TapRing = handle.ring
    guard = TapGuard(
        name or getattr(score_fn, "__name__", repr(score_fn)),
        budget,
        window=window,
        check_stride=check_stride,
        measure_alloc=measure_alloc,
    )
    facets = tuple(facet_keys)
    grader_name = guard.name
    seq = [0]

    def _facets(kwargs: Mapping[str, Any]) -> dict[str, Any] | None:
        if not facets:
            return None
        return {k: kwargs[k] for k in facets if k in kwargs}

    def _record(
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        value: Any,
        exc: BaseException | None,
        inner_ns: int,
        started_ns: int,
    ) -> GraderCall:
        if exc is not None:
            outcome = (
                CallOutcome.TIMED_OUT
                if (isinstance(exc, timeout_types) or type(exc).__name__ == "TimeoutError")
                else CallOutcome.RAISED
            )
            shape = None
        else:
            shape = classify_return(value)
            outcome = CallOutcome.SENTINEL if is_sentinel(value, shape) else CallOutcome.RETURNED
        seq[0] += 1
        # Positional, because keyword construction of this fifteen-field record costs 436 ns on the
        # reference machine against 185 ns positional, and it is built once per grader call.
        return GraderCall(
            seq[0],
            handle.run_id,
            grader_name,
            outcome,
            inner_ns,
            0,
            args if retain_args else None,
            kwargs if retain_args else None,
            value if (retain_value and exc is None) else None,
            shape,
            type(exc).__name__ if exc is not None else None,
            str(exc).split("\n", 1)[0][:400] if exc is not None else None,
            _facets(kwargs),
            getattr(handle, "step", None),
            started_ns,
        )

    def _observe(
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        value: Any,
        exc: BaseException | None,
        ended_ns: int,
        t1: int,
        prologue_ns: int,
        traced: bool,
        a0: int,
    ) -> None:
        """The recorder segment. Everything in here is caught; nothing in here reaches the host.

        ``added_ns`` is fixed on the record *before* the ring offer, so a concurrent drain never
        sees a record whose own cost is still a placeholder. The consequence is that the offer
        itself, 131 ns on the reference machine, is the one part of the tap that the tap does not
        count. The acceptance test measures the whole wrapper from outside against the bare
        callable, which is what catches an omission like that rather than trusting this note.
        """
        inner_ns = ended_ns - t1
        try:
            record = (
                _record(args, kwargs, value, exc, inner_ns, t1) if guard.should_sample() else None
            )
            if traced:
                guard.observe_alloc(
                    tracemalloc.get_traced_memory()[0] - a0, getattr(handle, "step", None)
                )
            added = prologue_ns + (_now() - ended_ns)
            guard.observe(added, inner_ns)
            if record is not None:
                record.added_ns = added
                the_ring.offer(record)
            guard.check(the_ring)
        except TapBudgetExceeded:
            raise
        except Exception as recorder_exc:
            guard.note_recorder_exception(recorder_exc)

    if inspect.iscoroutinefunction(score_fn):

        @functools.wraps(score_fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if guard.disabled:
                return await score_fn(*args, **kwargs)
            t0 = _now()
            traced = guard.alloc_enabled()
            a0 = tracemalloc.get_traced_memory()[0] if traced else 0
            t1 = _now()
            try:
                value = await score_fn(*args, **kwargs)
            except BaseException as exc:
                _observe(args, kwargs, None, exc, _now(), t1, t1 - t0, traced, a0)
                raise
            _observe(args, kwargs, value, None, _now(), t1, t1 - t0, traced, a0)
            return value

        wrapper: Callable[..., Any] = async_wrapper
    else:

        @functools.wraps(score_fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if guard.disabled:
                return score_fn(*args, **kwargs)
            t0 = _now()
            traced = guard.alloc_enabled()
            a0 = tracemalloc.get_traced_memory()[0] if traced else 0
            t1 = _now()
            try:
                value = score_fn(*args, **kwargs)
            except BaseException as exc:
                _observe(args, kwargs, None, exc, _now(), t1, t1 - t0, traced, a0)
                raise
            _observe(args, kwargs, value, None, _now(), t1, t1 - t0, traced, a0)
            return value

        wrapper = sync_wrapper

    # The seam. `effect()` is how the instrument effect leaves Plane A, and the tap does not emit
    # it: emitting is a step-boundary act and the tap does not know where a step boundary is. A
    # framework adapter calls this at the end of a step and hands the result to the record.
    # The numbers are cumulative with rolling quantiles, so a per-step effect is the difference
    # between two of these rather than something the guard has to be told to reset.
    #
    # `guard.breaches` is deliberately reachable the same way. A tap is most likely to be dropping
    # records exactly when it is breaching, so the breach lives in the guard's own bounded list and
    # not only in the ring.
    wrapper.guard = guard  # type: ignore[attr-defined]
    wrapper.ring = the_ring  # type: ignore[attr-defined]
    wrapper.run = handle  # type: ignore[attr-defined]
    wrapper.effect = lambda: guard.effect(handle, the_ring)  # type: ignore[attr-defined]
    return wrapper


__all__ = [
    "DISTINCT_EXCEPTION_LIMIT",
    "TapBudgetExceeded",
    "TapGuard",
    "instrument_grader",
    "tap",
]

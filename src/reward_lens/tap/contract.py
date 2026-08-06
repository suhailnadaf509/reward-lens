"""The Plane A vocabulary: what a tap declares, what it records, and what it costs.

Plane A runs inside somebody else's training loop, so its reliability contract is the inverse of
everything else in this library. Plane B fails closed and loud, because a wrong number is worse
than no number. Plane A **fails open**: it never raises into the host, it never blocks, it never
grows without bound, and when it cannot keep its promises it turns itself off and records that it
did. A refusal from a tap is a value the analysis layer reads later. The training loop never sees
one.

Four types carry that contract.

``TapBudget`` is the declaration. Three limits, stated up front rather than discovered in an
incident review, plus what to do when one is breached.

``GraderCall`` is the record. It holds the arguments as received and the return value **by
reference**, because the one thing a wrapper must not do is hand the host back a reconstruction of
what it was given. Everything else on the record is a small scalar measured at call time.

``CallOutcome`` is the field nobody currently records and it is the reason the wrapper is worth
building. ``verifiers``' ``rubrics/rubric.py:204-217`` catches any exception from a reward function
and substitutes ``ans = 0.0``; because ``score_group`` is mean-centred with no standard-deviation
division (``rubric.py:406-409``), a systematically crashing component shifts the entire advantage
baseline and nothing downstream can tell that zero from a real one. A tap wrapping the reward
function itself sees the exception before the rubric swallows it, which is the whole of instrument
B4.

``InstrumentEffect`` is the cost. The overhead a monitor imposes is a term in the uncertainty
budget, not a footnote, so it composes through ``core.budget.BudgetTerm`` rather than sitting
beside the number as prose.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Mapping,
    NamedTuple,
    Protocol,
    Sequence,
    runtime_checkable,
)

from reward_lens.core.budget import BudgetTerm
from reward_lens.core.provenance import Cost

if TYPE_CHECKING:  # a ring holds records; a record knows nothing about a ring
    from reward_lens.tap.ring import RingStats, TapRing


# ---------------------------------------------------------------------------
# The budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TapBudget:
    """What a tap promises the host, declared before it runs.

    The three limits are not decoration. Adoption at a lab ends permanently the first time a
    monitor crashes or stalls a job, and the documented failure modes are memory-shaped: a vLLM
    hidden-state export allocates 2.625 GiB in one block at 49,152 tokens, and chunked prefill does
    not bound that allocation. So a tap states its ceiling and is measured against it every step.

    ``max_added_latency_ms_p99`` is a **distributional** claim, so a single slow call does not
    breach it and the enforcement does not pretend otherwise: the tap recomputes a windowed p99
    on a stride and compares that. ``max_resident_bytes`` bounds what the tap's own containers
    hold, which is what the tap can actually control; the host objects it keeps alive by holding
    references are counted separately as ``retained_refs`` rather than folded into a byte number
    the tap did not allocate. ``max_added_alloc_bytes_per_step`` is measured with ``tracemalloc``
    when the host has it running and is reported as unchecked when it does not, because a check
    that could not run is not a check that passed.

    ``on_breach`` is the spec's three-way. ``"disable"`` is the default and the only one that keeps
    the Plane A contract unconditionally. ``"raise"`` exists because a test harness sometimes wants
    a breach to be loud; it is never correct inside a training loop and the constructor says so
    rather than trusting a comment.
    """

    max_added_latency_ms_p99: float
    max_resident_bytes: int
    max_added_alloc_bytes_per_step: int
    on_breach: Literal["disable", "downsample", "raise"] = "disable"

    def __post_init__(self) -> None:
        if self.max_added_latency_ms_p99 <= 0:
            raise ValueError(
                f"max_added_latency_ms_p99 must be positive; got "
                f"{self.max_added_latency_ms_p99}. A tap with a zero latency budget cannot run at "
                f"all, which is a different thing from a tap that is switched off."
            )
        if self.max_resident_bytes <= 0:
            raise ValueError(f"max_resident_bytes must be positive; got {self.max_resident_bytes}")
        if self.max_added_alloc_bytes_per_step < 0:
            raise ValueError(
                f"max_added_alloc_bytes_per_step must not be negative; got "
                f"{self.max_added_alloc_bytes_per_step}"
            )
        if self.on_breach not in ("disable", "downsample", "raise"):
            raise ValueError(
                f"on_breach must be one of disable, downsample, raise; got {self.on_breach!r}"
            )

    @property
    def max_added_latency_ns_p99(self) -> float:
        """The latency limit in the units the tap actually measures in."""
        return self.max_added_latency_ms_p99 * 1e6

    def __canonical__(self) -> dict[str, Any]:
        return {
            "max_added_latency_ms_p99": self.max_added_latency_ms_p99,
            "max_resident_bytes": self.max_resident_bytes,
            "max_added_alloc_bytes_per_step": self.max_added_alloc_bytes_per_step,
            "on_breach": self.on_breach,
        }


#: The default budget. The numbers in it are argued from a measurement rather than derived.
#:
#: The tap's own recorder segment measures 745 to 881 ns at the median and about 1.2 us at the p99
#: on the reference machine, an Intel i9-13900HX under CPython 3.12.3. 1 ms is therefore roughly
#: 800 times the p99 it is bounding, which leaves room for a slower host, a contended lock, or a
#: garbage collection landing inside the recorder, and is still small enough that a breach means
#: something has genuinely gone wrong rather than that the machine was briefly busy. 64 MiB is a
#: ring of about 200,000 records, which is more than a training step produces and small enough that
#: nobody notices it next to a model. 16 MiB of allocation per step is the same reasoning applied
#: to churn rather than to residency.
#:
#: A tap that needs different numbers should declare them. The point of the default is that there
#: is always a declared budget, not that this one fits every host.
DEFAULT = TapBudget(
    max_added_latency_ms_p99=1.0,
    max_resident_bytes=64 * 1024 * 1024,
    max_added_alloc_bytes_per_step=16 * 1024 * 1024,
)


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


class CallOutcome(enum.Enum):
    """How the call ended. The highest value-to-effort field in the record.

    ``RETURNED`` and ``SENTINEL`` are both normal returns; they are separated because a grader that
    returns ``None`` is abstaining and a grader that returns ``0.0`` is scoring, and four of the
    fifteen grader interfaces in Appendix C have no way to tell you which happened. TRL gets this
    right (``None`` maps to ``torch.nan`` at ``grpo_trainer.py:1663`` and is excluded), which is
    why TRL's behaviour is the model here.

    ``TIMED_OUT`` is a raise that happened to be a ``TimeoutError``. The tap does **not** impose a
    timeout of its own: doing so would need a thread or a signal handler and would change the
    host's behaviour, which is exactly what a Plane A component may not do.
    """

    RETURNED = "returned"
    RAISED = "raised"
    TIMED_OUT = "timed_out"
    SENTINEL = "sentinel"


#: How the return value is shaped, classified in constant time (or in the number of mapping keys).
#: ``awaitable`` matters because a synchronous wrapper around an async grader measures the cost of
#: building a coroutine rather than the cost of scoring, and a latency that is silently wrong is
#: worse than one that is missing.
ReturnKind = Literal["none", "scalar", "mapping", "sequence", "awaitable", "other"]


class ReturnShape(NamedTuple):
    """What the return value looked like at the moment it was returned.

    Recorded at call time and kept alongside the retained object so that a return value the host
    mutates afterwards is detectable rather than silently re-read. Component *values* are read from
    the retained object by the drain, which is Plane B work; recording the shape here is the
    reduction Plane A is allowed to do.

    A ``NamedTuple`` rather than a frozen dataclass, and measured rather than assumed: constructing
    one costs about 126 ns on the reference machine against about 438 ns for the equivalent frozen
    dataclass, because a frozen ``__init__`` routes every field through ``object.__setattr__``.
    This object is built once per grader call, so 312 ns is a large fraction of the whole tap.
    """

    kind: ReturnKind
    type_name: str
    length: int | None = None
    keys: tuple[Any, ...] | None = None
    #: True when the shape walk was cut short by a cap, so ``keys`` is a prefix and not the whole.
    truncated: bool = False

    def __canonical__(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "type_name": self.type_name,
            "length": self.length,
            "keys": [str(k) for k in self.keys] if self.keys is not None else None,
            "truncated": self.truncated,
        }


@dataclass(slots=True)
class GraderCall:
    """One call to a wrapped scoring callable, as it happened.

    Write-once by discipline rather than by ``frozen=True``, and the reason is measured rather than
    stylistic. Constructing these fifteen fields costs 185 ns positionally on the reference machine
    and 436 ns by keyword; the frozen equivalent costs more again, because ``frozen`` routes every
    field through ``object.__setattr__``. Against a recorder segment of about 800 ns that is a
    quarter of the whole tap, and it would buy an immutability that is shallow anyway, since the
    record holds references to the host's own mutable arguments and return value. The store, not
    the record type, is what makes evidence append-only.

    ``args``, ``kwargs`` and ``value`` are the host's objects, held by reference. Nothing is copied,
    coerced, rounded or round-tripped through JSON. That is the contract the wrapper exists to
    keep, and it has a memory consequence: a ring full of records pins whatever the host passed in
    until the drain runs. ``TapRing.stats().retained_refs`` reports how many objects are pinned,
    and a tap constructed with ``retain_args=False`` keeps the shapes and drops the references.
    """

    #: Monotone within one tap, so a drain can tell a gap from a reorder.
    seq: int
    run_id: str
    grader: str
    outcome: CallOutcome
    #: Wall time inside the wrapped callable. Not the tap's cost; the grader's.
    inner_ns: int
    #: Wall time the tap itself added, measured around the recorder segments only.
    added_ns: int
    args: tuple[Any, ...] | None = None
    kwargs: Mapping[str, Any] | None = None
    #: What the callable returned, by reference. ``None`` when it raised.
    value: Any = None
    shape: ReturnShape | None = None
    #: Class name and message of the exception the callable raised, then re-raised unchanged.
    error_type: str | None = None
    error_message: str | None = None
    #: Facets available from the call context: seed, order, opponent, rubric draw, step index.
    #: A tap reads only what the host already handed it; it computes none of these.
    facets: Mapping[str, Any] | None = None
    step: int | None = None
    #: Wall clock at the start of the call, for aligning a tap's records against a host's log.
    started_ns: int = 0

    @property
    def raised(self) -> bool:
        """Whether this call failed, however the framework went on to present it downstream."""
        return self.outcome in (CallOutcome.RAISED, CallOutcome.TIMED_OUT)

    @property
    def cost(self) -> Cost:
        """The grader's own wall time, as the kernel's cost type, so it aggregates."""
        return Cost(wall_seconds=self.inner_ns / 1e9)

    def components(self, *, cap: int = 4096) -> tuple[tuple[str, Any], ...]:
        """Every component of the return value, not just the total.

        Read from the retained object rather than extracted on the hot path, because extraction is
        O(n) in the number of components and Plane A may not spend that. A mapping yields one
        component per key, a sequence one per index, a scalar one unnamed component, and a call
        that raised yields nothing.

        ``cap`` bounds the walk. A sequence longer than the cap yields its first ``cap`` entries;
        the caller can see the truncation by comparing against ``shape.length``.
        """
        v = self.value
        if self.outcome in (CallOutcome.RAISED, CallOutcome.TIMED_OUT) or v is None:
            return ()
        kind = self.shape.kind if self.shape is not None else classify_return(v).kind
        if kind == "mapping":
            items = list(v.items())[:cap]
            return tuple((str(k), val) for k, val in items)
        if kind == "sequence":
            return tuple((str(i), val) for i, val in enumerate(list(v)[:cap]))
        if kind == "scalar":
            return (("", v),)
        return ()

    def __canonical__(self) -> dict[str, Any]:
        """The envelope a drain writes. The retained objects are not in it.

        ``args``, ``kwargs`` and ``value`` are the host's objects and may be anything at all, so
        serialising them is the drain's decision and not the record's. What is canonical here is
        the measured part: the outcome, the timings, the shape and the facets.
        """
        return {
            "seq": self.seq,
            "run_id": self.run_id,
            "grader": self.grader,
            "outcome": self.outcome.value,
            "inner_ns": self.inner_ns,
            "added_ns": self.added_ns,
            "shape": self.shape.__canonical__() if self.shape is not None else None,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "step": self.step,
            "started_ns": self.started_ns,
        }


#: Cost of one record's own containers, in bytes, on CPython 3.12 with ``slots=True``: the object
#: header plus one pointer per slot, plus the ``ReturnShape`` beside it. Measured in
#: ``tests/acceptance/test_w3_1_tap.py`` rather than assumed, and used to turn the byte budget of
#: ``TapBudget.max_resident_bytes`` into a ring capacity. It deliberately excludes the host objects
#: a record points at: the tap did not allocate those and cannot free them.
RECORD_FOOTPRINT_BYTES = 320


def classify_return(value: Any, *, key_cap: int = 256) -> ReturnShape:
    """Classify a return value's shape without touching its contents.

    Concrete-type checks come first because they are the common case and an ``isinstance`` against
    an ABC is several times the cost of an identity check on ``type()``. Nothing here calls
    ``float()``, iterates a sequence, or reads a mapping's values: those are Plane B's job and the
    retained object is what makes them possible later.

    A mapping's keys are read, capped at ``key_cap``, because the key set is what says which
    components exist and TRL's ``reward_kwargs`` (``grpo_trainer.py:1618``) splats every remaining
    dataset column in, so no tap may assume a closed one.

    Every ``ReturnShape`` here is built positionally. Keyword construction of a five-field
    ``NamedTuple`` costs 207 ns against 126 ns positional, and this runs once per grader call.
    """
    if value is None:
        return ReturnShape("none", "NoneType")
    t = type(value)
    if t is float:
        return ReturnShape("scalar", "float")
    if t is list:
        return ReturnShape("sequence", "list", len(value))
    if t is dict:
        keys = tuple(value.keys())
        n = len(keys)
        return ReturnShape("mapping", "dict", n, keys[:key_cap], n > key_cap)
    name = t.__name__
    if t is int or t is bool:
        return ReturnShape("scalar", name)
    if t is tuple:
        return ReturnShape("sequence", name, len(value))
    if hasattr(value, "__await__"):
        return ReturnShape("awaitable", name)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        n = len(keys)
        return ReturnShape("mapping", name, n, keys[:key_cap], n > key_cap)
    if hasattr(value, "__float__") and not hasattr(value, "__len__"):
        # numpy scalars land here, and so does anything else that is a number in all but name.
        return ReturnShape("scalar", name)
    if isinstance(value, (Sequence, bytes, bytearray)) and not isinstance(value, str):
        return ReturnShape("sequence", name, len(value))
    return ReturnShape("other", name)


def is_sentinel(value: Any, shape: ReturnShape) -> bool:
    """Whether a normal return was actually an abstention.

    Two cases, both cheap and both at the top level only. ``None`` is TRL's abstention channel and
    the only one in the ecosystem that a framework acts on. A scalar NaN is the other, caught by
    ``value != value``, which holds for a Python float and for a numpy scalar alike and needs no
    numpy import. The check is deliberately not applied elementwise to a sequence: that walk is
    O(n) and belongs to the drain, which has the retained object and a whole core to spend.
    """
    if value is None:
        return True
    if shape.kind == "scalar":
        try:
            return bool(value != value)
        except Exception:
            return False
    return False


# ---------------------------------------------------------------------------
# The breach and the effect
# ---------------------------------------------------------------------------


class BreachKind(enum.Enum):
    """Which promise the tap failed to keep."""

    LATENCY_P99 = "latency_p99"
    RESIDENT_BYTES = "resident_bytes"
    ALLOC_BYTES_PER_STEP = "alloc_bytes_per_step"
    RECORDER_EXCEPTIONS = "recorder_exceptions"


@dataclass(frozen=True)
class TapBreach:
    """One budget breach, with the number that breached it and the number it was measured against.

    Carrying both is the difference between a log line and evidence. "The tap disabled itself"
    invites a guess; "the windowed p99 added latency was 4.20 ms against a declared 1.00 ms over
    1024 calls" is a fact somebody can act on.
    """

    kind: BreachKind
    observed: float
    limit: float
    at_call: int
    detail: str = ""
    step: int | None = None

    def render(self) -> str:
        return (
            f"{self.kind.value}: observed {self.observed:.6g}, limit {self.limit:.6g}, "
            f"at call {self.at_call}" + (f" ({self.detail})" if self.detail else "")
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "observed": self.observed,
            "limit": self.limit,
            "at_call": self.at_call,
            "detail": self.detail,
            "step": self.step,
        }


@dataclass(frozen=True)
class InstrumentEffect:
    """What the tap cost the run, measured.

    No competitor has published an overhead number of any kind, so this type exists to make ours
    reportable rather than anecdotal, and ``as_term`` is what stops it being a footnote: it
    composes into the same ``UncertaintyBudget`` as everything else, next to the grader replication
    variance and the substrate noise floor, where it can be compared against them.

    ``unchecked`` keeps it honest. A tap running under a host that is not tracing allocations has
    not established that it stayed inside its allocation budget, and saying so is different from
    passing.
    """

    tap_name: str
    run_id: str
    calls: int
    #: Sum over all calls of the time the tap itself added. The headline number.
    added_ns_total: int = 0
    #: Sum over all calls of the time spent inside the wrapped callable.
    inner_ns_total: int = 0
    added_ns_p50: float = 0.0
    added_ns_p99: float = 0.0
    added_ns_max: int = 0
    #: p99 of the per-call ratio ``added / (added + inner)``. A ratio of two quantiles would not be
    #: a quantile of anything, so this is the quantile of the per-call ratio.
    fraction_p99: float = 0.0
    resident_bytes: int = 0
    added_alloc_bytes_this_step: int | None = None
    step: int | None = None
    ring: "RingStats | None" = None
    recorder_exceptions: int = 0
    distinct_recorder_exceptions: tuple[str, ...] = ()
    enabled: bool = True
    breaches: tuple[TapBreach, ...] = ()
    unchecked: tuple[str, ...] = ()
    budget: TapBudget | None = None
    #: Samples behind the quantiles, so a p99 from 40 calls is visibly weaker than one from 40,000.
    window_n: int = 0

    @property
    def fraction_mean(self) -> float:
        """Added time as a fraction of total time in the wrapped callable, over the whole run."""
        if self.inner_ns_total <= 0:
            return 0.0
        return self.added_ns_total / self.inner_ns_total

    @property
    def added_ms_p99(self) -> float:
        return self.added_ns_p99 / 1e6

    def as_term(self, name: str = "instrument_overhead", *, sensitivity: float = 1.0) -> BudgetTerm:
        """The overhead as a Type B budget term, so it composes.

        Type B and rectangular, because the perturbation is an **uncorrected bias**: the tap cannot
        undo the time it took, so the GUM treatment is a bound rather than a standard deviation,
        divided by sqrt(3). The half-width is the p99 of the per-call fraction of wall time the tap
        added, which is dimensionless and therefore composable against any reading whose
        sensitivity to a timing perturbation the caller can state.

        **The basis is the grader call, not the training step, and that matters.** The tap can time
        the callable it wraps and nothing else; it does not know what a step costs. So the fraction
        is added time over the grader's own time, which is an upper bound on the perturbation to the
        run and errs in the conservative direction. It also means a trivially cheap grader reports a
        large fraction: wrapping a callable that does one comparison, the tap is most of the call,
        and 0.5 is a true statement about that setup rather than an artefact. A real grader costs
        milliseconds (a parse) to seconds (a judge over HTTP) and the fraction lands between 1e-3
        and 1e-6.

        ``sensitivity`` is left at 1.0 and should usually be overridden. How much a given reading
        moves per unit of instrument perturbation is a property of that reading, not of the tap,
        and the kernel keeps ``sensitivity`` separate from ``value`` precisely so that a tap is not
        forced to invent one. A caller who knows the step time converts the basis here: pass the
        ratio of grader time to step time as the sensitivity and the term becomes a run-level
        perturbation. A reading with no plausible dependence on wall-clock timing takes sensitivity
        0 and the term drops out arithmetically rather than by being omitted.
        """
        return BudgetTerm.from_half_width(
            name,
            half_width=self.fraction_p99,
            distribution="rectangular",
            sensitivity=sensitivity,
            note=(
                f"uncorrected instrument perturbation: p99 of added/(added+inner) over "
                f"{self.window_n} calls, p99 added latency {self.added_ms_p99:.4g} ms, "
                f"tap {self.tap_name!r}" + ("" if self.enabled else ", tap disabled after a breach")
            ),
        )

    def as_step_record(self) -> dict[str, Any]:
        """Project onto the flat per-step summary the record schema carries on every ``Step``.

        Two ``InstrumentEffect`` types exist and they are different objects rather than a duplicate.
        This one is the producer: nanoseconds, quantiles, the ring's counters, the breach list, and
        the checks that could not run. ``record.schema.InstrumentEffect`` is the serialised summary
        that sits on a step, in milliseconds, with one breach string. This method is the one-way
        map, kept as a plain dict so that Plane A does not import the record schema: a hot-path
        package that depends on a serialisation layer has the dependency arrow backwards.

        **What the projection loses, said out loud rather than discovered later.** The ring's drop
        counts collapse into nothing, and a silent drop is the failure this library exists to
        prevent, so a step whose ring dropped records is indistinguishable in the flat form from one
        that did not. ``unchecked`` collapses too, which turns "the allocation budget could not be
        checked" into "the allocation budget was fine". Both belong in the flat form, and until they
        are there the full effect should be kept alongside it rather than instead of it.
        """
        return {
            "added_latency_ms_p50": self.added_ns_p50 / 1e6,
            "added_latency_ms_p99": self.added_ns_p99 / 1e6,
            "added_resident_bytes": self.resident_bytes,
            "added_alloc_bytes": self.added_alloc_bytes_this_step or 0,
            "invocations": self.calls,
            "exceptions": self.recorder_exceptions,
            "disabled": not self.enabled,
            "breach": self.breaches[-1].render() if self.breaches else None,
            "downsample_factor": 1,
        }

    def render(self) -> str:
        head = (
            f"instrument effect for {self.tap_name!r} on run {self.run_id}: {self.calls} calls, "
            f"added {self.added_ns_p50 / 1e3:.2f} us median and {self.added_ms_p99:.4g} ms at p99, "
            f"{self.fraction_mean:.3%} of the wrapped callable's own time"
        )
        lines = [head, f"    resident {self.resident_bytes} bytes"]
        if self.added_alloc_bytes_this_step is not None:
            lines.append(f"    allocated {self.added_alloc_bytes_this_step} bytes this step")
        if self.ring is not None:
            lines.append(f"    ring: {self.ring.render()}")
        if self.recorder_exceptions:
            lines.append(
                f"    recorder raised {self.recorder_exceptions} times, "
                f"{len(self.distinct_recorder_exceptions)} distinct: "
                f"{', '.join(self.distinct_recorder_exceptions)}"
            )
        for b in self.breaches:
            lines.append(f"    BREACH {b.render()}")
        if not self.enabled:
            lines.append("    tap is DISABLED; the host ran on unchanged")
        if self.unchecked:
            lines.append(f"    not checked: {', '.join(self.unchecked)}")
        return "\n".join(lines)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "tap_name": self.tap_name,
            "run_id": self.run_id,
            "calls": self.calls,
            "added_ns_total": self.added_ns_total,
            "inner_ns_total": self.inner_ns_total,
            "added_ns_p50": self.added_ns_p50,
            "added_ns_p99": self.added_ns_p99,
            "added_ns_max": self.added_ns_max,
            "fraction_p99": self.fraction_p99,
            "resident_bytes": self.resident_bytes,
            "added_alloc_bytes_this_step": self.added_alloc_bytes_this_step,
            "step": self.step,
            "ring": self.ring.__canonical__() if self.ring is not None else None,
            "recorder_exceptions": self.recorder_exceptions,
            "distinct_recorder_exceptions": list(self.distinct_recorder_exceptions),
            "enabled": self.enabled,
            "breaches": [b.__canonical__() for b in self.breaches],
            "unchecked": list(self.unchecked),
            "budget": self.budget.__canonical__() if self.budget is not None else None,
            "window_n": self.window_n,
        }


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted sequence.

    Written out rather than taken from numpy for two reasons. It runs on the tap's own bookkeeping
    path where an array allocation would be the largest thing the tap does, and ``tap/`` is part of
    the torch-free core whose import graph is worth keeping short.
    """
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_values[0])
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[int(pos)])
    frac = pos - lo
    return float(sorted_values[lo]) * (1.0 - frac) + float(sorted_values[hi]) * frac


# ---------------------------------------------------------------------------
# The run handle
# ---------------------------------------------------------------------------


@runtime_checkable
class RunHandle(Protocol):
    """The minimum a tap needs from a run.

    Deliberately three attributes and no methods. A method call on the hot path is a cost the tap
    does not control, and a protocol wide enough to be interesting is a protocol the record layer
    would have to be rewritten around. Everything the tap wants beyond this it reads from the call
    context, because that is where a facet actually lives: TRL hands the grader
    ``reward_kwargs["trainer_state"]`` (``grpo_trainer.py:1621``) and the step number is on it.

    ``run_id`` stamps every record. ``ring`` is where records go, and it is an attribute rather
    than a factory so the hot path does not call anything to find it. ``step`` is read with a
    ``getattr`` default, so a run object that does not carry one still satisfies this in practice
    and the tap records ``None`` rather than guessing.
    """

    run_id: str
    ring: "TapRing"
    step: int | None


@dataclass
class SimpleRun:
    """A run handle with nothing but a ring, for tests and for a tap with no record layer yet.

    ``record/`` owns the real one. This exists so that ``tap/`` has no dependency on it, and so
    that a user who wants a grader card from a callable and nothing else does not have to stand up
    a run to get one.
    """

    run_id: str = "anonymous"
    ring: "TapRing" = field(default=None)  # type: ignore[assignment]
    step: int | None = None

    def __post_init__(self) -> None:
        if self.ring is None:
            from reward_lens.tap.ring import TapRing

            self.ring = TapRing()


__all__ = [
    "DEFAULT",
    "RECORD_FOOTPRINT_BYTES",
    "BreachKind",
    "CallOutcome",
    "GraderCall",
    "InstrumentEffect",
    "ReturnKind",
    "ReturnShape",
    "RunHandle",
    "SimpleRun",
    "TapBreach",
    "TapBudget",
    "classify_return",
    "is_sentinel",
    "quantile",
]

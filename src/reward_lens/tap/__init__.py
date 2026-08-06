"""``reward_lens.tap`` — Plane A: the hot path, inside somebody else's run.

Plane A does almost nothing, on purpose. It may read values the host has already computed, reduce
them, and append the reduction to a bounded ring buffer that somebody else's thread drains. It may
not do statistics, inference, decisions or blocking I/O, and it may never raise into the host.
Everything interesting happens in Plane B, out of process, against the record this produces.

The split is forced by measured numbers rather than by taste. Adoption at a lab ends permanently
the first time a monitor crashes a job, and the failure modes are on the record: a vLLM
hidden-state export allocates 2.625 GiB in a single block at 49,152 tokens with chunked prefill
failing to bound it, ending in ``EngineDeadError``; arbitrary Python interventions and CUDA graphs
are mutually exclusive, and eager mode costs 2.7 to 6.3 times.

The core of the subpackage is three files and it imports nothing heavier than the standard library
and ``reward_lens.core``. In particular it imports no torch, and there is a test asserting that
``import reward_lens`` pulls none. The adapters below it import their framework lazily, inside
functions, so that stays true however many of them there are.

``instrument_grader`` is the wedge: one wrapper that records every grader call, losing nothing and
changing nothing, and it needs no framework patch, no GPU and no white-box access.

Attaching this to a training framework is ``tap/adapters/``, which ships adapters for TRL and for
Prime Intellect's ``verifiers``. The seam is deliberate: nothing in this package knows what a
framework is, so an adapter is a function that finds the right callable to wrap and supplies a
``RunHandle``, not a change to any of this. Three things an adapter has to do that this package
cannot do for it. It has to find *every* call site, because TRL calls its reward functions from two
places, ``grpo_trainer.py:1659-1661`` synchronously and ``:1671-1673`` under ``await``, and a tap
that wraps one and not the other sees half the calls. It has to wrap the individual reward function
rather than the aggregator, because ``verifiers``' rubric substitutes ``0.0`` for an exception
before the aggregator ever sees it. And it owns the step boundary, so it is the adapter that calls
``wrapped.effect()`` and hands the resulting ``InstrumentEffect`` to the record.
"""

from __future__ import annotations

from reward_lens.tap.contract import (
    DEFAULT,
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
)
from reward_lens.tap.grader_wrap import (
    DISTINCT_EXCEPTION_LIMIT,
    TapBudgetExceeded,
    TapGuard,
    instrument_grader,
    tap,
)
from reward_lens.tap.ring import RingStats, TapRing

__all__ = [
    "DEFAULT",
    "DISTINCT_EXCEPTION_LIMIT",
    "BreachKind",
    "CallOutcome",
    "GraderCall",
    "InstrumentEffect",
    "RingStats",
    "ReturnShape",
    "RunHandle",
    "SimpleRun",
    "TapBreach",
    "TapBudget",
    "TapBudgetExceeded",
    "TapGuard",
    "TapRing",
    "classify_return",
    "instrument_grader",
    "is_sentinel",
    "tap",
]

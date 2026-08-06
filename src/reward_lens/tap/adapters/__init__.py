"""Framework adapters: what binds ``tap/`` to somebody else's training loop.

Nothing in ``tap/`` knows what a framework is. That is deliberate, and it leaves three jobs that
only an adapter can do.

It has to find *every* call site. TRL calls its reward functions from two places,
``grpo_trainer.py:1659-1661`` synchronously and ``:1671-1673`` under ``await``, and the branch
between them is ``inspect.iscoroutinefunction(reward_func)`` at ``:1655``. A wrapper that is
synchronous around an asynchronous grader does not merely miss half the calls, it changes which
branch TRL takes and hands the host a coroutine where it expects a list.

It has to wrap the individual reward function rather than the aggregator, because an aggregator is
usually where an exception gets turned into a zero and the whole point of ``CallOutcome`` is to be
upstream of that.

And it owns the step boundary. ``tap/`` measures and buffers; it does not know where a step ends,
so it never emits. The adapter is what calls ``effect()`` at a boundary and turns a pile of
``GraderCall`` records into the five-level hierarchy in ``record/``.

Every module in here imports its framework lazily, inside functions. ``tap/`` is part of the
torch-free core and importing this package must not change that, so nothing at module scope here
imports ``trl``, ``transformers`` or ``torch``.
"""

from __future__ import annotations

#: Empty on purpose. Importing an adapter should be an explicit
#: ``from reward_lens.tap.adapters.trl import TRLTap``, so that a star-import of this package can
#: never be the thing that drags a framework into a process that did not ask for one.
__all__: list[str] = []

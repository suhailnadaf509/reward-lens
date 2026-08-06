"""One shared shape for the four control instruments: compute once, refuse or emit.

`Observable.measure` returns `Evidence` by contract and `Instrument.estimate` returns
`Reading`, which is `Evidence | Refusal`. Those two contracts are both right and they need a
seam, because the four instruments here decide to refuse partway through a computation rather
than in preflight: M5 cannot know whether it is looking at a null until it has read the claim, and
M4 cannot know whether a placebo arm is missing until it has looked.

So the seam is here, once. A subclass writes `compute`, which returns its own result type or a
`Refusal`. `estimate` runs preflight, computes, returns the refusal if there is one, and otherwise
hands the computed result to `run`, which applies the capability check and gate 2 before `measure`
turns it into Evidence. Nothing computes twice.

Calling `measure` directly on an instrument whose `compute` would refuse raises, and that is not a
softening of the refusal rule. A `Refusal` is a value on the `estimate` path, which is the path
every caller uses. Reaching past it into the narrow `Observable` protocol and asking for `Evidence`
from a measurement that declined to produce one is a programming error, and the exception says so.
"""

from __future__ import annotations

from typing import Any

from reward_lens.core.evidence import Evidence
from reward_lens.core.gates import require_frame_for_comparison
from reward_lens.core.reading import Reading, Refusal
from reward_lens.measure.base import BaseObservable, Context, run


class ControlInstrument(BaseObservable):
    """Preflight, compute, refuse or emit. The four control instruments share this."""

    #: Set by `estimate` for the duration of one call, so `measure` does not recompute.
    _computed: Any = None

    def compute(self) -> Any:  # pragma: no cover - abstract
        """The instrument's own work, with no `Context`. Returns a result or a `Refusal`.

        Written without a Context on purpose: all four of these are pure functions of data the
        caller already holds, and being able to call them without standing up a signal is what
        makes them usable in a preflight and in a test.
        """
        raise NotImplementedError

    def payload(self, computed: Any) -> dict[str, Any]:  # pragma: no cover - abstract
        """The Evidence value: a flat mapping, including the `baselines` key M3's lint looks for."""
        raise NotImplementedError

    def gated_emit(self, ctx: Context, computed: Any) -> Evidence:
        """Hand a computed result to the runner, or apply the runner's gates by hand.

        `Context.signal` is optional now, and these four never touch it: they read injected data
        rather than a network. But `run` still resolves `ctx.signal.caps` to enforce the capability
        check, so calling it with a signal-free context raises `AttributeError` instead of
        emitting. Until the runner guards that, the no-signal branch here does what `run` would
        have done minus the check that has nothing to check against.

        The frame requirement is **not** skipped. Gate 2 depends on the instrument's gauge status
        and the context's frame rather than on the signal, so it is applied in both branches. A
        gate that is convenient to drop is exactly the one worth keeping.
        """
        self._computed = computed
        try:
            if ctx.signal is not None:
                return run(self, ctx)
            if ctx.is_comparison:
                require_frame_for_comparison(self.gauge_status, ctx.frame)
            # `run` sets this before delegating and the no-signal branch has to do the same. Left
            # unset, `Context.emit` finds no observable and stamps `observable='anonymous'`,
            # `observable_version='0'` and `quantity=''`, so the reading is attributed to nobody
            # and the unit machinery has nothing to key on.
            ctx._observable = self
            try:
                return self.measure(ctx)
            finally:
                ctx._observable = None
        finally:
            self._computed = None

    def estimate(self, ctx: Context) -> Reading:
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        out = self.compute()
        if isinstance(out, Refusal):
            return out
        return self.gated_emit(ctx, out)

    def measure(self, ctx: Context) -> Evidence:
        out = self._computed if self._computed is not None else self.compute()
        if isinstance(out, Refusal):
            raise ValueError(
                f"{self.name}.measure was called on a measurement that declines to produce "
                f"Evidence: {out.reason.name}. Call `estimate`, which returns the refusal as a "
                f"value with its remedy."
            )
        return ctx.emit(self.payload(out))


__all__ = ["ControlInstrument"]

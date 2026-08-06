"""Preflight, compute, refuse or emit: the shape the five Level 0 instruments share.

`Observable.measure` returns `Evidence` by contract and `Instrument.estimate` returns `Reading`,
which is `Evidence | Refusal`. The five instruments here decide to refuse partway through the
arithmetic rather than in preflight, because none of them can know whether it will refuse until it
has looked at the weights: the horizon cannot know whether the floor binds before it has swept, and
the tail index cannot know whether it has enough exceedances before it has counted them. So the
seam sits here, once, and a subclass writes `compute`.

`measure.controls._base.ControlInstrument` has the same shape for the same reason. It is not
imported because it is a private module of a different package and one instrument family reaching
into another's underscore is a dependency that is invisible at the point where it breaks. The
duplicated part is thirty lines of dispatch.
"""

from __future__ import annotations

from typing import Any

from reward_lens.core.evidence import Evidence
from reward_lens.core.gates import require_frame_for_comparison
from reward_lens.core.reading import Reading, Refusal
from reward_lens.measure.base import BaseObservable, Context, run


class FrontierInstrument(BaseObservable):
    """Level 0's runner: preflight, compute once, return the refusal or emit the Evidence."""

    #: Set by `estimate` for the duration of one call so `measure` does not recompute.
    _computed: Any = None

    def compute(self) -> Any:  # pragma: no cover - abstract
        """The instrument's own arithmetic, with no `Context`. Returns a payload or a `Refusal`.

        Written without a Context on purpose. Every reading in this layer is a pure function of two
        arrays the caller already holds, which is what makes the whole layer runnable before any
        training exists and testable without standing up a signal.
        """
        raise NotImplementedError

    def gated_emit(self, ctx: Context, computed: Any) -> Evidence:
        """Hand a computed payload to the runner, or apply the runner's gates by hand.

        `run` resolves `ctx.signal.caps` to enforce the capability check, and these five never
        touch a signal: they read injected scores. The no-signal branch does what `run` would do
        minus the check that has nothing to check against. Gate 2 is applied in both branches
        because it depends on the instrument's gauge status and the context's frame rather than on
        the signal, and a gate that is convenient to drop is the one worth keeping.
        """
        self._computed = computed
        previous = ctx._observable
        try:
            if ctx.signal is not None:
                return run(self, ctx)
            if ctx.is_comparison:
                require_frame_for_comparison(self.gauge_status, ctx.frame)
            # `Context.emit` reads the instrument's name, version, gauge status and **quantity** off
            # `ctx._observable`, and `measure.base.run` is the only thing that used to set it. This
            # branch bypasses `run` because there is no signal to gate on, so every reading it
            # produced was emitted as `anonymous` with `quantity=""`. That silently unmakes the
            # unit discipline: a per-token reading with no quantity on it can be ranked against a
            # per-sequence one and the unit machinery has nothing to key on.
            ctx._observable = self
            return self.measure(ctx)
        finally:
            self._computed = None
            ctx._observable = previous

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
                f"value carrying its remedy."
            )
        return ctx.emit(out)


__all__ = ["FrontierInstrument"]

"""Preflight, compute, refuse or emit: the shape the six series E instruments share.

`Observable.measure` returns `Evidence` by contract and `Instrument.estimate` returns `Reading`,
which is `Evidence | Refusal`. Every instrument here decides to refuse partway through the
arithmetic rather than in preflight, because none of them can know whether it will refuse until it
has looked at the record: E2 cannot know whether the all-fail flag was ever populated until it has
read the groups, and E4 cannot know whether the recorded estimator z-scores at all until it has read
the spec. So the seam sits here, once, and a subclass writes `compute`.

`measure.controls._base.ControlInstrument` and `measure.frontier._base.FrontierInstrument` have the
same shape for the same reason. Neither is imported: both are private modules of other packages, and
one instrument family reaching into another's underscore is a dependency that is invisible at the
point where it breaks. The duplicated part is thirty lines of dispatch.

**One kernel behaviour this file works around, deliberately.** `measure/base.py`'s `run()` raises
`CapabilityError` where an instrument should return a `Refusal`. Nothing here depends on the
exception: every instrument in this package declares
`Capability.NONE` and reads a record through the access matrix, so the raising branch is
unreachable from `estimate`, and when the kernel starts returning a refusal instead, this file needs
no change.
"""

from __future__ import annotations

from typing import Any

from reward_lens.core.evidence import Evidence
from reward_lens.core.gates import require_frame_for_comparison
from reward_lens.core.reading import Reading, Refusal
from reward_lens.measure.base import BaseObservable, Context, run


class EstimatorInstrument(BaseObservable):
    """Series E's runner: preflight, compute once, return the refusal or emit the Evidence."""

    #: Set by `estimate` for the duration of one call so `measure` does not recompute.
    _computed: Any = None

    def compute(self) -> Any:  # pragma: no cover - abstract
        """The instrument's own arithmetic, with no `Context`. Returns a payload or a `Refusal`.

        Written without a Context on purpose. Every reading in this series is a pure function of a
        record the caller already holds, which is what makes the whole series testable against a
        hand-built group whose z-score you can do on paper.
        """
        raise NotImplementedError

    def gated_emit(self, ctx: Context, computed: Any) -> Evidence:
        """Hand a computed payload to the runner, or apply the runner's gates by hand.

        `run` resolves `ctx.signal.caps` to enforce the capability check, and none of these six
        touches a signal: they read a record. The no-signal branch does what `run` would do minus
        the check that has nothing to check against. Gate 2 is applied in both branches because it
        depends on the instrument's gauge status and the context's frame rather than on the signal.
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


__all__ = ["EstimatorInstrument"]

"""What the four contract instruments share: the runner seam and the declarations.

`Observable.measure` returns `Evidence` by contract and `Instrument.estimate` returns `Reading`,
which is `Evidence | Refusal`. The four instruments here decide to refuse partway through, because
none of them can know whether it will refuse until it has looked at which parameters the caller
stated. So the seam sits here once, the same shape `measure.metrology.gstudy.MetrologyInstrument` and
`measure.frontier._base.FrontierInstrument` use, and a subclass writes `compute` and `payload`.

Neither of those is imported. Both are private modules of other instrument families, and one family
reaching into another's underscore is a dependency that is invisible at the point where it breaks.
The duplicated part is forty lines of dispatch.

**Every payload here carries the five assumptions and the parameter provenance.** That is enforced in
`measure` rather than left to each subclass, because an instrument that could forget it eventually
would, and a weight recommendation with its assumptions missing is the one output this package must
not be able to produce.
"""

from __future__ import annotations

from typing import Any, Mapping

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Evidence, Uncertainty
from reward_lens.core.gates import require_frame_for_comparison
from reward_lens.core.reading import Reading, Refusal
from reward_lens.core.types import Access, Component, Phase, Substrate
from reward_lens.measure.base import BaseObservable, Context, run
from reward_lens.measure.decision.assumptions import assumptions_payload, render_assumptions
from reward_lens.measure.rate.regime import MEASURED_BY

# ---------------------------------------------------------------------------
# The shared declarations
# ---------------------------------------------------------------------------

#: Every substrate. The contract model is about a weighted sum of signals and says nothing about
#: what produced any of them, so a unit test and a judge are the same kind of object here.
ALL_SUBSTRATES: frozenset[Substrate] = frozenset(
    {
        Substrate.NEURAL_SCALAR,
        Substrate.NEURAL_GEN,
        Substrate.PROGRAM,
        Substrate.PROCEDURAL,
        Substrate.HUMAN,
        Substrate.COMPOSITE,
    }
)

#: `PRE_RUN` because the whole point is answering before anything is optimised, and `POST_RUN`
#: because a finished run's weights can be read against the same model after the fact.
#:
#: `IN_RUN` is excluded and the exclusion is a claim rather than an oversight. The model has one
#: period and full commitment, which is the assumption `COMMITMENT_ONE_PERIOD` prints on every
#: reading. A weight changed mid-run violates it, so an in-run reading of this layer would be a
#: number computed under a premise the same reading declares false.
CONTRACT_PHASES: frozenset[Phase] = frozenset({Phase.PRE_RUN, Phase.POST_RUN})

#: `STATIONARY_GRADER` is the measurable half of `COMMITMENT_ONE_PERIOD`. A grader whose weights or
#: rubric moved across the window is a multi-period problem wearing a single-period model, and the
#: static optimum is not the dynamic one, so the honest behaviour is to refuse rather than to report
#: a fixed point the run cannot reach.
CONTRACT_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
    measured_by={RegimeCondition.STATIONARY_GRADER: MEASURED_BY[RegimeCondition.STATIONARY_GRADER]},
    on_violation="refuse",
)

#: Sigma is the one measured parameter and A2 needs facet control to produce it.
NOISE_ACCESS: dict[Component, Access] = {Component.GRADER: Access.REPLICATE}

#: The composite's weights are in the record; the sensitivity that turns them into commissions is
#: not, and the rung that needs more checks for it itself and refuses by name.
WEIGHTS_ACCESS: dict[Component, Access] = {Component.GRADER: Access.RECORD}


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


class DecisionInstrument(BaseObservable):
    """Preflight, compute once, refuse or emit, with the assumptions attached on the way out."""

    #: Set by `estimate` for the duration of one call so `measure` does not recompute.
    _computed: Any = None

    substrates = ALL_SUBSTRATES
    phases = CONTRACT_PHASES
    envelope = CONTRACT_ENVELOPE

    def compute(self) -> Any:  # pragma: no cover - abstract
        """The instrument's arithmetic, with no `Context`. Returns a payload or a `Refusal`.

        Written without a Context on purpose. Every reading in this layer is a pure function of a
        parameter set the caller already holds, which is what makes the layer runnable before any
        training exists and testable without standing up a signal.
        """
        raise NotImplementedError

    def payload(self, computed: Any) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def uncertainty(self, computed: Any) -> Uncertainty | None:
        """No interval, and that is the honest answer rather than a gap.

        The uncertainty of a weight recommendation is dominated by the parameters nobody measured,
        not by the sampling error of the one that was. An interval around `alpha*` that carried only
        `Sigma`'s sampling error would be narrow, correct in its own terms, and read as the
        uncertainty of the recommendation, which it is not by orders of magnitude. The sweep is the
        interval this layer has, and it lives on the reading rather than in this field.
        """
        return None

    def gated_emit(self, ctx: Context, computed: Any) -> Evidence:
        """Hand a computed payload to the runner, or apply the runner's gates by hand.

        `run` resolves `ctx.signal.caps` to enforce the capability check and these four never touch
        a signal: they read a parameter set. The no-signal branch does what `run` would do minus the
        check that has nothing to check against, including setting `ctx._observable`, which is what
        `Context.emit` reads the name, version and quantity off. Gate 2 applies in both branches,
        because it depends on the instrument's gauge status and the context's frame rather than on
        the signal.
        """
        self._computed = computed
        try:
            if ctx.signal is not None:
                return run(self, ctx)
            if ctx.is_comparison:
                require_frame_for_comparison(self.gauge_status, ctx.frame)
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
                f"value carrying its remedy."
            )
        body = self.payload(out)
        body["assumptions"] = assumptions_payload()
        body["assumptions_rendered"] = render_assumptions()
        declared = body.get("baselines")
        return ctx.emit(
            body,
            uncertainty=self.uncertainty(out),
            baselines=declared if isinstance(declared, Mapping) else None,
        )


__all__ = [
    "ALL_SUBSTRATES",
    "CONTRACT_ENVELOPE",
    "CONTRACT_PHASES",
    "NOISE_ACCESS",
    "WEIGHTS_ACCESS",
    "DecisionInstrument",
]

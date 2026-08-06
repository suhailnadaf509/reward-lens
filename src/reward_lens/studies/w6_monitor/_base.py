"""What `w6_4` and `w6_5` share: the runner, the access matrices, the envelopes, and d-prime.

Both packages are monitor-under-pressure designs and both turn on the same separation. `w6_5`
splits a falling detection curve into *evasion* (the policy's hacks stopped looking suspicious) and
*degradation* (the monitor stopped discriminating). `w6_4` splits a rising false-positive mass into
*selection* (training moved probability onto failure modes the base policy already produced) and
*exploitation* (training found failure modes the base policy did not have). In both cases an
aggregate moved, two mechanisms could have moved it, and the reading is worthless unless it says
which.

The move that makes both separable is the same one, and it is metrology's rather than machine
learning's: hold something fixed and re-measure it. A check standard is a stable artefact measured
in every session so that instrument drift can be told from subject change; `monitor/check_standard`
is that instrument and J5 is its shipped form. `w6_5`'s frozen probe set and `w6_4`'s base-policy
family sample are the same idea applied to two different subjects.

**Neither package is run here.** Every instrument below is asserted against a planted subject whose
answer is known by construction, and each one's docstring names the real subject its claim needs and
what that subject costs. `study.py` carries the two prices and the two runbooks.

**The runner is thirty lines of dispatch duplicated from `measure/threshold/_base.py` rather than
imported from it.** That module says why in its own header and the argument holds here: it is a
private module of another package, and one instrument family reaching into another's underscore is
a dependency that is invisible at the point where it breaks. `monitor/_base.py` has a third copy
with a richer `payload`/`baseline_map`/`uncertainty` split; neither is imported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Evidence, register_payload
from reward_lens.core.gates import require_frame_for_comparison
from reward_lens.core.quantity import CostModel
from reward_lens.core.reading import Reading, Refusal
from reward_lens.core.types import Access, Component, Phase, Substrate
from reward_lens.measure.base import BaseObservable, Context, run

#: Where a monitor or a verifier can live. Both rows read a scoring decision and neither cares
#: whether the thing that produced it is a network or a program: a linear probe on activations, a
#: generative judge and a test suite are all monitors for this purpose, and `w6_5` ranks across
#: exactly that mixture on purpose.
ALL_SUBSTRATES = frozenset(
    {
        Substrate.NEURAL_SCALAR,
        Substrate.NEURAL_GEN,
        Substrate.PROGRAM,
        Substrate.PROCEDURAL,
        Substrate.HUMAN,
        Substrate.COMPOSITE,
    }
)

#: I4's declared minimum. `Access.CONTROL` is the honest entry: the monitor has to be *in the loop*,
#: which means intervening on training rather than reading a log of it. The record is named as a
#: second component because the curve is per step and the steps come off the record.
CONTROL_ACCESS: dict[Component, Access] = {
    Component.OPTIMIZER: Access.CONTROL,
    Component.RECORD: Access.RECORD,
}

#: D5's behavioural half. The grader is queried (to re-score held-out rollouts against a strict
#: reference), the record supplies the per-step rollouts, and `GOLD` supplies the strict reference
#: that decides which accepted answers were wrong. The static half needs the first and the third;
#: the behavioural half needs the record as well, and that difference is the whole package.
BEHAVIOURAL_ACCESS: dict[Component, Access] = {
    Component.GRADER: Access.QUERY,
    Component.GOLD: Access.QUERY,
    Component.RECORD: Access.RECORD,
    Component.OPTIMIZER: Access.CONTROL,
}

#: Both rows are questions about a run in progress or a run that finished. Neither can be asked of a
#: grader before training starts, which is what separates them from the static halves.
RUN_PHASES = frozenset({Phase.IN_RUN, Phase.POST_RUN})

#: What I4 needs, and the catalogue names it: `STATIONARY_GRADER`.
#:
#: The condition is on the *arm without the monitor*, and the reason is specific. The degradation
#: curve is a comparison between what the monitor scores now and what it scored at step zero. If
#: the grader was retuned mid-run, the rollout population changes for a reason that has nothing to
#: do with the monitor or with the policy's stealth, and the curve records the retune. That is the
#: direction that manufactures a finding: a grader change reliably moves both class means and would
#: be read as degradation.
#:
#: `on_violation="refuse"` rather than "downgrade", against the precedent in `measure/threshold`,
#: because the two are answering different questions. A McCrary density stays computable under a
#: drifting grader and only becomes less trustworthy. A degradation curve under a drifting grader is
#: not a weak measurement of monitor degradation, it is a measurement of something else.
DEGRADATION_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
    measured_by={RegimeCondition.STATIONARY_GRADER: "env.replay_fidelity"},
    on_violation="refuse",
)

#: What the behavioural half needs. The same condition, for a different reason worth stating
#: separately rather than sharing the object with `DEGRADATION_ENVELOPE`.
#:
#: The whole claim is that *training* moved false-positive mass. If the grader moved instead, the
#: mass moved for a reason the design is not measuring, and the selection-versus-exploitation split
#: is computed against a base-policy sample drawn under a grader that no longer exists. The
#: comparison is between two arms that differ only in their verifier, so a verifier that changes
#: within an arm destroys the contrast the arms were built to make.
BEHAVIOURAL_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
    measured_by={RegimeCondition.STATIONARY_GRADER: "env.replay_fidelity"},
    on_violation="refuse",
)

#: What reading a stored curve costs once the run exists. The run is the price, not the arithmetic;
#: `study.py` carries that number and its assumptions.
FREE_READ = CostModel(note="free; reads per-step scores already on the record")

#: I4's declared baselines, in the catalogue's own words. The first is the one that decides the
#: reading: a monitor scored on the same rollouts and never used in the reward has the same
#: distribution shift and none of the optimisation pressure, so the difference between the two
#: curves is what pressure did.
DEGRADATION_BASELINES: tuple[str, ...] = (
    "parallel_monitor_not_optimised_against",
    "baseline.gradnorm_peak",
)

#: D5's declared baseline is the random-mutation search, which the static half already runs and
#: which its kill condition already fired on. The second is the one the behavioural half turns on:
#: the base policy's own error distribution. A static audit that does not order realised
#: false-positive mass better than "look at what the untrained model already gets wrong" has not
#: earned the run, and the published mechanism evidence says that is a live possibility rather than
#: a formality.
BEHAVIOURAL_BASELINES: tuple[str, ...] = (
    "random_mutation",
    "base_policy_error_distribution",
)


class W6Instrument(BaseObservable):
    """Preflight, compute once, return the refusal or emit the Evidence.

    `compute` takes no `Context` on purpose, the same choice series I made and for the same reason:
    every reading in both packages is a pure function of arrays the caller already holds, which is
    what makes the whole thing testable against a curve whose decay you put there yourself.
    """

    #: Set by `estimate` for the duration of one call so `measure` does not recompute.
    _computed: Any = None

    def compute(self) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def gated_emit(self, ctx: Context, computed: Any, **kwargs: Any) -> Evidence:
        """Hand a computed payload to the runner, or apply the runner's gates by hand.

        ``ctx._observable`` is set around both branches. `Context.emit` reads the observable's name,
        version, gauge status and quantity off it, and `run()` is the only place in the kernel that
        sets it, so a no-signal branch that calls `measure` directly emits `observable="anonymous"`
        and `quantity=""` whatever the instrument declared. That is the road every record-only
        instrument takes; `measure/threshold/_base.py` found and fixed it, and this is the same fix
        in the same shape.
        """
        self._computed = computed
        self._emit_kwargs = kwargs
        previous = ctx._observable
        ctx._observable = self
        try:
            if ctx.signal is not None:
                return run(self, ctx)
            if ctx.is_comparison:
                require_frame_for_comparison(self.gauge_status, ctx.frame)
            return self.measure(ctx)
        finally:
            ctx._observable = previous
            self._computed = None
            self._emit_kwargs = {}

    _emit_kwargs: dict = {}

    def estimate(self, ctx: Context | None = None) -> Reading:
        ctx = ctx if ctx is not None else Context()
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        out = self.compute()
        if isinstance(out, Refusal):
            return out
        return self.gated_emit(ctx, out, **self.emit_kwargs(out))

    def emit_kwargs(self, computed: Any) -> dict:
        """Extra keywords for `Context.emit`. Overridden where a reading is dual-use."""
        return {}

    def measure(self, ctx: Context) -> Evidence:
        out = self._computed if self._computed is not None else self.compute()
        if isinstance(out, Refusal):
            raise ValueError(
                f"{self.name}.measure was called on a measurement that declines to produce "
                f"Evidence: {out.reason.name}. Call `estimate`, which returns the refusal as a "
                f"value carrying its remedy."
            )
        return ctx.emit(out, **self._emit_kwargs)


# ---------------------------------------------------------------------------
# d-prime
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class Discriminability:
    """Signal detection's `d'` for one monitor at one step, with the two terms it factors into.

    `d' = (mu_hack - mu_clean) / sigma_pooled`. It is written out rather than imported because
    nothing in the library computes it: `stats/roc.py` ships `roc_pr`, `calibration_curve` and
    `operating_point` and has no d-prime, no confidence interval and no DeLong test. Four lines of
    arithmetic is not a second implementation of anything.

    The two factors are the reason this type exists rather than a bare float. `separation` is the
    numerator and `spread` the denominator, and a `d'` that halved because the classes moved
    together means something different from one that halved because both classes got noisier. The
    first is a monitor losing the signal, the second is a monitor losing the resolution to see it,
    and only the pair distinguishes them.

    `auroc` is carried alongside because the published curve reports both and because they can
    disagree: `d'` assumes two Gaussians with a common variance and AUROC assumes nothing, so a
    heavy-tailed monitor score can hold its AUROC while its `d'` collapses. Where they disagree the
    AUROC is the one to believe and the `d'` is the one that decomposes.
    """

    separation: float
    spread: float
    d_prime: float
    auroc: float
    n_positive: int
    n_negative: int

    @property
    def usable(self) -> bool:
        """Whether both classes were populated and the spread is positive."""
        return bool(
            self.n_positive > 0
            and self.n_negative > 0
            and np.isfinite(self.d_prime)
            and self.spread > 0.0
        )

    def render(self) -> str:
        return (
            f"d' = {self.d_prime:.3f}  (separation {self.separation:.3f} / spread "
            f"{self.spread:.3f}), AUROC {self.auroc:.3f} on {self.n_positive}+/{self.n_negative}-"
        )


def discriminability(
    hack_scores: Sequence[float] | np.ndarray,
    clean_scores: Sequence[float] | np.ndarray,
) -> Discriminability:
    """`d'` and AUROC for one monitor against one pool of labelled rollouts.

    The pooled standard deviation is the unweighted root-mean-square of the two class standard
    deviations, which is the equal-variance convention `d'` is defined under. A weighted pooled
    variance would be the right choice if the two classes were samples of one population; they are
    not, and using the sample sizes here would let a class imbalance that changes over training move
    `d'` without either distribution moving.

    Returns a `Discriminability` with a non-finite `d_prime` where either class is empty or the
    spread is zero, rather than raising or substituting a number. `usable` is the field to check.
    """
    from reward_lens.stats.roc import roc_pr

    h = np.asarray(hack_scores, dtype=np.float64).ravel()
    c = np.asarray(clean_scores, dtype=np.float64).ravel()
    h = h[np.isfinite(h)]
    c = c[np.isfinite(c)]
    if h.size == 0 or c.size == 0:
        return Discriminability(
            float("nan"), float("nan"), float("nan"), float("nan"), h.size, c.size
        )

    separation = float(np.mean(h) - np.mean(c))
    sd_h = float(np.std(h, ddof=1)) if h.size > 1 else 0.0
    sd_c = float(np.std(c, ddof=1)) if c.size > 1 else 0.0
    spread = float(math.sqrt(0.5 * (sd_h**2 + sd_c**2)))
    d = separation / spread if spread > 0 else float("nan")

    scores = np.concatenate([h, c])
    labels = np.concatenate([np.ones(h.size, dtype=int), np.zeros(c.size, dtype=int)])
    auc = float(roc_pr(scores, labels).auc)
    return Discriminability(separation, spread, float(d), auc, int(h.size), int(c.size))


__all__ = [
    "ALL_SUBSTRATES",
    "BEHAVIOURAL_ACCESS",
    "BEHAVIOURAL_BASELINES",
    "BEHAVIOURAL_ENVELOPE",
    "CONTROL_ACCESS",
    "DEGRADATION_BASELINES",
    "DEGRADATION_ENVELOPE",
    "FREE_READ",
    "RUN_PHASES",
    "Discriminability",
    "W6Instrument",
    "discriminability",
]

"""A10 Dark reward: the fraction of reward variance through no named channel.

Formal definition, A10. Dark reward is the fraction of ``Var(r)`` not causally mediated by any
named channel (criterion or feature). It is a card statistic: the reward variance a full accounting of
the intended criteria still cannot explain, the leakage capacity theory (S5) predicts grows with
``K/d_eff`` as the reward tries to carry more criteria than its effective dimension supports (A9).

Deviation from A10: the pure function measures the variance of ``r`` not linearly explained by the
named-channel contributions (``1 − R²`` of ``r`` on the channels), which is the observational reading;
the causal-mediation reading substitutes steering-measured channel contributions and is the production
path. The synthetic test plants ``r`` as named channels plus a known dark component and recovers its
fraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.types import Access, AccessMatrix, Capability, Component, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context, PreflightResult
from reward_lens.measure.indices._support import (
    ANY_SUBSTRATE,
    GRADER_STUDY_PHASES,
    MEASURED_BY,
    measured_without_input,
    missing_injection,
    reward_scores,
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def dark_reward(reward: np.ndarray, named_contributions: np.ndarray) -> float:
    """Fraction of ``Var(r)`` not explained by the named channels: ``1 − R²`` (A10).

    Regresses the reward on the named-channel contributions (``n × K``) and returns one minus the
    fraction of variance explained. All variance captured by the channels gives ``0`` (nothing dark);
    a reward orthogonal to every channel gives ``1`` (entirely dark). ``reward`` is ``(n,)``. A
    constant term is included so the channels are not charged for the reward's mean.
    """
    r = np.asarray(reward, dtype=np.float64).ravel()
    c = np.asarray(named_contributions, dtype=np.float64)
    if c.ndim == 1:
        c = c[:, None]
    var_r = float(np.var(r, ddof=0))
    if var_r == 0:
        return float("nan")
    design = np.column_stack([np.ones(c.shape[0]), c])
    coef, *_ = np.linalg.lstsq(design, r, rcond=None)
    resid = r - design @ coef
    return float(np.var(resid, ddof=0) / var_r)


class DarkReward(BaseObservable):
    """A10 fraction of reward variance mediated by no named channel.

    Requires scores. The named-channel contributions are injected (the concept/criterion layer supplies
    them in production; the causal reading uses steering-measured contributions). Reports the dark
    fraction. Gauge is INVARIANT: a variance fraction is scale-free.

    What it cannot do. This is ``1 - R^2`` of a linear regression, so every channel that acts on
    the reward non-linearly is counted as dark, and a reward that is a perfectly legible quadratic
    of one named channel reads as almost entirely unexplained. The complement is just as wrong in
    the other direction: adding channels can only lower the dark fraction, so a battery large
    relative to the sample drives it toward zero for arithmetic reasons. The dark fraction is a
    residual, and a residual attributes nothing; it says a variance is unaccounted for, not that
    anything in particular is hiding in it.
    """

    name = "DarkReward"
    version = "1.0"
    capabilities = Capability.SCORES
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A10"
    deviations = (
        "observational reading (1 - R^2 of r on the named channels); the causal-mediation reading "
        "uses steering-measured channel contributions and is the production path",
    )

    # -- the observable declarations ---------------------------------------
    quantity = "grader.dark_fraction"
    #: The grader is scored on the view; the named-channel contributions come from an earlier
    #: measurement by the concept or criterion layer.
    requires: AccessMatrix = {
        Component.GRADER: Access.QUERY,
        Component.RECORD: Access.RECORD,
    }
    substrates = ANY_SUBSTRATE
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset(
            {RegimeCondition.STATIONARY_GRADER, RegimeCondition.GROUP_NONDEGENERATE}
        ),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: ``R^2`` is unchanged by an affine transformation of the response, so the dark fraction does
    #: not move when the reward is rescaled or shifted.
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = ("baseline.random_channels", "baseline.shuffled_channels")
    rung = 0

    def __init__(self, named_contributions: np.ndarray | None = None) -> None:
        self.named_contributions = named_contributions

    def preflight(self, ctx: Context) -> PreflightResult:
        """The channel decomposition or a refusal. Dark reward is a residual and needs something to be dark against.

        The injected input is absent, which makes this a `Refusal` rather than an Evidence
        carrying a note. Nothing has to be computed to know it, so the question belongs
        here: `estimate` returns this refusal before `measure` is reached, and the
        capability report gets it with no work at all.
        """
        if self.named_contributions is None:
            return missing_injection(
                self,
                needs={
                    "named_contributions": "an (n, K) matrix of per-sample contributions from the K named channels",
                },
                have="no channel decomposition was injected",
                remedy=(
                    "Construct `DarkReward(named_contributions=...)` with one column per named reward "
                    "channel and one row per sample, in the same order as the rewards the signal returns. "
                    "The reading is the share of reward variance those columns do not explain, so a reward "
                    "with no named channels has nothing to be dark against and would read 1.0 for the wrong "
                    "reason."
                ),
            )
        return super().preflight(ctx)

    def measure(self, ctx: Context) -> "Evidence":
        if self.named_contributions is None:
            raise measured_without_input(self)
        reward = reward_scores(ctx.signal, ctx.view, ctx.readout)
        dark = dark_reward(reward, self.named_contributions)
        n_channels = int(np.atleast_2d(np.asarray(self.named_contributions).T).shape[0])
        return ctx.emit(
            {"dark_reward": dark, "explained_fraction": 1.0 - dark, "n_channels": n_channels},
            uncertainty=Uncertainty(n=int(reward.size), method="none"),
        )


__all__ = ["dark_reward", "DarkReward"]

"""Level 0, the frontier: what would happen if we optimised, answered before anything is optimised.

The layer develops one potential function, ``K(lambda) = log E_0[e^{lambda r}]``, and everything
in this package is a derivative of it or a ratio built from its weights. Five readings come out,
carried by five instruments over four catalogue records:

- **N1** `GoldVersusKL`, `frontier.gold_vs_kl`. The reward-versus-gold curve, out to the horizon.
- **N2** `VisibilityHorizon`, `frontier.visibility_horizon`. Where the tilt extrapolation goes
  blind, in nats. This is the piece of the layer that is genuinely unoccupied.
- **N3** `RewardTailIndex`, `frontier.tail_index`. The `LIGHT_TAILED` precondition the layer rests on,
  measured rather than assumed, with a stability protocol and a refusal below the exceedance count a
  defensible estimate needs.
- **N4** `SurrogateChecklist`, `frontier.prentice_checklist`, and `ConcomitantBestOfN`,
  `frontier.concomitant_bon`. Falsifiable conditions on the proxy, and the exact finite-n
  distribution theory for the gold at the proxy-argmax.

The whole layer needs a callable grader and a gold channel on the same n base-policy rollouts, and
nothing else: no GPU, no policy checkpoint, no record, no gradients. That is the entire content of
the claim that the frontier is estimable before any optimisation is run, for any substrate,
including a closed API. It is the only layer in this library that answers a question
before training happens, and it imports no torch.

Two things about it are worth knowing before reading a number out of it.

**Refusing is the feature, not the fallback.** N2 exists so that N1 can decline to answer past the
horizon, and `ESS_BELOW_FLOOR` is a reading rather than an error. An instrument that reports the
frontier out to any lambda you ask for is not more capable than this one, it is quieter about the
same limitation.

**The turning point is occupied and this package says so on every reading.** arXiv 2506.19248
(NeurIPS 2025 Spotlight) defines the same tilt, states the same stationarity condition, and ships
HedgeTune. HedgeTune is implemented in `curve.py` and is a mandatory baseline: both turning-point
estimates appear side by side on every N1 reading. What is ours is the horizon, the concomitant
framing and the surrogate conditions.
"""

from __future__ import annotations

from reward_lens.measure.frontier.curve import (
    FRONTIER_ACCESS,
    FRONTIER_BASELINES,
    FrontierReading,
    GoldVersusKL,
    TurningPoint,
    cumulant_turning_point,
    hedgetune,
    measure_frontier,
)
from reward_lens.measure.frontier.horizon import (
    ALL_SUBSTRATES,
    HORIZON_ACCESS,
    HORIZON_BASELINES,
    HorizonReading,
    VisibilityHorizon,
    light_tailed_envelope,
    measure_horizon,
)
from reward_lens.measure.frontier.potential import (
    DEFAULT_ESS_FLOOR,
    DEFAULT_T_CAP,
    Potential,
    TiltPoint,
    horizon_lambda,
    logsumexp,
)
from reward_lens.measure.frontier.surrogate import (
    CHECKLIST_BASELINES,
    CONCOMITANT_BASELINES,
    N4_ACCESS,
    N4_ENVELOPE,
    ChecklistReading,
    ConcomitantBestOfN,
    ConcomitantReading,
    Criterion,
    SurrogateChecklist,
    Verdict,
    concomitant_expectation,
    measure_checklist,
    measure_concomitant,
    simulate_concomitant,
)
from reward_lens.measure.frontier.tails import (
    DEFAULT_GAMMA_MAX,
    DEFAULT_TAIL_QUANTILE,
    MIN_EXCEEDANCES,
    TAIL_BASELINES,
    TAIL_ENVELOPE,
    Plateau,
    RewardTailIndex,
    TailReading,
    find_plateau,
    hill,
    measure_tail_index,
    pickands,
)

#: The five instruments, in the order the layer is read: horizon first, because it bounds the rest.
FRONTIER: tuple[type, ...] = (
    VisibilityHorizon,
    GoldVersusKL,
    RewardTailIndex,
    SurrogateChecklist,
    ConcomitantBestOfN,
)

__all__ = [
    "ALL_SUBSTRATES",
    "CHECKLIST_BASELINES",
    "CONCOMITANT_BASELINES",
    "DEFAULT_ESS_FLOOR",
    "DEFAULT_GAMMA_MAX",
    "DEFAULT_TAIL_QUANTILE",
    "DEFAULT_T_CAP",
    "FRONTIER",
    "FRONTIER_ACCESS",
    "FRONTIER_BASELINES",
    "HORIZON_ACCESS",
    "HORIZON_BASELINES",
    "MIN_EXCEEDANCES",
    "N4_ACCESS",
    "N4_ENVELOPE",
    "TAIL_BASELINES",
    "TAIL_ENVELOPE",
    "ChecklistReading",
    "ConcomitantBestOfN",
    "ConcomitantReading",
    "Criterion",
    "FrontierReading",
    "GoldVersusKL",
    "HorizonReading",
    "Plateau",
    "Potential",
    "RewardTailIndex",
    "SurrogateChecklist",
    "TailReading",
    "TiltPoint",
    "TurningPoint",
    "Verdict",
    "VisibilityHorizon",
    "concomitant_expectation",
    "cumulant_turning_point",
    "find_plateau",
    "hedgetune",
    "hill",
    "horizon_lambda",
    "light_tailed_envelope",
    "logsumexp",
    "measure_checklist",
    "measure_concomitant",
    "measure_frontier",
    "measure_horizon",
    "measure_tail_index",
    "pickands",
    "simulate_concomitant",
]

"""N2, the visibility horizon: where a tilt extrapolation goes blind, in nats.

Weight degeneracy is the reason this package exists. Importance weights degenerate as lambda grows,
so a tilt estimate has a range over which it is a measurement and a range beyond which it is a
confident number produced by four rollouts. The Kish effective sample size

    ESS(lambda) = (sum w_i)^2 / sum w_i^2

says which is which, and the horizon is the largest lambda at which it still clears a stated floor,
default 0.05 of n. Past that this instrument declines to answer, and that is not caution:
it is the quantity a published lower bound is already written in. The best-of-n
coverage coefficient is

    C = E_{pi*}[pi* / pi_ref] = 1 + chi^2(pi* || pi_ref) = n / ESS

so their controlling constant is this horizon reciprocated, and the reason that literature can
concede the turn "may be impossible to know" is that nobody measures it. Measuring it is cheap:
it costs nothing beyond the n grader calls N1 already makes.

**Why the reading is in nats rather than in lambda.** A horizon is easy to report as
"past lambda = 2.4", and lambda carries the reciprocal of the reward's scale. Two graders
whose scores differ by a factor of ten have horizons differing by a factor of ten in lambda and by
nothing at all in KL, so a horizon in lambda cannot be compared across graders and is not invariant
under the `reward.affine` group the quantity is registered under. The reading is the KL at which ESS
crosses the floor, which is the x-axis of the frontier it bounds. ``lambda_max`` and ``n / ESS`` are
reported alongside, so the lambda is there for anyone who wants it and nothing is lost.

Kill condition, from the catalogue record: if the horizon never binds before the sweep's own range
on any real grader, it is a formality rather than an instrument and it collapses into N1's reporting
range. Non-binding is a real outcome and the reading says so rather than inventing a crossing: a
bounded reward keeps a fixed fraction of its mass at the top under any tilt, so a binary verifier
with a pass rate above the floor never crosses it. On the four open reward models tested here the
horizon binds in every case, between 1.5 and 1.8 nats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BaselineID
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.frontier._base import FrontierInstrument
from reward_lens.measure.frontier.potential import (
    DEFAULT_ESS_FLOOR,
    DEFAULT_T_CAP,
    Potential,
    horizon_lambda,
)

#: Every instrument in this package applies to any grader that returns a number, which is all six
#: substrates. Level 0 reads scores and never reaches inside anything, so the distinction between a
#: scalar head and a program has no purchase here: that is the whole content of the claim that the
#: frontier is estimable "for any substrate, including a closed API".
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

#: The access this layer needs, in full: "GRADER: QUERY, POLICY: QUERY, GOLD: QUERY on the same n
#: samples. Nothing else." POLICY:QUERY is there because the n rollouts have to come from somewhere,
#: and a horizon computed on somebody else's rollouts is a horizon for their base policy.
HORIZON_ACCESS: dict[Component, Access] = {
    Component.GRADER: Access.QUERY,
    Component.POLICY: Access.QUERY,
}

#: The two comparators the catalogue names for N2, as baseline ids. Both are what a frontier
#: reported without a horizon implicitly claims, which is why they are the right things to sit
#: beside the reading rather than in a footnote.
HORIZON_BASELINES: tuple[BaselineID, ...] = (
    "baseline.raw_n",
    "baseline.nonzero_weight_count",
)


def light_tailed_envelope(measured_by: str = "frontier.tail_index") -> EnvelopeSpec:
    """The envelope N1 and N2 share: the moment generating function has to exist.

    A sceptic could argue this instrument needs no envelope, because ESS is a function of the
    realised weights and a finite sum is finite whatever the tail does. The catalogue record makes
    that argument itself, in its `bias` field: ESS is "a property of the sample rather than an
    estimate of a population quantity".

    That is true of the ESS and false of the reading. The reading is the **KL** at which ESS crosses
    the floor, and ``KL(pi_lambda || pi_0) = lambda K'(lambda) - K(lambda)`` is a population object
    that exists only where ``K`` does. On a genuinely heavy-tailed reward the empirical ``K`` is
    still a finite sum and still returns a number, and that number converges to nothing. So the
    condition attaches to the axis the horizon is reported on rather than to the effective sample
    size, and registering the horizon in nats is what brings it in.
    """
    return EnvelopeSpec(
        requires=frozenset({RegimeCondition.LIGHT_TAILED}),
        measured_by={RegimeCondition.LIGHT_TAILED: measured_by},
        on_violation="refuse",
    )


@register_payload
@dataclass
class HorizonReading:
    """The horizon, the ESS curve behind it, and the sentence it produces.

    ``binding`` is the field to read first. False means the floor was never crossed inside the
    searched range, so ``lambda_max`` is the cap rather than a crossing and the horizon is a
    statement about the search rather than about the grader.
    """

    n: int
    floor: float
    lambda_max: float
    t_max: float
    kl_max: float
    ess_at_horizon: float
    coverage_at_horizon: float
    binding: bool
    #: The swept curve, for the plot. `t` is the dimensionless pressure `lambda * sd_0(r)`.
    t_grid: np.ndarray
    lambdas: np.ndarray
    kl: np.ndarray
    ess: np.ndarray
    ess_frac: np.ndarray
    baselines: dict[str, float] = field(default_factory=dict)
    says: str = ""

    def render(self) -> str:
        return self.says


def _horizon_says(reading: HorizonReading) -> str:
    if not reading.binding:
        return (
            f"ESS never falls below {reading.floor:.0%} of n = {reading.n} out to a dimensionless "
            f"pressure of {reading.t_max:.3g}, where KL = {reading.kl_max:.3g} nats and "
            f"n/ESS = {reading.coverage_at_horizon:.3g}. The horizon does not bind on this grader."
        )
    return (
        f"Past KL = {reading.kl_max:.3g} nats (lambda = {reading.lambda_max:.3g}) this instrument "
        f"declines to answer. n/ESS = {reading.coverage_at_horizon:.3g}."
    )


def measure_horizon(
    reward: Sequence[float] | np.ndarray | Potential,
    *,
    floor: float = DEFAULT_ESS_FLOOR,
    t_cap: float = DEFAULT_T_CAP,
    grid: int = 65,
    instrument: str = "VisibilityHorizon",
) -> HorizonReading | Refusal:
    """The horizon and the ESS curve out to it, or the refusal that says why there is neither.

    Callable without a `Context` so it can be used inside a preflight and inside N1, which needs
    the horizon before it can decide how far to sweep.
    """
    pot = reward if isinstance(reward, Potential) else Potential(reward)
    n = pot.n

    if not (0.0 < floor <= 1.0):
        raise ValueError(f"the ESS floor is a fraction of n and must lie in (0, 1]; got {floor}")

    if floor * n < 1.0:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ESS_BELOW_FLOOR,
            detail=(
                f"a floor of {floor:.0%} on n = {n} rollouts is {floor * n:.3g} effective samples, "
                f"which is less than one. Every lambda in the sweep would clear a floor that no "
                f"sample can fail, so the horizon reported would be the search range rather than a "
                f"property of the grader"
            ),
            remedy=(
                f"draw at least {int(np.ceil(1.0 / floor))} rollouts so the floor is worth at least "
                f"one effective sample, or state a floor of at least {1.0 / n:.3g} for this n and "
                f"read the horizon as the point where a single rollout carries the estimate."
            ),
            statistics={"n": n, "floor": floor, "floor_in_samples": floor * n},
        )

    if pot.is_degenerate:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.BELOW_LOD,
            detail=(
                f"the proxy has zero spread over all {n} rollouts (sd = 0, every score is "
                f"{pot.mean:.6g}). The tilt family is then a single point: pi_lambda = pi_0 at "
                f"every lambda, ESS = n everywhere, and KL = 0 everywhere. There is no horizon "
                f"because there is no axis to put one on"
            ),
            remedy=(
                "score a bank the grader actually separates. An all-pass verifier, or a rubric "
                "every rollout in the bank happens to satisfy, gives this: the finding is about "
                "the bank rather than about the grader, and a harder bank fixes it."
            ),
            statistics={"n": n, "sd": pot.sd, "mean": pot.mean},
        )

    lam_max, binding = horizon_lambda(pot, floor=floor, t_cap=t_cap)
    t_max = float(pot.t_of(lam_max))
    ts = np.linspace(0.0, t_max, int(grid))
    lams = np.asarray(pot.lam_of(ts), dtype=np.float64)
    points = pot.sweep(lams)
    at_horizon = points[-1]

    # The naive comparator: how many rollouts still carry a weight that float64 can represent. It
    # agrees with n over almost the whole sweep and collapses only where the exponent underflows,
    # which is exactly why counting it is not a horizon.
    nonzero = int(np.count_nonzero(np.exp(pot.log_weights(lam_max))))

    reading = HorizonReading(
        n=n,
        floor=float(floor),
        lambda_max=float(lam_max),
        t_max=t_max,
        kl_max=float(at_horizon.kl),
        ess_at_horizon=float(at_horizon.ess),
        coverage_at_horizon=float(at_horizon.coverage),
        binding=bool(binding),
        t_grid=ts,
        lambdas=lams,
        kl=np.array([p.kl for p in points]),
        ess=np.array([p.ess for p in points]),
        ess_frac=np.array([p.ess / n for p in points]),
        baselines={
            "baseline.raw_n": float(n),
            "baseline.nonzero_weight_count": float(nonzero),
        },
    )
    reading.says = _horizon_says(reading)
    return reading


class VisibilityHorizon(FrontierInstrument):
    """N2. The largest KL at which the tilt estimate is still carried by enough of the bank.

    Kill condition: if the horizon never binds before the sweep's own range on any real grader, it
    is a formality rather than an instrument and it collapses into N1's reporting range.
    """

    name = "VisibilityHorizon"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "N2"
    deviations = (
        "the reading is the KL at the crossing, not the lambda. A horizon is easy to report as "
        "a lambda; lambda carries the reciprocal of the reward's scale and is not comparable "
        "across graders, so lambda_max is reported alongside rather than as the quantity",
        "the search is over lambda >= 0 only. A negative tilt is a well-defined member of the "
        "family and has its own horizon, and nothing here asks for it: the question the "
        "layer answers is what happens under optimisation pressure, which is the positive axis",
    )

    quantity = "frontier.visibility_horizon"
    requires: dict[Component, Access] = HORIZON_ACCESS
    substrates = ALL_SUBSTRATES
    phases = frozenset({Phase.PRE_RUN})
    envelope = light_tailed_envelope()
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = HORIZON_BASELINES
    rung = 0

    def __init__(
        self,
        reward: Sequence[float] | np.ndarray | Potential | None = None,
        *,
        floor: float = DEFAULT_ESS_FLOOR,
        t_cap: float = DEFAULT_T_CAP,
        grid: int = 65,
    ) -> None:
        self.reward = reward
        self.floor = float(floor)
        self.t_cap = float(t_cap)
        self.grid = int(grid)

    def compute(self) -> Any:
        if self.reward is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no proxy scores were supplied, so there is nothing to tilt",
                remedy=(
                    "pass `reward=` the grader's score on each of n base-policy rollouts. That is "
                    "GRADER:QUERY and POLICY:QUERY and nothing else: no gradients, no activations, "
                    "no training run."
                ),
            )
        return measure_horizon(
            self.reward,
            floor=self.floor,
            t_cap=self.t_cap,
            grid=self.grid,
            instrument=self.name,
        )


__all__ = [
    "ALL_SUBSTRATES",
    "HORIZON_ACCESS",
    "HORIZON_BASELINES",
    "HorizonReading",
    "VisibilityHorizon",
    "light_tailed_envelope",
    "measure_horizon",
]

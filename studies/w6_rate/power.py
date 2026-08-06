"""Power for the W6 rate studies, simulated at the realised n rather than assumed.

Every design here is a nonlinear verdict taken on a bootstrap band or on a weighted extrapolation,
and no closed form covers either. So power is Monte Carlo: plant a system whose answer is known,
run the arms at exactly the sizes `price.py` is costing, push them through the shipped instrument,
and count. That is the same discipline `stats/power.py` applies to the paired binary case, where
every formula in the module is checked against its own simulator.

The planted system is a first-order tracker: an order parameter that relaxes toward an equilibrium
which is a logistic in `lambda` with a fixed critical point. Its relaxation time `tau` is the knob.
At small `tau` the system tracks its equilibrium, both arms trace the same curve against `lambda`,
and the truth is bifurcation-induced. At large `tau` the faster arm lags, its curve is displaced,
and the truth is rate-induced. **The critical point does not move with `tau`**, which is what makes
this a clean planted subject: any separation the instrument reports is lag and nothing else.

What this cannot tell anyone. It is a linear tracker, not a language policy, so the numbers below
size a design and do not predict a result. The one thing it is genuinely evidence for is the
instrument's own operating characteristic: a verdict rule that cannot separate the two planted
truths at the realised n would not separate anything at the real one, and that is worth knowing for
$80 less than it costs to find out on hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reward_lens.measure.rate.collapse import CollapseCriteria, RateArm, two_run_rate_test
from reward_lens.measure.rate.hysteresis import (
    HysteresisCriteria,
    SweepArea,
    rate_extrapolated_area,
)
from reward_lens.measure.rate.transition import TransitionCriteria
from reward_lens.stats.power import TARGET_POWER

#: A cheaper bootstrap than the analysis will use. The band's Monte Carlo error at 60 replicates is
#: larger than at 400, which makes the simulated verdict noisier rather than biased; the effect on a
#: power estimate is to blur the boundary, not to move it. Stated because it is a real difference
#: between what is simulated and what will be run.
POWER_CRITERIA = CollapseCriteria(n_boot=60, n_grid=100)
POWER_FIT = TransitionCriteria(n_boot=30)


@dataclass(frozen=True)
class SimulatedVerdictPower:
    """A verdict rule's operating characteristic at one design, both errors reported.

    `power` is the fraction of simulations under the alternative in which the verdict fired.
    `false_alarm` is the fraction under the null in which it fired anyway. A power number without
    its false-alarm rate is not an operating characteristic, and a rule that fires on everything has
    a power of 1.
    """

    label: str
    power: float
    false_alarm: float
    n_sims: int
    n_refused: int
    detail: str = ""

    @property
    def mc_se(self) -> float:
        """Monte Carlo standard error on the power, at its own estimate."""
        p = self.power
        return float(np.sqrt(max(p * (1.0 - p), 0.0) / max(self.n_sims, 1)))

    @property
    def adequate(self) -> bool:
        return self.power >= TARGET_POWER

    def render(self) -> str:
        verdict = "meets" if self.adequate else "falls short of"
        return (
            f"{self.label}: power {self.power:.2f} +/- {self.mc_se:.2f} against a false-alarm rate "
            f"of {self.false_alarm:.2f} over {self.n_sims} simulations, which {verdict} the "
            f"{TARGET_POWER:.0%} target. {self.n_refused} simulations refused."
            + (f" {self.detail}" if self.detail else "")
        )


# ---------------------------------------------------------------------------
# The planted tracker
# ---------------------------------------------------------------------------


def tracker_arm(
    label: str,
    n_steps: int,
    *,
    tau: float,
    seed: int,
    n_seeds: int = 1,
    noise: float = 0.03,
    lam0: float = 0.05,
    lam1: float = 1.0,
    critical: float = 0.5,
    sharpness: float = 0.05,
) -> RateArm:
    """One arm of a first-order tracker crossing a fixed critical point on an exponential schedule.

    `n_seeds` replicates are averaged before the arm is built, which is what a seed buys in this
    design: the order parameter is the mean over seeds, so its noise falls as the square root and
    the band narrows. The critical point is the same for every `tau` and every rate.
    """
    lam = np.exp(np.linspace(np.log(lam0), np.log(lam1), n_steps))
    acc = np.zeros(n_steps)
    for s in range(n_seeds):
        rng = np.random.default_rng(seed * 1000 + s)
        m = np.zeros(n_steps)
        for i in range(1, n_steps):
            eq = 1.0 / (1.0 + np.exp(-(lam[i] - critical) / sharpness))
            m[i] = m[i - 1] + (eq - m[i - 1]) / tau + rng.normal(0.0, noise)
        acc += m
    return RateArm.from_series(
        label, lam, acc / n_seeds, np.arange(n_steps, dtype=float), series="order_parameter"
    )


def power_two_run(
    *,
    n_slow: int = 200,
    n_fast: int = 50,
    n_seeds: int = 3,
    tau_alt: float = 20.0,
    tau_null: float = 2.0,
    noise: float = 0.03,
    n_sims: int = 40,
    seed: int = 0,
) -> SimulatedVerdictPower:
    """H2's operating characteristic at the realised arm sizes.

    `tau_alt` is the rate-induced truth and `tau_null` the bifurcation-induced one. Both are run
    through the identical instrument at the identical arm sizes, so the two numbers that come back
    are the two errors of one rule rather than two separate experiments.
    """
    fired_alt = fired_null = 0
    refused = 0
    for i in range(n_sims):
        for tau, is_alt in ((tau_alt, True), (tau_null, False)):
            slow = tracker_arm("slow", n_slow, tau=tau, seed=seed + i, n_seeds=n_seeds, noise=noise)
            fast = tracker_arm("fast", n_fast, tau=tau, seed=seed + i, n_seeds=n_seeds, noise=noise)
            out = two_run_rate_test(
                fast, slow, criteria=POWER_CRITERIA, fit_criteria=POWER_FIT, seed=seed + i
            )
            if not hasattr(out, "rate_induced"):
                refused += 1
                continue
            if out.rate_induced:
                if is_alt:
                    fired_alt += 1
                else:
                    fired_null += 1
    return SimulatedVerdictPower(
        label=(f"H2 rate-induced verdict, {n_slow}-step and {n_fast}-step arms at {n_seeds} seeds"),
        power=fired_alt / max(n_sims, 1),
        false_alarm=fired_null / max(n_sims, 1),
        n_sims=n_sims,
        n_refused=refused,
        detail=(
            f"Planted relaxation times {tau_alt:.0f} steps under the alternative and "
            f"{tau_null:.0f} under the null, order-parameter noise {noise:.3g} per step before "
            f"averaging over seeds."
        ),
    )


def power_hysteresis(
    *,
    area_alt: float = 0.10,
    slope: float = 2.0,
    rates: tuple[float, ...] = (0.0188, 0.0377, 0.0755, 0.1538),
    n_seeds: int = 3,
    seed_sd: float = 0.01,
    n_sims: int = 400,
    seed: int = 0,
) -> SimulatedVerdictPower:
    """H3's operating characteristic: does the intercept's interval exclude zero when it should?

    The alternative plants a genuine area of `area_alt` on top of a linear rate dependence; the null
    plants an intercept of exactly zero with the same slope, which is the pure-lag case. Both are
    given the same seed spread, so what is being measured is whether the weighted extrapolation and
    its Birge-widened interval separate a real intercept from a fitted one.
    """
    rng = np.random.default_rng(seed)
    fired_alt = fired_null = 0
    refused = 0
    for _ in range(n_sims):
        for a0, is_alt in ((area_alt, True), (0.0, False)):
            sweeps = []
            for v in rates:
                truth = a0 + slope * v
                draws = truth + rng.normal(0.0, seed_sd, n_seeds)
                sweeps.append(SweepArea.from_seeds(v, draws))
            out = rate_extrapolated_area(sweeps, criteria=HysteresisCriteria())
            if not hasattr(out, "genuine"):
                refused += 1
                continue
            if out.genuine:
                if is_alt:
                    fired_alt += 1
                else:
                    fired_null += 1
    return SimulatedVerdictPower(
        label=(
            f"H3 genuine-hysteresis verdict, {len(rates)} rates at {n_seeds} seeds, "
            f"planted area {area_alt:.3g}"
        ),
        power=fired_alt / max(n_sims, 1),
        false_alarm=fired_null / max(n_sims, 1),
        n_sims=n_sims,
        n_refused=refused,
        detail=(
            f"Seed spread {seed_sd:.3g} at every rate, rate dependence {slope:.3g} per unit rate, "
            f"so the raw area at the fastest rate is {a0 + slope * max(rates):.4g} under both "
            f"truths and the whole of the difference is in the intercept."
        ),
    )


def power_linearity(n_episodes: int) -> float:
    """The best two-sided p-value H1 rung 1's linear-response check can reach at `n` episodes.

    Not a simulation: it is exact. A perfectly monotone ordering of `n` fitted time constants
    against `n` perturbation magnitudes is one of `n!` equally likely permutations under the null, so
    the smallest achievable p-value is `2 / n!`. At three episodes that is 0.333 and the check
    cannot fire at any conventional level; at five it is 0.017 and it can.
    """
    from reward_lens.measure.rate.perturb import _kendall_p_floor

    return _kendall_p_floor(int(n_episodes))


__all__ = [
    "POWER_CRITERIA",
    "POWER_FIT",
    "SimulatedVerdictPower",
    "power_hysteresis",
    "power_linearity",
    "power_two_run",
    "tracker_arm",
]

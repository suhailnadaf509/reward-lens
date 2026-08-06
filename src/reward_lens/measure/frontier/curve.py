"""N1, the reward-versus-gold frontier, out to the horizon and not one nat further.

The claim is that the whole curve of ``E_lambda[g]`` against ``KL(pi_lambda || pi_0)``
exists before any optimisation is run. From n base-policy rollouts scored by both the proxy ``r``
and a gold channel ``g``, with ``w_i = exp(lambda r_i)``, both axes are ratios of weighted means:

    E_lambda[g] = E_0[g e^{lambda r}] / E_0[e^{lambda r}]
    KL(pi_lambda || pi_0) = lambda K'(lambda) - K(lambda)

and ``d/dlambda E_lambda[g] = Cov_lambda(g, r)`` gives the slope in closed form. No gradients, no
policy update, no white-box access. The sweep stops at N2's horizon, because past there the numbers
are produced by a handful of rollouts and this instrument declines to answer.

**What is ours here and what is not.** The turning point is occupied. arXiv 2506.19248 (NeurIPS 2025
Spotlight) defines the identical tilt, names the hacking threshold, proves that the true-reward
curve is either monotone or has a single unique interior extremum for any exponential family with a
strictly monotone statistic, states the stationarity condition ``Cov_lambda(g, r) = 0`` as its
Theorem 3, and ships **HedgeTune**, a bisection and Newton solver that consumes exactly the data
this estimator consumes. So the closed-form cumulant expression

    lambda* = -Cov_0(g, r) / kappa_3(g, r, r)

is a convenience over their root-finder rather than a new result, and the precise sense in which it
is one is visible in `hedgetune` below: it is the first Newton step taken from ``lambda = 0``.
HedgeTune is implemented here and is a mandatory baseline rather than a citation, and both
turning-point estimates are reported side by side on every reading.

What survives, and it is narrower and better, is the horizon (N2), which nobody publishes, and the
concomitant framing and the surrogate conditions (N4).

**The vocabulary.** This module says "the turn in `E_lambda[g]`" and never anything shorter. If `g`
is itself a proxy for something inexpressible, and it always is, then a measured turn is the turn
for `g` and for nothing else. Every reading here is a statement about the gold channel that was
supplied, on the bank that was scored.

Kill condition, from the catalogue record: if HedgeTune and the cumulant estimate agree within their
intervals on every grader tested, report HedgeTune and drop the second estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from reward_lens.core.evidence import Uncertainty, make_evidence, register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.provenance import Provenance
from reward_lens.core.quantity import BaselineID
from reward_lens.core.reading import Refusal, RefusalReason, bounded_refusal
from reward_lens.core.types import (
    Access,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    SubjectRef,
)
from reward_lens.measure.frontier._base import FrontierInstrument
from reward_lens.measure.frontier.horizon import (
    ALL_SUBSTRATES,
    HorizonReading,
    light_tailed_envelope,
    measure_horizon,
)
from reward_lens.measure.frontier.potential import (
    DEFAULT_ESS_FLOOR,
    DEFAULT_T_CAP,
    Potential,
    bootstrap_indices,
    percentile_interval,
)

#: The access this layer needs, in full: "GRADER: QUERY, POLICY: QUERY, GOLD: QUERY on the same n
#: samples. Nothing else."
FRONTIER_ACCESS: dict[Component, Access] = {
    Component.GRADER: Access.QUERY,
    Component.GOLD: Access.QUERY,
    Component.POLICY: Access.QUERY,
}

#: The catalogue's two baselines for N1. The first is mandatory rather than decorative.
FRONTIER_BASELINES: tuple[BaselineID, ...] = (
    "baseline.hedgetune",
    "baseline.gold_at_zero",
)


@register_payload
@dataclass
class TurningPoint:
    """One estimate of where `E_lambda[g]` turns, and how it was obtained.

    ``found`` is False for a curve with no interior stationary point in the visible range, which is
    a common and perfectly good answer: it means gold is still rising where the instrument goes
    blind. Reporting a turn there would be an extrapolation past the horizon wearing a decimal
    point.

    ``is_maximum`` distinguishes a root that is a peak from a root that is a trough. The sign of
    ``d/dlambda Cov_lambda(g, r)`` at the root is the second-derivative test, and a solver that
    reports a trough as the turn has made an error of sign rather than of precision.
    """

    method: str
    found: bool
    lam: float = float("nan")
    kl: float = float("nan")
    gold: float = float("nan")
    is_maximum: bool = False
    converged: bool = False
    iterations: int = 0
    detail: str = ""


def hedgetune(
    pot: Potential,
    *,
    lam_hi: float,
    lam_lo: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> TurningPoint:
    """Bisection then safeguarded Newton on `Cov_lambda(g, r) = 0`, over `[lam_lo, lam_hi]`.

    This is HedgeTune, reimplemented on our own weights so that the comparison in every reading is
    two estimators on one sample rather than two papers on two samples. The function whose root it
    finds is the stationarity condition of the tilt, which is arXiv 2506.19248's Theorem 3:
    ``d/dlambda E_lambda[g] = Cov_lambda(g, r)``.

    Bisection first, because it cannot diverge and the bracket is bounded above by the horizon
    anyway. Newton second, because near the root the derivative
    ``d/dlambda Cov_lambda(g, r) = E_lambda[(g - E_lambda g)(r - E_lambda r)^2]`` is available in
    closed form for the price of one more weighted mean, and a Newton step that leaves the bracket
    is discarded in favour of the bisection step. That safeguard is why this converges on the
    non-convex empirical covariance as well as on the smooth population one.

    A bracket whose endpoints share a sign returns ``found=False`` with the direction named. The
    published theorem says the curve is monotone or has a single interior extremum, so a bracket
    with no sign change means monotone **over that bracket**, and the bracket here stops at the
    horizon rather than at infinity.
    """
    lo, hi = float(lam_lo), float(lam_hi)
    if not hi > lo:
        return TurningPoint(
            method="hedgetune",
            found=False,
            detail=f"empty bracket [{lo:.6g}, {hi:.6g}]: the horizon leaves no range to search",
        )
    f_lo = pot.stationarity(lo)
    f_hi = pot.stationarity(hi)
    if f_lo == 0.0:
        slope = pot.stationarity_slope(lo)
        return TurningPoint(
            method="hedgetune",
            found=True,
            lam=lo,
            kl=pot.kl(lo),
            gold=pot.gold_mean(lo),
            is_maximum=slope < 0.0,
            converged=True,
            iterations=0,
            detail="Cov_0(g, r) is exactly zero: the base policy is already stationary in gold",
        )
    if np.sign(f_lo) == np.sign(f_hi):
        direction = "rising" if f_lo > 0 else "falling"
        return TurningPoint(
            method="hedgetune",
            found=False,
            converged=True,
            detail=(
                f"Cov_lambda(g, r) keeps the same sign across the visible range: "
                f"{f_lo:+.4g} at lambda = {lo:.4g} and {f_hi:+.4g} at lambda = {hi:.4g}. "
                f"E_lambda[g] is {direction} throughout, so there is no turn to report before the "
                f"horizon"
            ),
        )

    iterations = 0
    while hi - lo > tol * max(1.0, hi) and iterations < max_iter:
        iterations += 1
        mid = 0.5 * (lo + hi)
        f_mid = pot.stationarity(mid)
        if f_mid == 0.0:
            lo = hi = mid
            break
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid

    lam = 0.5 * (lo + hi)
    # Newton polish, safeguarded: a step that leaves the bracket is thrown away.
    for _ in range(8):
        f = pot.stationarity(lam)
        slope = pot.stationarity_slope(lam)
        if slope == 0.0 or not np.isfinite(slope):
            break
        step = f / slope
        candidate = lam - step
        if not (lo <= candidate <= hi) or not np.isfinite(candidate):
            break
        iterations += 1
        if abs(candidate - lam) <= tol * max(1.0, abs(lam)):
            lam = candidate
            break
        lam = candidate

    slope = pot.stationarity_slope(lam)
    return TurningPoint(
        method="hedgetune",
        found=True,
        lam=float(lam),
        kl=float(pot.kl(lam)),
        gold=float(pot.gold_mean(lam)),
        is_maximum=bool(slope < 0.0),
        converged=True,
        iterations=iterations,
        detail=(
            f"root of Cov_lambda(g, r) by bisection and Newton in {iterations} steps; "
            f"d/dlambda Cov = {slope:+.4g} there, so this is a "
            f"{'maximum' if slope < 0 else 'minimum'} of E_lambda[g]"
        ),
    )


def cumulant_turning_point(pot: Potential, *, lam_hi: float) -> TurningPoint:
    """`lambda* = -Cov_0(g, r) / kappa_3(g, r, r)`, the closed form, evaluated at zero pressure.

    This is one Newton step on the stationarity condition taken from ``lambda = 0``, and that is
    exactly what makes it a convenience over the root-finder rather than an independent result: the
    numerator is the value and the denominator is the derivative, both at the base policy. It costs
    two weighted means over the unweighted bank and no iteration at all, so it is worth having as
    the number a reader can compute by hand from a covariance and a third cumulant.

    Where it is honest it agrees with `hedgetune` to a few percent. Where the curve has real
    curvature between zero and the turn it does not, and the gap is the price of the linearisation.
    Both are reported and neither is presented as the answer.
    """
    c = pot.stationarity(0.0)
    k3 = pot.stationarity_slope(0.0)
    if k3 == 0.0 or not np.isfinite(k3):
        return TurningPoint(
            method="cumulant",
            found=False,
            detail=(
                "kappa_3(g, r, r) is zero at the base policy, so the linearisation of "
                "Cov_lambda(g, r) has no slope and the closed form has no root"
            ),
        )
    lam = -c / k3
    if not np.isfinite(lam) or lam <= 0.0:
        return TurningPoint(
            method="cumulant",
            found=False,
            lam=float(lam),
            detail=(
                f"the closed form puts the turn at lambda = {lam:.4g}, which is not on the positive "
                f"axis. Cov_0(g, r) = {c:+.4g} and kappa_3(g, r, r) = {k3:+.4g}: optimising the "
                f"proxy from here moves gold monotonically in one direction as far as the "
                f"linearisation can see"
            ),
        )
    if lam > lam_hi:
        return TurningPoint(
            method="cumulant",
            found=False,
            lam=float(lam),
            kl=float("nan"),
            detail=(
                f"the closed form puts the turn at lambda = {lam:.4g}, past the horizon at "
                f"lambda = {lam_hi:.4g}. It is reported as a lambda and not as a KL, because the "
                f"KL there would be an extrapolation past the point where this instrument can see"
            ),
        )
    return TurningPoint(
        method="cumulant",
        found=True,
        lam=float(lam),
        kl=float(pot.kl(lam)),
        gold=float(pot.gold_mean(lam)),
        is_maximum=bool(k3 < 0.0),
        converged=True,
        iterations=1,
        detail=(
            f"Cov_0(g, r) = {c:+.4g}, kappa_3(g, r, r) = {k3:+.4g}; one Newton step from lambda = 0"
        ),
    )


@register_payload
@dataclass
class FrontierReading:
    """The frontier out to the horizon, both turning-point estimates, and the sentence.

    The arrays are the plot. ``kl`` is the x-axis in nats, ``gold`` is ``E_lambda[g]``, and
    ``ess_frac`` is what makes the horizon markable on the same figure rather than in a caption.
    """

    n: int
    floor: float
    #: The swept curve. Every array is the same length and in grid order.
    t_grid: np.ndarray
    lambdas: np.ndarray
    kl: np.ndarray
    gold: np.ndarray
    reward: np.ndarray
    reward_var: np.ndarray
    ess: np.ndarray
    ess_frac: np.ndarray
    stationarity: np.ndarray
    #: The horizon this curve stops at.
    lambda_max: float
    kl_max: float
    coverage_at_horizon: float
    horizon_binding: bool
    #: The turn, twice, from two estimators on the same samples.
    hedgetune: TurningPoint
    cumulant: TurningPoint
    #: The grid's own argmax, which is the third and dumbest reading of the same thing.
    peak_kl: float
    peak_gold: float
    peak_lambda: float
    peak_is_interior: bool
    peak_kl_ci: tuple[float, float] = (float("nan"), float("nan"))
    peak_gold_ci: tuple[float, float] = (float("nan"), float("nan"))
    resamples: int = 0
    baselines: dict[str, float] = field(default_factory=dict)
    says: str = ""

    def render(self) -> str:
        lines = [self.says]
        for tp in (self.hedgetune, self.cumulant):
            if tp.found:
                lines.append(f"    {tp.method:<10} lambda = {tp.lam:.4g}, KL = {tp.kl:.4g} nats")
            else:
                lines.append(f"    {tp.method:<10} no turn: {tp.detail}")
        return "\n".join(lines)


def _frontier_says(reading: FrontierReading) -> str:
    horizon = (
        f"we can see out to {reading.kl_max:.3g}"
        if reading.horizon_binding
        else f"the sweep reaches {reading.kl_max:.3g} without the floor binding"
    )
    if reading.peak_is_interior:
        return (
            f"Gold peaks at KL = {reading.peak_kl:.3g} nats and {horizon} before ESS falls below "
            f"{reading.floor:.0%} of n = {reading.n}."
        )
    return (
        f"E_lambda[g] for this gold channel is still rising at KL = {reading.kl_max:.3g} nats, "
        f"where ESS falls below {reading.floor:.0%} of n = {reading.n}. No turn is visible before "
        f"the horizon, and {horizon}."
    )


def _bootstrap_peak(
    pot: Potential, lams: np.ndarray, resamples: int, seed: int
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Percentile intervals for the grid peak's KL and its gold value.

    The interval is conditional on the lambda grid, which is held at the point estimate's horizon
    rather than re-searched inside each resample. That is a deliberate narrowing and it is worth
    naming: this is an interval on where the turn is *given where we can see*, not one that also
    propagates the uncertainty in where we can see. Widening it to include the horizon's own
    sampling variability would need the horizon re-solved per resample, which costs forty times as
    much for a second-order effect on every case tested here.
    """
    if resamples <= 0 or pot.g is None:
        return (float("nan"), float("nan")), (float("nan"), float("nan"))
    idx = bootstrap_indices(pot.n, resamples, seed)
    kls = np.empty(resamples, dtype=np.float64)
    golds = np.empty(resamples, dtype=np.float64)
    for b in range(resamples):
        sub = Potential(pot.r[idx[b]], pot.g[idx[b]])
        pts = sub.sweep(lams)
        gold = np.array([p.gold_mean for p in pts])
        j = int(np.argmax(gold))
        kls[b] = pts[j].kl
        golds[b] = gold[j]
    return percentile_interval(kls), percentile_interval(golds)


def measure_frontier(
    reward: Sequence[float] | np.ndarray,
    gold: Sequence[float] | np.ndarray,
    *,
    floor: float = DEFAULT_ESS_FLOOR,
    t_cap: float = DEFAULT_T_CAP,
    grid: int = 65,
    lambda_max: float | None = None,
    resamples: int = 200,
    seed: int = 0,
    instrument: str = "GoldVersusKL",
) -> FrontierReading | Refusal:
    """The frontier out to the horizon, or a bounded refusal when more was asked for than is visible.

    With ``lambda_max`` left at None the sweep runs to the horizon and returns Evidence. Pinning
    ``lambda_max`` past the horizon returns a refusal carrying the visible part of the curve as its
    ``partial``, which is the difference between "I cannot tell you" and "I cannot tell you out to
    lambda = 4, and here is the curve out to 2.4 where I can."
    """
    pot = Potential(reward, gold)
    horizon = measure_horizon(pot, floor=floor, t_cap=t_cap, grid=grid, instrument=instrument)
    if isinstance(horizon, Refusal):
        return horizon

    to_horizon = np.linspace(0.0, horizon.t_max, int(grid))
    if lambda_max is None:
        return _frontier_to(pot, horizon, to_horizon, resamples, seed)

    requested = float(lambda_max)
    if requested <= horizon.lambda_max * (1.0 + 1e-9):
        ts = np.linspace(0.0, float(pot.t_of(requested)), int(grid))
        return _frontier_to(pot, horizon, ts, resamples, seed)

    reading = _frontier_to(pot, horizon, to_horizon, resamples, seed)
    at = pot.at(requested)
    return bounded_refusal(
        instrument,
        RefusalReason.ESS_BELOW_FLOOR,
        detail=(
            f"the frontier was requested out to lambda = {requested:.4g}, where the effective "
            f"sample size is {at.ess:.1f} of {pot.n} ({at.ess / pot.n:.2%}) and n/ESS = "
            f"{at.coverage:.4g}. The floor is {floor:.0%} of n and it is crossed at "
            f"lambda = {horizon.lambda_max:.4g}, KL = {horizon.kl_max:.4g} nats. Past that the "
            f"curve is carried by a handful of rollouts, so a value of E_lambda[g] there would be "
            f"a guess wearing an interval"
        ),
        remedy=(
            f"re-run with lambda_max <= {horizon.lambda_max:.4g} and read the bound attached to "
            f"this refusal, which is the frontier out to KL = {horizon.kl_max:.4g} nats. To see "
            f"further, draw more rollouts: the horizon moves roughly as log n, so the next nat "
            f"costs about e times the bank."
        ),
        bound=make_evidence(
            observable="frontier.gold_vs_kl",
            observable_version="1.0",
            subject=SubjectRef(readout="frontier"),
            value=reading,
            uncertainty=Uncertainty(n=pot.n, method="snis-percentile-bootstrap"),
            provenance=Provenance(),
        ),
        requested_lambda=requested,
        lambda_max=horizon.lambda_max,
        kl_max=horizon.kl_max,
        ess_at_request=at.ess,
        coverage_at_request=at.coverage,
        floor=float(floor),
    )


def _frontier_to(
    pot: Potential,
    horizon: HorizonReading,
    ts: np.ndarray,
    resamples: int,
    seed: int,
) -> FrontierReading:
    """Assemble the reading over a dimensionless grid that has already been bounded."""
    lams = np.asarray(pot.lam_of(ts), dtype=np.float64)
    pts = pot.sweep(lams)
    gold = np.array([p.gold_mean for p in pts])
    j = int(np.argmax(gold))
    interior = 0 < j < gold.size - 1

    lam_hi = float(lams[-1])
    tuned = hedgetune(pot, lam_hi=lam_hi)
    closed = cumulant_turning_point(pot, lam_hi=lam_hi)
    kl_ci, gold_ci = _bootstrap_peak(pot, lams, resamples, seed)

    reading = FrontierReading(
        n=pot.n,
        floor=horizon.floor,
        t_grid=ts,
        lambdas=lams,
        kl=np.array([p.kl for p in pts]),
        gold=gold,
        reward=np.array([p.reward_mean for p in pts]),
        reward_var=np.array([p.reward_var for p in pts]),
        ess=np.array([p.ess for p in pts]),
        ess_frac=np.array([p.ess / pot.n for p in pts]),
        stationarity=np.array([p.stationarity for p in pts]),
        lambda_max=horizon.lambda_max,
        kl_max=horizon.kl_max,
        coverage_at_horizon=horizon.coverage_at_horizon,
        horizon_binding=horizon.binding,
        hedgetune=tuned,
        cumulant=closed,
        peak_kl=float(pts[j].kl),
        peak_gold=float(gold[j]),
        peak_lambda=float(lams[j]),
        peak_is_interior=interior,
        peak_kl_ci=kl_ci,
        peak_gold_ci=gold_ci,
        resamples=int(resamples),
        baselines={
            "baseline.hedgetune": tuned.kl if tuned.found else float("nan"),
            "baseline.gold_at_zero": float(gold[0]),
        },
    )
    reading.says = _frontier_says(reading)
    return reading


class GoldVersusKL(FrontierInstrument):
    """N1. `E_lambda[g]` against `KL(pi_lambda || pi_0)`, from n rollouts and no optimisation run.

    Kill condition: if HedgeTune and the cumulant estimate agree within their intervals on every
    grader tested, report HedgeTune and drop the second estimator.
    """

    name = "GoldVersusKL"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "N1"
    deviations = (
        "the sweep is parametrised by the dimensionless pressure t = lambda * sd_0(r) rather than "
        "by lambda. A grid fixed in lambda means something different on every grader and would "
        "have made the reading covariant under reward.affine through the grid rather than through "
        "the estimator",
        "the bootstrap interval on the turn is conditional on the lambda grid, which is held at "
        "the point estimate's horizon rather than re-searched inside each resample",
        "the self-normalised ratio is biased at order 1/ESS and the bias grows exactly where the "
        "horizon is about to refuse. Nothing here corrects it; the horizon is the control for it",
    )

    quantity = "frontier.gold_vs_kl"
    requires: dict[Component, Access] = FRONTIER_ACCESS
    substrates = ALL_SUBSTRATES
    phases = frozenset({Phase.PRE_RUN})
    envelope = light_tailed_envelope()
    invariance = "reward.affine"
    invariance_relation = INVARIANT
    baselines = FRONTIER_BASELINES
    rung = 0

    def __init__(
        self,
        reward: Sequence[float] | np.ndarray | None = None,
        gold: Sequence[float] | np.ndarray | None = None,
        *,
        floor: float = DEFAULT_ESS_FLOOR,
        t_cap: float = DEFAULT_T_CAP,
        grid: int = 65,
        lambda_max: float | None = None,
        resamples: int = 200,
        seed: int = 0,
    ) -> None:
        self.reward = reward
        self.gold = gold
        self.floor = float(floor)
        self.t_cap = float(t_cap)
        self.grid = int(grid)
        self.lambda_max = lambda_max
        self.resamples = int(resamples)
        self.seed = int(seed)

    def compute(self) -> Any:
        if self.reward is None or self.gold is None:
            missing = {}
            if self.reward is None:
                missing["GRADER"] = "QUERY"
            if self.gold is None:
                missing["GOLD"] = "QUERY"
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    f"the frontier is a joint statement about the proxy and a gold channel and "
                    f"needs both on the same n rollouts; missing "
                    f"{', '.join(f'{k}:{v}' for k, v in sorted(missing.items()))}"
                ),
                remedy=(
                    "pass `reward=` and `gold=`, both scored on the same n base-policy rollouts in "
                    "the same order. Two independent samples do not give a joint distribution and "
                    "nothing in this layer is estimable from them. With only the proxy, "
                    "`VisibilityHorizon` and `RewardTailIndex` still run."
                ),
                statistics={"missing": missing},
            )
        return measure_frontier(
            self.reward,
            self.gold,
            floor=self.floor,
            t_cap=self.t_cap,
            grid=self.grid,
            lambda_max=self.lambda_max,
            resamples=self.resamples,
            seed=self.seed,
            instrument=self.name,
        )


__all__ = [
    "FRONTIER_ACCESS",
    "FRONTIER_BASELINES",
    "FrontierReading",
    "GoldVersusKL",
    "TurningPoint",
    "cumulant_turning_point",
    "hedgetune",
    "measure_frontier",
]

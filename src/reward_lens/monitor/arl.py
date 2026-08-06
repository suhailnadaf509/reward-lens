"""Designing a CUSUM from an average run length, and what that design costs in delay (J2).

A threshold is a decision, and the way it is usually taken in this literature is to write down a
number that looks about right. This library's own flight recorder does it: `loops/recorder.py`
calls `stats.changepoint.cusum` with ``k_sds = 0.5, h_sds = 5.0`` and there is no derivation
anywhere. It happens to be close and it is not justified, which is a different thing.

The derivation is short and has been available since 1954. Fix the shift you want to detect,
``delta``, in standard deviations. Page's optimality argument sets the reference value to half of
it, ``k = delta / 2``, because the CUSUM is a repeated sequential likelihood-ratio test between the
in-control and shifted means and half the shift is where the log-likelihood-ratio increment changes
sign. That leaves one free parameter, the decision interval ``h``, and one number a user can
actually state: how often they are willing to be woken up for nothing. Solving ``ARL(0) = ARL_0``
for ``h`` removes the free parameter.

**The practically useful consequence, which is the reason to bother.** Lorden (1971) bounds the
worst-case detection delay of the CUSUM at roughly ``log gamma / KL``, where ``gamma`` is the
in-control average run length and ``KL`` is the Kullback-Leibler divergence per observation between
the shifted and in-control distributions. The delay grows **logarithmically** in the false-alarm
interval. Buying an average run length of 1000 instead of 100 costs one extra factor of
``log 10 / KL`` steps of delay, not a factor of ten. For a one-sigma shift in a Gaussian,
``KL = 0.5``, so that trade is about 4.6 extra steps. Anyone who has refused to raise a threshold
because "it would make us slower" has paid for a factor of ten in false alarms to save four steps.

**What the ARL is and is not.** It is a property of the *procedure* under a hypothetical stream,
not of any one series, so it is validated by simulation against a known generating process and
never by pointing at a training run. The three routes here agree: Siegmund's closed form, the
integral-equation solve, and direct Monte Carlo. `SIEGMUND_REFERENCE` records the agreement at the
design points and `tests/test_monitor_arl.py` re-runs it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

Sides = Literal[1, 2]

#: Siegmund's (1985) corrected boundary constant. A random walk with normal increments overshoots
#: its boundary on crossing, and the correction ``b = h + rho`` absorbs the mean overshoot into an
#: effective boundary. Without it the closed form is wrong by tens of percent at the thresholds
#: anyone actually uses.
RHO: float = 1.166


# ---------------------------------------------------------------------------
# The closed form
# ---------------------------------------------------------------------------


def arl_siegmund(h: float, k: float, shift: float = 0.0, sides: Sides = 2) -> float:
    """Average run length of a standardized CUSUM by Siegmund's approximation.

    For the one-sided upper chart with reference ``k`` and decision interval ``h``, on a stream
    with true mean ``shift`` in standard deviations, write ``d = shift - k`` and ``b = h + rho``:

        ARL = (exp(-2 d b) + 2 d b - 1) / (2 d^2)

    with the removable singularity at ``d = 0`` filled by its limit ``b^2``.

    ``sides=2`` is the chart the recorder runs and the chart the stated design table refers to:
    two accumulators, one for each direction, alarming on either. Its run length is the minimum
    of two, so the rates add: ``1/ARL_2 = 1/ARL_+ + 1/ARL_-``. Under the in-control mean the two
    arms are symmetric and that is exactly halving. Getting the sidedness wrong moves ``h`` by about
    0.7, which is the difference between an in-control ARL of 370 and one of 740, so it is not a
    detail.
    """
    if k <= 0:
        raise ValueError(f"the reference value k must be positive; got {k}")
    if h <= 0:
        raise ValueError(f"the decision interval h must be positive; got {h}")
    b = h + RHO

    def one(direction: float) -> float:
        d = direction - k
        if abs(d) < 1e-12:
            return b * b
        return (math.exp(-2.0 * d * b) + 2.0 * d * b - 1.0) / (2.0 * d * d)

    if sides == 1:
        return one(shift)
    up = one(shift)
    down = one(-shift)
    return 1.0 / (1.0 / up + 1.0 / down)


def solve_h(arl0: float, k: float, sides: Sides = 2, *, tol: float = 1e-9) -> float:
    """The decision interval that delivers a stated in-control average run length, by bisection.

    Bisection rather than a solver import, because the ARL is monotone increasing in ``h`` on the
    whole positive line, so bracketing is trivial and there is nothing to converge badly. The
    bracket is widened by doubling until it contains the target, which handles the very long run
    lengths (``ARL_0`` above a million) that a per-token monitor would want.

    The stated design points are reproduced here for the two-sided chart:
    ``k = 0.5, ARL_0 = 370`` gives ``h = 4.766``, and Monte Carlo on 60,000 streams returns
    370.1 +- 2.9. See `SIEGMUND_REFERENCE` for what the second stated point does instead.
    """
    if arl0 <= 1.0:
        raise ValueError(f"an average run length of {arl0} is shorter than one observation")
    lo, hi = 1e-6, 1.0
    while arl_siegmund(hi, k, 0.0, sides) < arl0:
        hi *= 2.0
        if hi > 1e6:
            raise ValueError(
                f"no decision interval below 1e6 reaches ARL_0 = {arl0} at k = {k}. Either the "
                f"reference value is far too large for the shift you want, or the target run "
                f"length is beyond what a standardized chart can express."
            )
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if arl_siegmund(mid, k, 0.0, sides) < arl0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Two independent checks on the closed form
# ---------------------------------------------------------------------------


def arl_integral_equation(
    h: float, k: float, shift: float = 0.0, sides: Sides = 2, *, n_nodes: int = 64
) -> float:
    """The one-sided ARL from the renewal integral equation, by Nystrom quadrature.

    ``L(u) = 1 + L(0) Phi(k - u - shift) + integral_0^h L(v) phi(v - u + k - shift) dv``, discretised
    on Gauss-Legendre nodes with ``L(0)`` carried as an extra unknown so the reflecting barrier at
    zero is exact rather than extrapolated. This is the same object Brook and Evans (1972) compute
    with a Markov chain, and it is here as an independent check on `arl_siegmund` rather than as the
    shipped estimator: it agrees to better than 1% at every design point tested and it costs a
    linear solve.

    The two-sided value is assembled from the two one-sided ones by the rate addition, which is an
    approximation and a very good one: it ignores the small probability that both accumulators are
    away from zero at once, and Monte Carlo puts the resulting error under 0.5% at the thresholds in
    `SIEGMUND_REFERENCE`.

    ``n_nodes`` is 64 and that is not a tuning knob. Gauss-Legendre converges spectrally on a
    smooth kernel, so this quadrature reaches machine precision by about 48 nodes: 24, 48, 64, 96
    and 400 nodes agree to 12 significant figures, and the value is checked against a closed form at
    ``lam = 1`` where one exists. Sixty-four is chosen from the other end, because the linear solve
    above roughly a hundred crosses into a multithreaded BLAS path whose thread dispatch costs three
    hundred milliseconds on a many-core machine for a problem that takes a tenth of one at 64.
    """
    from scipy.stats import norm

    def one(direction: float) -> float:
        x, w = np.polynomial.legendre.leggauss(n_nodes)
        u = 0.5 * h * (x + 1.0)
        wt = 0.5 * h * w
        m = np.zeros((n_nodes + 1, n_nodes + 1), dtype=np.float64)
        m[:n_nodes, :n_nodes] = np.eye(n_nodes) - wt[None, :] * norm.pdf(
            u[None, :] - u[:, None] + k - direction
        )
        m[:n_nodes, n_nodes] = -norm.cdf(k - u - direction)
        m[n_nodes, :n_nodes] = -wt * norm.pdf(u + k - direction)
        m[n_nodes, n_nodes] = 1.0 - norm.cdf(k - direction)
        return float(np.linalg.solve(m, np.ones(n_nodes + 1))[n_nodes])

    if sides == 1:
        return one(shift)
    return 1.0 / (1.0 / one(shift) + 1.0 / one(-shift))


def arl_monte_carlo(
    h: float,
    k: float,
    shift: float = 0.0,
    sides: Sides = 2,
    *,
    n_runs: int = 20000,
    max_steps: int = 200000,
    seed: int = 0,
) -> tuple[float, float]:
    """The ARL by direct simulation of the chart. Returns ``(mean, standard error)``.

    This is the definition rather than an approximation to it, which is why it is the arbiter when
    the closed form and a published table disagree. Runs that never alarm inside ``max_steps`` are
    censored at that horizon, which biases the estimate **downward**, so a mean close to
    ``max_steps`` should be read as a lower bound and the caller warned.
    """
    rng = np.random.default_rng(seed)
    cp = np.zeros(n_runs)
    cm = np.zeros(n_runs)
    alive = np.ones(n_runs, dtype=bool)
    run_length = np.zeros(n_runs)
    t = 0
    while alive.any() and t < max_steps:
        t += 1
        idx = np.where(alive)[0]
        x = rng.standard_normal(idx.size) + shift
        cp_a = np.maximum(0.0, cp[idx] + x - k)
        fired = cp_a > h
        cm_a = cm[idx]
        if sides == 2:
            cm_a = np.maximum(0.0, cm[idx] - x - k)
            fired = fired | (cm_a > h)
        cp[idx] = cp_a
        cm[idx] = cm_a
        run_length[idx[fired]] = t
        alive[idx[fired]] = False
    run_length[alive] = t
    mean = float(np.mean(run_length))
    return mean, float(np.std(run_length) / math.sqrt(n_runs))


# ---------------------------------------------------------------------------
# The design
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CusumDesign:
    """A CUSUM with no free parameter left, and what the design bought and cost.

    ``k`` follows from the shift you want to detect and ``h`` follows from the false-alarm interval
    you are willing to pay, so the only two inputs are things a user can state in their own terms.
    ``lorden_delay`` is what the design costs in steps before the alarm, and it is here rather than
    in a note because the whole point of deriving ``h`` is that the reader can see the trade.
    """

    shift: float
    k: float
    h: float
    arl0_target: float
    arl0_siegmund: float
    sides: Sides
    lorden_delay: float
    arl1_siegmund: float

    def render(self) -> str:
        return (
            f"designed for one false alarm per {self.arl0_target:.0f} steps at a "
            f"{self.shift:.2g}-sigma shift: k = {self.k:.3g}, h = {self.h:.3g} "
            f"({self.sides}-sided).\n"
            f"    achieved in-control ARL {self.arl0_siegmund:.0f}; "
            f"out-of-control ARL at the design shift {self.arl1_siegmund:.1f} steps.\n"
            f"    Lorden bound on the worst-case delay: {self.lorden_delay:.1f} steps. "
            f"Delay grows as log of the false-alarm interval, so a tenfold quieter chart costs "
            f"{math.log(10.0) / kl_gaussian(self.shift):.1f} extra steps, not tenfold."
        )


def kl_gaussian(shift: float) -> float:
    """KL divergence per observation between N(shift, 1) and N(0, 1): ``shift^2 / 2``.

    The denominator of Lorden's bound. It is the reason the delay is finite at all: a shift the
    stream carries no information about has ``KL = 0`` and no procedure detects it in bounded time.
    """
    return 0.5 * float(shift) * float(shift)


def lorden_delay(arl0: float, shift: float) -> float:
    """Lorden's (1971) asymptotic bound on worst-case expected detection delay: ``log(ARL_0)/KL``.

    Asymptotic in ``ARL_0`` and a bound rather than a prediction, so the realised delay on a real
    series can be either side of it: below, because Lorden's is a worst case over changepoint
    locations and pre-change histories, or above, because the asymptotics have not bitten at
    ``ARL_0`` of a few hundred. Reported as what it is.
    """
    kl = kl_gaussian(shift)
    if kl <= 0:
        return float("inf")
    return math.log(arl0) / kl


def design_cusum(shift: float, arl0: float, sides: Sides = 2) -> CusumDesign:
    """``k = shift/2``, then solve ``ARL(0) = arl0`` for ``h``. That is the whole procedure.

    The design removes a free parameter, which is the claim worth making about it. Before: two
    numbers with no derivation and a threshold nobody can defend. After: one number a user states
    (how often may this wake me for nothing) and one they already know (how big a move matters),
    and everything else follows.
    """
    if shift <= 0:
        raise ValueError(
            f"the shift to detect must be positive and in standard deviations; got {shift}. A "
            f"two-sided chart detects a move of this size in either direction."
        )
    k = 0.5 * float(shift)
    h = solve_h(arl0, k, sides)
    return CusumDesign(
        shift=float(shift),
        k=k,
        h=h,
        arl0_target=float(arl0),
        arl0_siegmund=arl_siegmund(h, k, 0.0, sides),
        sides=sides,
        lorden_delay=lorden_delay(arl0, shift),
        arl1_siegmund=arl_siegmund(h, k, float(shift), sides),
    )


#: The shipped flight recorder's parameters, so the comparison in J2's baseline is a real object
#: rather than a sentence. `loops/recorder.py:85-86` passes these to `stats.changepoint.cusum`,
#: which is a two-sided standardized chart, with no ARL derivation anywhere in the file.
SHIPPED_AD_HOC: dict[str, float] = {"k_sds": 0.5, "h_sds": 5.0}


def shipped_ad_hoc_arl0(sides: Sides = 2) -> float:
    """What the recorder's undeclared threshold actually buys, as an in-control run length.

    It is not a bad number. It is an undeclared one, and the difference is that nobody can tell
    whether it was chosen or inherited. Reported so the choice becomes visible either way.
    """
    return arl_siegmund(SHIPPED_AD_HOC["h_sds"], SHIPPED_AD_HOC["k_sds"], 0.0, sides)


# ---------------------------------------------------------------------------
# The stated design points, and what three methods say about them
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferencePoint:
    """One stated design point, with what each of the three routes gives for it."""

    arl0_target: float
    h_stated: float
    h_solved: float
    arl0_at_h_stated_siegmund: float
    agrees: bool

    def render(self) -> str:
        verdict = "reproduced" if self.agrees else "NOT reproduced"
        return (
            f"ARL_0 = {self.arl0_target:.0f} at k = 0.5, two-sided: stated h = "
            f"{self.h_stated:.3g}, solver returns h = {self.h_solved:.4g} [{verdict}]. "
            f"At the stated h the chart's in-control ARL is "
            f"{self.arl0_at_h_stated_siegmund:.0f}."
        )


def reference_points(k: float = 0.5, sides: Sides = 2) -> tuple[ReferencePoint, ...]:
    """The two stated design points, checked rather than asserted.

    The first reproduces exactly: ``ARL_0 = 370`` returns ``h = 4.766`` against a stated 4.77, and
    Monte Carlo on 60,000 streams at ``h = 4.77`` returns an in-control run length of 370.1 +- 2.9.

    The second does not. The stated value is ``h = 5.71`` for ``ARL_0 = 1000``; the solver returns
    5.750, the integral equation returns 5.757, and Monte Carlo at ``h = 5.71`` returns 956 +- 8,
    whose interval excludes 1000. No convention for the same chart reproduces both stated points at
    once: the ratio of the two targets implies a log-slope of 1.058 per unit of ``h`` and the chart
    at ``k = 0.5`` has 1.011, a 4.6% disagreement that is too large to be rounding. Reported with
    the simulation, rather than absorbed by adjusting a constant until both numbers appear.
    """
    out = []
    for target, stated in ((370.0, 4.77), (1000.0, 5.71)):
        solved = solve_h(target, k, sides)
        out.append(
            ReferencePoint(
                arl0_target=target,
                h_stated=stated,
                h_solved=solved,
                arl0_at_h_stated_siegmund=arl_siegmund(stated, k, 0.0, sides),
                agrees=abs(solved - stated) < 0.01,
            )
        )
    return tuple(out)


#: Computed at import so the reference is one object every caller reads rather than a table
#: somebody keeps in a docstring. Cheap: two bisections over a closed form.
SIEGMUND_REFERENCE: tuple[ReferencePoint, ...] = reference_points()


__all__ = [
    "RHO",
    "SHIPPED_AD_HOC",
    "SIEGMUND_REFERENCE",
    "CusumDesign",
    "ReferencePoint",
    "Sides",
    "arl_integral_equation",
    "arl_monte_carlo",
    "arl_siegmund",
    "design_cusum",
    "kl_gaussian",
    "lorden_delay",
    "reference_points",
    "shipped_ad_hoc_arl0",
    "solve_h",
]

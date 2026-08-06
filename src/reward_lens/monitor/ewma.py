"""The EWMA chart, designed to the same average run length as the CUSUM (J2 rung 1).

An exponentially weighted moving average ``Z_t = lam x_t + (1 - lam) Z_{t-1}`` alarms when ``|Z_t|``
leaves ``+- L sigma_Z``, with the asymptotic ``sigma_Z = sigma sqrt(lam / (2 - lam))``. It is the
rung above the CUSUM in J2's ladder for one reason: the CUSUM is optimal for the shift it was tuned
to and drops off either side of it, while the EWMA at a small ``lam`` is competitive over a range of
shifts. Which matters here, because nobody designing a reward-hacking monitor knows the size of the
shift they are looking for.

**Designed to a stated ARL, not to a convention.** ``L = 3`` is the number everybody writes down and
it is a Shewhart reflex rather than a derivation: at ``lam = 1`` the EWMA *is* a Shewhart chart, and
``L = 3`` there gives an in-control run length of 370.4, which is where the convention comes from.
At ``lam = 0.1`` the same ``L = 3`` gives a run length of 842, more than twice as quiet as the
reader thinks, because the smoothing correlates successive statistics. `design_ewma` solves for
``L`` at the ``lam`` you are actually using.

Solved for ARL_0 = 370 it returns L = 2.7011 at lam = 0.1 and 2.8977 at lam = 0.25, against the
2.703 and 2.898 of Lucas and Saccucci (1990), which is agreement to the last digit they print.

The ARL comes from the integral equation, solved on Gauss-Legendre nodes. At ``lam = 1`` it must
reproduce the Shewhart identity ``ARL_0 = 1 / (2 Phi(-L))`` exactly, and
`tests/test_monitor_arl.py` asserts that, which is the strongest available check on the quadrature:
the limit has a closed form and the code does not know it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from reward_lens.monitor.arl import Sides


def sigma_z(lam: float, sigma: float = 1.0, *, asymptotic: bool = True, t: int = 0) -> float:
    """The standard deviation of the EWMA statistic.

    The asymptotic value ``sigma sqrt(lam / (2 - lam))`` is the steady state. The exact
    time-varying value carries a factor ``1 - (1 - lam)^(2t)``, which matters only in the first few
    steps and is what makes an EWMA chart with fixed limits insensitive early. Offered because a
    monitor that starts at step zero of a training run spends those steps.
    """
    base = sigma * math.sqrt(lam / (2.0 - lam))
    if asymptotic or t <= 0:
        return base
    return base * math.sqrt(1.0 - (1.0 - lam) ** (2 * t))


def arl_ewma(lam: float, control_limit: float, shift: float = 0.0, *, n_nodes: int = 64) -> float:
    """In-control or out-of-control ARL of a two-sided EWMA, from the integral equation.

    ``L(z) = 1 + (1/lam) integral_{-h}^{h} L(y) phi((y - (1 - lam) z) / lam - shift) dy`` with
    ``h = control_limit`` in the units of the raw observation, discretised by Nystrom. The chart is
    started at zero, so the reported run length is ``L(0)``.

    ``control_limit`` is an absolute limit on ``Z``, not a multiple of ``sigma_Z``. `design_ewma`
    converts. Passing the multiple by mistake gives an enormous ARL rather than an error, which is
    why the argument is named for what it is.

    ``n_nodes`` is 64 and that is not a tuning knob. Gauss-Legendre converges spectrally on a
    smooth kernel, so this quadrature reaches machine precision by about 48 nodes: 24, 48, 64, 96
    and 400 nodes agree to 12 significant figures, and the value is checked against a closed form at
    ``lam = 1`` where one exists. Sixty-four is chosen from the other end, because the linear solve
    above roughly a hundred crosses into a multithreaded BLAS path whose thread dispatch costs three
    hundred milliseconds on a many-core machine for a problem that takes a tenth of one at 64.
    """
    from scipy.stats import norm

    h = float(control_limit)
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    z = h * x
    wt = h * w
    # Built by broadcasting rather than row by row. The loop form called `norm.pdf` once per node
    # and the bisection in `design_ewma` calls this twenty times, which turned a design into thirty
    # seconds of scipy dispatch.
    arg = (z[None, :] - (1.0 - lam) * z[:, None]) / lam - shift / lam
    kernel = (wt[None, :] / lam) * norm.pdf(arg)
    solution = np.linalg.solve(np.eye(n_nodes) - kernel, np.ones(n_nodes))
    # L(0) by the same equation evaluated at z = 0, using the solved L on the nodes.
    row = (wt / lam) * norm.pdf(z / lam - shift / lam)
    return float(1.0 + row @ solution)


@dataclass(frozen=True)
class EwmaDesign:
    """An EWMA with its smoothing constant stated and its limit derived."""

    lam: float
    multiplier: float
    control_limit: float
    arl0_target: float
    arl0_achieved: float
    sides: Sides = 2

    def render(self) -> str:
        return (
            f"EWMA at lam = {self.lam:.3g}: L = {self.multiplier:.4g}, control limit "
            f"{self.control_limit:.4g} in observation units, achieved in-control ARL "
            f"{self.arl0_achieved:.0f} against a target of {self.arl0_target:.0f}."
        )


def design_ewma(
    arl0: float, lam: float = 0.2, *, sigma: float = 1.0, tol: float = 1e-4, n_nodes: int = 64
) -> EwmaDesign:
    """Solve for the control-limit multiplier that delivers a stated in-control ARL at this ``lam``.

    Bisection on ``L``, because the ARL is monotone increasing in the limit. The bracket starts at
    ``[0.5, 5]``, which spans every design anybody uses, and widens upward if it has to.
    """
    if not 0.0 < lam <= 1.0:
        raise ValueError(f"lam must lie in (0, 1]; got {lam}")
    sz = sigma_z(lam, sigma)

    def arl_of(mult: float) -> float:
        return arl_ewma(lam, mult * sz, 0.0, n_nodes=n_nodes)

    lo, hi = 0.5, 5.0
    while arl_of(hi) < arl0:
        hi *= 1.5
        if hi > 50:
            raise ValueError(f"no multiplier below 50 reaches ARL_0 = {arl0} at lam = {lam}")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if arl_of(mid) < arl0:
            lo = mid
        else:
            hi = mid
    mult = 0.5 * (lo + hi)
    return EwmaDesign(
        lam=float(lam),
        multiplier=mult,
        control_limit=mult * sz,
        arl0_target=float(arl0),
        arl0_achieved=arl_of(mult),
    )


def ewma_alarm(
    z_series: np.ndarray, design: EwmaDesign, *, exact_limits: bool = True
) -> int | None:
    """Run the chart over a standardized series and return the first alarm index, or None.

    ``exact_limits`` widens the limit in the first steps by the exact time-varying ``sigma_Z``,
    which is what stops a chart from being blind for its first ten observations at a small ``lam``.
    Turning it off reproduces the fixed-limit chart most references print.
    """
    z = np.asarray(z_series, dtype=np.float64).ravel()
    stat = 0.0
    for i, xi in enumerate(z):
        if not math.isfinite(xi):
            continue
        stat = design.lam * float(xi) + (1.0 - design.lam) * stat
        limit = (
            design.multiplier * sigma_z(design.lam, 1.0, asymptotic=False, t=i + 1)
            if exact_limits
            else design.control_limit
        )
        if abs(stat) > limit:
            return i
    return None


__all__ = ["EwmaDesign", "arl_ewma", "design_ewma", "ewma_alarm", "sigma_z"]

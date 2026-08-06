# Vendored, unmodified below this header, from
#     https://github.com/AsiaeeLab/certified-interventional-fidelity
#     path   code/cif.py
#     branch main (the only branch; `master` resolves to the same blob)
#     as of  2026-07-09T10:19:17Z, fetched 2026-08-01
#     461 lines, 14047 bytes, sha256
#     4c9824c9d46b681dfca7688c42080a5f8dd701aad0bd9933105ea68325812a51
#
# Vendored rather than depended on. `cif.py` is a module inside a research repository, not a
# package: there is no pyproject.toml, no release, and no name on any index, so there is nothing to
# put in a dependency list. The alternative in this lane, `confseq`, cannot be installed here at
# all: no cp312 wheel exists, pip falls back to the sdist, and CMake fails on
# Boost_INCLUDE_DIR-NOTFOUND. Vendoring is the only option rather than the preference.
#
# Nothing below is modified. Byte-for-byte the upstream file, so the sha256 above is checkable
# against the vendored body with the header stripped, and `tests/test_monitor_sequential.py` does
# exactly that. Wrapping it instead of editing it is what keeps that check meaningful: everything
# this library needs on top of `cif.py` lives in `reward_lens.stats.sequential` and
# `reward_lens.monitor.eprocess`, never here.
#
# ---------------------------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2026 Amir Asiaee, AsiaeeLab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ---------------------------------------------------------------------------------------------
# --- begin vendored code/cif.py, sha256 4c9824c9d46b681dfca7688c42080a5f8dd701aad0bd9933105ea68325812a51
"""Core CIF: confidence sequences, adaptive IS, certification."""

from __future__ import annotations

import math


def spending_delta(n: int, delta: float) -> float:
    """Spending schedule: delta_n = 6*delta / (pi^2 * n^2)."""
    return 6.0 * delta / (math.pi**2 * n**2)


def cs_radius(n: int, delta: float) -> float:
    """Anytime-valid CS radius for Z in [0,1], i.i.d. sampling."""
    delta_n = spending_delta(n, delta)
    return math.sqrt(math.log(2.0 / delta_n) / (2.0 * n))


def cs_radius_is(n: int, delta: float, W_max: float) -> float:
    """Anytime-valid CS radius under importance sampling with weight bound W_max."""
    delta_n = spending_delta(n, delta)
    return W_max * math.sqrt(math.log(2.0 / delta_n) / (2.0 * n))


def cs_radius_paired(n: int, delta: float) -> float:
    """Anytime-valid CS radius for paired comparison (Delta in [-1,1], range=2)."""
    delta_n = spending_delta(n, delta)
    return 2.0 * math.sqrt(math.log(2.0 / delta_n) / (2.0 * n))


def hoeffding_fixed_radius(n: int, delta: float) -> float:
    """Fixed-budget Hoeffding CI radius for Z in [0,1]."""
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def samples_for_width(epsilon: float, delta: float) -> int:
    """Minimum n to achieve half-width epsilon at confidence 1-delta (fixed budget)."""
    return int(math.ceil(math.log(2.0 / delta) / (2.0 * epsilon**2)))


class CIFTracker:
    """Maintains running CIF statistics for one estimand.

    Supports both i.i.d. and importance-weighted observations.

    Usage:
        tracker = CIFTracker(delta=0.05, W_max=1.0)
        for t in range(1, n+1):
            tracker.update(z_t, w_t=1.0)
            r_hat, b, lcb_f = tracker.r_hat, tracker.radius, tracker.lcb_fidelity
    """

    def __init__(self, delta: float = 0.05, W_max: float = 1.0):
        self.delta = float(delta)
        self.W_max = float(W_max)
        self.n = 0
        self._sum_wz = 0.0
        self.history_wz: list[float] = []  # store w_t * Z_t values for full trace if needed

    def update(self, z: float, w: float = 1.0) -> None:
        self.n += 1
        wz = float(w) * float(z)
        self._sum_wz += wz
        self.history_wz.append(wz)

    @property
    def r_hat(self) -> float:
        if self.n == 0:
            return 0.0
        return self._sum_wz / self.n

    @property
    def radius(self) -> float:
        if self.n == 0:
            return float("inf")
        if self.W_max == 1.0:
            return cs_radius(self.n, self.delta)
        return cs_radius_is(self.n, self.delta, self.W_max)

    @property
    def lcb_fidelity(self) -> float:
        """Lower confidence bound on F = 1 - R."""
        return 1.0 - (self.r_hat + self.radius)

    @property
    def ucb_fidelity(self) -> float:
        """Upper confidence bound on F = 1 - R."""
        return 1.0 - max(0.0, self.r_hat - self.radius)

    @property
    def cs_interval(self) -> tuple[float, float]:
        """(lower, upper) for R."""
        return (
            max(0.0, self.r_hat - self.radius),
            min(1.0, self.r_hat + self.radius),
        )

    def is_certified(self, F_0: float) -> bool:
        """True if LCB(F) >= F_0."""
        return self.lcb_fidelity >= float(F_0)

    def trace(self, step_size: int = 1) -> dict[str, list[float]]:
        """Return full trace of R_hat, radius, LCB(F) at each step."""
        ns: list[float] = []
        r_hats: list[float] = []
        radii: list[float] = []
        lcbs: list[float] = []
        cumsum = 0.0
        for t, wz in enumerate(self.history_wz, 1):
            cumsum += wz
            if t % step_size == 0 or t == len(self.history_wz):
                r_hat_t = cumsum / t
                if self.W_max == 1.0:
                    b_t = cs_radius(t, self.delta)
                else:
                    b_t = cs_radius_is(t, self.delta, self.W_max)
                ns.append(float(t))
                r_hats.append(r_hat_t)
                radii.append(b_t)
                lcbs.append(1.0 - (r_hat_t + b_t))
        return {"n": ns, "r_hat": r_hats, "radius": radii, "lcb_f": lcbs}


class PairedCIFTracker:
    """Maintains running paired-comparison CIF statistics.

    Delta_t = Z_t^(1) - Z_t^(2) in [-1, 1].
    Estimates mu_Delta = R^(1) - R^(2).
    """

    def __init__(self, delta: float = 0.05):
        self.delta = float(delta)
        self.n = 0
        self._sum_delta = 0.0
        self.history_delta: list[float] = []

    def update(self, z1: float, z2: float) -> None:
        self.n += 1
        d = float(z1) - float(z2)
        self._sum_delta += d
        self.history_delta.append(d)

    @property
    def mu_hat(self) -> float:
        if self.n == 0:
            return 0.0
        return self._sum_delta / self.n

    @property
    def radius(self) -> float:
        if self.n == 0:
            return float("inf")
        return cs_radius_paired(self.n, self.delta)

    @property
    def significant(self) -> bool:
        """True if CS for mu_Delta excludes 0."""
        low = self.mu_hat - self.radius
        high = self.mu_hat + self.radius
        return low > 0 or high < 0


class BettingCSTracker:
    """Betting-based confidence sequence using ONS strategy.

    Maintains the capital process to construct an anytime-valid confidence interval.

    For importance-weighted observations (W_max > 1), observations are rescaled to [0, 1]
    and the final CS is scaled back.
    """

    def __init__(self, delta: float = 0.05, W_max: float = 1.0, grid_size: int = 200):
        self.delta = float(delta)
        self.W_max = float(W_max)
        self.grid_size = int(grid_size)
        self.n = 0
        self._observations: list[float] = []  # store rescaled observations Y_t in [0,1]

    def update(self, z: float, w: float = 1.0) -> None:
        """Add observation z ∈ [0,1] with importance weight w."""
        self.n += 1
        wz = float(w) * float(z)
        wz = max(0.0, min(wz, self.W_max))
        y = wz / self.W_max if self.W_max > 0 else 0.0
        self._observations.append(y)

    @staticmethod
    def _ons_log_capital(
        observations: list[float],
        m: float,
        *,
        c: float = 0.5,
    ) -> float:
        """Compute log(K_n(m)) using ONS betting strategy on observations in [0,1]."""
        if not observations:
            return 0.0

        m = max(0.001, min(float(m), 0.999))
        eta = 2.0 / (2.0 - math.log(3.0))

        lam = 0.0  # λ_1 = 0
        A = 1.0  # A_0 = 1
        logK = 0.0

        for x in observations:
            x = max(0.0, min(float(x), 1.0))
            diff = x - m
            factor = 1.0 + lam * diff
            if factor <= 0.0:
                return float("-inf")
            logK += math.log(factor)

            g = diff / factor
            A += g * g
            lam += eta * g / A
            if lam > c:
                lam = c
            elif lam < -c:
                lam = -c

        return logK

    @staticmethod
    def _ons_max_log_capital(
        observations: list[float],
        m: float,
        *,
        check_every: int,
        c: float = 0.5,
    ) -> float:
        """Max log(K_t(m)) over t in {check_every, 2*check_every, ..., n}."""
        if not observations:
            return 0.0
        if check_every <= 0:
            raise ValueError("check_every must be >= 1")

        m = max(0.001, min(float(m), 0.999))
        eta = 2.0 / (2.0 - math.log(3.0))

        lam = 0.0
        A = 1.0
        logK = 0.0
        max_logK = float("-inf")

        for t, x in enumerate(observations, 1):
            x = max(0.0, min(float(x), 1.0))
            diff = x - m
            factor = 1.0 + lam * diff
            if factor <= 0.0:
                return float("inf")
            logK += math.log(factor)

            if t % check_every == 0 or t == len(observations):
                if logK > max_logK:
                    max_logK = logK

            g = diff / factor
            A += g * g
            lam += eta * g / A
            if lam > c:
                lam = c
            elif lam < -c:
                lam = -c

        return max_logK

    def _log_capital(self, m: float) -> float:
        return self._ons_log_capital(self._observations, m, c=0.5)

    def _find_boundary(self, x_bar: float, side: str) -> float:
        """Binary search for CS boundary. side='lower' or 'upper'."""
        tol = 1e-5
        max_iter = 50
        log_thresh = math.log(1.0 / self.delta)

        x_bar = max(0.0, min(float(x_bar), 1.0))

        if side == "lower":
            lo = 0.0
            hi = x_bar
            if hi <= lo + 1e-12:
                return 0.0
            if self._log_capital(lo) < log_thresh:
                return 0.0
            if self._log_capital(hi) >= log_thresh:
                return hi
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                if self._log_capital(mid) >= log_thresh:
                    lo = mid
                else:
                    hi = mid
                if hi - lo <= tol:
                    break
            return lo

        if side == "upper":
            lo = x_bar
            hi = 1.0
            if hi <= lo + 1e-12:
                return 1.0
            if self._log_capital(hi) < log_thresh:
                return 1.0
            if self._log_capital(lo) >= log_thresh:
                return lo
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                if self._log_capital(mid) >= log_thresh:
                    hi = mid
                else:
                    lo = mid
                if hi - lo <= tol:
                    break
            return hi

        raise ValueError("side must be 'lower' or 'upper'")

    @property
    def r_hat(self) -> float:
        if self.n == 0:
            return 0.0
        return self.W_max * (sum(self._observations) / float(self.n))

    @property
    def cs_interval(self) -> tuple[float, float]:
        """Return (lower, upper) confidence interval for R."""
        if self.n < 2:
            return (0.0, 1.0)

        y_bar = sum(self._observations) / float(self.n)
        lo_y = self._find_boundary(y_bar, side="lower")
        hi_y = self._find_boundary(y_bar, side="upper")

        lo = self.W_max * lo_y
        hi = self.W_max * hi_y
        lo = max(0.0, min(float(lo), 1.0))
        hi = max(0.0, min(float(hi), 1.0))
        if hi < lo:
            lo, hi = hi, lo
        return (lo, hi)

    @property
    def radius(self) -> float:
        lo, hi = self.cs_interval
        return (hi - lo) / 2.0

    @property
    def lcb_fidelity(self) -> float:
        _, hi_R = self.cs_interval
        return 1.0 - hi_R

    @property
    def ucb_fidelity(self) -> float:
        lo_R, _ = self.cs_interval
        return 1.0 - lo_R

    def is_certified(self, F_0: float) -> bool:
        return self.lcb_fidelity >= float(F_0)


class PairedBettingCSTracker:
    """Betting CS for paired differences.

    Delta_t = Z_t^(1) - Z_t^(2) in [-1, 1]. Map to Y_t = (Delta_t + 1) / 2 in [0, 1],
    run betting CS on Y_t, then map back.
    """

    def __init__(self, delta: float = 0.05, grid_size: int = 200):
        self.delta = float(delta)
        self.grid_size = int(grid_size)
        self.n = 0
        self._observations: list[float] = []  # Y_t in [0,1]

    def update(self, z1: float, z2: float) -> None:
        self.n += 1
        d = float(z1) - float(z2)
        d = max(-1.0, min(d, 1.0))
        y = 0.5 * (d + 1.0)
        self._observations.append(y)

    def _log_capital(self, m: float) -> float:
        return BettingCSTracker._ons_log_capital(self._observations, m, c=0.5)

    def _find_boundary(self, y_bar: float, side: str) -> float:
        tol = 1e-5
        max_iter = 50
        log_thresh = math.log(1.0 / self.delta)

        y_bar = max(0.0, min(float(y_bar), 1.0))

        if side == "lower":
            lo = 0.0
            hi = y_bar
            if hi <= lo + 1e-12:
                return 0.0
            if self._log_capital(lo) < log_thresh:
                return 0.0
            if self._log_capital(hi) >= log_thresh:
                return hi
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                if self._log_capital(mid) >= log_thresh:
                    lo = mid
                else:
                    hi = mid
                if hi - lo <= tol:
                    break
            return lo

        if side == "upper":
            lo = y_bar
            hi = 1.0
            if hi <= lo + 1e-12:
                return 1.0
            if self._log_capital(hi) < log_thresh:
                return 1.0
            if self._log_capital(lo) >= log_thresh:
                return lo
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                if self._log_capital(mid) >= log_thresh:
                    hi = mid
                else:
                    lo = mid
                if hi - lo <= tol:
                    break
            return hi

        raise ValueError("side must be 'lower' or 'upper'")

    @property
    def mu_hat(self) -> float:
        if self.n == 0:
            return 0.0
        y_bar = sum(self._observations) / float(self.n)
        return 2.0 * y_bar - 1.0

    @property
    def cs_interval(self) -> tuple[float, float]:
        if self.n < 2:
            return (-1.0, 1.0)
        y_bar = sum(self._observations) / float(self.n)
        lo_y = self._find_boundary(y_bar, side="lower")
        hi_y = self._find_boundary(y_bar, side="upper")
        lo = 2.0 * lo_y - 1.0
        hi = 2.0 * hi_y - 1.0
        lo = max(-1.0, min(float(lo), 1.0))
        hi = max(-1.0, min(float(hi), 1.0))
        if hi < lo:
            lo, hi = hi, lo
        return (lo, hi)

    @property
    def radius(self) -> float:
        lo, hi = self.cs_interval
        return (hi - lo) / 2.0

    @property
    def significant(self) -> bool:
        lo, hi = self.cs_interval
        return lo > 0.0 or hi < 0.0

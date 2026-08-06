"""One potential function. `K(lambda) = log E_0[e^{lambda r}]` and every ratio that comes off it.

The frontier layer is a single object with several faces. For the exponential tilt family
``pi_lambda(y) proportional to pi_0(y) e^{lambda r(y)}`` the cumulant generating function

    K(lambda) = log E_0[e^{lambda r}]

generates the rest: ``K'(lambda) = E_lambda[r]``, ``K''(lambda) = Var_lambda(r)``,
``E_lambda[g] = E_0[g e^{lambda r}] / E_0[e^{lambda r}]`` for any behavioural channel ``g``,
``d/dlambda E_lambda[g] = Cov_lambda(g, r)``, and ``KL(pi_lambda || pi_0) = lambda K'(lambda) -
K(lambda)``. These are identities rather than approximations, and every one of them is a ratio of
weighted means over ``n`` base-policy rollouts with ``w_i = exp(lambda r_i)``. Nothing here needs a
gradient, a policy or an optimisation run.

Three implementation commitments hold the module together.

**The max is subtracted before exponentiating, always.** ``exp(lambda r)`` overflows float64 at an
exponent of 710, which for a reward on a scale of 1 is a lambda of 710. That is well inside the
range a sweep visits on a low-variance grader, and the failure is silent: the weights become
``inf``, the self-normalised ratio becomes ``nan``, and a nan propagated into a frontier plot looks
like a missing point rather than an overflow. Subtracting ``max(lambda r_i)`` leaves every ratio
here algebraically unchanged, because all of them are self-normalised.

**The sweep is parametrised by a dimensionless pressure.** ``lambda`` carries the reciprocal of the
reward's scale, so a lambda grid fixed in absolute terms means something different on every grader
and is not comparable across two of them. The grid here is ``t = lambda * sd_0(r)``, and lambda is
recovered as ``t / sd_0(r)``. That single choice is what makes the whole layer invariant under
``r -> a r + b``: the affine map sends ``sd_0 -> a sd_0``, hence ``lambda -> lambda / a``, which is
exactly the reparametrisation under which the weights, the effective sample size, ``E_lambda[g]``
and ``KL(pi_lambda || pi_0)`` are all algebraically unchanged. A grid fixed in lambda would have
failed that test and the failure would have been an artefact of the grid rather than of the
estimator.

**Nothing in this module is the ancestor's job twice.** ``loops.tilt`` already does self-normalised
importance sampling over this family for the feature-level question (which feature drifts, and how
fast). It raises ``ESSGuardError`` when the weights degenerate, because there the degeneracy is a
guard on a prediction. Here the degeneracy is the reading: the whole argument of the horizon is that
the point at which the weights collapse is a publishable number. So this module computes ESS as a
quantity rather than as a guard, and the refusals live in the instruments. What is genuinely shared
is nine lines of arithmetic, and the two differ in what they return at the end of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

#: The Kish ESS floor, as a fraction of n, that the horizon uses by default. It is a
#: default and not a constant: an instrument may be constructed with a different floor and reports
#: whichever one it used. What it may not do is move silently.
DEFAULT_ESS_FLOOR = 0.05

#: How far the dimensionless pressure `t = lambda * sd_0(r)` is searched before a horizon search
#: gives up and reports that the floor never binds. On a Gaussian reward the 5% floor binds at
#: `t = sqrt(log 20) = 1.73`, so this is a factor of thirty of headroom. It exists because a
#: bounded reward can genuinely never cross the floor: a binary verifier with a pass rate of 0.4
#: has ESS/n -> 0.4 as t -> infinity, and searching forever for a crossing that does not exist is
#: worse than saying so.
DEFAULT_T_CAP = 64.0


def logsumexp(x: np.ndarray) -> float:
    """`log sum exp(x)`, with the max subtracted. Written out rather than imported.

    scipy has this and it is correct. It is written here because the max subtraction is the one
    explicit numerical commitment this layer makes, and a reader checking that commitment should
    be able to see it rather than take an import on trust.
    """
    m = float(np.max(x))
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.sum(np.exp(x - m))))


@dataclass(frozen=True)
class TiltPoint:
    """The tilt family at one lambda, as the six numbers everything else is built from.

    ``ess`` is the Kish effective sample size ``(sum w)^2 / sum w^2`` and ``coverage`` is
    ``n / ess``, which is the best-of-n coverage coefficient ``1 + chi^2(pi_lambda || pi_0)``. The
    two are the same number written twice because two literatures write it two ways, and carrying
    both is cheaper than making every reader do the division.
    """

    lam: float
    t: float
    k: float
    kl: float
    reward_mean: float
    reward_var: float
    ess: float
    coverage: float
    gold_mean: float = float("nan")
    stationarity: float = float("nan")


class Potential:
    """`K(lambda)` and its derivatives on one bank of base-policy rollouts.

    Constructed from the proxy reward ``r`` per rollout and, where the caller has one, a gold
    channel ``g`` scored on the same rollouts. Every method is a ratio of weighted means, so the
    whole family costs one pass over ``n`` numbers per lambda and no model calls at all.

    ``sd`` is the population standard deviation of the proxy under the base policy, with ddof 0,
    because it is an expectation under ``pi_0`` and the population form is the estimator of that
    object. It is also the scale that makes the sweep dimensionless, so getting the convention
    wrong here would show up as a failed invariance test rather than as a rounding difference.
    """

    def __init__(
        self,
        reward: Sequence[float] | np.ndarray,
        gold: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        r = np.asarray(reward, dtype=np.float64).ravel()
        if r.size < 2:
            raise ValueError(f"the tilt family needs at least 2 rollouts; got {r.size}")
        if not np.all(np.isfinite(r)):
            raise ValueError(
                "the proxy reward contains non-finite values. A nan or an inf in the exponent is "
                "not a hard case for this estimator, it is a scoring bug upstream of it."
            )
        g: np.ndarray | None = None
        if gold is not None:
            g = np.asarray(gold, dtype=np.float64).ravel()
            if g.size != r.size:
                raise ValueError(
                    f"the gold channel has {g.size} scores and the proxy has {r.size}. The tilt "
                    f"needs both channels on the *same* n rollouts; two independent samples do not "
                    f"give a joint distribution and nothing here is estimable from them."
                )
            if not np.all(np.isfinite(g)):
                raise ValueError("the gold channel contains non-finite values")
        self.r = r
        self.g = g
        self.n = int(r.size)
        self.mean = float(r.mean())
        self.sd = float(r.std(ddof=0))

    # -- the scale conversion -------------------------------------------------

    @property
    def is_degenerate(self) -> bool:
        """Whether the proxy has no spread, in which case the tilt family is a single point.

        ``sd_0(r) = 0`` means ``pi_lambda = pi_0`` at every lambda: the weights are uniform, the KL
        is zero, and there is no frontier to report because there is no x-axis. An all-pass
        verifier does this, and so does a rubric that every rollout in the bank happens to satisfy.
        """
        return not (self.sd > 0.0)

    def lam_of(self, t: float | np.ndarray) -> np.ndarray:
        """The lambda that a dimensionless pressure `t` names on this grader."""
        return np.asarray(t, dtype=np.float64) / self.sd

    def t_of(self, lam: float | np.ndarray) -> np.ndarray:
        """The dimensionless pressure a lambda corresponds to. The inverse of `lam_of`."""
        return np.asarray(lam, dtype=np.float64) * self.sd

    # -- the weights ----------------------------------------------------------

    def log_weights(self, lam: float) -> np.ndarray:
        """`lambda r_i` with its max removed. Every other method starts here."""
        z = float(lam) * self.r
        return z - float(np.max(z))

    def weights(self, lam: float) -> np.ndarray:
        """Self-normalised importance weights, summing to 1."""
        w = np.exp(self.log_weights(lam))
        return w / w.sum()

    # -- the potential and its derivatives ------------------------------------

    def log_mgf(self, lam: float) -> float:
        """`K(lambda) = log E_0[e^{lambda r}]`, the empirical cumulant generating function."""
        return logsumexp(float(lam) * self.r) - float(np.log(self.n))

    def tilted_mean(self, values: np.ndarray, lam: float) -> float:
        """`E_lambda[h] = sum_i w_i h_i` with normalised weights. The one ratio underneath all of them."""
        return float(self.weights(lam) @ np.asarray(values, dtype=np.float64))

    def reward_mean(self, lam: float) -> float:
        """`K'(lambda) = E_lambda[r]`. The y-axis of the reward curve."""
        return self.tilted_mean(self.r, lam)

    def reward_var(self, lam: float) -> float:
        """`K''(lambda) = Var_lambda(r)`."""
        w = self.weights(lam)
        m = float(w @ self.r)
        d = self.r - m
        return float(w @ (d * d))

    def kl(self, lam: float) -> float:
        """`KL(pi_lambda || pi_0) = lambda K'(lambda) - K(lambda)`, in nats.

        The x-axis of the frontier, and the axis the horizon is reported on. Clipped at zero from
        below: the identity is non-negative for every lambda and a value of -1e-16 on the flat part
        of the curve is float64 noise rather than a negative divergence.
        """
        value = float(lam) * self.reward_mean(lam) - self.log_mgf(lam)
        return max(0.0, value)

    def cov(self, a: np.ndarray, b: np.ndarray, lam: float) -> float:
        """`Cov_lambda(a, b)` under the tilted measure."""
        w = self.weights(lam)
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return float(w @ (a * b)) - float(w @ a) * float(w @ b)

    # -- the gold channel -----------------------------------------------------

    def _require_gold(self) -> np.ndarray:
        if self.g is None:
            raise ValueError(
                "this reading needs a gold channel scored on the same rollouts as the proxy. "
                "Construct the Potential with `gold=`."
            )
        return self.g

    def gold_mean(self, lam: float) -> float:
        """`E_lambda[g]`, the y-axis of the frontier."""
        return self.tilted_mean(self._require_gold(), lam)

    def stationarity(self, lam: float) -> float:
        """`d/dlambda E_lambda[g] = Cov_lambda(g, r)`. The function whose root is the turn.

        The tilt's own identity, and the quantity arXiv 2506.19248 states as its Theorem 3. Where
        this crosses zero from above, `E_lambda[g]` stops rising, which is the turn in the gold
        curve for this particular gold signal `g`.
        """
        return self.cov(self._require_gold(), self.r, lam)

    def stationarity_slope(self, lam: float) -> float:
        """`d/dlambda Cov_lambda(g, r) = E_lambda[(g - E_lambda g)(r - E_lambda r)^2]`.

        The third joint cumulant under the tilted measure, which is the derivative Newton needs.
        Two things follow from it and both are worth stating.

        At ``lambda = 0`` it is ``kappa_3(g, r, r)``, so a single Newton step from zero gives
        ``lambda* = -Cov_0(g, r) / kappa_3(g, r, r)``. That closed-form cumulant expression is a
        convenience, and this is the precise sense in which it is one: it is the first iterate of
        the root-finder, not a different result.

        Its sign is also the second-derivative test. A root with a negative slope is a maximum of
        `E_lambda[g]`, which is the turn worth reporting; a root with a positive slope is a
        minimum and reporting it as the turn would be an error of sign.
        """
        w = self.weights(lam)
        g = self._require_gold()
        gd = g - float(w @ g)
        rd = self.r - float(w @ self.r)
        return float(w @ (gd * rd * rd))

    # -- the horizon quantities ----------------------------------------------

    def ess(self, lam: float) -> float:
        """Kish `ESS(lambda) = (sum w_i)^2 / sum w_i^2`, in samples.

        Bounded above by ``n`` and below by 1, both attained: uniform weights give ``n`` and a
        single dominating weight gives 1.
        """
        w = self.weights(lam)
        return 1.0 / float(w @ w)

    def coverage(self, lam: float) -> float:
        """`n / ESS(lambda)`, which is `1 + chi^2(pi_lambda || pi_0)` exactly.

        Not a resemblance. ``chi^2(pi_lambda || pi_0) = E_0[(dpi_lambda/dpi_0)^2] - 1 =
        E_0[e^{2 lambda r}] / E_0[e^{lambda r}]^2 - 1``, and the plug-in of that is
        ``exp(K(2 lambda) - 2 K(lambda)) - 1``, which is ``n / ESS - 1`` on the same weights. The
        best-of-n coverage literature's controlling coefficient and this layer's visibility horizon
        are one quantity written two ways.
        """
        return self.n / self.ess(lam)

    def log_coverage(self, lam: float) -> float:
        """`log(n / ESS) = K(2 lambda) - 2 K(lambda)`, computed through the potential.

        The identity above, evaluated the other way round. It is here because it is what makes the
        monotonicity of the horizon a theorem rather than an observation: the derivative is
        ``2(K'(2 lambda) - K'(lambda))``, which is non-negative for ``lambda >= 0`` because ``K``
        is convex, so ESS is non-increasing on the positive axis and a bisection for the crossing
        is exact rather than a search over a possibly wiggly curve. The empirical ``K`` is convex
        for the same reason the population one is, so this holds on the sample as well.
        """
        return self.log_mgf(2.0 * float(lam)) - 2.0 * self.log_mgf(float(lam))

    # -- one lambda, packaged -------------------------------------------------

    def at(self, lam: float) -> TiltPoint:
        """Everything at one lambda, in one pass."""
        w = self.weights(lam)
        rm = float(w @ self.r)
        rd = self.r - rm
        k = self.log_mgf(lam)
        gold = float("nan")
        stat = float("nan")
        if self.g is not None:
            gold = float(w @ self.g)
            stat = float(w @ (self.g * self.r)) - gold * rm
        ess = 1.0 / float(w @ w)
        return TiltPoint(
            lam=float(lam),
            t=float(self.t_of(lam)),
            k=k,
            kl=max(0.0, float(lam) * rm - k),
            reward_mean=rm,
            reward_var=float(w @ (rd * rd)),
            ess=ess,
            coverage=self.n / ess,
            gold_mean=gold,
            stationarity=stat,
        )

    def sweep(self, lambdas: Sequence[float] | np.ndarray) -> list[TiltPoint]:
        """`at` over a grid, in grid order."""
        return [self.at(float(lam)) for lam in np.asarray(lambdas, dtype=np.float64)]


def horizon_lambda(
    pot: Potential,
    *,
    floor: float = DEFAULT_ESS_FLOOR,
    t_cap: float = DEFAULT_T_CAP,
    tol: float = 1e-9,
) -> tuple[float, bool]:
    """The largest lambda >= 0 at which `ESS(lambda)` still clears `floor * n`.

    Returns ``(lambda_max, binding)``. ``binding`` is False when the floor was never crossed inside
    the searched range, which is a real outcome rather than a numerical failure: a bounded reward
    keeps a fixed fraction of its mass at the top no matter how hard it is tilted, so a binary
    verifier passing 40% of the bank has ``ESS / n -> 0.4`` and never reaches a 5% floor. When that
    happens the search returns the cap and says so, and the instrument reports the horizon as
    non-binding rather than inventing a crossing.

    The search is a bisection and it is exact rather than approximate, because ``log(n / ESS) =
    K(2 lambda) - 2 K(lambda)`` has derivative ``2(K'(2 lambda) - K'(lambda)) >= 0`` for
    ``lambda >= 0`` by convexity of ``K``. ESS is therefore non-increasing on the positive axis and
    has at most one crossing of any floor.
    """
    if pot.is_degenerate:
        return 0.0, False
    target = float(floor) * pot.n
    lam_cap = float(pot.lam_of(t_cap))
    if pot.ess(lam_cap) >= target:
        return lam_cap, False
    lo, hi = 0.0, lam_cap
    while hi - lo > tol * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        if pot.ess(mid) >= target:
            lo = mid
        else:
            hi = mid
    return lo, True


def bootstrap_indices(n: int, resamples: int, seed: int) -> np.ndarray:
    """`(resamples, n)` index matrix for a nonparametric bootstrap over rollouts.

    Resampling rollouts rather than residuals is the right unit here because every reading in this
    layer is a functional of the joint empirical distribution of ``(r, g)``, and the pairing is the
    whole object. Resampling the two channels independently would destroy the covariance that the
    frontier, the checklist and the concomitant are all about.
    """
    return np.random.default_rng(seed).integers(0, n, size=(int(resamples), int(n)))


def percentile_interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """A percentile interval, with the non-finite resamples dropped and counted by the caller.

    Percentile rather than BCa. The readings here are smooth functionals of the empirical
    distribution and the samples are cheap, so the acceleration term buys very little, and a BCa
    interval computed from a jackknife over n rollouts costs n times more for a correction that is
    below the width of the interval on every case tested here.
    """
    finite = np.asarray(samples, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    alpha = 0.5 * (1.0 - float(level))
    return (
        float(np.quantile(finite, alpha)),
        float(np.quantile(finite, 1.0 - alpha)),
    )


__all__ = [
    "DEFAULT_ESS_FLOOR",
    "DEFAULT_T_CAP",
    "Potential",
    "TiltPoint",
    "bootstrap_indices",
    "horizon_lambda",
    "logsumexp",
    "percentile_interval",
]

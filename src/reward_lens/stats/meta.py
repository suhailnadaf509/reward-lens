"""Random-effects meta-analysis, with the prediction interval reported properly.

The reason this module exists is one sentence long: **the confidence interval is about the mean
effect, the prediction interval is about the effect in a new study, and with real heterogeneity
they are not close.** A meta-analysis that reports only the first has told you where the average
sits and left you to assume that is where the next result will land. It usually is not.

The formula is three lines (Higgins, Thompson & Spiegelhalter 2009 eq. 12; Riley, Higgins & Deeks
2011):

    PI = mu_hat  +/-  t(k-2, 1-alpha/2) * sqrt(tau2_hat + SE(mu_hat)^2)

and no machine-learning package exposes it. Two details in it are easy to get wrong and both matter
most exactly where the interval matters most, at small k:

  1. **The critical value is a t, not a z.** `metafor`'s `predict.rma` uses a standard normal by
     default and says so in its own documentation. At k = 20 that is a 3% difference and nobody
     notices. At k = 4 it is t(2, 0.975) = 4.303 against z = 1.96, so the normal version reports an
     interval 2.2 times too narrow. Both conventions are implemented here, named, and selectable;
     `PredictionRule.HTS` (the t) is the default because it is the one the interval's own literature
     recommends, and `PredictionRule.NORMAL` exists so a reader can reproduce what `metafor` prints.
  2. **The variance under the root is tau2 plus the squared standard error, not one or the other.**
     tau2 alone ignores that the centre of the interval is estimated. SE^2 alone is the confidence
     interval wearing a different name.

What else is here, and why each piece earns its place:

  - **Three estimators of tau2.** DerSimonian-Laird because it is what everyone ships and a reader
    will want to compare against; Paule-Mandel and REML because DL is known to be biased downward
    under substantial heterogeneity, which biases the prediction interval in the flattering
    direction. All three are computed on every fit and reported together, because at small k the
    spread between them is itself information. The default is PM.
  - **A confidence interval for tau2 itself** (Q-profile, Viechtbauer 2007). This is the piece the
    field's own critique asks for: an I2 of 0% or a tau2_hat of 0.00 at k = 6 is a point estimate
    with an enormous standard error, and reporting it bare invites the reader to conclude the
    studies agree when the data merely failed to prove they disagree.
  - **I2, reported below tau2 rather than as the headline.** Rucker, Schwarzer, Carpenter &
    Schumacher (2008) show I2 depends on the precision of the included studies: as study size grows,
    tau2 stays put and I2 climbs toward 100%. This bites much harder in machine learning than in
    medicine, because study size there is how many eval items somebody chose to run and is bounded
    only by budget. `Heterogeneity.caveat` prints that warning every time; it is not optional.
  - **Refusals where the arithmetic runs out.** k < 2 is not a meta-analysis. k = 2 has one degree
    of freedom for tau2 and no prediction interval at all, since t(0) does not exist, so the fit
    refuses and hands back the fixed-effect pooled estimate as a bound rather than a random-effects
    number that would look like a result. Egger's test refuses below k = 10 for the same kind of
    reason.

The proportion helpers at the bottom exist because the first real use of this module pools "what
fraction of published comparisons did not survive re-analysis", and a proportion cannot be pooled on
its raw scale when the denominators range from 4 to 40. Logit is the default and double-arcsine is
available; the reasons for that ordering are in `proportion_effects`.

Validation. Every estimator here is checked in `tests/test_stats_meta.py` against published output
rather than against itself: the BCG vaccine meta-analysis (Colditz et al. 1994, `metafor`'s
`dat.bcg`) for REML, Q, I2, H2 and the normal-form prediction interval, and DerSimonian & Kacker
(2007), which is a k = 6 example, for DL, PM and REML side by side. `power.py` next door states the
same discipline for power calculations and this module holds to it.

Torch-free, numpy and scipy only, like the rest of `stats/`.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from reward_lens.core.quantity import QUANTITIES, Quantity, Unit, register_quantity
from reward_lens.core.reading import Refusal, RefusalReason

#: The convention, stated rather than implied, matching `power.py`.
ALPHA = 0.05

#: Below this many studies there is no meta-analysis to do. Two studies give tau2 one degree of
#: freedom and give the prediction interval none at all, so the fit refuses at k < 3 rather than
#: printing a random-effects interval whose width is an artifact of the estimator.
MIN_STUDIES = 3

#: Between MIN_STUDIES and here, a fit is returned but every rendering of it carries the small-k
#: warning. Five is the number below which the tau2 literature stops recommending point estimates
#: without an accompanying interval; it is a convention, not a theorem, and it is named so it can be
#: argued with.
SMALL_K = 5

#: Cochrane's guidance on funnel-plot asymmetry tests: do not run one below ten studies. Egger's
#: test refuses rather than returning a p-value nobody should read.
MIN_STUDIES_EGGER = 10

TauMethod = Literal["DL", "PM", "REML"]


class PredictionRule(enum.Enum):
    """Which critical value the prediction interval uses.

    The two in circulation differ by a factor of 2.2 at k = 4 and by 3% at k = 20, so the choice is
    invisible in the settings where nobody checks it and decisive in the settings where the interval
    is the whole point.
    """

    #: t with k-2 degrees of freedom. Higgins, Thompson & Spiegelhalter (2009) eq. 12, restated in
    #: Riley, Higgins & Deeks (2011). The default here, and the default in R's `meta` up to 7.0-0.
    HTS = "t(k-2)"
    #: Standard normal. What `metafor`'s `predict.rma` uses unless asked otherwise. Reproducible,
    #: widely printed, and too narrow at small k.
    NORMAL = "z"


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------


def _z(p: float) -> float:
    from scipy.special import ndtri

    return float(ndtri(p))


def _t(p: float, df: int) -> float:
    from scipy.stats import t as student_t

    return float(student_t.ppf(p, df))


def _chi2_sf(x: float, df: int) -> float:
    from scipy.stats import chi2

    return float(chi2.sf(x, df))


def _chi2_ppf(p: float, df: int) -> float:
    from scipy.stats import chi2

    return float(chi2.ppf(p, df))


def _as_arrays(effects: Sequence[float], variances: Sequence[float]) -> tuple[Any, Any]:
    y = np.asarray(effects, dtype=float)
    v = np.asarray(variances, dtype=float)
    if y.shape != v.shape or y.ndim != 1:
        raise ValueError(
            f"effects and variances must be 1-D and the same length; got {y.shape} and {v.shape}"
        )
    if y.size and not np.all(np.isfinite(y)):
        raise ValueError("effects contain a non-finite value")
    if v.size and (not np.all(np.isfinite(v)) or np.any(v <= 0)):
        raise ValueError("variances must be finite and strictly positive")
    return y, v


# ---------------------------------------------------------------------------
# The fixed-effect fit, which is also the mandatory comparator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedEffect:
    """The common-effect (fixed-effect) fit: every study estimates the same thing.

    Kept as a first-class object rather than an intermediate because it is the honest comparator for
    the random-effects fit. If the two pooled estimates agree and the random-effects interval is
    barely wider, the random-effects machinery added nothing and the reader should be able to see
    that. If they disagree, the disagreement is driven by the small studies gaining weight under
    random effects, and that is worth knowing before anybody quotes the number.
    """

    k: int
    pooled: float
    se: float
    ci: tuple[float, float]
    weights: Any
    alpha: float = ALPHA

    def render(self) -> str:
        lo, hi = self.ci
        pct = int(round((1 - self.alpha) * 100))
        return f"fixed effect  {self.pooled:+.4f}  {pct}% CI [{lo:+.4f}, {hi:+.4f}]  (k={self.k})"


def fixed_effect(
    effects: Sequence[float], variances: Sequence[float], *, alpha: float = ALPHA
) -> FixedEffect:
    """Inverse-variance weighted pooling with no between-study variance term."""
    y, v = _as_arrays(effects, variances)
    if y.size == 0:
        raise ValueError("no studies")
    w = 1.0 / v
    sw = float(w.sum())
    mu = float((w * y).sum() / sw)
    se = math.sqrt(1.0 / sw)
    crit = _z(1 - alpha / 2)
    return FixedEffect(
        k=int(y.size),
        pooled=mu,
        se=se,
        ci=(mu - crit * se, mu + crit * se),
        weights=w / sw,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Heterogeneity: Q, the three tau2 estimators, the Q-profile interval, I2
# ---------------------------------------------------------------------------


def cochran_q(effects: Sequence[float], variances: Sequence[float]) -> tuple[float, int, float]:
    """Cochran's Q, its degrees of freedom, and its p-value.

    Q is the weighted sum of squared deviations from the fixed-effect mean. Under homogeneity it is
    approximately chi-square on k-1 degrees of freedom. It is underpowered at small k and
    overpowered at large k, which is why nothing downstream treats "Q was not significant" as
    "the studies agree".
    """
    y, v = _as_arrays(effects, variances)
    fe = fixed_effect(y, v)
    w = 1.0 / v
    q = float((w * (y - fe.pooled) ** 2).sum())
    df = int(y.size) - 1
    return q, df, _chi2_sf(q, df) if df > 0 else float("nan")


def generalised_q(effects: Sequence[float], variances: Sequence[float], tau2: float) -> float:
    """The generalised Q statistic at a given tau2, weights 1 / (v_i + tau2).

    Strictly decreasing in tau2, which is what makes both Paule-Mandel and the Q-profile interval
    solvable by bisection with no starting-value problem. At tau2 = 0 it is Cochran's Q.
    """
    y, v = _as_arrays(effects, variances)
    w = 1.0 / (v + tau2)
    mu = float((w * y).sum() / w.sum())
    return float((w * (y - mu) ** 2).sum())


def tau2_dersimonian_laird(effects: Sequence[float], variances: Sequence[float]) -> float:
    """The 1986 moment estimator: tau2 = max(0, (Q - df) / C).

    Closed form, no iteration, and the one every package defaults to. Its known defect is that it is
    biased downward when heterogeneity is substantial, and a downward-biased tau2 makes the
    prediction interval too narrow, which is the direction that flatters the analysis. Reported here
    alongside PM and REML rather than instead of them.
    """
    y, v = _as_arrays(effects, variances)
    if y.size < 2:
        return 0.0
    w = 1.0 / v
    sw = float(w.sum())
    c = sw - float((w**2).sum()) / sw
    q, df, _ = cochran_q(y, v)
    if c <= 0:
        return 0.0
    return max(0.0, (q - df) / c)


def tau2_paule_mandel(
    effects: Sequence[float],
    variances: Sequence[float],
    *,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """Paule-Mandel (1982): the tau2 at which the generalised Q equals its expectation, k-1.

    Solved by bisection on a bracket that is widened until it contains the root. Bisection rather
    than Newton because generalised Q is monotone and the bracket is guaranteed, so bisection cannot
    fail, and this is not a hot loop. Identical to the empirical-Bayes estimator, which is a useful
    cross-check: `metafor` reports the two as the same number and so does this.

    Returns 0.0 when Q at tau2 = 0 is already at or below k-1, which is the correct answer and not a
    failure to converge: the data give no evidence of between-study variance. It is emphatically not
    the same statement as "there is none", which is what the Q-profile interval is for.
    """
    y, v = _as_arrays(effects, variances)
    k = int(y.size)
    if k < 2:
        return 0.0
    target = float(k - 1)
    if generalised_q(y, v, 0.0) <= target:
        return 0.0
    lo, hi = 0.0, max(1.0, float(np.var(y)) + float(v.max()))
    for _ in range(100):
        if generalised_q(y, v, hi) <= target:
            break
        hi *= 2.0
    else:  # pragma: no cover - would need a pathological input
        return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if generalised_q(y, v, mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def tau2_reml(
    effects: Sequence[float],
    variances: Sequence[float],
    *,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> float:
    """Restricted maximum likelihood, by the standard fixed-point iteration.

    The REML estimating equation for the random-effects meta-analysis is

        tau2 = [ sum w_i^2 { (y_i - mu_hat)^2 - v_i } / sum w_i^2 ] + 1 / sum w_i

    with w_i = 1 / (v_i + tau2). The trailing term is the correction for having estimated mu, which
    is the whole difference between REML and plain ML and the reason ML is biased downward. Iterated
    from the DL estimate and truncated at zero.

    Damped, because the raw fixed point can oscillate when the estimate is near the zero boundary.
    Damping changes the path and not the fixed point.
    """
    y, v = _as_arrays(effects, variances)
    k = int(y.size)
    if k < 2:
        return 0.0
    tau2 = tau2_dersimonian_laird(y, v)
    for _ in range(max_iter):
        w = 1.0 / (v + tau2)
        sw = float(w.sum())
        mu = float((w * y).sum() / sw)
        w2 = float((w**2).sum())
        proposal = float((w**2 * ((y - mu) ** 2 - v)).sum()) / w2 + 1.0 / sw
        proposal = max(0.0, proposal)
        step = proposal - tau2
        tau2 = tau2 + 0.5 * step
        if abs(step) < tol:
            break
    return max(0.0, tau2)


TAU2_ESTIMATORS = {
    "DL": tau2_dersimonian_laird,
    "PM": tau2_paule_mandel,
    "REML": tau2_reml,
}

TAU2_NAMES = {
    "DL": "DerSimonian-Laird",
    "PM": "Paule-Mandel",
    "REML": "restricted maximum likelihood",
}


def tau2_q_profile_ci(
    effects: Sequence[float], variances: Sequence[float], *, alpha: float = ALPHA
) -> tuple[float, float]:
    """The Q-profile confidence interval for tau2 (Viechtbauer 2007).

    Invert the generalised Q statistic against its chi-square reference: the lower limit is the tau2
    at which generalised Q equals the upper chi-square quantile, the upper limit is where it equals
    the lower one. Because generalised Q is strictly decreasing in tau2, both limits are found by
    bisection and the lower limit is exactly 0 whenever Q at tau2 = 0 already sits below the upper
    quantile.

    This is the number that stops "tau2_hat = 0.00" from being read as "the studies agree". At k = 4
    the interval is typically wide enough to contain values of tau2 that would make the prediction
    interval span most of the parameter's support, and saying so is the difference between a
    meta-analysis and a summary table.
    """
    y, v = _as_arrays(effects, variances)
    k = int(y.size)
    df = k - 1
    if df < 1:
        return (0.0, float("inf"))
    q_lower_target = _chi2_ppf(1 - alpha / 2, df)  # gives the LOWER tau2 limit
    q_upper_target = _chi2_ppf(alpha / 2, df)  # gives the UPPER tau2 limit

    def solve(target: float) -> float:
        if generalised_q(y, v, 0.0) <= target:
            return 0.0
        lo, hi = 0.0, max(1.0, float(np.var(y)) + float(v.max()))
        for _ in range(200):
            if generalised_q(y, v, hi) <= target:
                break
            hi *= 2.0
        else:  # pragma: no cover
            return float("inf")
        for _ in range(300):
            mid = 0.5 * (lo + hi)
            if generalised_q(y, v, mid) > target:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-12:
                break
        return 0.5 * (lo + hi)

    return (solve(q_lower_target), solve(q_upper_target))


def typical_within_variance(variances: Sequence[float]) -> float:
    """Higgins & Thompson's s^2, the "typical" within-study variance.

        s^2 = (k - 1) * sum(w) / ( sum(w)^2 - sum(w^2) ),   w_i = 1 / v_i

    Not the mean of the v_i and not their harmonic mean. It is the quantity that makes
    I2 = tau2 / (tau2 + s^2) reduce exactly to (Q - df) / Q when tau2 is the DerSimonian-Laird
    estimate, which is the identity `test_i_squared_agrees_with_the_q_form_under_dl` checks.
    """
    _, v = _as_arrays(np.zeros(len(variances)), variances)
    k = len(v)
    if k < 2:
        return float("nan")
    w = 1.0 / v
    sw = float(w.sum())
    denom = sw**2 - float((w**2).sum())
    if denom <= 0:
        return float("nan")
    return (k - 1) * sw / denom


def i_squared(tau2: float, typical_variance: float) -> float:
    """I2 as a percentage, from tau2 and the typical within-study variance.

    Written in the tau2 form rather than as (Q - df) / Q because that form is correct for whichever
    tau2 estimator was actually used, and the two coincide when the estimator is DerSimonian-Laird.

    Read it with Rucker et al. (2008) in hand. I2 is the share of total variance that is
    between-study, so it moves when either part moves, and the within-study part is under the
    analyst's control in machine learning in a way it is not in medicine. Ten thousand eval items per
    model drives I2 toward 100% with tau2 unchanged. A high I2 computed on large studies is a
    statement about the item counts somebody chose.
    """
    if not math.isfinite(typical_variance) or tau2 + typical_variance <= 0:
        return float("nan")
    return 100.0 * tau2 / (tau2 + typical_variance)


@dataclass(frozen=True)
class Heterogeneity:
    """Everything the fit knows about between-study variance, with tau2 first.

    tau2 is in the squared units of the effect size and it is the interpretable one: it is the
    variance of the true effects across studies, so its square root is directly comparable to the
    effects themselves. I2 is a ratio and is reported second, with its caveat attached, because it is
    the one that gets quoted out of context.
    """

    tau2: float
    tau2_method: TauMethod
    tau2_ci: tuple[float, float]
    tau2_all: dict[str, float]
    q: float
    q_df: int
    q_p: float
    i2: float
    h2: float
    typical_variance: float
    k: int

    @property
    def tau(self) -> float:
        return math.sqrt(max(0.0, self.tau2))

    @property
    def is_reliable(self) -> bool:
        """Whether k is large enough for the tau2 point estimate to mean much on its own."""
        return self.k >= SMALL_K

    def caveat(self) -> str:
        """The sentences that must travel with these numbers. Not optional, not a footnote."""
        lo, hi = self.tau2_ci
        hi_s = "unbounded" if not math.isfinite(hi) else f"{hi:.4f}"
        parts = [
            f"tau2 = {self.tau2:.4f} by {TAU2_NAMES[self.tau2_method]}, "
            f"95% Q-profile interval [{lo:.4f}, {hi_s}]. The interval, not the point estimate, is "
            f"what the {self.k} studies support.",
            f"I2 = {self.i2:.1f}% depends on the precision of the included studies as well as on "
            f"their disagreement (Rucker et al. 2008): raise the item counts and I2 climbs toward "
            f"100% with tau2 unchanged. Read tau2 first.",
        ]
        if not self.is_reliable:
            parts.append(
                f"k = {self.k} is below {SMALL_K}. tau2 at this k is estimated with very few "
                f"degrees of freedom and a point estimate of 0.0000 means the data failed to "
                f"demonstrate heterogeneity, which is a different claim from its absence."
            )
        spread = max(self.tau2_all.values()) - min(self.tau2_all.values())
        if spread > 1e-9:
            listed = ", ".join(f"{m}={t:.4f}" for m, t in sorted(self.tau2_all.items()))
            parts.append(
                f"The three estimators do not agree ({listed}); at this k the choice of estimator "
                f"moves the answer and is a reported analysis decision rather than a default."
            )
        return " ".join(parts)

    def render(self) -> str:
        lo, hi = self.tau2_ci
        hi_s = "inf" if not math.isfinite(hi) else f"{hi:.4f}"
        return (
            f"tau2 = {self.tau2:.4f} [{lo:.4f}, {hi_s}] ({self.tau2_method})   "
            f"I2 = {self.i2:.1f}%   H2 = {self.h2:.2f}   "
            f"Q({self.q_df}) = {self.q:.4f}, p = {self.q_p:.4g}"
        )


def heterogeneity(
    effects: Sequence[float],
    variances: Sequence[float],
    *,
    tau2_method: TauMethod = "PM",
    alpha: float = ALPHA,
) -> Heterogeneity:
    """Every heterogeneity statistic in one pass, with all three tau2 estimates carried."""
    y, v = _as_arrays(effects, variances)
    q, df, p = cochran_q(y, v)
    all_tau2 = {name: float(fn(y, v)) for name, fn in TAU2_ESTIMATORS.items()}
    tau2 = all_tau2[tau2_method]
    s2 = typical_within_variance(v)
    return Heterogeneity(
        tau2=tau2,
        tau2_method=tau2_method,
        tau2_ci=tau2_q_profile_ci(y, v, alpha=alpha),
        tau2_all=all_tau2,
        q=q,
        q_df=df,
        q_p=p,
        i2=i_squared(tau2, s2),
        h2=(tau2 / s2 + 1.0) if math.isfinite(s2) and s2 > 0 else float("nan"),
        typical_variance=s2,
        k=int(y.size),
    )


# ---------------------------------------------------------------------------
# The prediction interval
# ---------------------------------------------------------------------------


def prediction_interval(
    pooled: float,
    se: float,
    tau2: float,
    k: int,
    *,
    alpha: float = ALPHA,
    rule: PredictionRule = PredictionRule.HTS,
) -> tuple[float, float, int | None]:
    """The interval a *new* study's true effect falls in, with the critical value named.

    Returns (low, high, df); df is None under the normal rule, which has none.

    The three lines are the whole thing:

        spread = sqrt(tau2 + se^2)
        crit   = t(k-2, 1-alpha/2)     under HTS, or z(1-alpha/2) under NORMAL
        return pooled -/+ crit * spread

    Raises for k < 3 under HTS, because t on zero or negative degrees of freedom does not exist.
    Callers inside this module refuse before reaching that point; the raise is here so that anyone
    calling the function directly gets a stop rather than a nan.

    When tau2 is 0 this collapses to the confidence interval scaled by t(k-2) / z, so it is still
    wider. That is correct and it is not an artifact: even with no estimated between-study variance,
    the interval for a new observation carries the uncertainty in the centre plus the uncertainty in
    having estimated tau2 as zero from k points.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    spread = math.sqrt(max(0.0, tau2) + se**2)
    if rule is PredictionRule.NORMAL:
        crit = _z(1 - alpha / 2)
        df: int | None = None
    else:
        df = k - 2
        if df < 1:
            raise ValueError(
                f"the HTS prediction interval needs t({k - 2}) and k = {k} leaves "
                f"{k - 2} degrees of freedom; there is no prediction interval at this k"
            )
        crit = _t(1 - alpha / 2, df)
    return (pooled - crit * spread, pooled + crit * spread, df)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetaAnalysis:
    """A random-effects fit, carrying both intervals and the statements that go with them."""

    k: int
    labels: tuple[str, ...]
    effects: Any
    variances: Any
    weights: Any
    pooled: float
    se: float
    ci: tuple[float, float]
    ci_method: str
    prediction: tuple[float, float]
    prediction_df: int | None
    prediction_rule: PredictionRule
    het: Heterogeneity
    fixed: FixedEffect
    alpha: float = ALPHA
    #: Anything the caller wants carried through to the write-up: source lines, extraction notes.
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def ci_width(self) -> float:
        return self.ci[1] - self.ci[0]

    @property
    def prediction_width(self) -> float:
        return self.prediction[1] - self.prediction[0]

    @property
    def width_ratio(self) -> float:
        """How much wider the prediction interval is than the confidence interval.

        The single number that says whether reporting only the CI would have misled. It is never
        below 1, and it is above 2 whenever k is small even if tau2 is exactly zero.
        """
        return self.prediction_width / self.ci_width if self.ci_width > 0 else float("inf")

    def excludes(self, null: float) -> tuple[bool, bool]:
        """(does the CI exclude `null`, does the prediction interval exclude it).

        The pair is the point of the module. A CI that excludes the null and a prediction interval
        that does not means: the average effect is established, and the effect in the next study is
        not.
        """
        return (
            not (self.ci[0] <= null <= self.ci[1]),
            not (self.prediction[0] <= null <= self.prediction[1]),
        )

    def interpretation(self, null: float | None = None, unit: str = "") -> str:
        """Prose that says which interval is which, in the terms a reader will act on."""
        pct = int(round((1 - self.alpha) * 100))
        u = f" {unit}" if unit else ""
        lines = [
            f"Pooled effect {self.pooled:+.4f}{u} over k = {self.k} studies, "
            f"{TAU2_NAMES[self.het.tau2_method]} tau2.",
            f"The {pct}% confidence interval [{self.ci[0]:+.4f}, {self.ci[1]:+.4f}] is about the "
            f"MEAN effect across the studies pooled here.",
            f"The {pct}% prediction interval [{self.prediction[0]:+.4f}, "
            f"{self.prediction[1]:+.4f}] is about the effect in a NEW study drawn from the same "
            f"population, which is the interval a reader deciding what to expect next should use. "
            f"It is {self.width_ratio:.2f} times wider "
            f"({self.prediction_rule.value} critical value"
            + (f", {self.prediction_df} df" if self.prediction_df is not None else "")
            + ").",
        ]
        if null is not None:
            ci_ex, pi_ex = self.excludes(null)
            if ci_ex and not pi_ex:
                lines.append(
                    f"The confidence interval excludes {null:+.4g} and the prediction interval "
                    f"does not. The average is established; the next study is not. Reporting only "
                    f"the confidence interval here would overstate what is known."
                )
            elif ci_ex and pi_ex:
                lines.append(
                    f"Both intervals exclude {null:+.4g}, so the claim survives being asked about "
                    f"a study that has not been run yet."
                )
            elif not ci_ex:
                lines.append(
                    f"The confidence interval contains {null:+.4g}, so the pooled effect is not "
                    f"distinguishable from it, and the prediction interval is necessarily wider."
                )
        lines.append(self.het.caveat())
        lines.append(
            f"Comparator: the fixed-effect pooled estimate is {self.fixed.pooled:+.4f} with "
            f"{pct}% CI [{self.fixed.ci[0]:+.4f}, {self.fixed.ci[1]:+.4f}]. It assumes every study "
            f"estimates the same effect and it is reported so the reader can see what the "
            f"random-effects model changed."
        )
        return " ".join(lines)

    def render(self) -> str:
        pct = int(round((1 - self.alpha) * 100))
        rows = [
            f"random-effects meta-analysis, k = {self.k}",
            f"  pooled          {self.pooled:+.4f}  (SE {self.se:.4f})",
            f"  {pct}% CI          [{self.ci[0]:+.4f}, {self.ci[1]:+.4f}]   "
            f"width {self.ci_width:.4f}   about the MEAN effect  [{self.ci_method}]",
            f"  {pct}% prediction  [{self.prediction[0]:+.4f}, {self.prediction[1]:+.4f}]   "
            f"width {self.prediction_width:.4f}   about a NEW study  "
            f"[{self.prediction_rule.value}]",
            f"  ratio           {self.width_ratio:.2f}x",
            f"  {self.het.render()}",
            f"  {self.fixed.render()}",
        ]
        w = np.asarray(self.weights, dtype=float)
        for i, label in enumerate(self.labels):
            rows.append(
                f"    {label:<34s} {self.effects[i]:+.4f}  "
                f"(v {self.variances[i]:.4f}, weight {100 * w[i]:5.1f}%)"
            )
        return "\n".join(rows)

    def as_dict(self) -> dict[str, Any]:
        """A plain dict, for a JSON artifact or an Evidence payload."""
        return {
            "k": self.k,
            "labels": list(self.labels),
            "effects": [float(x) for x in self.effects],
            "variances": [float(x) for x in self.variances],
            "weights": [float(x) for x in self.weights],
            "pooled": self.pooled,
            "se": self.se,
            "ci": list(self.ci),
            "ci_method": self.ci_method,
            "prediction_interval": list(self.prediction),
            "prediction_df": self.prediction_df,
            "prediction_rule": self.prediction_rule.value,
            "width_ratio": self.width_ratio,
            "tau2": self.het.tau2,
            "tau2_method": self.het.tau2_method,
            "tau2_ci": list(self.het.tau2_ci),
            "tau2_all": dict(self.het.tau2_all),
            "i2": self.het.i2,
            "h2": self.het.h2,
            "q": self.het.q,
            "q_df": self.het.q_df,
            "q_p": self.het.q_p,
            "fixed_effect": {
                "pooled": self.fixed.pooled,
                "se": self.fixed.se,
                "ci": list(self.fixed.ci),
            },
            "alpha": self.alpha,
            "context": dict(self.context),
        }


def random_effects(
    effects: Sequence[float],
    variances: Sequence[float],
    *,
    labels: Sequence[str] = (),
    tau2_method: TauMethod = "PM",
    alpha: float = ALPHA,
    knapp_hartung: bool = False,
    rule: PredictionRule = PredictionRule.HTS,
    instrument: str = "stats.meta.random_effects",
    context: dict[str, Any] | None = None,
) -> MetaAnalysis | Refusal:
    """Fit the random-effects model and report both intervals, or refuse and say why.

    ``knapp_hartung`` switches the confidence interval to the Hartung-Knapp-Sidik-Jonkman variance
    estimator with a t(k-1) critical value. It is the better-behaved interval at small k and it is
    off by default only so that the primary output matches what a reader reproducing this in
    `metafor` with default settings would get. Turning it on is a reported analysis decision.

    Refuses below three studies. Two studies give tau2 a single degree of freedom and give the
    prediction interval none, so a random-effects fit at k = 2 would print an interval whose width
    is a property of the estimator rather than of the evidence. The refusal carries the fixed-effect
    pooled estimate and its interval in ``statistics``, because "I cannot give you a random-effects
    answer, but the inverse-variance pooled estimate is 0.42 with CI [0.31, 0.53] and it assumes
    homogeneity you have no way to check" is a more useful reply than silence.
    """
    y, v = _as_arrays(effects, variances)
    k = int(y.size)
    names = tuple(labels) if labels else tuple(f"study {i + 1}" for i in range(k))
    if len(names) != k:
        raise ValueError(f"{len(names)} labels for {k} studies")
    if tau2_method not in TAU2_ESTIMATORS:
        raise ValueError(f"unknown tau2 estimator {tau2_method!r}; have {sorted(TAU2_ESTIMATORS)}")

    if k < 2:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ESS_BELOW_FLOOR,
            detail=(
                f"k = {k}. A meta-analysis of fewer than two studies has no between-study variance "
                f"to estimate and no pooling to do."
            ),
            remedy=(
                "Report the single study with its own interval. If you want a pooled number, add "
                "at least two more independent estimates of the same quantity."
            ),
            statistics={"k": k},
        )

    fe = fixed_effect(y, v, alpha=alpha)
    if k < MIN_STUDIES:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ABOVE_LOD_BELOW_LOQ,
            detail=(
                f"k = {k}, below the floor of {MIN_STUDIES}. tau2 would have {k - 1} degree(s) of "
                f"freedom and the prediction interval would need t({k - 2}), which does not exist. "
                f"A random-effects interval at this k reports the estimator, not the evidence."
            ),
            remedy=(
                "Add at least one more independent study, or report the estimates separately with "
                "their own intervals. The fixed-effect pooled value in `statistics` is available "
                "if you state alongside it that homogeneity was assumed and could not be checked."
            ),
            statistics={
                "k": k,
                "fixed_effect_pooled": fe.pooled,
                "fixed_effect_se": fe.se,
                "fixed_effect_ci": list(fe.ci),
                "q": cochran_q(y, v)[0],
            },
        )

    het = heterogeneity(y, v, tau2_method=tau2_method, alpha=alpha)
    w = 1.0 / (v + het.tau2)
    sw = float(w.sum())
    mu = float((w * y).sum() / sw)
    se = math.sqrt(1.0 / sw)

    if knapp_hartung:
        resid = float((w * (y - mu) ** 2).sum())
        se_ci = math.sqrt(resid / ((k - 1) * sw))
        crit = _t(1 - alpha / 2, k - 1)
        ci = (mu - crit * se_ci, mu + crit * se_ci)
        ci_method = f"Hartung-Knapp-Sidik-Jonkman, t({k - 1})"
    else:
        crit = _z(1 - alpha / 2)
        ci = (mu - crit * se, mu + crit * se)
        ci_method = "inverse-variance, z"

    lo, hi, df = prediction_interval(mu, se, het.tau2, k, alpha=alpha, rule=rule)

    return MetaAnalysis(
        k=k,
        labels=names,
        effects=y,
        variances=v,
        weights=w / sw,
        pooled=mu,
        se=se,
        ci=ci,
        ci_method=ci_method,
        prediction=(lo, hi),
        prediction_df=df,
        prediction_rule=rule,
        het=het,
        fixed=fe,
        alpha=alpha,
        context=dict(context or {}),
    )


# ---------------------------------------------------------------------------
# Baselines and diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VoteCount:
    """The naive comparator: how many studies point which way.

    This is what a reader does without a meta-analysis, and the six-row table this module was
    written for is exactly a vote count presented as a conclusion. Reporting it beside the pooled
    estimate is not a courtesy: a vote count has no interval, is not sensitive to how big the
    effects were, and gives the same answer whether every study was decisive or every study was a
    coin flip. Showing it makes the difference visible instead of assumed.
    """

    k: int
    positive: int
    negative: int
    threshold: float

    @property
    def fraction_positive(self) -> float:
        return self.positive / self.k if self.k else float("nan")

    def render(self) -> str:
        return (
            f"vote count: {self.positive} of {self.k} studies above {self.threshold:+.4g} "
            f"({100 * self.fraction_positive:.0f}%). No interval, no weighting, no sensitivity to "
            f"effect magnitude."
        )


def vote_count(effects: Sequence[float], *, threshold: float = 0.0) -> VoteCount:
    """Count studies on each side of a threshold. The mandatory naive comparator."""
    y = np.asarray(effects, dtype=float)
    pos = int((y > threshold).sum())
    return VoteCount(k=int(y.size), positive=pos, negative=int(y.size) - pos, threshold=threshold)


@dataclass(frozen=True)
class EggerTest:
    """Egger's regression test for funnel-plot asymmetry."""

    intercept: float
    se: float
    t: float
    p: float
    df: int
    k: int

    def render(self) -> str:
        return (
            f"Egger intercept {self.intercept:+.4f} (SE {self.se:.4f}), "
            f"t({self.df}) = {self.t:+.3f}, p = {self.p:.4g}"
        )


def eggers_test(
    effects: Sequence[float],
    variances: Sequence[float],
    *,
    instrument: str = "stats.meta.eggers_test",
) -> EggerTest | Refusal:
    """Regress the standardised effect on precision; a non-zero intercept is asymmetry.

    Refuses below ten studies, which is the Cochrane Handbook's guidance and not a house rule. The
    test has almost no power at small k, so a non-significant result there is uninformative and
    would be read as evidence of no publication bias. Returning the refusal is the only reading that
    does not mislead.
    """
    y, v = _as_arrays(effects, variances)
    k = int(y.size)
    if k < MIN_STUDIES_EGGER:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ESS_BELOW_FLOOR,
            detail=(
                f"k = {k}, below the floor of {MIN_STUDIES_EGGER} studies. The test has negligible "
                f"power here, so a non-significant intercept would be indistinguishable from no "
                f"asymmetry and would be reported as such."
            ),
            remedy=(
                "Do not test for funnel asymmetry at this k. State the number of studies and that "
                "publication bias could not be assessed, or assemble at least "
                f"{MIN_STUDIES_EGGER} estimates."
            ),
            statistics={"k": k, "floor": MIN_STUDIES_EGGER},
        )
    se = np.sqrt(v)
    x = 1.0 / se
    yy = y / se
    design = np.column_stack([np.ones(k), x])
    coef, *_ = np.linalg.lstsq(design, yy, rcond=None)
    resid = yy - design @ coef
    df = k - 2
    sigma2 = float((resid**2).sum()) / df
    cov = sigma2 * np.linalg.inv(design.T @ design)
    intercept = float(coef[0])
    se_int = math.sqrt(float(cov[0, 0]))
    tstat = intercept / se_int
    from scipy.stats import t as student_t

    p = float(2 * student_t.sf(abs(tstat), df))
    return EggerTest(intercept=intercept, se=se_int, t=tstat, p=p, df=df, k=k)


# ---------------------------------------------------------------------------
# Power, at the realised k
# ---------------------------------------------------------------------------


def power_for_pooled_effect(
    variances: Sequence[float],
    *,
    tau2: float,
    delta: float,
    alpha: float = ALPHA,
    tails: Literal[1, 2] = 2,
) -> float:
    """Power to detect a pooled effect of size `delta` at the realised within-study variances.

    Hedges & Pigott (2001). Under the random-effects model the pooled estimate has variance
    1 / sum(1 / (v_i + tau2)), so power against a two-sided level-alpha test is

        Phi( |delta| / SE - z(1 - alpha / tails) )

    to a very good approximation, ignoring the lower tail. Post-hoc power computed from an observed
    effect is a rationalisation, which is why this takes `delta` as an argument: it answers "what
    could this collection of studies have detected", which is a design question, and not "was my
    result significant", which is already answered by the interval.

    `tau2` is an input rather than an estimate for the same reason. At small k the estimate is noisy,
    so the honest use is to sweep it: power at tau2 = 0 is the optimistic bound and power at the
    upper Q-profile limit is the pessimistic one.
    """
    _, v = _as_arrays(np.zeros(len(variances)), variances)
    se = math.sqrt(1.0 / float((1.0 / (v + max(0.0, tau2))).sum()))
    from scipy.stats import norm

    return float(norm.cdf(abs(delta) / se - _z(1 - alpha / tails)))


def power_to_detect_heterogeneity(
    variances: Sequence[float],
    *,
    tau2: float,
    alpha: float = ALPHA,
    n_sim: int = 20000,
    seed: int = 0,
) -> float:
    """Power of Cochran's Q to detect between-study variance `tau2`, by simulation.

    By simulation and not by a formula, for the reason `power.py` gives at the top of its own
    module: the chi-square approximation to Q is the thing under test here, so validating a power
    calculation against it would be circular. Draws true effects from N(0, tau2), observed effects
    from N(true, v_i), and counts rejections of the chi-square reference at level alpha.

    The number this returns at k = 4 or k = 6 is the reason "Q was not significant" must never be
    written down as "the studies are homogeneous".
    """
    _, v = _as_arrays(np.zeros(len(variances)), variances)
    k = len(v)
    if k < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    true = rng.normal(0.0, math.sqrt(max(0.0, tau2)), size=(n_sim, k))
    obs = true + rng.normal(0.0, np.sqrt(v), size=(n_sim, k))
    w = 1.0 / v
    mu = (obs * w).sum(axis=1) / w.sum()
    q = (w * (obs - mu[:, None]) ** 2).sum(axis=1)
    crit = _chi2_ppf(1 - alpha, k - 1)
    return float((q > crit).mean())


# ---------------------------------------------------------------------------
# Proportions
# ---------------------------------------------------------------------------

ProportionScale = Literal["logit", "double-arcsine"]


def proportion_effects(
    counts: Sequence[int],
    totals: Sequence[int],
    *,
    scale: ProportionScale = "logit",
    correction: float = 0.5,
) -> tuple[Any, Any]:
    """Transform k-of-n proportions onto a scale that can be pooled, with their variances.

    A proportion cannot be pooled on its raw scale when the denominators differ by a factor of ten:
    the sampling variance p(1-p)/n depends on p, so the weights would depend on the answer, and the
    interval would run past 0 or 1. Two transforms are offered.

    **logit** (the default). y = log((k + c) / (n - k + c)), v = 1 / (k + c) + 1 / (n - k + c). The
    continuity correction c defaults to 0.5 and is applied to **every** study, not only to the ones
    with an empty cell. Applying it selectively means a study is analysed with a different estimator
    because of its own data, which makes the studies non-comparable in a way that is invisible in
    the output. The back-transform is the logistic function: exact, monotone, and it maps the
    prediction interval onto (0, 1) without any further approximation. That last property is why
    logit is the default here.

    **double-arcsine** (Freeman & Tukey 1950). y = (asin(sqrt(k/(n+1))) + asin(sqrt((k+1)/(n+1))))/2,
    v = 1 / (4n + 2). Variance-stabilising, so the weights do not depend on the observed proportion,
    and defined at k = 0 and k = n where the logit is not. Its weakness is the back-transform, which
    needs a sample size that the pooled estimate does not have; the usual substitute is the harmonic
    mean of the n_i, and Schwarzer, Chemaitelly, Abu-Raddad & Rucker (2019) show this can go badly
    wrong when the study sizes are very unequal. Use it when there is a boundary count, and report
    both when the answer changes.

    Set `correction=0.0` for the uncorrected logit when no count is at a boundary.
    """
    c = np.asarray(counts, dtype=float)
    n = np.asarray(totals, dtype=float)
    if c.shape != n.shape or c.ndim != 1:
        raise ValueError("counts and totals must be 1-D and the same length")
    if np.any(n <= 0) or np.any(c < 0) or np.any(c > n):
        raise ValueError("need 0 <= counts <= totals and totals > 0")
    if scale == "logit":
        a = c + correction
        b = n - c + correction
        if np.any(a <= 0) or np.any(b <= 0):
            raise ValueError(
                "a zero cell with correction=0 gives an infinite logit; use correction=0.5 "
                "or the double-arcsine scale"
            )
        return np.log(a / b), 1.0 / a + 1.0 / b
    if scale == "double-arcsine":
        y = 0.5 * (np.arcsin(np.sqrt(c / (n + 1.0))) + np.arcsin(np.sqrt((c + 1.0) / (n + 1.0))))
        return y, 1.0 / (4.0 * n + 2.0)
    raise ValueError(f"unknown scale {scale!r}")


def proportion_back(value: float, *, scale: ProportionScale, totals: Sequence[int] = ()) -> float:
    """Map a pooled value on the analysis scale back to a proportion in [0, 1].

    For **logit** this is the logistic function and nothing else is needed.

    For **double-arcsine** it is Miller's (1978) inverse evaluated at the harmonic mean of the study
    sizes, which is the standard choice and the one `metafor` uses. It reduces to sin^2(y) as n
    grows. `totals` is required for this scale and ignored for the other.
    """
    if scale == "logit":
        return 1.0 / (1.0 + math.exp(-value))
    if scale == "double-arcsine":
        n = np.asarray(totals, dtype=float)
        if n.size == 0:
            raise ValueError("the double-arcsine back-transform needs the study sizes")
        nh = float(n.size / np.sum(1.0 / n))
        s = math.sin(2.0 * value)
        if abs(s) < 1e-12:
            return 0.0 if math.cos(2.0 * value) > 0 else 1.0
        inner = s + (s - 1.0 / s) / nh
        inner = min(1.0, max(-1.0, inner))
        sign = 1.0 if math.cos(2.0 * value) >= 0 else -1.0
        return 0.5 * (1.0 - sign * math.sqrt(max(0.0, 1.0 - inner**2)))
    raise ValueError(f"unknown scale {scale!r}")


@dataclass(frozen=True)
class ProportionMeta:
    """A meta-analysis of proportions, carrying both the analysis scale and the raw scale."""

    fit: MetaAnalysis
    scale: ProportionScale
    counts: tuple[int, ...]
    totals: tuple[int, ...]
    pooled_p: float
    ci_p: tuple[float, float]
    prediction_p: tuple[float, float]
    correction: float

    @property
    def observed(self) -> tuple[float, ...]:
        return tuple(c / n for c, n in zip(self.counts, self.totals))

    def interpretation(self, null: float | None = None) -> str:
        pct = int(round((1 - self.fit.alpha) * 100))
        head = (
            f"Pooled proportion {self.pooled_p:.3f} over k = {self.fit.k} studies "
            f"(analysis on the {self.scale} scale, back-transformed). "
            f"The {pct}% confidence interval [{self.ci_p[0]:.3f}, {self.ci_p[1]:.3f}] is about the "
            f"MEAN proportion. The {pct}% prediction interval "
            f"[{self.prediction_p[0]:.3f}, {self.prediction_p[1]:.3f}] is about the proportion in a "
            f"NEW study, and it is the one to quote when the question is what to expect next."
        )
        null_scale = None
        if null is not None:
            null_scale = math.log(null / (1 - null)) if self.scale == "logit" else None
        tail = self.fit.interpretation(null=null_scale)
        return head + " " + tail

    def render(self) -> str:
        pct = int(round((1 - self.fit.alpha) * 100))
        rows = [
            f"proportion meta-analysis on the {self.scale} scale, k = {self.fit.k}",
            f"  pooled          {self.pooled_p:.4f}",
            f"  {pct}% CI          [{self.ci_p[0]:.4f}, {self.ci_p[1]:.4f}]   about the MEAN",
            f"  {pct}% prediction  [{self.prediction_p[0]:.4f}, {self.prediction_p[1]:.4f}]   "
            f"about a NEW study",
            "",
            self.fit.render(),
        ]
        return "\n".join(rows)

    def as_dict(self) -> dict[str, Any]:
        d = self.fit.as_dict()
        d.update(
            {
                "scale": self.scale,
                "correction": self.correction,
                "counts": list(self.counts),
                "totals": list(self.totals),
                "observed": list(self.observed),
                "pooled_p": self.pooled_p,
                "ci_p": list(self.ci_p),
                "prediction_p": list(self.prediction_p),
            }
        )
        return d


def proportion_meta(
    counts: Sequence[int],
    totals: Sequence[int],
    *,
    labels: Sequence[str] = (),
    scale: ProportionScale = "logit",
    correction: float = 0.5,
    tau2_method: TauMethod = "PM",
    alpha: float = ALPHA,
    knapp_hartung: bool = False,
    rule: PredictionRule = PredictionRule.HTS,
    context: dict[str, Any] | None = None,
) -> ProportionMeta | Refusal:
    """Pool k-of-n proportions and report both intervals on the proportion scale.

    Passes any refusal from `random_effects` straight through, so a caller handles one return type.
    """
    y, v = proportion_effects(counts, totals, scale=scale, correction=correction)
    fit = random_effects(
        y,
        v,
        labels=labels,
        tau2_method=tau2_method,
        alpha=alpha,
        knapp_hartung=knapp_hartung,
        rule=rule,
        instrument="stats.meta.proportion_meta",
        context=context,
    )
    if isinstance(fit, Refusal):
        return fit
    back = lambda value: proportion_back(value, scale=scale, totals=totals)  # noqa: E731
    return ProportionMeta(
        fit=fit,
        scale=scale,
        counts=tuple(int(c) for c in counts),
        totals=tuple(int(n) for n in totals),
        pooled_p=back(fit.pooled),
        ci_p=(back(fit.ci[0]), back(fit.ci[1])),
        prediction_p=(back(fit.prediction[0]), back(fit.prediction[1])),
        correction=correction,
    )


# ---------------------------------------------------------------------------
# Quantities this module estimates, proposed here and NOT registered
# ---------------------------------------------------------------------------
#
# `spec/QUANTITIES.yaml` carries `study.power`, `study.mde` and `study.resolution_ratio` and nothing
# for a pooled effect or a between-study variance, so every number this module returns is a reading
# of a quantity the registry has no id for. Registering one is a decision about what the library
# claims to measure and it belongs in that file, which this module does not write to. The proposals
# live here as data, next to the code they describe, in the shape the controls bank established:
# nothing runs at import, `register_proposed()` exists so a test can register them process-locally,
# and `as_yaml_rows()` emits the exact shape the registry uses so the landed rows cannot drift from
# the objects here.
#
# Three decisions inside them are worth challenging rather than inheriting.
#
# **Two of the three carry an OPEN unit and that is the honest answer, not a deferral.** A pooled
# effect is in whatever units the effect measure was in: a log odds ratio, a logit proportion, a
# standardised mean difference. The quantity is the same quantity in each case, and its unit is a
# property of the input rather than of the quantity, which is exactly the situation the registry
# already has 24 rows for. `study.tau2` is worse still, being in the squared units of that. Forcing
# a decomposition would make a log odds ratio compare cleanly against a logit proportion, and those
# are not the same thing.
#
# **`study.prediction_interval_ratio` is registered and the prediction interval itself is not.** An
# interval is a pair and the registry holds scalars. The ratio of the prediction interval's width to
# the confidence interval's is a scalar, it is dimensionless whatever was pooled, and it is the
# number that answers "would reporting only the confidence interval have misled anyone", which is
# the question this module exists to make askable. The interval endpoints travel in the Evidence
# payload; they are not a separate quantity.
#
# **I2 is deliberately NOT proposed.** It would be one line to add and the argument against it is
# the whole of Rucker et al. (2008): I2 moves with the precision of the included studies, and in
# machine learning precision is a budget decision, so a registered `study.i2` would be a quotable id
# for a number that is partly a statement about somebody's item count. It stays a field on
# `Heterogeneity` with its caveat attached and does not become a first-class quantity. Anyone who
# disagrees will find the row trivial to add, and this comment is the argument to overturn.

_OPEN_UNIT = Unit(dimension="OPEN", per="OPEN", scale="OPEN", as_printed="effect")
_SQ_OPEN_UNIT = Unit(dimension="OPEN", per="OPEN", scale="OPEN", as_printed="effect^2")
_ONE = Unit(dimension="1", per=None, scale=None, as_printed="1")

POOLED_EFFECT = Quantity(
    id="study.pooled_effect",
    definition=(
        "The random-effects pooled estimate over k independent studies: the inverse-variance "
        "weighted mean of the study effects with weights 1 / (v_i + tau2), where v_i is the "
        "within-study sampling variance and tau2 the estimated between-study variance."
    ),
    unit=_OPEN_UNIT,
    invariance="units",
    interpretation=(
        "The mean effect across the studies pooled, and only that. It is not a prediction about a "
        "new study, and its confidence interval is not a prediction interval. Quote it with "
        "study.prediction_interval_ratio beside it or the reader will read it as one."
    ),
    support=None,
    wedge=False,
)

TAU2 = Quantity(
    id="study.tau2",
    definition=(
        "The estimated variance of the true effects across studies in a random-effects "
        "meta-analysis, in the squared units of the effect size. Estimator named alongside: "
        "DerSimonian-Laird, Paule-Mandel or restricted maximum likelihood."
    ),
    unit=_SQ_OPEN_UNIT,
    invariance="units",
    interpretation=(
        "The interpretable heterogeneity statistic, because its square root is on the same scale "
        "as the effects. A point estimate of zero means the data failed to demonstrate "
        "between-study variance, which is not the same claim as its absence, so a reading of this "
        "quantity is incomplete without its confidence interval."
    ),
    support=(0.0, float("inf")),
    wedge=False,
)

PREDICTION_INTERVAL_RATIO = Quantity(
    id="study.prediction_interval_ratio",
    definition=(
        "The width of the 95% prediction interval for the effect in a new study, divided by the "
        "width of the 95% confidence interval for the pooled mean, both from the same "
        "random-effects fit."
    ),
    unit=_ONE,
    invariance="units",
    interpretation=(
        "How much a report that gave only the confidence interval would have understated its own "
        "uncertainty about the next study. Never below 1. Above roughly 2 whenever k is small even "
        "with no estimated between-study variance, because the critical value alone is t(k-2) "
        "rather than z."
    ),
    support=(1.0, float("inf")),
    wedge=False,
)

PROPOSED: tuple[Quantity, ...] = (POOLED_EFFECT, TAU2, PREDICTION_INTERVAL_RATIO)

#: `min_access` in the catalogue's spelling. `design` matches the rest of the `study.*` family: a
#: meta-analysis needs published effect sizes and variances, not a model, a grader or a run.
PROPOSED_MIN_ACCESS: dict[str, str] = {q.id: "design" for q in PROPOSED}

#: No catalogued instrument owns these yet. M10 is the nearest neighbour and estimates the rest of
#: the `study.*` family, so the instrument list is left empty rather than guessed at.
PROPOSED_INSTRUMENTS: dict[str, tuple[str, ...]] = {q.id: () for q in PROPOSED}


def register_proposed() -> list[str]:
    """Register the proposals in this process only. Not called at import, by design."""
    added = []
    for q in PROPOSED:
        if q.id not in QUANTITIES:
            register_quantity(q)
            added.append(q.id)
    return added


def _yaml_scalar(text: str) -> str:
    """Quote a string when leaving it bare would change what YAML reads.

    The definitions here contain colon-space, which ends a key in block context and makes the row
    unparseable. `spec/QUANTITIES.yaml` already single-quotes such values elsewhere. Emitting a row
    that does not load is the same failure as emitting the wrong number, so the output of this
    function is round-tripped through the yaml parser in the tests rather than eyeballed.
    """
    if any(token in text for token in (": ", " #", "'", '"', "\n")) or text.strip() != text:
        return "'" + text.replace("'", "''") + "'"
    return text


def as_yaml_rows() -> str:
    """The proposals in `spec/QUANTITIES.yaml`'s own field order, ready to paste."""
    lines: list[str] = []
    for q in PROPOSED:
        lines.append(f"- id: {q.id}")
        lines.append("  unit:")
        lines.append(f"    as_printed: '{q.unit.as_printed}'")
        lines.append(f"    dimension: '{q.unit.dimension}'")
        lines.append(f"    per: {q.unit.per if q.unit.per else 'null'}")
        lines.append(f"    scale: {q.unit.scale if q.unit.scale else 'null'}")
        lines.append(f"  invariance_group: {q.invariance}")
        lines.append(f"  min_access: {PROPOSED_MIN_ACCESS[q.id]}")
        lines.append("  rungs: 1")
        instruments = PROPOSED_INSTRUMENTS[q.id]
        lines.append("  instrument: []" if not instruments else "  instrument:")
        for inst in instruments:
            lines.append(f"  - {inst}")
        lines.append(f"  wedge: {'true' if q.wedge else 'false'}")
        lines.append(f"  definition: {_yaml_scalar(q.definition)}")
        lines.append(f"  interpretation: {_yaml_scalar(q.interpretation)}")
        if q.support is None:
            lines.append("  support: OPEN")
        else:
            lo, hi = q.support
            hi_s = ".inf" if hi == float("inf") else f"{hi}"
            lines.append(f"  support: [{lo}, {hi_s}]")
    return "\n".join(lines)


__all__ = [
    "ALPHA",
    "MIN_STUDIES",
    "MIN_STUDIES_EGGER",
    "POOLED_EFFECT",
    "PREDICTION_INTERVAL_RATIO",
    "PROPOSED",
    "PROPOSED_INSTRUMENTS",
    "PROPOSED_MIN_ACCESS",
    "SMALL_K",
    "TAU2",
    "TAU2_ESTIMATORS",
    "TAU2_NAMES",
    "EggerTest",
    "FixedEffect",
    "Heterogeneity",
    "MetaAnalysis",
    "PredictionRule",
    "ProportionMeta",
    "ProportionScale",
    "TauMethod",
    "VoteCount",
    "as_yaml_rows",
    "cochran_q",
    "eggers_test",
    "fixed_effect",
    "generalised_q",
    "heterogeneity",
    "i_squared",
    "power_for_pooled_effect",
    "power_to_detect_heterogeneity",
    "prediction_interval",
    "proportion_back",
    "proportion_effects",
    "proportion_meta",
    "random_effects",
    "register_proposed",
    "tau2_dersimonian_laird",
    "tau2_paule_mandel",
    "tau2_q_profile_ci",
    "tau2_reml",
    "typical_within_variance",
    "vote_count",
]

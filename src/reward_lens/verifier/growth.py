"""D6 Exploit-family coverage: how much of the exploit space you have already found.

Says, with your numbers in place of the specification's illustrative ones: "You have found 23
exploit families. Chao1 puts 8% of the families still unseen, and Good-Turing puts the probability
that the next find is novel at 0.06. Crow-AMSAA beta = 0.74, so the blacklist is converging."

That sentence used to attribute the 8% to Good-Turing, which is the confusion the section below
exists to head off and which had got into the module's own headline. Good-Turing's unseen mass is
`f1/n`, the 0.06; the 8% is a Chao1 richness fraction. On a log of 100 finds with `f1 = 7` and
`f2 = 3` the two read 0.070 and 0.262, so the label was on a number 3.7 times the one it named.

Every lab running code-based reward keeps this log. Kimi K3's anti-hacking blacklist is
"continuously extended with new safeguards as new hacking strategies are observed", which is a
sentence describing a dataset, and no lab publishes a number computed from it. The arithmetic that
turns it into one has been in ecology since Chao 1984 and in reliability engineering since Duane
1964. Nothing here needed inventing. It needed pointing at a log people already keep.

**The two estimators and what each actually claims.**

Good-Turing. With `f1` families seen exactly once and `n` observations, the probability that the
next find belongs to a family you have never seen is estimated by `f1/n`. That is a statement about
the *next draw* and it needs no assumption about how many families exist. Chao 1984 then bounds
the number of families you have missed by `f1**2 / (2*f2)`, and `Chao1 = S_obs + f1**2/(2*f2)` is
the resulting richness estimate. Note that these are two different quantities that the field's
shorthand runs together: `f1/n` is a probability *mass*, `f1**2/(2*f2)` is a *count* of families.
The specification's example sentence quotes 8% for one and 0.06 for the other, which only makes
sense if the 8% is a richness fraction rather than Good-Turing's unseen mass. So
`verifier.unseen_exploit_mass` is reported here as the Chao1 richness fraction
`f0_hat / (S_obs + f0_hat)`, with `f0_hat / S_obs` carried beside it, because the two differ by
about a point at the specification's own numbers and the registry's `definition` field is OPEN.
**Every place that number is printed says Chao1 and not Good-Turing**, which is not pedantry: they
are a count and a probability, they are not close numerically, and a reader who acts on one
believing it is the other is deciding whether to keep fuzzing on the wrong statistic.

**Both estimators now ship with an interval, and neither had one.** Good-Turing gets Esty (1983,
*Ann. Statist.* 11(3), 905-912), whose variance is `(f1 + 2*f2 - f1**2/n) / n**2`. At the log
above that is a 95% interval of [0.0007, 0.1393] around a point of 0.070, which is most of the
range the point estimate could have taken, and a team reading 0.070 without it would conclude the
search is nearly exhausted on evidence that does not distinguish that from a 14% novelty rate.
Chao1 gets Chao (1987) with the log-normal transform the ecology literature uses, which keeps the
lower end above zero. That interval covers sampling variation in `f1` and `f2` and **does not cover
Chao1's own downward bias**: simulated on uneven communities it contains the true unseen count
0.847 of the time at moderate unevenness and 0.611 at high unevenness, and essentially every miss
is the truth sitting *above* the interval. That is the same direction the estimator's bias
statement declares, and it is why the reading calls the fraction a floor.

Crow-AMSAA. Fit `N(t) = lambda * t**beta` to cumulative finds against cumulative test effort by
least squares on the log-log plot, which is the Duane plot. `beta < 1` means the instantaneous
discovery rate is falling, so the blacklist is converging on whatever is there to find; `beta > 1`
means finds are accelerating and the search has not saturated. The number without its interval is
not worth printing: on a four-family log the interval routinely spans 1, which is the difference
between "converging" and "we cannot tell yet", and reporting only the point estimate hides that.

**What can go wrong, and what this module does about it.**

`f2 = 0` is common on short logs and makes the Chao1 bound undefined rather than infinite. That is
a refusal carrying the bias-corrected bound `f1*(f1-1)/2` as a partial answer and a remedy naming
how many more finds the log needs, not a division by zero and not a very large number.

The effort axis is the other trap and it is silent. If a log carries no effort measurements, the
only axis available is the find ordinal, and fitting *cumulative finds* against the *find ordinal*
is fitting `N(t) = t`, which returns `beta = 1.0` with `R^2 = 1.0` for any log whatsoever. That is
not a measurement, it is the axis reading itself back, so `crow_amsaa` refuses the combination
rather than returning the tautology. Fitting cumulative *families* against the find ordinal is
fine, because families accumulate strictly more slowly than finds do.

Kill condition, from the catalogue: **nothing; it is arithmetic on a log.** That is the honest
entry and it is also the one to be careful with, because an instrument nothing can kill is an
instrument nothing constrains. An earlier version of this catalogue did name one, and it is worth
keeping in view as the practical failure: if `f1` is zero on every real log, every family having
been seen at least twice, then the estimator is undefined in practice and the log is telling you
the search stopped finding new things a while ago.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty, register_payload
from reward_lens.core.invariance import Relation
from reward_lens.core.quantity import (
    BiasStatement,
    CostModel,
    EstimatorEntry,
    register_estimator,
)
from reward_lens.core.reading import (
    Reading,
    Refusal,
    RefusalReason,
    bounded_refusal,
)
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
    content_hash,
)
from reward_lens.measure.base import BaseObservable, Context, run
from reward_lens.verifier import ensure_quantities

#: What the Duane fit is run over. ``families`` counts distinct families first seen by time `t`,
#: which is what a blacklist actually holds; ``finds`` counts every entry in the log.
FitOn = Literal["families", "finds"]

#: The smallest number of fitted points that leaves a residual degree of freedom, so that a
#: standard error and therefore an interval exist. Two points fit a line exactly and say nothing
#: about how well.
MIN_FIT_POINTS = 3


# ---------------------------------------------------------------------------
# The log
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class ExploitFind:
    """One entry in the log: a family, and the cumulative test effort when it was found.

    ``effort`` is the odometer reading at the moment of the find, in whatever unit the log keeps:
    rollouts graded, candidate inputs tried, GPU-hours, wall-clock days. Crow-AMSAA needs a
    *cumulative* axis and nothing else about it, so the unit only has to be consistent within one
    log. It is carried on the log rather than assumed, because a fit against an axis whose unit
    nobody wrote down is a slope with no meaning.
    """

    family: str
    effort: float | None = None
    id: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError(
                "an exploit find needs a family name. Good-Turing counts families, and an unnamed "
                "find cannot be counted as a repeat of anything."
            )
        if self.effort is not None and not math.isfinite(self.effort):
            raise ValueError(
                f"effort must be finite, got {self.effort!r} for family {self.family!r}"
            )


@dataclass(frozen=True)
class ExploitLog:
    """A sequence of finds in discovery order, and the effort that bought them.

    Order is load-bearing and is the caller's, not this class's: the log is read in the order it
    is given and a find's position is its discovery order. Sorting it here would silently rewrite
    the growth curve of any log whose effort field is absent.

    ``total_effort`` is the odometer at the end of the observation window, which is not the same as
    the effort of the last find: a log that ran for a further ten thousand trials and found nothing
    is strong evidence of convergence, and dropping the tail throws that evidence away.
    """

    finds: tuple[ExploitFind, ...]
    total_effort: float | None = None
    effort_unit: str = "trials"
    source: str = ""

    @classmethod
    def of(
        cls,
        finds: Sequence[ExploitFind] | Sequence[tuple[str, float]] | Sequence[str],
        *,
        total_effort: float | None = None,
        effort_unit: str = "trials",
        source: str = "",
    ) -> "ExploitLog":
        """Build a log from `ExploitFind`s, `(family, effort)` pairs, or bare family names."""
        rows: list[ExploitFind] = []
        for item in finds:
            if isinstance(item, ExploitFind):
                rows.append(item)
            elif isinstance(item, str):
                rows.append(ExploitFind(family=item))
            else:
                family, effort = item
                rows.append(ExploitFind(family=family, effort=float(effort)))
        return cls(tuple(rows), total_effort, effort_unit, source)

    def __len__(self) -> int:
        return len(self.finds)

    def __iter__(self) -> Any:
        return iter(self.finds)

    @property
    def n(self) -> int:
        """Observations. Every entry in the log, not distinct families."""
        return len(self.finds)

    @property
    def has_effort(self) -> bool:
        """Whether every find carries a measured effort, so the Duane axis is real.

        All or nothing on purpose. A log where half the finds carry an odometer reading and half
        do not has no consistent axis, and filling the gaps with ordinals would mix two units in
        one regression.
        """
        return bool(self.finds) and all(f.effort is not None for f in self.finds)

    def counts(self) -> dict[str, int]:
        """How many times each family appears, in first-appearance order."""
        out: dict[str, int] = {}
        for f in self.finds:
            out[f.family] = out.get(f.family, 0) + 1
        return out

    def spectrum(self) -> dict[int, int]:
        """The frequency-of-frequencies `f_r`: how many families appear exactly `r` times."""
        return dict(Counter(self.counts().values()))

    @property
    def s_obs(self) -> int:
        """Distinct families found. The catalogue's mandatory baseline, on its own."""
        return len(self.counts())

    def effort_axis(self) -> tuple[float, ...]:
        """The cumulative-effort value for each find, measured if present and ordinal if not."""
        if self.has_effort:
            return tuple(float(f.effort) for f in self.finds if f.effort is not None)
        return tuple(float(i) for i in range(1, len(self.finds) + 1))

    def discovery_curve(self, on: FitOn = "families") -> tuple[tuple[float, ...], tuple[int, ...]]:
        """`(t, N)` for the Duane fit: cumulative effort against cumulative count.

        For ``families`` a point is emitted only when a family appears for the first time, so `N`
        is the size of the blacklist at effort `t`. For ``finds`` every entry emits a point.
        """
        axis = self.effort_axis()
        ts: list[float] = []
        ns: list[int] = []
        seen: set[str] = set()
        for i, find in enumerate(self.finds):
            if on == "families":
                if find.family in seen:
                    continue
                seen.add(find.family)
            ts.append(axis[i])
            ns.append(len(seen) if on == "families" else i + 1)
        return tuple(ts), tuple(ns)

    def checksum(self) -> str:
        """Content identity of the log, so two readings of different logs are two rows."""
        return content_hash(
            {
                "finds": [{"family": f.family, "effort": f.effort, "id": f.id} for f in self.finds],
                "total_effort": self.total_effort,
                "effort_unit": self.effort_unit,
            },
            "xlog",
        )


@dataclass(frozen=True)
class _LogMeta:
    fingerprint: str
    lineage: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LogSubject:
    """An exploit log, shaped so a `Context` can carry it.

    The third subject shim in this series, after `ProgramSubject` and `QuerySubject`, and the
    weakest of the three: D6 never touches the grader at all, so the only thing it can name is the
    log. That is honest rather than sloppy. A reading whose subject is a log is a reading about
    what the search has turned up, and two labs searching the same grader with different search
    effort should get two different rows.
    """

    log: ExploitLog
    caps: Capability = Capability.SCORES
    intervention_fingerprints: tuple[str, ...] = ()

    @property
    def meta(self) -> _LogMeta:
        return _LogMeta(
            fingerprint=self.log.checksum(),
            lineage={"source": self.log.source, "effort_unit": self.log.effort_unit},
        )


# ---------------------------------------------------------------------------
# Good-Turing and Chao1
# ---------------------------------------------------------------------------


def novelty_probability(f1: int, n: int) -> float:
    """Good-Turing's estimate that the next find belongs to an unseen family: `f1 / n`.

    The one number in this module that assumes nothing about how many families exist. It is a
    statement about the next draw under the same search, which is exactly the question a team
    deciding whether to keep fuzzing is asking.
    """
    return float("nan") if n <= 0 else f1 / n


def chao1_unseen(f1: int, f2: int) -> float:
    """`f1**2 / (2*f2)`: the estimated number of families the log has missed.

    Undefined at `f2 = 0` and this returns NaN there rather than infinity, because the caller has
    to branch on it and NaN propagates into a comparison as False while infinity propagates as a
    confident answer. `exploit_coverage` turns the NaN into a refusal with a bound.
    """
    if f1 <= 0:
        return 0.0
    if f2 <= 0:
        return float("nan")
    return (f1 * f1) / (2.0 * f2)


def good_turing_interval(f1: int, f2: int, n: int, *, level: float = 0.95) -> tuple[float, float]:
    """Esty's interval for `f1/n`, the one interval Good-Turing has and did not have here.

    Esty (1983), *A Normal Limit Law for a Nonparametric Estimator of the Coverage of a Random
    Sample*, `Ann. Statist.` 11(3), 905-912, gives the asymptotic variance

        Var(f1/n) = (f1 + 2*f2 - f1**2 / n) / n**2

    and normality, so the interval is the ordinary symmetric one clipped to [0, 1]. It needs no
    assumption about how many families exist, which is the same thing that makes `f1/n` itself
    assumption-free.

    The reason this matters more than a missing interval usually does: on a log of 100 finds with
    `f1 = 7` and `f2 = 3` the point estimate is 0.070 and the interval is [0.0007, 0.1393]. That
    spans a factor of two hundred and it is most of the range the statistic could plausibly take.
    A team deciding whether to keep fuzzing on a novelty probability of 0.070 is deciding on a
    number that the same data would let be 0.14.
    """
    if n <= 0:
        return float("nan"), float("nan")
    from scipy.stats import norm

    p = f1 / n
    var = (f1 + 2.0 * f2 - (f1 * f1) / n) / (n * n)
    if var < 0.0:
        # Reachable when f1 is large relative to n and f2 is small, where the asymptotic form runs
        # out. Returning a NaN interval says the variance estimate is unavailable; returning zero
        # would say the estimate is exact.
        return float("nan"), float("nan")
    half = float(norm.ppf(0.5 * (1.0 + level))) * math.sqrt(var)
    return max(0.0, p - half), min(1.0, p + half)


def chao1_interval(f1: int, f2: int, *, level: float = 0.95) -> tuple[float, float]:
    """A log-normal interval for `f0_hat = f1**2/(2*f2)`, from Chao's variance.

    Chao (1987), *Biometrics* 43(4), 783-791. For the abundance-based estimator the variance of the
    unseen count is

        Var(f0_hat) = f2 * [ (f1/f2)**2 / 2 + (f1/f2)**3 + (f1/f2)**4 / 4 ]

    and the interval is built on the log scale, `[f0/K, f0*K]` with
    `K = exp(z * sqrt(log(1 + Var/f0**2)))`, which is what the ecology literature uses and which
    keeps the lower end above zero. A symmetric interval on a right-skewed count goes negative at
    the sample sizes an exploit log has, and a negative bound on a number of missing families is
    the kind of output that makes a reader stop trusting the whole reading.

    **This interval is about sampling variation in `f1` and `f2` and not about Chao1's own bias.**
    Chao1 is a lower bound on richness and the bound is loose when abundances are uneven, which is
    the regime an exploit log is in: a family the search cannot generate at all contributes to
    neither `f1` nor `f2`. Simulated on communities with gamma-distributed abundances, the interval
    contains the true unseen count 0.847 of the time at moderate unevenness and 0.611 at high
    unevenness, and the misses are one-sided: the truth sits above the interval 0.151 and 0.389 of
    the time respectively and below it 0.003 and 0.000. So read the interval as the precision of a
    floor, not as a range the answer is inside.
    """
    if f1 <= 0 or f2 <= 0:
        return float("nan"), float("nan")
    from scipy.stats import norm

    r = f1 / f2
    var = f2 * (0.5 * r * r + r**3 + 0.25 * r**4)
    f0 = (f1 * f1) / (2.0 * f2)
    k = math.exp(float(norm.ppf(0.5 * (1.0 + level))) * math.sqrt(math.log1p(var / (f0 * f0))))
    return f0 / k, f0 * k


def chao1_unseen_bias_corrected(f1: int) -> float:
    """`f1*(f1-1)/2`, the form that survives `f2 = 0`. A lower bound, and known to be one.

    Chao's bias-corrected estimator is what the ecology literature reaches for when the doubleton
    class is empty. It is biased low, which is the right direction for a refusal's partial answer:
    it says "at least this many are missing" rather than guessing at how many more.
    """
    return 0.0 if f1 <= 1 else f1 * (f1 - 1) / 2.0


def expected_doubletons(m: int, richness: float) -> float:
    """`E[f2]` after `m` draws from `S` equiprobable families: `S * C(m,2) * p**2 * (1-p)**(m-2)`.

    The exact binomial form rather than the usual Poisson approximation `(m**2/2S)exp(-m/S)`,
    because the logs this is used on are short and the approximation is only good for large `S`.
    At `S = 3` and `m = 8` the two differ by about ten percent, and the answer they are feeding is
    a remedy telling somebody how much longer to search.
    """
    if richness < 2 or m < 2:
        return 0.0
    p = 1.0 / richness
    return richness * (m * (m - 1) / 2.0) * p * p * math.pow(1.0 - p, m - 2)


def observations_for_a_doubleton(n: int, f1: int, s_obs: int) -> int:
    """How many more finds the log needs before `f2 = 0` should stop being the answer.

    A remedy has to name a number or it is not a remedy, and the number here comes from an
    explicitly stated model rather than from the data alone, because at `f2 = 0` the data contains
    no information about repeat rates by construction. The model is equiprobable families, and
    this walks `m` up from the log's own length to the first point where `E[f2]` reaches one.

    Two things make it more than one line. The richness the model needs is not observable when
    nothing has repeated at all (`f1 == n` makes `S_obs` equal `n` and useless as a proxy), so the
    birthday bound supplies a floor there: seeing no collision in `n` draws implies `S` is at
    least about `2*n**2/pi`. And `E[f2]` is **not monotone**, because `f2` counts families seen
    *exactly* twice and they keep moving to three. On a short log over few families the curve can
    peak below one and fall, and the walk would then run forever looking for a crossing that is
    not there. When that happens this falls back to the doubling rule the capture-recapture
    literature uses in the same situation: search again as long as you have searched so far.

    The number is a floor and reads as one. It says when to look again, not when the answer will
    be there.
    """
    if n <= 0:
        return 1
    richness = max(float(s_obs), 2.0 * n * n / math.pi) if f1 >= n else float(max(s_obs, 2))
    ceiling = 4 * max(int(richness), n) + 8
    best = 0.0
    m = max(n, 2)
    while m <= ceiling:
        expected = expected_doubletons(m, richness)
        if expected >= 1.0:
            return max(1, m - n)
        if expected < best:
            break
        best = expected
        m += 1
    return max(1, n)


# ---------------------------------------------------------------------------
# Crow-AMSAA
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class CrowFit:
    """The Duane fit: `N(t) = lambda * t**beta`, with the interval that decides what it means."""

    beta: float
    lam: float
    ci_low: float
    ci_high: float
    ci_level: float
    points: int
    r_squared: float
    fit_on: str
    #: The Crow maximum-likelihood exponent `m / sum(log(T/t_i))`, computed as a cross-check.
    #: Log-log least squares is what this instrument reports; the MLE is what MIL-HDBK-189 and
    #: every reliability text actually use, because the cumulative points a Duane plot regresses
    #: are not independent. Where the two disagree materially, the least-squares number is the one
    #: to distrust.
    beta_mle: float | None = None
    #: Crow's bias-corrected exponent, `(surviving terms - 1) / sum`. **This is the one to read at
    #: the sample sizes an exploit log has.** The MLE is badly biased upward there: against a
    #: planted 0.700 it recovers 0.922 at eight failures where this recovers 0.692. Both are
    #: reported because MIL-HDBK-189 gives both and a reader looking for the MLE should find it.
    beta_unbiased: float | None = None

    @property
    def converging(self) -> bool:
        """`beta < 1`: the discovery rate is falling. The point estimate, not the claim."""
        return self.beta < 1.0

    @property
    def converging_at_interval(self) -> bool | None:
        """Whether the *interval* settles it. None when it spans 1, which is the usual answer."""
        if self.ci_high < 1.0:
            return True
        if self.ci_low > 1.0:
            return False
        return None

    @property
    def crow_beta(self) -> float | None:
        """The Crow estimator to read: bias-corrected if it exists, else the raw MLE.

        Explicitly None-checked rather than written `beta_unbiased or beta_mle`, because a
        bias-corrected exponent of exactly 0.0 is falsy and that expression would silently fall
        through to the MLE on the one log where the correction says the discovery rate has stopped
        dead.
        """
        return self.beta_mle if self.beta_unbiased is None else self.beta_unbiased

    def render(self) -> str:
        verdict = {
            True: "the blacklist is converging",
            False: "finds are still accelerating",
            None: "the interval spans 1, so this does not yet settle whether it is converging",
        }[self.converging_at_interval]
        lines = [
            f"Crow-AMSAA beta = {self.beta:.3f} "
            f"[{self.ci_low:.3f}, {self.ci_high:.3f}] at {self.ci_level:.0%}, "
            f"{self.points} points on cumulative {self.fit_on}: {verdict}"
        ]
        # The caveat belongs here rather than only on the coverage reading, and that is the point
        # of moving it. The least-squares interval regresses cumulative counts, which are heavily
        # autocorrelated, so it is far narrower than its nominal level. Simulated on a homogeneous
        # Poisson process, where beta is exactly 1, this interval contains the truth in 0.401 /
        # 0.308 / 0.219 of runs at 8 / 12 / 25 finds against a nominal 0.95, and it declares the
        # blacklist *converging* in 0.462 / 0.533 / 0.574 of those runs where a one-sided 2.5%
        # claim should. The caveat was once carried on `ExploitCoverage.render` only, and
        # `ReliabilityGrowth` is the instrument whose entire quantity is beta.
        lines.append(
            "    that interval is narrower than its own level: on a process with beta exactly 1 "
            "it contains the truth in 0.22 to 0.40 of runs at 8 to 25 finds against a nominal "
            "0.95. "
            + (
                "It calls such a process converging in about half of them, so read the verdict "
                "above as the fit's precision rather than as the evidence."
                if self.converging_at_interval is not None
                else "The honest interval is wider still, so an interval that already spans 1 "
                "spans it under any correction."
            )
        )
        crow = self.crow_beta
        if crow is not None:
            lines.append(
                f"    Crow beta = {crow:.3f} "
                + ("bias-corrected" if self.beta_unbiased is not None else "raw MLE")
                + (
                    f", raw MLE {self.beta_mle:.3f}"
                    if self.beta_unbiased is not None and self.beta_mle is not None
                    else ""
                )
                + ". This is the point-process estimator and it is the one to trust at these "
                "sample sizes."
            )
            if (crow < 1.0) != (self.beta < 1.0):
                lines.append(
                    f"    the two estimators disagree about direction: least squares reads "
                    f"{self.beta:.3f} and Crow reads {crow:.3f}. Crow is the one to trust here, "
                    f"because the cumulative points a Duane plot regresses are not independent "
                    f"observations and its interval is too narrow in consequence."
                )
        return "\n".join(lines)


def crow_amsaa(
    t: Sequence[float],
    n: Sequence[int],
    *,
    ci_level: float = 0.95,
    fit_on: str = "families",
    total_effort: float | None = None,
) -> CrowFit | Refusal:
    """Least squares on `log N = log lambda + beta log t`, with a Student-t interval on `beta`.

    The interval is the textbook one for a simple regression: `se(beta) = s / sqrt(Sxx)` with
    `s**2 = RSS/(m-2)`, and `beta +- t(m-2, (1+level)/2) * se`. It is an interval on the *fit*, and
    it under-states the real uncertainty because the cumulative points being regressed are not
    independent observations. That is recorded in the estimator's bias statement rather than
    corrected here, because correcting it properly means fitting the point process instead, which
    is what ``beta_mle`` is.

    Refuses rather than returning a number when the fit would be an artifact of the axis: two
    points fit a line exactly, non-positive efforts have no logarithm, and a constant `t` has no
    slope.
    """
    import numpy as np

    ts = np.asarray(t, dtype=float)
    ns = np.asarray(n, dtype=float)
    if ts.size != ns.size:
        raise ValueError(f"the effort axis has {ts.size} points and the count axis {ns.size}")

    keep = (ts > 0) & (ns > 0) & np.isfinite(ts) & np.isfinite(ns)
    dropped = int(ts.size - keep.sum())
    ts, ns = ts[keep], ns[keep]
    m = int(ts.size)

    if m < MIN_FIT_POINTS:
        detail = f"{m} usable point{'s' if m != 1 else ''} on the growth curve" + (
            f" ({dropped} dropped for a non-positive effort or count)" if dropped else ""
        )
        if m < 2:
            return Refusal(
                instrument="ReliabilityGrowth",
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=f"{detail}; a power law cannot be fitted to fewer than two",
                remedy=(
                    f"supply a log with at least {MIN_FIT_POINTS} finds at distinct positive "
                    f"cumulative efforts. Below that there is no growth curve, only a point, and "
                    f"the raw family count is the whole of what the log supports."
                ),
                statistics={"points": m, "dropped": dropped},
            )
        # Two points determine beta exactly and leave no residual, so there is a slope and no way
        # to say how well it is determined. The slope is the honest partial answer.
        slope, intercept = _loglog_slope(np.log(ts), np.log(ns))
        return bounded_refusal(
            "ReliabilityGrowth",
            RefusalReason.ABOVE_LOD_BELOW_LOQ,
            detail=(
                f"{detail}; two points fit a line exactly, so beta = {slope:.3f} has no residual "
                f"and no interval"
            ),
            remedy=(
                f"supply at least {MIN_FIT_POINTS} finds. The specification asks for the interval "
                f"on beta, and an interval needs a residual degree of freedom, which arrives with "
                f"the third point."
            ),
            bound=_bare_evidence(float(slope)),
            beta_point_estimate=float(slope),
            lam=float(math.exp(intercept)),
            points=m,
        )

    x = np.log(ts)
    y = np.log(ns)
    if float(np.ptp(x)) <= 0.0:
        return Refusal(
            instrument="ReliabilityGrowth",
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail=f"every one of the {m} finds is recorded at the same cumulative effort {ts[0]:g}",
            remedy=(
                "record the cumulative test effort at each find. A growth curve needs an axis "
                "that moves; with a constant one there is no rate to estimate, and the log "
                "supports the raw family count and nothing more."
            ),
            statistics={"points": m, "effort": float(ts[0])},
        )

    slope, intercept = _loglog_slope(x, y)
    resid = y - (intercept + slope * x)
    rss = float(resid @ resid)
    sxx = float(((x - x.mean()) ** 2).sum())
    dof = m - 2
    s2 = rss / dof
    se = math.sqrt(s2 / sxx) if sxx > 0 else float("inf")

    from scipy.stats import t as student_t

    half = float(student_t.ppf(0.5 * (1.0 + ci_level), dof)) * se
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")

    # Crow's estimator, and the censoring scheme decides its numerator. With `total_effort` the
    # test is time-truncated, the horizon sits beyond every failure, all n terms survive the filter
    # and the MLE is `n / sum`. Without it the horizon *is* the last failure, that term contributes
    # log(1) = 0 and is filtered out, so the sum runs to n-1 while the MLE's numerator stays n.
    # This used to report `len(ratios) / sum`, which is the MLE in the first case and one short of
    # it in the second, and `total_effort` defaults to None so the second is the path most logs
    # take.
    mle: float | None = None
    unbiased: float | None = None
    horizon = total_effort if total_effort is not None else float(ts[-1])
    if horizon > 0:
        ratios = [math.log(horizon / ti) for ti in ts if 0 < ti < horizon]
        total = sum(ratios)
        if ratios and total > 0:
            mle = len(ts) / total
            # And the MLE is badly biased upward at the sample sizes an exploit log has: 0.92
            # against a true 0.70 at eight failures, 0.84 at twelve. Crow's bias correction drops
            # the numerator by one **relative to the surviving terms**, which is `(n-1)/sum` when
            # time-truncated and `(n-2)/sum` when failure-truncated, and both are `(len(ratios) -
            # 1) / sum`. Measured recovery of a planted 0.700 over 6,000 simulated processes:
            # 0.692 / 0.699 / 0.703 at n = 8 / 12 / 25, against the MLE's 0.922 / 0.838 / 0.764.
            if len(ratios) > 1:
                unbiased = (len(ratios) - 1) / total

    return CrowFit(
        beta=float(slope),
        lam=float(math.exp(intercept)),
        ci_low=float(slope - half),
        ci_high=float(slope + half),
        ci_level=ci_level,
        points=m,
        r_squared=r2,
        fit_on=fit_on,
        beta_mle=mle,
        beta_unbiased=unbiased,
    )


def _loglog_slope(x: Any, y: Any) -> tuple[float, float]:
    """Ordinary least squares by the closed form, so the arithmetic is checkable by hand."""
    xm = float(x.mean())
    ym = float(y.mean())
    sxx = float(((x - xm) ** 2).sum())
    sxy = float(((x - xm) * (y - ym)).sum())
    slope = sxy / sxx if sxx else float("nan")
    return slope, ym - slope * xm


def _bare_evidence(value: float) -> Any:
    """An `Evidence` for a refusal's `partial` field, outside any instrument's context.

    `Refusal.partial` is typed as an `Evidence`, and the bound here is produced inside a pure
    function that has no `Context` to emit from. `make_evidence` is the constructor the kernel
    exposes for exactly that; the trust level it assigns is EXPLORATORY, which is correct for a
    bound attached to a refusal.
    """
    from reward_lens.core.evidence import SubjectRef, make_evidence

    return make_evidence(
        observable="ReliabilityGrowth",
        observable_version="1.0",
        subject=SubjectRef(signals=(), readout="exploit_log"),
        value=value,
    )


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class ExploitCoverage:
    """What the log says about the exploit space. The value of both of D6's quantities.

    ``n_families`` on its own is the catalogue's mandatory baseline, and it is worth stating why
    that baseline is not a formality here. "We have found 23 families" is what every team already
    reports. The claim this instrument adds is that the 23 is nearly all of them, or is not, and
    the baseline is the number the claim has to beat by being more informative than a count.
    """

    source: str
    effort_unit: str
    rung: int
    n_finds: int
    n_families: int
    f1: int
    f2: int
    novelty_probability: float
    unseen_families: float
    unseen_fraction: float
    unseen_fraction_of_observed: float
    #: Esty's 95% interval on `novelty_probability`. Good-Turing shipped without one and the
    #: interval is most of the unit interval at the sample sizes an exploit log has.
    novelty_ci: tuple[float, float] = (float("nan"), float("nan"))
    #: The Chao1 interval, mapped onto the same fraction as `unseen_fraction`. Sampling variation
    #: in f1 and f2 only; Chao1's downward bias is not in it. See `chao1_interval`.
    unseen_fraction_ci: tuple[float, float] = (float("nan"), float("nan"))
    #: The Chao1 interval on the unseen *count*, in families.
    unseen_families_ci: tuple[float, float] = (float("nan"), float("nan"))
    ci_level: float = 0.95
    chao1: float | None = None
    fit: CrowFit | None = None
    family_counts: Mapping[str, int] = field(default_factory=dict)
    spectrum: Mapping[int, int] = field(default_factory=dict)
    total_effort: float | None = None
    notes: tuple[str, ...] = ()

    @property
    def headline(self) -> float:
        """`verifier.unseen_exploit_mass`: the fraction of the families still unseen.

        Not called ``value``. `Evidence.value` is this whole payload, and a payload carrying its
        own ``value`` makes every helper that walks `Evidence.value` ambiguous.
        """
        return self.unseen_fraction

    @property
    def beta(self) -> float | None:
        """`verifier.reliability_growth_beta`, or None when the fit did not run."""
        return None if self.fit is None else self.fit.beta

    def render(self) -> str:
        lines = [
            f"You have found {self.n_families} exploit "
            f"{'family' if self.n_families == 1 else 'families'} in "
            f"{self.n_finds} {self.effort_unit}-logged finds"
            + (f" from {self.source}" if self.source else "")
            + ".",
            f"    Chao1 puts {self.unseen_fraction:.1%} of the families still unseen "
            f"[{self.unseen_fraction_ci[0]:.1%}, {self.unseen_fraction_ci[1]:.1%}] "
            f"({self.unseen_families:.2f} unseen against {self.n_families} seen), and that is a "
            f"floor rather than an estimate.",
            f"    Good-Turing puts the probability that the next find is novel at "
            f"{self.novelty_probability:.3f} "
            f"[{self.novelty_ci[0]:.3f}, {self.novelty_ci[1]:.3f}] at "
            f"{self.ci_level:.0%}, by Esty's variance.",
            f"    f1 = {self.f1}, f2 = {self.f2}, n = {self.n_finds}",
        ]
        if self.chao1 is not None:
            lines.append(f"    Chao1 richness estimate {self.chao1:.2f}")
        if self.fit is not None:
            # One rendering of the fit, indented, so `ReliabilityGrowth` and this reading print the
            # same caveats. They used to differ, and the instrument whose headline is beta was the
            # one that printed fewer.
            lines += [f"    {line.strip()}" for line in self.fit.render().splitlines()]
        lines.append(f"    baseline (the raw count of families found): {self.n_families}")
        lines += [f"    note: {n}" for n in self.notes]
        return "\n".join(lines)


def exploit_coverage(
    log: ExploitLog,
    *,
    rung: int = 2,
    fit_on: FitOn = "families",
    ci_level: float = 0.95,
) -> ExploitCoverage | Refusal:
    """D6's arithmetic, end to end. Returns the reading or the refusal that replaces it.

    Rungs, from the catalogue: 0 is `f1/n` and `f1**2/(2*f2)`, 1 adds Crow-AMSAA, 2 adds Chao1.
    A lower rung is a smaller answer rather than a worse one, so asking for rung 0 on a log too
    short to fit a growth curve is the right move and returns evidence rather than a refusal.
    """
    if log.n == 0:
        return Refusal(
            instrument="ExploitFamilyCoverage",
            reason=RefusalReason.ACCESS_INSUFFICIENT,
            detail="the exploit log is empty, so there is no frequency spectrum to read",
            remedy=(
                "supply a log with at least one find. D6 is arithmetic on a log of exploit "
                "families your search has already turned up; with no finds there is no evidence "
                "about the unseen ones, and an empty log is not a claim that the grader is clean."
            ),
            statistics={"n": 0},
        )

    spectrum = log.spectrum()
    f1 = spectrum.get(1, 0)
    f2 = spectrum.get(2, 0)
    s_obs = log.s_obs
    notes: list[str] = []

    unseen = chao1_unseen(f1, f2)
    if math.isnan(unseen):
        needed = observations_for_a_doubleton(log.n, f1, s_obs)
        bound = chao1_unseen_bias_corrected(f1)
        return bounded_refusal(
            "ExploitFamilyCoverage",
            RefusalReason.ABOVE_LOD_BELOW_LOQ,
            detail=(
                f"f2 = 0: no family in this log of {log.n} finds appears exactly twice, so "
                f"f1**2/(2*f2) is undefined. f1 = {f1} over {s_obs} families found."
            ),
            remedy=(
                f"collect about {needed} more find{'s' if needed != 1 else ''} under the same "
                f"search and re-run. The doubleton class is what Chao1 divides by, and an "
                f"equiprobable-family model says a log this length should populate it after "
                f"roughly that much more searching. The bias-corrected bound f1*(f1-1)/2 = "
                f"{bound:.1f} unseen families is attached and is a floor, not an estimate. "
                f"Good-Turing's novelty probability f1/n = "
                f"{novelty_probability(f1, log.n):.3f} needs no doubletons and is available now "
                f"at rung 0."
            ),
            bound=_bare_evidence(bound),
            f1=f1,
            f2=f2,
            n=log.n,
            s_obs=s_obs,
            novelty_probability=novelty_probability(f1, log.n),
            additional_finds_needed=needed,
        )

    if f1 == 0:
        notes.append(
            "f1 = 0: every family in this log has been found more than once, so the estimated "
            "unseen mass is zero. That is a real reading and it is also the shape a log takes "
            "when the search stopped turning up new things some time ago; check whether the "
            "search changed before reading it as coverage."
        )

    chao1_value = s_obs + unseen if rung >= 2 else None
    f0_lo, f0_hi = chao1_interval(f1, f2, level=ci_level)
    frac_lo = (
        f0_lo / (s_obs + f0_lo) if math.isfinite(f0_lo) and (s_obs + f0_lo) > 0 else float("nan")
    )
    frac_hi = (
        f0_hi / (s_obs + f0_hi) if math.isfinite(f0_hi) and (s_obs + f0_hi) > 0 else float("nan")
    )

    fit: CrowFit | None = None
    if rung >= 1:
        if fit_on == "finds" and not log.has_effort:
            notes.append(
                "the Crow-AMSAA fit was not run on cumulative finds: this log carries no measured "
                "effort, so the only axis available is the find ordinal, and regressing cumulative "
                "finds on the find ordinal is fitting N(t) = t. It returns beta = 1 for every log "
                "ever written, which is the axis reading itself back rather than a measurement."
            )
        else:
            t, n = log.discovery_curve(fit_on)
            outcome = crow_amsaa(
                t, n, ci_level=ci_level, fit_on=fit_on, total_effort=log.total_effort
            )
            if isinstance(outcome, Refusal):
                notes.append(f"no growth fit: {outcome.detail}")
            else:
                fit = outcome
                if not log.has_effort:
                    notes.append(
                        "the growth fit is against the find ordinal rather than measured effort, "
                        "because the log carries none. beta is then a statement about finds per "
                        "find rather than finds per unit of search, which is weaker: record the "
                        "cumulative rollouts or trials at each find to make it the number the "
                        "catalogue asks for."
                    )

    return ExploitCoverage(
        source=log.source,
        effort_unit=log.effort_unit,
        rung=rung,
        n_finds=log.n,
        n_families=s_obs,
        f1=f1,
        f2=f2,
        novelty_probability=novelty_probability(f1, log.n),
        unseen_families=unseen,
        unseen_fraction=unseen / (s_obs + unseen) if (s_obs + unseen) > 0 else 0.0,
        unseen_fraction_of_observed=unseen / s_obs if s_obs else float("nan"),
        novelty_ci=good_turing_interval(f1, f2, log.n, level=ci_level),
        unseen_fraction_ci=(frac_lo, frac_hi),
        unseen_families_ci=(f0_lo, f0_hi),
        ci_level=ci_level,
        chao1=chao1_value,
        fit=fit,
        family_counts=log.counts(),
        spectrum=spectrum,
        total_effort=log.total_effort,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# The instruments
# ---------------------------------------------------------------------------

#: A log is a census of finds and the arithmetic on it is unconditional in the regime sense, with
#: one exception that is not a formality: Good-Turing and Crow-AMSAA both count families *of one
#: grader*. If the grader was patched inside the observation window, half the log describes a
#: program the other half does not, and the frequency spectrum mixes two populations. That is what
#: `STATIONARY_GRADER` says, and D10 is what measures it.
D6_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
    measured_by={RegimeCondition.STATIONARY_GRADER: "env.replay_fidelity"},
)

#: An exploit log is a record, and reading it needs nothing else. No source, no calls, no weights.
#: That is what makes this the best effort-to-impact ratio in the catalogue: every lab running
#: code-based reward already has the input.
ACCESS_LOG_ONLY: AccessMatrix = {Component.RECORD: Access.RECORD}

_D6_SUBSTRATES = frozenset(Substrate)
_D6_PHASES = frozenset({Phase.PRE_RUN, Phase.IN_RUN, Phase.POST_RUN, Phase.DEPLOYED})

_LOG_DEVIATION = (
    "the log is taken as given. This instrument cannot see whether two entries labelled with the "
    "same family are the same exploit, or whether the search that produced them changed partway "
    "through, and both make the frequency spectrum mean something else."
)


class _LogObservable(BaseObservable):
    """What D6's two instruments share: a log, a rung, and a context built from the log."""

    capabilities = Capability.SCORES
    requires = ACCESS_LOG_ONLY
    substrates = _D6_SUBSTRATES
    phases = _D6_PHASES
    envelope = D6_ENVELOPE
    invariance = "none"
    invariance_relation = Relation("invariant")
    gauge_status = GaugeStatus.INVARIANT
    faithful_to: str | None = None

    def __init__(
        self,
        log: ExploitLog | None = None,
        *,
        rung: int | None = None,
        fit_on: FitOn = "families",
        ci_level: float = 0.95,
    ) -> None:
        ensure_quantities()
        self.log = log
        if rung is not None:
            self.rung = rung
        self.fit_on: FitOn = fit_on
        self.ci_level = ci_level

    @property
    def subject(self) -> LogSubject:
        if self.log is None:
            raise ValueError(f"{self.name} was constructed without a log")
        return LogSubject(self.log)

    def _resolve(self, ctx: Context) -> ExploitLog | Refusal:
        log = self.log
        if log is None:
            log = getattr(ctx.view, "log", None) or (
                ctx.view if isinstance(ctx.view, ExploitLog) else None
            )
        if log is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no exploit log was supplied, on the context or to the constructor",
                remedy=(
                    "pass `log=ExploitLog.of([...])` here, or set `ctx.view` to the log. D6 reads "
                    "a log of exploit families and nothing else; without one there is no "
                    "frequency spectrum and no growth curve."
                ),
            )
        return log

    def estimate(self, ctx: Context | None = None) -> Reading:
        if ctx is None:
            # `Context.signal` is typed `RewardSignal` and an exploit log is not one. The ignore is
            # the visible form of the kernel gap every PROGRAM-substrate instrument in this series
            # carries; see the same line in `verifier.program_context`.
            ctx = Context(
                signal=self.subject,  # type: ignore[arg-type]
                view=self.log,
                readout="exploit_log",
            )
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        return run(self, ctx)


class ExploitFamilyCoverage(_LogObservable):
    """D6 `verifier.unseen_exploit_mass`: how much of the exploit space is still unseen.

    Kill condition, from the catalogue: **nothing; it is arithmetic on a log.** The honest reading
    of an unkillable instrument is that it is unfalsifiable rather than robust, so the practical
    failure worth watching is the one the ancestor catalogue named: if `f1` is zero on every real
    log, every family having been seen at least twice, the estimator is undefined in practice and
    the reading collapses to the baseline count it was supposed to improve on.
    """

    name = "ExploitFamilyCoverage"
    version = "1.0"
    quantity = "verifier.unseen_exploit_mass"
    baselines = ("the raw count of families found",)
    rung = 2
    deviations = (
        _LOG_DEVIATION,
        "`unseen_exploit_mass` is reported as the Chao1 richness fraction f0/(S_obs+f0), not as "
        "Good-Turing's unseen probability mass f1/n. The registry's `definition` field is OPEN and "
        "the specification's own example sentence quotes two different numbers for the two, so the "
        "alternative denominator f0/S_obs is carried on the reading rather than chosen silently.",
    )

    def measure(self, ctx: Context) -> Any:
        log = self._resolve(ctx)
        if isinstance(log, Refusal):
            return log
        result = exploit_coverage(log, rung=self.rung, fit_on=self.fit_on, ci_level=self.ci_level)
        if isinstance(result, Refusal):
            return result
        return ctx.emit(
            result,
            # The interval is the headline's own, which is Chao1's on the unseen *fraction*, not
            # Good-Turing's on the novelty probability. Those are a count and a probability and
            # they are not interchangeable; the Esty interval travels on the payload beside the
            # statistic it belongs to.
            uncertainty=Uncertainty(
                n=result.n_finds,
                ci_low=result.unseen_fraction_ci[0],
                ci_high=result.unseen_fraction_ci[1],
                ci_level=result.ci_level,
                method=(
                    "Chao1 unseen fraction, log-normal interval from Chao (1987); sampling "
                    "variation only, the estimator's downward bias is not in it"
                ),
            ),
            subject_extra={"baseline_family_count": str(result.n_families)},
        )


class ReliabilityGrowth(_LogObservable):
    """D6 `verifier.reliability_growth_beta`: whether the blacklist is converging.

    Kill condition, from the catalogue: **nothing; it is arithmetic on a log.** What would make
    this number worthless in practice is reporting it without its interval. On the logs that
    actually exist, a handful of families over a few dozen finds, the 95% interval on `beta`
    usually spans 1, and "beta = 0.74" printed alone converts "we cannot tell yet" into "it is
    converging" with no evidence having changed.
    """

    name = "ReliabilityGrowth"
    version = "1.0"
    quantity = "verifier.reliability_growth_beta"
    baselines = ("the raw count of families found",)
    rung = 1
    deviations = (
        _LOG_DEVIATION,
        "beta is fitted by least squares on the log-log plot, which is the Duane convention. The "
        "points are cumulative and therefore not independent, so the fitted interval is narrower "
        "than the truth; the Crow maximum-likelihood exponent is computed alongside as the "
        "cross-check and is on the reading.",
    )

    def measure(self, ctx: Context) -> Any:
        log = self._resolve(ctx)
        if isinstance(log, Refusal):
            return log
        if self.fit_on == "finds" and not log.has_effort:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail=(
                    f"the log's {log.n} finds carry no measured effort, so cumulative finds would "
                    f"be regressed on the find ordinal, which is fitting N(t) = t"
                ),
                remedy=(
                    "record the cumulative test effort at each find and pass it as "
                    "`ExploitFind.effort`, or fit on cumulative families instead "
                    "(`fit_on='families'`), where the ordinal axis is not degenerate because "
                    "families accumulate more slowly than finds."
                ),
                statistics={"n": log.n, "has_effort": False},
            )
        t, n = log.discovery_curve(self.fit_on)
        outcome = crow_amsaa(
            t, n, ci_level=self.ci_level, fit_on=self.fit_on, total_effort=log.total_effort
        )
        if isinstance(outcome, Refusal):
            return outcome
        return ctx.emit(
            outcome,
            uncertainty=Uncertainty(
                ci_low=outcome.ci_low,
                ci_high=outcome.ci_high,
                ci_level=outcome.ci_level,
                n=outcome.points,
                method="log-log least squares, Student-t on the fitted slope",
            ),
            subject_extra={"baseline_family_count": str(log.s_obs)},
        )


def exploit_family_coverage(log: ExploitLog, **kwargs: Any) -> Reading:
    """Run D6's unseen-mass half and return the Reading. The one-call form, for a card renderer."""
    return ExploitFamilyCoverage(log, **kwargs).estimate()


def reliability_growth(log: ExploitLog, **kwargs: Any) -> Reading:
    """Run D6's growth half and return the Reading."""
    return ReliabilityGrowth(log, **kwargs).estimate()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _register() -> None:
    """D6's ladder, so `capability_report` knows which rungs an exploit log reaches.

    Three estimators for `verifier.unseen_exploit_mass` and one for
    `verifier.reliability_growth_beta`, which is what `spec/QUANTITIES.yaml` declares: `rungs: 3`
    and `rungs: 1`. The catalogue's ladder puts beta at rung 1 and Chao1 at rung 2, so beta is
    carried by a single estimator that rung 2 does not replace.
    """
    ensure_quantities()
    for rung, impl, what in (
        (0, "verifier.unseen_exploit_mass.good_turing", "f1/n and f1**2/(2*f2)"),
        (1, "verifier.unseen_exploit_mass.growth", "adds the Crow-AMSAA growth exponent"),
        (2, "verifier.unseen_exploit_mass.chao1", "adds the Chao1 richness estimate"),
    ):
        register_estimator(
            EstimatorEntry(
                quantity="verifier.unseen_exploit_mass",
                impl=impl,
                requires=ACCESS_LOG_ONLY,
                envelope=D6_ENVELOPE,
                rung=rung,
                bias=BiasStatement(
                    direction="downward",
                    why=(
                        f"{what}. Chao1 is a lower bound on richness and is known to be one: it "
                        f"uses only the singleton and doubleton classes, so a family whose "
                        f"instances the search cannot generate at all contributes nothing to "
                        f"either. The unseen fraction it produces is a floor."
                    ),
                ),
                cost=CostModel(note="one pass over a log; no grader call, no GPU"),
                substrates=_D6_SUBSTRATES,
                phases=_D6_PHASES,
                run=None,
            )
        )
    register_estimator(
        EstimatorEntry(
            quantity="verifier.reliability_growth_beta",
            impl="verifier.reliability_growth_beta.duane_ols",
            requires=ACCESS_LOG_ONLY,
            envelope=D6_ENVELOPE,
            rung=1,
            bias=BiasStatement(
                direction="unknown",
                why=(
                    "least squares on a Duane plot regresses cumulative counts, which are not "
                    "independent across points. The slope's sign is robust; the interval around it "
                    "is narrower than the truth, and by how much depends on the log. The Crow "
                    "maximum-likelihood exponent is reported beside it as the cross-check."
                ),
            ),
            cost=CostModel(note="one least-squares fit over at most one point per find"),
            substrates=_D6_SUBSTRATES,
            phases=_D6_PHASES,
            run=None,
        )
    )


_register()


__all__ = [
    "ACCESS_LOG_ONLY",
    "D6_ENVELOPE",
    "MIN_FIT_POINTS",
    "CrowFit",
    "ExploitCoverage",
    "ExploitFamilyCoverage",
    "ExploitFind",
    "ExploitLog",
    "FitOn",
    "LogSubject",
    "ReliabilityGrowth",
    "chao1_interval",
    "chao1_unseen",
    "chao1_unseen_bias_corrected",
    "crow_amsaa",
    "good_turing_interval",
    "exploit_coverage",
    "expected_doubletons",
    "exploit_family_coverage",
    "novelty_probability",
    "observations_for_a_doubleton",
    "reliability_growth",
]

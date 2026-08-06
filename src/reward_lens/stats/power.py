"""M10, power and the minimum detectable effect. Before the run, not after.

A power calculation done after a null result is a rationalisation. Done before the run it is the
cheapest thing in this library: it costs no GPU, it costs no grader calls, and it tells you
whether the experiment you are about to pay for can answer the question you are about to ask.

**Everything here is validated against simulation, not against a formula, and that is a
requirement rather than a preference.** Three of the five standard calculators a practitioner
would reach for are roughly 2x wrong for close paired model comparisons, because they treat two
models scored on the *same* benchmark items as two independent samples. `compare_calculators`
runs all five against a Monte Carlo of the actual test and reports each one's ratio, so the claim
is a measurement this module makes rather than a warning it repeats. On the shipped default
scenario at a per-item correctness correlation of 0.5 the three uncorrected calculators come out
near 2x; at 0.8 they come out near 5x. Run it yourself with `compare_calculators()`.

**The resolution ratio.** `q = N / N*` is the number that says outright when a leaderboard row is
not resolved. A row with q = 0.4 is not "a close call between two models", it is an experiment
that could not have separated them, and `Resolution.render` prints NOT RESOLVED rather than a
verdict.

**The detection band.** Position bias is statistically detectable only within a window of base
accuracy, and above that window the effect in probability space collapses even though the
underlying bias is unchanged. So **absence of signal above the band reads as "not measurable", not
as "unbiased"**, and `DetectionBand.interpret` will not return the second sentence for a design
that could not have produced the first. The published figure for that window is roughly 60% to
95% base accuracy; this module does not assert it, it computes the band for the design in front of
it with `detection_band` and lets the two be compared.

Attribution: the "three of five calculators" finding, the 60% to 95% window, and the observation
that several refuted cards in this library's own history sit above the band are published or prior
findings; the numbers this module prints are its own simulations of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from reward_lens.core.quantity import DIMENSIONLESS, QuantityID, Unit
from reward_lens.core.reading import Refusal, RefusalReason
from reward_lens.stats.ess import effective_sample_size

#: The conventions, stated rather than implied. Both are arguments everywhere below; these are the
#: values a caller gets by not choosing, and a card that used something else has to say so.
ALPHA = 0.05
TARGET_POWER = 0.80

Tails = Literal[1, 2]


def _z(p: float) -> float:
    """The standard normal quantile, via the inverse error function."""
    from scipy.special import ndtri

    return float(ndtri(p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# The design
# ---------------------------------------------------------------------------


def rho_bounds(p_a: float, p_b: float) -> tuple[float, float]:
    """The achievable correlation between two Bernoullis with these marginals (Fréchet).

    Two 0/1 variables cannot be arbitrarily correlated once their means are fixed. Handing a power
    calculator a correlation outside these bounds produces a joint distribution with a negative
    cell, and the resulting n is not wrong so much as meaningless.
    """
    qa, qb = 1.0 - p_a, 1.0 - p_b
    denom = math.sqrt(p_a * qa * p_b * qb)
    if denom <= 0:
        return (0.0, 0.0)
    lo = max(-p_a * p_b, -qa * qb) / denom
    hi = min(p_a * qb, qa * p_b) / denom
    return (lo, hi)


@dataclass(frozen=True)
class PairedBinaryDesign:
    """Two systems scored right/wrong on the same `n` items, with a stated correlation.

    ``rho`` is the per-item correlation between the two systems' correctness and it is the
    parameter every wrong calculator drops. Two close models on the same benchmark agree on the
    easy items and disagree on a handful, so rho is high, and the paired design is far more
    powerful than an independent one at the same n. Treating them as independent inflates the
    required n by roughly `1 / (1 - rho)`.

    Nobody knows rho before the run, which is the usual objection. The answer is that you know it
    within a range, you can measure it on a pilot of fifty items for nothing, and planning at
    rho = 0 is not conservative: it is a different experiment.
    """

    n: int
    accuracy_a: float
    accuracy_b: float
    rho: float = 0.0
    alpha: float = ALPHA
    tails: Tails = 2

    def __post_init__(self) -> None:
        for name in ("accuracy_a", "accuracy_b"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly inside (0, 1); got {value}")
        lo, hi = rho_bounds(self.accuracy_a, self.accuracy_b)
        if not lo - 1e-9 <= self.rho <= hi + 1e-9:
            raise ValueError(
                f"rho={self.rho:.4f} is outside the achievable range [{lo:.4f}, {hi:.4f}] for "
                f"marginals ({self.accuracy_a:.4f}, {self.accuracy_b:.4f}). Two 0/1 variables "
                f"cannot be correlated beyond what their means allow."
            )
        if self.n < 1:
            raise ValueError(f"n must be at least 1; got {self.n}")

    @property
    def delta(self) -> float:
        """The effect: how much more often B is right than A."""
        return self.accuracy_b - self.accuracy_a

    @property
    def cells(self) -> tuple[float, float, float, float]:
        """`(p11, p10, p01, p00)`: both right, A only, B only, neither."""
        p_a, p_b = self.accuracy_a, self.accuracy_b
        cov = self.rho * math.sqrt(p_a * (1 - p_a) * p_b * (1 - p_b))
        p11 = p_a * p_b + cov
        p10 = p_a - p11
        p01 = p_b - p11
        p00 = 1.0 - p11 - p10 - p01
        return (max(p11, 0.0), max(p10, 0.0), max(p01, 0.0), max(p00, 0.0))

    @property
    def discordance(self) -> float:
        """`pi_d = p10 + p01`, the fraction of items the two systems disagree on.

        This is the only part of the sample McNemar's test looks at, which is why a paired
        comparison of two close models is powered by their disagreements rather than by n.
        """
        _, p10, p01, _ = self.cells
        return p10 + p01

    def at_n(self, n: int) -> "PairedBinaryDesign":
        return PairedBinaryDesign(
            n=n,
            accuracy_a=self.accuracy_a,
            accuracy_b=self.accuracy_b,
            rho=self.rho,
            alpha=self.alpha,
            tails=self.tails,
        )

    def at_delta(self, delta: float) -> "PairedBinaryDesign":
        """The same design with a different effect, with `rho` clamped into what is achievable.

        The Fréchet bounds tighten as the two marginals separate, so holding `rho` fixed while
        searching over candidate effects makes the large candidates unconstructible rather than
        merely large, and the search then reports that no effect is detectable when the truth is
        that the search walked off the edge of the parameter space. Clamping keeps the correlation
        as close to the stated one as the marginals allow, and the clamp only binds at effects far
        larger than any MDE worth reporting.
        """
        b = min(max(self.accuracy_a + delta, 1e-6), 1.0 - 1e-6)
        lo, hi = rho_bounds(self.accuracy_a, b)
        return PairedBinaryDesign(
            n=self.n,
            accuracy_a=self.accuracy_a,
            accuracy_b=b,
            rho=min(max(self.rho, lo), hi),
            alpha=self.alpha,
            tails=self.tails,
        )


# ---------------------------------------------------------------------------
# The simulator, which is the ground truth everything else is checked against
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulatedPower:
    """An empirical rejection rate, with its own Monte Carlo error.

    ``mc_se`` is reported because a simulated power of 0.80 from 400 replicates and one from
    40,000 are different claims, and a power calculation that hides its own noise is repeating the
    mistake it exists to catch.
    """

    power: float
    mc_se: float
    replicates: int
    n: int
    mean_discordant: float
    test: str

    def render(self) -> str:
        return (
            f"simulated power {self.power:.4f} +/- {self.mc_se:.4f} at n={self.n} "
            f"({self.replicates:,} replicates, {self.test}, mean discordant pairs "
            f"{self.mean_discordant:.1f})"
        )


def mcnemar_p_values(b: np.ndarray, c: np.ndarray, tails: Tails = 2) -> np.ndarray:
    """Exact McNemar p-values from discordant counts, vectorised over replicates.

    Under the null the `b + c` discordant pairs split as a fair coin, so the p-value is a binomial
    tail. Exact rather than the chi-square approximation because the whole point of this module is
    close comparisons, and close comparisons have few discordant pairs, which is exactly where the
    approximation is worst.
    """
    from scipy.stats import binom

    m = b + c
    if tails == 2:
        k = np.minimum(b, c)
        p = 2.0 * binom.cdf(k, m, 0.5)
    else:
        p = binom.sf(b - 1, m, 0.5)
    return np.minimum(np.where(m > 0, p, 1.0), 1.0)


def simulate_power(
    design: PairedBinaryDesign,
    *,
    replicates: int = 20_000,
    seed: int = 0,
    test: Literal["exact", "normal"] = "exact",
) -> SimulatedPower:
    """Monte Carlo the actual test. This is what every formula in this module is checked against.

    Draws `replicates` datasets of `n` paired items from the design's own four-cell distribution,
    runs McNemar on each, and returns the fraction rejected at the design's alpha. No formula is
    involved, so nothing here can be 2x wrong for the reason the formulas are.
    """
    p11, p10, p01, p00 = design.cells
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(design.n, [p11, p10, p01, p00], size=replicates)
    b = counts[:, 1].astype(np.int64)
    c = counts[:, 2].astype(np.int64)
    if test == "exact":
        p = mcnemar_p_values(b, c, design.tails)
    else:
        m = b + c
        with np.errstate(divide="ignore", invalid="ignore"):
            chi = np.where(m > 0, (np.abs(b - c) - 1.0) ** 2 / np.maximum(m, 1), 0.0)
        from scipy.stats import chi2

        p = np.where(m > 0, chi2.sf(np.maximum(chi, 0.0), 1), 1.0)
        if design.tails == 1:
            p = np.where(c >= b, p / 2.0, 1.0 - p / 2.0)
    rejected = p <= design.alpha
    power = float(np.mean(rejected))
    return SimulatedPower(
        power=power,
        mc_se=float(math.sqrt(max(power * (1.0 - power), 0.0) / replicates)),
        replicates=replicates,
        n=design.n,
        mean_discordant=float(np.mean(b + c)),
        test=f"McNemar {test}, {design.tails}-tailed",
    )


def required_n(
    design: PairedBinaryDesign,
    *,
    target_power: float = TARGET_POWER,
    replicates: int = 8_000,
    seed: int = 0,
    n_max: int = 2_000_000,
) -> int:
    """`N*`: the smallest n whose simulated power reaches the target. The honest denominator of q.

    Found by doubling to bracket and then bisecting on the simulator, with the same seed at every
    n so the search is on one Monte Carlo surface rather than on a fresh noisy draw each step.
    Discreteness makes the true power curve a staircase, so the answer is accurate to about the
    Monte Carlo error and is reported as an integer rather than a precise one.
    """
    if design.delta == 0.0:
        return n_max
    n = 8
    while n < n_max:
        if simulate_power(design.at_n(n), replicates=replicates, seed=seed).power >= target_power:
            break
        n *= 2
    else:
        return n_max
    lo, hi = max(n // 2, 1), n
    while lo < hi:
        mid = (lo + hi) // 2
        got = simulate_power(design.at_n(mid), replicates=replicates, seed=seed).power
        if got >= target_power:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


def minimum_detectable_effect(
    design: PairedBinaryDesign,
    *,
    target_power: float = TARGET_POWER,
    replicates: int = 8_000,
    seed: int = 0,
    tol: float = 1e-4,
) -> float:
    """The smallest effect this design could have found. The number a null result needs beside it.

    Bisects on `delta` at the design's own n, holding `accuracy_a` and `rho` fixed, using the
    simulator. Returns `nan` when even the largest admissible effect does not reach the target,
    which is a real answer: it means no effect of any size is detectable at this n, and reporting a
    number there would be worse than reporting nothing.
    """
    lo, hi = 0.0, min(1.0 - design.accuracy_a, design.accuracy_a) - 1e-3
    if hi <= 0:
        return float("nan")

    def powered(delta: float) -> float:
        try:
            candidate = design.at_delta(delta)
        except ValueError:
            return 0.0
        return simulate_power(candidate, replicates=replicates, seed=seed).power

    if powered(hi) < target_power:
        return float("nan")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if powered(mid) >= target_power:
            hi = mid
        else:
            lo = mid
    return float(hi)


# ---------------------------------------------------------------------------
# The five calculators, and the comparison that shows three of them are wrong
# ---------------------------------------------------------------------------


def n_two_proportion_z(design: PairedBinaryDesign, target_power: float = TARGET_POWER) -> float:
    """Two independent proportions, per arm. Ignores the pairing entirely.

    The default reach in every sample-size calculator and the one a leaderboard comparison is most
    often planned with. It is the wrong test for this design because the two arms are the same
    items, and it is wrong in the expensive direction.
    """
    p_a, p_b = design.accuracy_a, design.accuracy_b
    if design.delta == 0.0:
        return float("inf")
    z_a = _z(1.0 - design.alpha / design.tails)
    z_b = _z(target_power)
    pbar = 0.5 * (p_a + p_b)
    num = z_a * math.sqrt(2.0 * pbar * (1.0 - pbar)) + z_b * math.sqrt(
        p_a * (1 - p_a) + p_b * (1 - p_b)
    )
    return float(num**2 / design.delta**2)


def n_cohen_h(design: PairedBinaryDesign, target_power: float = TARGET_POWER) -> float:
    """The arcsine effect size `h`, per arm. Also an independent-samples calculator.

    `phi = 2 asin(sqrt(p))` is variance-stabilising with asymptotic variance `1/n`, so the
    difference of two independent arms of size n has variance `2/n` and the required n per arm is
    `2 (z_{1-a/2} + z_{1-b})^2 / h^2`. The factor of two is easy to drop and dropping it halves
    every answer; the check is Cohen's own table entry of 63 per group for a medium effect
    `h = 0.5` at alpha 0.05 and 80% power, which this formula reproduces and the halved one does
    not.
    """
    h = 2.0 * math.asin(math.sqrt(design.accuracy_b)) - 2.0 * math.asin(
        math.sqrt(design.accuracy_a)
    )
    if h == 0.0:
        return float("inf")
    z_a = _z(1.0 - design.alpha / design.tails)
    z_b = _z(target_power)
    return float(2.0 * (z_a + z_b) ** 2 / h**2)


def n_one_sample_normal(design: PairedBinaryDesign, target_power: float = TARGET_POWER) -> float:
    """One sample on the per-item difference, with the variance computed as if uncorrelated.

    This is the subtle one, and the one most likely to be written by someone who knows the design
    is paired. The test is right and the standard deviation is wrong: it uses
    `sqrt(p_a q_a + p_b q_b)` and drops the covariance term, which is precisely the term the
    pairing buys.
    """
    p_a, p_b = design.accuracy_a, design.accuracy_b
    if design.delta == 0.0:
        return float("inf")
    sd = math.sqrt(p_a * (1 - p_a) + p_b * (1 - p_b))
    z_a = _z(1.0 - design.alpha / design.tails)
    z_b = _z(target_power)
    return float(((z_a + z_b) * sd / design.delta) ** 2)


def n_paired_normal(design: PairedBinaryDesign, target_power: float = TARGET_POWER) -> float:
    """One sample on the difference with the covariance term kept. Uses `rho`."""
    p_a, p_b = design.accuracy_a, design.accuracy_b
    if design.delta == 0.0:
        return float("inf")
    var = (
        p_a * (1 - p_a)
        + p_b * (1 - p_b)
        - 2.0 * design.rho * math.sqrt(p_a * (1 - p_a) * p_b * (1 - p_b))
    )
    sd = math.sqrt(max(var, 1e-12))
    z_a = _z(1.0 - design.alpha / design.tails)
    z_b = _z(target_power)
    return float(((z_a + z_b) * sd / design.delta) ** 2)


def n_mcnemar_normal(design: PairedBinaryDesign, target_power: float = TARGET_POWER) -> float:
    """The discordant-pair calculator, which is the one matched to the test actually run."""
    _, p10, p01, _ = design.cells
    pi_d = p10 + p01
    delta = p01 - p10
    if delta == 0.0 or pi_d <= 0.0:
        return float("inf")
    z_a = _z(1.0 - design.alpha / design.tails)
    z_b = _z(target_power)
    num = z_a * math.sqrt(pi_d) + z_b * math.sqrt(max(pi_d - delta**2, 1e-12))
    return float(num**2 / delta**2)


#: The five a practitioner would reach for, named. Three of them drop `rho`.
CALCULATORS: dict[str, Callable[[PairedBinaryDesign, float], float]] = {
    "two_proportion_z": n_two_proportion_z,
    "cohen_h": n_cohen_h,
    "one_sample_normal": n_one_sample_normal,
    "paired_normal": n_paired_normal,
    "mcnemar_normal": n_mcnemar_normal,
}

#: Which calculators use the per-item correlation at all. This is the whole of the finding: the
#: three that do not are the three that come out roughly 2x wrong at rho = 0.5.
USES_CORRELATION: dict[str, bool] = {
    "two_proportion_z": False,
    "cohen_h": False,
    "one_sample_normal": False,
    "paired_normal": True,
    "mcnemar_normal": True,
}


@dataclass(frozen=True)
class CalculatorCheck:
    """One calculator's `N*` beside the simulator's, and the ratio between them."""

    name: str
    n_star: float
    n_star_simulated: int
    ratio: float
    uses_correlation: bool

    @property
    def roughly_2x_wrong(self) -> bool:
        """Off by at least 50% in either direction. A stated band, not a p-value."""
        return not (0.667 <= self.ratio <= 1.5)

    def render(self) -> str:
        flag = "  <- off" if self.roughly_2x_wrong else ""
        return (
            f"{self.name:<20} N* {self.n_star:>10,.0f}  ratio to simulation {self.ratio:>6.2f}"
            f"{flag}"
        )


def compare_calculators(
    design: PairedBinaryDesign | None = None,
    *,
    target_power: float = TARGET_POWER,
    replicates: int = 8_000,
    seed: int = 0,
) -> dict[str, CalculatorCheck]:
    """Run all five against the simulator and report each one's ratio.

    The default design is two close models on a shared benchmark: 82% against 85% at a per-item
    correctness correlation of 0.5, which is a mild correlation for two models of the same family
    scored on the same items. The three calculators that ignore the correlation come out near 2x;
    the two that use it come out near 1. Change `rho` and watch the three move: at 0.8 they are
    near 5x, because the inflation factor is `1 / (1 - rho)`.
    """
    design = design or DEFAULT_CLOSE_PAIR
    n_sim = required_n(design, target_power=target_power, replicates=replicates, seed=seed)
    out: dict[str, CalculatorCheck] = {}
    for name, fn in CALCULATORS.items():
        n_star = fn(design, target_power)
        out[name] = CalculatorCheck(
            name=name,
            n_star=n_star,
            n_star_simulated=n_sim,
            ratio=float(n_star / n_sim) if n_sim > 0 else float("inf"),
            uses_correlation=USES_CORRELATION[name],
        )
    return out


#: Two close models on a shared benchmark. The scenario the finding is about.
DEFAULT_CLOSE_PAIR = PairedBinaryDesign(n=500, accuracy_a=0.82, accuracy_b=0.85, rho=0.5)


# ---------------------------------------------------------------------------
# The resolution ratio
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolution:
    """`q = N / N*`. Below 1 the row is not resolved, and this type will not say otherwise.

    ``basis`` records whether N is a raw item count or a lineage-aware effective sample size. A
    leaderboard row built from 5,000 items expanded from 200 seeds has N = 5,000 and ESS = 200,
    and the two give q values a factor of 25 apart. The second one is the true one.
    """

    n: float
    n_star: float
    target_power: float = TARGET_POWER
    alpha: float = ALPHA
    basis: str = "items"
    note: str = ""

    @property
    def q(self) -> float:
        return float(self.n / self.n_star) if self.n_star > 0 else float("inf")

    @property
    def resolved(self) -> bool:
        return self.q >= 1.0

    def render(self) -> str:
        head = (
            f"q = N/N* = {self.n:,.0f}/{self.n_star:,.0f} = {self.q:.3f} "
            f"({self.basis}, {self.target_power:.0%} power at alpha {self.alpha:g})"
        )
        if self.resolved:
            return f"{head}\n    RESOLVED: this comparison had the sample to separate them."
        shortfall = self.n_star - self.n
        return (
            f"{head}\n    NOT RESOLVED: this comparison could not have separated them. It is "
            f"short by {shortfall:,.0f} {self.basis}. Report the ordering as unresolved rather "
            f"than as a result; a rank here is a coin flip wearing a decimal."
        ) + (f"\n    {self.note}" if self.note else "")


def resolution_ratio(
    n: float,
    n_star: float,
    *,
    target_power: float = TARGET_POWER,
    alpha: float = ALPHA,
    basis: str = "items",
) -> Resolution:
    """Build the resolution ratio. Kept as a named function so `q` is never an inline division."""
    return Resolution(n=n, n_star=n_star, target_power=target_power, alpha=alpha, basis=basis)


def resolution_from_lineage(
    seed_labels: Sequence[Any],
    n_star: float,
    *,
    target_power: float = TARGET_POWER,
    alpha: float = ALPHA,
) -> Resolution:
    """The resolution ratio computed on effective sample size rather than on row count.

    Delegates to `stats.ess.effective_sample_size`, which is the module that already knows a
    stimulus cloned fifty times is worth one. Using ESS here rather than n is the difference
    between a q that reflects the data and a q that reflects the expansion factor.
    """
    ess = effective_sample_size(seed_labels)
    return Resolution(
        n=ess,
        n_star=n_star,
        target_power=target_power,
        alpha=alpha,
        basis="effective sample size",
        note=(
            f"{len(list(seed_labels)):,} rows collapse to {ess:.1f} independent lineages; the "
            f"ratio above uses the lineages"
        ),
    )


def alpha_for_family(
    alpha: float, n_comparisons: int, *, method: Literal["bonferroni", "bh"] = "bonferroni"
) -> float:
    """The alpha to plan at when this reading is one of a family.

    Planning at the nominal alpha and correcting afterwards is how a battery of thirty instruments
    ends with thirty underpowered readings and one significant one. Bonferroni is the honest
    planning choice because it needs nothing from the data. The Benjamini-Hochberg option is the
    optimistic one and it assumes every hypothesis in the family is a true effect, which makes it
    the level BH would use in the best case rather than the level it will use; it is offered for
    the case where you genuinely expect most of the family to be real.

    `stats.multiplicity.bh_fdr` is what runs afterwards. This only chooses what to plan at.
    """
    m = max(int(n_comparisons), 1)
    if method == "bonferroni":
        return alpha / m
    return alpha  # BH's least conservative rank; see the docstring


# ---------------------------------------------------------------------------
# The detection band
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionBiasDesign:
    """A judge scored on the same items in both orders, with a position advantage in logits.

    ``logit_advantage`` is the bias, held fixed. ``base_accuracy`` is the accuracy the judge would
    have with no position effect. The two orders' accuracies are `sigmoid(logit(base) +/- b/2)`,
    which is the point of the whole construction: as base accuracy goes to 1 the difference between
    those two probabilities collapses toward zero even though `b` never changes. The bias does not
    go away, it stops being expressible in the units the test measures.
    """

    n: int
    base_accuracy: float
    logit_advantage: float
    rho: float = 0.5
    alpha: float = ALPHA
    tails: Tails = 2

    @property
    def accuracies(self) -> tuple[float, float]:
        a = _logit(self.base_accuracy)
        b = self.logit_advantage
        return (_sigmoid(a + b / 2.0), _sigmoid(a - b / 2.0))

    def as_paired(self) -> PairedBinaryDesign:
        first, second = self.accuracies
        lo, hi = rho_bounds(second, first)
        return PairedBinaryDesign(
            n=self.n,
            accuracy_a=second,
            accuracy_b=first,
            rho=min(max(self.rho, lo), hi),
            alpha=self.alpha,
            tails=self.tails,
        )


Readability = Literal["measurable", "not_measurable_above", "not_measurable_below"]


@dataclass(frozen=True)
class DetectionBand:
    """The window of base accuracy over which a design can see the effect it is looking for.

    ``low`` and ``high`` are computed by simulation for one design, not quoted. Outside the window
    the design's power is below target, which means a null there carries no information: the
    correct reading is "not measurable", and `interpret` will not produce the word "unbiased" for a
    base accuracy outside the band no matter how clean the null looks.
    """

    low: float
    high: float
    target_power: float
    n: int
    logit_advantage: float
    alpha: float
    grid: tuple[float, ...] = ()
    powers: tuple[float, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.low <= self.high)

    @property
    def low_is_grid_edge(self) -> bool:
        """Whether the lower edge is just where the grid stopped rather than where power did.

        Worth checking before quoting a lower bound. In the logit model used here the effect in
        probability space is *largest* near chance, so the design usually has power all the way
        down and the reported `low` is the first grid point rather than a real edge. The published
        60% lower bound for position bias comes from somewhere this model does not represent, and
        quoting `low` as if it were measured would be borrowing that bound.
        """
        return bool(self.grid) and abs(self.low - self.grid[0]) < 1e-12

    def contains(self, base_accuracy: float) -> bool:
        return self.low <= base_accuracy <= self.high

    def read(self, base_accuracy: float) -> Readability:
        if self.contains(base_accuracy):
            return "measurable"
        return "not_measurable_above" if base_accuracy > self.high else "not_measurable_below"

    def interpret(self, base_accuracy: float, *, detected: bool) -> str:
        """The sentence a card prints. The one it must not print is the one this refuses to.

        A design that could not have detected the effect produces "not measurable", never
        "unbiased". That distinction is the whole lesson: several refuted cards in this library's
        own history report an absence of signal at a base accuracy where an absence of signal was
        the only outcome available.
        """
        where = self.read(base_accuracy)
        if detected:
            return (
                f"bias detected at base accuracy {base_accuracy:.3f}. A detection outside the "
                f"band is still a detection; the band bounds what a null can mean, not what a "
                f"hit can."
                if where != "measurable"
                else f"bias detected at base accuracy {base_accuracy:.3f}, inside the measurable "
                f"band [{self.low:.3f}, {self.high:.3f}]."
            )
        if where == "measurable":
            return (
                f"no bias detected at base accuracy {base_accuracy:.3f}, inside the measurable "
                f"band [{self.low:.3f}, {self.high:.3f}] where this design has "
                f"{self.target_power:.0%} power. This null is informative."
            )
        side = "above" if where == "not_measurable_above" else "below"
        return (
            f"NOT MEASURABLE at base accuracy {base_accuracy:.3f}, which is {side} the band "
            f"[{self.low:.3f}, {self.high:.3f}]. This design has under {self.target_power:.0%} "
            f"power here, so the absence of signal is a property of the design and not of the "
            f"judge. Read it as not measurable, not as unbiased. To make it informative, raise n "
            f"or move the evaluation to a difficulty inside the band."
        )

    def render(self) -> str:
        if self.is_empty:
            return (
                f"detection band: EMPTY at n={self.n:,}. No base accuracy on the grid reaches "
                f"{self.target_power:.0%} power for a logit advantage of "
                f"{self.logit_advantage:.3f}. Nothing this design reports about position bias is "
                f"informative either way."
            )
        edge = (
            "\n    the lower edge is the first grid point, so this design shows no lower edge in "
            "this model; do not quote it as a measured bound."
            if self.low_is_grid_edge
            else ""
        )
        return (
            f"detection band [{self.low:.3f}, {self.high:.3f}] at n={self.n:,}, "
            f"logit advantage {self.logit_advantage:.3f}, "
            f"{self.target_power:.0%} power, alpha {self.alpha:g}. Outside it, an absence of "
            f"signal means not measurable." + edge
        )


def detection_band(
    *,
    n: int,
    logit_advantage: float,
    rho: float = 0.5,
    alpha: float = ALPHA,
    target_power: float = TARGET_POWER,
    grid: Sequence[float] | None = None,
    replicates: int = 4_000,
    seed: int = 0,
) -> DetectionBand:
    """Compute the band of base accuracy where this design reaches the target power.

    Simulated at every grid point, so the band is a measurement of the design rather than a quoted
    window. The upper edge is the interesting one and it is structural: at high base accuracy the
    two orders are both right almost always, the discordant-pair count collapses, and McNemar has
    nothing to test. The published window for LLM position bias is roughly 60% to 95%; whether
    this design's band matches it is a question to answer by running this, not by assuming it.
    """
    points = tuple(grid) if grid is not None else tuple(np.round(np.arange(0.51, 0.995, 0.01), 4))
    powers: list[float] = []
    for base in points:
        design = PositionBiasDesign(
            n=n, base_accuracy=float(base), logit_advantage=logit_advantage, rho=rho, alpha=alpha
        )
        powers.append(simulate_power(design.as_paired(), replicates=replicates, seed=seed).power)
    ok = [p >= target_power for p in powers]
    if not any(ok):
        return DetectionBand(
            low=float("nan"),
            high=float("nan"),
            target_power=target_power,
            n=n,
            logit_advantage=logit_advantage,
            alpha=alpha,
            grid=points,
            powers=tuple(powers),
        )
    first = ok.index(True)
    last = len(ok) - 1 - ok[::-1].index(True)
    return DetectionBand(
        low=float(points[first]),
        high=float(points[last]),
        target_power=target_power,
        n=n,
        logit_advantage=logit_advantage,
        alpha=alpha,
        grid=points,
        powers=tuple(powers),
    )


# ---------------------------------------------------------------------------
# The plan: what M10 hands to a preflight
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PowerPlan:
    """Everything M10 knows about a design, computed before anything runs.

    ``validated_against`` is a field rather than a docstring claim because the requirement is
    structural: a plan whose numbers came from a formula and were never checked against the test
    they describe is the failure mode this module exists to close.
    """

    design: PairedBinaryDesign
    power: float
    power_mc_se: float
    n_star: int
    mde: float
    resolution: Resolution
    target_power: float = TARGET_POWER
    replicates: int = 8_000
    validated_against: str = "simulation"
    calculators: Mapping[str, CalculatorCheck] = field(default_factory=dict)

    @property
    def adequate(self) -> bool:
        return self.power >= self.target_power

    def render(self) -> str:
        lines = [
            f"power plan: n={self.design.n:,}, "
            f"{self.design.accuracy_a:.3f} vs {self.design.accuracy_b:.3f} "
            f"(delta {self.design.delta:+.3f}), rho={self.design.rho:.2f}, "
            f"alpha={self.design.alpha:g}",
            f"    simulated power {self.power:.3f} +/- {self.power_mc_se:.3f} "
            f"({self.replicates:,} replicates)",
            f"    N* for {self.target_power:.0%} power: {self.n_star:,}",
            "    minimum detectable effect at this n: "
            + (f"{self.mde:.4f}" if np.isfinite(self.mde) else "none; no effect is detectable"),
            "    " + self.resolution.render().replace("\n    ", "\n        "),
        ]
        if self.calculators:
            lines.append("    standard calculators against the simulation:")
            lines += [f"        {c.render()}" for c in self.calculators.values()]
        return "\n".join(lines)


def plan(
    design: PairedBinaryDesign,
    *,
    target_power: float = TARGET_POWER,
    replicates: int = 8_000,
    seed: int = 0,
    with_calculators: bool = False,
    ess: float | None = None,
) -> PowerPlan:
    """The whole M10 reading for one design, simulated.

    ``ess`` overrides the numerator of q when the items are not independent, which is the usual
    case for an expanded stimulus set. Pass it and the resolution ratio is computed on effective
    sample size, which is the only version of q worth printing.
    """
    simulated = simulate_power(design, replicates=replicates, seed=seed)
    n_star = required_n(design, target_power=target_power, replicates=replicates, seed=seed)
    mde = minimum_detectable_effect(
        design, target_power=target_power, replicates=replicates, seed=seed
    )
    resolution = Resolution(
        n=float(ess if ess is not None else design.n),
        n_star=float(n_star),
        target_power=target_power,
        alpha=design.alpha,
        basis="effective sample size" if ess is not None else "items",
    )
    checks = (
        compare_calculators(design, target_power=target_power, replicates=replicates, seed=seed)
        if with_calculators
        else {}
    )
    return PowerPlan(
        design=design,
        power=simulated.power,
        power_mc_se=simulated.mc_se,
        n_star=n_star,
        mde=mde,
        resolution=resolution,
        target_power=target_power,
        replicates=replicates,
        calculators=checks,
    )


# ---------------------------------------------------------------------------
# Units, so a power and an MDE cannot be compared to each other
# ---------------------------------------------------------------------------

#: The unit an MDE is expressed in, matching `study.mde` in the registry. It is not dimensionless
#: and it is not a probability: it is an effect, on whatever scale the study's own metric uses.
EFFECT = Unit(dimension="effect", as_printed="effect")


@dataclass(frozen=True)
class PowerQuantity:
    """One of M10's three numbers, carrying the unit that decides what it may be compared to."""

    quantity: QuantityID
    value: float
    unit: Unit = DIMENSIONLESS


def difference(a: PowerQuantity, b: PowerQuantity) -> Any:
    """Subtract two of M10's readings, or refuse when the units do not admit it.

    This is the `units` group's assertion made executable: a power is dimensionless and an MDE is
    an effect, and subtracting one from the other is the unit error that is the most common silent
    failure in this literature. The conversion factor between them is not a property of the unit,
    so there is nothing to convert.
    """
    if not a.unit.compatible_with(b.unit):
        return Refusal(
            instrument="stats.power.difference",
            reason=RefusalReason.UNIT_MISMATCH,
            detail=(
                f"{a.quantity} is in {a.unit} and {b.quantity} is in {b.unit}; these are "
                f"different quantities, not one quantity in two clothes"
            ),
            remedy=(
                "compare readings of the same quantity, or convert explicitly with a factor you "
                "can name. A power and a minimum detectable effect do not subtract."
            ),
            statistics={"unit_a": str(a.unit), "unit_b": str(b.unit)},
        )
    return PowerQuantity(quantity=a.quantity, value=a.value - b.value, unit=a.unit)


__all__ = [
    "ALPHA",
    "CALCULATORS",
    "DEFAULT_CLOSE_PAIR",
    "DIMENSIONLESS",
    "EFFECT",
    "TARGET_POWER",
    "USES_CORRELATION",
    "CalculatorCheck",
    "DetectionBand",
    "PairedBinaryDesign",
    "PositionBiasDesign",
    "PowerPlan",
    "PowerQuantity",
    "Readability",
    "Resolution",
    "SimulatedPower",
    "Tails",
    "alpha_for_family",
    "compare_calculators",
    "detection_band",
    "difference",
    "mcnemar_p_values",
    "minimum_detectable_effect",
    "n_cohen_h",
    "n_mcnemar_normal",
    "n_one_sample_normal",
    "n_paired_normal",
    "n_two_proportion_z",
    "plan",
    "required_n",
    "resolution_from_lineage",
    "resolution_ratio",
    "rho_bounds",
    "simulate_power",
]

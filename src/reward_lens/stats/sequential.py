"""Anytime-valid sequential statistics: Ville, stitching, e-merging, e-BH.

The one fact this module is built on is Ville's inequality (Ville 1939). For a nonnegative
supermartingale ``M`` with ``E[M_0] = 1``,

    P(exists t : M_t >= a) <= 1/a

so ``p_t = 1 / max_{s <= t} M_s`` is a p-value that is valid **at every stopping time at once**,
not merely at a pre-declared sample size. That is what makes continuous peeking free, and it is the
only reason a monitor is allowed to look at every training step and still quote a false-alarm rate.

What lives here is the mathematics that ``monitor/_vendor/cif.py`` does not already ship: the
Ville conversion, the polynomial-stitched boundary of Howard, Ramdas, McAuliffe and Sekhon (2021),
the two e-value merging rules, e-BH, and the simulation that measures what a fixed-sample interval
actually costs under peeking. The Hoeffding alpha-spending radius and the ONS betting sequence are
**not** reimplemented here; they come from the vendored file, and `monitor.eprocess` composes the
two. `tests/test_monitor_sequential.py` pins the one overlap (the fixed-sample Hoeffding radius,
which is the mandatory baseline and belongs in the statistics layer) against the vendored version.

**Two things this module will not do.** It will not convert a p-value to an e-value by ``1/p``,
which is not a calibrator and is not valid. And it will not multiply e-values across channels: the
merging rule that is valid under arbitrary dependence is the *arithmetic mean*, and `merge_e` says
so in its signature rather than in a comment. See `merge_e` for why that matters here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.special import zeta

# ---------------------------------------------------------------------------
# Ville's inequality
# ---------------------------------------------------------------------------


def ville_pvalue(running_max: float) -> float:
    """The anytime-valid p-value from a nonnegative martingale's running maximum.

    ``p_t = 1 / max_{s <= t} M_s``, clipped into [0, 1]. Ville's inequality says
    ``P(exists t : M_t >= a) <= 1/a`` for a nonnegative supermartingale started at 1, so this
    number is a valid p-value at every stopping time simultaneously. It is the whole licence to
    peek.

    A running maximum below 1 gives a p-value of 1, which is correct rather than a clamp: the
    process never accumulated evidence, so no level of any test is crossed.
    """
    if not math.isfinite(running_max) or running_max <= 0.0:
        return 1.0
    return float(min(1.0, 1.0 / running_max))


def ville_threshold(alpha: float) -> float:
    """The martingale value at which a level-``alpha`` anytime-valid test rejects: ``1/alpha``.

    Rejecting the first time ``M_t >= 1/alpha`` has false-alarm probability at most ``alpha`` over
    an unbounded horizon. There is no multiplicity correction to apply afterwards, because there
    was never a fixed number of looks to correct for.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    return 1.0 / alpha


# ---------------------------------------------------------------------------
# The fixed-sample baseline
# ---------------------------------------------------------------------------


def fixed_sample_radius(n: int, delta: float) -> float:
    """The Hoeffding half-width for a mean of ``n`` observations in [0, 1], at one look.

    This is the **mandatory baseline** for J1 and it is here rather than in `monitor/` because it
    is what every reader already computes. It is valid at exactly one sample size, declared in
    advance. Recomputing it as the data arrive and reading it whenever it looks interesting is what
    `peeking_miscoverage` measures the cost of, and that cost is the entire argument for the
    anytime-valid machinery.

    Identical to `monitor._vendor.cif.hoeffding_fixed_radius`; a test asserts the two agree to
    machine precision so the duplication cannot drift.
    """
    if n <= 0:
        return float("inf")
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))


# ---------------------------------------------------------------------------
# Rung 1: the polynomial-stitched boundary
# ---------------------------------------------------------------------------

#: Howard et al. (2021) tune the boundary to be tightest at one intrinsic time and valid at all of
#: them. This is that time in units of the variance process, and 1.0 puts the tight point early,
#: which is where a training-run monitor spends the steps that matter. A run that will be read at
#: step 200 should pass ``v_opt = 200 * sigma**2`` instead and get a narrower interval there.
DEFAULT_V_OPT: float = 1.0


def stitched_radius(
    n: int,
    delta: float,
    *,
    sigma: float = 0.5,
    v_opt: float = DEFAULT_V_OPT,
    s: float = 1.4,
    eta: float = 2.0,
) -> float:
    """The sub-Gaussian polynomial-stitched confidence radius (Howard et al. 2021, Theorem 1).

    Rung 1 of J1's ladder. Rung 0 spends a fixed alpha budget ``6 delta / (pi^2 n^2)`` at every
    step, which is simple and pays ``2 log n`` inside the square root forever. Stitching pays
    ``log log n`` instead, by covering geometrically spaced epochs with a union bound and taking
    the envelope. On the [0, 1]-bounded case at delta = 0.05 the measured narrowing over rung 0 is
    1.33x at n = 30, 1.41x at n = 100, 1.56x at n = 1,000 and 1.70x at n = 10,000. Those are
    computed by `tests/test_monitor_sequential.py` and not quoted from the paper.

    ``sigma`` is the sub-Gaussian parameter of a single observation. For a variable supported on
    [0, 1], Hoeffding's lemma gives ``sigma = 1/2`` and that is the default. Supplying a smaller
    one is a claim about the data that this function cannot check, and it makes the interval
    narrower, so it is the one argument here that can produce a confident wrong number.

    The boundary is ``k1 * sqrt(v * l(v))`` with ``v = n sigma^2`` the intrinsic time and

        l(v) = s * log log(eta * max(v, v_opt) / v_opt) + log(zeta(s) / (delta * log(eta)^s))

    with ``k1 = (eta^(1/4) + eta^(-1/4)) / sqrt(2)``. The sub-exponential term of the general
    theorem is dropped because a bounded observation has none, and dropping it is what makes this
    the sub-Gaussian rather than the sub-gamma boundary. The radius on the mean is the boundary
    divided by ``n``.
    """
    if n <= 0:
        return float("inf")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1); got {delta}")
    if eta <= 1.0:
        raise ValueError(f"eta must exceed 1 so the epochs grow; got {eta}")
    if s <= 1.0:
        raise ValueError(f"s must exceed 1 for zeta(s) to converge; got {s}")
    v = float(n) * sigma * sigma
    k1 = (eta**0.25 + eta**-0.25) / math.sqrt(2.0)
    inner = eta * max(v, v_opt) / v_opt
    ell = s * math.log(math.log(inner)) + math.log(float(zeta(s)) / (delta * math.log(eta) ** s))
    if ell <= 0.0:
        # The log-log term is negative near v_opt and can in principle swamp the constant for a
        # very loose delta. A negative "boundary" is not conservative, it is nonsense, so refuse to
        # produce one and fall back to the rung-0 width, which is always valid.
        return float("inf")
    return float(k1 * math.sqrt(v * ell) / n)


# ---------------------------------------------------------------------------
# Merging e-values, and the rule that is actually valid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergedEValue:
    """One merged e-value, with the dependence assumption that licenses it written down.

    The assumption is a field rather than a docstring because the two rules differ by a factor that
    grows with the number of channels, and a merged e-value whose assumption is invisible is the
    exact shape of the overclaim this layer exists to prevent.
    """

    value: float
    rule: str
    assumption: str
    n_merged: int

    def render(self) -> str:
        return (
            f"merged e-value {self.value:.4g} over {self.n_merged} channels by {self.rule} "
            f"({self.assumption})"
        )


def merge_e(e_values: Sequence[float], *, dependence: str = "arbitrary") -> MergedEValue:
    """Combine e-values across channels, under a dependence assumption you have to state.

    ``dependence="arbitrary"`` uses the **arithmetic mean**. Vovk and Wang (2021) show the mean is
    an e-merging function under arbitrary dependence and that it is essentially the only admissible
    symmetric one: no symmetric rule that dominates it exists.

    ``dependence="independent"`` uses the **product**, which is valid when the e-values are
    independent, and also when they are sequentially valid, meaning each is conditionally an
    e-value given everything before it. That second case is what makes an e-process legal over
    *time* within one channel: the running product of per-step betting factors is a test
    martingale.

    The distinction is load-bearing for J3. E-values are often said to "multiply legally under
    arbitrary dependence"; they do not. Entropy decline, prediction saturation and episode-length
    pinning are strongly dependent channels, so multiplying their e-values inflates the evidence by
    a factor that can reach the number of channels and is not bounded in general. Over time within
    one channel the product is right; across channels at one time the mean is right.
    """
    e = np.asarray(e_values, dtype=np.float64).ravel()
    e = e[np.isfinite(e)]
    if e.size == 0:
        return MergedEValue(1.0, "none", "no finite e-values to merge", 0)
    if np.any(e < 0):
        raise ValueError("an e-value is nonnegative by definition; got a negative one")
    if dependence == "arbitrary":
        return MergedEValue(
            float(np.mean(e)),
            "arithmetic mean",
            "valid under arbitrary dependence (Vovk and Wang 2021)",
            int(e.size),
        )
    if dependence == "independent":
        return MergedEValue(
            float(np.prod(e)),
            "product",
            "valid only under independence or sequential validity, not under arbitrary dependence",
            int(e.size),
        )
    raise ValueError(f"dependence must be 'arbitrary' or 'independent'; got {dependence!r}")


# ---------------------------------------------------------------------------
# e-BH
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EBHResult:
    """What e-BH rejected, and the threshold it rejected at."""

    rejected: np.ndarray
    threshold: float
    n_rejected: int
    alpha: float
    n_hypotheses: int

    def render(self) -> str:
        return (
            f"e-BH at alpha={self.alpha:.3g}: {self.n_rejected} of {self.n_hypotheses} rejected "
            f"at e >= {self.threshold:.4g}"
        )


def ebh(e_values: Sequence[float], alpha: float = 0.05) -> EBHResult:
    """The e-Benjamini-Hochberg procedure (Wang and Ramdas 2022). FDR control under any dependence.

    Sort the e-values descending as ``e_(1) >= ... >= e_(K)``, find the largest ``k`` with
    ``e_(k) >= K / (alpha k)``, and reject those ``k``. The false discovery rate is at most
    ``alpha`` **with no assumption whatever about the dependence between the e-values**, which is
    what makes it the right procedure for a ledger of alarms raised on overlapping channels of the
    same training run. Plain BH on p-values needs positive dependence for the same guarantee and
    the channels of a monitor are not positively dependent by construction.

    Non-finite e-values are treated as zero rather than dropped, so a channel whose e-process
    diverged numerically cannot be silently removed from the denominator and make the surviving
    channels easier to reject.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
    e = np.asarray(e_values, dtype=np.float64).ravel()
    k_total = e.size
    if k_total == 0:
        return EBHResult(np.zeros(0, dtype=bool), float("inf"), 0, alpha, 0)
    e = np.where(np.isfinite(e), e, 0.0)
    order = np.argsort(-e)
    ranked = e[order]
    ranks = np.arange(1, k_total + 1, dtype=np.float64)
    admissible = ranked >= k_total / (alpha * ranks)
    if not np.any(admissible):
        return EBHResult(np.zeros(k_total, dtype=bool), float("inf"), 0, alpha, k_total)
    k_star = int(np.max(np.where(admissible)[0])) + 1
    threshold = float(k_total / (alpha * k_star))
    rejected = np.zeros(k_total, dtype=bool)
    rejected[order[:k_star]] = True
    return EBHResult(rejected, threshold, k_star, alpha, k_total)


# ---------------------------------------------------------------------------
# What peeking costs, measured rather than asserted
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeekingCost:
    """How often each interval would have been wrong, under exactly the same looks.

    Three comparators, because two of them answer different questions and reporting only one is how
    this comparison gets rigged in either direction.

    ``hoeffding_miscoverage`` is the fixed-sample **Hoeffding** interval, recomputed and read at
    every step. It is the matched comparator: same bound, same boundedness assumption, and the only
    difference from rung 0 is the alpha-spending. It isolates what the spending buys and nothing
    else. On a rate well away from 1/2 it is so conservative that it rarely fails even under
    peeking, which is honest and is not a demonstration of anything.

    ``wilson_miscoverage`` is the fixed-sample **Wilson score** interval, which is the best
    fixed-sample interval anybody actually uses for a Bernoulli rate: it beats the normal
    approximation everywhere and it does not collapse to zero width when no positive has been seen
    yet, which is the case that would rig this comparison. It is the comparator that shows what
    peeking costs, and its assumptions differ from rung 0's in more than the spending, which the
    reading says.

    The normal approximation is deliberately not one of the three. Its half-width is exactly zero
    until the first positive arrives, so on a rate of 0.1 it excludes the truth on essentially every
    run at step 1, and reporting that 100% would be reporting a degeneracy as a demonstration.
    """

    nominal: float
    hoeffding_miscoverage: float
    wilson_miscoverage: float
    anytime_miscoverage: float
    n_runs: int
    n_steps: int
    truth: float
    hoeffding_final_width: float
    wilson_final_width: float
    anytime_final_width: float
    burn_in: int = 1

    @property
    def inflation(self) -> float:
        """How many times its own advertised level the practitioner's interval actually spends."""
        return self.wilson_miscoverage / self.nominal if self.nominal > 0 else float("nan")

    def render(self) -> str:
        return (
            f"over {self.n_runs} streams of {self.n_steps} steps with true mean {self.truth:.3g}, "
            f"all read at every step from step {self.burn_in}:\n"
            f"    Wilson {1 - self.nominal:.0%} interval, fixed-sample: wrong on "
            f"{self.wilson_miscoverage:.1%} of runs "
            f"({self.inflation:.1f}x its advertised {self.nominal:.1%}), "
            f"final half-width {self.wilson_final_width:.4f}\n"
            f"    Hoeffding {1 - self.nominal:.0%} interval, fixed-sample: wrong on "
            f"{self.hoeffding_miscoverage:.1%} of runs, "
            f"final half-width {self.hoeffding_final_width:.4f}\n"
            f"    anytime-valid interval: wrong on {self.anytime_miscoverage:.1%} of runs "
            f"(Ville guarantees at most {self.nominal:.1%}), "
            f"final half-width {self.anytime_final_width:.4f}"
        )


def wilson_interval(
    successes: np.ndarray, n: np.ndarray, z: float = 1.959963984540054
) -> tuple[np.ndarray, np.ndarray]:
    """The Wilson score interval for a binomial proportion, vectorised.

    ``centre = (k + z^2/2) / (n + z^2)`` with half-width
    ``z sqrt(n) / (n + z^2) * sqrt(p(1-p) + z^2/(4n))``. It is the fixed-sample interval a careful
    practitioner reaches for and it is well defined at ``k = 0``, unlike the normal approximation,
    which is why it is the comparator here.
    """
    n = np.asarray(n, dtype=np.float64)
    k = np.asarray(successes, dtype=np.float64)
    p = k / n
    denom = n + z * z
    centre = (k + 0.5 * z * z) / denom
    half = (z * np.sqrt(n) / denom) * np.sqrt(p * (1.0 - p) + z * z / (4.0 * n))
    return np.clip(centre - half, 0.0, 1.0), np.clip(centre + half, 0.0, 1.0)


def peeking_miscoverage(
    anytime_radius: Callable[[int], float],
    *,
    truth: float = 0.1,
    delta: float = 0.05,
    n_steps: int = 200,
    n_runs: int = 2000,
    seed: int = 0,
    burn_in: int = 1,
    z: float = 1.959963984540054,
) -> PeekingCost:
    """Simulate Bernoulli streams and count how often each interval excludes the truth.

    The experiment is the one a practitioner actually runs: a rate is estimated as the data arrive,
    the interval is recomputed at every step, and the reader looks at every step. A fixed-sample
    interval is valid at one declared ``n`` and this reads it at all of them, which is precisely the
    use that makes its stated level meaningless. The anytime-valid interval is valid at all of them
    by Ville and this reads it the same way.

    Bernoulli rather than a general [0, 1] variable because a hack rate is a Bernoulli rate.

    ``burn_in`` is the first step anybody is allowed to look at, and it applies to all three
    intervals equally so the comparison stays paired. A burn-in of 1 is the honest default: a reader
    who peeks peeks from the start. Raising it makes the fixed-sample intervals look better and the
    reading records the value used.
    """
    rng = np.random.default_rng(seed)
    draws = rng.random((n_runs, n_steps)) < truth
    ns = np.arange(1, n_steps + 1, dtype=np.float64)
    counts = np.cumsum(draws, axis=1)
    means = counts / ns
    r_hoeff = np.array([fixed_sample_radius(int(n), delta) for n in ns], dtype=np.float64)
    r_any = np.array([anytime_radius(int(n)) for n in ns], dtype=np.float64)
    w_lo, w_hi = wilson_interval(counts, np.broadcast_to(ns, counts.shape), z)
    look = slice(burn_in - 1, None)
    dev = np.abs(means - truth)[:, look]
    return PeekingCost(
        nominal=delta,
        hoeffding_miscoverage=float(np.mean(np.any(dev > r_hoeff[None, look], axis=1))),
        wilson_miscoverage=float(
            np.mean(np.any((w_lo[:, look] > truth) | (w_hi[:, look] < truth), axis=1))
        ),
        anytime_miscoverage=float(np.mean(np.any(dev > r_any[None, look], axis=1))),
        n_runs=n_runs,
        n_steps=n_steps,
        truth=truth,
        hoeffding_final_width=float(r_hoeff[-1]),
        wilson_final_width=float(np.mean(0.5 * (w_hi[:, -1] - w_lo[:, -1]))),
        anytime_final_width=float(r_any[-1]),
        burn_in=int(burn_in),
    )


# ---------------------------------------------------------------------------
# A mixture e-process for a bounded stream, used by the conjunction detector
# ---------------------------------------------------------------------------


def bounded_evalue_increment(x: float, mean_null: float, lam: float) -> float:
    """One betting factor ``1 + lam (x - mean_null)`` for an observation in [0, 1].

    The running product of these is a nonnegative martingale with expectation 1 under the null that
    ``E[x] = mean_null``, provided ``lam`` is chosen from the past only. That is the whole
    construction: no distributional assumption beyond boundedness, no asymptotics, and validity at
    every stopping time.

    ``lam`` is clipped so the factor stays positive. A nonpositive factor is not a bet, it is
    bankruptcy, and it would make the log capital negative infinity for the rest of the run.
    """
    span_low = -1.0 / (1.0 - mean_null) if mean_null < 1.0 else -float("inf")
    span_high = 1.0 / mean_null if mean_null > 0.0 else float("inf")
    lam = float(min(max(lam, span_low * 0.999), span_high * 0.999))
    return float(1.0 + lam * (x - mean_null))


def running_evalue(
    xs: Sequence[float],
    mean_null: float,
    *,
    lam: float = 0.5,
    adaptive: bool = True,
) -> np.ndarray:
    """The e-process for "the mean of this bounded stream is at most ``mean_null``".

    Returns the capital after each observation, starting at 1. With ``adaptive=True`` the bet size
    follows the ONS-style rule the vendored `cif.BettingCSTracker` uses, so the sequence adapts to
    an effect it did not know the size of in advance; with ``adaptive=False`` the bet is the fixed
    ``lam``, which is what a hand-computed unit test can check.

    One-sided by construction: it accumulates evidence when the observed mean runs **above**
    ``mean_null`` and loses capital when it runs below. A monitor for a rising hack rate wants
    exactly that, and a two-sided version would halve the power for a direction nobody is
    monitoring.
    """
    x = np.asarray(xs, dtype=np.float64).ravel()
    out = np.empty(x.size, dtype=np.float64)
    capital = 1.0
    eta = 2.0 / (2.0 - math.log(3.0))
    a_sum = 1.0
    current = 0.0 if adaptive else lam
    for i, xi in enumerate(x):
        if not math.isfinite(xi):
            out[i] = capital
            continue
        factor = bounded_evalue_increment(float(xi), mean_null, current)
        if factor <= 0.0:
            capital = 0.0
            out[i] = capital
            continue
        capital *= factor
        out[i] = capital
        if adaptive:
            g = (float(xi) - mean_null) / factor
            a_sum += g * g
            current = float(np.clip(current + eta * g / a_sum, -0.5, 0.5))
    return out


__all__ = [
    "DEFAULT_V_OPT",
    "EBHResult",
    "MergedEValue",
    "PeekingCost",
    "bounded_evalue_increment",
    "ebh",
    "fixed_sample_radius",
    "merge_e",
    "peeking_miscoverage",
    "wilson_interval",
    "running_evalue",
    "stitched_radius",
    "ville_pvalue",
    "ville_threshold",
]

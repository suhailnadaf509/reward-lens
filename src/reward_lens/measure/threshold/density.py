"""I1: is the density of the running variable discontinuous at the gate?

**A hard reward threshold is a sharp regression discontinuity with a deterministic, perfectly
measured assignment rule.** Everything the RD literature spends its effort on is free here. There
is no fuzzy compliance, because the rule is code. There is no measurement error in the running
variable, because the trainer wrote it down. There is no unknown cutoff, because it is a constant
in a config file. That is the ideal case for a set of tools that normally has to work much harder,
and two independent literature searches found nothing pointing them at reward gates.

The test itself is McCrary (2008). Under continuity of the counterfactual density at the cutoff,
the density of the running variable is smooth there; a policy that manipulates the running variable
to stay on the profitable side of a gate piles mass just below it and empties the region just
above, and the density jumps. The reading is a log difference in the density across the cutoff and
its standard error, reported as a z statistic.

What this cannot do, stated where it belongs rather than on a caveats page:

- It is a **local** test. It says nothing about the density anywhere except within one bandwidth of
  the cutoff, and a policy that responds to a gate by changing its whole length distribution
  without piling up at the boundary produces a smooth density and a z of zero.
- A discontinuity is evidence of **sorting**, not of intent. A sampler that stops at a cap, a
  tokenizer that cannot emit a partial word, and a task distribution with a mode at the cutoff all
  produce the same jump. `DecodeLength` exists so the first of those cannot be missed, and the
  placebo baseline is what separates the rest from an artifact of the estimator.
- The standard error is asymptotic and assumes the bins are Poisson. The smooth-density null
  baseline measures whether that holds on the data in hand, and **the reading is refused with
  `ENVELOPE_VIOLATED` when it does not**: a band whose centre sits more than
  `MAX_NULL_CENTRE_SPREADS` of its own spreads from zero, or whose spread is off the implied 1 by
  more than a factor of `MAX_NULL_SPREAD_RATIO`, is the estimator saying its premise fails on this
  density. The refusal carries both numbers and the statistic it would have reported. This is the
  one condition here that is a precondition on the subject rather than on the run, which is why it
  refuses in the arithmetic rather than through a regime condition in `GATE_ENVELOPE`.

Two baselines run on every reading and neither is optional. A **smooth-density null** draws samples
from a single polynomial fitted across the whole support, which cannot be discontinuous anywhere,
and reruns the test on them; that gives the null distribution of z under this sample size, this
binwidth and this bandwidth rather than under the asymptotic theory. **The same test at a placebo
cutoff** is the one that separates a finding from an artifact: an estimator that reports a large z
wherever you point it has told you about itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from reward_lens.core.evidence import register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BaselineID
from reward_lens.core.reading import Refusal, RefusalReason, refuse_incomplete
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.threshold._base import (
    ALL_SUBSTRATES,
    GATE_ACCESS,
    GATE_ENVELOPE,
    RECORD_PHASES,
    ThresholdInstrument,
)
from reward_lens.measure.threshold.gates import DecodeLength, Gate, RunningVariable

#: The two the catalogue names for I1, and both run on every reading.
DENSITY_BASELINES: tuple[BaselineID, ...] = (
    "baseline.smooth_density_null",
    "baseline.placebo_cutoff",
)

#: McCrary's constant in the automatic bandwidth rule, from the AMISE-optimal local linear boundary
#: estimator with a triangular kernel. Not a tuning parameter: it falls out of the derivation.
_MCCRARY_C = 3.348

#: The asymptotic variance constant for the log density difference under a triangular kernel,
#: 24/5. Also not a tuning parameter.
_VAR_CONST = 24.0 / 5.0

#: The fewest bins with positive kernel weight on one side that will support a local linear fit
#: with an intercept and a slope. Two points fit a line exactly and leave no residual, so three is
#: the smallest number at which the fit is a fit.
_MIN_BINS_PER_SIDE = 3

#: How far the smooth-density null's centre may sit from zero before the reading is refused, in
#: units of the band's own spread.
#:
#: The asymptotic null this test reports its z and its p against is N(0, 1). The smooth-density
#: baseline measures what the null actually is on the density in hand, and when its centre sits
#: many of its own spreads away from zero the estimator is biased here and the reported statistic
#: is mostly that bias. Three spreads is the working tolerance: at the 300 draws the baseline takes
#: by default the band's own centre is estimated to about a sixth of a spread, so three spreads is
#: far outside what the baseline's sampling noise can produce.
MAX_NULL_CENTRE_SPREADS = 3.0

#: How far the smooth-density null's spread may sit from the implied 1 before the reading is
#: refused, as a ratio in either direction.
#:
#: A band wider than 1 makes the reported p anti-conservative and a band narrower than 1 makes it
#: conservative, and both are the asymptotic formula being wrong about this density rather than a
#: property of the run. A factor of 1.5 is where a nominal p of 0.05 becomes about 0.19, which is
#: the point at which a reader acting on the printed p is acting on a different number.
MAX_NULL_SPREAD_RATIO = 1.5


# ---------------------------------------------------------------------------
# binning
# ---------------------------------------------------------------------------


def automatic_binsize(x: np.ndarray) -> float:
    """McCrary's first-step binwidth, ``2 * sd(x) * n**(-1/2)``."""
    finite = x[np.isfinite(x)]
    n = finite.size
    if n < 2:
        return float("nan")
    sd = float(np.std(finite, ddof=1))
    return float(2.0 * sd * n**-0.5) if sd > 0 else float("nan")


@dataclass(frozen=True)
class Histogram:
    """The first-step histogram, with the cutoff on a bin edge.

    The alignment is the part that is easy to get wrong and impossible to see afterwards. If a bin
    straddles the cutoff it contains observations from both sides, and the local linear fit that is
    supposed to approach the cutoff from one side is fitted to a point that averages over both. The
    jump is then attenuated by exactly the fraction of the bin on the wrong side.
    """

    midpoints: np.ndarray
    density: np.ndarray
    counts: np.ndarray
    binsize: float
    cutoff: float
    n: int

    @property
    def left(self) -> np.ndarray:
        return self.midpoints < self.cutoff

    @property
    def right(self) -> np.ndarray:
        return self.midpoints >= self.cutoff


def histogram(x: Sequence[float] | np.ndarray, cutoff: float, binsize: float) -> Histogram:
    """Bin ``x`` with the cutoff on a bin edge and normalise so the bin heights integrate to one."""
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0 or not math.isfinite(binsize) or binsize <= 0:
        empty = np.zeros(0, dtype=np.float64)
        return Histogram(empty, empty, empty, float(binsize), float(cutoff), 0)
    lo = int(math.floor((v.min() - cutoff) / binsize))
    hi = int(math.ceil((v.max() - cutoff) / binsize))
    edges = cutoff + binsize * np.arange(lo, hi + 1, dtype=np.float64)
    if edges.size < 2:
        edges = np.asarray([cutoff - binsize, cutoff, cutoff + binsize], dtype=np.float64)
    counts, _ = np.histogram(v, bins=edges)
    midpoints = edges[:-1] + binsize / 2.0
    density = counts.astype(np.float64) / (n * binsize)
    return Histogram(
        midpoints, density, counts.astype(np.float64), float(binsize), float(cutoff), n
    )


def automatic_bandwidth(hist: Histogram) -> float:
    """McCrary's automatic bandwidth: a fourth-order fit per side, then the AMISE rule.

    A fourth-order polynomial is fitted to the binned density on each side, its implied second
    derivative is evaluated at every bin on that side, and the rule
    ``3.348 * (sigma^2 * range / sum(f'')^2)**(1/5)`` gives a bandwidth for that side. The two are
    averaged, which is what the reference implementation does and which matters when the density is
    much flatter on one side than the other.
    """
    hs: list[float] = []
    for mask in (hist.left, hist.right):
        u = hist.midpoints[mask] - hist.cutoff
        y = hist.density[mask]
        if u.size < 6:
            continue
        design = np.vander(u, 5, increasing=True)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        dof = max(1, u.size - 5)
        sigma2 = float(resid @ resid) / dof
        second = 2.0 * coef[2] + 6.0 * coef[3] * u + 12.0 * coef[4] * u**2
        denom = float(second @ second)
        span = float(u.max() - u.min())
        if denom <= 0 or span <= 0 or sigma2 <= 0:
            continue
        hs.append(_MCCRARY_C * (sigma2 * span / denom) ** 0.2)
    if not hs:
        return float("nan")
    return float(np.mean(hs))


def _local_linear(u: np.ndarray, y: np.ndarray, h: float) -> tuple[float, int]:
    """Triangular-kernel local linear fit at ``u = 0``. Returns the intercept and the bins used."""
    w = np.clip(1.0 - np.abs(u) / h, 0.0, None)
    keep = w > 0
    if int(keep.sum()) < _MIN_BINS_PER_SIDE:
        return float("nan"), int(keep.sum())
    uu, yy, ww = u[keep], y[keep], w[keep]
    design = np.column_stack([np.ones_like(uu), uu])
    sqrt_w = np.sqrt(ww)
    coef, *_ = np.linalg.lstsq(design * sqrt_w[:, None], yy * sqrt_w, rcond=None)
    return float(coef[0]), int(keep.sum())


# ---------------------------------------------------------------------------
# the readings
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class NullBand:
    """The null distribution of the statistic, measured rather than assumed."""

    label: str
    n_draws: int
    mean: float
    sd: float
    q95_abs: float
    p_empirical: float
    detail: str = ""

    def render(self) -> str:
        return (
            f"{self.label}: {self.n_draws} draws, z ~ ({self.mean:+.3f} +/- {self.sd:.3f}), "
            f"95th pct |z| {self.q95_abs:.3f}, empirical p {self.p_empirical:.4g}"
            + (f". {self.detail}" if self.detail else "")
        )


@register_payload
@dataclass(frozen=True)
class McCraryReading:
    """The log density difference at the cutoff, its standard error, and its two baselines."""

    gate: Gate
    running: str
    unit: str
    n: int
    cutoff: float
    binsize: float
    bandwidth: float
    density_left: float
    density_right: float
    theta: float
    se: float
    z: float
    p: float
    bins_left: int
    bins_right: int
    rung: int
    estimator: str
    smooth_null: NullBand | None = None
    placebo: NullBand | None = None
    placebo_cutoffs: tuple[float, ...] = ()
    placebo_z: tuple[float, ...] = ()
    decode: DecodeLength | None = None

    @property
    def says(self) -> str:
        direction = "above" if self.theta > 0 else "below"
        return (
            f"the density of {self.running} is {math.exp(abs(self.theta)):.2f} times higher "
            f"{direction} {self.cutoff:g} {self.unit} than on the other side of it "
            f"(z = {self.z:+.2f}, p = {self.p:.3g})"
        )

    def render(self) -> str:
        lines = [
            f"I1 McCrary  {self.gate.render()}",
            f"  {self.says}",
            f"  log difference {self.theta:+.4f} +/- {self.se:.4f}; "
            f"f- = {self.density_left:.5g}, f+ = {self.density_right:.5g}",
            f"  n = {self.n}, binsize {self.binsize:.4g}, bandwidth {self.bandwidth:.4g}, "
            f"bins used {self.bins_left} left / {self.bins_right} right, rung {self.rung} "
            f"({self.estimator})",
        ]
        if self.smooth_null is not None:
            lines.append(f"  baseline {self.smooth_null.render()}")
        if self.placebo is not None:
            lines.append(f"  baseline {self.placebo.render()}")
        if self.decode is not None:
            lines.append(f"  {self.decode.render()}")
        if self.gate.installed:
            lines.append(
                "  the gate is installed, so this is a property of the estimator and of this "
                "density, not evidence that any policy manipulated anything."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# rung 0 and rung 1
# ---------------------------------------------------------------------------


def mccrary(
    x: Sequence[float] | np.ndarray,
    cutoff: float,
    *,
    binsize: float | None = None,
    bandwidth: float | None = None,
) -> tuple[float, float, float, float, float, int, int, float, float]:
    """Rung 0. Returns theta, se, z, f_left, f_right, bins_left, bins_right, binsize, bandwidth.

    A tuple rather than a payload because both rungs and every baseline call it in a loop, and the
    payload is assembled once at the end by `density_discontinuity`.
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    b = binsize if binsize is not None else automatic_binsize(v)
    hist = histogram(v, cutoff, b)
    h = bandwidth if bandwidth is not None else automatic_bandwidth(hist)
    if not math.isfinite(h) or h <= 0:
        nan = float("nan")
        return nan, nan, nan, nan, nan, 0, 0, float(b), nan
    u = hist.midpoints - cutoff
    f_left, n_left = _local_linear(u[hist.left], hist.density[hist.left], h)
    f_right, n_right = _local_linear(u[hist.right], hist.density[hist.right], h)
    if not (math.isfinite(f_left) and math.isfinite(f_right)) or f_left <= 0 or f_right <= 0:
        nan = float("nan")
        return nan, nan, nan, f_left, f_right, n_left, n_right, float(b), float(h)
    theta = math.log(f_right) - math.log(f_left)
    se = math.sqrt((1.0 / (hist.n * h)) * _VAR_CONST * (1.0 / f_right + 1.0 / f_left))
    z = theta / se if se > 0 else float("nan")
    return theta, se, z, f_left, f_right, n_left, n_right, float(b), float(h)


def _cdf_local_poly(v: np.ndarray, cutoff: float, h: float, order: int) -> tuple[float, float]:
    """Cattaneo-Jansson-Ma: local polynomial regression of the empirical CDF, per side.

    The density is the coefficient on the linear term of a weighted polynomial fit of the empirical
    distribution function against the running variable, done separately on each side of the cutoff
    with a triangular kernel. Fitting the CDF rather than a histogram is what removes the binwidth
    from the estimator entirely, which is the point of the newer estimator: two analysts who choose
    different bins get the same answer.
    """
    order_terms = order + 1
    sorted_v = np.sort(v)
    cdf = np.searchsorted(sorted_v, v, side="right") / v.size
    out: list[float] = []
    for mask in (v < cutoff, v >= cutoff):
        u = v[mask] - cutoff
        y = cdf[mask]
        w = np.clip(1.0 - np.abs(u) / h, 0.0, None)
        keep = w > 0
        if int(keep.sum()) <= order_terms:
            out.append(float("nan"))
            continue
        uu, yy, ww = u[keep], y[keep], w[keep]
        design = np.vander(uu, order_terms, increasing=True)
        sqrt_w = np.sqrt(ww)
        coef, *_ = np.linalg.lstsq(design * sqrt_w[:, None], yy * sqrt_w, rcond=None)
        out.append(float(coef[1]))
    return out[0], out[1]


def mccrary_robust(
    x: Sequence[float] | np.ndarray,
    cutoff: float,
    *,
    bandwidth: float | None = None,
    order: int = 2,
    n_boot: int = 400,
    seed: int = 0,
) -> tuple[float, float, float, float, float, float]:
    """Rung 1: the local polynomial density estimator with robust bias correction.

    The returned point estimate is the order-``p+1`` fit, which is the robust-bias-correction
    construction: rather than estimating the order-``p`` fit's leading bias term and subtracting it,
    fit one order higher and take that fit's own variance, so the interval is valid for the
    bias-corrected point estimate instead of for the uncorrected one. ``order`` names ``p``.

    **Deviation from Cattaneo, Jansson and Ma (2020), stated because it is a real one.** Their
    variance is analytic and this is a nonparametric bootstrap over rollouts. The bootstrap is
    honest about the finite sample and costs 400 small least-squares fits; the analytic form would
    be exact asymptotically and is not implemented here. Where the two disagree the bootstrap is the
    wider of the two on every sample this was checked on, so the reported z is the conservative one.
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    h = bandwidth
    if h is None:
        h = automatic_bandwidth(histogram(v, cutoff, automatic_binsize(v)))
    if not math.isfinite(h) or h <= 0:
        nan = float("nan")
        return nan, nan, nan, nan, nan, nan
    f_left, f_right = _cdf_local_poly(v, cutoff, h, order + 1)
    if not (math.isfinite(f_left) and math.isfinite(f_right)) or f_left <= 0 or f_right <= 0:
        nan = float("nan")
        return nan, nan, nan, f_left, f_right, float(h)
    theta = math.log(f_right) - math.log(f_left)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(int(n_boot)):
        sample = rng.choice(v, size=v.size, replace=True)
        left, right = _cdf_local_poly(sample, cutoff, h, order + 1)
        if math.isfinite(left) and math.isfinite(right) and left > 0 and right > 0:
            draws.append(math.log(right) - math.log(left))
    if len(draws) < 20:
        nan = float("nan")
        return theta, nan, nan, f_left, f_right, float(h)
    se = float(np.std(draws, ddof=1))
    z = theta / se if se > 0 else float("nan")
    return theta, se, z, f_left, f_right, float(h)


# ---------------------------------------------------------------------------
# the two mandatory baselines
# ---------------------------------------------------------------------------


def smooth_density_null(
    x: Sequence[float] | np.ndarray,
    cutoff: float,
    *,
    observed_z: float,
    n_draws: int = 300,
    binsize: float | None = None,
    bandwidth: float | None = None,
    degree: int = 7,
    seed: int = 0,
) -> NullBand:
    """Draw from the smoothest density of this shape, and rerun the test on the draws.

    The null density is a single polynomial in the log of the binned counts, fitted across the
    **whole** support with no break at the cutoff. One polynomial cannot be discontinuous anywhere,
    so a sample drawn from it has no jump at the cutoff by construction, while keeping the shape,
    the sample size and the support of the data in hand. Running the same test on those samples
    gives the null distribution of z for this n, this binwidth and this bandwidth, which is what
    the asymptotic standard error claims to be and is worth checking rather than assuming.

    A smoothed bootstrap is the obvious alternative and it is the wrong one here: jittering the
    observed points by a kernel narrower than the bunching region leaves the bunching in, so the
    "null" inherits the very discontinuity it is meant to exclude. On a planted spike that showed
    up as a null centred at z = -5.9 rather than at zero, which would have made a real detection
    look like an estimator artifact.

    A null band whose sd is far from 1 is not a failure of the test. It is the finite-sample
    correction the asymptotic formula does not carry, and the honest reading of a z of 3 against a
    null sd of 1.8 is a p near 0.1 rather than near 0.003.
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    n = v.size
    if n < 20:
        return NullBand(
            "smooth-density null",
            0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            detail="fewer than 20 observations; a resampled null would be noise",
        )
    b = binsize if binsize is not None else automatic_binsize(v)
    hist = histogram(v, cutoff, b)
    if hist.counts.size < degree + 2:
        return NullBand(
            "smooth-density null",
            0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            detail=f"{hist.counts.size} bins cannot support a degree-{degree} smooth fit",
        )
    u = hist.midpoints - cutoff
    scale = float(np.max(np.abs(u))) or 1.0
    design = np.vander(u / scale, degree + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(design, np.log(hist.counts + 0.5), rcond=None)
    weights = np.exp(design @ coef)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        return NullBand(
            "smooth-density null",
            0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            detail="the smooth fit did not produce a usable density",
        )
    pmf = weights / total
    rng = np.random.default_rng(seed)
    zs: list[float] = []
    for _ in range(int(n_draws)):
        picked = rng.choice(hist.midpoints.size, size=n, replace=True, p=pmf)
        sample = hist.midpoints[picked] + rng.uniform(-b / 2.0, b / 2.0, size=n)
        z = mccrary(sample, cutoff, binsize=b, bandwidth=bandwidth)[2]
        if math.isfinite(z):
            zs.append(z)
    if len(zs) < 20:
        return NullBand(
            "smooth-density null",
            len(zs),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            detail="the test did not return a finite z on enough draws to form a null",
        )
    arr = np.asarray(zs, dtype=np.float64)
    p = (
        float(np.mean(np.abs(arr) >= abs(observed_z)))
        if math.isfinite(observed_z)
        else float("nan")
    )
    return NullBand(
        label="smooth-density null",
        n_draws=len(zs),
        mean=float(arr.mean()),
        sd=float(arr.std(ddof=1)),
        q95_abs=float(np.percentile(np.abs(arr), 95)),
        p_empirical=p,
        detail=(
            f"samples of {n} drawn from a degree-{degree} log-density fitted across the whole "
            f"support; the asymptotic SE would imply a null sd of 1.0"
        ),
    )


def null_band_failure(band: NullBand | None) -> tuple[str, dict[str, float]] | None:
    """Whether the measured null says the asymptotic approximation does not hold here.

    Returns None when the band is consistent with N(0, 1), and otherwise a sentence naming what
    failed with its numbers, plus the statistics behind it. Two ways to fail and they are checked
    separately because they are different faults: a displaced centre is bias, which standardising
    against the band cannot repair, and a spread away from 1 is the asymptotic variance being wrong
    about this density, which makes the printed p a different number from the one it claims to be.

    A band that could not be measured at all (fewer than 20 usable draws, which is what
    :func:`smooth_density_null` returns a NaN band for) is not treated as a failure here. There is
    then no evidence either way, and turning "the baseline could not run" into "the premise is
    violated" would refuse readings for a reason nobody measured.
    """
    if band is None or band.n_draws == 0:
        return None
    if not (math.isfinite(band.mean) and math.isfinite(band.sd)) or band.sd <= 0.0:
        return None

    spreads = abs(band.mean) / band.sd
    stats = {
        "null_mean": float(band.mean),
        "null_sd": float(band.sd),
        "null_draws": float(band.n_draws),
        "null_centre_spreads": float(spreads),
        "max_null_centre_spreads": float(MAX_NULL_CENTRE_SPREADS),
        "max_null_spread_ratio": float(MAX_NULL_SPREAD_RATIO),
    }
    faults: list[str] = []
    if spreads > MAX_NULL_CENTRE_SPREADS:
        faults.append(
            f"its centre is {spreads:.1f} of its own spreads from zero, past the "
            f"{MAX_NULL_CENTRE_SPREADS:g} this instrument accepts, so most of the reported "
            f"statistic is the estimator's bias on this density"
        )
    if band.sd > MAX_NULL_SPREAD_RATIO or band.sd < 1.0 / MAX_NULL_SPREAD_RATIO:
        nominal = math.erfc(1.959963985 / (band.sd * math.sqrt(2.0)))
        stats["p_at_nominal_five_percent"] = float(nominal)
        faults.append(
            f"its spread is {band.sd:.2f} against the implied 1, past the factor of "
            f"{MAX_NULL_SPREAD_RATIO:g} this instrument accepts, so a printed p of 0.05 on this "
            f"density is really near {nominal:.2f}"
        )
    if not faults:
        return None
    return (" and ".join(faults), stats)


def placebo_cutoffs(
    x: Sequence[float] | np.ndarray,
    cutoff: float,
    *,
    observed_z: float,
    n_placebos: int = 40,
    exclude_within: float | None = None,
    binsize: float | None = None,
    bandwidth: float | None = None,
) -> tuple[NullBand, tuple[float, ...], tuple[float, ...]]:
    """The same test everywhere else in the support. The baseline that separates a finding.

    Placebo cutoffs are laid on an even grid between the 2.5th and 97.5th percentiles of the
    running variable, with a neighbourhood of the real cutoff removed so the real discontinuity
    does not leak into its own null. The neighbourhood is one bandwidth, which is the smallest
    principled choice: a placebo closer than that has the real jump inside its own kernel and is
    not a placebo. Reporting the real cutoff's rank among them is randomisation inference, it needs
    no distributional assumption at all, and it is the reading to trust when the smooth null and
    the asymptotic standard error disagree.
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size < 20:
        band = NullBand(
            "placebo cutoff",
            0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            detail="fewer than 20 observations",
        )
        return band, (), ()
    b = binsize if binsize is not None else automatic_binsize(v)
    h = bandwidth
    if h is None:
        h = automatic_bandwidth(histogram(v, cutoff, b))
    keep_out = exclude_within if exclude_within is not None else (h if math.isfinite(h) else b)
    lo, hi = np.percentile(v, [2.5, 97.5])
    grid = np.linspace(float(lo), float(hi), int(n_placebos) + 2)[1:-1]
    grid = grid[np.abs(grid - cutoff) > keep_out]
    zs: list[float] = []
    used: list[float] = []
    for c in grid:
        z = mccrary(v, float(c), binsize=b, bandwidth=h)[2]
        if math.isfinite(z):
            zs.append(z)
            used.append(float(c))
    if len(zs) < 5:
        band = NullBand(
            "placebo cutoff",
            len(zs),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            detail="fewer than 5 placebo cutoffs returned a finite z",
        )
        return band, tuple(used), tuple(zs)
    arr = np.asarray(zs, dtype=np.float64)
    p = (
        float(np.mean(np.abs(arr) >= abs(observed_z)))
        if math.isfinite(observed_z)
        else float("nan")
    )
    band = NullBand(
        label="placebo cutoff",
        n_draws=len(zs),
        mean=float(arr.mean()),
        sd=float(arr.std(ddof=1)),
        q95_abs=float(np.percentile(np.abs(arr), 95)),
        p_empirical=p,
        detail=(
            f"same test at {len(zs)} cutoffs across the support, excluding within {keep_out:.4g} "
            f"of the real one"
        ),
    )
    return band, tuple(used), tuple(zs)


# ---------------------------------------------------------------------------
# the estimator with its refusals
# ---------------------------------------------------------------------------


def density_discontinuity(
    running: RunningVariable,
    gate: Gate,
    *,
    rung: int = 1,
    instrument: str = "DensityDiscontinuity",
    binsize: float | None = None,
    bandwidth: float | None = None,
    n_boot: int = 400,
    n_null: int = 300,
    n_placebos: int = 40,
    decode: DecodeLength | None = None,
    seed: int = 0,
) -> McCraryReading | Refusal:
    """I1's reading, or the refusal that says why there is none."""
    v = np.asarray(running.values, dtype=np.float64).ravel()
    finite = v[np.isfinite(v)]
    if finite.size < 20:
        return refuse_incomplete(
            instrument,
            field=f"{running.name} on more than {finite.size} of {v.size} rollouts",
            subject="the record",
            remedy=(
                f"record {running.name} per rollout. For a token-count running variable that means "
                f"writing `token_ids` or a token count onto each turn; the converter reads it from "
                f"there and no discontinuity test can run without it."
            ),
            n=int(v.size),
            n_finite=int(finite.size),
        )

    distinct = int(np.unique(finite).size)
    below = int((finite < gate.cutoff).sum())
    above = int((finite >= gate.cutoff).sum())
    if distinct < _MIN_BINS_PER_SIDE * 2 or below == 0 or above == 0:
        return refuse_incomplete(
            instrument,
            field=f"variation in {running.name} on both sides of {gate.cutoff:g}",
            subject=(
                f"{finite.size} rollouts taking {distinct} distinct values, {below} below the "
                f"cutoff and {above} at or above it"
            ),
            remedy=(
                "raise the sampler's completion cap above the gate and re-run, so the running "
                "variable has support on both sides of the cutoff. A running variable that is a "
                "point mass has no density for a discontinuity to be in, and no reweighting of "
                "this record recovers one."
            ),
            n=int(finite.size),
            n_distinct=distinct,
            n_below=below,
            n_above=above,
            cutoff=float(gate.cutoff),
        )

    theta, se, z, f_left, f_right, n_left, n_right, b, h = mccrary(
        finite, gate.cutoff, binsize=binsize, bandwidth=bandwidth
    )
    estimator = "local linear on the binned density, McCrary (2008)"
    if not math.isfinite(theta):
        thin = min(n_left, n_right) < _MIN_BINS_PER_SIDE
        if thin:
            detail = (
                f"the local linear fit has {n_left} usable bins below the cutoff and {n_right} at "
                f"or above it, at binwidth {b:.4g} and bandwidth {h:.4g}. A density estimated from "
                f"fewer than {_MIN_BINS_PER_SIDE} bins on a side is a line through two points."
            )
            remedy = (
                "widen the bandwidth explicitly with `bandwidth=`, or collect more rollouts near "
                "the cutoff. The automatic bandwidth is chosen for the whole support and is too "
                "narrow when the density near this cutoff is thin."
            )
        else:
            detail = (
                f"the local linear fit puts the density at f- = {f_left:.5g} and f+ = "
                f"{f_right:.5g} using {n_left} and {n_right} bins at bandwidth {h:.4g}. The "
                f"statistic is a difference of logs, and a fitted density at or below zero means "
                f"the linear approximation has extrapolated past the data rather than that the "
                f"density is zero."
            )
            remedy = (
                f"test a cutoff further inside the support of {running.name}, or widen the "
                f"bandwidth with `bandwidth=` so the fit has data on both sides of it. At "
                f"{gate.cutoff:g} the sample carries {below} rollouts below and {above} at or "
                f"above, which is thin enough for a local linear fit to go negative."
            )
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=detail,
            remedy=remedy,
            statistics={
                "bins_left": n_left,
                "bins_right": n_right,
                "binsize": b,
                "bandwidth": h,
                "density_left": f_left,
                "density_right": f_right,
                "n_below": below,
                "n_above": above,
            },
        )

    if rung >= 1:
        r_theta, r_se, r_z, r_left, r_right, r_h = mccrary_robust(
            finite, gate.cutoff, bandwidth=h, n_boot=n_boot, seed=seed
        )
        if math.isfinite(r_z):
            theta, se, z = r_theta, r_se, r_z
            f_left, f_right, h = r_left, r_right, r_h
            estimator = (
                "local polynomial on the empirical CDF with robust bias correction, "
                "Cattaneo-Jansson-Ma (2020), bootstrap variance"
            )
        else:
            rung = 0

    p = math.erfc(abs(z) / math.sqrt(2.0)) if math.isfinite(z) else float("nan")
    null = smooth_density_null(
        finite, gate.cutoff, observed_z=z, n_draws=n_null, binsize=b, bandwidth=h, seed=seed + 1
    )

    # The premise, checked against the instrument's own baseline rather than assumed. McCrary's
    # identifying assumption is usually stated as continuity of the counterfactual density at the
    # cutoff, but what the local linear estimator needs is for the density to be close to linear
    # over the bandwidth, and on a running variable spanning orders of magnitude that is a much
    # stronger requirement. The smooth-density null measures it directly: it draws from a density
    # of this shape that cannot be discontinuous anywhere, so anything it reports other than
    # N(0, 1) is the estimator rather than the run.
    #
    # This refuses rather than downgrades. Measured on 25,664 real completion lengths with no gate
    # anywhere in them, the full-range test returned z from -50.4 to -76.3 with p below any
    # printable floor at three cutoffs, and standardising against the band left the statistic
    # 23.5 to 31.3 spreads out, so a reader who treated the band as a correction would have
    # published a gate anyway. A baseline that detects a violated premise and does not repair it is
    # a reason to withhold the reading, not a reason to annotate it.
    failure = null_band_failure(null)
    if failure is not None:
        why, stats = failure
        lo, hi = float(finite.min()), float(finite.max())
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ENVELOPE_VIOLATED,
            detail=(
                f"the smooth-density null at {gate.cutoff:g} {running.unit} comes back as "
                f"({null.mean:+.3f} +/- {null.sd:.3f}) where the normal approximation this test "
                f"reports its z and its p against implies (0.000 +/- 1.000): {why}. The local "
                f"linear fit runs at a bandwidth of {h:.4g} {running.unit} on {finite.size} values "
                f"of {running.name} spanning {lo:g} to {hi:g}, and over that range the density is "
                f"not close enough to linear for the asymptotics to hold. The statistic it would "
                f"have reported is z = {z:+.3f}, p = {p:.3g}, and on this density that is a "
                f"measurement of the fit rather than of the policy."
            ),
            remedy=(
                f"restrict {running.name} to a range over which its density is locally smooth and "
                f"re-run: build the RunningVariable from the values inside that range and keep the "
                f"cutoff inside it. On 25,664 recorded completion lengths spanning 9 to 26,870 "
                f"characters, restricting to [200, 2000] brought the null band to a centre within "
                f"0.25 of zero and a spread within 0.03 of 1, and the same three cutoffs then "
                f"returned |z| below 1.61. That interval is a worked example rather than a "
                f"default: the range that works is a property of your own density, and the check "
                f"to run on a candidate range is this same null band."
            ),
            statistics={
                **stats,
                "z": float(z),
                "p": float(p),
                "theta": float(theta),
                "se": float(se),
                "cutoff": float(gate.cutoff),
                "bandwidth": float(h),
                "binsize": float(b),
                "n": float(finite.size),
                "rung": float(rung),
                "running_min": lo,
                "running_max": hi,
            },
        )

    band, used, placebo_z = placebo_cutoffs(
        finite, gate.cutoff, observed_z=z, n_placebos=n_placebos, binsize=b, bandwidth=h
    )
    return McCraryReading(
        gate=gate,
        running=running.name,
        unit=running.unit,
        n=int(finite.size),
        cutoff=float(gate.cutoff),
        binsize=float(b),
        bandwidth=float(h),
        density_left=float(f_left),
        density_right=float(f_right),
        theta=float(theta),
        se=float(se),
        z=float(z),
        p=float(p),
        bins_left=int(n_left),
        bins_right=int(n_right),
        rung=int(rung),
        estimator=estimator,
        smooth_null=null,
        placebo=band,
        placebo_cutoffs=used,
        placebo_z=placebo_z,
        decode=decode,
    )


# ---------------------------------------------------------------------------
# the instrument
# ---------------------------------------------------------------------------


class DensityDiscontinuity(ThresholdInstrument):
    """I1. Whether the density of the running variable jumps at the gate.

    Kill condition, from the catalogue record: if no real pipeline has a hard gate, which is false.
    """

    name = "DensityDiscontinuity"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "I1"
    deviations = (
        "rung 1's variance is a nonparametric bootstrap over rollouts rather than "
        "Cattaneo-Jansson-Ma's analytic asymptotic variance. The point estimate and the bias "
        "correction are theirs; the interval is wider and finite-sample",
        "the bandwidth is McCrary's automatic rule computed on the binned density, and it is "
        "reused for the rung-1 estimator rather than re-selected by the CJM MSE-optimal rule. "
        "Both are reported on the reading so a reader can see which one produced the number",
        "the p value is the two-sided normal tail of z. The smooth-density null baseline measures "
        "whether that reference distribution holds on the sample in hand and reports the empirical "
        "p beside it; where the two disagree the empirical one is the reading",
    )

    quantity = "gate.mccrary_statistic"
    requires = GATE_ACCESS
    substrates = ALL_SUBSTRATES
    phases = RECORD_PHASES
    envelope = GATE_ENVELOPE
    #: `units` in the registry, whose assertion is a refusal rather than a numeric relation. A z on
    #: a density in tokens and a z on a density in characters are not the same quantity, and
    #: `check_invariance` routes this group to `check_unit_refusal`.
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = DENSITY_BASELINES
    rung = 1

    def __init__(
        self,
        running: RunningVariable | None = None,
        gate: Gate | None = None,
        *,
        rung: int = 1,
        binsize: float | None = None,
        bandwidth: float | None = None,
        n_boot: int = 400,
        n_null: int = 300,
        n_placebos: int = 40,
        decode: DecodeLength | None = None,
        seed: int = 0,
    ) -> None:
        self.running = running
        self.gate = gate
        self.rung = int(rung)
        self.binsize = binsize
        self.bandwidth = bandwidth
        self.n_boot = int(n_boot)
        self.n_null = int(n_null)
        self.n_placebos = int(n_placebos)
        self.decode = decode
        self.seed = int(seed)

    def compute(self) -> McCraryReading | Refusal:
        if self.running is None or self.gate is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no running variable or no gate was supplied, so there is nothing to test",
                remedy=(
                    "pass `running=` a RunningVariable and `gate=` a Gate. "
                    "`measure.threshold.gates.completion_lengths(run)` builds the first from a "
                    "record and `recorded_gates(...)` reads the second off its composition trees."
                ),
            )
        return density_discontinuity(
            self.running,
            self.gate,
            rung=self.rung,
            instrument=self.name,
            binsize=self.binsize,
            bandwidth=self.bandwidth,
            n_boot=self.n_boot,
            n_null=self.n_null,
            n_placebos=self.n_placebos,
            decode=self.decode,
            seed=self.seed,
        )


__all__ = [
    "DENSITY_BASELINES",
    "MAX_NULL_CENTRE_SPREADS",
    "MAX_NULL_SPREAD_RATIO",
    "DensityDiscontinuity",
    "Histogram",
    "McCraryReading",
    "NullBand",
    "automatic_bandwidth",
    "automatic_binsize",
    "density_discontinuity",
    "histogram",
    "mccrary",
    "mccrary_robust",
    "null_band_failure",
    "placebo_cutoffs",
    "smooth_density_null",
]

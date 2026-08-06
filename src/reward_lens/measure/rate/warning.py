"""The early-warning statistics, with the nulls that make them evidence rather than decoration.

Rising variance and rising lag-1 autocorrelation are the standard pair for critical slowing down,
and on their own they are not enough. A rolling autocorrelation computed on an autocorrelated series
drifts upward for reasons that have nothing to do with an approaching bifurcation. Four
requirements follow from that, and this module implements all four:

**A null built from surrogate series.** Kendall's tau of the indicator against time is compared
against the same statistic computed on surrogates that preserve the spectrum and destroy the trend.
Two constructions are offered because they fail differently: the Fourier surrogate keeps the
periodogram exactly and randomises the phases, and the first-order autoregressive bootstrap keeps
only the lag-1 structure and is the narrower null. **Reporting a rising AC(1) with no null attached
is not evidence**, and `trend_significance` is the only entry point here that returns a tau.

**Flickering by bimodality rather than by a moment.** A system rattling between two attractors is
bimodal, and skewness and kurtosis both move for a dozen other reasons. `flickering` compares a
two-component Gaussian mixture against one component by BIC, and calibrates the difference against a
parametric bootstrap under the fitted single component so the answer is a p-value rather than a
rule of thumb.

**The relaxation time against the driver timescale.** `driver_comparison` is the adiabaticity check
arrived at from the other side: an early autoregressive fit gives a relaxation time, the schedule
gives a driving rate, and the ratio is `Ad`. It calls `adiabaticity_number` rather than recomputing,
because there is one definition of that number in this library.

**A noise control and a window sweep.** `noise_control` runs the identical pipeline on a channel
that has no business responding to the transition; `window_sensitivity` runs it at several window
lengths. Both exist because a rolling statistic has one free parameter and one arbitrary channel,
and a result that survives neither sweep is a result about the choices.

Two caveats, carried here rather than in a page nobody opens.

**There are whole classes of systems that always show early-warning signals and never feature a
critical transition.** A rising autocorrelation is consistent with an approaching bifurcation and it
is also consistent with the noise becoming redder for its own reasons, and no amount of surrogate
testing separates those, because the surrogate null answers "is the trend real" and not "is the
trend a slowing-down". The instrument that answers the second question is the two-run rate test in
`collapse.py`, and this module's output is not a substitute for it.

**Early warnings are too late when the parameter moves fast**, which is a direct hit on the case
this library is built for. Slowing down is measurable only while the system still tracks its
equilibrium, so on a run with `Ad` of order 1 the indicator has no lead to give and the rolling
window is measuring the transition rather than anticipating it. `driver_comparison` is here so that
number travels with every reading rather than being looked up afterwards.

Torch-free, pure numpy plus scipy for Kendall's tau and scikit-learn for the mixture fit, all three
of which are already base dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.stats import kendalltau

from reward_lens.core.evidence import register_payload
from reward_lens.core.reading import Refusal, RefusalReason, refuse_incomplete

Indicator = Literal["ac1", "variance"]
SurrogateMethod = Literal["fourier", "ar1"]


@dataclass(frozen=True)
class WarningCriteria:
    """Every number a verdict here is compared against, in one place, with where it came from.

    None of these follows from the statistics themselves, which fix the method and not the
    thresholds it is read against. They are this module's defaults.
    """

    #: Rolling window as a fraction of the series length. **Chosen: 0.5**, which is the value the
    #: ecological early-warning literature settled on and the value `window_sensitivity` sweeps
    #: around. It is a genuine trade: a shorter window resolves the approach to the transition and
    #: estimates each lag-1 coefficient from fewer points.
    window_fraction: float = 0.5

    #: Bandwidth of the Gaussian detrending kernel, as a fraction of the series length.
    #: **Chosen: 0.2.** Detrending matters more than the rolling window does, because an undetrended
    #: series carries its trend straight into the lag-1 coefficient.
    detrend_bandwidth: float = 0.2

    #: Surrogate replicates for the trend null. **Chosen: 500**, which puts the Monte Carlo standard
    #: error on a p-value near 0.05 at about 0.010 and keeps a null under a second on a 400-point
    #: series.
    n_surrogates: int = 500

    #: Bootstrap replicates for the flickering null. **Chosen: 200.** The statistic is a BIC
    #: difference rather than a tail probability, so it needs fewer.
    n_flicker_boot: int = 200

    #: Finite points needed before any of this is attempted. **Chosen: 30.** Below it the rolling
    #: window holds fewer than 15 points, a lag-1 coefficient from 15 points has a standard error
    #: near 0.25, and a Kendall tau over a handful of such coefficients is noise with a sign.
    min_points: int = 30

    #: Points needed inside one rolling window.
    min_window_points: int = 10

    #: Window fractions the sensitivity sweep visits.
    window_sweep: tuple[float, ...] = (0.25, 0.375, 0.5, 0.625, 0.75)

    #: Significance level the verdicts are read at.
    alpha: float = 0.05


# ---------------------------------------------------------------------------
# Detrending and the rolling indicators
# ---------------------------------------------------------------------------


def gaussian_smooth(x: np.ndarray, bandwidth_points: float) -> np.ndarray:
    """A Gaussian kernel smoother of a series against its own index, evaluated at every point.

    The one smoother in this package. `gaussian_detrend` is its residual and `collapse.py` uses it
    as the curve the two arms' bands are built around, so a change to the kernel changes both
    together rather than leaving them to drift apart.
    """
    n = x.size
    if n == 0:
        return x
    h = max(float(bandwidth_points), 1e-9)
    idx = np.arange(n, dtype=np.float64)
    # An (n, n) kernel matrix is fine at the sizes this runs on: 400 points is 1.3 MB.
    d = idx[:, None] - idx[None, :]
    w = np.exp(-0.5 * (d / h) ** 2)
    w /= w.sum(axis=1, keepdims=True)
    return w @ x


def gaussian_detrend(x: np.ndarray, bandwidth_points: float) -> np.ndarray:
    """Residuals of a Gaussian kernel smoother, which is what a rolling indicator is computed on.

    A linear detrend is the cheaper option and it is the wrong one here: the series this is pointed
    at bends through a transition, a straight line leaves that bend in the residual, and the bend
    lands in the lag-1 coefficient as memory the system does not have. The kernel follows the bend.

    The cost of following it is real and is the reason `window_sensitivity` exists: a bandwidth
    narrow enough to track the transition also removes some of the slowing down being looked for,
    so the indicator is attenuated toward finding nothing. That direction is the safe one.
    """
    if x.size == 0:
        return x
    return x - gaussian_smooth(x, bandwidth_points)


def _lag1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation, or NaN where the window has no variance to correlate."""
    if x.size < 3:
        return float("nan")
    r = x - x.mean()
    denominator = float(np.dot(r, r))
    if denominator <= 0:
        return float("nan")
    return float(np.dot(r[:-1], r[1:]) / denominator)


def rolling_indicator(
    resid: np.ndarray, window: int, kind: Indicator = "ac1"
) -> tuple[np.ndarray, np.ndarray]:
    """The indicator over a trailing window, and the index of the window's right edge.

    Trailing rather than centred, and that is the whole point of the exercise: a centred window at
    step `t` has seen steps after `t`, so an early warning computed on one is not an early warning.
    The returned index is the last step each value used, so a lead time measured from it is a lead
    time something could have acted on.
    """
    n = resid.size
    if window < 3 or window > n:
        return np.empty(0), np.empty(0)
    out = np.empty(n - window + 1, dtype=np.float64)
    where = np.empty(n - window + 1, dtype=np.float64)
    for i in range(out.size):
        chunk = resid[i : i + window]
        out[i] = _lag1(chunk) if kind == "ac1" else float(np.var(chunk, ddof=1))
        where[i] = float(i + window - 1)
    return out, where


# ---------------------------------------------------------------------------
# The surrogates
# ---------------------------------------------------------------------------


def fourier_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A phase-randomised surrogate: the periodogram is preserved exactly, the trend is destroyed.

    Take the discrete Fourier transform, replace every phase with a uniform draw while keeping the
    amplitudes and the conjugate symmetry, and transform back. The result has the same power
    spectrum as the input and therefore the same autocorrelation function, and it is a stationary
    Gaussian series by construction. What it does not have is the input's ordering, which is exactly
    what a trend in a rolling indicator is a claim about.

    What this null cannot do: the surrogate is Gaussian, so a strongly non-Gaussian marginal makes
    the null slightly too narrow and the p-value slightly too small. The iterated amplitude-adjusted
    construction fixes that and is not built here; the AR(1) null below is the conservative
    alternative available today.
    """
    n = x.size
    spectrum = np.fft.rfft(x - x.mean())
    phases = rng.uniform(0.0, 2.0 * np.pi, spectrum.size)
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0
    return np.fft.irfft(np.abs(spectrum) * np.exp(1j * phases), n=n) + x.mean()


def ar1_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A parametric first-order autoregressive surrogate at the series' own fitted coefficient.

    Narrower than the Fourier null, because it keeps one number from the spectrum instead of all of
    it. It is here because the two disagree usefully: a series whose memory really is first-order
    gives the same answer under both, and a series with structure at longer lags gives a smaller
    p-value under this one. When they disagree, the Fourier null is the one to report, and
    `trend_significance` runs whichever it is asked for and names it on the reading.
    """
    n = x.size
    phi = _lag1(x)
    if not math.isfinite(phi):
        phi = 0.0
    phi = float(np.clip(phi, -0.999, 0.999))
    resid = x[1:] - phi * x[:-1] if n > 1 else np.zeros(0)
    sd = float(np.std(resid)) if resid.size else float(np.std(x))
    if not math.isfinite(sd) or sd <= 0:
        sd = 1.0
    out = np.empty(n, dtype=np.float64)
    out[0] = rng.normal(0.0, sd / math.sqrt(max(1.0 - phi * phi, 1e-6)))
    for i in range(1, n):
        out[i] = phi * out[i - 1] + rng.normal(0.0, sd)
    return out + x.mean()


_SURROGATES = {"fourier": fourier_surrogate, "ar1": ar1_surrogate}


# ---------------------------------------------------------------------------
# The trend, and its null
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class TrendNull:
    """Kendall's tau of an indicator against time, and what the surrogates said about it.

    `tau` on its own is the number the literature reports and it is not interpretable, because a
    rolling lag-1 coefficient computed on a red-noise series trends upward at a rate that depends on
    the window and the redness. `p` is the fraction of surrogates whose tau reached the observed
    one, and it is the number this reading is read on.
    """

    tau: float
    p: float
    n_surrogates: int
    null_median: float
    null_q95: float
    indicator: Indicator
    method: SurrogateMethod
    window: int
    n_indicator_points: int
    series: str

    @property
    def significant(self) -> bool:
        return self.p <= 0.05

    def render(self) -> str:
        verdict = "beats" if self.significant else "does not beat"
        return (
            f"Kendall tau of the rolling {self.indicator} of {self.series!r} against time is "
            f"{self.tau:+.3f} over {self.n_indicator_points} windows of {self.window} steps, and "
            f"{verdict} its null: p = {self.p:.4g} against {self.n_surrogates} {self.method} "
            f"surrogates whose tau has median {self.null_median:+.3f} and 95th percentile "
            f"{self.null_q95:+.3f}"
        )


def trend_significance(
    series: Sequence[float] | np.ndarray,
    *,
    indicator: Indicator = "ac1",
    method: SurrogateMethod = "fourier",
    criteria: WarningCriteria | None = None,
    name: str = "series",
    instrument: str = "EarlyWarning",
    seed: int = 0,
) -> "TrendNull | Refusal":
    """The rolling indicator's trend, with the surrogate null attached. The only way to get a tau.

    There is deliberately no function here that returns a bare Kendall tau. The tau without the null
    is the statistic this module exists to stop being reported, and making the null the cheap path
    is the only mechanism that reliably works.

    The pipeline: detrend with a Gaussian kernel, roll the indicator over a trailing window, take
    Kendall's tau against the window's right edge. Then run the identical pipeline over
    `n_surrogates` surrogates of the detrended residual, and report where the observed tau falls.
    Identical is load-bearing: the surrogates go through the same detrend and the same window, so
    whatever bias the pipeline has is in the null as well as in the estimate.
    """
    criteria = criteria or WarningCriteria()
    x = np.asarray([float(v) for v in series], dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n < criteria.min_points:
        return refuse_incomplete(
            instrument,
            field=f"at least {criteria.min_points} finite points",
            subject=f"the series {name!r} ({n} recorded)",
            remedy=(
                f"log {name!r} on every step, or widen the window until it holds "
                f"{criteria.min_points} points. A rolling indicator over fewer than that is a "
                f"handful of lag-1 coefficients each fitted on a handful of points, and a "
                f"Kendall tau over those has a sign and no information."
            ),
            n=n,
            floor=criteria.min_points,
        )

    window = max(criteria.min_window_points, int(round(criteria.window_fraction * n)))
    if window > n:
        return refuse_incomplete(
            instrument,
            field=f"a rolling window of {window} points inside a series of {n}",
            subject=f"the series {name!r}",
            remedy=(
                f"lower window_fraction below {float(n) / window:.2f}, or record more steps. The "
                f"window cannot be longer than the series it rolls over."
            ),
            n=n,
            window=window,
        )

    resid = gaussian_detrend(x, criteria.detrend_bandwidth * n)
    values, where = rolling_indicator(resid, window, indicator)
    ok = np.isfinite(values)
    if int(ok.sum()) < 5:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"the rolling {indicator} of {name!r} is defined at {int(ok.sum())} of "
                f"{values.size} windows; a window with no variance in it has no autocorrelation"
            ),
            remedy=(
                "point this at a channel that fluctuates within the window. A series pinned to one "
                "value inside every window has no residual to correlate, and that is a fact about "
                "the channel rather than about the run."
            ),
            statistics={"n_defined": int(ok.sum()), "n_windows": int(values.size)},
        )

    observed = float(kendalltau(where[ok], values[ok]).statistic)
    surrogate = _SURROGATES[method]
    rng = np.random.default_rng(seed)
    null = np.empty(criteria.n_surrogates, dtype=np.float64)
    for i in range(criteria.n_surrogates):
        s = surrogate(resid, rng)
        sv, sw = rolling_indicator(
            gaussian_detrend(s, criteria.detrend_bandwidth * n), window, indicator
        )
        m = np.isfinite(sv)
        null[i] = float(kendalltau(sw[m], sv[m]).statistic) if int(m.sum()) >= 5 else np.nan

    finite = null[np.isfinite(null)]
    if finite.size < criteria.n_surrogates // 2:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.NO_MATCHED_CONTROL,
            detail=(
                f"only {finite.size} of {criteria.n_surrogates} surrogates produced a defined tau, "
                f"so the null this tau of {observed:+.3f} would be read against is not estimated"
            ),
            remedy=(
                "use method='ar1', which generates from a fitted model rather than from the "
                "observed spectrum and cannot degenerate, or lengthen the series. A tau with a "
                "null this thin is a tau with no null."
            ),
            statistics={"tau": observed, "n_null_ok": int(finite.size)},
        )

    # One-sided, upward, because slowing down is a claim about the sign. The plus-one is the
    # standard finite-sample correction: with B surrogates a p-value of exactly zero is not
    # available evidence, it is the resolution of the null.
    p = float((1.0 + float(np.sum(finite >= observed))) / (1.0 + finite.size))
    return TrendNull(
        tau=observed,
        p=p,
        n_surrogates=int(finite.size),
        null_median=float(np.median(finite)),
        null_q95=float(np.quantile(finite, 0.95)),
        indicator=indicator,
        method=method,
        window=int(window),
        n_indicator_points=int(ok.sum()),
        series=name,
    )


# ---------------------------------------------------------------------------
# Flickering, by bimodality rather than by a moment
# ---------------------------------------------------------------------------


def _mixture_bic(values: np.ndarray, k: int, seed: int) -> float:
    """BIC of a `k`-component Gaussian mixture, or positive infinity if it will not fit."""
    from sklearn.mixture import GaussianMixture

    try:
        model = GaussianMixture(
            n_components=k, covariance_type="full", n_init=3, random_state=seed, reg_covar=1e-8
        ).fit(values.reshape(-1, 1))
        return float(model.bic(values.reshape(-1, 1)))
    except (ValueError, FloatingPointError):
        return float("inf")


@register_payload
@dataclass(frozen=True)
class Flickering:
    """Whether the series is rattling between two states, tested as bimodality with a null.

    `delta_bic` is BIC of the one-component fit minus BIC of the two-component fit, so positive
    favours two states. `p` calibrates it against a parametric bootstrap under the fitted single
    component, which is the part that makes the number mean anything: a two-component mixture fits
    almost any real sample better than one component, and the question is whether it fits better
    than it would on data that genuinely has one mode.

    `separation` is the distance between the fitted means in pooled standard deviations. Two
    components sitting on top of each other are a better likelihood and not a second state, so a
    significant `delta_bic` at a separation below about 2 is a fit artifact rather than flickering
    and `render` says so.
    """

    delta_bic: float
    p: float
    separation: float
    weights: tuple[float, float]
    means: tuple[float, float]
    n: int
    n_boot: int
    series: str

    @property
    def flickering(self) -> bool:
        """Bimodal at the 5 percent level and separated enough for the modes to be distinct."""
        return self.p <= 0.05 and self.separation >= 2.0

    def render(self) -> str:
        if self.p > 0.05:
            return (
                f"{self.series!r} is not detectably bimodal: delta BIC {self.delta_bic:+.3g} at "
                f"p = {self.p:.4g} over {self.n_boot} unimodal bootstraps of {self.n} points"
            )
        tail = (
            ""
            if self.separation >= 2.0
            else (
                f", but the two fitted means are only {self.separation:.2f} pooled standard "
                f"deviations apart, which is a better likelihood rather than a second state"
            )
        )
        return (
            f"{self.series!r} is bimodal: delta BIC {self.delta_bic:+.3g} at p = {self.p:.4g}, "
            f"modes at {self.means[0]:.4g} and {self.means[1]:.4g} with weights "
            f"{self.weights[0]:.2f} and {self.weights[1]:.2f}{tail}"
        )


def flickering(
    series: Sequence[float] | np.ndarray,
    *,
    criteria: WarningCriteria | None = None,
    name: str = "series",
    instrument: str = "EarlyWarning",
    seed: int = 0,
) -> "Flickering | Refusal":
    """Bimodality by mixture BIC, calibrated against a unimodal parametric bootstrap.

    There are two tests for this and this is the second of them. Hartigan's dip is the first and it
    is not built: the dip is nonparametric where this is not, which matters, and a from-scratch
    implementation of the greatest-convex-minorant algorithm is a piece of statistics that is easy
    to get subtly wrong and hard to notice. The choice is recorded rather than hidden.

    **What this cannot do, and it is the reason the choice is worth recording.** The null is a
    single Gaussian, so the p-value answers "is this more bimodal than one Gaussian" and not "is
    this more bimodal than one mode". A skewed or heavy-tailed unimodal distribution is fitted
    better by two Gaussians than by one, and this test will call that bimodal. `separation` is the
    guard that catches the commonest form of it, and it is a guard rather than a fix. If the channel
    is skewed by construction, such as a rate bounded below at zero, transform it to symmetry before
    calling this or read the answer as a fit comparison rather than as a claim about modes.
    """
    criteria = criteria or WarningCriteria()
    x = np.asarray([float(v) for v in series], dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n < criteria.min_points:
        return refuse_incomplete(
            instrument,
            field=f"at least {criteria.min_points} finite points",
            subject=f"the series {name!r} ({n} recorded)",
            remedy=(
                f"record at least {criteria.min_points} points of {name!r}. A two-component "
                f"mixture has five parameters and fitting one to fewer points than that recovers "
                f"the sample rather than the distribution."
            ),
            n=n,
            floor=criteria.min_points,
        )
    sd = float(np.std(x))
    if not math.isfinite(sd) or sd <= 0:
        return refuse_incomplete(
            instrument,
            field="any variation to find modes in",
            subject=f"the series {name!r}, every value of which is {float(x[0]):.6g}, and so",
            remedy=(
                "point this at a channel that moves. A constant series has one mode by definition "
                "and no bootstrap changes that."
            ),
            n=n,
            sd=sd,
        )

    from sklearn.mixture import GaussianMixture

    bic1 = _mixture_bic(x, 1, seed)
    bic2 = _mixture_bic(x, 2, seed)
    observed = bic1 - bic2
    two = GaussianMixture(
        n_components=2, covariance_type="full", n_init=3, random_state=seed, reg_covar=1e-8
    ).fit(x.reshape(-1, 1))
    means = np.sort(two.means_.ravel())
    order = np.argsort(two.means_.ravel())
    pooled = float(np.sqrt(np.mean(two.covariances_.ravel())))
    separation = float(abs(means[1] - means[0]) / pooled) if pooled > 0 else float("inf")

    rng = np.random.default_rng(seed + 1)
    mu, sigma = float(np.mean(x)), sd
    null = np.empty(criteria.n_flicker_boot, dtype=np.float64)
    for i in range(criteria.n_flicker_boot):
        draw = rng.normal(mu, sigma, n)
        null[i] = _mixture_bic(draw, 1, seed) - _mixture_bic(draw, 2, seed)
    finite = null[np.isfinite(null)]
    p = float((1.0 + float(np.sum(finite >= observed))) / (1.0 + finite.size))

    weights = two.weights_.ravel()[order]
    return Flickering(
        delta_bic=float(observed),
        p=p,
        separation=separation,
        weights=(float(weights[0]), float(weights[1])),
        means=(float(means[0]), float(means[1])),
        n=n,
        n_boot=int(finite.size),
        series=name,
    )


# ---------------------------------------------------------------------------
# The driver check, the window sweep and the noise control
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class DriverComparison:
    """The relaxation time against the timescale the driver is moving on, which is `Ad` again.

    This is one of the early-warning checks and it is the same quantity `H1` reports,
    reached from the other side: an early autoregressive fit gives the relaxation time, the schedule
    gives the driving rate, and the product is the adiabaticity number. It is here because the
    second caveat on every early-warning reading is that the warning arrives too late when the
    driver moves fast, and this is the number that says whether it did.
    """

    tau_relax: float
    tau_high: float
    driver_timescale: float
    ad: float
    ad_high: float
    licensed: bool

    def render(self) -> str:
        if self.licensed:
            return (
                f"the relaxation time is {self.tau_relax:.4g} steps against a driver timescale of "
                f"{self.driver_timescale:.4g} steps, so Ad is {self.ad:.3g} (upper end "
                f"{self.ad_high:.3g}) and the system tracks its equilibrium: an early warning has "
                f"room to be early"
            )
        return (
            f"the relaxation time is {self.tau_relax:.4g} steps against a driver timescale of "
            f"{self.driver_timescale:.4g} steps, so Ad reaches {self.ad_high:.3g} and the system "
            f"does not track its equilibrium. Slowing down is measurable only while it does, so a "
            f"rolling indicator on this run is measuring the transition rather than anticipating it"
        )


def driver_comparison(
    series: Sequence[float] | np.ndarray,
    drive_rate: float,
    *,
    ad_max: float = 1.0,
    name: str = "series",
    instrument: str = "EarlyWarning",
    seed: int = 0,
) -> "DriverComparison | Refusal":
    """`tau_relax` from the early autoregressive fit against `1 / |d log lambda / dt|`.

    Composes `adiabaticity.relaxation_time` and `adiabaticity.adiabaticity_number` rather than
    fitting a second coefficient. There is one bias-corrected relaxation-time estimator in this
    library and one definition of `Ad`, and a warning module that grew its own copies of both would
    be two more things to keep in agreement.
    """
    from reward_lens.measure.rate.adiabaticity import (
        STEP_AXIS,
        DriveRate,
        adiabaticity_number,
        relaxation_time,
    )

    if not math.isfinite(drive_rate) or drive_rate < 0:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.RECORD_INCOMPLETE,
            detail=f"the driving rate supplied is {drive_rate!r}, which is not a rate",
            remedy=(
                "supply |d log lambda / dt| in units of one over optimizer steps, taken from the "
                "schedule the run actually wrote down. `adiabaticity.drive_rates` computes it from "
                "a Run."
            ),
            statistics={"drive_rate": drive_rate},
        )
    tau = relaxation_time(series, name=name, instrument=instrument, seed=seed)
    if isinstance(tau, Refusal):
        return tau
    rate = DriveRate(from_step=0, to_step=1, rate=float(drive_rate), parameter=name, axis=STEP_AXIS)
    ad = adiabaticity_number(tau, rate)
    if isinstance(ad, Refusal):
        return ad
    ad_high = float(tau.tau_high * drive_rate)
    timescale = float("inf") if drive_rate == 0 else 1.0 / drive_rate
    return DriverComparison(
        tau_relax=float(tau.tau),
        tau_high=float(tau.tau_high),
        driver_timescale=timescale,
        ad=float(ad),
        ad_high=ad_high,
        licensed=bool(ad_high < ad_max),
    )


@register_payload
@dataclass(frozen=True)
class WindowSensitivity:
    """The same tau at several window lengths, because the window is a free parameter.

    `stable` is the field to read: whether every window in the sweep agreed on the sign and on
    significance. A result that appears at one window length and not at its neighbours is a result
    about the window.
    """

    fractions: tuple[float, ...]
    taus: tuple[float, ...]
    p_values: tuple[float, ...]
    n_ok: int
    n_tried: int

    @property
    def stable(self) -> bool:
        if self.n_ok < 2:
            return False
        signs = {t > 0 for t in self.taus}
        calls = {p <= 0.05 for p in self.p_values}
        return len(signs) == 1 and len(calls) == 1

    def render(self) -> str:
        pairs = ", ".join(
            f"{f:.3f}:{t:+.3f}(p={p:.3g})"
            for f, t, p in zip(self.fractions, self.taus, self.p_values)
        )
        verdict = "agree" if self.stable else "do not agree"
        return (
            f"across {self.n_ok} of {self.n_tried} window fractions the readings {verdict} on sign "
            f"and significance: {pairs}"
        )


def window_sensitivity(
    series: Sequence[float] | np.ndarray,
    *,
    indicator: Indicator = "ac1",
    method: SurrogateMethod = "fourier",
    criteria: WarningCriteria | None = None,
    name: str = "series",
    seed: int = 0,
) -> WindowSensitivity:
    """Run `trend_significance` at every window fraction in the sweep and collect what came back.

    Windows that refuse are dropped and counted rather than treated as absent, so a sweep in which
    most window lengths could not be computed is visible as `n_ok` well below `n_tried` instead of
    as a confident agreement between the two that worked.
    """
    criteria = criteria or WarningCriteria()
    fractions: list[float] = []
    taus: list[float] = []
    ps: list[float] = []
    for f in criteria.window_sweep:
        out = trend_significance(
            series,
            indicator=indicator,
            method=method,
            criteria=WarningCriteria(
                window_fraction=f,
                detrend_bandwidth=criteria.detrend_bandwidth,
                n_surrogates=criteria.n_surrogates,
                n_flicker_boot=criteria.n_flicker_boot,
                min_points=criteria.min_points,
                min_window_points=criteria.min_window_points,
                window_sweep=criteria.window_sweep,
                alpha=criteria.alpha,
            ),
            name=name,
            seed=seed,
        )
        if isinstance(out, Refusal):
            continue
        fractions.append(float(f))
        taus.append(out.tau)
        ps.append(out.p)
    return WindowSensitivity(
        fractions=tuple(fractions),
        taus=tuple(taus),
        p_values=tuple(ps),
        n_ok=len(fractions),
        n_tried=len(criteria.window_sweep),
    )


@register_payload
@dataclass(frozen=True)
class EarlyWarning:
    """Every early-warning statement about one series, with all four of the required checks.

    Read `credible` and then read why. A rising autocorrelation with a significant null, a stable
    window sweep, a quiet noise control and a licensed driver comparison is the strongest statement
    this evidence supports, and it is still a statement about slowing down rather than about a
    bifurcation. The instrument that separates those is `collapse.two_run_rate_test`.
    """

    ac1: TrendNull
    variance: "TrendNull | None"
    flicker: "Flickering | None"
    driver: "DriverComparison | None"
    sensitivity: WindowSensitivity
    control: "TrendNull | None"
    control_series: str

    @property
    def control_quiet(self) -> bool:
        """Whether the unrelated channel stayed quiet. No control at all is not quiet."""
        return self.control is not None and self.control.p > 0.05

    @property
    def credible(self) -> bool:
        """All four checks, and the driver only if one was supplied.

        The driver check is the one that is allowed to be absent, because a record with no annealed
        schedule has no driver timescale and that is a fact about the run. Every other check missing
        makes this False: an absent null is not a passed null.
        """
        driver_ok = self.driver is None or self.driver.licensed
        return bool(
            self.ac1.significant and self.sensitivity.stable and self.control_quiet and driver_ok
        )

    def render(self) -> str:
        lines = [self.ac1.render(), self.sensitivity.render()]
        if self.variance is not None:
            lines.append(self.variance.render())
        if self.flicker is not None:
            lines.append(self.flicker.render())
        if self.driver is not None:
            lines.append(self.driver.render())
        lines.append(
            self.control.render()
            if self.control is not None
            else (
                "no noise control was supplied, so nothing here separates a slowing-down signal "
                "from anything that moved every channel at once"
            )
        )
        lines.append(
            "This is evidence of slowing down and not of a bifurcation. Systems with no critical "
            "transition in them show these signals, and a warning arrives too late when the driver "
            "moves faster than the system relaxes; the two-run rate test is what separates the two "
            "cases."
        )
        return "\n".join(lines)


def early_warning(
    series: Sequence[float] | np.ndarray,
    *,
    control: Sequence[float] | np.ndarray | None = None,
    drive_rate: float | None = None,
    criteria: WarningCriteria | None = None,
    method: SurrogateMethod = "fourier",
    name: str = "series",
    control_name: str = "control",
    instrument: str = "EarlyWarning",
    seed: int = 0,
) -> "EarlyWarning | Refusal":
    """Every required check, on one series, in one call.

    The autocorrelation trend is required and everything else is best-effort: a variance trend, a
    flickering test, the driver comparison when a rate is supplied, the window sweep, and the noise
    control when a second channel is supplied. If the autocorrelation trend itself refuses, that
    refusal is returned, because the rest of the reading has nothing to qualify.
    """
    criteria = criteria or WarningCriteria()
    ac1 = trend_significance(
        series,
        indicator="ac1",
        method=method,
        criteria=criteria,
        name=name,
        instrument=instrument,
        seed=seed,
    )
    if isinstance(ac1, Refusal):
        return ac1

    var = trend_significance(
        series,
        indicator="variance",
        method=method,
        criteria=criteria,
        name=name,
        instrument=instrument,
        seed=seed,
    )
    flick = flickering(series, criteria=criteria, name=name, instrument=instrument, seed=seed)
    drive = (
        driver_comparison(series, drive_rate, name=name, instrument=instrument, seed=seed)
        if drive_rate is not None
        else None
    )
    ctrl = (
        trend_significance(
            control,
            indicator="ac1",
            method=method,
            criteria=criteria,
            name=control_name,
            instrument=instrument,
            seed=seed,
        )
        if control is not None
        else None
    )
    return EarlyWarning(
        ac1=ac1,
        variance=None if isinstance(var, Refusal) else var,
        flicker=None if isinstance(flick, Refusal) else flick,
        driver=None if drive is None or isinstance(drive, Refusal) else drive,
        sensitivity=window_sensitivity(
            series, method=method, criteria=criteria, name=name, seed=seed
        ),
        control=None if ctrl is None or isinstance(ctrl, Refusal) else ctrl,
        control_series=control_name,
    )


__all__ = [
    "DriverComparison",
    "EarlyWarning",
    "Flickering",
    "Indicator",
    "SurrogateMethod",
    "TrendNull",
    "WarningCriteria",
    "WindowSensitivity",
    "ar1_surrogate",
    "driver_comparison",
    "early_warning",
    "flickering",
    "fourier_surrogate",
    "gaussian_detrend",
    "gaussian_smooth",
    "rolling_indicator",
    "trend_significance",
    "window_sensitivity",
]

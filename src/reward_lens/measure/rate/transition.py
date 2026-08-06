"""H4, the transition width: the unit every lead time in this library is reported in.

Lead time has four incommensurable conventions in the literature it is reported in. One paper
counts training steps, one reports a fraction of episode elapsed, one reports distance to onset,
and one reports precision and recall on a consecutive-decline rule. No two of those can be placed
on one axis, and the raw step count is the worst of them, because a lab reporting that most of the
change in reward-hacking rate happens inside the first 64 steps has also told you that a 40-step
lead inside a 64-step window is most of the run rather than a comfortable margin.

This library fixes it: **lead time is a fraction of the fitted transition width**, where the width
comes from a changepoint or sigmoid fit to the outcome series, and the fit is reported with it. The
second half of that sentence is not decoration. A width with no fit quality beside it is a unit
with no scale, because the number 58 means one thing when the logistic explains 94 percent of the
series and nothing at all when a straight line fits just as well. So `fit_transition` returns the
width and the fit together in one object, or it returns a `Refusal`, and there is no path through
this module that hands a caller a width without the evidence for it.

**What this module will not do.** It will not report a width for a series a transition model does
not fit. On a 200-step optimisation trace with no behavioural transition in it, the logistic and a
straight line explain the series equally well, and the fitted "width" is then the width of the
noise. That case returns `BELOW_LOD` with the model comparison in the refusal, and the remedy says
what subject the question needs. It also will not report a width the observation window does not
contain: a fitted width at or beyond the span of the recorded steps is an extrapolation, and it
comes back as a lower bound attached to `ABOVE_LOD_BELOW_LOQ` rather than as a number.

The interval on the width is a moving-block bootstrap over the residuals rather than the covariance
matrix `curve_fit` returns. The covariance matrix assumes independent residuals; a training series
is autocorrelated, so both it and an ordinary residual bootstrap understate the interval, and the
width interval is the number a reader uses to decide whether two runs' widths differ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import SubjectRef, make_evidence, register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.provenance import Provenance
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
    refuse_incomplete,
)
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    GaugeStatus,
    Phase,
    Substrate,
)
from reward_lens.measure.base import BaseObservable, Context, run
from reward_lens.measure.rate.regime import MEASURED_BY, Window
from reward_lens.record.schema import Run, Step

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence

#: `2 ln 9` is the number of logistic scale lengths between the 10 percent and 90 percent points of
#: the rise, which is what makes the 10-to-90 convention a fixed multiple of the fitted scale.
TEN_TO_NINETY = 2.0 * math.log(9.0)


# ---------------------------------------------------------------------------
# The numbers this module decides against
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionCriteria:
    """Every number a verdict here is compared against, in one place, with where it came from.

    None of these follows from the estimator itself ("a changepoint or sigmoid fit"), which fixes
    the method and not the thresholds it is read against. They are this module's defaults, chosen
    for the reasons at each field. A default is not a decision; it behaves like one until somebody
    says otherwise, which is why they are gathered here rather than buried in the fitting code.
    """

    #: AICc of the best no-transition model minus AICc of the logistic, below which the transition
    #: is not established. Burnham and Anderson's conventional reading of an information-criterion
    #: difference puts "essentially no support" for the weaker model at a difference above 10, and
    #: the weaker model here is the one this instrument needs to rule out. **Chosen: 10.0.** A
    #: smaller cut would let a series whose trend is a straight line report a width.
    delta_aicc_min: float = 10.0

    #: Recorded steps needed before a four-parameter fit is attempted. Three observations per
    #: parameter is the usual rough floor and twelve is that. **Chosen: 12.** Below it the fit is
    #: not identified rather than imprecise, and AICc's own small-sample correction divides by
    #: ``n - k - 1``, which is 7 here and is already small.
    min_points: int = 12

    #: The fitted width, as a multiple of the observed span, above which the transition is not
    #: contained in the window. **Chosen: 1.0**, which is the definition rather than a taste: a
    #: rise wider than everything you recorded was not observed, it was extrapolated, and the only
    #: honest statement left is that the width is at least the span.
    max_width_in_spans: float = 1.0

    #: Bootstrap replicates for the width interval. **Chosen: 200**, which puts the Monte Carlo
    #: error on a 95 percent percentile interval well under the sampling error it is estimating and
    #: keeps a fit on a 200-step record under a second.
    n_boot: int = 200

    #: Coverage of the width interval.
    ci_level: float = 0.95


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def _logistic(
    t: np.ndarray, baseline: float, amplitude: float, midpoint: float, width: float
) -> np.ndarray:
    """``baseline + amplitude * expit((t - midpoint) * 2 ln 9 / width)``.

    Parameterised by the 10-to-90 width directly rather than by a rate, so the quantity being
    reported is a fitted parameter with its own interval instead of a transform of one. `width` is
    kept positive by the optimiser's bounds; a falling transition is a negative `amplitude`, which
    keeps the sign of the direction in one place instead of two.
    """
    z = (t - midpoint) * (TEN_TO_NINETY / width)
    # expit written out, clipped, because a 200-step axis divided by a width the optimiser is
    # driving toward zero overflows the exponential before the bounds catch it.
    return baseline + amplitude / (1.0 + np.exp(-np.clip(z, -700.0, 700.0)))


def _aicc(sse: float, n: int, k: int) -> float:
    """Akaike's criterion with the small-sample correction, on a Gaussian likelihood.

    Returns positive infinity when ``n - k - 1`` is not positive, which is the honest answer: the
    correction is undefined there and a model with more parameters than the data can support has
    not been fitted, it has been interpolated.
    """
    if n - k - 1 <= 0:
        return float("inf")
    if sse <= 0:
        return float("-inf")
    return n * math.log(sse / n) + 2.0 * k + (2.0 * k * (k + 1.0)) / (n - k - 1.0)


def _sse_constant(y: np.ndarray) -> float:
    return float(np.sum((y - y.mean()) ** 2))


def _sse_line(t: np.ndarray, y: np.ndarray) -> float:
    tc = t - t.mean()
    denominator = float(np.dot(tc, tc))
    if denominator <= 0:
        return _sse_constant(y)
    slope = float(np.dot(tc, y - y.mean())) / denominator
    fitted = slope * tc + y.mean()
    return float(np.sum((y - fitted) ** 2))


def _lag1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of a residual vector, or NaN when it is not defined."""
    r = x - x.mean()
    denominator = float(np.dot(r, r))
    if denominator <= 0 or r.size < 3:
        return float("nan")
    return float(np.dot(r[:-1], r[1:]) / denominator)


# ---------------------------------------------------------------------------
# The reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class FitQuality:
    """Whether the fitted width means anything, in the numbers a reader can check it with.

    `delta_aicc_line` is the load-bearing field and it is the one nobody reports. A logistic has
    four parameters and will always fit a monotone series better than a straight line has any right
    to; the question is whether it fits enough better to have found a transition, and an
    information criterion is what answers that at these sample sizes.

    `residual_ac1` is here because r-squared cannot see the failure that matters. A fit that misses
    the shape of the transition leaves residuals with structure in them, and a lag-1
    autocorrelation near one on the residuals says the model is wrong in a way a high r-squared
    will happily hide. Training series are autocorrelated to begin with, so this is read as a
    comparison against the series' own autocorrelation rather than against zero.
    """

    r2: float
    rmse: float
    #: AICc of the better of the two no-transition models minus AICc of the logistic. Positive
    #: favours the transition.
    delta_aicc_line: float
    delta_aicc_constant: float
    #: Lag-1 autocorrelation of the residuals, against the same statistic on the raw series.
    residual_ac1: float
    series_ac1: float
    #: Whether the optimiser was satisfied at the parameters, as against stopping at its evaluation
    #: cap. A fit that stopped at the cap and still beats a straight line by an information
    #: criterion is a usable fit at parameters nobody proved optimal, and saying which it was is
    #: cheaper than arguing about it later.
    converged: bool
    width_ci: tuple[float, float]
    n: int
    n_boot_ok: int
    identified: bool
    note: str = ""

    def render(self) -> str:
        tail = "" if self.converged else "; the optimiser stopped at its evaluation cap"
        if self.note:
            tail += f"; {self.note}"
        return (
            f"r2 = {self.r2:.3g}, rmse = {self.rmse:.4g}, AICc favours the transition over a "
            f"straight line by {self.delta_aicc_line:.4g} on {self.n} points; residual AC(1) "
            f"{self.residual_ac1:.3g} against {self.series_ac1:.3g} on the raw series{tail}"
        )


@register_payload
@dataclass(frozen=True)
class TransitionFit:
    """A fitted behavioural transition: where it happened, how wide it was, and how well it fits.

    `width` is the 10-to-90 rise of the fitted logistic, on the step axis it was given. It is the
    denominator of every lead time in this library, so it never travels without `quality`.

    `midpoint` is the 50 percent point and is what `lead_time` measures against by default.
    `onset_10` and `onset_90` are the ends of the rise, carried so that a caller who wants the lead
    measured from the start of the transition rather than from its centre does not have to
    reconstruct them and get the factor wrong.
    """

    width: float
    midpoint: float
    amplitude: float
    baseline: float
    #: +1 for a rising transition, -1 for a falling one.
    direction: int
    onset_10: float
    onset_90: float
    #: The half-open span of the step axis the fit was taken on, and its median spacing.
    span: tuple[float, float]
    cadence: float
    quality: FitQuality
    series: str
    method: str

    @property
    def usable(self) -> bool:
        """Whether a lead time may be divided by this width.

        Two things, because a caller checking only the first is the failure mode: the width is
        finite and positive, and the transition beat both no-transition models. A width with no
        interval on it is still usable, and `quality.note` says it has none: the interval is the
        precision of the denominator rather than its existence, and it is missing only when the
        caller turned the bootstrap off or the refits would not converge. Every rendering of the
        quality carries that note, so it cannot be lost between here and a report.
        """
        return math.isfinite(self.width) and self.width > 0 and self.quality.identified

    @property
    def resolution_in_widths(self) -> float:
        """The finest lead this series can resolve, as a fraction of a width.

        A lead shorter than one sampling interval is not measured, it is rounded, and reporting it
        beside a lead is what stops a coarse log from producing a confident fraction. On a
        checkpoint ladder logged every 148 steps, resolving 0.24 of a width needs a transition
        wider than 617 steps, and saying so is more useful than the fraction itself.
        """
        if not math.isfinite(self.width) or self.width <= 0:
            return float("nan")
        return float(self.cadence / self.width)

    def render(self) -> str:
        lo, hi = self.quality.width_ci
        return (
            f"the transition in {self.series} has a fitted width of {self.width:.4g} steps "
            f"[{lo:.4g}, {hi:.4g}], centred at step {self.midpoint:.4g} "
            f"({'rising' if self.direction > 0 else 'falling'} by {abs(self.amplitude):.4g}). "
            f"{self.quality.render()}."
        )


@register_payload
@dataclass(frozen=True)
class LeadTime:
    """One alarm's lead, in this library's unit and in the unit the literature uses.

    `widths` is the number this library scores in. `steps` is carried beside it for continuity with
    the four conventions in circulation and **is not comparable across runs**, because two runs'
    transitions are not the same width; `render` says so every time it prints one, which is the
    only reliable way to stop the step count being quoted on its own.
    """

    widths: float
    #: The same lead measured from the start of the rise rather than from its centre. Reported
    #: because the two differ by exactly half a width and the difference is the commonest way to
    #: read a lead-time claim wrong.
    widths_from_onset: float
    steps: float
    alarm_step: float
    resolution_in_widths: float
    fit: TransitionFit

    @property
    def resolved(self) -> bool:
        """Whether the lead is larger than the sampling interval that measured it."""
        return abs(self.widths) >= self.resolution_in_widths

    def render(self) -> str:
        tail = (
            ""
            if self.resolved
            else (
                f" This lead is below the {self.resolution_in_widths:.3g}-width sampling "
                f"resolution of the series, so it is rounded rather than measured."
            )
        )
        return (
            f"a {self.steps:.4g}-step lead is {self.widths:.3f} of a fitted "
            f"{self.fit.width:.4g}-step window ({self.widths_from_onset:.3f} measured from the "
            f"start of the rise). The step count is not comparable across runs. "
            f"{self.fit.quality.render()}.{tail}"
        )


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def _starts(t: np.ndarray, y: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Starting points for the optimiser, taken from the data rather than from a constant.

    Four of them, because one is not enough and a hundred is a search. The midpoint starts at the
    step of steepest observed change and at the two tertile boundaries, which covers the case where
    the steepest single difference is noise; the width starts at a quarter and at half the span.
    """
    span = float(t[-1] - t[0]) or 1.0
    lo, hi = float(np.min(y)), float(np.max(y))
    amplitude = (hi - lo) or 1.0
    rising = float(np.mean(y[y.size // 2 :])) >= float(np.mean(y[: y.size // 2]))
    baseline = lo if rising else hi
    signed = amplitude if rising else -amplitude
    steepest = float(t[int(np.argmax(np.abs(np.gradient(y, t))))])
    thirds = [float(t[0] + span / 3.0), float(t[0] + 2.0 * span / 3.0)]
    out = [(baseline, signed, m, span / 4.0) for m in (steepest, *thirds)]
    out.append((baseline, signed, steepest, span / 2.0))
    return out


#: Function evaluations the optimiser gets per starting point. A four-parameter fit that has not
#: converged in two thousand evaluations is not converging; the case that eats them is a series with
#: no transition, where the optimiser walks the width down toward the resolution floor forever, and
#: that case is about to be refused anyway. Measured on the 200-step GRPO record: with no cap and no
#: floor the four starting points took 1.3 seconds per series, which times 200 bootstrap refits is
#: four minutes to answer a question whose answer is "there is no transition here".
_MAX_NFEV = 2000


def _fit_bounds(t: np.ndarray, cadence: float) -> tuple[list[float], list[float]]:
    """Parameter bounds, with the width floored at the sampling interval.

    The floor is a statement rather than a speed fix. A rise faster than one sample was not
    observed to be that fast, it was observed to happen between two samples, so a width below the
    cadence is a property of the optimiser's search and not of the run. Flooring it there also
    stops the pathological excursion where a series with no transition is fitted as an instant jump
    at one step, which is the global optimum of the least-squares problem on white noise and is not
    a transition.
    """
    span = float(t[-1] - t[0]) or 1.0
    floor = cadence if math.isfinite(cadence) and cadence > 0 else span / max(t.size - 1, 1)
    return (
        [-np.inf, -np.inf, float(t[0]) - span, float(floor)],
        [np.inf, np.inf, float(t[-1]) + span, 10.0 * span],
    )


def _fit_once(
    t: np.ndarray, y: np.ndarray, cadence: float
) -> tuple[np.ndarray, float, bool] | None:
    """The best of the starting points: the parameters, their SSE, and whether it converged.

    `least_squares` rather than `curve_fit`, because `curve_fit` raises when the optimiser stops at
    the evaluation cap and throws away the parameters it had reached. Those parameters are the
    thing this module needs: on a series with no transition the optimiser does not converge, and
    the informative answer is the model comparison against a straight line rather than the sentence
    "it did not converge". So the parameters come back with a flag saying whether the optimiser was
    satisfied with them, and the acceptance test stays the information criterion.
    """
    bounds = _fit_bounds(t, cadence)

    def residual(p: np.ndarray) -> np.ndarray:
        return np.asarray(_logistic(t, *p) - y, dtype=float)

    best: tuple[np.ndarray, float, bool] | None = None
    for p0 in _starts(t, y):
        start = np.array(
            [p0[0], p0[1], p0[2], min(max(p0[3], bounds[0][3]), bounds[1][3])], dtype=float
        )
        try:
            res = least_squares(residual, start, bounds=bounds, max_nfev=_MAX_NFEV)
        except ValueError:
            continue
        sse = float(2.0 * res.cost)
        if best is None or sse < best[1]:
            best = (np.asarray(res.x, dtype=float), sse, bool(res.status > 0))
    return best


def _block_bootstrap_widths(
    t: np.ndarray,
    y: np.ndarray,
    popt: np.ndarray,
    cadence: float,
    criteria: TransitionCriteria,
    seed: int,
) -> np.ndarray:
    """Widths from refits on moving-block resamples of the residuals.

    Blocks rather than single residuals, with length ``n ** (1/3)`` rounded up, which is the
    standard choice for a stationary block bootstrap. An ordinary residual bootstrap resamples
    residuals independently and a training series' residuals are not independent, so it would
    return an interval narrower than the truth on exactly the data this instrument is for.
    """
    n = y.size
    resid = y - _logistic(t, *popt)
    block = max(2, int(math.ceil(n ** (1.0 / 3.0))))
    n_blocks = int(math.ceil(n / block))
    rng = np.random.default_rng(seed)
    fitted = _logistic(t, *popt)
    widths: list[float] = []
    for _ in range(criteria.n_boot):
        starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
        drawn = np.concatenate([resid[s : s + block] for s in starts])[:n]
        out = _fit_once(t, fitted + drawn, cadence)
        if out is not None and math.isfinite(out[0][3]):
            widths.append(abs(float(out[0][3])))
    return np.asarray(widths, dtype=float)


def fit_transition(
    outcome: Sequence[float] | np.ndarray,
    steps: Sequence[float] | np.ndarray | None = None,
    *,
    series: str = "outcome",
    instrument: str = "TransitionWidth",
    criteria: TransitionCriteria | None = None,
    seed: int = 0,
) -> "TransitionFit | Refusal":
    """Fit a transition to an outcome series and return its width with the fit, or refuse.

    This is the one entry point. Everything in this library that scores a lead time calls it once
    per run and divides by `TransitionFit.width`; nothing else in the library fits a second one.

    ``outcome`` is the series whose transition is being measured: a labelled hacking rate, a gold
    reward, a held-out probe, whatever the claim is about. ``steps`` is the axis it was sampled on
    and defaults to ``0, 1, 2, ...``; pass the real step indices whenever the series is logged at a
    cadence, because the width comes back in the units of this axis and the sampling resolution is
    computed from its spacing.

    Four ways this returns a `Refusal` instead of a width, and each is a different fact:

    `RECORD_INCOMPLETE` when the series is shorter than the floor, or is constant. Neither is
    recoverable from this record and the remedy is upstream.

    `BELOW_LOD` when the fit converges and an information criterion prefers a straight line or a
    constant. The transition is not distinguishable from the trend and the noise, and a width read
    off that fit is the width of the noise. This is the answer on a real optimisation trace that
    contains no behavioural transition, which is most of them.

    `BELOW_LOD` again when no starting point converges at all, with the linear residual reported
    beside it so a reader can see whether that is because there is nothing to fit.

    `ABOVE_LOD_BELOW_LOQ`, carrying a bound, when the transition is established and its fitted
    width is at least the observed span, or its midpoint falls outside the recorded steps. The
    transition is detected and not quantifiable from this window, and the bound is that the width
    is at least the span.
    """
    criteria = criteria or TransitionCriteria()
    y = np.asarray([float(v) for v in outcome], dtype=np.float64).ravel()
    t = (
        np.arange(y.size, dtype=np.float64)
        if steps is None
        else np.asarray([float(v) for v in steps], dtype=np.float64).ravel()
    )
    if t.size != y.size:
        raise ValueError(
            f"the step axis has {t.size} entries and the series has {y.size}. These have to be "
            f"the same length; a series logged at a cadence needs its own step indices, not a "
            f"range."
        )
    finite = np.isfinite(y) & np.isfinite(t)
    y, t = y[finite], t[finite]
    order = np.argsort(t)
    y, t = y[order], t[order]
    n = int(y.size)

    if n < criteria.min_points:
        return refuse_incomplete(
            instrument,
            field=f"at least {criteria.min_points} finite points",
            subject=f"the series {series!r} ({n} recorded)",
            remedy=(
                f"log {series!r} on every step, or on a cadence fine enough to leave "
                f"{criteria.min_points} points inside the window you are fitting. A "
                f"four-parameter transition fit on fewer points is not imprecise, it is "
                f"unidentified, and the width it returns is a property of the starting guess."
            ),
            n=n,
            floor=criteria.min_points,
        )

    span = float(t[-1] - t[0])
    cadence = float(np.median(np.diff(t))) if n > 1 else float("nan")
    spread = float(np.max(y) - np.min(y))
    if spread <= 0 or span <= 0:
        return refuse_incomplete(
            instrument,
            field="variation to fit a transition to",
            subject=(
                f"the series {series!r} over steps {t[0]:.0f} to {t[-1]:.0f}, where every value "
                f"is {float(y[0]):.6g}, and so"
                if spread <= 0
                else f"the step axis of {series!r}, where every index is {t[0]:.0f}, and so"
            ),
            remedy=(
                "point this at a series that moves. A channel pinned to one value has no "
                "transition to be wide, and that is a fact about the channel rather than about "
                "the run: a completion length pinned at the generation cap, for instance, is "
                "reporting the cap."
            ),
            n=n,
            spread=spread,
        )

    best = _fit_once(t, y, cadence)
    sse_line = _sse_line(t, y)
    sse_constant = _sse_constant(y)
    if best is None:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.BELOW_LOD,
            detail=(
                f"the transition model could not be evaluated on {series!r} from any of four "
                f"starting points over {n} steps. A straight line leaves a residual sum of squares "
                f"of {sse_line:.6g} against {sse_constant:.6g} for a constant, so the series is "
                f"{'close to flat' if sse_line >= 0.9 * sse_constant else 'trending'} and there "
                f"is no rise for a logistic to place a midpoint inside"
            ),
            remedy=(
                "report the lead in steps and label it as not comparable across runs, or measure "
                "this against a series that contains a transition. A width fitted to a series "
                "with no transition in it is the width of the noise, and dividing a lead by it "
                "would make the lead look precise."
            ),
            statistics={"n": n, "sse_line": sse_line, "sse_constant": sse_constant},
        )

    popt, sse, converged = best
    baseline, amplitude, midpoint, width = (float(v) for v in popt)
    width = abs(width)
    resid = y - _logistic(t, *popt)
    rmse = float(math.sqrt(sse / n))
    sst = _sse_constant(y)
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    delta_line = _aicc(sse_line, n, 2) - _aicc(sse, n, 4)
    delta_constant = _aicc(sse_constant, n, 1) - _aicc(sse, n, 4)
    identified = min(delta_line, delta_constant) >= criteria.delta_aicc_min

    if not identified:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.BELOW_LOD,
            detail=(
                f"a transition fitted to {series!r} over {n} steps is not distinguishable from a "
                f"trend: AICc favours the logistic over a straight line by only "
                f"{delta_line:.4g} and over a constant by {delta_constant:.4g}, against a "
                f"threshold of {criteria.delta_aicc_min:.4g}. The fit's own width would be "
                f"{width:.4g} steps at r2 = {r2:.3g}, and on a series with no transition in it "
                f"that number is the width of the noise"
            ),
            remedy=(
                "report the lead in steps for this run and label it as not comparable across "
                "runs, because there is no width here to divide by. If the claim needs a lead in "
                "width units, it needs a run that contains a behavioural transition: this "
                "estimator can measure the width of one and cannot manufacture one."
            ),
            statistics={
                "n": n,
                "delta_aicc_line": delta_line,
                "delta_aicc_constant": delta_constant,
                "threshold": criteria.delta_aicc_min,
                "width_if_forced": width,
                "r2": r2,
            },
        )

    boot = _block_bootstrap_widths(t, y, popt, cadence, criteria, seed)
    alpha = (1.0 - criteria.ci_level) / 2.0
    ci = (
        (float(np.quantile(boot, alpha)), float(np.quantile(boot, 1.0 - alpha)))
        if boot.size >= 20
        else (float("nan"), float("nan"))
    )
    quality = FitQuality(
        r2=r2,
        rmse=rmse,
        delta_aicc_line=float(delta_line),
        delta_aicc_constant=float(delta_constant),
        residual_ac1=_lag1(resid),
        series_ac1=_lag1(y),
        converged=converged,
        width_ci=ci,
        n=n,
        n_boot_ok=int(boot.size),
        identified=True,
        note=(
            ""
            if boot.size >= 20
            else (
                f"only {boot.size} of {criteria.n_boot} bootstrap refits converged, so the width "
                f"carries no interval. Read the width as a point with no precision attached."
            )
        ),
    )
    direction = 1 if amplitude >= 0 else -1
    fit = TransitionFit(
        width=width,
        midpoint=midpoint,
        amplitude=amplitude,
        baseline=baseline,
        direction=direction,
        onset_10=midpoint - width / 2.0,
        onset_90=midpoint + width / 2.0,
        span=(float(t[0]), float(t[-1])),
        cadence=cadence,
        quality=quality,
        series=series,
        method="logistic 10-to-90, AICc against a line and a constant",
    )

    contained = width < criteria.max_width_in_spans * span and t[0] <= midpoint <= t[-1]
    if not contained:
        why = (
            f"its fitted width of {width:.4g} steps is at least the {span:.4g}-step span"
            if width >= criteria.max_width_in_spans * span
            else (
                f"its midpoint falls at step {midpoint:.4g}, outside the recorded range "
                f"{t[0]:.0f} to {t[-1]:.0f}"
            )
        )
        return bounded_refusal(
            instrument,
            RefusalReason.ABOVE_LOD_BELOW_LOQ,
            detail=(
                f"a transition in {series!r} is established (AICc favours it over a straight line "
                f"by {delta_line:.4g}) and is not quantifiable from this window, because {why}. A "
                f"rise the window does not contain was extrapolated rather than observed"
            ),
            remedy=(
                f"widen the window past step {t[-1]:.0f}, or read the bound: the width is at "
                f"least {span:.4g} steps, so any lead measured inside this window is at most "
                f"{1.0:.2g} of a width and is a lower bound on the fraction rather than a "
                f"measurement of it."
            ),
            bound=make_evidence(
                observable=instrument,
                observable_version="1.0",
                subject=SubjectRef(readout=series),
                value=fit,
                provenance=Provenance(),
                quantity="run.transition_width",
            ),
            n=n,
            width=width,
            span=span,
            midpoint=midpoint,
        )
    return fit


def lead_time(alarm_step: float, fit: TransitionFit) -> LeadTime:
    """A lead in this library's unit: the fraction of a fitted transition width.

    Positive means the alarm fired before the transition. The step count travels with it and is
    labelled at every rendering as not comparable across runs, which is the whole reason this
    function exists rather than a division at each call site.

    This takes a `TransitionFit` rather than a width, deliberately. A bare width is how a lead
    ends up divided by a number nobody checked, and `fit_transition` already refuses rather than
    returning an unusable fit, so a caller holding one of these has something it may divide by.
    """
    if not fit.usable:
        raise ValueError(
            f"this fit is not usable as a denominator: width {fit.width!r}, identified "
            f"{fit.quality.identified}. `fit_transition` returns a Refusal rather than an "
            f"unusable fit, so reaching here means one was constructed by hand. Read the refusal's "
            f"remedy instead of dividing by this."
        )
    steps = float(fit.midpoint - alarm_step)
    return LeadTime(
        widths=steps / fit.width,
        widths_from_onset=float(fit.onset_10 - alarm_step) / fit.width,
        steps=steps,
        alarm_step=float(alarm_step),
        resolution_in_widths=fit.resolution_in_widths,
        fit=fit,
    )


def compare_lead_times(a: Any, b: Any) -> "float | Refusal":
    """Two leads, differenced, or the refusal that says they are not in one unit.

    This is the `units` group's assertion for H4 and it is also the instrument's entire argument
    stated as a check. A lead in steps and a lead in widths are not the same quantity in different
    clothes: converting between them needs the transition width of each run, which the step count
    does not carry, so the comparison refuses rather than converting.
    """
    for name, value in (("first", a), ("second", b)):
        if not isinstance(value, LeadTime):
            return Refusal(
                instrument="TransitionWidth",
                reason=RefusalReason.UNIT_MISMATCH,
                detail=(
                    f"the {name} lead is a bare {type(value).__name__} and carries no transition "
                    f"width, so it is a count of training steps rather than a fraction of a "
                    f"window. Two runs' transitions are not the same width, so their step counts "
                    f"are not on one axis"
                ),
                remedy=(
                    "fit the transition on each run with `fit_transition` and take each lead "
                    "through `lead_time`, then compare the `widths` field. If one run has no "
                    "fitted transition, there is no width to express its lead in and the two "
                    "runs cannot be compared on this axis at all."
                ),
                statistics={"side": name, "type": type(value).__name__},
            )
    return float(a.widths - b.widths)


# ---------------------------------------------------------------------------
# Reading a series off a record
# ---------------------------------------------------------------------------

#: Per-step channels a `Run` can supply, and where each lives. The names are the record's own, so a
#: caller naming one gets what the record wrote rather than a re-derivation. Anything else is
#: looked up in `OptimizerTelemetry.extra`, which is where a converter puts its framework's own log
#: keys, and in the step's probes by name.
SERIES: Mapping[str, str] = {
    "group_mean": "the mean of every group's mean score at the step",
    "entropy": "OptimizerTelemetry.entropy",
    "grad_norm": "OptimizerTelemetry.grad_norm_clipped",
    "kl_to_previous": "OptimizerTelemetry.kl_to_previous",
    "kl_to_ref": "OptimizerTelemetry.kl_to_ref",
}


def window_steps(run: Run, window: Window | None) -> list[Step]:
    """The steps of a window, in index order, with the whole run as the default.

    The default is resolved from `run.steps.indices` rather than from a wide sentinel span, and
    that is not a style point. `RecordReader.partitions_for` builds `range(first, last + 1, chunk)`
    and filters it, so a caller asking for `slice(-2**62, 2**62)` to mean "everything" does not get
    a wide read, it gets a loop over 2**63 partition starts that never returns. Ask for the range
    the record says it has.
    """
    indices = run.steps.indices
    if not indices:
        return []
    lo, hi = window if window is not None else (min(indices), max(indices) + 1)
    return sorted(run.steps.slice(lo, hi), key=lambda s: s.index)


def series_from_run(
    run: Run, name: str, *, window: Window | None = None, channel: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """One named per-step series off a record, with its step indices.

    Four places are searched, in order, and the order is the order of authority: the named channels
    of `SERIES`, then a probe of that name (a `gold` or `held_out` probe is the closest thing a
    record carries to a labelled outcome), then `OptimizerTelemetry.extra`, which is where a
    converter puts the framework's own log keys. Steps where the value is absent are dropped rather
    than filled, so the returned axis is the steps that actually carry it.

    ``channel`` restricts the probe search to one of `held_out`, `gold` or `check_standard`.
    Averaging a gold probe with a held-out eval is how tooling drift gets reported as model
    improvement, so the two are never pooled here.
    """
    return _series_from_steps(window_steps(run, window), name, channel)


def _series_from_steps(
    steps: Sequence[Step], name: str, channel: str | None
) -> tuple[np.ndarray, np.ndarray]:
    """The body of `series_from_run`, on steps already decoded.

    Split out so `available_series` decodes the record once rather than once per candidate channel.
    Decoding a 200-step record is a tenth of a second and doing it six times to answer one question
    is the kind of cost that stops a helper from being used.
    """
    xs: list[float] = []
    ts: list[float] = []
    for s in steps:
        value: float | None = None
        if name == "group_mean":
            means = [g.group_stats.mean for g in s.groups if g.group_stats.mean is not None]
            value = float(np.mean(means)) if means else None
        elif name in ("entropy", "kl_to_previous", "kl_to_ref"):
            value = getattr(s.optimizer, name)
        elif name == "grad_norm":
            value = s.optimizer.grad_norm_clipped
            if value is None:
                value = s.optimizer.grad_norm_unclipped
        else:
            probed = [
                float(p.value)
                for p in s.probes
                if p.name == name
                and p.value is not None
                and (channel is None or p.channel == channel)
            ]
            if probed:
                value = float(np.mean(probed))
            elif name in s.optimizer.extra:
                raw = s.optimizer.extra[name]
                value = float(raw) if isinstance(raw, (int, float)) else None
        if value is not None and math.isfinite(float(value)):
            xs.append(float(value))
            ts.append(float(s.index))
    return np.asarray(xs, dtype=float), np.asarray(ts, dtype=float)


def available_series(run: Run, *, window: Window | None = None) -> dict[str, int]:
    """Every per-step channel this record carries, with how many steps carry each.

    The first thing to run before asking for a width, because the commonest failure is naming a
    channel the converter did not write and reading the refusal as a statement about the run.
    """
    steps = window_steps(run, window)
    counts: dict[str, int] = {}
    for name in SERIES:
        values, _ = _series_from_steps(steps, name, None)
        if values.size:
            counts[name] = int(values.size)
    for s in steps:
        for p in s.probes:
            if p.value is not None:
                counts[p.name] = counts.get(p.name, 0) + 1
        for key, raw in s.optimizer.extra.items():
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

#: What has to be true for a fitted width to be a property of the policy rather than of the setup.
#: Both conditions downgrade rather than refuse, which is the right behaviour for this shape: the
#: width is still a real fit of the series that was recorded, and what it loses outside the
#: envelope is the right to be called a property of the run. Requiring a refusal here would
#: withhold the fit on exactly the records where a reader most needs to see it and then decide.
TRANSITION_ENVELOPE = EnvelopeSpec(
    requires=frozenset({RegimeCondition.STATIONARY_GRADER, RegimeCondition.EXOGENOUS_CURRICULUM}),
    measured_by=MEASURED_BY,
    on_violation="downgrade",
)

_TRANSITION_ACCESS: AccessMatrix = {Component.RECORD: Access.RECORD}

TRANSITION_BASELINES = (
    "baseline.first_crossing",
    "baseline.whole_run_width",
)


class TransitionWidth(BaseObservable):
    """H4. The fitted width of a behavioural transition, and the unit lead time is reported in.

    Reads a record and nothing else. The width is fitted to one named per-step series, which the
    caller chooses, because the transition whose width matters is a property of the outcome being
    claimed about rather than of the run: a width fitted to the training reward and a width fitted
    to a labelled hacking rate are two different quantities and this instrument will not pretend
    otherwise. `available_series` lists what a given record can supply.

    What it cannot do, stated here rather than on a caveats page. It cannot tell a transition from
    a trend on a series short enough that four parameters interpolate it, which is why the floor is
    twelve points and the acceptance test is an information criterion rather than r-squared. It
    cannot see a transition wider than the window, and returns a lower bound instead. And it has no
    opinion about whether the fitted transition is a *behavioural* one: that is a property of the
    series it was pointed at, so pointing it at a training reward and reading the result as a
    hacking transition is an error this instrument cannot catch and its `series` field is what
    makes visible.
    """

    name = "TransitionWidth"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to: str | None = "H4"
    deviations: tuple[str, ...] = (
        "the width can come from 'a changepoint or sigmoid fit'. This is the sigmoid "
        "half only. A changepoint fit answers a different question (where did it change) and "
        "`stats.changepoint` already answers it; what a lead time needs is a width, and a "
        "changepoint has none.",
        "the width is the 10-to-90 rise of the fitted logistic. No convention is fixed for it "
        "elsewhere, and 10-to-90 is the one that makes the width a fixed multiple of the fitted "
        "scale and is the convention rise time is reported in everywhere else it is measured.",
    )

    quantity = "run.transition_width"
    requires: AccessMatrix = _TRANSITION_ACCESS
    substrates = frozenset(Substrate)
    #: Not PRE_RUN: there is no series before the run. Not DEPLOYED: only the artifact survives
    #: there, and a transition is a property of the process that produced it.
    phases = frozenset({Phase.IN_RUN, Phase.POST_RUN})
    envelope = TRANSITION_ENVELOPE
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = TRANSITION_BASELINES
    rung = 0

    def __init__(
        self,
        run: Run,
        *,
        series: str = "group_mean",
        window: Window | None = None,
        channel: str | None = None,
        criteria: TransitionCriteria | None = None,
        seed: int = 0,
    ) -> None:
        self.run = run
        self.series = series
        self.window = window
        self.channel = channel
        self.criteria = criteria or TransitionCriteria()
        self.seed = seed
        self._computed: TransitionFit | None = None
        self._series: tuple[np.ndarray, np.ndarray] | None = None

    def _resolve(self) -> tuple[np.ndarray, np.ndarray]:
        """The named series, decoded once. A record read is a tenth of a second per pass."""
        if self._series is None:
            self._series = series_from_run(
                self.run, self.series, window=self.window, channel=self.channel
            )
        return self._series

    def compute(self) -> "TransitionFit | Refusal":
        values, indices = self._resolve()
        if values.size == 0:
            have = available_series(self.run, window=self.window)
            return refuse_incomplete(
                self.name,
                field=f"a per-step series named {self.series!r}",
                subject=f"run {self.run.id}",
                remedy=(
                    "name a channel this record carries. It carries "
                    + (", ".join(sorted(have)[:12]) if have else "no per-step numeric channel")
                    + ". If the series you want is a labelled outcome, it has to be written at "
                    "record time as a `ProbeResult(channel='gold')`; nothing downstream can "
                    "recover a label the run did not take."
                ),
                available=sorted(have),
            )
        return fit_transition(
            values,
            indices,
            series=self.series,
            instrument=self.name,
            criteria=self.criteria,
            seed=self.seed,
        )

    def estimate(self, ctx: Context) -> Reading:
        """Preflight, compute, refuse or emit. Never a bare width, never a silent zero."""
        pre = self.preflight(ctx)
        if not pre.ok and pre.refusal is not None:
            return pre.refusal
        out = self.compute()
        if isinstance(out, Refusal):
            return out
        self._computed = out
        try:
            return run(self, ctx)
        finally:
            self._computed = None

    def measure(self, ctx: Context) -> "Evidence":
        out = self._computed if self._computed is not None else self.compute()
        if isinstance(out, Refusal):
            raise ValueError(
                f"{self.name}.measure was called on a series that declines to produce Evidence: "
                f"{out.reason.name}. Call `estimate`, which returns the refusal as a value with "
                f"its remedy."
            )
        return ctx.emit(out, baselines=self.baseline_scores(out))

    def baseline_scores(self, fit: TransitionFit) -> dict[str, float]:
        """What the two dumb comparators say the width is.

        `baseline.whole_run_width` is what a reader does with no fit at all, which is to treat the
        whole observed span as the transition. `baseline.first_crossing` is the other reflex: the
        distance between the first and last steps at which the series crosses the midpoint of its
        own range, which needs no model and is what a threshold rule measures. Both are widths in
        the same unit as the reading, so the comparison is a real one.
        """
        lo, hi = fit.span
        values, indices = self._resolve()
        half = float((np.max(values) + np.min(values)) / 2.0)
        above = values >= half
        crossings = np.flatnonzero(np.diff(above.astype(int)) != 0)
        crossing_width = (
            float(indices[crossings[-1] + 1] - indices[crossings[0]])
            if crossings.size
            else float("nan")
        )
        return {
            "baseline.whole_run_width": float(hi - lo),
            "baseline.first_crossing": crossing_width,
        }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _register() -> None:
    """One rung for `run.transition_width`, and the honest note that the ladder has no second.

    `spec/QUANTITIES.yaml` gives this quantity two rungs and the catalogue's H4 record prints its
    ladder as `OPEN`, so what the second rung is has not been decided anywhere. It is not invented
    here: a rung registered with no estimator behind it reads as a plan and this one would be a
    guess.
    """
    register_estimator(
        EstimatorEntry(
            quantity="run.transition_width",
            impl="run.transition_width.logistic_10_90",
            requires=_TRANSITION_ACCESS,
            envelope=TRANSITION_ENVELOPE,
            rung=0,
            bias=BiasStatement(
                direction="upward",
                why=(
                    "measured on planted logistics, 200 replicates per cell at n = 200 and a "
                    "planted width of 15 or 40 steps: the mean fitted width runs +0.17 percent "
                    "high at residual sd 0.02 and +2.2 percent high at sd 0.2, and the drift is "
                    "upward at every noise level in that range. Upward is the safe direction, "
                    "because a width biased up makes every lead measured against it smaller. The "
                    "one bias that runs the other way is truncation: a transition running past "
                    "the last recorded step cannot be fitted wider than the window, and that case "
                    "is refused with a lower bound rather than reported."
                ),
                magnitude=0.022,
            ),
            cost=CostModel(
                note="one pass over the window's steps plus 200 four-parameter refits; no grader "
                "calls, no GPU"
            ),
            phases=frozenset({Phase.IN_RUN, Phase.POST_RUN}),
            run=None,
        )
    )


_register()


__all__ = [
    "SERIES",
    "TEN_TO_NINETY",
    "TRANSITION_BASELINES",
    "TRANSITION_ENVELOPE",
    "FitQuality",
    "LeadTime",
    "TransitionCriteria",
    "TransitionFit",
    "TransitionWidth",
    "available_series",
    "compare_lead_times",
    "fit_transition",
    "lead_time",
    "series_from_run",
    "window_steps",
]

"""I5: the derivative of within-group reward variance, against the gradient-norm peak.

The argument is precise and it is somebody else's measurement: hidden reward variance was measured
to peak within 0 to 2 steps of liftoff while the ramp climbs over roughly 20 steps beforehand, so
the **level** is not predictive and the **derivative** might be. veRL already logs `E[A^2]`, so the
instrument costs reading logs that already exist. Nothing here reproduces that measurement; it is
the reason the quantity is worth a rung.

**Units, which is the whole point of reporting a lead time at all.** A lead of 14 steps is not
comparable across runs, and no two published lead-time numbers in this literature are. The unit is
the fitted transition width: the lead divided by the 10-to-90 rise time of a logistic fitted to the
outcome series. A 40-step lead against a 58-step width is 0.69 of a window, and that number
transfers. Without an outcome series there is no onset to lead and no width to divide by, and this
instrument returns the alarm steps as a bound rather than inventing a denominator.

**The alarm** is Page's CUSUM with a threshold derived from a target in-control average run length
rather than picked. The shipped `stats.changepoint.cusum` defaults, drift 0.5 and threshold 5.0,
imply one false alarm every 469 steps under Siegmund's approximation, which is a design nobody
made. `cusum_threshold` inverts the approximation so the threshold follows from the false-alarm
budget. The detector itself is the library's; nothing here ships a second changepoint
implementation.

**Two baselines, and the scoping correction that makes the comparison fair.** "The level of
variance is not predictive" is a claim about where the level *peaks*, and it is not the same claim
as "a CUSUM on the level fires late". A CUSUM is itself a change detector, so it alarms on the
level's rise, which starts before the peak. The level baseline is therefore run as a CUSUM on the
level and not as the level's argmax, or the comparison is rigged toward I5. The gradient-norm
baseline is reported both ways for the same reason: as a CUSUM, and as the smoothed peak that
`stats.baselines.series.gradnorm_peak` computes, which is the free leading indicator every trainer
already logs and the one any new statistic has to beat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import brentq, curve_fit
from scipy.stats import f as _f_dist

from reward_lens.core.evidence import make_evidence, register_payload
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.quantity import BaselineID
from reward_lens.core.reading import Refusal, RefusalReason, bounded_refusal, refuse_incomplete
from reward_lens.core.types import Capability, GaugeStatus, SubjectRef
from reward_lens.measure.threshold._base import (
    ALL_SUBSTRATES,
    RECORD_ACCESS,
    RECORD_PHASES,
    VARIANCE_ENVELOPE,
    ThresholdInstrument,
)
from reward_lens.record.schema import Run
from reward_lens.stats.baselines.series import gradnorm_peak
from reward_lens.stats.changepoint import ChangePoint, cusum


def f_sf(statistic: float, dfn: int, dfd: int) -> float:
    """The upper tail of the F distribution. Named so the F test reads as one."""
    return float(_f_dist.sf(statistic, dfn, dfd))


#: The two the catalogue names, and both run on every reading. The level is not optional and the
#: gradient norm is the kill condition: an alarm that does not beat it is not worth adding.
VARIANCE_BASELINES: tuple[BaselineID, ...] = (
    "baseline.variance_level",
    "baseline.gradnorm_peak",
)

#: The false-alarm budget the CUSUM threshold is solved for. One alarm per thousand in-control
#: steps, which is a design rather than a default: the shipped threshold of 5.0 implies 469.
DEFAULT_ARL0 = 1000.0


# ---------------------------------------------------------------------------
# transition width, which is the unit
# ---------------------------------------------------------------------------


def _logistic(t: np.ndarray, baseline: float, amplitude: float, midpoint: float, rate: float):
    return baseline + amplitude / (1.0 + np.exp(-(t - midpoint) * rate))


@register_payload
@dataclass(frozen=True)
class TransitionFit:
    """A fitted behavioural transition: where it happened, how wide it was, and whether it is one.

    ``r_squared`` and ``p_value`` are the fields that stop this returning a transition on a flat
    series, and they were added because it did. A four-parameter logistic fitted to two hundred
    steps of a run that is not going anywhere converges happily on a step through the noise: on the
    real 200-step record in the fixtures it returned a midpoint of 169.2 and a width of 2.7 steps
    against a residual root-mean-square of 0.098 on a series whose own spread is 0.10. Every lead
    time divided by that width would have been a confident wrong number, and the number it produced
    was a lead of 12.5 transition windows, which is arithmetic rather than a measurement.

    ``p_value`` is an F test of the logistic against the constant model, on the same series. A
    transition that does not beat a horizontal line is not a transition.
    """

    midpoint: float
    width: float  # the 10-to-90 rise time, in units of the step axis
    amplitude: float
    baseline: float
    rate: float
    rmse: float
    method: str
    r_squared: float = float("nan")
    p_value: float = float("nan")
    n: int = 0

    @property
    def converged(self) -> bool:
        """Whether the optimiser returned a finite positive width. Not the same as `valid`."""
        return bool(np.isfinite(self.width) and self.width > 0)

    def valid_at(self, alpha: float) -> bool:
        """Whether this is a transition at the stated false-positive rate."""
        return bool(self.converged and np.isfinite(self.p_value) and self.p_value < alpha)

    @property
    def valid(self) -> bool:
        """Converged and beating the constant model at one in a thousand.

        The level matches `DEFAULT_ARL0`, so the fit and the alarm are designed against the same
        false-positive budget rather than against two numbers chosen separately.
        """
        return self.valid_at(1.0 / DEFAULT_ARL0)

    def render(self) -> str:
        if not self.converged:
            return f"transition fit failed ({self.method})"
        verdict = "" if self.valid else "  NOT a transition at 1 in 1000: "
        return (
            f"{verdict}transition at step {self.midpoint:.1f}, 10-to-90 width {self.width:.1f} "
            f"steps, amplitude {self.amplitude:.4g} (rmse {self.rmse:.5g}, R2 {self.r_squared:.3f}, "
            f"F test against a constant p {self.p_value:.3g}, {self.method})"
        )


def fit_transition(
    outcome: Sequence[float] | np.ndarray, steps: Sequence[float] | np.ndarray | None = None
) -> TransitionFit:
    """Fit a logistic to an outcome series and report its 10-to-90 rise time as the width.

    ``width = 2 * ln(9) / rate``, the interval over which the fitted curve travels from 10 percent
    to 90 percent of its total rise. That is the denominator every lead time here is divided by, so
    it is reported alongside every lead and never left implicit.

    The fit is least squares with a starting point taken from the data: the midpoint starts at the
    step of steepest observed change and the rate at one over a quarter of the series length. If
    the optimiser does not converge the returned fit has ``width = nan`` and ``converged`` is
    False, which callers must check rather than divide by.

    Convergence is not the same as a transition, and the difference is the whole reason this
    returns an F test. The logistic has four parameters and the constant has one, so the comparison
    is an F test on three extra degrees of freedom against the residual of the constant fit. On a
    flat series the optimiser still converges, on a step through the noise, and it is the F test
    rather than the convergence flag that says so.
    """
    y = np.asarray(outcome, dtype=np.float64).ravel()
    t = (
        np.arange(y.size, dtype=np.float64)
        if steps is None
        else np.asarray(steps, dtype=np.float64)
    )
    keep = np.isfinite(y) & np.isfinite(t)
    y, t = y[keep], t[keep]
    if y.size < 6:
        return TransitionFit(
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "logistic:too-short", n=int(y.size)
        )
    span = float(t[-1] - t[0]) or 1.0
    d = np.gradient(y, t)
    p0 = (
        float(y.min()),
        float(y.max() - y.min()) or 1.0,
        float(t[int(np.argmax(np.abs(d)))]),
        4.0 / span,
    )
    try:
        popt, _ = curve_fit(_logistic, t, y, p0=p0, maxfev=20000)
    except (RuntimeError, ValueError):
        return TransitionFit(
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            "logistic:no-convergence",
            n=int(y.size),
        )
    baseline, amplitude, midpoint, rate = (float(v) for v in popt)
    resid = y - _logistic(t, *popt)
    rss = float(resid @ resid)
    rmse = float(np.sqrt(rss / y.size))
    width = float(2.0 * np.log(9.0) / abs(rate)) if rate else float("inf")
    null_resid = y - float(np.mean(y))
    rss0 = float(null_resid @ null_resid)
    r_squared = 1.0 - rss / rss0 if rss0 > 0 else float("nan")
    dof = y.size - 4
    if rss > 0 and rss0 > rss and dof > 0:
        f_stat = ((rss0 - rss) / 3.0) / (rss / dof)
        p_value = float(f_sf(f_stat, 3, dof))
    else:
        p_value = float("nan")
    return TransitionFit(
        midpoint,
        width,
        amplitude,
        baseline,
        rate,
        rmse,
        "logistic-10-90",
        r_squared=float(r_squared),
        p_value=p_value,
        n=int(y.size),
    )


def lead_time_in_widths(alarm_step: float, onset_step: float, width: float) -> float:
    """``(onset - alarm) / width``. Positive means the alarm fired before the onset."""
    if not np.isfinite(width) or width <= 0:
        return float("nan")
    return float((onset_step - alarm_step) / width)


def cadence_resolution_in_widths(cadence: float, width: float) -> float:
    """The finest lead a series sampled every ``cadence`` steps can resolve, in width units.

    A lead smaller than one sampling interval is not measured, it is rounded. Reporting this beside
    a lead time is what stops a coarse log from producing a confident fraction.
    """
    if not np.isfinite(width) or width <= 0:
        return float("nan")
    return float(cadence / width)


# ---------------------------------------------------------------------------
# the ARL-designed alarm
# ---------------------------------------------------------------------------


def arl0(threshold: float, drift: float = 0.5) -> float:
    """Siegmund's approximation to the in-control average run length of a two-sided CUSUM.

    One-sided: ``(exp(2 k (h + 1.166)) - 2 k (h + 1.166) - 1) / (2 k^2)``. Two arms running in
    parallel halve it. This is an approximation and it is the standard one; it is used here to turn
    a false-alarm budget into a threshold, not to make a claim about the exact run length.
    """
    d = 2.0 * drift * (threshold + 1.166)
    return float((np.exp(d) - d - 1.0) / (2.0 * drift * drift) / 2.0)


def cusum_threshold(target_arl0: float, drift: float = 0.5) -> float:
    """Invert ``arl0``: the threshold ``h`` giving one false alarm per ``target_arl0`` steps.

    At ``target_arl0 = 1000`` and ``drift = 0.5`` this returns 5.75; the source quotes 5.71
    for the same design, which is the same number to the precision the approximation supports
    (5.71 scores an ARL of 960 under this form).
    """
    if target_arl0 <= 1:
        raise ValueError(f"target_arl0 must exceed 1; got {target_arl0}")
    return float(brentq(lambda h: arl0(h, drift) - target_arl0, 1e-6, 100.0))


def derivative(
    series: Sequence[float] | np.ndarray, steps: Sequence[float] | np.ndarray | None = None
):
    """``d/dt`` of a series by central differences, honouring a non-uniform step axis."""
    y = np.asarray(series, dtype=np.float64).ravel()
    t = (
        np.arange(y.size, dtype=np.float64)
        if steps is None
        else np.asarray(steps, dtype=np.float64)
    )
    return np.gradient(y, t)


def peak_index(series: Sequence[float] | np.ndarray) -> int:
    """The index of the maximum. The retrospective reading of "the gradient-norm peak"."""
    return int(np.argmax(np.asarray(series, dtype=np.float64).ravel()))


def first_alarm(
    series: Sequence[float] | np.ndarray,
    *,
    target_arl0: float = DEFAULT_ARL0,
    drift: float = 0.5,
    baseline: int | None = None,
) -> ChangePoint:
    """The library CUSUM with a threshold derived from a false-alarm budget.

    ``baseline`` is the number of leading samples treated as in control, which must be chosen from
    the run's own pre-transition period and not from the whole series: standardising against a
    window that already contains the transition is what makes an alarm look late.
    """
    return cusum(
        np.asarray(series, dtype=np.float64).ravel(),
        threshold=cusum_threshold(target_arl0, drift),
        drift=drift,
        baseline=baseline,
    )


@register_payload
@dataclass(frozen=True)
class AlarmCalibration:
    """How often this detector fires on this run's own series with the time order destroyed.

    The ARL design assumes the in-control series is independent and Gaussian, and a within-group
    reward variance series is neither. Measured on this implementation: on 400 samples of 200 iid
    standard normal steps the ARL-1000 design fired on 119 of them against the 73 implied by
    ``1 - exp(-200/1000)``, which is the price of estimating the baseline mean and spread from 60
    points; on lognormal steps it fired on 303 and on an AR(1) with rho 0.5 on 388. A heavy tail or
    a little autocorrelation turns a one-in-a-thousand design into a coin flip.

    So the design rate is not the reference to compare an alarm against, and this is. A circular
    block bootstrap of the run's own **in-control prefix** is grown to the full length of the
    series, which keeps that period's marginal distribution and its short-range dependence and
    contains no change point at all by construction. The fraction of those surrogates that fire is
    the false-alarm rate for this series. An alarm that fires on most surrogates too is not a
    detection, whatever its step index says.

    The prefix matters and the first version of this got it wrong: bootstrapping the **whole**
    series preserves a marginal that already contains the excursion, so on a planted run with a
    real bump every surrogate fired and the reference said nothing. The in-control window is the
    same one the CUSUM standardises against, so the two agree about what "before" means.
    """

    detector: str
    n_surrogates: int
    fired_fraction: float
    design_rate: float
    block: int
    n_in_control: int = 0

    @property
    def informative(self) -> bool:
        """Whether firing on this series says more than a coin flip would."""
        return bool(np.isfinite(self.fired_fraction) and self.fired_fraction < 0.5)

    @property
    def excess(self) -> float:
        """How many times the design rate this detector actually fires at on this series."""
        if not np.isfinite(self.fired_fraction) or self.design_rate <= 0:
            return float("nan")
        return float(self.fired_fraction / self.design_rate)

    def render(self) -> str:
        return (
            f"{self.detector:<30}fires on {self.fired_fraction:.1%} of {self.n_surrogates} "
            f"in-control surrogates (block {self.block} from {self.n_in_control} steps), against a "
            f"design rate of {self.design_rate:.1%}: {self.excess:.1f} times the budget"
        )


def alarm_calibration(
    series: Sequence[float] | np.ndarray,
    *,
    detector: str,
    target_arl0: float = DEFAULT_ARL0,
    baseline: int | None = None,
    n_surrogates: int = 200,
    seed: int = 0,
) -> AlarmCalibration:
    """Grow surrogates from the in-control prefix, and count how often the detector fires anyway.

    The block length is ``m**(1/3)`` rounded over the ``m`` in-control samples, which is the
    standard rule for a block bootstrap and carries no constant anybody chose. Blocks are drawn with
    replacement from a circularly extended prefix, so every in-control observation is equally likely
    to appear, the prefix's marginal and short-range dependence survive, and the surrogate contains
    no change point at any position.

    ``baseline`` names the in-control window and defaults to the first third of the series, which is
    reported on the reading rather than assumed.
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n < 12:
        return AlarmCalibration(detector, 0, float("nan"), float("nan"), 0, 0)
    m = int(baseline) if baseline else max(12, n // 3)
    m = min(m, n)
    prefix = x[:m]
    block = max(1, int(round(m ** (1.0 / 3.0))))
    rng = np.random.default_rng(seed)
    extended = np.concatenate([prefix, prefix[: block - 1]]) if block > 1 else prefix
    n_blocks = int(np.ceil(n / block))
    fired = 0
    for _ in range(int(n_surrogates)):
        starts = rng.integers(0, m, size=n_blocks)
        pieces = [extended[s : s + block] for s in starts]
        surrogate = np.concatenate(pieces)[:n]
        if first_alarm(surrogate, target_arl0=target_arl0, baseline=baseline).index is not None:
            fired += 1
    return AlarmCalibration(
        detector=detector,
        n_surrogates=int(n_surrogates),
        fired_fraction=fired / float(n_surrogates),
        design_rate=float(1.0 - math.exp(-n / target_arl0)),
        block=block,
        n_in_control=m,
    )


@register_payload
@dataclass(frozen=True)
class DetectorResult:
    """One detector's answer on one run."""

    name: str
    alarm_step: float | None
    lead_steps: float | None
    lead_in_widths: float | None

    def render(self) -> str:
        if self.alarm_step is None:
            return f"{self.name:<30}no alarm"
        widths = "-" if self.lead_in_widths is None else f"{self.lead_in_widths:+.3f}"
        steps = "-" if self.lead_steps is None else f"{self.lead_steps:+.1f}"
        return f"{self.name:<30}alarm at {self.alarm_step:>7.1f}  lead {steps:>8} steps  {widths:>8} widths"


def score_detectors(
    *,
    variance: Sequence[float] | np.ndarray,
    grad_norm: Sequence[float] | np.ndarray | None,
    outcome: Sequence[float] | np.ndarray,
    steps: Sequence[float] | np.ndarray | None = None,
    target_arl0: float = DEFAULT_ARL0,
    baseline: int | None = None,
) -> tuple[TransitionFit, list[DetectorResult]]:
    """Run I5's statistic and its two mandatory baselines against one labelled run.

    Returns the fitted transition and one ``DetectorResult`` per detector. The onset the leads are
    measured against is the fitted midpoint of the outcome series, not the first labelled nonzero,
    because the midpoint is what the width is defined around.
    """
    v = np.asarray(variance, dtype=np.float64).ravel()
    t = (
        np.arange(v.size, dtype=np.float64)
        if steps is None
        else np.asarray(steps, dtype=np.float64)
    )
    fit = fit_transition(outcome, t)

    def result(name: str, cp: ChangePoint) -> DetectorResult:
        if cp.index is None:
            return DetectorResult(name, None, None, None)
        step = float(t[cp.index])
        return DetectorResult(
            name,
            step,
            float(fit.midpoint - step),
            lead_time_in_widths(step, fit.midpoint, fit.width),
        )

    out = [
        result(
            "variance derivative, CUSUM",
            first_alarm(derivative(v, t), target_arl0=target_arl0, baseline=baseline),
        ),
        result("variance level, CUSUM", first_alarm(v, target_arl0=target_arl0, baseline=baseline)),
    ]
    if grad_norm is not None:
        g = np.asarray(grad_norm, dtype=np.float64).ravel()
        out.append(
            result(
                "gradient norm, CUSUM", first_alarm(g, target_arl0=target_arl0, baseline=baseline)
            )
        )
        peak = gradnorm_peak(g)
        if peak.index is not None:
            gp = float(t[peak.index])
            out.append(
                DetectorResult(
                    "gradient norm, peak",
                    gp,
                    float(fit.midpoint - gp),
                    lead_time_in_widths(gp, fit.midpoint, fit.width),
                )
            )
        else:
            out.append(DetectorResult("gradient norm, peak", None, None, None))
    return fit, out


# ---------------------------------------------------------------------------
# the series, off a record
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class RunSeries:
    """The three per-step series I5 reads, with the ones the record did not carry named.

    ``outcome`` is the one that is usually missing and it is the one the unit depends on. A run
    with no labelled behavioural series has no onset to lead and no width to divide by, and the
    instrument bounds rather than guessing a denominator.
    """

    steps: np.ndarray
    variance: np.ndarray
    grad_norm: np.ndarray | None
    outcome: np.ndarray | None
    outcome_source: str
    grad_norm_source: str
    degenerate_fraction: float
    n_groups: int
    missing: tuple[str, ...] = ()

    def render(self) -> str:
        parts = [
            f"{self.steps.size} steps over {self.n_groups} groups, "
            f"degenerate group fraction {self.degenerate_fraction:.1%}"
        ]
        parts.append(f"gradient norm from {self.grad_norm_source}")
        parts.append(
            f"outcome from {self.outcome_source}" if self.outcome is not None else "no outcome"
        )
        if self.missing:
            parts.append(f"missing: {', '.join(self.missing)}")
        return "; ".join(parts)


def run_series(run: Run, *, span: tuple[int, int] | None = None) -> RunSeries:
    """Pull the within-group variance, the gradient norm and any outcome series off a record.

    The within-group variance is the mean over the step's groups of the squared group standard
    deviation, which is the quantity veRL logs as `E[A^2]` before normalisation. Groups whose
    standard deviation was not recorded are excluded from the mean and counted, rather than treated
    as zero variance, because a group whose spread nobody wrote down is not a group with no spread.

    The gradient norm prefers the unclipped one where the record carries it. Where only the clipped
    norm is there the source says so, and the reading says so: a clipped norm makes every
    clip-crossing step look like a step at the threshold, which biases exactly the steps a change
    detector cares about.
    """
    steps: list[int] = []
    variance: list[float] = []
    grads: list[float] = []
    outcome: list[float] = []
    n_degenerate = 0
    n_groups = 0
    grad_source = "OptimizerTelemetry.grad_norm_unclipped"
    used_clipped = False
    missing: list[str] = []

    stream = run.steps.slice(*span) if span is not None else iter(run.steps)
    for step in stream:
        spreads = []
        for group in step.groups:
            n_groups += 1
            if group.group_stats.degenerate:
                n_degenerate += 1
            if group.group_stats.std is not None:
                spreads.append(float(group.group_stats.std) ** 2)
        steps.append(int(step.index))
        variance.append(float(np.mean(spreads)) if spreads else float("nan"))
        g = step.optimizer.grad_norm_unclipped
        if g is None:
            g = step.optimizer.grad_norm_clipped
            if g is not None:
                used_clipped = True
        grads.append(float("nan") if g is None else float(g))
        mean_reward = step.optimizer.extra.get("reward")
        outcome.append(float("nan") if mean_reward is None else float(mean_reward))

    if used_clipped:
        grad_source = "OptimizerTelemetry.grad_norm_clipped (the unclipped norm is not recorded)"
    grad_arr: np.ndarray | None = np.asarray(grads, dtype=np.float64)
    if grad_arr is not None and not np.any(np.isfinite(grad_arr)):
        grad_arr = None
        grad_source = "not recorded"
        missing.append("gradient norm")
    outcome_arr: np.ndarray | None = np.asarray(outcome, dtype=np.float64)
    outcome_source = "mean reward per step, `reward` in OptimizerTelemetry.extra"
    if outcome_arr is not None and not np.any(np.isfinite(outcome_arr)):
        outcome_arr = None
        outcome_source = "not recorded"
        missing.append("outcome series")
    return RunSeries(
        steps=np.asarray(steps, dtype=np.float64),
        variance=np.asarray(variance, dtype=np.float64),
        grad_norm=grad_arr,
        outcome=outcome_arr,
        outcome_source=outcome_source,
        grad_norm_source=grad_source,
        degenerate_fraction=(n_degenerate / n_groups) if n_groups else float("nan"),
        n_groups=n_groups,
        missing=tuple(missing),
    )


# ---------------------------------------------------------------------------
# the reading
# ---------------------------------------------------------------------------


@register_payload
@dataclass(frozen=True)
class VarianceDerivativeReading:
    """Every detector's alarm, scored in transition-width units against the same onset."""

    fit: TransitionFit
    detectors: tuple[DetectorResult, ...]
    series: RunSeries
    target_arl0: float
    threshold: float
    cadence: float
    resolution_in_widths: float
    n_steps: int
    expected_false_alarms: float
    n_alarms: int
    calibration: tuple[AlarmCalibration, ...] = ()

    def detector(self, name: str) -> DetectorResult | None:
        for d in self.detectors:
            if d.name == name:
                return d
        return None

    @property
    def beats_gradnorm(self) -> bool | None:
        """Whether I5's alarm leads the gradient-norm peak. The catalogue's kill condition."""
        mine = self.detector("variance derivative, CUSUM")
        theirs = self.detector("gradient norm, peak")
        if mine is None or theirs is None:
            return None
        if mine.lead_in_widths is None or theirs.lead_in_widths is None:
            return None
        return bool(mine.lead_in_widths > theirs.lead_in_widths)

    @property
    def says(self) -> str:
        mine = self.detector("variance derivative, CUSUM")
        if mine is None or mine.alarm_step is None:
            return "the variance derivative did not cross its CUSUM threshold on this run"
        if mine.lead_in_widths is None or not math.isfinite(mine.lead_in_widths):
            return (
                f"d/dt of within-group reward variance crossed its CUSUM threshold at step "
                f"{mine.alarm_step:.0f}. No transition was fitted, so the lead has no unit"
            )
        return (
            f"d/dt of within-group reward variance crossed its CUSUM threshold at step "
            f"{mine.alarm_step:.0f}, {mine.lead_steps:.0f} steps before the outcome series moved. "
            f"Lead = {mine.lead_in_widths:.2f} of the transition window"
        )

    def render(self) -> str:
        lines = [
            "I5 variance derivative",
            f"  {self.says}",
            f"  {self.fit.render()}",
            f"  {self.series.render()}",
            f"  CUSUM threshold {self.threshold:.4f} from a budget of one false alarm per "
            f"{self.target_arl0:.0f} steps; over {self.n_steps} steps that is "
            f"{self.expected_false_alarms:.2f} expected, {self.n_alarms} observed",
            f"  cadence {self.cadence:g} steps, so the finest resolvable lead is "
            f"{self.resolution_in_widths:.3f} widths",
        ]
        lines.extend(f"  {d.render()}" for d in self.detectors)
        lines.extend(f"  {c.render()}" for c in self.calibration)
        verdict = self.beats_gradnorm
        if verdict is not None:
            lines.append(
                "  beats the gradient-norm peak"
                if verdict
                else "  does NOT beat the gradient-norm peak, which is the catalogue's kill "
                "condition for this instrument"
            )
        return "\n".join(lines)


def variance_derivative(
    series: RunSeries,
    *,
    target_arl0: float = DEFAULT_ARL0,
    baseline_steps: int | None = None,
    n_surrogates: int = 200,
    instrument: str = "VarianceDerivative",
) -> VarianceDerivativeReading | Refusal:
    """I5's reading, or the refusal that says why there is none.

    The refusal that matters here is the one for a run with no behavioural transition. The alarms
    are still computable and still worth having, so it is a `bounded_refusal`: the bound carries
    every detector's alarm step and its lead in raw steps, and what is refused is the conversion to
    transition widths, which is the only unit in which two runs can be compared.
    """
    v = np.asarray(series.variance, dtype=np.float64)
    finite = v[np.isfinite(v)]
    if finite.size < 8:
        return refuse_incomplete(
            instrument,
            field=f"a within-group reward variance on more than {finite.size} of {v.size} steps",
            subject="the record",
            remedy=(
                "record the group reward tensor or its per-group standard deviation on every step. "
                "veRL's `E[A^2]` is sufficient and is already computed; the record's `GroupStats` "
                "field carries it where the tap is attached."
            ),
            n_steps=int(v.size),
            n_finite=int(finite.size),
        )
    if series.degenerate_fraction >= 1.0:
        return Refusal(
            instrument=instrument,
            reason=RefusalReason.ENVELOPE_VIOLATED,
            detail=(
                f"every one of {series.n_groups} groups is degenerate, so the within-group reward "
                f"variance is identically zero and its derivative is the derivative of a constant."
            ),
            remedy=(
                "restrict the window to steps where at least some groups have spread, or raise the "
                "group size. A run in which no group ever disagrees with itself has no within-group "
                "variance to take a derivative of."
            ),
            statistics={
                "degenerate_fraction": series.degenerate_fraction,
                "n_groups": series.n_groups,
            },
        )

    steps = np.asarray(series.steps, dtype=np.float64)
    cadence = float(np.median(np.diff(steps))) if steps.size > 1 else float("nan")
    threshold = cusum_threshold(target_arl0)
    expected = float(steps.size / target_arl0)

    calibration: list[AlarmCalibration] = [
        alarm_calibration(
            derivative(v, steps),
            detector="variance derivative, CUSUM",
            target_arl0=target_arl0,
            baseline=baseline_steps,
            n_surrogates=n_surrogates,
        ),
        alarm_calibration(
            v,
            detector="variance level, CUSUM",
            target_arl0=target_arl0,
            baseline=baseline_steps,
            n_surrogates=n_surrogates,
        ),
    ]
    if series.grad_norm is not None:
        calibration.append(
            alarm_calibration(
                series.grad_norm,
                detector="gradient norm, CUSUM",
                target_arl0=target_arl0,
                baseline=baseline_steps,
                n_surrogates=n_surrogates,
            )
        )

    if series.outcome is None:
        alarms = [
            (
                "variance derivative, CUSUM",
                first_alarm(derivative(v, steps), target_arl0=target_arl0, baseline=baseline_steps),
            ),
            (
                "variance level, CUSUM",
                first_alarm(v, target_arl0=target_arl0, baseline=baseline_steps),
            ),
        ]
        if series.grad_norm is not None:
            alarms.append(
                (
                    "gradient norm, CUSUM",
                    first_alarm(series.grad_norm, target_arl0=target_arl0, baseline=baseline_steps),
                )
            )
        unlabelled = tuple(
            DetectorResult(name, None if cp.index is None else float(steps[cp.index]), None, None)
            for name, cp in alarms
        )
        n_alarms = sum(1 for d in unlabelled if d.alarm_step is not None)
        bound = make_evidence(
            observable=instrument,
            observable_version="1.0",
            subject=SubjectRef(signals=(), dataset=None, readout="reward"),
            value=unlabelled,
            gauge=GaugeStatus.INVARIANT,
        )
        return bounded_refusal(
            instrument,
            RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"the record carries no labelled outcome series, so there is no onset for an alarm "
                f"to lead and no transition width to express the lead in. The alarms themselves are "
                f"computable and are in the bound: {n_alarms} of {len(unlabelled)} detectors fired "
                f"over {steps.size} steps against {expected:.2f} expected from the "
                f"false-alarm budget."
            ),
            remedy=(
                "record one labelled outcome per step, a hack rate, a gold-reward gap, or any "
                "behavioural series with a transition in it, together with how it was labelled. "
                "Then re-read: every number here becomes a lead time in transition-width units, "
                "which is the only unit in which two runs are comparable."
            ),
            bound=bound,
            n_steps=int(steps.size),
            n_alarms=n_alarms,
            expected_false_alarms=expected,
            cusum_threshold=threshold,
            outcome_source=series.outcome_source,
            surrogate_alarm_rate={c.detector: c.fired_fraction for c in calibration},
        )

    fit, detectors = score_detectors(
        variance=v,
        grad_norm=series.grad_norm,
        outcome=series.outcome,
        steps=steps,
        target_arl0=target_arl0,
        baseline=baseline_steps,
    )
    alpha = 1.0 / target_arl0
    resolvable = math.isfinite(cadence) and fit.converged and fit.width >= cadence
    if not fit.valid_at(alpha) or not resolvable:
        # The bound carries the alarm steps and nothing derived from the width. Leaving the
        # width-scaled leads on it would put the exact numbers this refusal exists to withhold
        # inside the object a caller reaches for when the refusal says there is still an answer.
        stripped = tuple(DetectorResult(d.name, d.alarm_step, None, None) for d in detectors)
        bound = make_evidence(
            observable=instrument,
            observable_version="1.0",
            subject=SubjectRef(signals=(), dataset=None, readout="reward"),
            value=stripped,
            gauge=GaugeStatus.INVARIANT,
        )
        n_alarms = sum(1 for d in stripped if d.alarm_step is not None)
        if not fit.converged:
            why = f"the logistic did not converge on the outcome series ({fit.method})"
        elif not fit.valid_at(alpha):
            why = (
                f"the fitted logistic does not beat a constant on the outcome series: R2 "
                f"{fit.r_squared:.3f}, F test p {fit.p_value:.3g} against the {alpha:.4g} implied "
                f"by the false-alarm budget. The optimiser converged, on a step through the noise, "
                f"and its width of {fit.width:.1f} steps would have divided every lead below"
            )
        else:
            why = (
                f"the fitted width of {fit.width:.2f} steps is finer than the {cadence:g}-step "
                f"logging cadence, so the transition is not resolved by this series and dividing "
                f"by the width rounds rather than measures"
            )
        return bounded_refusal(
            instrument,
            RefusalReason.RECORD_INCOMPLETE,
            detail=(
                f"{why}. Without a width the leads have no unit, and a lead in raw steps is not "
                f"comparable with any other run's. The alarms are in the bound: {n_alarms} of "
                f"{len(detectors)} detectors fired over {steps.size} steps against "
                f"{expected:.2f} expected from the false-alarm budget."
            ),
            remedy=(
                "supply an outcome series that contains a behavioural transition, or restrict the "
                "window to the steps around one. A flat outcome series is a real reading about the "
                "run and it means there is nothing for an early-warning statistic to be early "
                "about."
            ),
            bound=bound,
            n_steps=int(steps.size),
            n_alarms=n_alarms,
            expected_false_alarms=expected,
            cusum_threshold=threshold,
            fit_method=fit.method,
            fit_r_squared=fit.r_squared,
            fit_p_value=fit.p_value,
            fit_width=fit.width,
            cadence=cadence,
            surrogate_alarm_rate={c.detector: c.fired_fraction for c in calibration},
        )

    n_alarms = sum(1 for d in detectors if d.alarm_step is not None and "peak" not in d.name)
    return VarianceDerivativeReading(
        fit=fit,
        detectors=tuple(detectors),
        series=series,
        target_arl0=float(target_arl0),
        threshold=float(threshold),
        cadence=cadence,
        resolution_in_widths=cadence_resolution_in_widths(cadence, fit.width),
        n_steps=int(steps.size),
        expected_false_alarms=expected,
        n_alarms=n_alarms,
        calibration=tuple(calibration),
    )


# ---------------------------------------------------------------------------
# a planted run, for validating the machinery
# ---------------------------------------------------------------------------


def planted_run(
    *,
    n_steps: int = 240,
    onset: float = 140.0,
    outcome_scale: float = 8.0,
    variance_peak_offset: float = 0.0,
    variance_sigma: float = 10.0,
    noise: float = 0.02,
    seed: int = 0,
) -> dict[str, np.ndarray | float]:
    """A synthetic run with an analytically known answer for each detector.

    The outcome is a logistic with midpoint ``onset`` and rate ``1 / outcome_scale``, so its
    10-to-90 width is ``2 ln(9) * outcome_scale`` in closed form. The within-group reward variance
    is a Gaussian bump centred ``variance_peak_offset`` steps from the onset, so its level peaks
    there and its derivative peaks exactly ``variance_sigma`` steps earlier. The gradient norm is a
    second bump centred on the onset.

    Where each peak sits is a choice made here. This plant fixes what each detector should say and
    therefore tests the machinery; it does not adjudicate between detectors, because the ordering
    among them is something the plant was told.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps, dtype=np.float64)
    outcome = 1.0 / (1.0 + np.exp(-(t - onset) / outcome_scale))
    v_peak = onset + variance_peak_offset
    variance = 1.0 + 3.0 * np.exp(-0.5 * ((t - v_peak) / variance_sigma) ** 2)
    grad = 1.0 + 2.0 * np.exp(-0.5 * ((t - onset) / variance_sigma) ** 2)
    return {
        "steps": t,
        "outcome": outcome + rng.normal(0.0, noise * 0.25, n_steps),
        "variance": variance + rng.normal(0.0, noise, n_steps),
        "grad_norm": grad + rng.normal(0.0, noise, n_steps),
        "planted_onset": onset,
        "planted_width": float(2.0 * np.log(9.0) * outcome_scale),
        "planted_variance_peak": float(v_peak),
        "planted_derivative_peak": float(v_peak - variance_sigma),
    }


# ---------------------------------------------------------------------------
# the instrument
# ---------------------------------------------------------------------------


class VarianceDerivative(ThresholdInstrument):
    """I5. The derivative of within-group reward variance, as an early warning.

    Kill condition, from the catalogue record: if it does not beat the gradient-norm peak.
    `VarianceDerivativeReading.beats_gradnorm` is that comparison, in transition-width units, on
    every reading that has a width to divide by.
    """

    name = "VarianceDerivative"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "I5"
    deviations = (
        "the within-group variance is the mean over a step's groups of the squared recorded group "
        "standard deviation. veRL logs `E[A^2]` after normalisation, which is the same quantity "
        "up to the estimator's epsilon; where a record carries `E[A^2]` directly it should be "
        "passed rather than recomputed",
        "the transition width is the 10-to-90 rise time of a logistic fitted to the outcome "
        "series. Any monotone sigmoid would give a similar width and a non-monotone transition "
        "gives none, which is why the fit's convergence is checked and refused on rather than "
        "reported",
        "the CUSUM threshold comes from Siegmund's approximation to the in-control run length, "
        "which is an approximation. It is used to turn a false-alarm budget into a threshold and "
        "not to claim an exact run length",
    )

    quantity = "run.variance_derivative"
    requires = RECORD_ACCESS
    substrates = ALL_SUBSTRATES
    phases = RECORD_PHASES
    envelope = VARIANCE_ENVELOPE
    #: `units` in the registry. A variance per step and a variance are not the same quantity, and
    #: this is the instrument where confusing them is easiest.
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = VARIANCE_BASELINES
    rung = 0

    def __init__(
        self,
        series: RunSeries | Run | None = None,
        *,
        target_arl0: float = DEFAULT_ARL0,
        baseline_steps: int | None = None,
        n_surrogates: int = 200,
        span: tuple[int, int] | None = None,
    ) -> None:
        self.series = series
        self.target_arl0 = float(target_arl0)
        self.baseline_steps = baseline_steps
        self.n_surrogates = int(n_surrogates)
        self.span = span

    def compute(self) -> VarianceDerivativeReading | Refusal:
        subject = self.series
        if subject is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ACCESS_INSUFFICIENT,
                detail="no run and no series were supplied",
                remedy=(
                    "pass `series=` a Run, or a RunSeries built by "
                    "`measure.threshold.variance.run_series(run)`. This instrument reads a record "
                    "and computes nothing that needs a policy."
                ),
            )
        series = run_series(subject, span=self.span) if isinstance(subject, Run) else subject
        return variance_derivative(
            series,
            target_arl0=self.target_arl0,
            baseline_steps=self.baseline_steps,
            n_surrogates=self.n_surrogates,
            instrument=self.name,
        )


__all__ = [
    "DEFAULT_ARL0",
    "VARIANCE_BASELINES",
    "AlarmCalibration",
    "DetectorResult",
    "RunSeries",
    "TransitionFit",
    "VarianceDerivative",
    "VarianceDerivativeReading",
    "alarm_calibration",
    "arl0",
    "cadence_resolution_in_widths",
    "cusum_threshold",
    "derivative",
    "first_alarm",
    "fit_transition",
    "lead_time_in_widths",
    "peak_index",
    "planted_run",
    "run_series",
    "score_detectors",
    "variance_derivative",
]

"""The ARL-designed alarm, and its delay in fraction-of-transition-window units (J2).

Three things happen here. The chart runs, the delay is put into a unit that is comparable across
papers, and the whole detector bank is measured against a planted changepoint so the delay and the
false-alarm rate are properties of the procedures rather than anecdotes about one series.

**The unit, which is free to claim because nobody has one.** Lead time is reported in training steps
by one paper, as a fraction of episode elapsed by another, as distance-to-onset by a third, and as
precision and recall on a consecutive-decline rule by a fourth. No two are commensurable, and
Anthropic reports that most of the change in reward-hacking rate occurs within the first 64 steps,
so a 40-to-60-step lead inside a 64-step window is not the margin it sounds like. The unit here is
**delay as a fraction of the fitted transition width**, and the fit travels with the number.

**Where the width comes from.** Instrument H4 in `measure/rate/` fits it, and this module does not
fit a second one. `TransitionWindow` is the interface: hand it H4's fit and the delay is reported
against H4's width with `source="H4"`. `local_transition_width` is the stand-in used until H4 lands,
it is a four-parameter logistic fitted by least squares, it reports its own R-squared, and it
refuses rather than returning a width when the fit does not describe the series. Every reading says
which of the two produced its denominator.

**What the simulation is for and what the record is for.** An average run length is a property of a
procedure under a hypothetical stream, so it is established by simulation against a known planted
changepoint of known width, and that is where the delay and false-alarm numbers come from.
Whether the code runs on a real optimisation trace with real per-step statistics is a different
question, answered on the shipped GRPO record. **The record contains no reward-hacking transition**,
so no lead time is claimed on it, and `DetectionDelay` refuses with the reason rather than reporting
the distance from an alarm to a changepoint that is not there.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import Uncertainty
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.reading import Refusal, RefusalReason, refuse_incomplete
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.base import Context
from reward_lens.monitor._base import (
    MONITOR_ENVELOPE,
    NO_ACCESS,
    RECORD_ACCESS,
    Channel,
    MonitorInstrument,
)
from reward_lens.monitor.arl import (
    SHIPPED_AD_HOC,
    CusumDesign,
    arl_siegmund,
    design_cusum,
    lorden_delay,
    shipped_ad_hoc_arl0,
)
from reward_lens.monitor.ewma import EwmaDesign, design_ewma, ewma_alarm
from reward_lens.monitor.operating_point import ppv_curve
from reward_lens.stats.baselines.series import gradnorm_peak

# ---------------------------------------------------------------------------
# Running the chart
# ---------------------------------------------------------------------------


def standardize(series: Sequence[float], baseline: int | None = None) -> np.ndarray:
    """Centre and scale by a baseline window, so ``k`` and ``h`` are in standard deviations.

    ``baseline`` is the number of leading steps taken to be in control. Using the whole series is
    the default and it is the conservative choice for a retrospective read: it inflates the
    denominator with the very shift the chart is looking for, so the chart under-detects rather than
    over-detects. An online monitor should pass a real baseline window.
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    b = x[:baseline] if baseline else x
    b = b[np.isfinite(b)]
    if b.size == 0:
        return np.zeros_like(x)
    mu = float(np.mean(b))
    sd = float(np.std(b))
    if not np.isfinite(sd) or sd <= 0:
        alt = np.std(x[np.isfinite(x)])
        sd = float(alt) if np.isfinite(alt) and alt > 0 else 1.0
    return (x - mu) / sd


@dataclass(frozen=True)
class CusumRun:
    """The chart's path and its first alarm.

    ``upper`` and ``lower`` are the two accumulators. ``alarm_at`` is an index into the series, not
    a step number: a caller holding a window has to map it back, and returning a step number here
    would hide whether the mapping happened.
    """

    upper: np.ndarray
    lower: np.ndarray
    alarm_at: int | None
    peak: float
    design: CusumDesign

    @property
    def fired(self) -> bool:
        return self.alarm_at is not None


def run_cusum(
    series: Sequence[float],
    design: CusumDesign,
    *,
    baseline: int | None = None,
    standardized: bool = False,
) -> CusumRun:
    """Page's two-sided CUSUM with a derived threshold. The chart, run.

    Two-sided because that is what the design in `arl.py` solved for and what
    `stats.changepoint.cusum` already runs. A one-sided design at the same target ``ARL_0`` needs a
    different ``h`` (4.09 rather than 4.77 at ``k = 0.5``, ``ARL_0 = 370``), so mixing the two is
    not a detail: it is the difference between a chart that alarms every 370 steps and one that
    alarms every 740.
    """
    z = np.asarray(series, dtype=np.float64) if standardized else standardize(series, baseline)
    up = np.zeros(z.size)
    lo = np.zeros(z.size)
    hi_acc = 0.0
    lo_acc = 0.0
    alarm: int | None = None
    peak = 0.0
    for i, zi in enumerate(z):
        v = float(zi) if math.isfinite(zi) else 0.0
        hi_acc = max(0.0, hi_acc + v - design.k)
        lo_acc = max(0.0, lo_acc - v - design.k)
        up[i] = hi_acc
        lo[i] = lo_acc
        peak = max(peak, hi_acc, lo_acc)
        if alarm is None and (hi_acc > design.h or lo_acc > design.h):
            alarm = i
    return CusumRun(upper=up, lower=lo, alarm_at=alarm, peak=float(peak), design=design)


# ---------------------------------------------------------------------------
# The unit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionWindow:
    """The denominator of `monitor.detection_delay`, with where it came from.

    ``width_steps`` is the 10-to-90-percent rise time of the fitted transition and
    ``midpoint_step`` is its 50% point. ``onset_step`` and ``completion_step`` are the 10% and 90%
    points, derived from those two, and they are properties rather than fields so the three numbers
    cannot disagree.

    **Delay is measured from ``onset_step``, the beginning of the transition.** That convention
    makes a delay positive and a lead negative. `lead_to_midpoint` is the other number people
    quote, the distance from the alarm back to the 50% point, and it is reported beside the delay
    rather than instead of it because papers in this area use both and call both "lead time".

    ``source`` is ``"H4"`` when instrument H4 produced the fit, ``"local"`` when the stand-in in
    this module did, and ``"planted"`` when the transition was simulated and the width is known by
    construction. A delay divided by a width nobody can reproduce is worse than a delay in steps, so
    ``source`` and ``r_squared`` travel on every reading that quotes a fraction.
    """

    width_steps: float
    midpoint_step: float
    source: Literal["H4", "local", "planted"]
    r_squared: float = float("nan")
    detail: str = ""

    @property
    def onset_step(self) -> float:
        """The 10% point: where the transition begins, and what a delay is measured from."""
        return self.midpoint_step - 0.5 * self.width_steps

    @property
    def completion_step(self) -> float:
        """The 90% point: where the transition is essentially over."""
        return self.midpoint_step + 0.5 * self.width_steps

    def fraction(self, delay_steps: float) -> float:
        if not (self.width_steps > 0 and math.isfinite(self.width_steps)):
            return float("nan")
        return float(delay_steps / self.width_steps)

    def delay_from_onset(self, alarm_index: float) -> float:
        return float(alarm_index) - self.onset_step

    def lead_to_midpoint(self, alarm_index: float) -> float:
        return self.midpoint_step - float(alarm_index)

    @classmethod
    def from_fit(cls, fit: Any) -> "TransitionWindow":
        """Build one from instrument H4's `TransitionFit`. The wiring, in one method.

        Read structurally rather than by importing `measure.rate.transition`, for two reasons. This
        package must not fail to import because another package moved, and H4 refuses rather than
        returning an unusable fit, so anything that reaches here already carries a width somebody
        checked. What is read is ``width``, ``midpoint`` and, when present, ``quality.r2``.

        **Two conventions differ between the packages and both are carried on both sides**, so
        nothing is lost and the difference is worth knowing. H4's `lead_time` is positive when the
        alarm fires *before* the transition and measures against the midpoint by default; this
        module's ``delay_windows`` is positive when the alarm fires *after* the 10% point. They are
        the same number with opposite signs and different origins, and each object reports the
        other's form beside its own.
        """
        width = float(getattr(fit, "width"))
        midpoint = float(getattr(fit, "midpoint"))
        quality = getattr(fit, "quality", None)
        r2 = float(getattr(quality, "r2", float("nan"))) if quality is not None else float("nan")
        usable = bool(getattr(fit, "usable", True))
        return cls(
            width_steps=width,
            midpoint_step=midpoint,
            source="H4",
            r_squared=r2,
            detail=(
                "instrument H4, measure.rate.transition"
                + ("" if usable else "; H4 reports this fit as not usable as a denominator")
            ),
        )

    def render(self) -> str:
        r2 = "" if math.isnan(self.r_squared) else f", R^2 {self.r_squared:.3f}"
        return (
            f"transition window: width {self.width_steps:.3g} steps, 10% at "
            f"{self.onset_step:.4g}, 50% at {self.midpoint_step:.4g}, 90% at "
            f"{self.completion_step:.4g} [{self.source}{r2}]"
            + (f"  {self.detail}" if self.detail else "")
        )


#: The 10-to-90 rise time of a logistic in units of its scale parameter: ``log(81) = 4.394``.
LOGISTIC_10_90: float = math.log(81.0)


def local_transition_width(
    series: Sequence[float],
    *,
    min_r_squared: float = 0.5,
) -> TransitionWindow | Refusal:
    """A four-parameter logistic fit, as a stand-in until H4's fit is wired in.

    **This is deliberately the smallest fit that produces the unit, and it is not a second
    implementation of H4.** H4 in `measure/rate/` owns the transition-width measurement, including
    the model comparison, the changepoint alternative and the uncertainty on the width. What is
    needed from it here is exactly three numbers, and `TransitionWindow` is the shape they arrive
    in: a 10-to-90 width in steps, a midpoint, and a fit quality. Pass one built from H4's reading
    with ``source="H4"`` and nothing else in this module changes.

    Refuses rather than returning a width when the logistic does not describe the series, because a
    delay expressed as a fraction of a width that does not exist is the confident wrong number this
    unit was invented to prevent.
    """
    from scipy.optimize import OptimizeWarning, curve_fit

    y = np.asarray(series, dtype=np.float64).ravel()
    ok = np.isfinite(y)
    y = y[ok]
    t = np.arange(y.size, dtype=np.float64)
    if y.size < 8:
        return refuse_incomplete(
            "local_transition_width",
            field="a series long enough to fit a transition",
            subject=f"{y.size} finite points",
            remedy=(
                "Supply at least 8 finite points. A four-parameter logistic has four parameters "
                "and a width fitted on fewer than twice that many points is not a measurement."
            ),
        )
    span = float(np.ptp(y))
    if span == 0.0:
        return Refusal(
            instrument="local_transition_width",
            reason=RefusalReason.BELOW_LOD,
            detail=f"the series is constant at {y[0]:.6g}, so there is no transition to fit.",
            remedy=(
                "Widen the window to a span where the series moves, or read the delay in steps "
                "instead. A fraction of a transition window is undefined when there is no "
                "transition."
            ),
        )

    def logistic(x, lo, hi, mid, scale):
        # Clipped because the optimiser walks through scales near zero on a series with no
        # transition, where the exponent overflows and the fit reports a warning instead of the
        # refusal it is on its way to producing.
        return lo + (hi - lo) / (1.0 + np.exp(np.clip(-(x - mid) / scale, -700.0, 700.0)))

    p0 = [
        float(y[: max(1, y.size // 4)].mean()),
        float(y[-max(1, y.size // 4) :].mean()),
        float(y.size / 2),
        max(1.0, y.size / 10.0),
    ]
    try:
        with warnings.catch_warnings():
            # On a series with no transition the optimiser cannot estimate a covariance, which it
            # warns about. That case is on its way to a refusal below and the warning adds nothing;
            # the covariance is not read here in any case.
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(logistic, t, y, p0=p0, maxfev=20000)
    except (RuntimeError, ValueError) as exc:
        return Refusal(
            instrument="local_transition_width",
            reason=RefusalReason.ENVELOPE_VIOLATED,
            detail=f"the logistic fit did not converge on this series: {exc}",
            remedy=(
                "Restrict the window to the span containing the transition, or supply H4's fitted "
                "width directly as a `TransitionWindow` with source='H4'. Reporting a delay in "
                "steps is the honest fallback when no width can be fitted."
            ),
        )
    resid = y - logistic(t, *popt)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if not (r2 >= min_r_squared):
        return Refusal(
            instrument="local_transition_width",
            reason=RefusalReason.ENVELOPE_VIOLATED,
            detail=(
                f"the logistic fit explains R^2 = {r2:.3f} of the variance, below the "
                f"{min_r_squared:.2f} floor. This series is not a transition of the shape the unit "
                f"assumes."
            ),
            remedy=(
                "Report the delay in steps and say so, or restrict the window to a span that does "
                "contain a transition. Do not divide by a width fitted to noise: the resulting "
                "fraction would look like a measurement and would be an artefact of the fit."
            ),
            statistics={"r_squared": r2, "n": int(y.size)},
        )
    width = LOGISTIC_10_90 * abs(float(popt[3]))
    return TransitionWindow(
        width_steps=width,
        midpoint_step=float(popt[2]),
        source="local",
        r_squared=r2,
        detail="four-parameter logistic, least squares, 10-to-90 rise time",
    )


# ---------------------------------------------------------------------------
# The detector bank
# ---------------------------------------------------------------------------

#: One detector: a name, a callable taking a standardized series and returning the first alarm
#: index or None, and a one-line description of where its threshold came from.
Detector = Callable[[np.ndarray], "int | None"]


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    fn: Detector
    provenance: str
    online: bool = True


def _cusum_detector(design: CusumDesign) -> Detector:
    def fn(z: np.ndarray) -> int | None:
        return run_cusum(z, design, standardized=True).alarm_at

    return fn


def _fixed_threshold_detector(limit: float) -> Detector:
    def fn(z: np.ndarray) -> int | None:
        hit = np.where(np.abs(z) > limit)[0]
        return int(hit[0]) if hit.size else None

    return fn


def _ewma_detector(design: EwmaDesign) -> Detector:
    def fn(z: np.ndarray) -> int | None:
        return ewma_alarm(z, design)

    return fn


def _anytime_llr_detector(shift: float, alpha: float) -> Detector:
    """The non-resetting likelihood-ratio martingale, alarming at ``1/alpha``.

    ``M_t = exp(shift * S_t - t shift^2 / 2)`` is a nonnegative martingale under the null, so Ville
    bounds the probability that it *ever* alarms at ``alpha`` over an unbounded horizon. That is a
    different guarantee from an ARL and the difference is the point: this alarm may never fire, and
    its false-alarm probability is a lifetime total rather than a rate per step. A CUSUM with
    ``ARL_0 = 370`` fires eventually with probability one. To compare them the level is set to
    ``horizon / ARL_0``, so both spend the same false-alarm budget over the window being watched.
    """
    log_thresh = math.log(1.0 / alpha)

    def fn(z: np.ndarray) -> int | None:
        s = np.cumsum(np.nan_to_num(z, nan=0.0))
        t = np.arange(1, z.size + 1, dtype=np.float64)
        log_m = shift * s - 0.5 * t * shift * shift
        hit = np.where(log_m >= log_thresh)[0]
        return int(hit[0]) if hit.size else None

    return fn


def _gradnorm_peak_detector(z: np.ndarray) -> int | None:
    """The baseline. Retrospective: it reads the whole series and returns its smoothed argmax.

    It is not an online detector and it is in the bank anyway, because it is the thing a monitor has
    to beat: if a designed alarm fires no earlier than the point where the logged scalar peaked, the
    alarm contributed nothing an existing dashboard did not already show. Its "false-alarm rate" in
    the table below is the fraction of runs whose peak falls before the change, which is a different
    quantity from an online chart's and is labelled as such.
    """
    return gradnorm_peak(np.abs(np.asarray(z, dtype=np.float64))).index


def default_bank(
    *,
    shift: float = 1.0,
    horizon: int = 200,
    arl0: float = 370.0,
    ewma_lam: float = 0.2,
) -> tuple[DetectorSpec, ...]:
    """The comparison that decides the design: designed alarms against fixed thresholds.

    ``horizon`` sets the level of the anytime-valid alarm so that it spends the same false-alarm
    budget over the watched window as an ``ARL_0``-designed chart does. Without that conversion the
    two are not comparable at all, because one quotes a lifetime probability and the other a rate.
    """
    d370 = design_cusum(shift, arl0)
    d1000 = design_cusum(shift, 1000.0)
    ad_hoc = CusumDesign(
        shift=shift,
        k=SHIPPED_AD_HOC["k_sds"],
        h=SHIPPED_AD_HOC["h_sds"],
        arl0_target=float("nan"),
        arl0_siegmund=shipped_ad_hoc_arl0(),
        sides=2,
        lorden_delay=lorden_delay(shipped_ad_hoc_arl0(), shift),
        arl1_siegmund=arl_siegmund(SHIPPED_AD_HOC["h_sds"], SHIPPED_AD_HOC["k_sds"], shift, 2),
    )
    ewma = design_ewma(arl0, ewma_lam)
    return (
        DetectorSpec(
            f"cusum.arl{int(arl0)}",
            _cusum_detector(d370),
            f"designed: k = {d370.k:.3g}, h = {d370.h:.4g} solves ARL(0) = {arl0:.0f}",
        ),
        DetectorSpec(
            "cusum.arl1000",
            _cusum_detector(d1000),
            f"designed: k = {d1000.k:.3g}, h = {d1000.h:.4g} solves ARL(0) = 1000",
        ),
        DetectorSpec(
            "cusum.ad_hoc",
            _cusum_detector(ad_hoc),
            f"the shipped recorder's k = {ad_hoc.k}, h = {ad_hoc.h}, no derivation; "
            f"implies ARL(0) = {ad_hoc.arl0_siegmund:.0f}",
        ),
        DetectorSpec(
            "ewma.designed",
            _ewma_detector(ewma),
            f"designed: lam = {ewma.lam}, L = {ewma.multiplier:.4g} solves ARL(0) = {arl0:.0f}",
        ),
        DetectorSpec(
            "fixed.3sigma",
            _fixed_threshold_detector(3.0),
            "fixed threshold at 3 standard deviations, the Shewhart convention, no derivation",
        ),
        DetectorSpec(
            "fixed.2sigma",
            _fixed_threshold_detector(2.0),
            "fixed threshold at 2 standard deviations, no derivation",
        ),
        DetectorSpec(
            "anytime.llr",
            _anytime_llr_detector(shift, min(0.99, horizon / arl0)),
            f"Ville at level {min(0.99, horizon / arl0):.4g}, matched to spend the same "
            f"false-alarm budget over {horizon} steps as ARL(0) = {arl0:.0f}",
        ),
        DetectorSpec(
            "baseline.gradnorm_peak",
            _gradnorm_peak_detector,
            "retrospective argmax of the smoothed series; not an online detector",
            online=False,
        ),
    )


# ---------------------------------------------------------------------------
# Measuring the bank against a planted changepoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectorPerformance:
    """One detector's realised delay and false-alarm rate against a planted transition."""

    name: str
    provenance: str
    online: bool
    false_alarm_rate: float
    detection_rate: float
    median_delay_steps: float
    mean_delay_steps: float
    median_delay_windows: float
    mean_delay_windows: float
    n_runs: int

    def render(self) -> str:
        return (
            f"{self.name:<24} FAR {self.false_alarm_rate:6.2%}  detected {self.detection_rate:6.1%}"
            f"  delay {self.median_delay_steps:6.1f} steps = "
            f"{self.median_delay_windows:5.2f} windows   [{self.provenance}]"
        )


@dataclass(frozen=True)
class BankReport:
    """The whole comparison, with the planted transition it was measured against."""

    window: TransitionWindow
    shift: float
    n_pre: int
    n_post: int
    n_runs: int
    rows: tuple[DetectorPerformance, ...] = ()
    lorden_bound_steps: float = float("nan")

    def by_name(self) -> dict[str, DetectorPerformance]:
        return {r.name: r for r in self.rows}

    def render(self) -> str:
        head = [
            f"detector bank against a planted {self.shift:.2g}-sigma logistic transition of width "
            f"{self.window.width_steps:.3g} steps, {self.n_pre} pre-change and {self.n_post} "
            f"post-change steps, {self.n_runs} runs.",
            f"    Lorden bound on worst-case delay at the designed chart: "
            f"{self.lorden_bound_steps:.1f} steps "
            f"= {self.window.fraction(self.lorden_bound_steps):.2f} windows.",
            "    false-alarm rate is the fraction of runs alarming before the 10% point of the "
            "transition; delay is measured from that same point.",
        ]
        return "\n".join(head + ["    " + r.render() for r in self.rows])


def _logistic_ramp(n_pre: int, n_post: int, width: float, shift: float) -> np.ndarray:
    """The mean path: flat at zero, then a logistic rise of stated 10-to-90 width to ``shift``."""
    t = np.arange(n_pre + n_post, dtype=np.float64)
    scale = width / LOGISTIC_10_90
    return shift / (1.0 + np.exp(-(t - n_pre) / scale))


def measure_bank(
    *,
    shift: float = 1.0,
    width: float = 20.0,
    n_pre: int = 100,
    n_post: int = 150,
    n_runs: int = 2000,
    arl0: float = 370.0,
    seed: int = 0,
    bank: Sequence[DetectorSpec] | None = None,
) -> BankReport:
    """Plant a transition of known width and measure every detector's delay and false-alarm rate.

    This is how an ARL-designed alarm is validated, because an average run length and a detection
    delay are properties of the *procedure* and no single series can establish either. The
    transition is a logistic ramp rather than a step, so the transition window has a width by
    construction and the delay divides by a denominator nobody had to fit.

    The false-alarm rate is the fraction of runs whose first alarm lands **before** the transition
    begins, over ``n_pre`` steps. The steady-state prediction for a chart designed at
    ``ARL_0 = 370`` watched for 100 steps is ``1 - exp(-100/370) = 24%``; the measured rate at the
    defaults is 18%, and it is below the prediction for two reasons that are both real. The chart
    starts from a cold accumulator at zero, so its first crossing takes longer than the steady-state
    rate implies, and the series is standardized against the same pre-change window the false alarms
    are counted in, which shrinks the apparent deviations there. Both are what an online monitor
    actually does, so 18% is the honest number for this design in this use and 24% is the number for
    a chart that has been running forever.

    The series is standardized against its own first ``n_pre`` steps, which is what an online
    monitor can actually do. Standardizing against the whole series would leak the post-change data
    into the denominator and flatter every detector equally.
    """
    specs = tuple(bank) if bank is not None else default_bank(shift=shift, horizon=n_pre, arl0=arl0)
    rng = np.random.default_rng(seed)
    mean_path = _logistic_ramp(n_pre, n_post, width, shift)
    n = n_pre + n_post
    noise = rng.standard_normal((n_runs, n))
    series = noise + mean_path[None, :]
    # Standardize against the pre-change window, per run, as an online monitor would.
    mu = series[:, :n_pre].mean(axis=1, keepdims=True)
    sd = series[:, :n_pre].std(axis=1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    z = (series - mu) / sd

    window = TransitionWindow(
        width_steps=float(width),
        midpoint_step=float(n_pre),
        source="planted",
        r_squared=1.0,
        detail="logistic ramp planted by `measure_bank`; the width is known, not fitted",
    )
    rows: list[DetectorPerformance] = []
    for spec in specs:
        raw = [spec.fn(z[i]) for i in range(n_runs)]
        alarms = np.array([-1 if a is None else a for a in raw], dtype=np.int64)
        fired = alarms >= 0
        onset = window.onset_step
        early = fired & (alarms < onset)
        detected = fired & (alarms >= onset)
        delays = alarms[detected].astype(np.float64) - onset
        rows.append(
            DetectorPerformance(
                name=spec.name,
                provenance=spec.provenance,
                online=spec.online,
                false_alarm_rate=float(np.mean(early)),
                detection_rate=float(np.mean(detected)),
                median_delay_steps=float(np.median(delays)) if delays.size else float("nan"),
                mean_delay_steps=float(np.mean(delays)) if delays.size else float("nan"),
                median_delay_windows=window.fraction(float(np.median(delays)))
                if delays.size
                else float("nan"),
                mean_delay_windows=window.fraction(float(np.mean(delays)))
                if delays.size
                else float("nan"),
                n_runs=n_runs,
            )
        )
    return BankReport(
        window=window,
        shift=shift,
        n_pre=n_pre,
        n_post=n_post,
        n_runs=n_runs,
        rows=tuple(rows),
        lorden_bound_steps=lorden_delay(arl0, shift),
    )


# ---------------------------------------------------------------------------
# The two J2 instruments
# ---------------------------------------------------------------------------

J2_DESIGN_BASELINES: tuple[str, ...] = (
    "baseline.shipped_ad_hoc_cusum",
    "baseline.fixed_threshold_3sigma",
)
J2_DELAY_BASELINES: tuple[str, ...] = (
    "baseline.shipped_ad_hoc_cusum",
    "baseline.fixed_threshold_3sigma",
    "baseline.gradnorm_peak",
)

#: The design instrument reads no record: it solves an equation. Its premises are the standardized
#: normal increments the ARL approximation assumes, which is a property of the estimator and not of
#: any run, and it is printed on the reading rather than checked against a regime.
DESIGN_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "a threshold solved from a stated false-alarm interval. It reads no record, so no regime "
        "of any run can make it wrong. Its one premise, that the monitored statistic is "
        "approximately standard normal in control, is a property of the series the chart is later "
        "run on and is checked there rather than here."
    ),
)


class AlarmDesign(MonitorInstrument):
    """J2. The threshold, derived. "Designed for one false alarm per 1000 steps: k = 0.5, h = 5.75."

    The whole content is that a free parameter is removed. Before: two numbers with no derivation,
    which is what this library's own flight recorder ships. After: one number the user states, how
    often they will accept being woken for nothing, and one they already know, how big a move
    matters, with everything else following.

    The reading carries the Lorden bound beside the threshold because that is the trade a reader
    actually has to make, and the shape of it is the useful part: detection delay grows as the
    logarithm of the false-alarm interval, so a tenfold quieter chart costs 4.6 extra steps at a
    one-sigma shift and not tenfold.

    What it cannot do: the ARL is an average over a hypothetical stream of independent standard
    normal observations. A monitored series with autocorrelation has a shorter in-control run length
    than the design says, in the direction of more false alarms, and this instrument cannot see that
    from a design alone. `DetectionDelay` measures it on the series.
    """

    name = "AlarmDesign"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "Page (1954); Siegmund (1985) ARL approximation; Lorden (1971) delay bound"
    deviations = (
        "Siegmund's approximation is asymptotic in the boundary and uses the corrected boundary "
        "b = h + 1.166 for the mean overshoot. It agrees with the integral-equation solve to under "
        "1% and with Monte Carlo to within its own standard error at the design points, and it is "
        "an approximation rather than the ARL.",
    )

    quantity = "monitor.arl0"
    requires = NO_ACCESS
    envelope = DESIGN_ENVELOPE
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = J2_DESIGN_BASELINES
    rung = 0

    def __init__(self, *, shift: float = 1.0, arl0: float = 370.0, sides: int = 2) -> None:
        self.shift = float(shift)
        self.arl0 = float(arl0)
        self.sides = int(sides)

    def compute(self, ctx: Context) -> CusumDesign | Refusal:
        if self.arl0 <= 1.0:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.ENVELOPE_VIOLATED,
                detail=f"an in-control run length of {self.arl0} is shorter than one observation.",
                remedy=(
                    "State an average run length above 1. A useful monitor is designed somewhere "
                    "between 100 and 10,000 steps between false alarms; below that the chart is "
                    "alarming on noise and above it the delay starts to bite."
                ),
            )
        return design_cusum(self.shift, self.arl0, self.sides)  # type: ignore[arg-type]

    def payload(self, computed: CusumDesign) -> dict:
        # Every detector reports a PPV curve (J4). For a design the false-positive rate per step is
        # the reciprocal of the in-control run length, which is what the design just fixed, and
        # sensitivity is set to 1 because a design has not seen any data. So these are upper bounds
        # on what an alarm from this chart can be worth, and that is the useful reading: at one
        # false alarm per 370 steps, an alarm is right less than half the time unless something is
        # going wrong on more than 0.3% of steps.
        far = 1.0 / computed.arl0_siegmund
        curve = ppv_curve(1.0, far)
        return {
            "arl0_target": computed.arl0_target,
            "arl0_achieved": computed.arl0_siegmund,
            "k": computed.k,
            "h": computed.h,
            "shift": computed.shift,
            "sides": computed.sides,
            "lorden_delay_steps": computed.lorden_delay,
            "arl1_at_design_shift": computed.arl1_siegmund,
            "cost_of_ten_times_quieter_steps": math.log(10.0) / (0.5 * computed.shift**2),
            "ppv_curve": {
                "sensitivity_assumed": 1.0,
                "fpr": far,
                "prevalences": list(curve.prevalences),
                "ppv": list(curve.ppv),
                "prevalence_for_half_ppv": curve.prevalence_for_ppv,
                "note": (
                    "per step. The false-positive rate is one over the in-control run length this "
                    "design just fixed, and sensitivity is 1, which is the best case, so these are "
                    "upper bounds on what an alarm from this chart can be worth."
                ),
            },
            "baselines": self.baseline_map(computed),
            "rendered": computed.render(),
        }

    def baseline_map(self, computed: CusumDesign) -> Mapping[str, float]:
        return {
            "baseline.shipped_ad_hoc_cusum": shipped_ad_hoc_arl0(),
            # A 3-sigma fixed threshold on independent standard normals alarms with probability
            # 2*Phi(-3) per step, so its in-control run length is the reciprocal.
            "baseline.fixed_threshold_3sigma": 1.0 / (2.0 * 0.001349898031630094),
        }


class DetectionDelay(MonitorInstrument):
    """J2. Realised delay and false-alarm rate on a series, in fraction-of-transition-window units.

    Two things it will not do, and both are the reason it exists.

    It will not quote a lead time on a run with no transition in it. A delay is the distance from a
    changepoint to an alarm and a run with no changepoint has no such distance, so the instrument
    refuses with the fitted width's own refusal attached rather than reporting the alarm's index and
    letting a reader take it for a lead.

    It will not divide by a width it could not fit. `local_transition_width` refuses below an
    R-squared of 0.5, and this passes that refusal through. A delay in steps is the honest fallback
    and the reading says which unit it is in.
    """

    name = "DetectionDelay"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "Page (1954); the transition-window unit convention for lead time"
    deviations = (
        "the transition width is fitted here by a four-parameter logistic when H4's fit is not "
        "supplied. H4 owns that measurement and this is a stand-in; every reading records which of "
        "the two produced the denominator.",
    )

    quantity = "monitor.detection_delay"
    requires = RECORD_ACCESS
    envelope = MONITOR_ENVELOPE
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = J2_DELAY_BASELINES
    rung = 0

    def __init__(
        self,
        channel: Channel,
        *,
        shift: float = 1.0,
        arl0: float = 370.0,
        baseline_steps: int | None = None,
        window: TransitionWindow | None = None,
        gradnorm: Channel | None = None,
    ) -> None:
        self.channel = channel
        self.shift = float(shift)
        self.arl0 = float(arl0)
        self.baseline_steps = baseline_steps
        self.window = window
        self.gradnorm = gradnorm

    def compute(self, ctx: Context) -> dict | Refusal:
        design = design_cusum(self.shift, self.arl0)
        oriented = self.channel.oriented
        run = run_cusum(oriented, design, baseline=self.baseline_steps)
        ad_hoc = CusumDesign(
            shift=self.shift,
            k=SHIPPED_AD_HOC["k_sds"],
            h=SHIPPED_AD_HOC["h_sds"],
            arl0_target=float("nan"),
            arl0_siegmund=shipped_ad_hoc_arl0(),
            sides=2,
            lorden_delay=lorden_delay(shipped_ad_hoc_arl0(), self.shift),
            arl1_siegmund=arl_siegmund(
                SHIPPED_AD_HOC["h_sds"], SHIPPED_AD_HOC["k_sds"], self.shift, 2
            ),
        )
        ad_hoc_run = run_cusum(oriented, ad_hoc, baseline=self.baseline_steps)
        z = standardize(oriented, self.baseline_steps)
        fixed3 = _fixed_threshold_detector(3.0)(z)
        peak = gradnorm_peak(self.gradnorm.values).index if self.gradnorm is not None else None

        window = self.window
        if window is None:
            fitted = local_transition_width(self.channel.values)
            if isinstance(fitted, Refusal):
                return Refusal(
                    instrument=self.name,
                    reason=fitted.reason,
                    detail=(
                        f"channel `{self.channel.name}` over {self.channel.n} steps: "
                        f"{fitted.detail} A detection delay is the distance from a transition to "
                        f"an alarm, and this series carries no transition to measure from. The "
                        f"designed chart "
                        + (f"fired at index {run.alarm_at}" if run.fired else "did not fire at all")
                        + ", which is an alarm index and not a lead time."
                    ),
                    remedy=(
                        "Supply a `TransitionWindow` from instrument H4 if you have a fitted "
                        "width for this run, or read `alarm_index` and the false-alarm rate from "
                        "the design and treat them as what they are. On a run with no transition "
                        "there is no lead time to report and quoting the alarm index as one is "
                        "how a monitoring claim gets made about a run that never hacked."
                    ),
                    statistics={
                        "alarm_index": run.alarm_at,
                        "peak_statistic": run.peak,
                        "channel": self.channel.name,
                        "n_steps": self.channel.n,
                    },
                )
            window = fitted

        if run.alarm_at is None:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.BELOW_LOD,
                detail=(
                    f"the chart designed for ARL_0 = {self.arl0:.0f} (k = {design.k:.3g}, "
                    f"h = {design.h:.4g}) never crossed on channel `{self.channel.name}`; the peak "
                    f"statistic reached {run.peak:.3g} of a threshold of {design.h:.4g}."
                ),
                remedy=(
                    f"Design for a shorter in-control run length if a false alarm is cheap for "
                    f"you: at ARL_0 = 100 the threshold falls to "
                    f"{design_cusum(self.shift, 100.0).h:.3g}. Or accept the null: a chart that "
                    f"did not fire is evidence of no shift of the size it was designed for, and "
                    f"that is a result rather than a failure."
                ),
                statistics={
                    "peak_statistic": run.peak,
                    "threshold": design.h,
                    "transition_width_steps": window.width_steps,
                },
            )

        delay_steps = window.delay_from_onset(run.alarm_at)
        return {
            "design": design,
            "window": window,
            "alarm_index": run.alarm_at,
            "delay_steps": delay_steps,
            "delay_windows": window.fraction(delay_steps),
            "lead_to_midpoint_steps": window.lead_to_midpoint(run.alarm_at),
            "lead_to_midpoint_windows": window.fraction(window.lead_to_midpoint(run.alarm_at)),
            "ad_hoc_alarm_index": ad_hoc_run.alarm_at,
            "fixed3_alarm_index": fixed3,
            "gradnorm_peak_index": peak,
            "peak_statistic": run.peak,
        }

    def payload(self, computed: dict) -> dict:
        design: CusumDesign = computed["design"]
        window: TransitionWindow = computed["window"]
        body = {
            "delay_windows": computed["delay_windows"],
            "delay_steps": computed["delay_steps"],
            "lead_to_midpoint_windows": computed["lead_to_midpoint_windows"],
            "lead_to_midpoint_steps": computed["lead_to_midpoint_steps"],
            "alarm_index": computed["alarm_index"],
            "transition_width_steps": window.width_steps,
            "transition_onset_step": window.onset_step,
            "transition_midpoint_step": window.midpoint_step,
            "transition_source": window.source,
            "transition_r_squared": window.r_squared,
            "k": design.k,
            "h": design.h,
            "arl0_target": design.arl0_target,
            "arl0_achieved": design.arl0_siegmund,
            "lorden_delay_steps": design.lorden_delay,
            "lorden_delay_windows": window.fraction(design.lorden_delay),
            "channel": self.channel.name,
            "n_steps": self.channel.n,
            "baselines": self.baseline_map(computed),
            "ppv_curve": None,
        }
        # Every detector reports a PPV curve (J4). The false-alarm rate over the watched window is
        # what a designed ARL buys, and sensitivity is not measurable without labels, so the curve
        # is drawn at perfect sensitivity and says so.
        far = min(0.999, self.channel.n / design.arl0_siegmund)
        curve = ppv_curve(1.0, far)
        body["ppv_curve"] = {
            "sensitivity_assumed": 1.0,
            "fpr": far,
            "prevalences": list(curve.prevalences),
            "ppv": list(curve.ppv),
            "note": (
                "false-positive rate is the designed chart's expected false alarms over this "
                "window; sensitivity is set to 1, which is the best case, so the realised PPV "
                "cannot exceed these."
            ),
        }
        return body

    def baseline_map(self, computed: dict) -> Mapping[str, float]:
        out: dict[str, float] = {}
        window: TransitionWindow = computed["window"]
        for key, idx in (
            ("baseline.shipped_ad_hoc_cusum", computed["ad_hoc_alarm_index"]),
            ("baseline.fixed_threshold_3sigma", computed["fixed3_alarm_index"]),
            ("baseline.gradnorm_peak", computed["gradnorm_peak_index"]),
        ):
            out[key] = (
                float("nan") if idx is None else window.fraction(window.delay_from_onset(idx))
            )
        return out

    def uncertainty(self, computed: dict) -> Uncertainty:
        return Uncertainty(
            n=self.channel.n,
            method="single realisation; the delay distribution is a property of the procedure and "
            "is measured by `measure_bank`",
        )


__all__ = [
    "DESIGN_ENVELOPE",
    "J2_DELAY_BASELINES",
    "J2_DESIGN_BASELINES",
    "LOGISTIC_10_90",
    "AlarmDesign",
    "BankReport",
    "CusumRun",
    "DetectionDelay",
    "Detector",
    "DetectorPerformance",
    "DetectorSpec",
    "TransitionWindow",
    "default_bank",
    "local_transition_width",
    "measure_bank",
    "run_cusum",
    "standardize",
]

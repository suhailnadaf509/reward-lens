"""The registered prediction on `Λ`, frozen before the labelled series was looked at.

The question is fixed in advance and the freeze is what makes that checkable: **does `Λ` move
before the labelled hack rate does, and by how much, as a fraction of the transition window?** The
comparator it has to beat is the gradient-norm peak, which is free and which every trainer already
logs, so a lead time that does not beat it is a lead time nobody needs a new instrument for.

`studies.freeze.freeze` hashes the spec and records the git sha, and the resulting `StudyID` stamps
every Evidence produced under it as REGISTERED. Two specs differing in any registered field get
different ids, which is what makes editing a prediction after seeing the data visible as a new
version rather than invisible. The spec below is written out in full and its hash is asserted in the
acceptance test, so a later edit fails a test rather than passing quietly.

**The window fit is local and it is not the canonical one.** Expressing a lead time as a fraction of
a transition width needs the transition fitted, and the canonical estimator for that is not in the
library yet. `transition_window` here is a four-parameter logistic least squares on the rate series,
validated in this package's own tests against a planted transition. When the canonical estimator
lands, this should be deleted and the prediction rescored against it; the frozen metric names the
quantity rather than the implementation, so that substitution does not change the prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from reward_lens.stats.baselines.series import gradnorm_peak, smooth
from reward_lens.stats.changepoint import ChangePoint, cusum
from reward_lens.studies.freeze import FrozenStudy, freeze
from reward_lens.studies.spec import Hypothesis, KillCriterion, Prediction, StudySpec, SubjectQuery


@dataclass(frozen=True)
class TransitionWindow:
    """A fitted rise in a rate series: where it happened and how long it took.

    ``t50`` is the midpoint of the fitted logistic and ``width`` is its 10-to-90 rise, which is the
    unit a lead time is reported in. ``amplitude`` is the total rise; a transition whose amplitude is
    within the noise is not a transition, and ``fitted`` is False in that case rather than the fit
    reporting a midpoint of a rise that did not happen.
    """

    t50: float
    width: float
    low: float
    high: float
    amplitude: float
    residual_rms: float
    fitted: bool
    detail: str = ""

    def fraction(self, index: float | int | None) -> float:
        """How far ahead of the midpoint ``index`` is, in transition widths. NaN when undefined."""
        if index is None or not self.fitted or not np.isfinite(self.width) or self.width <= 0:
            return float("nan")
        return float((self.t50 - float(index)) / self.width)


def _logistic(t: np.ndarray, low: float, span: float, t50: float, scale: float) -> np.ndarray:
    """The four-parameter logistic, through `expit` so a wide step does not overflow the exponential.

    `curve_fit` walks the parameter space and reaches scales small enough that `exp(-(t-t50)/s)`
    overflows a float64 on the way. Overflowing to infinity is arithmetically harmless here (the
    ratio goes to zero) and it emits a warning per evaluation, which buries a real numerical problem
    under thousands of cosmetic ones. `expit` is the same function computed without the overflow.
    """
    from scipy.special import expit

    return np.asarray(low + span * expit((t - t50) / scale), dtype=np.float64)


def transition_window(
    series: Sequence[float] | np.ndarray,
    steps: Sequence[int] | np.ndarray | None = None,
    *,
    min_amplitude_sigmas: float = 2.0,
) -> TransitionWindow:
    """Fit a four-parameter logistic to a rate series and report its midpoint and 10-90 width.

    The 10-to-90 width of a logistic with scale `s` is `s·ln(81)`, which is where the 4.394 comes
    from; it is the standard rise-time convention and it is stated because a width quoted without
    its percentiles is not a width.

    The fit is declined, rather than reported with a wide interval, when the amplitude does not clear
    ``min_amplitude_sigmas`` times the residual scatter. A logistic will fit a midpoint to pure noise
    and the midpoint will land wherever the noise happened to be, so a lead time measured against it
    would be a lead time against an arbitrary index.
    """
    y = np.asarray(series, dtype=np.float64).ravel()
    t = (
        np.asarray(steps, dtype=np.float64).ravel()
        if steps is not None
        else np.arange(y.size, dtype=np.float64)
    )
    finite = np.isfinite(y) & np.isfinite(t)
    y, t = y[finite], t[finite]
    if y.size < 8:
        return TransitionWindow(
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            False,
            f"{y.size} finite points; a four-parameter fit needs at least eight",
        )

    from scipy.optimize import curve_fit

    span0 = float(y[-max(3, y.size // 5) :].mean() - y[: max(3, y.size // 5)].mean())
    guess = [
        float(y[: max(3, y.size // 5)].mean()),
        span0 or 1e-6,
        float(np.median(t)),
        float(max((t[-1] - t[0]) / 10.0, 1e-6)),
    ]
    try:
        # A series with no transition drives the fit to a degenerate covariance, which scipy warns
        # about. That is the answer rather than a problem: the amplitude test below declines the
        # fit, and the warning would otherwise fire on every well-behaved flat series.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(_logistic, t, y, p0=guess, maxfev=20000)
    except Exception as exc:  # the fit failing is an answer about the series, not a crash
        return TransitionWindow(
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            False,
            f"the logistic fit did not converge: {type(exc).__name__}",
        )
    low, span, t50, scale = (float(v) for v in popt)
    residual = float(np.sqrt(np.mean((y - _logistic(t, low, span, t50, scale)) ** 2)))
    width = abs(scale) * float(np.log(81.0))
    amplitude = abs(span)
    ok = (
        np.isfinite(t50)
        and np.isfinite(width)
        and width > 0
        and residual > 0
        and amplitude >= min_amplitude_sigmas * residual
        and t[0] <= t50 <= t[-1]
    )
    return TransitionWindow(
        t50=t50,
        width=width,
        low=low,
        high=low + span,
        amplitude=amplitude,
        residual_rms=residual,
        fitted=bool(ok),
        detail=(
            f"logistic midpoint {t50:.4g}, 10-90 width {width:.4g}, amplitude {amplitude:.4g} "
            f"against residual rms {residual:.4g}"
            + (
                ""
                if ok
                else "; declined, the rise does not clear the scatter or sits outside the "
                "observed range"
            )
        ),
    )


def onset_of(
    series: Sequence[float] | np.ndarray,
    steps: Sequence[int] | np.ndarray | None = None,
    *,
    threshold: float = 5.0,
    drift: float = 0.5,
    baseline: int | None = None,
    smooth_window: int = 1,
) -> tuple[ChangePoint, float]:
    """A two-sided CUSUM onset on a series, returned as ``(ChangePoint, step index)``.

    Two-sided on purpose. Whether `Λ` rises or falls as a run approaches a reward-hacking transition
    is exactly the thing nobody has measured, so a one-sided detector would encode the answer in the
    detector. The returned step index is in the series' own step units rather than in array
    positions, because the two differ whenever the series is sampled every `n` steps.

    The CUSUM defaults are `stats.changepoint`'s own, and they imply one false alarm every 469 steps
    under Siegmund's approximation. That is not a rate anybody chose, and the fix is to solve
    `ARL(0) = ARL_0` for the threshold instead. The defaults are used here so that the comparator and
    the claim run under identical settings; when that fix lands, both sides move together and the
    comparison is unaffected.
    """
    y = np.asarray(series, dtype=np.float64).ravel()
    if smooth_window > 1:
        y = smooth(y, smooth_window)
    point = cusum(y, threshold=threshold, drift=drift, baseline=baseline)
    if point.index is None:
        return point, float("nan")
    if steps is None:
        return point, float(point.index)
    axis = np.asarray(steps, dtype=np.float64).ravel()
    return point, float(axis[min(point.index, axis.size - 1)])


def gradnorm_onset(
    series: Sequence[float] | np.ndarray,
    steps: Sequence[int] | np.ndarray | None = None,
) -> dict[str, float]:
    """The free comparator, run two ways, because running it one way rigs the comparison.

    The catalogue names "the gradient-norm peak". A peak is a late statistic: it fires where the
    change is largest rather than where it began, so scoring a CUSUM on `Λ` against a peak on the
    gradient norm compares an onset detector against a magnitude detector and the onset detector wins
    by construction. The same objection applies to I5's variance-level baseline and the fix is the
    same: run the comparator as a CUSUM as well and take **whichever is earlier**.
    """
    y = np.asarray(series, dtype=np.float64).ravel()
    axis = (
        np.asarray(steps, dtype=np.float64).ravel()
        if steps is not None
        else np.arange(y.size, dtype=np.float64)
    )
    peak = gradnorm_peak(y)
    peak_step = (
        float(axis[min(peak.index, axis.size - 1)]) if peak.index is not None else float("nan")
    )
    _, cusum_step = onset_of(y, axis)
    candidates = [s for s in (peak_step, cusum_step) if np.isfinite(s)]
    return {
        "peak_step": peak_step,
        "peak_strength": float(peak.strength),
        "cusum_step": cusum_step,
        "earliest": float(min(candidates)) if candidates else float("nan"),
    }


# ---------------------------------------------------------------------------
# The frozen spec
# ---------------------------------------------------------------------------

#: The three metrics the analysis produces, named here so the prediction and the kill criterion
#: reference the same strings the scoring function writes.
METRIC_LAMBDA_LEAD = "lambda_lead_fraction"
METRIC_BASELINE_LEAD = "gradnorm_lead_fraction"
METRIC_MARGIN = "lambda_minus_gradnorm_lead_fraction"
METRIC_WIDTH = "hack_rate_transition_width_steps"

LAMBDA_LEAD_TIME_SPEC = StudySpec(
    id="f2-lambda-lead-time",
    title="Does the selection-explained fraction move before the labelled hack rate?",
    science="S01-selection",
    hypotheses=(
        Hypothesis(
            id="H1",
            statement=(
                "On a labelled reinforcement-learning run that undergoes a reward-hacking "
                "transition, the CUSUM onset of the selection-explained fraction Lambda precedes "
                "the midpoint of the fitted transition in the labelled hack rate. Reported as a "
                "fraction of the 10-to-90 transition width, the lead is positive."
            ),
            prediction=Prediction(
                metric=METRIC_LAMBDA_LEAD,
                comparator=">",
                threshold=0.0,
                rationale=(
                    "Lambda measures whether the first-order selection term still explains what "
                    "moved. A policy entering a reward-hacking transition is by hypothesis moving "
                    "along a direction the current step's selection pressure does not account for, "
                    "so Lambda should fall while the labelled outcome rate has not yet risen. The "
                    "direction of the effect is registered as a lead in time and deliberately not "
                    "as a sign on Lambda, because whether Lambda falls or rises through a "
                    "transition has not been measured and encoding a guess about it in the "
                    "detector would make the detector the hypothesis."
                ),
            ),
            scoreboard_row="F2",
        ),
        Hypothesis(
            id="H2",
            statement=(
                "That lead beats the free comparator. The gradient-norm onset, taken as the earlier "
                "of its peak and its own CUSUM, does not lead the labelled transition by as much."
            ),
            prediction=Prediction(
                metric=METRIC_MARGIN,
                comparator=">",
                threshold=0.0,
                rationale=(
                    "Every trainer already logs the gradient norm, so an instrument that does not "
                    "beat it has told nobody anything they could not read off the dashboard they "
                    "already have. The comparator is run as a CUSUM as well as a peak and the "
                    "earlier of the two is taken, because scoring an onset detector against a "
                    "magnitude detector would hand this hypothesis a win it did not earn."
                ),
            ),
            scoreboard_row="F2",
        ),
    ),
    analysis="reward_lens.measure.ledger.prediction.score_lead_time",
    subjects=SubjectQuery(
        datasets=(
            "ai-safety-institute/reward-hacking-olmo3.1-32b-kl0.0-seed2-rollouts",
            "ai-safety-institute/reward-hacking-olmo3.1-32b-kl0.02-seed2-rollouts",
        ),
        extra={
            "label_column": "reward_hacked",
            "reward_column": "training_passed",
            "group_column": "problem_id",
            "step_column": "rollout_index",
            "lambda_context": 5,
        },
    ),
    kill_criteria=(
        KillCriterion(
            id="K1",
            metric=METRIC_WIDTH,
            comparator="<",
            threshold=1.0,
            description=(
                "The labelled hack rate has no fitted transition on this series, so there is no "
                "window to report a lead time as a fraction of and the question is void rather "
                "than answered. A width below one step is the fit having collapsed, not a "
                "transition that happened instantly."
            ),
        ),
    ),
    version=1,
    notes=(
        "Frozen before the labelled series was read. The transition window is fitted by the local "
        "four-parameter logistic in this module; rescore against a fitted estimator once one lands. "
        "The metric names a quantity and not an implementation, so that substitution does not "
        "change what was predicted."
    ),
)


def freeze_prediction(repo_dir: str | None = None, frozen_at: str | None = None) -> FrozenStudy:
    """Freeze the `Λ` lead-time prediction. The StudyID stamps readings taken under it REGISTERED."""
    return freeze(LAMBDA_LEAD_TIME_SPEC, repo_dir=repo_dir, frozen_at=frozen_at)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


#: The detector-free comparison, added after the freeze and therefore **not preregistered**.
#:
#: Running the frozen analysis on a real labelled series showed that the registered CUSUM metric
#: does not resolve the question there, for a reason that is a property of the detector rather than
#: of `Λ`. Two symptoms, both measured on the AISI kl0.0 series and both reproduced in this
#: package's tests. With `baseline=None` the CUSUM standardises against the whole series, so every
#: early point sits about 1.3 sigma **below** a mean the post-transition regime raised, the negative
#: accumulator crosses within seven points, and the reported "onset at step 11" is the detector
#: using the future to define normal. With a pre-transition baseline it instead fires at the end of
#: its own baseline window on an isolated spike inside the early noise band. Under the shipped
#: defaults (drift 0.5, threshold 5.0) the detector already fires one false alarm every 469
#: steps on a Gaussian series, and a sliding-window `Λ` is neither Gaussian nor homoskedastic.
#:
#: So the same transition fit is applied to `Λ` itself and the two midpoints are compared. It is
#: symmetric, it has no threshold to tune, and "as a fraction of the transition window" is already
#: the frozen framing. It is reported beside the registered metric and never in place of it.
METRIC_MIDPOINT_LEAD = "lambda_midpoint_lead_fraction"


@dataclass(frozen=True)
class LeadTimeResult:
    """What the frozen prediction resolves to on one series, including "it does not"."""

    transition: TransitionWindow
    lambda_step: float
    lambda_lead: float
    baseline_step: float
    baseline_lead: float
    margin: float
    resolved: bool
    lambda_transition: TransitionWindow | None = None
    midpoint_lead: float = float("nan")
    detail: str = ""

    def metrics(self) -> dict[str, float]:
        return {
            METRIC_LAMBDA_LEAD: self.lambda_lead,
            METRIC_BASELINE_LEAD: self.baseline_lead,
            METRIC_MARGIN: self.margin,
            METRIC_WIDTH: self.transition.width,
            METRIC_MIDPOINT_LEAD: self.midpoint_lead,
        }

    def render(self) -> str:
        if not self.resolved:
            return f"unresolved: {self.detail}"
        lines = [
            f"hack-rate transition at step {self.transition.t50:.1f}, 10-90 width "
            f"{self.transition.width:.1f} steps",
            f"    registered CUSUM metric: Lambda onset at step {self.lambda_step:.0f}, lead "
            f"{self.lambda_lead:+.3f} widths",
        ]
        if self.lambda_transition is not None and self.lambda_transition.fitted:
            lines.append(
                f"    detector-free (not preregistered): Lambda's own transition at step "
                f"{self.lambda_transition.t50:.1f}, lead {self.midpoint_lead:+.3f} widths"
            )
        lines.append(
            f"    gradient-norm onset at step {self.baseline_step:.0f}, lead "
            f"{self.baseline_lead:+.3f} widths; margin {self.margin:+.3f} widths"
        )
        if self.detail:
            lines.append(f"    {self.detail}")
        return "\n".join(lines)


def score_lead_time(
    lambda_series: Sequence[float],
    lambda_steps: Sequence[int],
    hack_rate: Sequence[float],
    hack_steps: Sequence[int],
    baseline_series: Sequence[float] | None = None,
    baseline_steps: Sequence[int] | None = None,
    *,
    cusum_baseline: int | None = None,
) -> LeadTimeResult:
    """Resolve the frozen prediction on one series. Returns unresolved rather than a number.

    ``cusum_baseline`` defaults to None, which is what the analysis did at freeze time and is
    therefore what the registered metric means. It is exposed rather than changed, because editing a
    frozen analysis after seeing the data is exactly what the freeze exists to make visible: a
    different value here produces a different number under the same study id, so the caller has to
    say which they ran. `METRIC_MIDPOINT_LEAD` is the comparison to read instead, and the reason is
    written out at that constant.

    The gradient-norm baseline is optional and its absence is carried rather than filled: a published
    rollout table does not carry optimiser telemetry, and scoring H2 against a comparator that was
    never computed would be scoring it against nothing.
    """
    window = transition_window(hack_rate, hack_steps)
    if not window.fitted:
        return LeadTimeResult(
            transition=window,
            lambda_step=float("nan"),
            lambda_lead=float("nan"),
            baseline_step=float("nan"),
            baseline_lead=float("nan"),
            margin=float("nan"),
            resolved=False,
            detail=(
                "the labelled hack rate has no fitted transition on this series, so there is no "
                f"window to express a lead time in. {window.detail}"
            ),
        )
    _, lam_step = onset_of(lambda_series, lambda_steps, baseline=cusum_baseline)
    lam_lead = window.fraction(lam_step) if np.isfinite(lam_step) else float("nan")
    lam_window = transition_window(lambda_series, lambda_steps)
    midpoint_lead = window.fraction(lam_window.t50) if lam_window.fitted else float("nan")

    if baseline_series is None:
        return LeadTimeResult(
            transition=window,
            lambda_step=lam_step,
            lambda_lead=lam_lead,
            baseline_step=float("nan"),
            baseline_lead=float("nan"),
            margin=float("nan"),
            resolved=bool(np.isfinite(lam_lead)),
            lambda_transition=lam_window,
            midpoint_lead=midpoint_lead,
            detail=(
                "no gradient-norm series was supplied, so H2 is unresolved. A published rollout "
                "table carries rewards and labels and no optimiser telemetry; the comparator needs "
                "the training log from the same run."
            ),
        )
    base = gradnorm_onset(baseline_series, baseline_steps)
    base_lead = window.fraction(base["earliest"])
    return LeadTimeResult(
        transition=window,
        lambda_step=lam_step,
        lambda_lead=lam_lead,
        baseline_step=base["earliest"],
        baseline_lead=base_lead,
        margin=float(lam_lead - base_lead),
        resolved=bool(np.isfinite(lam_lead) and np.isfinite(base_lead)),
        lambda_transition=lam_window,
        midpoint_lead=midpoint_lead,
    )


__all__ = [
    "LAMBDA_LEAD_TIME_SPEC",
    "LeadTimeResult",
    "METRIC_BASELINE_LEAD",
    "METRIC_LAMBDA_LEAD",
    "METRIC_MARGIN",
    "METRIC_MIDPOINT_LEAD",
    "METRIC_WIDTH",
    "TransitionWindow",
    "freeze_prediction",
    "gradnorm_onset",
    "onset_of",
    "score_lead_time",
    "transition_window",
]

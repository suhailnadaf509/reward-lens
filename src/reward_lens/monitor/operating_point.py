"""The operating point from an asymmetric loss, and the PPV line every detector owes (J4).

A monitor's threshold is a decision, so it needs a loss. The convention in this literature is a
threshold of 0.5 or a false-positive rate of 5%, neither of which is derived from anything, and
both of which are indefensible the moment the two errors cost different amounts. Missing a reward
hack that ships costs more than waking somebody at 3am. State the ratio and the threshold follows.

**The line that prevents a recurring class of overclaim.** At a 2.5% base rate and a 31.7%
false-positive rate the positive predictive value is 7.5% **even with perfect sensitivity**. That
is Bayes' rule and nothing else:

    PPV = se * pi / (se * pi + fpr * (1 - pi))
        = 1.0 * 0.025 / (1.0 * 0.025 + 0.317 * 0.975)
        = 0.025 / 0.334 = 0.0748

It is one line, it was predictable rather than discoverable, and it is the shape of this library's
own `EVAL-AWARE` result. **Every detector in this package attaches a PPV curve**, computed by
`ppv_curve` and carried on the reading, so a detector that looks impressive at a stated sensitivity
cannot be read without the prevalence that decides what it is worth.

**What this instrument cannot do.** It cannot tell you the prevalence. Nothing can: a base rate of
reward hacking in your runs is a property of your runs, and the whole point of the curve is that the
answer swings by an order of magnitude across the range anyone would guess. So the curve is the
reading and the point estimate is what you get once you commit to a number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec
from reward_lens.core.evidence import Uncertainty
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.reading import Reading, Refusal, RefusalReason
from reward_lens.core.types import Capability, GaugeStatus
from reward_lens.measure.base import Context
from reward_lens.monitor._base import NO_ACCESS, MonitorInstrument

# ---------------------------------------------------------------------------
# Bayes' rule, and the numbers of the worked case
# ---------------------------------------------------------------------------

#: The prevalence grid every reading carries by default. Log-ish rather than linear because the
#: interesting behaviour is all below 10%: PPV is nearly linear in prevalence down there and the
#: whole failure mode is a reader who assumed 50%.
DEFAULT_PREVALENCES: tuple[float, ...] = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.10,
    0.20,
    0.50,
)

#: The worked case, kept as data so the docstring's number is checkable rather than asserted.
#: `tests/acceptance/test_w4_7_monitor.py` recomputes it.
WORKED_CASE: dict[str, float] = {"prevalence": 0.025, "fpr": 0.317, "sensitivity": 1.0}


def ppv(sensitivity: float, fpr: float, prevalence: float) -> float:
    """Positive predictive value from Bayes' rule. The probability an alarm is a real event.

    ``se * pi / (se * pi + fpr * (1 - pi))``. Returns NaN when the denominator is zero, which
    happens only when a detector never fires at all: with no alarms there is no such thing as the
    fraction of alarms that are real, and returning 1.0 or 0.0 there would be inventing one.
    """
    num = sensitivity * prevalence
    den = num + fpr * (1.0 - prevalence)
    if den <= 0.0:
        return float("nan")
    return float(num / den)


def npv(sensitivity: float, fpr: float, prevalence: float) -> float:
    """Negative predictive value: the probability that silence means nothing is happening."""
    num = (1.0 - fpr) * (1.0 - prevalence)
    den = num + (1.0 - sensitivity) * prevalence
    if den <= 0.0:
        return float("nan")
    return float(num / den)


@dataclass(frozen=True)
class PPVCurve:
    """PPV against assumed prevalence, at one detector's operating characteristic.

    ``prevalence_for_ppv`` is the field a reader acts on: it inverts the curve and says what the
    base rate would have to be for the detector to be worth acting on at all. When that number is
    above anything plausible, the detector is not usable at this operating point however good its
    AUC looks, and saying so is the point of the type.
    """

    sensitivity: float
    fpr: float
    prevalences: tuple[float, ...]
    ppv: tuple[float, ...]
    npv: tuple[float, ...]
    target_ppv: float = 0.5

    @property
    def prevalence_for_ppv(self) -> float:
        """The base rate at which this detector reaches ``target_ppv``, by inverting Bayes.

        ``pi* = t f / (s (1 - t) + t f)`` for target ``t``, sensitivity ``s`` and false-positive
        rate ``f``. Returns NaN when the detector's sensitivity is zero, since no prevalence saves a
        detector that never fires on a real event.
        """
        s, f, t = self.sensitivity, self.fpr, self.target_ppv
        den = s * (1.0 - t) + t * f
        if den <= 0.0:
            return float("nan")
        return float(t * f / den)

    def at(self, prevalence: float) -> float:
        return ppv(self.sensitivity, self.fpr, prevalence)

    def render(self) -> str:
        head = (
            f"PPV against assumed prevalence at sensitivity {self.sensitivity:.3g}, "
            f"false-positive rate {self.fpr:.3g}"
        )
        rows = [
            f"    prevalence {p:>7.3%}   PPV {v:>7.2%}   NPV {n:>7.2%}"
            for p, v, n in zip(self.prevalences, self.ppv, self.npv)
        ]
        tail = (
            f"    a base rate of {self.prevalence_for_ppv:.2%} is what this detector would need "
            f"for an alarm to be right half the time."
        )
        return "\n".join([head, *rows, tail])


def ppv_curve(
    sensitivity: float,
    fpr: float,
    prevalences: Sequence[float] = DEFAULT_PREVALENCES,
    *,
    target_ppv: float = 0.5,
) -> PPVCurve:
    """The curve every detector in this package attaches to its reading. One line, one class of bug."""
    ps = tuple(float(p) for p in prevalences)
    return PPVCurve(
        sensitivity=float(sensitivity),
        fpr=float(fpr),
        prevalences=ps,
        ppv=tuple(ppv(sensitivity, fpr, p) for p in ps),
        npv=tuple(npv(sensitivity, fpr, p) for p in ps),
        target_ppv=float(target_ppv),
    )


# ---------------------------------------------------------------------------
# The loss, and the threshold that minimises it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AsymmetricLoss:
    """What each error costs, in whatever unit the caller thinks in.

    Only the ratio matters for where the threshold lands, so ``cost_ratio`` is the field to read.
    Both are kept because the expected loss at the optimum is reported in the caller's own units,
    and a number in units of "missed hacks" is easier to argue with than a dimensionless one.
    """

    cost_false_negative: float = 20.0
    cost_false_positive: float = 1.0

    def __post_init__(self) -> None:
        if self.cost_false_negative <= 0 or self.cost_false_positive <= 0:
            raise ValueError(
                "both costs must be positive. A zero cost on one side puts the optimal threshold "
                "at a boundary, which is a statement that one of the two errors does not matter, "
                "and if that is meant it should be said rather than encoded as a zero."
            )

    @property
    def cost_ratio(self) -> float:
        """How many false alarms one missed event is worth."""
        return self.cost_false_negative / self.cost_false_positive


@dataclass(frozen=True)
class OperatingPoint:
    """The threshold, where it came from, and what it is worth at the stated base rate."""

    threshold: float
    sensitivity: float
    fpr: float
    prevalence: float
    loss: AsymmetricLoss
    expected_loss: float
    curve: PPVCurve
    baseline_half: "OperatingPoint | None" = None
    baseline_fpr5: "OperatingPoint | None" = None
    n: int = 0
    n_positive: int = 0

    @property
    def ppv(self) -> float:
        return ppv(self.sensitivity, self.fpr, self.prevalence)

    def render(self) -> str:
        lines = [
            f"at a {self.loss.cost_ratio:.0f}:1 cost of a missed event over a false alarm, the "
            f"threshold is {self.threshold:.4g}.",
            f"    sensitivity {self.sensitivity:.3f}, false-positive rate {self.fpr:.3f}; "
            f"at a {self.prevalence:.1%} base rate that gives PPV {self.ppv:.3f}.",
            f"    expected loss per item {self.expected_loss:.4g} in units of one false alarm.",
        ]
        for label, other in (
            ("threshold at 0.5", self.baseline_half),
            ("5% false-positive convention", self.baseline_fpr5),
        ):
            if other is not None:
                lines.append(
                    f"    baseline, {label}: threshold {other.threshold:.4g}, "
                    f"PPV {other.ppv:.3f}, expected loss {other.expected_loss:.4g} "
                    f"({other.expected_loss / self.expected_loss:.2f}x the optimum)"
                    if self.expected_loss > 0
                    else f"    baseline, {label}: threshold {other.threshold:.4g}"
                )
        lines.append(self.curve.render())
        return "\n".join(lines)


def _rates_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    """Sensitivity and false-positive rate at one threshold, alarm when ``score >= threshold``."""
    fired = scores >= threshold
    pos = labels == 1
    neg = ~pos
    se = float(np.mean(fired[pos])) if pos.any() else float("nan")
    fpr = float(np.mean(fired[neg])) if neg.any() else float("nan")
    return se, fpr


def expected_loss(sensitivity: float, fpr: float, prevalence: float, loss: AsymmetricLoss) -> float:
    """Expected cost per item: ``pi (1 - se) C_FN + (1 - pi) fpr C_FP``.

    Per item rather than per alarm, because that is the quantity a threshold minimises and because
    per-alarm cost is undefined when the detector is silent.
    """
    return float(
        prevalence * (1.0 - sensitivity) * loss.cost_false_negative
        + (1.0 - prevalence) * fpr * loss.cost_false_positive
    )


def choose_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    loss: AsymmetricLoss | None = None,
    prevalence: float | None = None,
    prevalences: Sequence[float] = DEFAULT_PREVALENCES,
) -> OperatingPoint:
    """Minimise expected loss over the empirical thresholds, and report the two conventions beside it.

    The candidate set is the observed scores themselves, which is exhaustive: expected loss is
    piecewise constant in the threshold and changes only where a score sits, so a grid would either
    miss the optimum or waste evaluations.

    ``prevalence`` defaults to the empirical positive rate in ``labels``, and when it does the
    reading says so. That default is right for a labelled slice drawn from the population and wrong
    for an enriched one, which is the usual case in this field: a hand-built set of known hacks has
    a base rate near 50% and a training run has one near 2%. The curve is the defence against
    getting that wrong and it is attached either way.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(int)
    if s.size != y.size:
        raise ValueError(f"{s.size} scores and {y.size} labels")
    lo = loss or AsymmetricLoss()
    pi = float(np.mean(y == 1)) if prevalence is None else float(prevalence)

    candidates = np.unique(s)
    candidates = np.concatenate([candidates, [candidates.max() + 1.0]])
    best: tuple[float, float, float, float] | None = None
    for t in candidates:
        se, fpr = _rates_at(s, y, float(t))
        if not (np.isfinite(se) and np.isfinite(fpr)):
            continue
        val = expected_loss(se, fpr, pi, lo)
        if best is None or val < best[0]:
            best = (val, float(t), se, fpr)
    if best is None:
        raise ValueError(
            "no threshold has both a sensitivity and a false-positive rate, which means the labels "
            "carry only one class. A detection threshold is not defined there."
        )
    val, t, se, fpr = best

    def _at(threshold: float) -> OperatingPoint:
        se_b, fpr_b = _rates_at(s, y, threshold)
        return OperatingPoint(
            threshold=float(threshold),
            sensitivity=se_b,
            fpr=fpr_b,
            prevalence=pi,
            loss=lo,
            expected_loss=expected_loss(se_b, fpr_b, pi, lo),
            curve=ppv_curve(se_b, fpr_b, prevalences),
            n=int(s.size),
            n_positive=int(np.sum(y == 1)),
        )

    # The 5% convention: the smallest threshold whose false-positive rate is at most 0.05.
    fpr5 = candidates[-1]
    for t5 in candidates:
        _, f = _rates_at(s, y, float(t5))
        if np.isfinite(f) and f <= 0.05:
            fpr5 = t5
            break

    return OperatingPoint(
        threshold=t,
        sensitivity=se,
        fpr=fpr,
        prevalence=pi,
        loss=lo,
        expected_loss=val,
        curve=ppv_curve(se, fpr, prevalences),
        baseline_half=_at(0.5),
        baseline_fpr5=_at(float(fpr5)),
        n=int(s.size),
        n_positive=int(np.sum(y == 1)),
    )


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------

#: J4 asserts arithmetic about numbers the caller states. No regime of any run can make Bayes' rule
#: wrong, and the premises that can be wrong are the caller's prevalence and costs, which travel on
#: the reading rather than being checked against a record the instrument never reads.
LOSS_ENVELOPE = EnvelopeSpec(
    unconditional=True,
    justification=(
        "Bayes' rule applied to a stated prevalence, a stated cost ratio and a measured operating "
        "characteristic. No property of the run under measurement can make the arithmetic wrong, "
        "and the two stated numbers are recorded as stated rather than assumed to be right."
    ),
)

J4_BASELINES: tuple[str, ...] = (
    "baseline.threshold_at_half",
    "baseline.fpr_at_5_percent",
)


class OperatingPointFromLoss(MonitorInstrument):
    """J4. The threshold that minimises a stated asymmetric loss, with the PPV curve beside it.

    Needs no access to anything: a labelled slice of scores, a cost ratio and a prevalence. That is
    why it is the instrument every other detector in this package borrows from rather than the
    other way round.

    What it cannot do: it cannot tell you your base rate, and the threshold it returns is optimal
    for the base rate you stated and no other. The curve on the reading is there because the honest
    answer to "is this detector any good" depends on a number nobody in this field measures.
    """

    name = "OperatingPointFromLoss"
    version = "1.0"
    capabilities = Capability.NONE
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "Bayes' rule; minimum expected loss over a two-error decision problem"
    deviations = (
        "the threshold is chosen on the same labelled slice its operating characteristic is "
        "measured on, so both are optimistic by the usual selection amount. A held-out split is "
        "the fix and it needs labels this instrument is not given.",
    )

    quantity = "monitor.operating_point"
    requires = NO_ACCESS
    envelope = LOSS_ENVELOPE
    invariance = "units"
    invariance_relation = INVARIANT
    baselines = J4_BASELINES
    rung = 0

    def __init__(
        self,
        scores: Sequence[float],
        labels: Sequence[int],
        *,
        loss: AsymmetricLoss | None = None,
        prevalence: float | None = None,
        prevalences: Sequence[float] = DEFAULT_PREVALENCES,
    ) -> None:
        self.scores = np.asarray(scores, dtype=np.float64).ravel()
        self.labels = np.asarray(labels).ravel().astype(int)
        self.loss = loss or AsymmetricLoss()
        self.prevalence = prevalence
        self.prevalences = tuple(float(p) for p in prevalences)

    def compute(self, ctx: Context) -> OperatingPoint | Refusal:
        if np.unique(self.labels).size < 2:
            return Refusal(
                instrument=self.name,
                reason=RefusalReason.LABEL_QUALITY_UNKNOWN,
                detail=(
                    f"the {self.labels.size} labels carry one class only "
                    f"({int(np.sum(self.labels == 1))} positive), so no threshold has both a "
                    f"sensitivity and a false-positive rate."
                ),
                remedy=(
                    "Supply a slice containing at least one positive and one negative. If your "
                    "labelled set genuinely has one class, what you have is a prevalence estimate "
                    "and not a detection problem, and `ppv_curve` will still tell you what any "
                    "operating characteristic would be worth at that rate."
                ),
                statistics={
                    "n": int(self.labels.size),
                    "n_positive": int(np.sum(self.labels == 1)),
                },
            )
        return choose_threshold(
            self.scores,
            self.labels,
            loss=self.loss,
            prevalence=self.prevalence,
            prevalences=self.prevalences,
        )

    def payload(self, computed: OperatingPoint) -> dict:
        return {
            "threshold": computed.threshold,
            "sensitivity": computed.sensitivity,
            "fpr": computed.fpr,
            "prevalence": computed.prevalence,
            "prevalence_stated": self.prevalence is not None,
            "cost_ratio": computed.loss.cost_ratio,
            "ppv": computed.ppv,
            "expected_loss": computed.expected_loss,
            "ppv_curve": {
                "prevalences": list(computed.curve.prevalences),
                "ppv": list(computed.curve.ppv),
                "npv": list(computed.curve.npv),
                "prevalence_for_half_ppv": computed.curve.prevalence_for_ppv,
            },
            "baselines": self.baseline_map(computed),
            "n": computed.n,
            "n_positive": computed.n_positive,
            "rendered": computed.render(),
        }

    def baseline_map(self, computed: OperatingPoint) -> dict[str, float]:
        out: dict[str, float] = {}
        if computed.baseline_half is not None:
            out["baseline.threshold_at_half"] = computed.baseline_half.expected_loss
        if computed.baseline_fpr5 is not None:
            out["baseline.fpr_at_5_percent"] = computed.baseline_fpr5.expected_loss
        return out

    def uncertainty(self, computed: OperatingPoint) -> Uncertainty:
        return Uncertainty(n=computed.n, method="empirical operating characteristic, in-sample")


def estimate(instrument: OperatingPointFromLoss, ctx: Context) -> Reading:
    """Convenience for callers who hold an instrument and a context and want the reading."""
    return instrument.estimate(ctx)


__all__ = [
    "DEFAULT_PREVALENCES",
    "J4_BASELINES",
    "LOSS_ENVELOPE",
    "WORKED_CASE",
    "AsymmetricLoss",
    "OperatingPoint",
    "OperatingPointFromLoss",
    "PPVCurve",
    "choose_threshold",
    "expected_loss",
    "npv",
    "ppv",
    "ppv_curve",
]

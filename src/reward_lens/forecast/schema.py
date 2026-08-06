"""The forecast object, and the third clock it turns on.

A measurement that cannot support a claim about the future cannot support a decision, so this
package is not a section of the science layer. It is what makes the rest of the library worth
paying for, and the only thing it has that the four competing methods do not is that its predictor
is fitted strictly before the outcome it predicts, enforced by type rather than by review.

Everything here is frozen at issue. The resolution rule, the reference class, the baselines and the
decision spec are all part of the forecast, not annotations added when the answer is known, because
every one of them is a place a forecast can be improved after the fact without anybody noticing.

Read `barrier.py` next. This module defines the object; that one is the guarantee.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, NewType, Sequence

from reward_lens.core.errors import RewardLensError
from reward_lens.core.provenance import Provenance
from reward_lens.core.types import EvidenceID, SubjectRef, content_hash

QuantityID = str

#: A content-derived forecast id. Two forecasts made from the same inputs at the same information
#: time about the same target under the same rule are the same forecast and share an id.
ForecastID = NewType("ForecastID", str)

#: The population a calibration claim is conditional on. "Our forecasts are well calibrated" is not
#: a claim until this says over which runs.
ReferenceClassID = NewType("ReferenceClassID", str)


class ForecastError(RewardLensError):
    """A forecast that could not be constructed as stated. Distinct from a refusal.

    A refusal is a measurement declined with a remedy. This is a forecast that does not exist:
    an input that postdates its own issue, a mandatory baseline that was never run, a clock read
    off the wrong axis. None of those produce a worse forecast, they produce a thing that is not a
    forecast, so they raise at construction the way `EnvelopeSpec` does.
    """


# ---------------------------------------------------------------------------
# The third clock
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InformationTime:
    """When a thing became available to the forecaster. The third clock.

    There are three clocks in a training run and they are routinely confused. **Wall time** is when
    the bytes were written. **Run position** is which optimiser step produced them. **Information
    time** is when they became available to whoever is making the forecast, and it is not derivable
    from either of the others. A checkpoint produced at step 200 has run position 200 and an
    information time of whenever the forecaster could first read it, and those come apart the moment
    anyone reanalyses an archive: every input to a 2026 reanalysis of a 2024 run has run position in
    the hundreds, wall time in 2024, and information time in 2026.

    Getting that wrong is the failure this whole package exists to catch, so it lives in the type
    rather than in a naming convention. Three things follow from that and all three are deliberate:

    ``instant`` must be timezone-aware. A naive timestamp compared against an aware one either
    raises or silently orders wrong depending on which side it lands, and an ordering that silently
    fails is a barrier that silently passes.

    ``basis`` says what made it available, and it is required. "2026-07-19T14:38:14+00:00" is not an
    answer to "when could you have known this", it is a number; "the ladder intervals were written
    at runbook 5b, before the 8B arc" is. The barrier reports the basis alongside the timestamp when
    it refuses, because the reader's next question is always why that instant and not another.

    `from_run_position` **raises**. It exists so that the commonest mistake in this area has a name
    at the point it is made rather than an implicit coercion three modules away.
    """

    #: ISO-8601, normalised to UTC at construction so string order is time order.
    instant: str
    #: What made this available to the forecaster. Required; see the class docstring.
    basis: str

    def __post_init__(self) -> None:
        if not self.basis.strip():
            raise ForecastError(
                "an InformationTime needs a basis: what made this available to the forecaster at "
                "that instant. A bare timestamp cannot be checked by a reader and cannot be "
                "reconstructed by the next person to reanalyse the archive. One clause is enough: "
                "'the eval log was written', 'the checkpoint was published', 'reanalysed from the "
                "archive on this date'."
            )
        dt = _parse_instant(self.instant)
        object.__setattr__(self, "instant", dt.astimezone(timezone.utc).isoformat())

    @classmethod
    def parse(cls, instant: str, *, basis: str) -> "InformationTime":
        """Build from an ISO-8601 timestamp, which must carry an offset."""
        return cls(instant=instant, basis=basis)

    @classmethod
    def at(cls, dt: datetime, *, basis: str) -> "InformationTime":
        if dt.tzinfo is None:
            raise ForecastError(
                f"{dt!r} carries no timezone. An information time compared across timezones is the "
                f"one comparison in this package that must never be approximate; attach "
                f"timezone.utc, or the zone the log was written in."
            )
        return cls(instant=dt.isoformat(), basis=basis)

    @classmethod
    def from_run_position(cls, step: int, **_: Any) -> "InformationTime":
        """Always raises. Run position is not information time and never was.

        Kept as a named method rather than left absent so that the mistake fails where it is made.
        An `AttributeError` three frames up reads as a missing helper; this reads as the thing it
        is, which is a category error about which clock is being asked for.
        """
        raise ForecastError(
            f"step {step} is a run position, not an information time. A checkpoint produced at "
            f"step {step} became available to the forecaster at some instant, and which instant "
            f"that was is a fact about your workflow rather than about the run: on a live run it is "
            f"roughly when the step finished, and on a reanalysis of an archive it is the day you "
            f"opened the archive. Supply the instant with `InformationTime.parse(..., basis=...)`. "
            f"If you genuinely mean 'the moment step {step} completed', say so in the basis, "
            f"because that is a claim somebody may later need to check."
        )

    @property
    def epoch(self) -> float:
        """Seconds since the Unix epoch. The only thing comparisons use."""
        return _parse_instant(self.instant).timestamp()

    def __lt__(self, other: "InformationTime") -> bool:
        return self.epoch < other.epoch

    def __le__(self, other: "InformationTime") -> bool:
        return self.epoch <= other.epoch

    def __gt__(self, other: "InformationTime") -> bool:
        return self.epoch > other.epoch

    def __ge__(self, other: "InformationTime") -> bool:
        return self.epoch >= other.epoch

    def __str__(self) -> str:
        return f"{self.instant} ({self.basis})"

    def __canonical__(self) -> dict[str, Any]:
        return {"instant": self.instant, "basis": self.basis}


def _parse_instant(text: str) -> datetime:
    """Parse an ISO-8601 instant, refusing a naive one.

    ``Z`` is handled here rather than by `datetime.fromisoformat`, which only learned it in 3.11
    and this package supports the 3.10 floor in `pyproject.toml`.
    """
    raw = text.strip()
    if raw.endswith(("Z", "z")):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ForecastError(
            f"{text!r} is not an ISO-8601 instant. Write it as 2026-07-19T14:38:14+00:00, which is "
            f"what every timestamp in an evidence store already looks like."
        ) from exc
    if dt.tzinfo is None:
        raise ForecastError(
            f"{text!r} carries no UTC offset. A naive timestamp compared against an aware one "
            f"either raises or orders wrong depending on which side of the comparison it lands, "
            f"and an ordering that fails quietly is a barrier that passes quietly. Append the "
            f"offset, or +00:00 if the log is already UTC."
        )
    return dt


# ---------------------------------------------------------------------------
# The reference class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceClass:
    """The population a forecast is conditional on, and the base rate in it.

    A calibration claim with no reference class is not a claim. "We are well calibrated" over what
    population of runs? Climatology is the base rate *in this class*, so this object is also the
    first mandatory baseline's entire input, which is why it carries `n` and `base_rate` rather than
    only a description: a class whose base rate nobody counted cannot produce a climatology, and
    the forecast is then refused rather than issued against three baselines.
    """

    id: ReferenceClassID
    definition: str
    #: How many members of the class have been observed. `None` means nobody counted, which makes
    #: climatology unavailable and is therefore a refusable condition rather than a missing field.
    n: int | None = None
    #: The observed frequency of the event in the class. `None` for the same reason.
    base_rate: float | None = None

    def __post_init__(self) -> None:
        if not self.definition.strip():
            raise ForecastError(
                f"reference class {self.id!r} has no definition. The definition is what a reader "
                f"checks their own run against to decide whether the calibration number applies to "
                f"them, so a class id with no membership rule is a label rather than a population."
            )
        if self.base_rate is not None and not 0.0 <= self.base_rate <= 1.0:
            raise ForecastError(f"base rate {self.base_rate} for {self.id!r} is not a probability.")

    @property
    def has_climatology(self) -> bool:
        return self.base_rate is not None and self.n is not None and self.n > 0

    def __canonical__(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "definition": self.definition,
            "n": self.n,
            "base_rate": self.base_rate,
        }


# ---------------------------------------------------------------------------
# The distribution. Not a point.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinaryProbability:
    """A probability for a binary event. The commonest forecast and the one the campaign made."""

    p: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0 or math.isnan(self.p):
            raise ForecastError(f"{self.p} is not a probability")

    @property
    def kind(self) -> str:
        return "binary"

    @property
    def point(self) -> float:
        return self.p

    def probability_of(self, outcome: bool) -> float:
        return self.p if outcome else 1.0 - self.p

    def covers(self, value: float) -> bool:  # pragma: no cover - not meaningful here
        raise ForecastError(
            "a binary probability has no interval to cover a value with. Score it with `brier` or "
            "`log_score`, or issue an IntervalForecast if the target is continuous."
        )

    def render(self) -> str:
        return f"P = {self.p:.3f}"

    def __canonical__(self) -> dict[str, Any]:
        return {"kind": "binary", "p": self.p}


@dataclass(frozen=True)
class IntervalForecast:
    """A central interval at a stated nominal level. Scored by coverage, not by hit or miss."""

    lo: float
    hi: float
    level: float = 0.8
    point: float | None = None

    def __post_init__(self) -> None:
        if self.hi < self.lo:
            raise ForecastError(f"interval [{self.lo}, {self.hi}] has its bounds the wrong way up")
        if not 0.0 < self.level < 1.0:
            raise ForecastError(f"nominal level {self.level} is not in (0, 1)")

    @property
    def kind(self) -> str:
        return "interval"

    def covers(self, value: float) -> bool:
        return bool(self.lo <= value <= self.hi)

    def probability_of(self, outcome: bool) -> float:  # pragma: no cover - not meaningful here
        raise ForecastError(
            "an interval forecast does not assign a probability to a binary outcome. Score it by "
            "coverage against its nominal level."
        )

    def render(self) -> str:
        mid = "" if self.point is None else f" (point {self.point:.4g})"
        return f"[{self.lo:.4g}, {self.hi:.4g}] at {self.level:.0%}{mid}"

    def __canonical__(self) -> dict[str, Any]:
        return {
            "kind": "interval",
            "lo": self.lo,
            "hi": self.hi,
            "level": self.level,
            "point": self.point,
        }


@dataclass(frozen=True)
class QuantileForecast:
    """A predictive distribution given as quantiles. Scored by the pinball loss or by coverage."""

    levels: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.levels) != len(self.values):
            raise ForecastError(
                f"{len(self.levels)} quantile levels against {len(self.values)} values"
            )
        if not self.levels:
            raise ForecastError("a quantile forecast with no quantiles is not a distribution")
        if list(self.levels) != sorted(self.levels):
            raise ForecastError("quantile levels must be ascending")
        if list(self.values) != sorted(self.values):
            raise ForecastError(
                "quantile values must be non-decreasing in the level, or the distribution "
                "they describe has negative density somewhere"
            )

    @property
    def kind(self) -> str:
        return "quantile"

    @property
    def point(self) -> float:
        """The median, or the closest quantile to it."""
        idx = min(range(len(self.levels)), key=lambda i: abs(self.levels[i] - 0.5))
        return self.values[idx]

    def interval(self, level: float = 0.8) -> IntervalForecast:
        tail = (1.0 - level) / 2.0
        lo = self._at(tail)
        hi = self._at(1.0 - tail)
        return IntervalForecast(lo=lo, hi=hi, level=level, point=self.point)

    def _at(self, level: float) -> float:
        idx = min(range(len(self.levels)), key=lambda i: abs(self.levels[i] - level))
        return self.values[idx]

    def covers(self, value: float) -> bool:
        return bool(self.values[0] <= value <= self.values[-1])

    def probability_of(self, outcome: bool) -> float:  # pragma: no cover - not meaningful here
        raise ForecastError("a quantile forecast does not assign a probability to a binary outcome")

    def render(self) -> str:
        return ", ".join(
            f"q{level:.2f}={value:.4g}" for level, value in zip(self.levels, self.values)
        )

    def __canonical__(self) -> dict[str, Any]:
        return {"kind": "quantile", "levels": list(self.levels), "values": list(self.values)}


#: What a forecast carries instead of a point estimate. A point is not a forecast: it cannot be
#: scored by a proper rule, it cannot be checked for coverage, and it cannot enter a decision.
Distribution = Any  # statically: BinaryProbability | IntervalForecast | QuantileForecast


# ---------------------------------------------------------------------------
# The resolution rule
# ---------------------------------------------------------------------------


class Comparator(enum.Enum):
    """The six comparators a mechanically evaluable rule needs.

    These are the campaign's, verbatim, because the first thing this package has to do is re-score
    the campaign and a rule vocabulary that cannot express the frozen specs would have to be
    translated, which is exactly the step where a prediction quietly becomes a different prediction.
    """

    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="
    EQ = "=="
    ABS_LT = "abs<"

    def evaluate(self, value: float, threshold: float) -> bool:
        if self is Comparator.GT:
            return value > threshold
        if self is Comparator.GE:
            return value >= threshold
        if self is Comparator.LT:
            return value < threshold
        if self is Comparator.LE:
            return value <= threshold
        if self is Comparator.EQ:
            return value == threshold
        return abs(value) < threshold

    def render(self, metric: str, threshold: float) -> str:
        if self is Comparator.ABS_LT:
            return f"|{metric}| < {threshold:g}"
        return f"{metric} {self.value} {threshold:g}"


@dataclass(frozen=True)
class ResolutionRule:
    """How this forecast is decided, written down before the answer is known.

    Mechanically evaluable is the whole requirement: a rule a person has to interpret is a rule the
    person can interpret in the direction the result went. `evaluate` takes the metric mapping the
    analysis produced and returns True, False, or `None` when the metric is absent, and the third
    case is the one that matters. A missing metric is a void, never a miss.
    """

    metric: str
    comparator: Comparator
    threshold: float
    #: What the metric means, in one sentence, for whoever resolves this in six months.
    definition: str = ""

    def evaluate(self, metrics: Mapping[str, float]) -> bool | None:
        value = metrics.get(self.metric)
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric):
            return None
        return self.comparator.evaluate(numeric, self.threshold)

    def render(self) -> str:
        return self.comparator.render(self.metric, self.threshold)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "comparator": self.comparator.value,
            "threshold": self.threshold,
            "definition": self.definition,
        }


@dataclass(frozen=True)
class HorizonSpec:
    """How far ahead this reaches, and when it stops being answerable.

    ``kind`` is the unit the horizon is stated in. ``steps`` is the literature's unit and is not
    comparable across runs; ``widths`` is a fraction of the fitted transition width, and is.
    ``time`` is for a forecast about a calendar deadline rather than about a run.

    ``expires_at`` is what makes a void possible. A forecast whose horizon has passed with no
    resolvable metric is void with a reason, not a miss, and without an expiry there is no instant
    at which that can be said.
    """

    kind: Literal["steps", "widths", "time"] = "steps"
    value: float = 0.0
    expires_at: InformationTime | None = None

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ForecastError(f"a horizon cannot be negative; got {self.value}")

    def render(self) -> str:
        if self.kind == "widths":
            return f"{self.value:.3g} transition widths ahead"
        if self.kind == "time":
            return f"by {self.expires_at}" if self.expires_at else "an unbounded time horizon"
        return f"{self.value:g} steps ahead"

    def __canonical__(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "expires_at": None if self.expires_at is None else self.expires_at.__canonical__(),
        }


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionSpec:
    """What action this forecast is for, at what threshold, at what loss.

    A forecast with no action attached is a number. This is the cost-loss model, which is the oldest
    decision-analytic framing in forecasting and still the only one that answers "so what": you can
    pay ``cost`` now to protect against an event that would cost ``loss`` if it happened and you had
    not. The threshold at which acting is worth it is `cost / loss`, and it is derived rather than
    declared, because a threshold somebody typed in is the place a decision analysis gets tuned
    until it agrees with what was going to be done anyway.

    Worked with illustrative figures, since the point is the arithmetic and not the numbers: suppose
    a run costs 480 GPU-hours, killing it partway and restarting against a fixed grader costs 90, and
    letting a hacked run finish and discovering it at the end costs the whole 480. Then `cost=90`,
    `loss=480`, and the threshold is 0.1875. That threshold is not a preference, it is arithmetic on
    two costs the operator already knows, which is why it is derived here rather than typed in.
    """

    action: str
    cost: float
    loss: float
    unit: str = "gpu_hours"
    #: Set only when the operator deliberately acts away from the cost-loss optimum, which is a real
    #: thing (a safety case may require acting earlier than break-even). Left None it is derived.
    override_threshold: float | None = None

    def __post_init__(self) -> None:
        if not self.action.strip():
            raise ForecastError("a decision spec with no action is a forecast with no decision")
        if self.cost < 0 or self.loss <= 0:
            raise ForecastError(
                f"cost {self.cost} and loss {self.loss} must be non-negative and the loss "
                f"positive; a decision whose loss is zero has nothing to protect against"
            )
        if self.cost >= self.loss:
            raise ForecastError(
                f"acting costs {self.cost} {self.unit} and the event it protects against costs "
                f"{self.loss}. Protection that costs more than the thing it prevents is never "
                f"worth taking at any probability, so no forecast can change this decision and "
                f"scoring one against it would report a decision value of zero for a reason that "
                f"has nothing to do with the forecast."
            )

    @property
    def threshold(self) -> float:
        """The probability above which acting has lower expected loss than not acting."""
        if self.override_threshold is not None:
            return self.override_threshold
        return self.cost / self.loss

    def expected_loss(self, *, acted: bool, event: bool) -> float:
        """The loss realised by one decision, in `unit`.

        Acting pays `cost` whether or not the event arrives, which is the point of protection.
        Not acting pays `loss` if the event arrives and nothing otherwise.
        """
        if acted:
            return self.cost
        return self.loss if event else 0.0

    def render(self) -> str:
        return (
            f"{self.action}: cost {self.cost:g} {self.unit}, loss {self.loss:g} {self.unit}, "
            f"act above P = {self.threshold:.4g}"
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "cost": self.cost,
            "loss": self.loss,
            "unit": self.unit,
            "threshold": self.threshold,
        }


# ---------------------------------------------------------------------------
# The mandatory baselines
# ---------------------------------------------------------------------------


class BaselineKind(enum.Enum):
    """The four kinds that are mandatory. A forecast missing any of them is refused.

    They are four kinds rather than four named methods because each is a family: the dumb statistic
    for hacking onset is the gradient-norm peak, and for a text-level call it is a string match, and
    a forecast should name whichever one is dumb for its own target rather than the one that happens
    to be implemented.
    """

    #: The base rate in the reference class. A forecaster that cannot beat "hacking happens in 12%
    #: of runs like this" is not forecasting.
    CLIMATOLOGY = "climatology"
    #: Whatever is true now stays true.
    PERSISTENCE = "persistence"
    #: The zero-or-one-parameter statistic that is free and already in the log.
    DUMB_STATISTIC = "dumb_statistic"
    #: The strongest published method that needs no gold evaluator and no activations. Currently
    #: contrastive belief-flipping. It is the first thing a reviewer raises, so it is not optional.
    SCAFFOLDED_BLACK_BOX = "scaffolded_black_box"


MANDATORY_BASELINE_KINDS: frozenset[BaselineKind] = frozenset(BaselineKind)


@dataclass(frozen=True)
class BaselineForecast:
    """One comparator's forecast for the same target, under the same rule, at the same instant.

    ``refused`` is what keeps this honest, and it is the same argument the dumb-baseline bank makes
    for its own refusals: a forecast that never ran the black-box comparator and a forecast that ran
    it and beat it look identical from the outside unless the refusal is written down. A refused
    baseline satisfies the mandatory-kind check and carries no distribution, so it can never be
    scored as though it had made a call.
    """

    id: str
    kind: BaselineKind
    distribution: Distribution | None = None
    method: str = ""
    detail: str = ""
    refused: str = ""

    def __post_init__(self) -> None:
        if self.distribution is None and not self.refused.strip():
            raise ForecastError(
                f"baseline {self.id!r} carries neither a distribution nor a refusal. A baseline "
                f"that produced nothing has to say why, or it is indistinguishable from one that "
                f"was quietly dropped because it was winning."
            )

    @property
    def is_scored(self) -> bool:
        return self.distribution is not None

    def render(self) -> str:
        distribution = self.distribution
        if distribution is None:
            return f"{self.id:<32} refused: {self.refused}"
        return f"{self.id:<32} {distribution.render()}  {self.detail}"

    def __canonical__(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "distribution": None
            if self.distribution is None
            else self.distribution.__canonical__(),
            "method": self.method,
            "refused": self.refused,
        }


# ---------------------------------------------------------------------------
# Lead time, in fractions of a fitted transition width
# ---------------------------------------------------------------------------
#
# There is nothing here, and that is deliberate. Lead time is reported in fractions of a fitted
# transition width, instrument H4 owns `run.transition_width`, and a second fit would be a second
# answer to one question. `reward_lens.measure.rate.transition` holds the fit, the `TransitionFit`
# it produces, and the `LeadTime` that divides by it; `forecast.score.transition_fit` and
# `forecast.score.forecast_lead_time` call through to them and this package re-exports their types
# so a forecast user meets one lead-time object rather than two.
#
# An earlier draft of this module carried its own logistic fallback, written before H4 landed. It
# was measured against H4's on the same planted logistic of width 21.9722 steps: the fallback
# returned 23.11 clean and 62.80 at noise sd 0.03, and H4 returned 21.9722 and 21.955. It was
# deleted rather than kept behind a flag, because the one thing worse than no width unit is two.


# ---------------------------------------------------------------------------
# The forecast
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Forecast:
    """One frozen prediction, plus what it takes to enforce it.

    Construct with `barrier.issue`, never directly. The direct constructor still enforces the
    mandatory baselines, because a check that can be dodged by calling the dataclass is not a check,
    but it cannot enforce the information barrier: the barrier needs the evidence store the inputs
    live in, and only `issue` takes one.

    Two fields the prior design did not have and both earn their place. ``reference_class``, because
    calibration is meaningless without one and because climatology is the base rate *in it*.
    ``decision``, because a forecast without an action is a number, and decision value is the number
    that answers "so what".
    """

    id: ForecastID
    target: QuantityID
    subject: SubjectRef
    resolution: ResolutionRule
    issued_at: InformationTime
    horizon: HorizonSpec
    reference_class: ReferenceClass
    distribution: Distribution
    method: str
    inputs: tuple[EvidenceID, ...]
    baselines: tuple[BaselineForecast, ...]
    decision: DecisionSpec | None = None
    meta_plan: str | None = None
    provenance: Provenance = field(default_factory=Provenance)
    #: Where in the run this was issued, when the subject is a run. Kept beside `issued_at` rather
    #: than folded into it, because they are different clocks and the barrier reads only the clock.
    issued_step: int | None = None

    def __post_init__(self) -> None:
        if not self.method.strip():
            raise ForecastError(
                f"forecast {self.id} names no method. The method is what a replication reads, and "
                f"'the model said so' is a method if you write it down."
            )
        missing = MANDATORY_BASELINE_KINDS - {b.kind for b in self.baselines}
        if missing:
            names = ", ".join(sorted(k.value for k in missing))
            raise ForecastError(
                f"forecast {self.id} on {self.target!r} carries no {names} baseline, and section "
                f"7.3 makes all four mandatory. Run the comparator, or attach a "
                f"BaselineForecast(kind=..., refused='why it could not run') so the gap is on the "
                f"record instead of being invisible. A forecast that never ran the black-box "
                f"comparator and one that ran it and won look identical from the outside."
            )

    @property
    def scored_baselines(self) -> tuple[BaselineForecast, ...]:
        return tuple(b for b in self.baselines if b.is_scored)

    @property
    def refused_baselines(self) -> tuple[BaselineForecast, ...]:
        return tuple(b for b in self.baselines if not b.is_scored)

    def baseline(self, kind: BaselineKind) -> BaselineForecast:
        for b in self.baselines:
            if b.kind is kind:
                return b
        raise KeyError(f"forecast {self.id} declares no {kind.value} baseline")

    def with_provenance(self, provenance: Provenance) -> "Forecast":
        return replace(self, provenance=provenance)

    def render(self) -> str:
        lines = [
            f"{self.id}",
            f"    target      {self.target}",
            f"    issued      {self.issued_at}",
            f"    horizon     {self.horizon.render()}",
            f"    class       {self.reference_class.id}  {self.reference_class.definition}",
            f"    rule        {self.resolution.render()}",
            f"    forecast    {self.distribution.render()}   [{self.method}]",
            f"    inputs      {len(self.inputs)} evidence rows",
        ]
        if self.decision is not None:
            lines.append(f"    decision    {self.decision.render()}")
        for b in self.baselines:
            lines.append(f"    baseline    {b.render()}")
        return "\n".join(lines)

    def __canonical__(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target": self.target,
            "subject": self.subject.__canonical__(),
            "resolution": self.resolution.__canonical__(),
            "issued_at": self.issued_at.__canonical__(),
            "horizon": self.horizon.__canonical__(),
            "reference_class": self.reference_class.__canonical__(),
            "distribution": self.distribution.__canonical__(),
            "method": self.method,
            "inputs": list(self.inputs),
            "baselines": [b.__canonical__() for b in self.baselines],
            "decision": None if self.decision is None else self.decision.__canonical__(),
            "meta_plan": self.meta_plan,
            "issued_step": self.issued_step,
        }


def forecast_id(
    *,
    target: QuantityID,
    subject: SubjectRef,
    resolution: ResolutionRule,
    issued_at: InformationTime,
    distribution: Distribution,
    inputs: Sequence[EvidenceID],
    method: str,
) -> ForecastID:
    """A content-derived forecast id over what makes two forecasts the same forecast.

    The information time is in the material, deliberately. The same call made from the same inputs
    at two different instants is two forecasts, because the second one was made by somebody who had
    had longer to look, and collapsing them would let a forecast be re-issued after the fact under
    the id of the earlier one.

    The baselines and the decision spec are not in it. Both are attached to a call rather than being
    part of what the call says, so adding the black-box comparator you forgot to run does not change
    which prediction you made.
    """
    return ForecastID(
        content_hash(
            {
                "target": target,
                "subject": subject.__canonical__(),
                "resolution": resolution.__canonical__(),
                "issued_at": issued_at.__canonical__(),
                "distribution": distribution.__canonical__(),
                "inputs": sorted(inputs),
                "method": method,
            },
            "fc",
        )
    )


__all__ = [
    "BaselineForecast",
    "BaselineKind",
    "BinaryProbability",
    "Comparator",
    "DecisionSpec",
    "Distribution",
    "Forecast",
    "ForecastError",
    "ForecastID",
    "HorizonSpec",
    "InformationTime",
    "IntervalForecast",
    "MANDATORY_BASELINE_KINDS",
    "QuantileForecast",
    "ReferenceClass",
    "ReferenceClassID",
    "ResolutionRule",
    "forecast_id",
]

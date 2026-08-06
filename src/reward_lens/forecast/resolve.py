"""Resolution: `Resolved` or `Void`, and an expired forecast is never a miss.

The only interesting decision in this module is the one it exists for. A forecast whose metric never
arrived, or whose horizon passed with the question still open, has **not** been got wrong. Scoring
it as a miss is a specific and common way to make a ledger say something false, and it is false in
whichever direction the missing metric happened to be pointing: a card whose analysis crashed on the
run where the effect was largest scores as a confident wrong call, and a card whose analysis crashed
on a null scores as a lucky one.

So resolution has three outcomes and not two, the third carries a reason, and the reason is on the
ledger row: a missing metric at adjudication is `VOID` with a named reason, and a kill criterion
whose metric is absent is `VOID`, never a non-firing.

The campaign is the worked case and it is why the count matters. Twenty-three directional calls were
registered; sixteen resolved and seven went void, and one of the seven carried the campaign's most
confident call at 0.9. Scoring those seven as misses would have moved the published Brier a long way
and every step of the move would have been a lie about what was measured.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from reward_lens.core.evidence import register_payload
from reward_lens.forecast.schema import (
    Forecast,
    ForecastID,
    InformationTime,
    IntervalForecast,
    QuantileForecast,
)


class VoidReason(enum.Enum):
    """Why a forecast could not be scored. Not a `RefusalReason`, and deliberately not.

    A `Refusal` is an instrument declining to measure. This is a forecast that was made honestly and
    cannot be graded, which is a different event with a different remedy and a different place in
    the ledger: a refusal never enters the score at all, and a void enters it as a counted
    non-resolution that the denominator has to show.
    """

    #: The metric the rule names was not produced by the analysis.
    METRIC_ABSENT = "metric_absent"
    #: The horizon passed and the question is still open.
    EXPIRED = "expired"
    #: The run the forecast is about is itself void (an arm diverged, the record is unreadable).
    SUBJECT_VOID = "subject_void"
    #: The metric arrived and is not a number: NaN, a string, an error marker.
    METRIC_UNEVALUABLE = "metric_unevaluable"
    #: The analysis reported the question as inconclusive under its own registered criteria.
    INCONCLUSIVE = "inconclusive"


VOID_MEANING: dict[VoidReason, str] = {
    VoidReason.METRIC_ABSENT: (
        "The resolution rule names a metric the analysis did not produce, so there is nothing to "
        "compare against the threshold. Scoring this as a miss would score the analysis, not the "
        "forecast."
    ),
    VoidReason.EXPIRED: (
        "The horizon passed with the question still open. An expired forecast is void, never a "
        "miss: the forecaster said 'by then', and 'not yet' is not 'no'."
    ),
    VoidReason.SUBJECT_VOID: (
        "The subject of the forecast is itself void, so no outcome exists to compare with. A "
        "forecast about a run whose record is unreadable has no answer, correct or otherwise."
    ),
    VoidReason.METRIC_UNEVALUABLE: (
        "The metric arrived and is not a number. A NaN compared against a threshold returns False "
        "under every comparator, which is a silent miss rather than a missing value."
    ),
    VoidReason.INCONCLUSIVE: (
        "The analysis reported the question as inconclusive under its own registered criteria. The "
        "forecast is not wrong; the experiment did not decide."
    ),
}


@register_payload
@dataclass(frozen=True)
class Resolved:
    """A forecast graded against its own frozen rule, with the number that decided it."""

    forecast_id: ForecastID
    outcome: bool
    metric: str
    metric_value: float
    rule: str
    resolved_at: InformationTime
    #: For an interval forecast: whether the realised value fell inside it.
    covered: bool | None = None
    detail: str = ""

    @property
    def is_void(self) -> bool:
        return False

    def render(self) -> str:
        verdict = "CONFIRMED" if self.outcome else "REFUTED"
        cover = "" if self.covered is None else f"  interval {'hit' if self.covered else 'miss'}"
        return (
            f"{self.forecast_id}  {verdict}  {self.rule}  observed "
            f"{self.metric}={self.metric_value:.6g}{cover}"
        )

    def __canonical__(self) -> dict[str, Any]:
        return {
            "forecast_id": self.forecast_id,
            "outcome": self.outcome,
            "metric": self.metric,
            "metric_value": self.metric_value,
            "rule": self.rule,
            "resolved_at": self.resolved_at.__canonical__(),
            "covered": self.covered,
        }


@register_payload
@dataclass(frozen=True)
class Void:
    """A forecast that cannot be graded, with the reason on the record."""

    forecast_id: ForecastID
    reason: VoidReason
    detail: str
    rule: str
    resolved_at: InformationTime
    statistics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_void(self) -> bool:
        return True

    @property
    def meaning(self) -> str:
        return VOID_MEANING[self.reason]

    def render(self) -> str:
        return f"{self.forecast_id}  VOID ({self.reason.value})  {self.rule}  {self.detail}"

    def __canonical__(self) -> dict[str, Any]:
        return {
            "forecast_id": self.forecast_id,
            "reason": self.reason.value,
            "detail": self.detail,
            "rule": self.rule,
            "resolved_at": self.resolved_at.__canonical__(),
        }


#: What `resolve` returns. Statically `Resolved | Void`; kept as a name so call sites read right.
Resolution = Any


def resolve(
    forecast: Forecast,
    metrics: Mapping[str, float],
    *,
    at: InformationTime,
    subject_void: str = "",
    inconclusive: str = "",
) -> Resolution:
    """Grade a forecast against its own frozen rule, or void it with a reason.

    Order matters and it is the order of how little the forecast had to do with the outcome. A void
    subject beats everything: if the run is unreadable there is no question. An analysis that
    declared itself inconclusive beats a missing metric, because it is a stronger statement about
    the same gap. An expired horizon beats a present metric, because a metric that arrived after the
    forecast's own deadline is not what the forecast was about.

    ``at`` is the information time of the resolution, not of the forecast, and the two are separate
    on purpose: the gap between them is the realised lead time, and it is the only quantity in this
    module that a reader is likely to care about more than the verdict.
    """
    rule = forecast.resolution
    rendered = rule.render()

    if subject_void:
        return Void(
            forecast_id=forecast.id,
            reason=VoidReason.SUBJECT_VOID,
            detail=subject_void,
            rule=rendered,
            resolved_at=at,
        )
    if inconclusive:
        return Void(
            forecast_id=forecast.id,
            reason=VoidReason.INCONCLUSIVE,
            detail=inconclusive,
            rule=rendered,
            resolved_at=at,
        )
    expiry = forecast.horizon.expires_at
    if expiry is not None and at > expiry and rule.metric not in metrics:
        return Void(
            forecast_id=forecast.id,
            reason=VoidReason.EXPIRED,
            detail=(
                f"the horizon expired at {expiry.instant} and {rule.metric!r} had still not been "
                f"produced when resolution was attempted at {at.instant}"
            ),
            rule=rendered,
            resolved_at=at,
            statistics={"expires_at": expiry.instant, "attempted_at": at.instant},
        )

    if rule.metric not in metrics:
        return Void(
            forecast_id=forecast.id,
            reason=VoidReason.METRIC_ABSENT,
            detail=(
                f"the analysis produced no {rule.metric!r}. Available metrics: "
                f"{', '.join(sorted(metrics)) or 'none'}"
            ),
            rule=rendered,
            resolved_at=at,
            statistics={"available": sorted(metrics)},
        )

    outcome = rule.evaluate(metrics)
    if outcome is None:
        return Void(
            forecast_id=forecast.id,
            reason=VoidReason.METRIC_UNEVALUABLE,
            detail=(
                f"{rule.metric!r} is {metrics[rule.metric]!r}, which is not a number. A NaN "
                f"compared against a threshold is False under every comparator, so this would have "
                f"scored as a confident miss."
            ),
            rule=rendered,
            resolved_at=at,
            statistics={"value": repr(metrics[rule.metric])},
        )

    value = float(metrics[rule.metric])
    covered = None
    distribution = forecast.distribution
    if isinstance(distribution, (IntervalForecast, QuantileForecast)):
        covered = bool(distribution.covers(value))
    return Resolved(
        forecast_id=forecast.id,
        outcome=bool(outcome),
        metric=rule.metric,
        metric_value=value,
        rule=rendered,
        resolved_at=at,
        covered=covered,
    )


def partition(resolutions: Sequence[Resolution]) -> tuple[tuple[Resolved, ...], tuple[Void, ...]]:
    """Split a batch into the graded and the void. The denominator lives in the second one."""
    resolved = tuple(r for r in resolutions if isinstance(r, Resolved))
    void = tuple(r for r in resolutions if isinstance(r, Void))
    return resolved, void


__all__ = [
    "VOID_MEANING",
    "Resolution",
    "Resolved",
    "Void",
    "VoidReason",
    "partition",
    "resolve",
]

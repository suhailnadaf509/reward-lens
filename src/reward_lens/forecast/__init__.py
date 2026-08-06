"""Forecasting: the layer that makes every other measurement worth something.

A measurement that cannot support a claim about the future cannot support a decision, so this is
not a section of the science layer. It is what the rest of the library is for.

The competitive position is narrow and it is real. Prospective monitoring of reinforcement-learning
internals is a small live subfield with at least four methods, so "we watch internals early" is not
a differentiator. But across every method read in the scan, **not one fits its predictor strictly
before the outcome it predicts.** So the position is temporal validity, enforced by type, with
baselines and a public score.

Five things, in the order you meet them:

`issue` refuses to build a forecast whose inputs, transitively, postdate it. No override.

Four baselines are mandatory and a forecast without them cannot be constructed: climatology,
persistence, the dumb statistic, and the scaffolded black-box method, which currently means
contrastive belief-flipping and is the first thing a reviewer will raise.

`RunCorpus` splits over runs and time, never over items, because splitting over items is what makes
a temporal claim mean nothing.

`resolve` returns `Resolved` or `Void`, and an expired or unresolvable forecast is void with a
reason, never a miss.

`CalibrationLedger` is append-only and prints its own worst result at the top.

    from reward_lens.forecast import issue, resolve, CalibrationLedger, entry_from

    forecast = issue(target, subject, rule, distribution, inputs, at=now, store=store, ...)
    outcome = resolve(forecast, metrics, at=later)
    ledger.append(entry_from(forecast, outcome))
    print(ledger.render())
"""

from reward_lens.forecast.barrier import (
    ForecastLeakageError,
    ancestry,
    check_barrier,
    information_time_of,
    issue,
)
from reward_lens.forecast.baselines import (
    BELIEF_FLIP_VERSION,
    BeliefFlipReading,
    RecordsTest,
    belief_flip_hash,
    climatology,
    climatology_interval,
    contrastive_belief_flip,
    dumb_statistic,
    dumb_statistic_from_gradnorm,
    harmonic,
    persistence,
    persistence_rate,
    records_null_forecast,
    records_test,
    render_belief_flip,
)
from reward_lens.forecast.corpus import (
    AISI_TRAPS,
    Fold,
    RunCorpus,
    SplitKind,
    corpus_from_runs,
    iter_folds,
)
from reward_lens.forecast.ledger import (
    FORECAST_BRIER_QUANTITY,
    PUBLISHED_COIN_BRIER,
    PUBLISHED_DIRECTIONAL_BRIER,
    PUBLISHED_INTERVAL_COVERAGE,
    REGISTERED_NOMINAL_COVERAGE,
    CalibrationLedger,
    ForecastCalibration,
    LedgerEntry,
    LedgerScore,
    entry_from,
)
from reward_lens.forecast.resolve import (
    Resolution,
    Resolved,
    Void,
    VoidReason,
    partition,
    resolve,
)
from reward_lens.forecast.schema import (
    MANDATORY_BASELINE_KINDS,
    BaselineForecast,
    BaselineKind,
    BinaryProbability,
    Comparator,
    DecisionSpec,
    Distribution,
    Forecast,
    ForecastError,
    ForecastID,
    HorizonSpec,
    InformationTime,
    IntervalForecast,
    QuantileForecast,
    ReferenceClass,
    ReferenceClassID,
    ResolutionRule,
    forecast_id,
)
from reward_lens.forecast.score import (
    CoverageScore,
    DecisionValue,
    MurphyDecomposition,
    ReliabilityDiagram,
    SkillScore,
    brier,
    brier_terms,
    coverage_score,
    decision_value,
    forecast_lead_time,
    log_score,
    murphy_decomposition,
    recalibrate,
    reliability_diagram,
    skill_score,
    transition_fit,
)

# Lead time is reported in fractions of a fitted transition width and instrument H4 owns that fit,
# so its two types are re-exported here rather than reimplemented. A forecast user meets one
# `LeadTime` and one `TransitionFit`, which is the point: two would be two width units.
from reward_lens.measure.rate.transition import LeadTime, TransitionFit  # noqa: E402

__all__ = [
    "AISI_TRAPS",
    "BELIEF_FLIP_VERSION",
    "FORECAST_BRIER_QUANTITY",
    "MANDATORY_BASELINE_KINDS",
    "PUBLISHED_COIN_BRIER",
    "PUBLISHED_DIRECTIONAL_BRIER",
    "PUBLISHED_INTERVAL_COVERAGE",
    "REGISTERED_NOMINAL_COVERAGE",
    "BaselineForecast",
    "BaselineKind",
    "BeliefFlipReading",
    "BinaryProbability",
    "CalibrationLedger",
    "Comparator",
    "CoverageScore",
    "DecisionSpec",
    "DecisionValue",
    "Distribution",
    "Fold",
    "Forecast",
    "ForecastCalibration",
    "ForecastError",
    "ForecastID",
    "ForecastLeakageError",
    "HorizonSpec",
    "InformationTime",
    "IntervalForecast",
    "LeadTime",
    "LedgerEntry",
    "LedgerScore",
    "MurphyDecomposition",
    "QuantileForecast",
    "RecordsTest",
    "ReferenceClass",
    "ReferenceClassID",
    "ReliabilityDiagram",
    "Resolution",
    "ResolutionRule",
    "Resolved",
    "RunCorpus",
    "SkillScore",
    "SplitKind",
    "TransitionFit",
    "Void",
    "VoidReason",
    "ancestry",
    "belief_flip_hash",
    "brier",
    "brier_terms",
    "check_barrier",
    "climatology",
    "climatology_interval",
    "contrastive_belief_flip",
    "corpus_from_runs",
    "coverage_score",
    "decision_value",
    "dumb_statistic",
    "dumb_statistic_from_gradnorm",
    "entry_from",
    "forecast_id",
    "harmonic",
    "information_time_of",
    "issue",
    "iter_folds",
    "forecast_lead_time",
    "transition_fit",
    "log_score",
    "murphy_decomposition",
    "partition",
    "persistence",
    "persistence_rate",
    "records_null_forecast",
    "records_test",
    "recalibrate",
    "reliability_diagram",
    "render_belief_flip",
    "resolve",
    "skill_score",
]

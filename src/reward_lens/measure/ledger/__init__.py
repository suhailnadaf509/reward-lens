"""The Price ledger (F1) and the selection-explained fraction (F2).

Four instruments over one identity. For any measurable feature `f` of a rollout, with
`z(θ) = E_{y~π_θ}[f(y)]`, a Fisher-preconditioned step gives, to first order,

    Δz_i = η · Cov_group(A, f_i) + ρ_i

`SelectionTerm` and `SelectionResidual` report the two halves per feature per step, which is F1.
`SelectionExplainedFraction` and `EffectiveStepSize` report the fit of the left side on the right
across steps, which is F2's `Λ` and `η_eff`.

**This is a measurement instrument on behavioural traits in a live run, not a derivation of an
update rule in parameter space.** Frank's Price partition of optimiser updates (arXiv 2507.18549) is
the nearest prior art and is a different use of the same equation; the full statement of the
distinction is at the top of `measure.ledger.price` and it belongs on page one rather than in a
footnote.

`Λ` is the validity certificate for every other Level 1 claim in the library. `RegimeCondition.
LINEAR_RESPONSE` names `selection.explained_fraction` as the quantity that measures it, so F1's own
envelope cannot be satisfied until F2 has run, and neither can the envelope of anything else that
expands to first order about the current step.

Both instruments need a record and a featuriser and nothing else, so they run at `RECORD` access on
a training run somebody else did. `measure.ledger.features` is the featuriser contract and a
surface bank that works on any record carrying turn text; `measure.ledger.labelled` adapts a
published per-rollout table, which is what puts a labelled series inside reach.
"""

from reward_lens.measure.ledger.explained import (
    EXPLAINED_ENVELOPE,
    EffectiveStepSize,
    LambdaFit,
    SelectionExplainedFraction,
    feature_scales,
    fit_lambda,
    lambda_by_step,
)
from reward_lens.measure.ledger.features import (
    RecordedFeatures,
    SurfaceFeatures,
    TrajectoryFeaturiser,
    assistant_text,
    matrix_of,
)
from reward_lens.measure.ledger.nulls import (
    NullResult,
    permuted_advantage_null,
    permuted_step_null,
    random_feature_null,
    summarise,
)
from reward_lens.measure.ledger.price import (
    LEDGER_ACCESS,
    LEDGER_ENVELOPE,
    Differential,
    LedgerRow,
    SelectionResidual,
    SelectionTerm,
    StepLedger,
    StepSample,
    advantages_from_rewards,
    learning_rates,
    ledger_between,
    ledger_series,
    selection_differential,
    steps_from_run,
)

__all__ = [
    "Differential",
    "EXPLAINED_ENVELOPE",
    "EffectiveStepSize",
    "LEDGER_ACCESS",
    "LEDGER_ENVELOPE",
    "LambdaFit",
    "LedgerRow",
    "NullResult",
    "RecordedFeatures",
    "SelectionExplainedFraction",
    "SelectionResidual",
    "SelectionTerm",
    "StepLedger",
    "StepSample",
    "SurfaceFeatures",
    "TrajectoryFeaturiser",
    "advantages_from_rewards",
    "assistant_text",
    "feature_scales",
    "fit_lambda",
    "lambda_by_step",
    "ledger_between",
    "ledger_series",
    "learning_rates",
    "matrix_of",
    "permuted_advantage_null",
    "permuted_step_null",
    "random_feature_null",
    "selection_differential",
    "steps_from_run",
    "summarise",
]

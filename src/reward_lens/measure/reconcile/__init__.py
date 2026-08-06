"""F4, the reconciliation residual budgeted, and F6, the Lande slope.

The four books give two independent predictions of the same thing. The effect book measures
`Δz_obs`. Cause and capacity together predict `Δz_pred = η·G·C⁻¹·S`. The difference is the
reconciliation residual, and it is not noise: it is a budget with named terms, and the
question is whether `Var(ρ)` is accounted for by `Σ u_i²`. A closed budget characterises the
instrument, an open one names an unmodelled term, and either outcome is publishable.

F6 regresses the observed response on the predicted one across a window. Lande's equation holds at
slope 1 and a slope near zero retires Level 1, which makes it the load-bearing assumption of the
whole story and the one worth testing rather than assuming.

**What this package does not compute.** `G` is `measure.efficiency`'s book and arrives as an
argument; `Δz`, `S` and `η` are `measure.ledger`'s and are read from it. The join key is the feature
basis and it is `StepSample.names`, compared element for element. Everything here
is the arithmetic between those books plus the uncertainty budget, and the budget machinery is
`core.budget`'s.

**The scope limit that decides how to read any number from it.** With `G` at the rung-0 covariance
bound, `G = C` makes `Gβ = S` exactly: F4's residual reduces to F1's and F6's slope reduces to F2's
`η_eff`. Both readings carry that as a field rather than as a caveat, and the independent content of
both needs a rung-2 Fisher `G` at `POLICY: BACKWARD`.
"""

from reward_lens.measure.reconcile.books import (
    BasisMismatch,
    BookRow,
    CostConsistency,
    FeatureCovariance,
    MetricGLike,
    SelectionGradient,
    StepCostLike,
    StepReconciliation,
    cost_consistency,
    reconcile_series,
    selection_gradient,
    within_group_covariance,
)
from reward_lens.measure.reconcile.closure import (
    MIN_CLUSTERS,
    ClosureResult,
    FeatureClosure,
    closure_of,
)
from reward_lens.measure.reconcile.facts import Absent, RunFacts, facts_from_run
from reward_lens.measure.reconcile.instruments import (
    LANDE_ENVELOPE,
    RECONCILE_ACCESS,
    RECONCILE_ENVELOPE,
    BudgetClosure,
    LandeSlope,
    Reconciliation,
    ReconciliationResidual,
)
from reward_lens.measure.reconcile.lande import (
    CircularEstimator,
    LandeFit,
    fit_lande,
    permuted_lande_null,
)
from reward_lens.measure.reconcile.prediction import (
    BUDGET_CLOSURE_SPEC,
    LANDE_SLOPE_SPEC,
    ClosureResolution,
    LandeResolution,
    freeze_closure,
    freeze_lande,
    score_closure,
    score_lande,
)
from reward_lens.measure.reconcile.residual import (
    TERM_ORDER,
    FeatureBudget,
    MissingTerm,
    advantage_r_squared,
    itemise,
)

__all__ = [
    "BUDGET_CLOSURE_SPEC",
    "LANDE_ENVELOPE",
    "LANDE_SLOPE_SPEC",
    "MIN_CLUSTERS",
    "RECONCILE_ACCESS",
    "RECONCILE_ENVELOPE",
    "TERM_ORDER",
    "Absent",
    "BasisMismatch",
    "BookRow",
    "BudgetClosure",
    "CircularEstimator",
    "ClosureResolution",
    "ClosureResult",
    "CostConsistency",
    "FeatureBudget",
    "FeatureClosure",
    "FeatureCovariance",
    "LandeFit",
    "LandeResolution",
    "LandeSlope",
    "MetricGLike",
    "MissingTerm",
    "Reconciliation",
    "ReconciliationResidual",
    "RunFacts",
    "SelectionGradient",
    "StepCostLike",
    "StepReconciliation",
    "advantage_r_squared",
    "closure_of",
    "cost_consistency",
    "facts_from_run",
    "fit_lande",
    "freeze_closure",
    "freeze_lande",
    "itemise",
    "permuted_lande_null",
    "reconcile_series",
    "score_closure",
    "score_lande",
    "selection_gradient",
    "within_group_covariance",
]

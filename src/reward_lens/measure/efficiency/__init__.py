"""F3, the cost book: the KL budget, the efficiency, and where the nats went.

Three objects and two entry points, over an interface fixed in advance so that
`measure.reconcile`'s reconciliation could be written against it.

    from reward_lens.measure.efficiency import metric_g, cost_series, MetricG, StepCost

`metric_g` estimates `G = J F⁻¹ Jᵀ`, the behavioural covariance a parameter move can reach.
`cost_series` divides each step's `KL_min = ½ Δzᵀ G⁻¹ Δz` by what the step spent and attributes the
result to named features. When the record carries no per-step KL, which is every `beta = 0` run,
`cost_series` refuses with `RECORD_INCOMPLETE` and `kl_min_series` returns the half that needs no
denominator.

The feature basis is the join key and it is `measure.ledger.price.StepSample.names`, whole and in
order. `G`, the ledger's `Δz` and this package's shares are vectors in that one basis, so all three
come from one `TrajectoryFeaturiser` and none of them invents one.

`measure.efficiency.scores` is the only module here that imports torch, and it does so lazily.
Importing this package does not.
"""

from reward_lens.measure.efficiency.cost import (
    COST_BASELINES,
    COST_ENVELOPE,
    MAX_EXACT_FEATURES,
    NO_DENOMINATOR_REMEDY,
    StepCost,
    StepKlMin,
    UpdateEfficiency,
    UpdateKLMin,
    UpdateKLShare,
    UpdateKLSpent,
    cost_series,
    kl_min_series,
    kl_spent_from_record,
    noise_floor,
    shapley_shares,
)
from reward_lens.measure.efficiency.metric import (
    DEFAULT_DAMPING,
    DEFAULT_STABILITY_TOL,
    RANK_TOLERANCE,
    MetricG,
    metric_g,
    pooled_rollouts,
)

__all__ = [
    "COST_BASELINES",
    "COST_ENVELOPE",
    "DEFAULT_DAMPING",
    "DEFAULT_STABILITY_TOL",
    "MAX_EXACT_FEATURES",
    "NO_DENOMINATOR_REMEDY",
    "RANK_TOLERANCE",
    "MetricG",
    "StepCost",
    "StepKlMin",
    "UpdateEfficiency",
    "UpdateKLMin",
    "UpdateKLShare",
    "UpdateKLSpent",
    "cost_series",
    "kl_min_series",
    "kl_spent_from_record",
    "metric_g",
    "noise_floor",
    "pooled_rollouts",
    "shapley_shares",
]

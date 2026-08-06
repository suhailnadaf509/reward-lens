"""Four-dimensional access resolution and the capability report.

Which instrument applies is decided by four independent things: what you can touch, when you are
asking, what kind of thing the grader is, and whether the estimator's assumptions are live on this
run. All four live in the type system. This package resolves all four from what a user
actually has on disk, and turns the result into the one output most users will ever see.

Four modules, in the order the report consumes them:

``matrix``
    The access matrix, resolved by probing. A grader endpoint that accepts a seed and returns
    different values for different seeds has `REPLICATE`; one that does not has `QUERY` only, and
    the difference decides whether variance decomposition exists at all. Probing costs calls, so it
    is opt-in and budgeted, and the default resolution makes none.
``substrate``
    Which of the six substrates the grader is, by walking its score tree. `COMPOSITE` is the
    frontier default, so walking is the normal case rather than the hard one.
``phase``
    When the question is being asked. Small, and deliberately blind to the access matrix, because
    phase is not derivable from access and the attempt is what makes an auditor and a lab
    indistinguishable.
``report``
    The four sections of the report, rendered, with a remedy on every refusal.

The package imports no torch and touches no network unless a probe budget authorises it.
"""

from reward_lens.access.matrix import (
    DRY,
    PROBE_ITEM,
    SEED_PROBE_CALLS,
    AccessResolution,
    ComponentAccess,
    ProbeBudget,
    ProbeResult,
    RecordFacts,
    ScoreEndpoint,
    http_endpoint,
    probe_endpoint,
    record_facts,
    render_access,
    resolution_from_matrix,
    resolve_access,
)
from reward_lens.access.phase import PhaseResolution, resolve_phase
from reward_lens.access.report import (
    REMEDY_FOR_CONDITION,
    Assessment,
    CapabilityReport,
    CatalogueInstrument,
    NotBuilt,
    capability_report,
    load_instrument_catalogue,
)
from reward_lens.access.substrate import (
    LeafReading,
    ScoreNode,
    SubstrateReading,
    classify_leaf,
    classify_substrate,
    is_score_node,
)

__all__ = [
    "DRY",
    "PROBE_ITEM",
    "REMEDY_FOR_CONDITION",
    "SEED_PROBE_CALLS",
    "AccessResolution",
    "Assessment",
    "CapabilityReport",
    "CatalogueInstrument",
    "ComponentAccess",
    "LeafReading",
    "NotBuilt",
    "PhaseResolution",
    "ProbeBudget",
    "ProbeResult",
    "RecordFacts",
    "ScoreEndpoint",
    "ScoreNode",
    "SubstrateReading",
    "capability_report",
    "classify_leaf",
    "classify_substrate",
    "http_endpoint",
    "is_score_node",
    "load_instrument_catalogue",
    "probe_endpoint",
    "record_facts",
    "render_access",
    "resolution_from_matrix",
    "resolve_access",
    "resolve_phase",
]

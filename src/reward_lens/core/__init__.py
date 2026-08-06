"""``reward_lens.core`` — types, evidence, provenance, the gates, and the store.

This package has no model dependencies. Everything above it imports it, and it imports nothing
heavier than numpy and pydantic, so ``import reward_lens.core`` is fast and torch-free. It is the
frozen contract of the kernel: the identity types, the Evidence atom, the three gates, and the
append-only store. Changing anything here is an ADR-level decision.
"""

from __future__ import annotations

from reward_lens.core.closure import (
    ArcSpec,
    ClosureError,
    ClosureReport,
    CostBudget,
    Demand,
    Gap,
    GapKind,
    MetricBinding,
    Output,
    analyse,
    on,
    subject_key,
)
from reward_lens.core.config import Settings, get_settings, set_settings
from reward_lens.core.errors import (
    CalibrationWarning,
    CapabilityError,
    ConformanceError,
    DataError,
    GaugeError,
    NumericsError,
    ProvenanceError,
    RegistryError,
    RewardLensError,
)
from reward_lens.core.evidence import (
    Evidence,
    Uncertainty,
    ValueCodec,
    evidence_from_envelope,
    make_evidence,
    register_payload,
)
from reward_lens.core.gates import (
    CalibrationRef,
    compute_trust,
    require_frame_for_comparison,
)
from reward_lens.core.provenance import Cost, Provenance, capture_provenance, git_sha
from reward_lens.core.registry import (
    CARD_SECTIONS,
    DATASETS,
    INTERVENTIONS,
    OBSERVABLES,
    ORACLES,
    ORGANISMS,
    SIGNALS,
    Registry,
)
from reward_lens.core.store import EvidenceStore, default_store, set_default_store
from reward_lens.core.types import (
    Capability,
    DatasetID,
    DirectionID,
    EvidenceID,
    FrameID,
    GaugeStatus,
    ModelFP,
    OrganismID,
    Site,
    Span,
    StudyID,
    SubjectRef,
    TrustLevel,
    content_hash,
    hash_bytes,
)

__all__ = [
    # plan closure
    "ArcSpec",
    "ClosureError",
    "ClosureReport",
    "CostBudget",
    "Demand",
    "Gap",
    "GapKind",
    "MetricBinding",
    "Output",
    "analyse",
    "on",
    "subject_key",
    # types
    "Capability",
    "TrustLevel",
    "GaugeStatus",
    "Site",
    "Span",
    "SubjectRef",
    "ModelFP",
    "DatasetID",
    "DirectionID",
    "FrameID",
    "EvidenceID",
    "StudyID",
    "OrganismID",
    "content_hash",
    "hash_bytes",
    # evidence
    "Evidence",
    "Uncertainty",
    "make_evidence",
    "evidence_from_envelope",
    "register_payload",
    "ValueCodec",
    # gates
    "CalibrationRef",
    "compute_trust",
    "require_frame_for_comparison",
    # provenance
    "Provenance",
    "Cost",
    "capture_provenance",
    "git_sha",
    # store
    "EvidenceStore",
    "default_store",
    "set_default_store",
    # registry
    "Registry",
    "SIGNALS",
    "OBSERVABLES",
    "INTERVENTIONS",
    "ORGANISMS",
    "DATASETS",
    "ORACLES",
    "CARD_SECTIONS",
    # config
    "Settings",
    "get_settings",
    "set_settings",
    # errors
    "RewardLensError",
    "CapabilityError",
    "GaugeError",
    "CalibrationWarning",
    "ConformanceError",
    "ProvenanceError",
    "RegistryError",
    "DataError",
    "NumericsError",
]

# The registry loads at import, so `QUANTITIES` is populated for anyone who has imported anything
# from this package. The lint rule ("an Instrument whose quantity is not registered fails
# at import") is only meaningful if the registry is there to check against at import; a lazy load
# means the first thing to look is the thing that decides, which is import order deciding a
# contract. Registration is idempotent and the catalogue is a packaged JSON read once, so the cost
# is one file read.
from reward_lens.core.quantity import load_quantities as _load_quantities

_LOAD_REPORT = _load_quantities()

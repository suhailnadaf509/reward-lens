"""Series L, reference materials and label metrology.

The layer that says whether the answer key is any good. Nothing in series C or K means anything
without it, and every prior version of this project assumed it away.

Five instruments, and the thread running through them is that a measurement against an answer key is
only as good as the answer key, which nobody measures:

    L1  `reference.u_characterisation`, `reference.u_homogeneity`, `reference.u_stability`
        A planted organism is a reference material and it ships with no uncertainty on its assigned
        dose. This measures the three terms an analytical laboratory would require, and reports
        `None` where nobody has measured, which caps the trust ladder at `CALIBRATED`.
    L2  `labels.error_rate`, `labels.score_ceiling`
        What fraction of a label set is wrong, and the score a perfect model is therefore capped at.
    L3  `labels.false_success_rate`, `labels.corrupt_success_rate`
        A programmatic verifier wrong in both directions on the same benchmark.
    L4  `labels.fs_signal_locality`
        Whether the tell is in the closing message or across the episode. The residual-stream
        version is registered as an unimplemented rung, which is what makes it nameable.
    L5  `labels.position_prior`
        A localiser scored against the position distribution it actually faces, rather than against
        a uniform prior over positions.

`register_all` populates the estimator ladder for every quantity in the series. It is idempotent and
it is not called at import: registering a global side effect on import is how a registry ends up
depending on import order, and a quantity whose definition depends on import order is not a
quantity.

**Torch-free.** Nothing here imports torch, directly or through a dependency. L4's residual-stream
rung is the one thing in the series that would need it, and it is registered with no implementation
rather than gated behind an extra, because the experiment has not been run by anyone.
"""

from __future__ import annotations

from reward_lens.measure.labels._common import (
    Proportion,
    label_quality_refusal,
    stratified_auc,
    wilson_interval,
)
from reward_lens.measure.labels.error_rate import (
    AuditSample,
    LabelErrorAudit,
    LabelErrorRate,
    MislabelCandidates,
    audit_error_rate,
    bound_from_surfacing,
    independent_rater_rate,
    irt_surface,
    two_rater_bounds,
)
from reward_lens.measure.labels.error_rate import register as _register_l2
from reward_lens.measure.labels.locality import (
    SignalLocality,
    SignalLocalityText,
    TranscriptSplit,
    signal_locality,
    split_closing,
)
from reward_lens.measure.labels.locality import register as _register_l4
from reward_lens.measure.labels.position import (
    LocalisationSeries,
    PositionPrior,
    PositionStratifiedNull,
    load_step_scores,
    rescore_against_position,
)
from reward_lens.measure.labels.position import register as _register_l5
from reward_lens.measure.labels.reference import (
    Certificate,
    DoseResponseFit,
    Plant,
    ReferenceCertificate,
    StabilityPoint,
    between_seed_homogeneity,
    campaign_plants,
    certify,
    compose_chain,
    dose_response_characterisation,
    foundry_plants,
    stability_uncertainty,
)
from reward_lens.measure.labels.reference import register as _register_l1
from reward_lens.measure.labels.verifier_error import (
    TwoSidedError,
    TwoSidedVerifierError,
    Verdict,
    compare_detector,
    false_success_task,
    two_sided_error,
)
from reward_lens.measure.labels.verifier_error import register as _register_l3

#: The five instrument classes, in catalogue order, so a sweep over the series is one import.
INSTRUMENTS = (
    ReferenceCertificate,
    LabelErrorAudit,
    TwoSidedVerifierError,
    SignalLocalityText,
    PositionStratifiedNull,
)


def register_all() -> None:
    """Register every estimator in series L. Idempotent, and never called at import."""
    _register_l1()
    _register_l2()
    _register_l3()
    _register_l4()
    _register_l5()


__all__ = [
    "INSTRUMENTS",
    "AuditSample",
    "Certificate",
    "DoseResponseFit",
    "LabelErrorAudit",
    "LabelErrorRate",
    "LocalisationSeries",
    "MislabelCandidates",
    "Plant",
    "PositionPrior",
    "PositionStratifiedNull",
    "Proportion",
    "ReferenceCertificate",
    "SignalLocality",
    "SignalLocalityText",
    "StabilityPoint",
    "TranscriptSplit",
    "TwoSidedError",
    "TwoSidedVerifierError",
    "Verdict",
    "audit_error_rate",
    "between_seed_homogeneity",
    "bound_from_surfacing",
    "campaign_plants",
    "certify",
    "compare_detector",
    "compose_chain",
    "dose_response_characterisation",
    "false_success_task",
    "foundry_plants",
    "independent_rater_rate",
    "irt_surface",
    "label_quality_refusal",
    "load_step_scores",
    "register_all",
    "rescore_against_position",
    "signal_locality",
    "split_closing",
    "stability_uncertainty",
    "stratified_auc",
    "two_rater_bounds",
    "two_sided_error",
    "wilson_interval",
]

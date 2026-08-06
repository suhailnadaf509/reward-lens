"""Typed exceptions for the reward-lens kernel.

The design's recurring lesson is that silent failure has a blast radius: a loader that
returns an empty list, a capability that is duck-typed and missing, a covariant number
compared in raw coordinates. Every one of those becomes a typed, named exception here so
the failure is loud and its cause is legible. Nothing in the kernel is allowed to fail by
returning a plausible-looking wrong answer.
"""

from __future__ import annotations


class RewardLensError(Exception):
    """Base class for every error the kernel raises deliberately."""


class CapabilityError(RewardLensError):
    """An Observable requires a Capability the signal does not declare.

    Raised before any GPU work, with the required and available capability sets in the
    message, so an incompatible measurement fails fast rather than deep inside a hook.
    """


class GaugeError(RewardLensError):
    """A covariant quantity was compared across signals without a Frame (gate 2, I3).

    The comparison APIs take a frame argument with no default; omitting it for a
    COVARIANT observable raises this rather than silently comparing raw coordinates,
    which is the E19 failure class made impossible.
    """


class CalibrationWarning(RewardLensError):
    """An instrument was asked to produce a non-EXPLORATORY claim without a scorecard.

    Most calibration gaps are handled by downgrading trust rather than raising (gates
    never silently block work; they downgrade visibly). This is raised only where a
    caller explicitly demands calibrated output and none exists, for example the MI
    estimators that refuse to over-claim on production signals.
    """


class ConformanceError(RewardLensError):
    """A signal adapter failed the conformance suite.

    New adapters are not registered until conformance passes; this is the structural fix
    for the InternLM2/QRM class of silent exclusion, where a failed load simply dropped a
    model from the campaign.
    """


class ProvenanceError(RewardLensError):
    """A required provenance or lineage field is missing (I5).

    The store rejects a signal, dataset, or derived Evidence whose parents or fingerprints
    cannot be resolved. Lineage is cheap to collect at creation and impossible to
    reconstruct afterwards, so its absence is an error, not a warning.
    """


class RegistryError(RewardLensError):
    """A registry lookup failed or a duplicate key was registered."""


class DataError(RewardLensError):
    """A dataset loader violated its declared contract.

    Declared counts that do not match, checksums that do not verify, or a network
    failure that would otherwise return an empty cell all raise this. The limit/subset
    loader bug that shrank "200 held-out pairs" to about 30 dies at this boundary.
    """


class NumericsError(RewardLensError):
    """A per-family numerics policy was violated or produced a non-finite result.

    NaN propagation on the cosine path (the all-NaN failure) and soft-cap
    interactions (the damping artifact) are caught here rather than flowing into a
    published number.
    """


__all__ = [
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

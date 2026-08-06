"""Wiring the calibration gate: scorecards to the measurement runner (gate 1).

The scorecard (`organisms/scorecard.py`) grades an instrument against planted ground truth and
produces a `CalibrationRef`. The measurement runner (`measure/base.py`) asks a pluggable provider,
at the moment it builds Evidence, whether the observable has a scorecard covering this subject and
regime. This module is the connection between the two: a registry of scorecard entries and the
provider that measure consults, plus ``install`` which registers that provider.

Keeping the connection here, rather than importing organisms into measure, preserves the dependency
direction (measure knows nothing of organisms; organisms installs itself into measure). ``install``
is a no-op until a scorecard is registered: with an empty registry the provider returns None and
every ad hoc number is correctly EXPLORATORY, so wiring the gate never silently upgrades anything.
Trust rises only once an instrument has actually earned a scorecard entry.
"""

from __future__ import annotations

from reward_lens.core.gates import CalibrationRef
from reward_lens.core.types import SubjectRef
from reward_lens.measure.base import set_calibration_provider
from reward_lens.organisms.scorecard import ScorecardEntry

# Scorecard entries indexed for the two lookups the provider needs: by observable name (the coarse
# match) and by (observable, organism_family) (the regime-aware match).
_BY_OBSERVABLE: dict[str, ScorecardEntry] = {}
_BY_FAMILY: dict[tuple[str, str], ScorecardEntry] = {}


def register_scorecard(observable: str, entry: ScorecardEntry) -> None:
    """Register a scorecard entry so downstream measurements of ``observable`` can cite it (gate 1).

    An instrument earns calibration by being graded against an organism family; registering the
    resulting entry is what makes that calibration visible to the runner. The entry is indexed both
    by observable name and by (observable, family), so a measurement that declares its regime gets
    the family-specific scorecard and one that does not still finds the observable's calibration.
    """
    _BY_OBSERVABLE[observable] = entry
    _BY_FAMILY[(observable, entry.calibration_ref.organism_family)] = entry


def scorecard_calibration_provider(
    observable: str, subject: SubjectRef, regime: dict
) -> CalibrationRef | None:
    """The gate-1 provider: return the CalibrationRef covering this observable and regime, or None.

    When the measurement's ``regime`` names an ``organism_family`` and a scorecard exists for that
    exact (observable, family) pair, the family-specific reference is returned. Otherwise, if the
    observable has any registered scorecard, its reference is returned so the number is CALIBRATED
    with the regime caveat carried in the reference's ``regime_match`` note. A missing entry returns
    None, which keeps the number EXPLORATORY.
    """
    family = regime.get("organism_family") if regime else None
    if family is not None and (observable, family) in _BY_FAMILY:
        return _BY_FAMILY[(observable, family)].calibration_ref
    entry = _BY_OBSERVABLE.get(observable)
    return entry.calibration_ref if entry is not None else None


def install() -> None:
    """Install the scorecard provider into the measurement runner (makes gate 1 live)."""
    set_calibration_provider(scorecard_calibration_provider)


def clear() -> None:
    """Forget all registered scorecards (used by tests to isolate the registry)."""
    _BY_OBSERVABLE.clear()
    _BY_FAMILY.clear()


__all__ = [
    "register_scorecard",
    "scorecard_calibration_provider",
    "install",
    "clear",
]

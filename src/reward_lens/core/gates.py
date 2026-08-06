"""The three gates, as runtime policy.

The gates convert the corpus's epistemology from prose into mechanism. They are implemented
once, here, and applied everywhere Evidence is produced or compared. Gates never silently block
work; they downgrade trust and make the downgrade visible, or, in the one case where a caller
explicitly demands a covariant cross-signal comparison without a frame, they raise so the
mistake cannot be committed to the store.

- Gate 1 (calibration): an Observable with no scorecard entry cannot claim more than
  EXPLORATORY trust when run ad hoc. `compute_trust` encodes the ladder.
- Gate 2 (gauge): a COVARIANT quantity compared across signals requires a `Frame`;
  `require_frame_for_comparison` enforces the no-default frame argument.
- Gate 3 (registration): Evidence produced under a frozen study is REGISTERED; this is threaded
  through `compute_trust` via the ``registered`` flag, which the study runner sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from reward_lens.core.errors import GaugeError
from reward_lens.core.types import FrameID, GaugeStatus, TrustLevel


class SupportsTrustCap(Protocol):
    """Anything that caps the trust of a reading derived from it.

    `ReferenceMaterial` and `CalibrationChain` implement this. Structural rather than nominal so
    `gates` does not import `reference`, which would pull the budget machinery into the module
    every Evidence construction goes through.
    """

    def trust_cap(self) -> TrustLevel: ...


@dataclass(frozen=True)
class CalibrationRef:
    """A reference to the scorecard entry that calibrates a measurement (gate 1).

    ``scorecard_entry`` is the Evidence id of the answer-key ROC that certifies this observable
    on this signal family and data regime. ``regime_match`` is a free-text note on how closely
    the calibrated regime matches the production regime, so a mismatch (calibrated on 0.5B
    organisms, applied to an 8B model) is visible rather than assumed away. ``organism_family``
    names the planted-structure family the scorecard was built from.
    """

    scorecard_entry: str
    organism_family: str
    regime_match: str = "exact"
    operating_point: dict[str, float] | None = None

    def __canonical__(self) -> dict[str, Any]:
        return {
            "scorecard_entry": self.scorecard_entry,
            "organism_family": self.organism_family,
            "regime_match": self.regime_match,
            "operating_point": self.operating_point,
        }


def compute_trust(
    *,
    calibration: CalibrationRef | None,
    registered: bool,
    adjudicated: bool = False,
    reference: "SupportsTrustCap | None" = None,
) -> TrustLevel:
    """Compute the trust level of an Evidence from the gate inputs.

    The trust level is never set by a caller. It is derived from three facts the gates
    establish: whether a calibration reference is present (gate 1), whether the run happened
    under a frozen study (gate 3), and whether the study runner has determined the result
    survived its own kill criteria and review.

    The ladder is EXPLORATORY < CALIBRATED < REGISTERED < ADJUDICATED. REGISTERED ranks above
    CALIBRATED deliberately: a preregistered prediction is a stronger epistemic claim than a
    calibrated but exploratory measurement. A REGISTERED result that lacks calibration still
    carries ``calibration: None`` on the Evidence and renders as unvalidated on cards, so the
    two axes stay independently visible even though the headline trust level is a single rung.

    ADJUDICATED requires the full set: registered, calibrated, and adjudicated. A caller that
    passes ``adjudicated=True`` without the other two gets the highest rung the facts actually
    support, never ADJUDICATED on the strength of the flag alone.

    **The reference cap.** A reading calibrated against a reference material whose
    own uncertainty was never characterised cannot climb past `CALIBRATED`, whatever else is true
    of it. `u_homogeneity is None` is not a missing field: it means nobody checked whether two
    plants with different seeds give the same answer, and the Model Organism Lottery says they do
    not. Freezing a prediction against a ruler of unknown length does not make the reading better;
    it makes the prediction precise about something unmeasured. This is the one rule that would
    have changed how the `CAL-TRANSFER` result reads.

    ``reference`` is anything exposing ``trust_cap() -> TrustLevel``, which
    `core.reference.ReferenceMaterial` and `core.reference.CalibrationChain` both do. Duck-typed
    rather than imported so the gate module keeps its shallow import graph. Absent means no cap,
    so every existing caller is unaffected.
    """
    if adjudicated and registered and calibration is not None:
        level = TrustLevel.ADJUDICATED
    elif registered:
        level = TrustLevel.REGISTERED
    elif calibration is not None:
        level = TrustLevel.CALIBRATED
    else:
        level = TrustLevel.EXPLORATORY
    if reference is not None:
        level = min(level, reference.trust_cap())
    return level


def require_frame_for_comparison(gauge_status: GaugeStatus, frame: FrameID | None) -> None:
    """Enforce gate 2 at a cross-signal comparison site.

    A COVARIANT quantity (a direction, an angle, a subspace overlap) is only comparable across
    signals in a shared frame. If the comparison is attempted without one, this raises
    `GaugeError` rather than returning a number that conflates a coordinate change with a
    functional change, which is exactly the E19 ``cos = 0.005`` failure. INVARIANT quantities
    need no frame; RAW_ONLY quantities are allowed through but the caller is responsible for
    typing the resulting Evidence as raw.
    """
    if gauge_status is GaugeStatus.COVARIANT and frame is None:
        raise GaugeError(
            "cross-signal comparison of a COVARIANT quantity requires a Frame; "
            "the comparison API takes a frame argument with no default (gate 2, I3). "
            "Fit a shared frame with geometry.fit_frame(...) and pass it, or type the "
            "measurement RAW_ONLY if a raw-coordinate number is what you intend."
        )


__all__ = [
    "CalibrationRef",
    "SupportsTrustCap",
    "compute_trust",
    "require_frame_for_comparison",
]

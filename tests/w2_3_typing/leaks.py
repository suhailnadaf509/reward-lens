"""Deliberate oracle leakage. **mypy must reject this file.**

This is the static half of the clause: *a function annotated to take features cannot
be passed a `Blind`, checked by the type checker in CI.* A clean type check here is a test failure
rather than a silent pass, which is the point: the clause is about a checker rejecting something,
and the only way to assert that is to hand it something it must reject.

Every line that must produce an error carries a `# EXPECT: <error-code>` marker.
`tests/test_record_labels_typing.py` reads those markers, runs mypy, and asserts the reported codes
and line numbers match exactly. Exactly, in both directions: an expected error that stops firing
fails the test, and so does an unexpected one, because an import error would otherwise make this
file "rejected" for a reason that has nothing to do with `Blind`.

Not collected by pytest (no `test_` prefix) and not imported by anything. It is only ever read by
mypy.
"""

from __future__ import annotations

from collections.abc import Mapping

from reward_lens.record.labels import (
    Blind,
    Detector,
    LabelValue,
    OracleFrame,
    RolloutFrame,
    blind,
)
from reward_lens.record.schema import FeatureID


def rate_by_length(features: Mapping[FeatureID, float]) -> float:
    """A detector annotated to take features, which is the clause's exact wording."""
    return features.get(FeatureID("len_tokens"), 0.0)


def detect(frame: RolloutFrame) -> float:
    """A detector annotated to take the visible half."""
    return float(len(frame.turns))


def run(detector: Detector, frame: RolloutFrame) -> float:
    """A harness that takes anything satisfying the detector protocol."""
    return detector(frame)


def reads_the_oracle(oracle: OracleFrame) -> float:
    """The scorer. This one is allowed to hold the oracle; nothing it does below is allowed."""
    label: Blind[LabelValue] = oracle.labels["hacked"]

    # The clause itself: a features function cannot be passed a Blind.
    rate_by_length(label)  # EXPECT: arg-type

    # Nor can a frame function be passed the held-out frame.
    detect(oracle)  # EXPECT: arg-type

    # There is no `.unwrap()`. That is a static fact, not a convention.
    label.unwrap()  # EXPECT: attr-defined

    # And no `.value`.
    label.value  # EXPECT: attr-defined

    # The features mapping is `Mapping[FeatureID, float]`, so a label smuggled in as a feature is
    # a type error too. The runtime name blocklist covers the case where the smuggler casts to
    # float first, which no checker can see.
    features: dict[FeatureID, float] = {FeatureID("is_hack"): label}  # EXPECT: dict-item

    return sum(features.values())


def bad_detector(oracle: OracleFrame) -> float:
    """A detector whose parameter is the held-out frame."""
    return float(len(oracle.labels))


def hand_it_to_the_harness(frame: RolloutFrame) -> float:
    """A detector that takes the oracle does not satisfy the detector protocol."""
    return run(bad_detector, frame)  # EXPECT: arg-type


def launder(blind_label: Blind[bool]) -> bool:
    """Assigning a Blind where the payload type is expected does not work either."""
    out: bool = blind_label  # EXPECT: assignment
    return out


def rebuild() -> Blind[bool]:
    """A Blind cannot be built out of the wrong shape, which keeps `blind()` the one entry point."""
    return blind(True, key=7)  # EXPECT: arg-type

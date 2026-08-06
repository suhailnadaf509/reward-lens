"""The same work, done correctly. **mypy must accept this file with no errors.**

The negative fixture beside this one proves the barrier catches leakage. This one proves the
barrier is not simply a wall: a detector, a scoring pass and an audit all type check, so someone
who follows the intended path never has to reach for `Any` or a cast to get their work done. A
`Blind` that nothing legitimate can be written against would be abandoned within a week, and the
type check would go on passing while everyone routed around it.

Not collected by pytest and not imported by anything. Only mypy reads it.
"""

from __future__ import annotations

from collections.abc import Mapping

from reward_lens.core.reading import Refusal
from reward_lens.core.store import EvidenceStore
from reward_lens.record.labels import (
    LabelQuality,
    LabelValue,
    OracleFrame,
    ReadPurpose,
    RolloutFrame,
    adjudicate,
    adjudicate_frame,
    blind,
    check_detector,
)
from reward_lens.record.schema import FeatureID


def detect(frame: RolloutFrame) -> float:
    """A detector: the visible half in, a score out."""
    length = float(frame.n_tokens)
    hedges = sum(text.count("perhaps") for text in frame.turns)
    return hedges / length if length else 0.0


def rate_by_length(features: Mapping[FeatureID, float]) -> float:
    """The features function from the clause, called with features."""
    return features.get(FeatureID("len_tokens"), 0.0)


def score_one(frame: RolloutFrame, oracle: OracleFrame, store: EvidenceStore) -> float | Refusal:
    """Score a detector against the held-out labels, through the audited path."""
    check_detector(detect)
    opened = adjudicate_frame(
        oracle,
        instrument="w2.3.example",
        purpose=ReadPurpose.SCORING,
        why="scoring the hedging detector against the held-out hack labels",
        store=store,
    )
    if isinstance(opened, Refusal):
        return opened
    truth: LabelValue = opened["hacked"]
    predicted = detect(frame) > 0.5 or rate_by_length(frame.features) > 100.0
    return 1.0 if bool(truth) == predicted else 0.0


def audit_one(oracle: OracleFrame, store: EvidenceStore) -> LabelQuality | Refusal:
    """Measure the labels' own error rate, which is the read that scoring depends on."""
    opened = adjudicate(
        oracle.labels["hacked"],
        instrument="L2",
        purpose=ReadPurpose.AUDIT,
        why="hand-auditing 200 labels to measure the answer key's error rate",
        subject=oracle.trajectory_ref,
        store=store,
    )
    if isinstance(opened, Refusal):
        return opened
    return LabelQuality(
        error_rate=0.0 if opened else 0.03,
        n_audited=200,
        method="two raters on a stratified sample, third adjudicating disagreements",
        measured_by="pool-a",
    )


def build(trajectory_id: str, hacked: bool) -> tuple[RolloutFrame, OracleFrame]:
    """Both halves, built by hand, as a converter would."""
    rollout = RolloutFrame(
        trajectory_id=trajectory_id,
        task_id="task-1",
        turns=("perhaps this is right",),
        n_tokens=4,
        advantage=0.25,
        features={FeatureID("len_tokens"): 4.0},
    )
    oracle = OracleFrame(
        trajectory_ref=trajectory_id,
        labels={"hacked": blind(hacked, key="hacked")},
        labeller="aisi-2026-03",
    )
    return rollout, oracle

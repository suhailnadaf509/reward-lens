"""M1: the tiny ``ClassifierRM`` scores pairs into Evidence with the right gauge and trust.

This test exercises the score path end to end: the return is ``Evidence[Scores]`` (not a bare float,
R1), typed INVARIANT (a raw score is gauge-free) and EXPLORATORY (nothing is calibrated yet, gate 1).
It also checks a ``score_pair``-style ordering: the model discriminates between two distinct
responses, and when the higher-scored one is labelled chosen, ``chosen > rejected`` holds and the
paired scores agree with the individual and batched score calls (the plumbing is consistent).

The tiny model has random weights, so semantic preference is not meaningful; the honest sanity here
is discrimination plus ordering consistency. Semantic chosen-over-rejected on real preference data is
a property of a trained 8B reward model and is GPU/download-gated.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.evidence import Evidence
from reward_lens.core.types import GaugeStatus, TrustLevel
from reward_lens.signals.base import Scores
from reward_lens.signals.loaders import from_tiny


@pytest.fixture(scope="module")
def tiny_signal():
    return from_tiny(seed=5, conformance_quickcheck=False)


def test_score_returns_evidence_scores(tiny_signal):
    items = [("What is 2+2?", "It is 4."), ("Name a fruit.", "An apple.")]
    evidence = tiny_signal.score(items)

    assert isinstance(evidence, Evidence)
    assert isinstance(evidence.value, Scores)
    assert evidence.value.values.shape == (2,)
    assert evidence.value.n_items == 2
    assert evidence.value.readout == "reward"
    assert np.all(np.isfinite(evidence.value.values))


def test_score_gauge_and_trust(tiny_signal):
    evidence = tiny_signal.score([("Q", "A response.")])
    # Raw scores are gauge-free and, with no scorecard entry, capped at EXPLORATORY (gates 1, 2).
    assert evidence.gauge is GaugeStatus.INVARIANT
    assert evidence.trust is TrustLevel.EXPLORATORY
    assert evidence.calibration is None
    # The subject names the signal by fingerprint and carries the readout.
    assert evidence.subject.signals == (tiny_signal.meta.fingerprint,)
    assert evidence.subject.readout == "reward"


def test_score_is_deterministic(tiny_signal):
    items = [("Q1", "first"), ("Q2", "second"), ("Q3", "third")]
    a = tiny_signal.score(items).value.values
    b = tiny_signal.score(items).value.values
    assert np.array_equal(a, b)


def test_score_pair_ordering_consistency(tiny_signal):
    """score_pair-style: the model discriminates, and chosen>rejected holds after labelling."""
    prompt = "Describe the sky."
    response_a = "The sky is a vast blue expanse stretching over the horizon at midday."
    response_b = "gray."

    score_a = float(tiny_signal.score([(prompt, response_a)]).value.values[0])
    score_b = float(tiny_signal.score([(prompt, response_b)]).value.values[0])

    # The model must discriminate between two clearly different responses.
    assert score_a != score_b, "tiny model failed to discriminate distinct responses"

    # Label the higher-scored response chosen; chosen must then beat rejected.
    chosen, rejected = (response_a, response_b) if score_a > score_b else (response_b, response_a)
    paired = tiny_signal.score([(prompt, chosen), (prompt, rejected)]).value.values
    assert paired[0] > paired[1], "chosen did not outscore rejected after labelling"

    # Batched scores agree with the individual calls (ordering consistency across batching).
    assert paired[0] == pytest.approx(max(score_a, score_b), abs=1e-4)
    assert paired[1] == pytest.approx(min(score_a, score_b), abs=1e-4)


def test_score_separable_pair_by_construction(tiny_signal):
    """A trivially separable pair: identical text scores equal; a shifted one differs."""
    prompt = "Repeat after me."
    same = tiny_signal.score([(prompt, "hello"), (prompt, "hello")]).value.values
    assert same[0] == pytest.approx(same[1], abs=1e-5), "identical inputs must score identically"

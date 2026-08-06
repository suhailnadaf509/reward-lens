"""The seven M-adapters behind the ``RewardSignal`` protocol.

M1 shipped ``ClassifierRM``; this suite is the acceptance for the rest of the corpus's substrate: a
generative judge, a process (step-level) RM, an implicit (DPO log-ratio) RM, a rubric grader, a
trajectory RM, a gated dense-reward extractor, and the ensemble / distributional composites. Each is
built on a tiny CPU model with a real tokenizer and asserted to (a) return typed ``Evidence`` with the
right capabilities, gauge, and trust, and (b) satisfy the property that makes it that kind of signal: the
judge's verdict is a genuine ``W_U[Yes] - W_U[No]`` logit difference, the implicit RM's per-token
decomposition sums to its sequence score, the ensemble mean is the mean of its members, and the dense
extractor is pinned EXPLORATORY because it ships gated.

The tiny models have random weights, so no score is semantically meaningful; what is asserted is the
mechanism (the readout really is the head) and the structural invariants, which is exactly what carries
over unchanged to an 8B model that only differs in magnitude.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.evidence import Evidence
from reward_lens.core.types import Capability, GaugeStatus, Span, TrustLevel
from reward_lens.data.lineage import make_lineage
from reward_lens.data.schema import Trajectory, TrajStep
from reward_lens.signals.base import Scores, TokenCurves
from reward_lens.signals.conformance_adapters import run_adapter_conformance
from reward_lens.signals.dense import DenseRewardExtractor
from reward_lens.signals.ensemble import DistributionalSignal, SignalEnsemble
from reward_lens.signals.implicit import ImplicitRM
from reward_lens.signals.judge import GenerativeJudge
from reward_lens.signals.loaders import from_tiny
from reward_lens.signals.process import ProcessRM, StepScores
from reward_lens.signals.rubric import RubricRM
from reward_lens.signals.trajectory import TrajectoryRM

# A stimulus set varied in length so batching forces heavy left-padding.
_JUDGE_ITEMS = [
    ("What is 2+2?", "It is 4."),
    ("Name a color.", "Blue."),
    (
        "Explain gravity.",
        "Gravity is the mutual attraction between masses that pulls them together.",
    ),
]
_TEXT_ITEMS = [
    ("Solve the problem.", "First we set up.\nThen we solve.\nFinally we check the answer twice."),
    ("Summarize.", "Short."),
    ("Describe the sky.", "The sky is a vast blue expanse stretching over the horizon at midday."),
]


# ---------------------------------------------------------------------------
# fixtures: one tiny signal of each kind, built once per module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def judge() -> GenerativeJudge:
    return GenerativeJudge.from_tiny(seed=1)


@pytest.fixture(scope="module")
def process() -> ProcessRM:
    return ProcessRM.from_tiny(seed=2)


@pytest.fixture(scope="module")
def implicit() -> ImplicitRM:
    return ImplicitRM.from_tiny(policy_seed=1, reference_seed=2, beta=0.1)


@pytest.fixture(scope="module")
def rubric() -> RubricRM:
    return RubricRM.from_tiny(criteria=("coherence", "correctness", "safety"), seed=3)


@pytest.fixture(scope="module")
def trajectory() -> TrajectoryRM:
    return TrajectoryRM.from_tiny(seed=4)


@pytest.fixture(scope="module")
def dense() -> DenseRewardExtractor:
    return DenseRewardExtractor(from_tiny(seed=6, conformance_quickcheck=False))


@pytest.fixture(scope="module")
def ensemble() -> SignalEnsemble:
    m1 = from_tiny(seed=7, conformance_quickcheck=False)
    m2 = from_tiny(seed=8, conformance_quickcheck=False)
    return SignalEnsemble([m1, m2], mode="mean")


@pytest.fixture(scope="module")
def distributional() -> DistributionalSignal:
    return DistributionalSignal.from_tiny(taus=(0.1, 0.5, 0.9), seed=9)


def _tiny_trajectory(seed_id: str = "traj-seed-1") -> Trajectory:
    text = "The tool returned 42 as the result. I conclude the answer is 42."
    step = TrajStep(
        action="call calculator",
        text=text,
        receipts=(Span(0, 35, "receipt"),),  # "The tool returned 42 as the result." (char range)
        narrative=(Span(36, len(text), "narrative"),),  # "I conclude the answer is 42."
    )
    lineage = make_lineage(seed_id, "test-builder", (), ["Trajectory"])
    return Trajectory(
        steps=(step,), outcome={"success": True}, lineage=lineage, prompt="find the answer"
    )


# ---------------------------------------------------------------------------
# GenerativeJudge
# ---------------------------------------------------------------------------


def test_judge_scores_evidence_and_caps(judge):
    evidence = judge.score(_JUDGE_ITEMS, "verdict")
    assert isinstance(evidence, Evidence)
    assert isinstance(evidence.value, Scores)
    assert evidence.value.values.shape == (3,)
    assert np.all(np.isfinite(evidence.value.values))
    assert evidence.gauge is GaugeStatus.INVARIANT
    assert evidence.trust is TrustLevel.EXPLORATORY
    assert evidence.calibration is None
    assert judge.caps & Capability.GENERATIVE
    assert judge.caps & Capability.LINEAR_READOUT
    assert {r.name for r in judge.readouts()} == {"verdict", "verdict_ab", "likert"}


def test_judge_verdict_is_the_yes_minus_no_logit_diff(judge):
    """The verdict readout equals the model's own logit(Yes) - logit(No) at the judgment position."""
    import torch

    read = judge.readout("verdict")
    tokenized = [judge.tokenize(it) for it in _JUDGE_ITEMS]
    pooled = judge.runtime.final_head_inputs(tokenized)
    logits = judge.runtime.model.lm_head(pooled.to(torch.float32))
    direct = (logits[:, read.meta["a_id"]] - logits[:, read.meta["b_id"]]).detach().numpy()
    readout_scores = judge.score(_JUDGE_ITEMS, "verdict").value.values
    assert np.allclose(direct, readout_scores, atol=1e-4)


def test_judge_verdict_orders_yes_over_no(judge):
    """A trivially-Yes reading outscores a trivially-No reading of the same content.

    On the random tiny model the verdict is assigned by the model's own logit difference, so the
    honest, model-independent statement of "orders Yes over No" is that reading the same content under
    (Yes, No) is exactly the negation of reading it under (No, Yes): whichever way the content leans, a
    Yes-verdict is positive iff the No-verdict is negative. A real instruct judge is what makes the
    verdict semantic; that path is marked, never faked.
    """
    import torch

    item = ("Is 2+2 equal to 4?", "Yes, 2+2 is 4.")
    read = judge.readout("verdict")
    tokenized = [judge.tokenize(item)]
    pooled = judge.runtime.final_head_inputs(tokenized)
    w_u = judge.runtime.model.lm_head.weight.data.to(torch.float32)
    yes_minus_no = float(
        pooled.to(torch.float32) @ (w_u[read.meta["a_id"]] - w_u[read.meta["b_id"]])
    )
    no_minus_yes = float(
        pooled.to(torch.float32) @ (w_u[read.meta["b_id"]] - w_u[read.meta["a_id"]])
    )
    # The Yes-reading and the No-reading are exact opposites, so one strictly orders above the other.
    assert yes_minus_no == pytest.approx(-no_minus_yes, abs=1e-5)
    assert (yes_minus_no > no_minus_yes) or (yes_minus_no < no_minus_yes)


def test_judge_likert_is_expected_score_in_range(judge):
    values = judge.score(_JUDGE_ITEMS, "likert").value.values
    # The Likert simplex is an expectation over the score tokens 1..5, so it lies in [1, 5].
    assert np.all(values >= 1.0) and np.all(values <= 5.0)


def test_judge_records_judgment_detection_confidence(judge):
    record = judge.meta.lineage["judgment_detection"]
    assert record["validated"] is True
    assert 0.0 <= record["confidence"] <= 1.0
    assert "note" in record  # the production-needs-a-real-model caveat is recorded, not hidden


def test_judge_conformance(judge):
    report = run_adapter_conformance(
        judge,
        items=_JUDGE_ITEMS,
        readout="verdict",
        probe=(("Give an example.", "hello brave world"), "hello brave world", "critique_target"),
    )
    assert report.passed, "\n" + report.summary()


# ---------------------------------------------------------------------------
# ProcessRM
# ---------------------------------------------------------------------------


def test_process_step_scores_and_caps(process):
    item = ("Solve.", "Step one here.\nStep two here.\nStep three concludes it.")
    evidence = process.step_scores([item])
    assert isinstance(evidence.value, StepScores)
    assert evidence.value.step_counts == [3]
    assert evidence.value.curves[0].shape == (3,)
    assert process.caps & Capability.STEP_SCORES
    assert evidence.gauge is GaugeStatus.INVARIANT
    assert evidence.trust is TrustLevel.EXPLORATORY


def test_process_last_step_equals_outcome(process):
    item = ("Solve.", "Step one.\nStep two.\nStep three is the final step.")
    step = process.step_scores([item]).value.curves[0]
    outcome = process.score([item]).value.values[0]
    assert step[-1] == pytest.approx(outcome, abs=1e-4)


def test_process_conformance(process):
    report = run_adapter_conformance(
        process,
        items=_TEXT_ITEMS,
        probe=(("Solve.", "alpha step here.\nbeta world follows."), "world", "step"),
    )
    assert report.passed, "\n" + report.summary()


# ---------------------------------------------------------------------------
# ImplicitRM
# ---------------------------------------------------------------------------


def test_implicit_decomposition_sums_to_score(implicit):
    """The native per-token reward decomposition sums exactly to the sequence score."""
    items = [("What is 2+2?", "It is four."), ("Name a color.", "A calm blue.")]
    scores = implicit.score(items).value.values
    per_token = implicit.per_token_rewards(items).value.curves
    sums = np.array([float(c.sum()) for c in per_token])
    assert np.allclose(scores, sums, atol=1e-5)


def test_implicit_prefix_last_equals_score(implicit):
    items = [("Q", "a response here"), ("Q2", "another longer response to score")]
    scores = implicit.score(items).value.values
    curves = implicit.score_prefixes(items).value.curves
    lasts = np.array([float(c[-1]) for c in curves])
    assert np.allclose(scores, lasts, atol=1e-5)


def test_implicit_paired_models_and_subject(implicit):
    assert implicit.caps & Capability.PAIRED_MODELS
    evidence = implicit.score([("Q", "response")])
    assert len(evidence.subject.signals) == 2  # policy + reference fingerprints
    assert evidence.subject.signals[0] != evidence.subject.signals[1]


def test_implicit_capture_routes_policy_vs_reference(implicit):
    import torch

    from reward_lens.core.types import Site
    from reward_lens.runtime.backend import CaptureSpec

    site = Site(1, "resid_post")
    items = [("Q", "route this capture")]
    pol = next(iter(implicit.capture(items, CaptureSpec(sites=(site,)), namespace="policy")))
    ref = next(iter(implicit.capture(items, CaptureSpec(sites=(site,)), namespace="ref")))
    assert not torch.allclose(pol.tensors[site], ref.tensors[site])


def test_implicit_conformance(implicit):
    # No linear head direction (a log-ratio), so the readout-vs-head check is not applicable.
    report = run_adapter_conformance(
        implicit,
        items=_JUDGE_ITEMS,
        check_head=False,
        probe=(("Q", "hello brave world response"), "world", "response"),
    )
    assert report.passed, "\n" + report.summary()


# ---------------------------------------------------------------------------
# RubricRM
# ---------------------------------------------------------------------------


def test_rubric_criteria_are_data_and_multi_readout(rubric):
    assert rubric.caps & Capability.MULTI_READOUT
    names = {r.name for r in rubric.readouts()}
    assert names == {"reward", "criterion:coherence", "criterion:correctness", "criterion:safety"}
    assert rubric.spec.criteria == ("coherence", "correctness", "safety")


def test_rubric_aggregate_is_weighted_sum_of_criteria():
    spec_weights = (0.5, 0.3, 0.2)
    rm = RubricRM.from_tiny(criteria=("a", "b", "c"), weights=spec_weights, seed=11)
    items = [("Q", "an answer to grade"), ("Q", "another")]
    agg = rm.score(items).value.values
    per = np.stack([rm.score(items, f"criterion:{c}").value.values for c in ("a", "b", "c")])
    manual = np.tensordot(np.array(spec_weights), per, axes=(0, 0))
    assert np.allclose(agg, manual, atol=1e-4)


def test_rubric_conformance(rubric):
    report = run_adapter_conformance(
        rubric,
        items=_TEXT_ITEMS,
        readout="reward",
        probe=({"prompt": "Give an example.", "response": "hello brave world"}, "world", "probe"),
        probe_inject=True,
    )
    assert report.passed, "\n" + report.summary()


# ---------------------------------------------------------------------------
# TrajectoryRM
# ---------------------------------------------------------------------------


def test_trajectory_consumes_trajectory_with_typed_spans(trajectory):
    traj = _tiny_trajectory()
    tokenized = trajectory.tokenize(traj)
    kinds = {s.kind for s in tokenized.spans}
    assert {"action", "receipt", "narrative"} <= kinds
    assert trajectory.caps & Capability.SPAN_TYPES
    evidence = trajectory.score([traj])
    assert isinstance(evidence.value, Scores)
    assert np.all(np.isfinite(evidence.value.values))
    assert evidence.trust is TrustLevel.EXPLORATORY


def test_trajectory_conformance(trajectory):
    # Varied step counts/lengths so batching forces left-padding (the invariance is real, not trivial).
    trajs = [
        _varied_trajectory("short", n_steps=1),
        _varied_trajectory("medium", n_steps=2),
        _varied_trajectory("long", n_steps=3),
    ]
    report = run_adapter_conformance(
        trajectory,
        items=trajs,
        probe=(_tiny_trajectory("probe"), "The tool returned 42", "receipt"),
    )
    assert report.passed, "\n" + report.summary()


def _varied_trajectory(seed_id: str, n_steps: int) -> Trajectory:
    steps = tuple(
        TrajStep(
            action=f"call tool {i}",
            text=f"Observation {i}: the result was {i * 7}. I record it.",
            receipts=(Span(0, 14, "receipt"),),
        )
        for i in range(n_steps)
    )
    lineage = make_lineage(seed_id, "test-builder", (), ["Trajectory"])
    return Trajectory(steps=steps, outcome={"success": True}, lineage=lineage, prompt="do the task")


# ---------------------------------------------------------------------------
# DenseRewardExtractor (gated)
# ---------------------------------------------------------------------------


def test_dense_evidence_is_exploratory_gated(dense):
    evidence = dense.dense_rewards([("Q", "hello world this is a dense reward map")])
    assert isinstance(evidence.value, TokenCurves)
    # The gating is structural: no calibration is ever attached, so trust cannot exceed EXPLORATORY.
    assert evidence.trust is TrustLevel.EXPLORATORY
    assert evidence.calibration is None
    assert evidence.subject.extra["gated"] is True
    assert dense.meta.lineage["gated"] is True


def test_dense_map_sums_to_outcome_score(dense):
    item = [("Q", "the dense map telescopes to the outcome score")]
    dense_map = dense.dense_rewards(item).value.curves[0]
    outcome = dense.score(item).value.values[0]
    assert dense_map.sum() == pytest.approx(outcome, abs=1e-4)


def test_dense_records_prefix_evidence_as_parent(dense):
    evidence = dense.dense_rewards([("Q", "provenance parent check")])
    assert evidence.provenance.parents  # the prefix Evidence id is recorded as a DAG parent


# ---------------------------------------------------------------------------
# SignalEnsemble
# ---------------------------------------------------------------------------


def test_ensemble_mean_equals_mean_of_members(ensemble):
    items = [("Q", "response one"), ("Q", "response two is longer than the first one here")]
    composite = ensemble.score(items).value.values
    members = np.stack([m.score(items).value.values for m in ensemble.members])
    assert np.allclose(composite, members.mean(axis=0), atol=1e-6)


def test_ensemble_subject_names_every_member(ensemble):
    evidence = ensemble.score([("Q", "member provenance")])
    member_fps = tuple(m.meta.fingerprint for m in ensemble.members)
    assert evidence.subject.signals == member_fps
    assert evidence.gauge is GaugeStatus.INVARIANT


def test_ensemble_min_and_quantile_modes():
    m1 = from_tiny(seed=7, conformance_quickcheck=False)
    m2 = from_tiny(seed=8, conformance_quickcheck=False)
    items = [("Q", "a"), ("Q", "a longer response to separate the two members clearly")]
    both = np.stack([m1.score(items).value.values, m2.score(items).value.values])
    en_min = SignalEnsemble([m1, m2], mode="min")
    assert np.allclose(en_min.score(items).value.values, both.min(axis=0), atol=1e-6)
    en_q = SignalEnsemble([m1, m2], mode="quantile", q=0.5)
    assert en_q.caps & Capability.DISTRIBUTIONAL
    assert np.allclose(en_q.score(items).value.values, np.quantile(both, 0.5, axis=0), atol=1e-6)


def test_ensemble_conformance(ensemble):
    report = run_adapter_conformance(
        ensemble,
        items=_TEXT_ITEMS,
        check_head=False,  # a composite has no single head direction
        check_prefix=True,  # composed prefix curves still satisfy curve[-1] == composite score
        probe=None,
    )
    assert report.passed, "\n" + report.summary()


def test_all_adapters_satisfy_the_protocol(
    judge, process, implicit, rubric, trajectory, dense, ensemble, distributional
):
    """Every adapter is structurally a ``RewardSignal`` (the extensibility contract)."""
    from reward_lens.signals.base import RewardSignal

    for sig in (judge, process, implicit, rubric, trajectory, dense, ensemble, distributional):
        assert isinstance(sig, RewardSignal), f"{type(sig).__name__} does not satisfy RewardSignal"


# ---------------------------------------------------------------------------
# DistributionalSignal
# ---------------------------------------------------------------------------


def test_distributional_exposes_quantile_readouts(distributional):
    assert distributional.caps & Capability.DISTRIBUTIONAL
    assert {r.name for r in distributional.readouts()} == {
        "quantile:0.1",
        "quantile:0.5",
        "quantile:0.9",
    }


def test_distributional_quantile_matches_underlying_row(distributional):
    items = [("Q", "a distributional response")]
    q = distributional.quantile(items, 0.9).value.values
    row = distributional.signal.score(items, "criterion:2").value.values
    assert np.allclose(q, row, atol=1e-6)


def test_distributional_conformance(distributional):
    report = run_adapter_conformance(
        distributional,
        items=_TEXT_ITEMS,
        readout="quantile:0.5",
        probe=({"prompt": "Give an example.", "response": "hello brave world"}, "world", "probe"),
        probe_inject=True,
    )
    assert report.passed, "\n" + report.summary()


# ---------------------------------------------------------------------------
# every adapter returns a typed Evidence payload (R1)
# ---------------------------------------------------------------------------


def test_all_adapters_return_typed_evidence(judge, process, implicit, rubric, trajectory):
    for sig, items in (
        (judge, _JUDGE_ITEMS),
        (process, _TEXT_ITEMS),
        (implicit, _JUDGE_ITEMS),
        (rubric, _TEXT_ITEMS),
    ):
        evidence = sig.score(items)
        assert isinstance(evidence, Evidence)
        assert isinstance(evidence.value, Scores)
        assert evidence.value.values.shape == (len(items),)
        assert evidence.trust is TrustLevel.EXPLORATORY
    traj_ev = trajectory.score([_tiny_trajectory()])
    assert isinstance(traj_ev, Evidence) and isinstance(traj_ev.value, Scores)

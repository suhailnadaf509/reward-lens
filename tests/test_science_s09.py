"""S9 runs end to end as a frozen study, emitting REGISTERED Evidence and feeding S8 at T10.

The Verification Score, the span-patch separation, and the dense-map localization are exercised on a
planted verifier where the causal-anchoring fraction and the error span are known by construction, so
the instrument is validated before it is turned on a production model (gate 1). The
Verification Score must recover the planted anchored fraction alpha across the sweep, span-patching the
error span must shift the score by the anchored fraction and not the style fraction, and the real
DenseRewardExtractor's per-token map must light up the labeled error span. The per-model VS leaderboard,
the (step x layer) propagation lens, and the cross-paradigm comparison on ProcessBench are recorded as
inconclusive-because-gated on the reward-model population and GPU.
"""

from __future__ import annotations

from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import TrustLevel
from reward_lens.studies import Scoreboard, render_report, run_study
from studies.s09_verification.analysis import build_spec


def test_s09_runs_and_registers(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    # All three calibration hypotheses confirmed; the study is not killed by its own kill criterion.
    assert result.outcomes["H1-vs-recovery"] == "confirmed", result.metrics
    assert result.outcomes["H2-causal-anchoring"] == "confirmed", result.metrics
    assert result.outcomes["H3-dense-localization"] == "confirmed", result.metrics
    assert not result.killed, result.killed_by

    # The headline Evidence is REGISTERED and descends from the planted construction (a real DAG).
    root = store.find(observable="S09.PlantedVerifier")
    assert root and root[0].trust is TrustLevel.REGISTERED

    vs = store.find(observable="S09.VerificationScore")
    assert vs and vs[0].trust is TrustLevel.REGISTERED
    assert store.parents(vs[0]), "verification-score Evidence should cite a parent"

    dense = store.find(observable="S09.DenseLocalization")
    assert dense and dense[0].trust is TrustLevel.REGISTERED
    assert store.parents(dense[0]), "dense-localization Evidence should cite a parent"


def test_s09_recovers_alpha_and_localizes(tmp_path):
    store = EvidenceStore(tmp_path)
    _, result = run_study(build_spec(), store=store)

    # The Verification Score recovered the planted anchored fraction exactly (up to float precision),
    # and the StyleShare index recovered its complement.
    assert result.metrics["vs_alpha_recovery_max_abs_error"] < 1e-6, result.metrics
    assert result.metrics["style_share_recovery_max_abs_error"] < 1e-6, result.metrics

    # Span-patching the error span shifted the score by the anchored fraction and the style patch by the
    # style fraction, with no cross-leakage: the reward gap separated cleanly.
    assert result.metrics["patch_separation_error"] < 1e-6, result.metrics

    # The DenseRewardExtractor's per-token map localized to the labeled error span well above chance.
    assert result.metrics["dense_localization_auc"] > 0.9, result.metrics

    # The per-alpha table records VS = alpha for each rung of the sweep.
    vs = store.find(observable="S09.VerificationScore")[0]
    scores = vs.value["verification_scores"]
    for alpha, measured in zip(vs.value["alphas"], scores):
        assert abs(measured - alpha) < 1e-6, (alpha, measured)


def test_s09_dense_product_is_gated_before_certification(tmp_path):
    store = EvidenceStore(tmp_path)
    run_study(build_spec(), store=store)

    # The DenseRewardExtractor's own Evidence is EXPLORATORY by construction (it attaches no
    # calibration): the S9 study is the certification that validates it against labeled error spans.
    dense_product = store.find(observable="signals.dense.dense_rewards")
    assert dense_product, "the real DenseRewardExtractor should have produced Evidence"
    assert dense_product[0].trust is TrustLevel.EXPLORATORY


def test_s09_span_patch_plumbing(tmp_path):
    store = EvidenceStore(tmp_path)
    run_study(build_spec(), store=store)

    plumbing = store.find(observable="S09.SpanPatchPlumbing")
    assert plumbing and plumbing[0].trust is TrustLevel.REGISTERED
    value = plumbing[0].value
    if value.get("ran"):
        # The production interventions path ran: the identity clean-twin replace reproduced the clean
        # score, so run_patched_scores is faithful to the operation the span patch relies on.
        assert value["status"] == "proven-on-tiny-vehicle"
        assert value["identity_fidelity_abs"] < 1e-2, value
    else:
        # If the torch path was unavailable it is gated honestly with the exact requirement.
        assert value["status"] == "gated"
        assert value["needs"]


def test_s09_gated_arms_recorded(tmp_path):
    store = EvidenceStore(tmp_path)
    run_study(build_spec(), store=store)

    gated = store.find(observable="S09.GatedArm")
    arms = {ev.value["arm"] for ev in gated}
    assert arms == {"vs-leaderboard", "step-layer-propagation", "cross-paradigm"}
    for ev in gated:
        assert ev.value["status"] == "inconclusive-because-gated"
        assert ev.value["needs"] and ev.value["produces"]
        assert ev.trust is TrustLevel.REGISTERED


def test_s09_updates_scoreboard(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    board = Scoreboard(tmp_path / "scoreboard.json")
    board.update_from_result(frozen.study_id, frozen.spec.hypotheses, result)
    # S9 feeds S8's belief-factorization row: verification is KUI applied to "this step is wrong".
    assert board.rows["T10"].status == "confirmed"
    assert board.rows["T10"].adjudicating_evidence  # cites the evidence that settled it

    report = render_report(frozen, result, store)
    assert "CONFIRMED" in report
    assert frozen.study_id in report

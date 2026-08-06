"""S8 runs end to end as a frozen study, emitting REGISTERED Evidence and updating T2 and T10.

This is one of milestone M6's cheap sciences, exercised entirely on planted ground truth where the
answer is known by construction (gate 1), so each instrument is calibrated before it would
ever be turned on a production model. Four things are proven here. First, the Knowledge-Utilization
Index recovers a positive represented-but-unpriced gap on a planted battery and ~none for a
decoded-and-priced control (the mechanistic precondition of hacking). Second, per-dimension distortion
is coverage-gated sensitivity, lighting up a priced-but-not-intended property and staying near zero for
an equally sensitive intended one. Third, the crown proof: a belief-patch factorization built from
``concepts.vectors.concept_direction`` and ``interventions.patch`` classifies a planted-epistemic
sycophancy organism epistemic and a planted-axiological one axiological, so the method tells the two
constructions apart. Fourth, the ``stats.mi`` estimator recovers a closed-form Gaussian mutual
information within tolerance and then measures the alignment channel's kept fraction and the gauge=kernel
identity. The real-model arms (the four-campaign KUI matrix, the real-model factorization, the real
channel bit counts) are population/GPU-gated and invent no number. The study folds its outcomes into T2
(distortion) and T10 (belief factorization and the channel identity).
"""

from __future__ import annotations

import numpy as np

from reward_lens.core.store import EvidenceStore
from reward_lens.core.types import TrustLevel
from reward_lens.studies import Scoreboard, render_report, run_study
from studies.s08_factorization.analysis import build_spec


def test_s08_runs_and_registers(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    # Every proof-arm hypothesis is confirmed, and the study is not killed by its own kill criteria.
    for hid in (
        "H1-kui-gap",
        "H2-kui-control",
        "H3-distortion",
        "H4-epistemic",
        "H5-axiological",
        "H6-mi-calibration",
        "H7-channel",
        "H8-gauge-kernel",
    ):
        assert result.outcomes[hid] == "confirmed", (hid, result.metrics)
    assert not result.killed, result.killed_by

    # Arm A/B (T2): KUI recovers the represented-but-unpriced gap; the control sits on the diagonal;
    # distortion separates priced-but-not-intended from priced-and-intended.
    assert result.metrics["kui_gap"] > 0.3
    assert result.metrics["kui_ignored"] > 0.3
    assert result.metrics["kui_control_abs"] < 0.2
    assert result.metrics["distortion_separation"] > 0.3
    assert result.metrics["distortion_spurious"] > result.metrics["distortion_intended"]

    # Arm C (T10, the crown): the two planted mechanisms are told apart. The planted-epistemic organism
    # is classified epistemic (severing belief removes the premium) and the planted-axiological organism
    # axiological (severing belief leaves it intact), with a wide margin.
    assert result.metrics["factorization_epistemic_share_epi"] > 0.7
    assert result.metrics["factorization_epistemic_share_axi"] < 0.3
    assert result.metrics["factorization_margin"] > 0.5

    # Arm D (T10): the KSG estimator recovers the closed-form Gaussian MI within tolerance, the
    # high-fidelity channel keeps ~all of H(V), and a reward-null direction transmits ~0 bits.
    assert result.metrics["mi_ksg_gaussian_abs_bias"] < 0.1
    assert result.metrics["channel_kept_fraction_hifi"] > 0.9
    assert (
        result.metrics["channel_kept_fraction_lossy"] < result.metrics["channel_kept_fraction_hifi"]
    )
    assert result.metrics["channel_null_bits"] < 0.15

    # The headline Evidence is REGISTERED and descends from the planted-organism root (a real DAG).
    for observable in (
        "S08.Organisms",
        "S08.KnowledgeUtilizationGap",
        "S08.Distortion",
        "S08.Factorization",
        "S08.MICalibration",
        "S08.AlignmentChannel",
    ):
        found = store.find(observable=observable)
        assert found and found[0].trust is TrustLevel.REGISTERED, observable
    kui = store.find(observable="S08.KnowledgeUtilizationGap")
    assert store.parents(kui[0]), "KUI Evidence should cite a parent (the organism root)"


def test_s08_factorization_classifies_both_organisms(tmp_path):
    """The crown proof, read straight off the Factorization Evidence: epistemic vs axiological."""
    store = EvidenceStore(tmp_path)
    _frozen, _result = run_study(build_spec(), store=store)

    fac = store.find(observable="S08.Factorization")
    assert fac
    value = fac[0].value
    # The method, blind to the construction, assigns the right class to each planted organism.
    assert value["epistemic_organism"]["classification"] == "epistemic"
    assert value["axiological_organism"]["classification"] == "axiological"
    assert value["epistemic_organism"]["planted"] == "epistemic"
    assert value["axiological_organism"]["planted"] == "axiological"
    # The belief direction is genuinely priced by the reward on both organisms (by construction).
    assert value["epistemic_organism"]["belief_reward_cos_abs"] > 0.3
    # The inline belief patch is bit-equal to the real ResidualAddPatch hook (it rides interventions).
    assert value["epistemic_organism"]["patch_matches_intervention"] is True
    assert value["axiological_organism"]["patch_matches_intervention"] is True


def test_s08_real_model_arms_are_gated(tmp_path):
    """The population/GPU arms emit REGISTERED gated Evidence with no adjudicated metric (no fabrication)."""
    store = EvidenceStore(tmp_path)
    _frozen, result = run_study(build_spec(), store=store)

    for observable in ("S08.KUIMatrixReal", "S08.FactorizationReal", "S08.ChannelBitsReal"):
        gated = store.find(observable=observable)
        assert gated and gated[0].trust is TrustLevel.REGISTERED, observable
        assert gated[0].value.get("gated") is True, observable

    # The real channel bit counts are recorded as a target to measure, not a measurement: no number is
    # promoted into a metric the runner could adjudicate.
    channel_real = store.find(observable="S08.ChannelBitsReal")[0].value
    assert "target_to_measure_not_measured" in channel_real
    for key in result.metrics:
        assert "real" not in key.lower(), f"no real-model metric should exist; found {key}"

    # The production belief-probe path reports which frozen contracts are present; concepts.beliefs and
    # concepts.probes are built in parallel and may be absent, in which case the arm is gated. The
    # steering primitive already exists, so it is reported present.
    prod = store.find(observable="S08.ProductionBeliefProbe")
    assert prod and prod[0].trust is TrustLevel.REGISTERED
    contracts = prod[0].value["contracts_present"]
    assert contracts["interventions.steer.SteeringIntervention"] is True
    assert prod[0].value["gated"] == (not prod[0].value["production_path_available"])


def test_s08_updates_scoreboard(tmp_path):
    store = EvidenceStore(tmp_path)
    frozen, result = run_study(build_spec(), store=store)

    board = Scoreboard(tmp_path / "scoreboard.json")
    board.update_from_result(frozen.study_id, frozen.spec.hypotheses, result)
    # S8 fills T2 (distortion) and originates T10 (belief factorization and the channel identity).
    assert board.rows["T2"].status == "confirmed"
    assert board.rows["T10"].status == "confirmed"
    assert board.rows["T10"].adjudicating_evidence
    assert frozen.study_id in board.rows["T2"].studies

    report = render_report(frozen, result, store)
    assert "CONFIRMED" in report
    assert frozen.study_id in report


# ---------------------------------------------------------------------------
# stats/mi: the estimator proven on synthetic where the true MI is known
# ---------------------------------------------------------------------------


def test_mi_ksg_recovers_gaussian_mutual_information():
    """The KSG estimator recovers a closed-form Gaussian MI within tolerance, at several correlations."""
    from reward_lens.stats.mi import calibrate_gaussian, gaussian_mi_bits

    for rho in (0.3, 0.6, 0.8):
        cal = calibrate_gaussian(rho=rho, n=2000, k=3, repeats=5, estimator="ksg")
        assert abs(cal.true_bits - gaussian_mi_bits(rho)) < 1e-12
        # The estimator recovers the truth within a tenth of a bit, and the reported bias is honest.
        assert abs(cal.bias_bits) < 0.1, (rho, cal.bias_bits)
        assert cal.estimate_bits > 0.0


def test_mi_binned_bias_is_reported_and_bounded():
    """The binned cross-check is also calibrated; its (small, here) bias is reported, not hidden."""
    from reward_lens.stats.mi import calibrate_gaussian

    cal = calibrate_gaussian(rho=0.6, n=2000, bins=8, estimator="binned", repeats=5)
    assert cal.estimator == "binned"
    assert abs(cal.bias_bits) < 0.2  # looser than KSG; the point is it is measured and small here


def test_mi_discrete_continuous_recovers_source_entropy_at_high_fidelity():
    """A discrete source through a near-lossless continuous channel transmits ~its full entropy."""
    from reward_lens.stats.mi import entropy_discrete_bits, mi_discrete_continuous

    rng = np.random.default_rng(0)
    m, n = 4, 1600
    v = rng.integers(0, m, size=n)
    centroids = np.arange(m) * 4.0
    r_hifi = centroids[v] + rng.standard_normal(n) * 0.15
    r_lossy = centroids[v] + rng.standard_normal(n) * 4.0

    h_v = entropy_discrete_bits(v)
    i_hifi = mi_discrete_continuous(v, r_hifi, k=3)
    i_lossy = mi_discrete_continuous(v, r_lossy, k=3)

    assert abs(h_v - 2.0) < 0.1  # uniform over 4 symbols is 2 bits
    assert abs(i_hifi - h_v) < 0.15  # lossless channel transmits the whole source entropy
    assert i_lossy < 0.6 * h_v  # the lossy channel drops most of it


def test_mi_import_surface():
    """The MI module is importable and exposes the estimators and the calibration harness."""
    from reward_lens.stats import mi

    for name in (
        "gaussian_mi_bits",
        "mi_ksg",
        "mi_discrete_continuous",
        "mi_binned",
        "calibrate_gaussian",
        "MICalibration",
    ):
        assert hasattr(mi, name), name

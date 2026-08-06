"""Unit tests for F5's arithmetic: the two dimensionality statistics, and the estimator's identity.

The arithmetic here is small and every piece of it has a closed form on a matrix chosen to have one,
which is the point: the spike measured an estimator's noise on a real policy, and it can only
be believed if the estimator itself is right. The score identity in `test_score_identity_holds` is
the one that would catch a wrong pooling convention, and it is checked against the analytic answer
rather than against another implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from reward_lens.core.invariance import GROUPS, INVARIANT
from reward_lens.measure.base import lint_instrument
from reward_lens.measure.frontier.covector import (
    F5,
    P8_COSINE_THRESHOLD,
    P8_RESOLUTION,
    P8_RSE_THRESHOLD,
    POOLED_N_FLOOR,
    SelectionStableRank,
    activation_metric,
    leave_one_out_baseline,
    p8_resolution,
    p8_study,
    participation_ratio,
    required_pooled_n,
    selection_covector,
    selection_second_moment,
    stable_rank,
    whitened_spectrum,
)

# ---------------------------------------------------------------------------
# stable rank and the participation ratio, against closed forms
# ---------------------------------------------------------------------------


def test_stable_rank_of_a_rank_one_matrix_is_one():
    m = np.zeros((6, 6))
    m[0, 0] = 3.7
    assert stable_rank(m) == pytest.approx(1.0)
    assert participation_ratio(m) == pytest.approx(1.0)


def test_stable_rank_of_the_identity_is_d():
    assert stable_rank(np.eye(9)) == pytest.approx(9.0)
    assert participation_ratio(np.eye(9)) == pytest.approx(9.0)


def test_stable_rank_is_full_rank_and_still_near_one():
    """The whole argument in one assertion.

    A matrix with 99% of its Frobenius mass in one direction has full numerical rank, because no
    eigenvalue is exactly zero, and a stable rank near 1. Reporting the rank would call this
    eight-dimensional; reporting the stable rank calls it one-dimensional, and it is.
    """
    ev = np.array([1.0, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    m = np.diag(ev)
    assert np.linalg.matrix_rank(m) == 8
    assert stable_rank(m) == pytest.approx(1.007, abs=1e-3)
    assert participation_ratio(m) < 1.02


def test_the_two_participation_ratio_conventions_disagree():
    """They are different numbers on the same matrix, which is why the convention is stated."""
    ev = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
    m = np.diag(ev)
    moment = participation_ratio(m, convention="moment_ratio")
    share = participation_ratio(m, convention="variance_share")
    # sum = 1.0 and sum of squares = 0.325, so the moment ratio is 1 / 0.325.
    assert moment == pytest.approx(3.0769, abs=0.001)
    assert share == pytest.approx(3.0)
    assert moment != share


def test_dimensionality_statistics_are_invariant_under_an_orthogonal_map():
    """`M -> Q M Q^T` leaves the spectrum alone, which is what `repr.basis` INVARIANT asserts."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((60, 7))
    m = x.T @ x
    q, r = np.linalg.qr(rng.standard_normal((7, 7)))
    q = q * np.sign(np.diag(r))
    rotated = q @ m @ q.T
    assert stable_rank(rotated) == pytest.approx(stable_rank(m), rel=1e-10)
    assert participation_ratio(rotated) == pytest.approx(participation_ratio(m), rel=1e-10)


def test_stable_rank_of_a_zero_matrix_is_nan_rather_than_one():
    """A matrix with no energy has no shape, and 1.0 would read as perfect concentration."""
    assert np.isnan(stable_rank(np.zeros((4, 4))))
    assert np.isnan(participation_ratio(np.zeros((4, 4))))


# ---------------------------------------------------------------------------
# the estimator
# ---------------------------------------------------------------------------


def test_leave_one_out_baseline_excludes_the_member():
    r = np.array([[1.0, 2.0, 3.0, 4.0]])
    b = leave_one_out_baseline(r)
    assert b[0, 0] == pytest.approx((2 + 3 + 4) / 3)
    assert b[0, 3] == pytest.approx((1 + 2 + 3) / 3)


def test_score_identity_holds_for_the_position_summed_gradient():
    """`E[(r - b) g] = Cov(r, g)` when `E[g] = 0`, which is the identity the convention buys.

    Built analytically: draw `g` with mean zero and `r` linear in `g` plus noise, so the population
    covector is known exactly. An estimator that pooled positions differently, or that used a
    baseline correlated with its own member, misses this.
    """
    rng = np.random.default_rng(7)
    p, k, d = 2000, 8, 3
    g = rng.standard_normal((p, k, d))
    true = np.array([2.0, -1.0, 0.5])
    r = (g @ true) + 0.1 * rng.standard_normal((p, k))
    est = selection_covector(r, g, baseline="leave_one_out")
    # 16,000 samples at a per-coordinate standard error near 0.019, so 0.08 is about four of them
    # and a wrong pooling or a self-correlated baseline misses by far more than that.
    assert est == pytest.approx(true, abs=0.08)


def test_a_constant_reward_gives_a_zero_covector():
    """Selection with nothing to select on is exactly zero, not merely small."""
    rng = np.random.default_rng(1)
    g = rng.standard_normal((10, 6, 4))
    r = np.full((10, 6), 3.14)
    est = selection_covector(r, g, baseline="leave_one_out")
    assert np.allclose(est, 0.0, atol=1e-12)


def test_the_baseline_choices_are_named_and_an_unknown_one_raises():
    rng = np.random.default_rng(2)
    g = rng.standard_normal((5, 4, 3))
    r = rng.standard_normal((5, 4))
    for baseline in ("none", "group_mean", "leave_one_out"):
        assert selection_covector(r, g, baseline=baseline).shape == (3,)
    with pytest.raises(ValueError, match="group_mean"):
        selection_covector(r, g, baseline="mean")  # type: ignore[arg-type]


def test_second_moment_is_positive_semi_definite_and_symmetric():
    rng = np.random.default_rng(3)
    g = rng.standard_normal((12, 5, 4))
    r = rng.standard_normal((12, 5))
    m = selection_second_moment(r, g)
    assert np.allclose(m, m.T)
    assert np.linalg.eigvalsh(m).min() > -1e-12


def test_whitened_problem_reports_its_damping_and_survives_a_singular_metric():
    """`G` is singular below `d` pooled rollouts, and the ridge is returned rather than hidden."""
    rng = np.random.default_rng(4)
    g = rng.standard_normal((2, 2, 6))  # 4 rollouts, d = 6, so G has rank 4
    r = rng.standard_normal((2, 2))
    m = selection_second_moment(r, g)
    metric = activation_metric(g)
    assert np.linalg.matrix_rank(metric) < 6
    values, vectors, damping = whitened_spectrum(m, metric, ridge=1e-6)
    assert damping > 0.0
    assert values.size == 6
    assert vectors.shape == (6, 6)
    assert np.all(np.diff(values) <= 1e-12)  # descending


def test_required_pooled_n_follows_the_root_n_scaling():
    assert required_pooled_n(2.0, 100) == pytest.approx(400.0)
    assert required_pooled_n(0.5, 100) == pytest.approx(25.0)
    assert np.isnan(required_pooled_n(float("nan"), 100))


# ---------------------------------------------------------------------------
# the declarations
# ---------------------------------------------------------------------------


def test_all_four_instruments_pass_lint():
    for cls in F5:
        assert lint_instrument(cls(rewards=[1.0, 2.0])) == []


def test_the_four_quantities_are_the_catalogue_record():
    assert sorted(cls.quantity for cls in F5) == [
        "selection.covector",
        "selection.dimensionality",
        "selection.spectrum",
        "selection.stable_rank",
    ]


def test_the_four_mandatory_baselines_are_declared():
    """The catalogue's Base cell, verbatim: a random direction of matched norm, a coherent
    irrelevant semantic direction, the vanilla logit lens, and string matching."""
    for cls in F5:
        declared = set(cls.baselines)
        assert declared == {
            "baseline.random_direction_matched_norm",
            "baseline.semantic_placebo",
            "baseline.logit_lens",
            "baseline.string_match",
        }


def test_the_invariance_relation_is_declared_per_group():
    """E55's mapping form: the scalars are invariant under `repr.basis` and the payload is not."""
    from reward_lens.core.invariance import resolve_relation

    for cls in F5:
        inst = cls(rewards=[1.0, 2.0])
        assert resolve_relation(inst, "repr.basis") == INVARIANT
        assert "invariant" in GROUPS["repr.basis"].admits


def test_the_generated_invariance_test_passes():
    """Rule four: no instrument merges without the property test its declared group generates.

    `check_invariance` asserts a relation on a scalar, so `run` projects the reading onto the stable
    rank, which is the scalar the relation is declared about. A Haar-random orthogonal map acts on
    the activations and the reading does not move: 3.6e-15 against a tolerance near 3.8e-07.
    """
    from reward_lens.core.invariance import InvariancePayload, check_invariance

    rng = np.random.default_rng(11)
    p, k, d = 12, 6, 7
    g = rng.standard_normal((p, k, d))
    r = rng.standard_normal((p, k))
    instrument = SelectionStableRank(rewards=r.ravel(), groups=np.repeat(np.arange(p), k))
    payload = InvariancePayload(scores=r.ravel(), activations=g.reshape(-1, d))

    def run(_instrument, transformed):
        grads = np.asarray(transformed.activations, dtype=np.float64).reshape(p, k, d)
        scores = np.asarray(transformed.scores, dtype=np.float64).reshape(p, k)
        return stable_rank(selection_second_moment(scores, grads))

    report = check_invariance(instrument, "repr.basis", payload, n=32, run=run)
    assert report.passed, report.render() if hasattr(report, "render") else report
    assert report.max_deviation < report.tol


def test_the_gauge_status_is_covariant_because_the_payload_carries_a_direction():
    from reward_lens.core.types import GaugeStatus

    for cls in F5:
        assert cls.gauge_status is GaugeStatus.COVARIANT


# ---------------------------------------------------------------------------
# the spike's record
# ---------------------------------------------------------------------------


def test_the_recorded_resolution_is_a_no_go_and_says_why():
    assert P8_RESOLUTION["verdict"] == "no-go"
    assert P8_RESOLUTION["go"] is False
    assert P8_RESOLUTION["rse_k64"] >= P8_RSE_THRESHOLD
    assert P8_RESOLUTION["cosine_k64"] <= P8_COSINE_THRESHOLD


def test_the_curve_covers_the_registered_k_grid_for_every_ablation():
    for arm, curve in P8_RESOLUTION["rse_by_k"].items():
        assert sorted(curve, key=int) == ["4", "8", "16", "64"], arm
        values = [curve[k] for k in ("4", "8", "16", "64")]
        assert values == sorted(values, reverse=True), f"{arm} is not monotone in K"


def test_the_group_mean_baseline_is_the_reduction_that_worked():
    """4.99 to 1.08 at K = 64 for the baseline; 1.099 to 1.075 for everything else together."""
    at64 = {k: v["64"] for k, v in P8_RESOLUTION["rse_by_k"].items()}
    assert at64["no_baseline"] / at64["group_mean_baseline"] > 4.0
    assert at64["leave_one_out_baseline"] / at64["antithetic_plus_all"] < 1.05


def test_the_differentiable_surrogate_beats_the_score_function_by_the_recorded_factor():
    ratio = P8_RESOLUTION["noise_ratio_score_function_over_surrogate"]
    assert ratio == pytest.approx(
        P8_RESOLUTION["rse_k64"] / P8_RESOLUTION["surrogate_rse_k64"], rel=1e-9
    )
    assert ratio > 50.0


def test_the_floor_matches_the_measured_extrapolation():
    """`POOLED_N_FLOOR` is the measured 591.5 rounded up, and it is not a round number chosen first."""
    assert POOLED_N_FLOOR >= P8_RESOLUTION["pooled_n_for_rse_1"]
    assert POOLED_N_FLOOR < P8_RESOLUTION["pooled_n_for_rse_1"] + 100


def test_the_spike_recorded_its_own_import_graph():
    assert P8_RESOLUTION["apparatus"]["nnsight_imported"] is False


def test_freezing_the_study_is_deterministic_and_the_analysis_path_resolves():
    a = p8_study(frozen_at="2026-08-05T00:00:00+00:00")
    b = p8_study(frozen_at="2026-08-05T00:00:00+00:00")
    assert a.study_id == b.study_id
    assert a.spec_hash == b.spec_hash
    module, _, name = a.spec.analysis.rpartition(".")
    import importlib

    assert callable(getattr(importlib.import_module(module), name))


def test_the_resolution_refutes_both_hypotheses_and_fires_the_kill():
    result = p8_resolution()
    assert result.outcomes == {"H1": "refuted", "H2": "refuted"}
    assert result.killed is True
    assert result.kill_outcomes == {"K1": "fired"}
    spec = p8_study().spec
    for hypothesis in spec.hypotheses:
        value = result.metrics[hypothesis.prediction.metric]
        assert hypothesis.prediction.check(value) is False
    assert spec.kill_criteria[0].fired(result.metrics["covector_relative_standard_error_k64"])

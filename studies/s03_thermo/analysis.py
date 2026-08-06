"""S3 - Reward Thermodynamics (DESIGN Part III, S3; scoreboard T9).

The claim S3 preregisters is that the feature-level consequences of optimizing against a grader are
derivable from base-policy statistics before any RL. Optimizing pulls the policy toward the
exponentially tilted family ``pi_lambda proportional to pi_0 exp(lambda r)``, and for an exponential
family the initial per-feature drift is a base-policy covariance,
``d/dlambda E_lambda[f_i]|_0 = Cov_0(f_i, r) = chi_i`` (A12, the fluctuation-dissipation
identity). The ``f = r`` diagonal ``chi = Var_0(r)`` is exactly Razin's teacher variance
``TC = w_r^T Sigma_pi w_r`` (A3), the zeroth-order term of the same law. So the susceptibility
spectrum predicts which feature drifts and how fast, and best-of-n previews where the optimization is
headed with no gradient step at all.

This study runs the calibration first, on a synthetic-but-honest base-policy draw where the answer is
known by construction, so the instrument is validated before it is turned on a production model
(gate 1). The draw is a bank of activations with a planted reward direction and a feature
bank of projections, for which every ``chi_i = u_i^T Sigma_pi w_r`` and the teacher variance
``w_r^T Sigma_pi w_r`` are analytically known. The two things S3 must recover are (H1) the planted
susceptibility spectrum, computed as ``Cov_0(f_i, r)`` and matched against the analytic values, and
(H2, the loops arm) the rank agreement between the predicted chi and the realized per-feature drift
under a best-of-n ladder.

Two arms consume ``reward_lens.loops`` (built concurrently). They are lazily imported: the
susceptibility can also be computed inline from the same covariance definition, so H1 runs either way;
the best-of-n transfer arm needs ``loops.bon`` and is gated with a recorded note when it is not
importable. Tail metrology (a Hill estimate of the reward tail and the critical pressure ``lambda_c``,
A4/A5) is computed inline here because ``measure.indices.tail`` is not yet importable, and is
calibrated against a planted Pareto draw whose tail exponent is known.

The headline: reward hacking obeys a fluctuation-dissipation theorem, and Razin's teacher variance is
the zeroth-order term. The kill criterion is real: if the predicted chi does not rank-correlate with
the realized best-of-n drift, the Gibbs family is the wrong idealization of practical RLHF, which is
itself a publishable negative (DESIGN S3 kill criterion).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reward_lens.core.evidence import Uncertainty, make_evidence
from reward_lens.core.provenance import Provenance
from reward_lens.core.reading import Reading
from reward_lens.core.types import Access, Component, GaugeStatus, SubjectRef
from reward_lens.measure.indices.chi import susceptibility
from reward_lens.measure.indices.tail import tail_estimate
from reward_lens.record.schema import Run
from reward_lens.stats import spearman_with_ci
from reward_lens.studies.spec import (
    Hypothesis,
    KillCriterion,
    Prediction,
    StudyResult,
    StudySpec,
    SubjectQuery,
)
from studies._retype import (
    MetricSpec,
    ScienceRetype,
    count_trajectories,
    declared_conditions,
    trajectory_features,
)

_VERSION = "1.0"

# The best-of-n ladder used for the transfer arm. It stays within the per-prompt bank size so the
# plug-in expected-maximum estimator is well resolved, and the realized per-feature drift is read at
# a single moderate rung where the tilt is gentle and the drift is monotone in chi.
_BON_LADDER: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
_BON_DRIFT_N: int = 16


def build_spec() -> StudySpec:
    """The frozen S3 spec: the susceptibility calibration and the best-of-n transfer arm, both T9."""
    return StudySpec(
        id="s03-thermo",
        title="Reward thermodynamics: base-policy covariances recover the planted susceptibility "
        "spectrum and predict the best-of-n feature-drift ordering",
        science="S03-thermo",
        hypotheses=(
            Hypothesis(
                id="H1-chi-recovery",
                statement="the susceptibility spectrum chi_i = Cov_0(f_i, r) recovers the planted "
                "per-feature susceptibility on a construction where chi is analytically known, and "
                "the f = r diagonal recovers the teacher variance w_r^T Sigma_pi w_r (calibration)",
                prediction=Prediction(metric="chi_recovery_corr", comparator=">", threshold=0.95),
                scoreboard_row="T9",
            ),
            Hypothesis(
                id="H2-bon-transfer",
                statement="the predicted susceptibility chi rank-correlates with the realized "
                "per-feature drift under a best-of-n ladder (the fluctuation-dissipation prediction "
                "transfers to the quasi-static optimization frontier)",
                prediction=Prediction(metric="chi_bon_spearman", comparator=">", threshold=0.3),
                scoreboard_row="T9",
            ),
        ),
        analysis="studies.s03_thermo.analysis.analyze",
        subjects=SubjectQuery(
            extra={
                "note": "synthetic-but-honest base-policy draw with a planted reward direction and a "
                "feature bank of projections, so chi and the teacher variance are analytically known; "
                "the four campaign RMs + Skywork with a real base-policy draw are the GPU-gated "
                "follow-on (DESIGN S3 first experiment)"
            }
        ),
        kill_criteria=(
            KillCriterion(
                id="K1-gibbs-wrong",
                metric="chi_bon_spearman",
                comparator="<",
                threshold=0.3,
                description="predicted chi fails to rank-correlate with the realized best-of-n drift, "
                "so the Gibbs (exponential-family) idealization is wrong for this optimizer and the "
                "non-equilibrium residual is the result (a publishable negative)",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# The synthetic-but-honest base-policy draw
# ---------------------------------------------------------------------------


@dataclass
class _Draw:
    """A base-policy draw with a planted, analytically known susceptibility structure.

    ``scores`` is the pooled per-sample reward ``r = h w_r`` (shape ``(N,)``); ``features`` the pooled
    per-sample feature-bank activations ``f_i = h u_i`` (shape ``(N, k)``). ``reward_banks`` and
    ``feature_banks`` hold the same samples reshaped per prompt (``(P, m)`` and ``(P, m, k)``) for the
    best-of-n arm. ``planted_chi`` is the analytic ``Cov_0(f_i, r) = u_i^T Sigma w_r``,
    ``planted_teacher_var`` the analytic ``w_r^T Sigma w_r``, ``gold_cov`` the measured
    ``Cov_0(f_i, gold)`` for a second (gold) reward direction, and ``planted_hack_modes`` the features
    with analytic ``chi_i > 0`` while ``Cov_0(f_i, gold) <= 0`` (the predicted hack modes, A12).
    """

    scores: np.ndarray
    features: np.ndarray
    reward_banks: np.ndarray
    feature_banks: np.ndarray
    feature_names: list[str]
    planted_chi: np.ndarray
    planted_teacher_var: float
    gold_cov: np.ndarray
    planted_hack_modes: list[str]


def _base_policy_draw(
    d: int = 24,
    k: int = 12,
    n_prompts: int = 40,
    samples_per_prompt: int = 64,
    proxy_gold_cos: float = 0.6,
    seed: int = 0,
) -> _Draw:
    """Draw base-policy activations with a planted reward direction and a feature bank of projections.

    The activations are zero-mean Gaussian with a fixed diagonal covariance ``Sigma`` (a non-isotropic
    on-distribution covariance), so ``r = h w_r`` has population variance exactly
    ``w_r^T Sigma w_r`` and each feature ``f_i = h u_i`` has population covariance with the reward
    exactly ``u_i^T Sigma w_r``. Both are the analytic ground truth the instrument is graded against.
    A second reward direction ``w_g`` at a fixed angle from ``w_r`` stands in for the gold objective,
    so the predicted hack modes (features the proxy loves but the gold does not) are known by
    construction. The feature directions are an orthonormal set, which spreads the planted chi across
    positive, near-zero, and negative values so the recovery correlation is a real test.
    """
    rng = np.random.default_rng(seed)

    # A non-isotropic on-distribution covariance, so the teacher-variance identity is non-trivial.
    scales = np.linspace(0.5, 3.0, d)
    sigma_diag = scales**2

    w_r = rng.standard_normal(d)
    w_r /= np.linalg.norm(w_r)
    # A gold direction at a controlled angle: it agrees with the proxy in part and disagrees in part,
    # which is what makes some high-susceptibility features genuine hack modes.
    g_raw = rng.standard_normal(d)
    g_perp = g_raw - (g_raw @ w_r) * w_r
    g_perp /= np.linalg.norm(g_perp)
    w_g = proxy_gold_cos * w_r + np.sqrt(max(1.0 - proxy_gold_cos**2, 0.0)) * g_perp
    w_g /= np.linalg.norm(w_g)

    # An orthonormal feature-direction bank.
    q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    u = q[:, :k]

    n = n_prompts * samples_per_prompt
    z = rng.standard_normal((n, d))
    h = z * scales  # Cov(h) = diag(scales**2) = Sigma

    scores = h @ w_r
    gold = h @ w_g
    features = h @ u

    planted_chi = (u * sigma_diag[:, None]).T @ w_r  # u_i^T Sigma w_r
    planted_teacher_var = float(w_r @ (sigma_diag * w_r))  # w_r^T Sigma w_r
    planted_gold_cov = (u * sigma_diag[:, None]).T @ w_g  # u_i^T Sigma w_g

    # The measured gold covariance is what a real study would have; the flagger runs on it.
    gold_c = gold - gold.mean()
    feat_c = features - features.mean(axis=0, keepdims=True)
    gold_cov = (feat_c * gold_c[:, None]).mean(axis=0)

    names = [f"feat{i:02d}" for i in range(k)]
    planted_hack_modes = [
        names[i] for i in range(k) if planted_chi[i] > 0 and planted_gold_cov[i] <= 0
    ]

    reward_banks = scores.reshape(n_prompts, samples_per_prompt)
    feature_banks = features.reshape(n_prompts, samples_per_prompt, k)

    return _Draw(
        scores=scores.astype(np.float64),
        features=features.astype(np.float64),
        reward_banks=reward_banks.astype(np.float64),
        feature_banks=feature_banks.astype(np.float64),
        feature_names=names,
        planted_chi=planted_chi.astype(np.float64),
        planted_teacher_var=planted_teacher_var,
        gold_cov=gold_cov.astype(np.float64),
        planted_hack_modes=planted_hack_modes,
    )


# ---------------------------------------------------------------------------
# Susceptibility, hack modes, tail metrology (loops arm, with inline fallbacks)
# ---------------------------------------------------------------------------


def _susceptibility_inline(scores: np.ndarray, features: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute ``chi_i = Cov_0(f_i, r)`` and the ``f = r`` diagonal ``Var_0(r)`` from the draw.

    The population covariance (``ddof = 0``) is the estimator of the theory object, because the
    covariance is an expectation under the base policy (A12). This is the exact definition
    ``reward_lens.loops.tilt.susceptibility`` implements; it is reproduced here only so the calibration
    still runs when the loops subsystem is not yet importable.
    """
    r = np.asarray(scores, dtype=np.float64).ravel()
    f = np.asarray(features, dtype=np.float64)
    r_c = r - r.mean()
    f_c = f - f.mean(axis=0, keepdims=True)
    chi = (f_c * r_c[:, None]).mean(axis=0)
    teacher_var = float((r_c * r_c).mean())
    return chi, teacher_var


def _flag_hack_modes_inline(chi: np.ndarray, gold_cov: np.ndarray, names: list[str]) -> list[str]:
    """Names of the predicted hack modes: ``chi_i > 0`` while ``Cov_0(f_i, gold) <= 0`` (A12)."""
    return [names[i] for i in range(len(names)) if chi[i] > 0 and gold_cov[i] <= 0]


def _hill_tail_index(x: np.ndarray, tail_frac: float = 0.1) -> float:
    """Hill estimate of the right-tail index ``alpha`` of positive data (A4).

    The Hill estimator uses the top ``k`` order statistics: with ``X_(n) >= ... >= X_(n-k)`` the
    reciprocal tail index is ``gamma = (1/k) sum_{i=0}^{k-1} log X_(n-i) - log X_(n-k)`` and
    ``alpha = 1/gamma``. A polynomial (heavy) tail has finite ``alpha``; a light tail pushes ``alpha``
    large. Only the positive part of the data enters, since the estimator is defined on a positive
    right tail. This is the simple inline estimate the design asks for while ``measure.indices.tail``
    is not importable; it is not the full peaks-over-threshold observable that will replace it.
    """
    xs = np.sort(np.asarray(x, dtype=np.float64).ravel())
    xs = xs[xs > 0]
    n = xs.size
    if n < 10:
        return float("nan")
    k = max(5, int(n * tail_frac))
    k = min(k, n - 1)
    top = xs[-(k + 1) :]
    threshold = top[0]
    gamma = float(np.mean(np.log(top[1:]) - np.log(threshold)))
    if gamma <= 0:
        return float("inf")
    return 1.0 / gamma


def _critical_lambda_inline(scores: np.ndarray, tail_quantile: float = 0.9) -> float:
    """Critical pressure ``lambda_c = 1 / tau`` from the mean excess of the reward tail (A5).

    The peaks-over-threshold scale ``tau`` is the mean excess of the scores above their
    ``tail_quantile`` quantile; an exponential right tail with scale ``tau`` has critical pressure
    ``1 / tau`` (A5). Returns ``inf`` when the excess is degenerate. Reproduced inline only as
    the fallback for ``loops.tilt.critical_lambda_from_tail``.
    """
    r = np.asarray(scores, dtype=np.float64).ravel()
    threshold = float(np.quantile(r, tail_quantile))
    excess = r[r > threshold] - threshold
    if excess.size < 2:
        return float("inf")
    tau = float(excess.mean())
    return 1.0 / tau if tau > 0 else float("inf")


def _bon_feature_drift(reward_banks: np.ndarray, feature_banks: np.ndarray, n: int) -> np.ndarray:
    """Realized per-feature drift under best-of-n, from the same plug-in identity ``loops.bon`` uses.

    Best-of-n keeps, per prompt, the completion the reward model scores highest. The exact expected
    feature value of that selected completion under ``n`` draws with replacement puts weight
    ``(rank/m)^n - ((rank-1)/m)^n`` on the completion holding the ``rank``-th smallest reward (the
    probability the maximum of ``n`` draws is that order statistic; the same Abel-summation weight
    ``loops.bon.expected_bon_reward`` applies to the reward). The drift is the selected-feature mean
    minus the base-policy feature mean, averaged over prompts. This is the realized-drift arm the
    predicted chi is scored against.
    """
    n_prompts, m = reward_banks.shape
    k = feature_banks.shape[2]
    base_mean = feature_banks.reshape(-1, k).mean(axis=0)
    ranks = np.arange(1, m + 1, dtype=np.float64)
    weights = (ranks / m) ** n - ((ranks - 1.0) / m) ** n
    acc = np.zeros(k, dtype=np.float64)
    for p in range(n_prompts):
        order = np.argsort(reward_banks[p])  # ascending by reward
        acc += weights @ feature_banks[p][order]
    return acc / n_prompts - base_mean


def _pareto_hill_calibration(alpha_true: float = 3.0, n: int = 6000, seed: int = 7) -> float:
    """Draw a Pareto sample with known tail index and return the Hill estimate on it.

    A Pareto(alpha) sample has ``P(X > x) proportional to x^{-alpha}``, so its Hill tail index
    recovers ``alpha``. Recovering the planted exponent within tolerance is what calibrates the inline
    Hill estimator before it is read on the (light-tailed) base-policy reward.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=n)
    x = (1.0 - u) ** (-1.0 / alpha_true)  # inverse-CDF Pareto draw, x_min = 1
    return _hill_tail_index(x)


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------


def analyze(run) -> StudyResult:
    """Run the susceptibility calibration and the best-of-n transfer arm, recording REGISTERED Evidence.

    H1 recovers the planted susceptibility spectrum and the teacher variance from a construction where
    both are analytically known. H2 (the loops arm) scores the rank agreement between predicted chi and
    the realized best-of-n feature drift; it is gated with a recorded note when ``loops.bon`` is not
    importable. Tail metrology reports a Hill index and the critical pressure, calibrated against a
    planted Pareto tail.
    """
    study_id = run.study.study_id
    subject = SubjectRef(extra={"study": study_id})
    draw = _base_policy_draw()

    # The root Evidence documents the synthetic scored draw; the susceptibility and best-of-n Evidence
    # descend from it, so the store stays a DAG rooted at the base-policy sampling.
    ev_draw = make_evidence(
        observable="S03.BasePolicyDraw",
        observable_version=_VERSION,
        subject=subject,
        value={
            "n_samples": int(draw.scores.size),
            "n_prompts": int(draw.reward_banks.shape[0]),
            "samples_per_prompt": int(draw.reward_banks.shape[1]),
            "n_features": int(draw.features.shape[1]),
            "planted_teacher_variance": draw.planted_teacher_var,
            "planted_hack_modes": list(draw.planted_hack_modes),
        },
        uncertainty=Uncertainty(n=int(draw.scores.size), method="none"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id),
        registered=True,
    )
    run.record(ev_draw)

    # Susceptibility spectrum: prefer loops.tilt (built concurrently), fall back to the inline
    # covariance so the calibration always runs.
    chi_parent = ev_draw.id
    tilt_source = "inline"
    try:
        from reward_lens.loops.tilt import flag_hack_modes, susceptibility

        chi_ev = susceptibility(
            draw.scores,
            draw.features,
            draw.feature_names,
            subject=subject,
            parents=(ev_draw.id,),
        )
        run.record(chi_ev)
        chi = np.asarray(chi_ev.value.chi, dtype=np.float64)
        teacher_var = float(chi_ev.value.teacher_variance)
        flagged = flag_hack_modes(chi_ev.value, draw.gold_cov)
        chi_parent = chi_ev.id
        tilt_source = "loops.tilt.susceptibility"
    except Exception:
        chi, teacher_var = _susceptibility_inline(draw.scores, draw.features)
        flagged = _flag_hack_modes_inline(chi, draw.gold_cov, draw.feature_names)

    chi_recovery_corr = float(np.corrcoef(chi, draw.planted_chi)[0, 1])
    spread = float(np.std(draw.planted_chi))
    chi_recovery_nrmse = float(np.sqrt(np.mean((chi - draw.planted_chi) ** 2)) / spread)
    tv_rel_error = abs(teacher_var - draw.planted_teacher_var) / abs(draw.planted_teacher_var)
    hack_mode_match = set(flagged) == set(draw.planted_hack_modes)

    ev_chi = make_evidence(
        observable="S03.Susceptibility",
        observable_version=_VERSION,
        subject=subject,
        value={
            "chi_recovery_corr": chi_recovery_corr,
            "chi_recovery_nrmse": chi_recovery_nrmse,
            "teacher_variance": teacher_var,
            "planted_teacher_variance": draw.planted_teacher_var,
            "teacher_variance_rel_error": tv_rel_error,
            "hack_modes_flagged": list(flagged),
            "hack_mode_match": bool(hack_mode_match),
            "source": tilt_source,
        },
        uncertainty=Uncertainty(n=int(draw.scores.size), method="none"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=(chi_parent,)),
        registered=True,
    )
    run.record(ev_chi)

    # Tail metrology: a Hill index on the base-policy reward and the critical pressure, calibrated
    # against a planted Pareto tail. Prefer loops.tilt for lambda_c, fall back inline.
    try:
        from reward_lens.loops.tilt import critical_lambda_from_tail

        lambda_c = float(critical_lambda_from_tail(draw.scores))
        lambda_source = "loops.tilt.critical_lambda_from_tail"
    except Exception:
        lambda_c = _critical_lambda_inline(draw.scores)
        lambda_source = "inline"
    hill_base = _hill_tail_index(draw.scores)
    pareto_alpha_true = 3.0
    hill_pareto = _pareto_hill_calibration(alpha_true=pareto_alpha_true)
    pareto_hill_rel_error = abs(hill_pareto - pareto_alpha_true) / pareto_alpha_true

    ev_tail = make_evidence(
        observable="S03.TailMetrology",
        observable_version=_VERSION,
        subject=subject,
        value={
            "hill_tail_index_base": hill_base,
            "lambda_c": lambda_c,
            "lambda_c_source": lambda_source,
            "pareto_alpha_true": pareto_alpha_true,
            "hill_pareto": hill_pareto,
            "pareto_hill_rel_error": pareto_hill_rel_error,
            "note": "measure.indices.tail is not importable; the Hill index and lambda_c are computed "
            "inline and calibrated on a planted Pareto tail. The full critical-KL table (A5) "
            "needs the tilted free-energy integral, which is the S16 safety-case arm.",
        },
        uncertainty=Uncertainty(n=int(draw.scores.size), method="none"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_draw.id,)),
        registered=True,
    )
    run.record(ev_tail)

    metrics: dict[str, float] = {
        "chi_recovery_corr": chi_recovery_corr,
        "chi_recovery_nrmse": chi_recovery_nrmse,
        "teacher_variance_rel_error": tv_rel_error,
        "hack_mode_match": 1.0 if hack_mode_match else 0.0,
        "lambda_c": lambda_c,
        "hill_tail_index_base": hill_base,
        "pareto_hill_rel_error": pareto_hill_rel_error,
    }

    # The best-of-n transfer arm: predicted chi against realized per-feature drift. It needs
    # loops.bon, so it is gated with a recorded note when that is not importable.
    bon_status = "gated"
    try:
        from reward_lens.loops.bon import bon_ladder

        ev_bon = bon_ladder(
            draw.reward_banks, ns=_BON_LADDER, subject=subject, parents=(ev_draw.id,)
        )
        run.record(ev_bon)
        drift = _bon_feature_drift(draw.reward_banks, draw.feature_banks, _BON_DRIFT_N)
        sp = spearman_with_ci(chi, drift, n_resamples=2000, seed=0)
        chi_bon_spearman = float(sp.point)
        metrics["chi_bon_spearman"] = chi_bon_spearman
        bon_status = "ran"
        ev_transfer = make_evidence(
            observable="S03.BoNTransfer",
            observable_version=_VERSION,
            subject=subject,
            value={
                "chi_bon_spearman": chi_bon_spearman,
                "ci_low": float(sp.ci_low),
                "ci_high": float(sp.ci_high),
                "bon_drift_n": _BON_DRIFT_N,
                "predictor": "measured chi = Cov_0(f_i, r)",
                "realized": f"best-of-{_BON_DRIFT_N} per-feature drift over "
                f"{draw.reward_banks.shape[0]} prompts",
            },
            uncertainty=Uncertainty(n=int(chi.size), method="bootstrap"),
            gauge=GaugeStatus.INVARIANT,
            provenance=Provenance(study=study_id, parents=(ev_chi.id, ev_bon.id)),
            registered=True,
        )
        run.record(ev_transfer)
    except Exception as exc:
        ev_gate = make_evidence(
            observable="S03.BoNTransfer",
            observable_version=_VERSION,
            subject=subject,
            value={
                "gated": True,
                "needs": "reward_lens.loops.bon",
                "reason": f"loops.bon not importable ({type(exc).__name__}: {exc})",
                "note": "the susceptibility calibration (H1) ran; the best-of-n transfer arm (H2) is "
                "inconclusive-because-gated until loops.bon is importable. No metric is emitted, so "
                "the kill criterion cannot fire on a gated arm.",
            },
            gauge=GaugeStatus.INVARIANT,
            provenance=Provenance(study=study_id, parents=(ev_chi.id,)),
            registered=True,
        )
        run.record(ev_gate)

    if bon_status == "ran":
        transfer = (
            f"predicted chi rank-correlates with the realized best-of-{_BON_DRIFT_N} drift at "
            f"Spearman {metrics['chi_bon_spearman']:.3f}"
        )
    else:
        transfer = "the best-of-n transfer arm is gated (loops.bon not importable) and inconclusive"

    summary = (
        f"On a base-policy draw where the susceptibility is analytically known, chi = Cov_0(f_i, r) "
        f"recovered the planted spectrum at correlation {chi_recovery_corr:.3f} (nrmse "
        f"{chi_recovery_nrmse:.3f}) and the f = r diagonal recovered the teacher variance "
        f"w_r^T Sigma_pi w_r to {tv_rel_error:.1%}; the hack-mode flagger "
        f"{'matched' if hack_mode_match else 'did not match'} the planted modes. The inline Hill "
        f"estimator recovered a planted Pareto tail to {pareto_hill_rel_error:.1%} and put the base "
        f"reward's critical pressure at lambda_c = {lambda_c:.3g}. {transfer}."
    )

    return StudyResult(outcomes={}, metrics=metrics, summary=summary)


# ---------------------------------------------------------------------------
# The retype: S3 on the kernel
# ---------------------------------------------------------------------------

RETYPE = ScienceRetype(
    science="s03_thermo",
    spec=build_spec(),
    headline="selection.differential_S",
    destination=(
        "measure/frontier/ at Level 0. The susceptibility chi_i = Cov_0(f_i, r) is the "
        "selection differential S at zero pressure, which is why both bind to the same registered "
        "id and why the frontier's tilt family is the same object this science calls the Gibbs "
        "family."
    ),
    needs={Component.GRADER: Access.RECORD, Component.RECORD: Access.RECORD},
    metrics=(
        MetricSpec(
            metric="chi_recovery_corr",
            quantity="selection.differential_S",
            arc="chi-calibration",
            frame="planted",
            source="organism",
            note=(
                "the correlation between the recovered susceptibility spectrum and the analytic "
                "one on a draw where every chi_i = u_i' Sigma_pi w_r is known by construction. The "
                "arc that produces it is the one that measures chi on the planted organism; the "
                "number itself is a recovery correlation rather than a covariance, which is why it "
                "is source='organism' and never stamped on a record reading."
            ),
        ),
        MetricSpec(
            metric="chi_bon_spearman",
            quantity="selection.differential_S",
            arc="chi-transfer",
            arm="best-of-n",
            note=(
                "the rank agreement between predicted chi and the realised per-feature drift under "
                "a best-of-n ladder. Answerable from a record whose kind is 'bon'; a training "
                "record is a different tilt path and reading one as the other is the substitution "
                "this science's kill criterion is about."
            ),
        ),
    ),
    arc_requires={"chi-transfer": ("chi-calibration",)},
)


def read(run: Run) -> Reading:
    """S3 against a real record: the susceptibility spectrum, measured rather than planted.

    The one thing S3 needs from a record is a feature bank and a reward on the same trajectories,
    because chi_i = Cov_0(f_i, r) is a covariance between them and nothing else. That is a low bar
    and most recorders clear it, which makes this the science that transfers to a record most
    directly of the fifteen.

    Scope limit, three lines in: the base-policy expectation is taken over the earliest quarter of
    the run, not over the base policy itself. Those agree while the policy has not moved far, and
    the record's declared NEAR_POLICY condition is what says whether that held. On a run that has
    already drifted, this is a susceptibility about the current policy and not about pi_0.
    """
    if (refusal := RETYPE.access_refusal(run, remedy=_ACCESS_REMEDY)) is not None:
        return refusal

    names, rows, rewards = trajectory_features(run)
    n_traj = count_trajectories(run)
    if not names or not rewards:
        return RETYPE.incomplete(
            field="a feature bank on the trajectories" if rewards else "trajectory scores",
            subject=f"run {run.id}",
            remedy=(
                "record per-trajectory features alongside the score, through the tap's feature "
                "hook or by attaching them to the trajectory. chi is a covariance between features "
                "and reward, so with no features there is no spectrum, only its f = r diagonal."
            ),
            trajectories=n_traj,
            features=len(names),
        )

    features = np.asarray(rows, dtype=np.float64)
    reward = np.asarray(rewards, dtype=np.float64)
    chi = susceptibility(features, reward)
    induced_variance = float(reward.var(ddof=0))
    tail = tail_estimate(reward)

    measured = {
        "chi_max_abs": (float(np.max(np.abs(chi))), "selection.differential_S"),
        "induced_variance": (induced_variance, "grader.induced_variance"),
        "reward_tail_index": (float(tail["shape_xi"]), "grader.tail_index"),
    }

    refusals = {
        "chi_recovery_corr": (
            "the recovery correlation is against an analytically known spectrum, and a record has "
            "no ground truth in it. Run the calibration arm through analyze(), which plants the "
            "spectrum it then recovers."
        )
    }
    if run.kind != "bon":
        refusals["chi_bon_spearman"] = (
            f"this record has kind {run.kind!r} and the transfer arm is defined on a best-of-n "
            f"ladder, where the realised drift is a plug-in order statistic at each rung. Point "
            f"this at a run recorded with kind='bon' over the same grader, or read the drift "
            f"against training step instead and accept that step and KL are different axes."
        )
    elif len(names) < 2:
        refusals["chi_bon_spearman"] = (
            f"the record carries {len(names)} feature and a rank agreement over one feature is not "
            f"defined. Record a bank of at least two independent features."
        )

    # chi_max == Var(r) to within rounding means the feature bank is the reward wearing another
    # name, so the spectrum has one entry and no content. It is worth saying out loud: a caller who
    # sees a susceptibility and does not notice it is the variance will read a degenerate bank as a
    # measurement.
    collinear = abs(float(np.max(np.abs(chi))) - induced_variance) <= 1e-9 * max(
        1.0, induced_variance
    )
    caveat = (
        " The largest susceptibility equals the induced reward variance exactly, which means the "
        "recorded feature bank is the realised reward itself: Cov(r, r) = Var(r). The spectrum has "
        "no content on this record, and recording features that are not the score is what would "
        "give it some."
        if collinear
        else ""
    )
    summary = (
        f"{n_traj} trajectories over {run.n_steps} steps, {len(names)} feature(s) "
        f"({', '.join(names)}): the largest susceptibility is {float(np.max(np.abs(chi))):.4g}, "
        f"the induced reward variance {induced_variance:.4g}, and the reward tail shape "
        f"{float(tail['shape_xi']):.4g} ({tail['regime']} regime). Both frozen metrics are refused "
        f"on this record, for different reasons.{caveat}"
    )
    return RETYPE.evidence(
        run,
        {},
        measured=measured,
        quantity="selection.differential_S",
        refusals=refusals,
        summary=summary,
        gauge=GaugeStatus.COVARIANT,
        features=list(names),
        chi=[float(c) for c in chi],
        n_trajectories=n_traj,
        feature_bank_collinear_with_reward=collinear,
        declared_regime=dict(declared_conditions(run)),
    )


_ACCESS_REMEDY = (
    "open a run written by the recorder whose grader scores were captured. S3 needs no activations: "
    "the susceptibility is a covariance between recorded features and recorded rewards."
)


__all__ = ["RETYPE", "build_spec", "analyze", "read"]

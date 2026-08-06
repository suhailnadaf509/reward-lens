"""S5 - Capacity Theory of Bias (DESIGN Part III, S5; scoreboard T12, coherence/Welch half).

The claim S5 preregisters is that a scalar grader aggregating K criteria through a d-dimensional
bottleneck cannot make those criteria independent once K exceeds the effective dimension, so some of
its biases are not learned at all: they are compressed into existence by the geometry of the head.
Three consequences follow, and each is checkable on a construction where the answer is known exactly.

The first is a floor. For any K unit criterion directions living in d dimensions, the largest
off-diagonal coherence ``mu_jk = v_j . v_k`` obeys the Welch bound
``max_{j!=k} |mu_jk| >= sqrt((K - d) / (d(K - 1)))`` once ``K > d`` (A9, faithful_to Welch
1974). Below that dimension there is room for orthogonality and the bound is vacuous; above it the
criteria must start to overlap, and an equiangular tight frame meets the floor with equality. So a
share of every over-packed grader's cross-criterion interference is obligatory before any data is
seen.

The second is that the interference is causal, not cosmetic. Steering criterion j's direction and
reading the change in criterion k's contribution defines the contamination ``C_jk``, and to first
order under a linear head this is exactly the coherence ``mu_jk`` (A9). So contamination
scales with coherence pair by pair, and on a planted rubric where both are known the relationship is
the linear identity ``C_jk = c * mu_jk``.

The third is that the surplus leaks. The dark reward (A10) is the fraction of ``Var(r)`` no
named criterion mediates, and capacity theory predicts it grows with ``K/d_eff`` as the reward tries
to carry more criteria than its effective dimension supports. The interference cross-terms are where
it hides, and a best-of-n policy that selects on the true reward mines exactly those terms: its gain
concentrates in the dark channel, invisible to an audit that reads only the per-criterion scores.

This study proves all three on planted geometries where the coherence, the effective dimension, and
the reward are exact by construction, in the corpus's discipline of calibrating the instrument before
turning it on a model (gate 1). It consumes the primitives already on disk:
``measure/indices/coherence`` (the Gram matrix, the Welch bound, ``d_eff``), ``geometry`` (the
participation ratio that reads ``d_eff``), ``measure/indices/dark_reward`` (the unmediated variance
fraction), and the organism foundry's planted-rubric generator (``rubric_organism`` at controlled
``(K, d, correlation)``). The contamination steer is computed inline, adding a small displacement
along criterion j's direction and reading the change in criterion k, which needs no external module;
the production path lazily imports ``interventions.steer.SteeringIntervention`` and, when it is
present, verifies the same first-order operation compiles as an addressed intervention. Running that
steer on a real multi-objective reward model's residual stream, the ArmoRM nineteen-objective
coherence matrix, the population mean-contamination Welch-curve fit, and the interference-hacking
best-of-n test on real models are all population/GPU-gated: they are built and proven on organisms
here, marked, and skipped rather than invented.

The headline: some reward-model biases are compressed into existence, the geometry of fitting K
values through a d-dimensional head predicts which pairs of criteria contaminate each other, and
optimization mines exactly those cross-terms. The kill criterion is real: if measured contamination
were uncorrelated with coherence on planted-rubric organisms, where both are exact, the linear
superposition picture would be wrong for reward heads and the wreckage would go to nonlinear circuits
(DESIGN S5 kill criterion). S5 originates the coherence/Welch half of T12; S6 originates the Hodge
half separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reward_lens.core.evidence import Uncertainty, make_evidence
from reward_lens.core.provenance import Provenance
from reward_lens.core.reading import Reading
from reward_lens.core.types import Access, Component, GaugeStatus, Site, SubjectRef
from reward_lens.geometry import participation_ratio
from reward_lens.measure.indices.coherence import (
    coherence_matrix,
    effective_dimension,
    max_offdiagonal_coherence,
    welch_bound,
)
from reward_lens.measure.indices.dark_reward import dark_reward
from reward_lens.organisms import make_rubric_directions, rubric_organism
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
from studies._retype import MetricSpec, ScienceRetype, count_trajectories

_VERSION = "1.0"

# The exact-coherence anchor: a planted rubric organism with K < d criteria at a fixed pairwise
# cosine, where the coherence matrix off-diagonals must equal the planted correlation to machine
# precision. This is the K <= d branch where orthogonality has room and the Welch floor is vacuous.
_ANCHOR_K = 4
_ANCHOR_D = 8
_ANCHOR_CORRELATION = 0.3

# The over-packed regime (K > d) where the Welch floor bites. The simplex equiangular tight frames
# meet it with equality (a tolerance, not a margin); the random frames sit above it.
_ETF_ORDERS: tuple[int, ...] = (3, 4, 6, 10, 16)
_RANDOM_OVERPACKED: tuple[tuple[int, int], ...] = ((6, 3), (10, 4), (20, 6), (30, 8), (50, 10))

# The contamination arm: heterogeneous criteria (a random planted frame, K < d so the head is full
# rank and linear) give the pairwise coherence spread a constant-coherence rubric lacks, so the
# scaling of contamination with coherence is a real correlation rather than a degenerate constant.
_CONTAM_K = 8
_CONTAM_D = 20
_CONTAM_STEER = 0.5
_CONTAM_READOUT_NOISE = 0.15

# The dark-reward and interference sweeps. Two near-orthonormal rungs (K <= d) anchor the low end at
# zero interference; the over-packed rungs rise. Modest K keeps the dark fraction off its ceiling so
# the gradient with K/d_eff is visible, not a single step.
_DARK_SWEEP: tuple[tuple[int, int, bool], ...] = (
    (5, 14, True),
    (4, 4, False),
    (5, 5, False),
    (6, 5, False),
    (7, 6, False),
    (8, 6, False),
    (10, 7, False),
    (12, 7, False),
    (14, 8, False),
    (16, 8, False),
)
_INTERFERENCE_ETA = 0.15  # per-pair coherence-weighted interference coefficient
_DARK_SEEDS = 10

# The interference-hacking sweep: over-packed frames where a best-of-n policy selects on the true
# reward and its gain is split into the audited (linear) and dark (interference) channels.
_HACK_SWEEP: tuple[tuple[int, int], ...] = (
    (4, 4),
    (6, 5),
    (8, 6),
    (10, 7),
    (14, 8),
    (20, 9),
    (28, 10),
)
_HACK_N = 32
_HACK_PROMPTS = 200
_HACK_SEEDS = 8

# Registered thresholds. The Welch floor is an inequality the measured coherence must not fall below
# (a tiny negative tolerance absorbs the floating-point equality of the tight frames); the ETF
# equality gap is a tolerance on the exact floor formula; the contamination correlation and the
# dark-reward trend are the two scaling predictions; the dark share is the hacking prediction.
_FLOOR_TOL = -1e-9
_ETF_TOL = 1e-9
_CONTAM_CORR_MIN = 0.9
_CONTAM_CORR_KILL = 0.3
_DARK_SPEARMAN_MIN = 0.8
_HACK_SHARE_MIN = 0.5


def build_spec() -> StudySpec:
    """The frozen S5 spec: the Welch floor, contamination scaling, dark reward, and hacking (T12)."""
    return StudySpec(
        id="s05-capacity",
        title="Capacity theory of bias: the Welch floor on cross-criterion coherence, contamination "
        "that scales with coherence, and dark reward that grows with K/d_eff",
        science="S05-capacity",
        hypotheses=(
            Hypothesis(
                id="H1-welch-floor-holds",
                statement="on planted over-packed criterion frames (K > d_eff), the measured maximum "
                "off-diagonal coherence sits at or above the Welch floor sqrt((K - d)/(d(K - 1))), so "
                "the interference floor is obligatory geometry, not a fitting artifact",
                prediction=Prediction(
                    metric="welch_floor_min_slack", comparator=">=", threshold=_FLOOR_TOL
                ),
                scoreboard_row="T12",
            ),
            Hypothesis(
                id="H2-etf-meets-floor-exactly",
                statement="the simplex equiangular tight frames meet the Welch floor with equality, "
                "so the floor formula is verified against the planted geometry to machine precision",
                prediction=Prediction(
                    metric="etf_equality_gap_max", comparator="<", threshold=_ETF_TOL
                ),
                scoreboard_row="T12",
            ),
            Hypothesis(
                id="H3-contamination-scales-with-coherence",
                statement="steering criterion j and reading the change in criterion k, the "
                "contamination C_jk correlates with the coherence mu_jk across pairs on a planted "
                "rubric where both are exact (the first-order linear identity C_jk = c * mu_jk)",
                prediction=Prediction(
                    metric="contamination_coherence_corr",
                    comparator=">",
                    threshold=_CONTAM_CORR_MIN,
                ),
                scoreboard_row="T12",
            ),
            Hypothesis(
                id="H4-dark-reward-grows-with-k-over-deff",
                statement="sweeping K/d_eff on planted organisms, the unmediated fraction of Var(r) "
                "(the dark reward) rises, as capacity theory predicts the surplus leaks into the "
                "cross-criterion channel",
                prediction=Prediction(
                    metric="dark_reward_kdeff_spearman",
                    comparator=">",
                    threshold=_DARK_SPEARMAN_MIN,
                ),
                scoreboard_row="T12",
            ),
            Hypothesis(
                id="H5-policies-mine-interference",
                statement="a best-of-n policy selecting on the true reward gains reward chiefly in the "
                "dark interference channel, a hack class invisible to an audit that reads only the "
                "per-criterion scores",
                prediction=Prediction(
                    metric="interference_dark_share", comparator=">", threshold=_HACK_SHARE_MIN
                ),
                scoreboard_row="T12",
            ),
        ),
        analysis="studies.s05_capacity.analysis.analyze",
        subjects=SubjectQuery(
            organisms=("rubric",),
            extra={
                "note": "planted-rubric organisms and planted criterion frames at controlled (K, d, "
                "correlation), where coherence, d_eff, and reward are exact by construction; the "
                "ArmoRM nineteen-objective coherence matrix, the population mean-contamination "
                "Welch-curve fit, the SteeringIntervention residual-stream contamination, and the "
                "interference-hacking best-of-n on real models are the population/GPU-gated follow-on "
                "(DESIGN S5 first experiment)"
            },
        ),
        kill_criteria=(
            KillCriterion(
                id="K1-contamination-uncorrelated",
                metric="contamination_coherence_corr",
                comparator="<",
                threshold=_CONTAM_CORR_KILL,
                description="measured contamination is uncorrelated with coherence on a planted rubric "
                "where both are exact, so the linear superposition picture is wrong for reward heads "
                "and the mechanism must be handed to nonlinear circuits (a publishable negative that "
                "also weakens the corpus-wide linear-direction premise)",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Planted criterion geometries (exact coherence, exact d_eff)
# ---------------------------------------------------------------------------


def _simplex_etf(k: int) -> np.ndarray:
    """The simplex equiangular tight frame: ``k`` unit vectors in ``R^{k-1}`` (the Welch equality case).

    The ``k`` vertices of a regular simplex centered at the origin have pairwise inner product exactly
    ``-1/(k - 1)``, which is exactly ``welch_bound(k, k - 1)``, so this frame meets the floor with
    equality. It is built by centering the standard basis of ``R^k`` (which lands the points in the
    zero-sum hyperplane, a ``(k - 1)``-dimensional subspace) and reading their coordinates in an
    orthonormal basis of that hyperplane, then unit-normalizing the rows.
    """
    centered = np.eye(k) - 1.0 / k
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, : k - 1] * s[: k - 1]
    coords /= np.linalg.norm(coords, axis=1, keepdims=True)
    return np.asarray(coords, dtype=np.float64)


def _random_frame(k: int, d: int, seed: int) -> np.ndarray:
    """``k`` random unit criterion directions in ``R^d`` (a planted frame with heterogeneous coherence).

    The rows are independent standard-Gaussian directions, unit-normalized. When ``k > d`` this is an
    over-packed frame whose maximum coherence must clear the Welch floor; when ``k < d`` it is a
    full-rank heterogeneous rubric whose pairwise coherences spread across a range, which the
    contamination-scaling test needs.
    """
    rng = np.random.default_rng([int(seed), int(k), int(d)])
    v = rng.standard_normal((k, d))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.asarray(v, dtype=np.float64)


def _anchor_directions() -> tuple[np.ndarray, float]:
    """The planted rubric organism's criterion directions and the coherence they must show exactly.

    Consumes the foundry's generator 7 (``rubric_organism``) at controlled ``(K, d, correlation)``,
    then reads the planted criterion directions off the answer key. Because the generator plants an
    exact pairwise cosine, the coherence matrix of these directions must have every off-diagonal equal
    to ``correlation`` to machine precision, which is the K <= d calibration of the coherence reader.
    """
    _view, key = rubric_organism(K=_ANCHOR_K, d=_ANCHOR_D, correlation=_ANCHOR_CORRELATION, seed=0)
    assert key.true_directions is not None
    dirs = np.stack(
        [
            np.asarray(key.true_directions[f"criterion_{k}"], dtype=np.float64)
            for k in range(_ANCHOR_K)
        ]
    )
    return dirs, _ANCHOR_CORRELATION


# ---------------------------------------------------------------------------
# Arm 1: the Welch floor (over-packed frames at or above the floor; ETFs at equality)
# ---------------------------------------------------------------------------


@dataclass
class _FloorRow:
    label: str
    k: int
    d: int
    max_coherence: float
    welch_bound: float
    slack: float
    d_eff: float
    overpacked: bool


def _welch_floor_rows() -> list[_FloorRow]:
    """Measure the maximum coherence against the Welch floor on the over-packed planted frames.

    The simplex ETFs are the equality case (max coherence equals the bound); the random frames are the
    strict case (max coherence above the bound). Every frame here has ``K > d``, so the bound is a real
    floor, and each row also records ``d_eff`` (the participation ratio of the criteria Gram) to
    confirm the over-packed condition ``K > d_eff`` the floor is stated under.
    """
    rows: list[_FloorRow] = []
    for k in _ETF_ORDERS:
        v = _simplex_etf(k)
        d = k - 1
        mu = coherence_matrix(v)
        max_coh = max_offdiagonal_coherence(mu)
        bound = welch_bound(k, d)
        d_eff = effective_dimension(v)
        rows.append(
            _FloorRow(f"simplex-etf-{k}", k, d, max_coh, bound, max_coh - bound, d_eff, k > d_eff)
        )
    for k, d in _RANDOM_OVERPACKED:
        v = _random_frame(k, d, seed=0)
        mu = coherence_matrix(v)
        max_coh = max_offdiagonal_coherence(mu)
        bound = welch_bound(k, d)
        d_eff = effective_dimension(v)
        rows.append(
            _FloorRow(f"random-{k}x{d}", k, d, max_coh, bound, max_coh - bound, d_eff, k > d_eff)
        )
    return rows


# ---------------------------------------------------------------------------
# Arm 2: contamination scales with coherence (inline steer on a planted rubric)
# ---------------------------------------------------------------------------


def _inline_contamination(v: np.ndarray, latents: np.ndarray, steer: float) -> np.ndarray:
    """Contamination ``C[k, j]``: the mean change in criterion k when criterion j's direction is steered.

    Criterion k reads a latent ``h`` as ``s_k = v_k . h``. Steering criterion j adds ``steer * v_j`` to
    the latent, so criterion k's readout moves by ``steer * (v_k . v_j) = steer * mu_jk`` for every
    latent. Averaging that change over the planted population is the contamination, and under this
    linear head it equals ``steer * mu_jk`` exactly. This is the inline steer the design specifies
    (add a small displacement along criterion j and read criterion k); it needs no external module.
    """
    s0 = latents @ v.T
    k = v.shape[0]
    contam = np.zeros((k, k), dtype=np.float64)
    for j in range(k):
        s_steered = (latents + steer * v[j]) @ v.T
        contam[:, j] = (s_steered - s0).mean(axis=0)
    return contam


def _noisy_contamination(
    v: np.ndarray, latents: np.ndarray, steer: float, noise: float, seed: int
) -> np.ndarray:
    """The same contamination read through a noisy criterion readout, so the correlation is not a tautology.

    A real steer-and-read carries measurement noise: the clean and steered readouts of criterion k are
    each observed with independent additive noise, so the measured contamination is
    ``steer * mu_jk`` plus an error that averages down over the population. Correlating this noisy
    measurement with the coherence across pairs is a genuine estimate (high but not exactly one),
    which the exact-identity check below complements.
    """
    rng = np.random.default_rng([int(seed), 101])
    k = v.shape[0]
    s0 = latents @ v.T + rng.standard_normal((latents.shape[0], k)) * noise
    contam = np.zeros((k, k), dtype=np.float64)
    for j in range(k):
        s_steered = (latents + steer * v[j]) @ v.T + rng.standard_normal(
            (latents.shape[0], k)
        ) * noise
        contam[:, j] = (s_steered - s0).mean(axis=0)
    return contam


def _contamination_arm() -> dict:
    """Steer each criterion on a heterogeneous planted rubric and score contamination against coherence.

    Returns the exact first-order deviation ``max |C_jk - c * mu_jk|`` (which must be at machine
    precision for the linear head), the Pearson correlation and slope of the noisy contamination
    against the coherence across off-diagonal pairs (the registered scaling metric), and the frame the
    steer ran on.
    """
    v = _random_frame(_CONTAM_K, _CONTAM_D, seed=1)
    rng = np.random.default_rng([1, 202])
    latents = rng.standard_normal((400, _CONTAM_D))
    mu = coherence_matrix(v)
    off = ~np.eye(_CONTAM_K, dtype=bool)

    contam_exact = _inline_contamination(v, latents, _CONTAM_STEER)
    exact_deviation = float(np.max(np.abs(contam_exact - _CONTAM_STEER * mu)[off]))

    contam_noisy = _noisy_contamination(v, latents, _CONTAM_STEER, _CONTAM_READOUT_NOISE, seed=1)
    x = contam_noisy[off]
    y = (_CONTAM_STEER * mu)[off]
    corr = float(np.corrcoef(x, y)[0, 1])
    slope = float(np.polyfit(mu[off], contam_noisy[off], 1)[0])

    return {
        "contamination_coherence_corr": corr,
        "contamination_slope": slope,
        "steer": _CONTAM_STEER,
        "exact_first_order_deviation": exact_deviation,
        "n_pairs": int(off.sum()),
        "K": _CONTAM_K,
        "d": _CONTAM_D,
    }


# ---------------------------------------------------------------------------
# Arm 3: dark reward grows with K/d_eff (coherence-weighted interference)
# ---------------------------------------------------------------------------


def _reward_with_interference(
    v: np.ndarray, latents: np.ndarray, eta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build the criterion readouts and a reward that is their equal-weight sum plus interference.

    The reward is ``r = mean_k s_k + eta * sum_{j<k} mu_jk s_j s_k``. The linear part is what a
    per-criterion audit accounts for; the interference part is the coherence-weighted cross-terms,
    which are zero when the criteria are orthogonal and grow as over-packing forces the coherences up.
    Returns the readouts ``S`` (the named channels) and the reward ``r``.
    """
    s = latents @ v.T
    k = v.shape[0]
    r = s.mean(axis=1)
    inter = np.zeros(latents.shape[0], dtype=np.float64)
    mu = coherence_matrix(v)
    for j in range(k):
        for kk in range(j + 1, k):
            inter += mu[j, kk] * s[:, j] * s[:, kk]
    return s, r + eta * inter


def _dark_reward_sweep() -> dict:
    """Sweep K/d_eff and measure the dark reward (the unmediated variance fraction) at each rung.

    Each rung builds a planted frame (near-orthonormal for the under-packed anchors, random for the
    over-packed rungs), draws a latent population, forms the reward with coherence-weighted
    interference, and measures the dark reward as the fraction of ``Var(r)`` the named channels leave
    unexplained (``measure.indices.dark_reward``). The dark reward is averaged over seeds per rung and
    scored for a monotone rise against ``K/d_eff`` by Spearman correlation.
    """
    ratios: list[float] = []
    darks: list[float] = []
    rung_rows: list[dict] = []
    for k, d, orthonormal in _DARK_SWEEP:
        rung_dark: list[float] = []
        rung_deff: list[float] = []
        for seed in range(_DARK_SEEDS):
            if orthonormal and k <= d:
                v = make_rubric_directions(K=k, d=d, correlation=0.0, seed=seed)
            else:
                v = _random_frame(k, d, seed=seed)
            rng = np.random.default_rng([seed, 303, k, d])
            latents = rng.standard_normal((6000, d))
            s, r = _reward_with_interference(v, latents, _INTERFERENCE_ETA)
            rung_dark.append(dark_reward(r, s))
            rung_deff.append(participation_ratio(np.linalg.eigvalsh(coherence_matrix(v))))
        d_eff = float(np.mean(rung_deff))
        dark = float(np.mean(rung_dark))
        ratio = k / d_eff
        ratios.append(ratio)
        darks.append(dark)
        rung_rows.append(
            {"K": k, "d": d, "d_eff": d_eff, "k_over_deff": ratio, "dark_reward": dark}
        )
    sp = spearman_with_ci(ratios, darks, n_resamples=2000, seed=0)
    return {
        "dark_reward_kdeff_spearman": float(sp.point),
        "spearman_ci_low": float(sp.ci_low),
        "spearman_ci_high": float(sp.ci_high),
        "dark_low": float(darks[0]),
        "dark_high": float(darks[-1]),
        "rungs": rung_rows,
    }


# ---------------------------------------------------------------------------
# Arm 4: policies mine the interference terms (a hack invisible to per-criterion audits)
# ---------------------------------------------------------------------------


def _interference_hacking() -> dict:
    """Best-of-n on the true reward: how much of the gain lands in the audited vs the dark channel.

    Per prompt, a bank of responses is drawn, the reward (linear plus coherence-weighted interference)
    is formed, and a best-of-n policy selects the highest-reward response. The gain over the bank mean
    is split into the audited linear part (what a per-criterion audit attributes) and the dark
    interference part (what it cannot). The dark share of the gain is the hacking signal, and it is
    scored for a rise against K/d_eff, which capacity theory predicts because the interference channel
    swells with over-packing. The production interference-hacking test over a real model population
    would run this through ``loops.bon``; its availability is recorded, the real-model run is gated.
    """
    try:
        import reward_lens.loops.bon as _bon  # noqa: F401

        bon_available = True
    except Exception:
        bon_available = False

    ratios: list[float] = []
    shares: list[float] = []
    rung_rows: list[dict] = []
    for k, d in _HACK_SWEEP:
        lin_gains: list[float] = []
        dark_gains: list[float] = []
        deffs: list[float] = []
        for seed in range(_HACK_SEEDS):
            v = _random_frame(k, d, seed=seed)
            mu = coherence_matrix(v)
            rng = np.random.default_rng([seed, 404, k, d])
            lin_gain = 0.0
            dark_gain = 0.0
            for _ in range(_HACK_PROMPTS):
                latents = rng.standard_normal((_HACK_N, d))
                s = latents @ v.T
                lin = s.mean(axis=1)
                inter = np.zeros(_HACK_N, dtype=np.float64)
                for j in range(k):
                    for kk in range(j + 1, k):
                        inter += mu[j, kk] * s[:, j] * s[:, kk]
                inter *= _INTERFERENCE_ETA
                sel = int(np.argmax(lin + inter))  # best-of-n on the true reward
                lin_gain += lin[sel] - lin.mean()
                dark_gain += inter[sel] - inter.mean()
            lin_gains.append(lin_gain / _HACK_PROMPTS)
            dark_gains.append(dark_gain / _HACK_PROMPTS)
            deffs.append(participation_ratio(np.linalg.eigvalsh(mu)))
        mean_lin = float(np.mean(lin_gains))
        mean_dark = float(np.mean(dark_gains))
        total = mean_lin + mean_dark
        share = mean_dark / total if total > 0 else float("nan")
        d_eff = float(np.mean(deffs))
        ratios.append(k / d_eff)
        shares.append(share)
        rung_rows.append(
            {
                "K": k,
                "d": d,
                "k_over_deff": k / d_eff,
                "audited_gain": mean_lin,
                "dark_gain": mean_dark,
                "dark_share": share,
            }
        )
    sp = spearman_with_ci(ratios, shares, n_resamples=2000, seed=0)
    return {
        "interference_dark_share": float(np.mean(shares)),
        "dark_share_kdeff_spearman": float(sp.point),
        "dark_share_low": float(shares[0]),
        "dark_share_high": float(shares[-1]),
        "bon_available": bon_available,
        "rungs": rung_rows,
    }


# ---------------------------------------------------------------------------
# Production wiring and the gated real-model arms
# ---------------------------------------------------------------------------


def _production_contamination_status() -> dict:
    """Lazily import ``SteeringIntervention`` and verify the production contamination contract compiles.

    The inline steer proves the mechanism on the planted rubric. The production contamination arm runs
    the identical first-order operation as an addressed intervention on a real reward model's residual
    stream, which needs the model and is gated. Here the ``SteeringIntervention(direction, site,
    strength)`` contract is imported and compiled (so a fingerprinted, site-addressed steer is
    demonstrably constructible), but reading contamination through it requires activations the planted
    numpy organism does not carry, so the real-model run is marked pending rather than invented.
    """
    try:
        from reward_lens.interventions.steer import SteeringIntervention
    except Exception as exc:
        return {
            "steer_module": "absent",
            "reason": f"{type(exc).__name__}: {exc}",
            "note": "the inline readout-space steer proof stands; the production contamination arm is "
            "pending until interventions.steer is importable",
        }

    direction = _random_frame(2, 8, seed=7)[0]
    steer = SteeringIntervention(
        direction=direction, site=Site(0, "resid_post"), strength=_CONTAM_STEER
    )
    compiled = steer.compile(None)
    return {
        "steer_module": "present",
        "contract": "SteeringIntervention(direction, site, strength)",
        "fingerprint": steer.fingerprint(),
        "compiled_fingerprint": compiled.fingerprint,
        "note": "the SteeringIntervention contract imports and compiles to a fingerprinted, "
        "site-addressed steer; running it on a real multi-objective reward model's residual stream to "
        "read causal contamination is the GPU-gated production arm (the planted organism has no "
        "activations to mount into, so no real-model contamination number is invented here)",
    }


def _armorm_nineteen_objective_proof() -> dict:
    """Prove the nineteen-objective coherence path on a synthetic frame; gate the real ArmoRM head.

    ArmoRM exposes nineteen objective directions. The coherence matrix, the Welch floor, and ``d_eff``
    are computed here on a synthetic nineteen-criterion frame to prove the computation works at that
    shape, so the only missing input is the real head rows. Those rows need the model and are gated, so
    no ArmoRM coherence number is invented; this records that the reader is ready for them.
    """
    v = _random_frame(19, 12, seed=0)
    mu = coherence_matrix(v)
    max_coh = max_offdiagonal_coherence(mu)
    d_eff = effective_dimension(v)
    return {
        "n_objectives": 19,
        "synthetic_max_coherence": float(max_coh),
        "synthetic_welch_bound": float(welch_bound(19, 12)),
        "synthetic_d_eff": float(d_eff),
        "gated": True,
        "needs": "ArmoRM head rows (the nineteen real objective directions)",
        "note": "the nineteen-objective coherence matrix, Welch floor, and d_eff compute on a "
        "synthetic frame of the right shape; the real ArmoRM objective directions are the GPU-gated "
        "input and no real coherence number is invented",
    }


def _population_welch_curve(floor_rows: list[_FloorRow]) -> dict:
    """The planted-frame mean-contamination-vs-K/d_eff shape; the real-population fit is gated.

    Capacity theory predicts a population of graders traces a Welch curve of mean contamination against
    ``K/d_eff``. The shape is exhibited here on the planted over-packed frames (their mean absolute
    coherence against the Welch floor at each K/d_eff); fitting it across a real reward-model population
    needs the models and is gated, so no population number is invented.
    """
    ratios = [row.k / row.d_eff for row in floor_rows]
    mean_contam = []
    for row in floor_rows:
        v = (
            _simplex_etf(row.k)
            if row.label.startswith("simplex")
            else _random_frame(row.k, row.d, 0)
        )
        mu = coherence_matrix(v)
        off = ~np.eye(row.k, dtype=bool)
        mean_contam.append(float(np.mean(np.abs(mu[off]))))
    return {
        "planted_k_over_deff": ratios,
        "planted_mean_contamination": mean_contam,
        "gated": True,
        "needs": "a real reward-model population (Skywork, ArmoRM, Tulu-3, GRM/URM/QRM, ...)",
        "note": "the mean-contamination-vs-K/d_eff shape is exhibited on planted frames; the "
        "population Welch-curve fit across real reward models is the Atlas GPU-gated arm and no "
        "population number is invented",
    }


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------


def analyze(run) -> StudyResult:
    """Prove the Welch floor, the contamination scaling, the dark-reward growth, and the hacking arm.

    Every number here is exact by construction on planted geometry: the coherence, the effective
    dimension, and the reward are all known, so the passing checks are the deliverable. The four proof
    Evidences descend from a root that documents the planted frames, the production contamination arm
    verifies the SteeringIntervention contract, and the real-model arms are recorded as gated with no
    invented numbers. The headline capacity-law Evidence traces to the proofs and is the number a card
    or paper cites for T12.
    """
    study_id = run.study.study_id
    subject = SubjectRef(extra={"study": study_id})

    # Root: the exact-coherence anchor from the foundry's planted-rubric organism, documenting that
    # the coherence reader recovers the planted pairwise cosine to machine precision (the K <= d
    # branch, where orthogonality has room and the Welch floor is vacuous).
    anchor_dirs, anchor_corr = _anchor_directions()
    anchor_mu = coherence_matrix(anchor_dirs)
    anchor_off = anchor_mu[~np.eye(_ANCHOR_K, dtype=bool)]
    anchor_deviation = float(np.max(np.abs(anchor_off - anchor_corr)))
    anchor_bound = welch_bound(_ANCHOR_K, _ANCHOR_D)

    ev_root = make_evidence(
        observable="S05.PlantedRubric",
        observable_version=_VERSION,
        subject=subject,
        value={
            "anchor_K": _ANCHOR_K,
            "anchor_d": _ANCHOR_D,
            "anchor_correlation": anchor_corr,
            "anchor_coherence_deviation": anchor_deviation,
            "anchor_welch_bound": anchor_bound,
            "anchor_d_eff": float(effective_dimension(anchor_dirs)),
            "note": "planted rubric organism (foundry generator 7) at controlled (K, d, correlation); "
            "the coherence off-diagonals equal the planted correlation to machine precision and the "
            "Welch bound is zero because K <= d",
        },
        uncertainty=Uncertainty(n=_ANCHOR_K, method="none"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id),
        registered=True,
    )
    run.record(ev_root)

    # Arm 1: the Welch floor on over-packed frames, with the simplex ETFs at equality.
    floor_rows = _welch_floor_rows()
    min_slack = float(min(row.slack for row in floor_rows))
    etf_gap_max = float(
        max(abs(row.slack) for row in floor_rows if row.label.startswith("simplex"))
    )
    all_overpacked = all(row.overpacked for row in floor_rows)
    ev_floor = make_evidence(
        observable="S05.WelchFloor",
        observable_version=_VERSION,
        subject=subject,
        value={
            "welch_floor_min_slack": min_slack,
            "etf_equality_gap_max": etf_gap_max,
            "all_overpacked": all_overpacked,
            "rows": [
                {
                    "label": row.label,
                    "K": row.k,
                    "d": row.d,
                    "max_coherence": row.max_coherence,
                    "welch_bound": row.welch_bound,
                    "slack": row.slack,
                    "d_eff": row.d_eff,
                }
                for row in floor_rows
            ],
        },
        uncertainty=Uncertainty(n=len(floor_rows), method="none"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_root.id,)),
        registered=True,
    )
    run.record(ev_floor)

    # Arm 2: contamination scales with coherence (inline steer on a heterogeneous planted rubric).
    contam = _contamination_arm()
    ev_contam = make_evidence(
        observable="S05.Contamination",
        observable_version=_VERSION,
        subject=subject,
        value=contam,
        uncertainty=Uncertainty(n=contam["n_pairs"], method="none"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_root.id,)),
        registered=True,
    )
    run.record(ev_contam)

    # The production contamination arm: verify the SteeringIntervention contract, gate the real run.
    production = _production_contamination_status()
    ev_production = make_evidence(
        observable="S05.ProductionContamination",
        observable_version=_VERSION,
        subject=subject,
        value=production,
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_contam.id,)),
        registered=True,
    )
    run.record(ev_production)

    # Arm 3: dark reward grows with K/d_eff.
    dark = _dark_reward_sweep()
    ev_dark = make_evidence(
        observable="S05.DarkReward",
        observable_version=_VERSION,
        subject=subject,
        value=dark,
        uncertainty=Uncertainty(n=len(_DARK_SWEEP), method="bootstrap"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_root.id,)),
        registered=True,
    )
    run.record(ev_dark)

    # Arm 4: policies mine the interference terms.
    hacking = _interference_hacking()
    ev_hacking = make_evidence(
        observable="S05.InterferenceHacking",
        observable_version=_VERSION,
        subject=subject,
        value=hacking,
        uncertainty=Uncertainty(n=len(_HACK_SWEEP), method="bootstrap"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_root.id,)),
        registered=True,
    )
    run.record(ev_hacking)

    # The gated real-model arms: the nineteen-objective ArmoRM path and the population Welch-curve fit.
    armorm = _armorm_nineteen_objective_proof()
    population = _population_welch_curve(floor_rows)
    ev_gated = make_evidence(
        observable="S05.GatedRealArms",
        observable_version=_VERSION,
        subject=subject,
        value={"armorm_nineteen_objective": armorm, "population_welch_curve": population},
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_root.id, ev_floor.id)),
        registered=True,
    )
    run.record(ev_gated)

    metrics: dict[str, float] = {
        "welch_floor_min_slack": min_slack,
        "etf_equality_gap_max": etf_gap_max,
        "contamination_coherence_corr": contam["contamination_coherence_corr"],
        "contamination_slope": contam["contamination_slope"],
        "contamination_exact_deviation": contam["exact_first_order_deviation"],
        "dark_reward_kdeff_spearman": dark["dark_reward_kdeff_spearman"],
        "interference_dark_share": hacking["interference_dark_share"],
        "dark_share_kdeff_spearman": hacking["dark_share_kdeff_spearman"],
    }

    # The registered headline: the coherence/Welch capacity law, tracing to the four proof arms.
    ev_law = make_evidence(
        observable="S05.CapacityLaw",
        observable_version=_VERSION,
        subject=subject,
        value={
            "welch_floor_min_slack": min_slack,
            "etf_equality_gap_max": etf_gap_max,
            "contamination_coherence_corr": contam["contamination_coherence_corr"],
            "dark_reward_kdeff_spearman": dark["dark_reward_kdeff_spearman"],
            "interference_dark_share": hacking["interference_dark_share"],
            "scoreboard_row": "T12",
            "half": "coherence/Welch (S6 originates the T12-Hodge half separately)",
        },
        uncertainty=Uncertainty(n=len(floor_rows), method="none"),
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(
            study=study_id,
            parents=(ev_floor.id, ev_contam.id, ev_dark.id, ev_hacking.id),
        ),
        registered=True,
    )
    run.record(ev_law)

    summary = (
        f"On planted criterion frames where coherence and d_eff are exact, the maximum off-diagonal "
        f"coherence sat at or above the Welch floor on every over-packed frame (min slack "
        f"{min_slack:+.2e}) and the simplex tight frames met the floor with equality to "
        f"{etf_gap_max:.1e}. Steering criterion j and reading criterion k, contamination tracked "
        f"coherence at Pearson {contam['contamination_coherence_corr']:.3f} (slope "
        f"{contam['contamination_slope']:.3f} against a planted steer of {contam['steer']}, and the "
        f"first-order identity C_jk = c mu_jk held to {contam['exact_first_order_deviation']:.1e}). "
        f"Sweeping K/d_eff, the dark reward rose from {dark['dark_low']:.3f} to {dark['dark_high']:.3f} "
        f"(Spearman {dark['dark_reward_kdeff_spearman']:.3f}), and a best-of-n policy on the true "
        f"reward put {hacking['interference_dark_share']:.1%} of its gain in the dark interference "
        f"channel, rising with K/d_eff at Spearman {hacking['dark_share_kdeff_spearman']:.3f}. The "
        f"SteeringIntervention production contract compiled; the ArmoRM nineteen-objective matrix, the "
        f"population Welch-curve fit, and the real-model interference-hacking best-of-n are gated."
    )

    return StudyResult(outcomes={}, metrics=metrics, summary=summary)


# ---------------------------------------------------------------------------
# The retype: S5 on the kernel
# ---------------------------------------------------------------------------

RETYPE = ScienceRetype(
    science="s05_capacity",
    spec=build_spec(),
    headline="grader.criterion_coherence",
    destination=(
        "the Welch floor and the coherence matrix, feeding measure/decision/'s N8. "
        "grader.criterion_coherence names the Welch floor for (K, d) in its own definition, which "
        "is why four of the five metrics resolve to it and none needed a new id."
    ),
    needs={Component.GRADER: Access.RECORD, Component.RECORD: Access.RECORD},
    metrics=(
        MetricSpec(
            metric="welch_floor_min_slack",
            quantity="grader.criterion_coherence",
            arc="welch-floor",
            frame="random-frames",
            source="organism",
            note=(
                "the smallest margin by which a measured maximum coherence clears the Welch bound "
                "over the frame sweep. The bound is part of grader.criterion_coherence's stated "
                "definition, so this is a feature of that quantity rather than a separate one. It "
                "sweeps (K, d) configurations, which is a construction and not a record."
            ),
        ),
        MetricSpec(
            metric="etf_equality_gap_max",
            quantity="grader.criterion_coherence",
            arc="welch-floor",
            frame="equiangular-tight-frames",
            source="organism",
            note="the same quantity on the tight frames that meet the floor with equality.",
        ),
        MetricSpec(
            metric="contamination_coherence_corr",
            quantity="grader.criterion_coherence",
            arc="contamination",
            arm="readout-contamination",
            source="organism",
            note=(
                "how strongly cross-criterion contamination tracks measured coherence. Needs the "
                "criterion directions, so it is a repr.basis reading and a record at RECORD access "
                "carries scores rather than directions."
            ),
        ),
        MetricSpec(
            metric="dark_reward_kdeff_spearman",
            quantity="grader.dark_fraction",
            arc="dark-sweep",
            arm="k-over-deff",
            source="organism",
            note=(
                "the rank agreement between the dark fraction and K/d_eff across the sweep. The "
                "dark fraction itself is exactly computable from a record, and read reports it; "
                "the trend across configurations needs the sweep."
            ),
        ),
        MetricSpec(
            metric="interference_dark_share",
            quantity="grader.dark_fraction",
            arc="interference",
            arm="best-of-n-mining",
            source="organism",
            note=(
                "the share of a best-of-n policy's gain that arrives through the interference "
                "channel rather than the audited linear one. Needs the counterfactual decomposition "
                "of a gain into two channels, which a record does not carry."
            ),
        ),
    ),
    arc_requires={
        "contamination": ("welch-floor",),
        "dark-sweep": ("welch-floor",),
        "interference": ("dark-sweep",),
    },
)


def read(run: Run) -> Reading:
    """S5 against a real record: the dark fraction, computed from the score tree's named channels.

    One of S5's two quantities transfers to a record exactly and the other does not.
    ``grader.dark_fraction`` is one minus the R-squared of the reward regressed on the named-channel
    contributions, and a ``ScoreTree`` is precisely a set of named channels with a total, so the
    definition maps onto the record with nothing left over. ``grader.criterion_coherence`` is a Gram
    matrix of criterion *directions*, and directions live in the representation: at RECORD access
    there are scores and no vectors, and the correlation of two score columns is not the cosine of
    two readout vectors.

    Scope limit, three lines in: with one named channel the dark fraction is identically zero,
    because a single channel regressed on itself explains everything. That is a fact about the
    record and not about the grader, so it is refused rather than reported.
    """
    if (refusal := RETYPE.access_refusal(run, remedy=_ACCESS_REMEDY)) is not None:
        return refusal

    names, contributions, totals = _named_channels(run)
    n_traj = count_trajectories(run)
    if len(names) < 2:
        return RETYPE.incomplete(
            field=(
                f"second named score channel (the tree has {len(names)}: "
                f"{', '.join(names) or 'no leaves at all'})"
            ),
            subject=f"run {run.id}",
            remedy=(
                "score with a composed grader and record each component as its own leaf. With one "
                "channel the dark fraction is zero by construction and the coherence matrix is the "
                "scalar 1, so neither number means anything. A rubric of K criteria recorded as K "
                "leaves is what makes this run answerable."
            ),
            trajectories=n_traj,
            channels=list(names),
        )

    reward = np.asarray(totals, dtype=np.float64)
    named = np.asarray(contributions, dtype=np.float64)
    dark = float(dark_reward(reward, named))

    measured = {"dark_fraction": (dark, "grader.dark_fraction")}
    refusals = {
        m.metric: _ORGANISM_REFUSAL[m.metric]
        for m in RETYPE.metrics
        if m.metric in _ORGANISM_REFUSAL
    }

    summary = (
        f"{n_traj} trajectories over {run.n_steps} steps with {len(names)} named channels "
        f"({', '.join(names)}): the dark fraction is {dark:.4g}, the share of reward variance no "
        f"named channel explains. The coherence half of S5 is not answerable at RECORD access, "
        f"because it is a Gram matrix of criterion directions and this record carries scores."
    )
    return RETYPE.evidence(
        run,
        {},
        measured=measured,
        quantity="grader.dark_fraction",
        refusals=refusals,
        summary=summary,
        gauge=GaugeStatus.INVARIANT,
        channels=list(names),
        n_trajectories=n_traj,
    )


#: Every frozen S5 metric needs a sweep, a construction or a counterfactual. None is a record read.
_ORGANISM_REFUSAL = {
    "welch_floor_min_slack": (
        "the floor is checked over a sweep of (K, d) frame configurations built to order. A record "
        "is one configuration and carries no directions to measure coherence on. Run the sweep "
        "through analyze()."
    ),
    "etf_equality_gap_max": (
        "the equality case needs equiangular tight frames constructed at each K, which is a "
        "construction rather than an observation. Run it through analyze()."
    ),
    "contamination_coherence_corr": (
        "contamination is measured between criterion readout vectors, so it needs LINEAR_READOUT on "
        "the grader. Raise the access to the readout and supply the K criterion directions, or read "
        "this from the planted rubric organism through analyze()."
    ),
    "dark_reward_kdeff_spearman": (
        "the trend is across a sweep of (K, d) configurations and a record is one of them. The dark "
        "fraction of this configuration is reported as `dark_fraction`; the rank agreement needs "
        "the others."
    ),
    "interference_dark_share": (
        "this splits a best-of-n policy's gain into an audited linear channel and a dark "
        "interference channel, which needs both channels evaluated counterfactually on the same "
        "draws. A record carries the realised gain and not its decomposition."
    ),
}

_ACCESS_REMEDY = (
    "open a run written by the recorder whose per-component grader scores were captured. S5's dark "
    "fraction needs the named channels of the score tree and nothing deeper."
)


def _named_channels(run: Run) -> tuple[list[str], list[list[float]], list[float]]:
    """The score tree's leaves as an aligned matrix, with the composed total beside them.

    Trajectories missing any leaf are dropped rather than zero-filled. A zero-filled channel enters
    the regression as a constant, which inflates the explained share and pushes the dark fraction
    down: the direction that makes a grader look better audited than it is.
    """
    names: list[str] = []
    for step in run.steps:
        for group in step.groups:
            for traj in group.trajectories:
                for name, _ in _leaves_of(traj.scores):
                    if name not in names:
                        names.append(name)
        if names:
            break
    rows: list[list[float]] = []
    totals: list[float] = []
    for step in run.steps:
        for group in step.groups:
            for traj in group.trajectories:
                leaves = dict(_leaves_of(traj.scores))
                if any(n not in leaves for n in names):
                    continue
                rows.append([leaves[n] for n in names])
                totals.append(float(sum(leaves[n] for n in names)))
    return names, rows, totals


def _leaves_of(node: object) -> list[tuple[str, float]]:
    children = getattr(node, "children", None)
    if children:
        out: list[tuple[str, float]] = []
        for child in children:
            out.extend(_leaves_of(child))
        return out
    name, value = getattr(node, "name", None), getattr(node, "value", None)
    if name is None or value is None:
        return []
    return [(str(name), float(value))]


__all__ = ["RETYPE", "build_spec", "analyze", "read"]

"""S2 — Gauge and the scalar bottleneck (DESIGN Part III, S2; scoreboard T6, T8).

Two registered experiments run here, both over the geometry subsystem and both on controlled inputs
where the answer is known by construction, so the study calibrates the method before it is turned on
a production model.

Experiment A (T6, identifiability up to shift and scale): apply a synthetic reward-gauge transform
to a reward direction (a per-prompt shift, a positive affine map, and noise confined to the
estimated on-distribution null subspace, all of which preserve preferences), then show that the
canonicalized cosine is near one while the raw cosine has collapsed. This is E19's cos = 0.005
turned from an anomaly into a measurement: raw-coordinate orthogonality is a coordinate change, and
canonicalization sees through it. The real Skywork v0.1-to-v0.2 comparison is the same analysis with
the two models' actual reward directions and a shared frame; that variant needs the 8B score heads
and is GPU-gated, so it is recorded as a follow-on rather than run here.

Experiment B (T8, the scalar head cannot express intransitivity): plant a genuinely cyclic
preference structure from a known skew operator and show `PreferenceRankTest` recovers held-out
cyclic preferences well above the best transitive (scalar) baseline, which provably cannot express a
cycle. If learned preference were empirically rank-1, this excess would be near zero.

The analysis builds its Evidence directly and records it under the frozen study, so the reported
numbers are REGISTERED and each headline metric names the geometry Evidence it was derived from.
"""

from __future__ import annotations

import numpy as np

from reward_lens.core.evidence import make_evidence
from reward_lens.core.provenance import Provenance
from reward_lens.core.reading import Reading
from reward_lens.core.types import Access, Component, GaugeStatus, SubjectRef
from reward_lens.geometry import PreferenceRankTest, effective_angle, fit_frame
from reward_lens.record.schema import Run
from reward_lens.studies.spec import (
    Hypothesis,
    KillCriterion,
    Prediction,
    StudyResult,
    StudySpec,
    SubjectQuery,
)
from studies._retype import MetricSpec, ScienceRetype, count_trajectories, leaf_scores

_VERSION = "1.0"


def build_spec() -> StudySpec:
    """The frozen S2 spec: two hypotheses with registered predictions, addressing T6 and T8."""
    return StudySpec(
        id="s02-gauge",
        title="Gauge and the scalar bottleneck: canonicalized directions are stable, "
        "and learned preference is not rank-1",
        science="S02-gauge",
        hypotheses=(
            Hypothesis(
                id="H1-canonical-stability",
                statement="under a pure reward-gauge transform, the canonicalized cosine exceeds "
                "the raw cosine by a wide margin (the orthogonality is a coordinate change)",
                prediction=Prediction(metric="canonical_minus_raw", comparator=">", threshold=0.4),
                scoreboard_row="T6",
            ),
            Hypothesis(
                id="H2-cyclic-recovery",
                statement="a rank-k skew operator recovers held-out cyclic preferences the scalar "
                "head cannot express (learned preference is not rank-1)",
                prediction=Prediction(metric="cyclic_recovery", comparator=">", threshold=0.1),
                scoreboard_row="T8",
            ),
        ),
        analysis="studies.s02_gauge.analysis.analyze",
        subjects=SubjectQuery(
            extra={
                "note": "controlled synthetic organisms; the real Skywork "
                "v0.1/v0.2 angle is the GPU-gated follow-on"
            }
        ),
        kill_criteria=(
            KillCriterion(
                id="K1-linear-gauge-insufficient",
                metric="canonical_cos",
                comparator="<",
                threshold=0.3,
                description="canonicalized cosine stays near zero while preferences are preserved, "
                "so linear gauge theory is insufficient and E19 becomes a reward-multiplicity "
                "candidate",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Experiment A helpers: a valid reward-gauge transform
# ---------------------------------------------------------------------------


def _orthonormal(dim: int, seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, dim)))
    return q


def _gauge_scenario(d: int = 64, active_dim: int = 40, n: int = 600, seed: int = 0):
    """An activation matrix with an active and a near-null subspace, a planted reward direction in
    the active subspace, and the per-item margins. Mirrors the geometry gauge property test so the
    science runs on the same validated construction."""
    rng = np.random.default_rng(seed)
    q = _orthonormal(d, seed + 1)
    p_act, p_null = q[:, :active_dim], q[:, active_dim:]
    z_act = rng.standard_normal((n, active_dim)) * rng.uniform(0.5, 2.0, active_dim)
    z_null = rng.standard_normal((n, d - active_dim)) * 1e-3
    h = (z_act @ p_act.T + z_null @ p_null.T).astype(np.float32)
    w = p_act @ rng.standard_normal(active_dim)
    w = (w / np.linalg.norm(w)).astype(np.float32)
    margins = (h @ w).astype(np.float32)
    return h, w, p_act, p_null, margins


def _apply_gauge(h, w, p_act, p_null, seed: int = 7):
    """A pure reward-gauge transform of the direction: a positive scale, a rotation confined to the
    null subspace, and a null-space shift of the direction. None changes on-distribution
    preferences, so the reward function is unchanged and only the coordinates move."""
    rng = np.random.default_rng(seed)
    null_dim = p_null.shape[1]
    alpha = float(rng.uniform(1.5, 3.0))
    o_null = _orthonormal(null_dim, seed + 3)
    r = (p_act @ p_act.T + p_null @ o_null @ p_null.T).astype(np.float64)
    w_t = alpha * (r @ w) + (p_null @ rng.standard_normal(null_dim)) * 2.0
    w_t = (w_t / np.linalg.norm(w_t)).astype(np.float32)
    return w_t


# ---------------------------------------------------------------------------
# Experiment B helpers: a planted cyclic preference structure
# ---------------------------------------------------------------------------


def _planted_cyclic_dataset(n: int = 70, d: int = 8, seed: int = 0):
    """Activations plus preferences generated by a known rank-1 skew operator: a cyclic tournament
    the scalar head cannot express. With ``A = u q^T - q u^T``, ``s(i, j) = phi_i^T A phi_j`` is the
    cross product of the items' coordinates in the (u, q) plane, a rotational rock-paper-scissors no
    scalar order can reproduce. The low dimension keeps the cyclic plane dominant, which is what
    makes the intransitive structure recoverable well above the transitive baseline. Returns
    ``(phi, pairs)`` with each pair as (winner, loser)."""
    rng = np.random.default_rng(seed)
    phi = rng.standard_normal((n, d))
    basis, _ = np.linalg.qr(rng.standard_normal((d, 2)))
    u, q = basis[:, 0], basis[:, 1]
    a_true = np.outer(u, q) - np.outer(q, u)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(phi[i] @ a_true @ phi[j])
            if abs(s) < 1e-9:
                continue
            pairs.append((i, j) if s > 0 else (j, i))
    return phi, np.array(pairs, dtype=np.int64)


def analyze(run) -> StudyResult:
    """Run both experiments, record REGISTERED Evidence, and return the adjudicated metrics."""
    study_id = run.study.study_id
    subject = SubjectRef(extra={"study": study_id})

    # Experiment A: canonicalized vs raw cosine under a pure gauge transform.
    h, w, p_act, p_null, margins = _gauge_scenario()
    w_t = _apply_gauge(h, w, p_act, p_null)
    frame = fit_frame(h, margins=margins)
    ev_angle = effective_angle(w, w_t, frame, activations_for_bound=h, n_boot=200, seed=0)
    run.record(ev_angle)
    ar = ev_angle.value
    canonical_cos = abs(float(ar.canonical_cos))
    raw_cos = abs(float(ar.raw_cos))
    ev_stab = make_evidence(
        observable="S02.GaugeStability",
        observable_version=_VERSION,
        subject=subject,
        value={
            "canonical_cos": canonical_cos,
            "raw_cos": raw_cos,
            "canonical_minus_raw": canonical_cos - raw_cos,
            "regret_bound": float(ar.regret_bound),
        },
        gauge=GaugeStatus.COVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_angle.id,)),
        registered=True,
    )
    run.record(ev_stab)

    # Experiment B: cyclic recovery above the transitive baseline.
    phi, pairs = _planted_cyclic_dataset()
    ev_rank = PreferenceRankTest(phi, pairs, rank_k=1).run(test_frac=0.3, seed=0)
    run.record(ev_rank)
    rr = ev_rank.value
    ev_cyclic = make_evidence(
        observable="S02.PreferenceRank",
        observable_version=_VERSION,
        subject=subject,
        value={
            "cyclic_recovery": float(rr.cyclic_recovery),
            "transitive_acc": float(rr.transitive_acc),
            "skew_acc": float(rr.skew_acc),
            "effective_rank": int(rr.effective_rank),
        },
        gauge=GaugeStatus.INVARIANT,
        provenance=Provenance(study=study_id, parents=(ev_rank.id,)),
        registered=True,
    )
    run.record(ev_cyclic)

    return StudyResult(
        outcomes={},
        metrics={
            "canonical_cos": canonical_cos,
            "raw_cos": raw_cos,
            "canonical_minus_raw": canonical_cos - raw_cos,
            "cyclic_recovery": float(rr.cyclic_recovery),
        },
        summary=(
            f"Under a pure gauge transform the raw cosine fell to {raw_cos:.3f} while the "
            f"canonicalized cosine held at {canonical_cos:.3f}; a rank-1 skew operator recovered "
            f"held-out cyclic preferences with a {rr.cyclic_recovery:.3f} margin over the scalar "
            f"baseline, so learned preference is not rank-1."
        ),
    )


# ---------------------------------------------------------------------------
# The retype: S2 on the kernel
# ---------------------------------------------------------------------------

RETYPE = ScienceRetype(
    science="s02_gauge",
    spec=build_spec(),
    headline="grader.objective_geometry",
    destination=(
        "the invariance machinery in core/invariance.py, where `repr.basis` is the group this "
        "science is an argument for, and `grader.objective_geometry` in the registry, which is the "
        "cosine between two readout vectors that the canonicalisation is applied to. The gauge "
        "transform this study plants is a member of `repr.basis` by construction: a rotation "
        "confined to the estimated null subspace plus a positive scale, which is why the raw cosine "
        "moves and the canonical one does not. The cyclic arm lands nowhere yet."
    ),
    # GRADER:FORWARD, because both cosines are between readout vectors and the frame they are read
    # in is fitted on the grader's own activations. A record carries neither.
    needs={Component.GRADER: Access.FORWARD},
    metrics=(
        MetricSpec(
            metric="canonical_cos",
            quantity="grader.objective_geometry",
            arc="canonicalise",
            frame="canonical",
            source="organism",
            note=(
                "the cosine between two reward directions read in the frame fitted on the "
                "on-distribution activations. `grader.objective_geometry` is the pairwise cosine of "
                "a head's readout vectors and this is that cosine for the pair (w, gauge(w)); the "
                "frame is what makes it canonical, which is why the frame is on the subject rather "
                "than folded into the metric name."
            ),
        ),
        MetricSpec(
            metric="canonical_minus_raw",
            quantity="grader.objective_geometry",
            arc="canonicalise",
            frame="canonical-minus-raw",
            source="organism",
            note=(
                "the same pair of directions read in two frames and subtracted, so it carries the "
                "same unit as each term. It is a contrast rather than a third quantity: the "
                "prediction H1 registers is that the frame changes the cosine, and a contrast "
                "between two frames of one quantity is how that is stated."
            ),
        ),
        MetricSpec(
            metric="cyclic_recovery",
            arc="preference-rank",
            source="organism",
            gap=(
                "held-out pairwise accuracy of a rank-k skew operator minus the best transitive "
                "(scalar) model's on the same held-out pairs. Unit: accuracy difference, dimension "
                "1, per null, support [-1, 1]. Invariance: `group.permutation`, invariant, because "
                "relabelling the items permutes both models' predictions together; also invariant "
                "under `reward.affine`, since a monotone rescaling of the scalar score changes no "
                "ordering. Nothing registered fits. `baseline.margin` is the closest and is a "
                "different comparison: it is M3's margin over the six dumb baselines from the "
                "controls bank, measured on discrimination, and the baseline here is the scalar "
                "head itself, which is the object under test rather than a control. Recommend "
                "registering `grader.intransitivity_recoverable` with the definition above."
            ),
        ),
    ),
)


def read(run: Run) -> Reading:
    """S2 against a record, which cannot answer it, and the refusal is the useful part.

    Both experiments need something a `record/` object does not contain. Experiment A needs two
    readout vectors and the activations to fit a frame in; a record carries the scores a grader
    produced and not the grader. Experiment B needs pairwise preferences over the same items, and a
    training record scores each rollout once with no comparison in it.

    Scope limit, three lines in: even at GRADER:FORWARD this returns a refusal, because forward
    access buys activations and a callable head, and the canonical cosine additionally needs the
    readout vector itself. That is a `Capability.LINEAR_READOUT` question rather than an access
    rung, and the remedy says so.
    """
    if (refusal := RETYPE.access_refusal(run, remedy=_ACCESS_REMEDY)) is not None:
        return refusal
    leaves = leaf_scores(run, limit=256)
    return RETYPE.incomplete(
        field="readout vector, and no activations to fit a frame in",
        subject=(
            f"run {run.id}, which carries {count_trajectories(run)} scored trajectories under "
            f"{len(leaves)} grader leaf/leaves and no weights,"
        ),
        remedy=(
            "load the reward model and read its score head, then run this through analyze() or "
            "through geometry.effective_angle directly on the two directions you are comparing. A "
            "record is the output of a grader and the gauge question is about the grader's "
            "coordinates, so no amount of recording answers it."
        ),
        trajectories=count_trajectories(run),
        grader_leaves=len(leaves),
    )


_ACCESS_REMEDY = (
    "supply GRADER:FORWARD on a model whose score head is readable, or run analyze(), which plants "
    "the gauge transform it then sees through. The canonical cosine is a cosine between two readout "
    "vectors measured in a frame fitted on activations, and a RECORD-access run has neither."
)


__all__ = ["RETYPE", "build_spec", "analyze", "read"]

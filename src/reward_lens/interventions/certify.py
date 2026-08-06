"""Post-hoc certificates for erasers, returned as Evidence.

An eraser is a claim: "no linear probe recovers this concept from the erased feature." A claim is
worth exactly what certifies it, and the design makes the certificate the thing that lifts an
eraser out of EXPLORATORY. Two certificates live here.

The erasure certificate trains a fresh linear probe on *held-out* data (data the eraser was not fit
on) and reports its recovery AUC. If the probe cannot beat chance by more than a small margin the
erasure holds, and the certificate carries a :class:`~reward_lens.core.gates.CalibrationRef` so the
certified eraser rises to CALIBRATED; if the probe recovers the concept the certificate refuses to
calibrate and the Evidence stays EXPLORATORY (gate 1). That asymmetry is the whole
point: the certificate is not decoration, it genuinely discriminates a real erasure from a fake one.
A sham eraser (a random affine map of the same rank) leaves the concept linearly present, its
recovery AUC stays high, and it is denied calibration by the same code path that grants it to a real
LEACE erase.

The robustness certificate reports the attack budget an adversary needs to rebreak an erasure under
a stated attack family. Its search uses ``geometry.hessian.gradient_ascent_probe``, which is marked
SENSITIVE / dual-use (RK8): the arm below is defensive (it measures how hard the eraser is to
defeat, which is the fix, not the attack), but it touches the hack generator, so that generator is
lazy-imported by name, is never re-exported from this module, and, when it is absent, this arm is
gated honestly by skipping. It never fabricates a budget.

torch is not needed here at all; certificates are computed on captured numpy matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from reward_lens.core.evidence import Evidence, Uncertainty, make_evidence, register_payload
from reward_lens.core.gates import CalibrationRef
from reward_lens.core.provenance import Provenance, capture_provenance
from reward_lens.core.types import GaugeStatus, SubjectRef
from reward_lens.stats.roc import roc_pr

if TYPE_CHECKING:
    from reward_lens.interventions.erase import Eraser

# Mirrors ``geometry.hessian.SENSITIVE`` without importing that module at load time, so importing
# certify never pulls in the hack-generation surface. The robustness arm imports the generator
# lazily and stamps this marker onto the sensitive Evidence's provenance (RK8).
_SENSITIVE = "sensitive:dual-use"


# ---------------------------------------------------------------------------
# The linear probe (the certificate's instrument)
# ---------------------------------------------------------------------------


def _fit_logistic(
    X: np.ndarray, y: np.ndarray, *, l2: float = 1e-3, max_iter: int = 500
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit an L2-regularized logistic-regression probe by L-BFGS with an analytic gradient.

    Features are standardized with the training mean and standard deviation (returned, so the same
    transform is applied at evaluation). The objective is the mean logistic loss plus
    ``0.5 * l2 * ||w||^2``; the loss uses ``logaddexp`` and ``expit`` so it is stable for large
    logits. Returns ``(w, b, mean, std)``. A near-constant feature column gets unit scale rather
    than dividing by a vanishing standard deviation.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    Xs = (X - mean) / std
    n, d = Xs.shape

    def loss_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        w = theta[:d]
        b = theta[d]
        z = Xs @ w + b
        loss = float(np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * l2 * np.dot(w, w))
        p = expit(z)
        gw = Xs.T @ (p - y) / n + l2 * w
        gb = float(np.mean(p - y))
        return loss, np.concatenate([gw, [gb]])

    res = minimize(
        loss_grad, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": max_iter}
    )
    return res.x[:d], float(res.x[d]), mean, std


def probe_recovery_auc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    l2: float = 1e-3,
) -> float:
    """Train a linear probe for a binary concept on ``(X_train, y_train)``, return its held-out AUC.

    This is the certificate's core measurement and the same instrument the erasure proof uses to
    show a concept is decodable before erasure and at chance after it. A trained logistic probe
    orients itself toward the concept, so its evaluation AUC sits at or above 0.5 when any linear
    signal survives and collapses to 0.5 when none does. The AUC is the exact rank / Mann-Whitney
    statistic from ``stats.roc`` (ties handled, no threshold grid).
    """
    w, b, mean, std = _fit_logistic(X_train, y_train, l2=l2)
    scores = ((np.asarray(X_eval, dtype=np.float64) - mean) / std) @ w + b
    return float(roc_pr(scores, np.asarray(y_eval, dtype=np.float64).ravel()).auc)


def _as_columns(Z: np.ndarray) -> np.ndarray:
    """Concept labels as an ``(n, k)`` matrix of binary columns for per-concept probing."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    return Z


def _split(n: int, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic shuffle-split of ``n`` row indices into (probe-train, probe-eval)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(round(n * frac))
    return idx[:cut], idx[cut:]


# ---------------------------------------------------------------------------
# Erasure certificate
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class ErasureCertificate:
    """The payload of :func:`certify_erasure`.

    ``recovery_auc`` is the worst-case held-out probe AUC over the concept columns (the most
    recoverable concept sets the bar); ``per_concept_auc`` breaks it out. ``passed`` is
    ``recovery_auc <= threshold`` with ``threshold = 0.5 + eps``. ``eraser_fingerprint`` names the
    eraser certified and ``fit_data_id`` the data it was fit on, so the certificate is bound to a
    specific artifact and can never be read as certifying a different eraser. The counts record how
    much held-out data the probe saw.
    """

    recovery_auc: float
    per_concept_auc: list[float]
    threshold: float
    eps: float
    passed: bool
    n_holdout: int
    n_probe_train: int
    n_probe_eval: int
    eraser_fingerprint: str
    concept_id: str | None
    fit_data_id: str | None
    method: str = "probe-recovery"


def certify_erasure(
    eraser: "Eraser",
    X_holdout: np.ndarray,
    Z_holdout: np.ndarray,
    *,
    eps: float = 0.05,
    probe_train_frac: float = 0.5,
    l2: float = 1e-3,
    seed: int = 0,
    concept_id: str | None = None,
    provenance: Provenance | None = None,
) -> Evidence[ErasureCertificate]:
    """Certify an eraser by held-out probe recovery, as Evidence.

    The held-out features are erased, split into a probe-train and a probe-eval portion, and a fresh
    linear probe is trained and scored per concept column. The reported ``recovery_auc`` is the
    worst case across columns. If it is at most ``0.5 + eps`` the erasure holds on data the eraser
    never saw, and the returned Evidence carries a :class:`CalibrationRef`, so gate 1 lifts it to
    CALIBRATED; otherwise the Evidence carries no calibration and stays EXPLORATORY. That is the
    mechanism by which a certificate lifts an eraser above EXPLORATORY, and by which a fake erasure
    is refused: the pass/fail is computed from the same held-out probe either way.

    ``X_holdout`` must be disjoint from the eraser's fit data for the certificate to mean anything;
    the caller is responsible for that split, and ``eraser.fit_data_id`` records what to exclude.
    The certificate is INVARIANT (a scalar recovery statistic, not a cross-signal geometric
    quantity), so no frame is required.
    """
    X_holdout = np.asarray(X_holdout, dtype=np.float64)
    Z = _as_columns(Z_holdout)
    n = X_holdout.shape[0]
    X_er = eraser.apply(X_holdout)
    train_idx, eval_idx = _split(n, probe_train_frac, seed)

    per_concept: list[float] = []
    for j in range(Z.shape[1]):
        zj = (Z[:, j] > 0.5).astype(np.float64)  # binarize (one-hot columns are already 0/1)
        auc = probe_recovery_auc(
            X_er[train_idx], zj[train_idx], X_er[eval_idx], zj[eval_idx], l2=l2
        )
        per_concept.append(auc)

    recovery_auc = max(per_concept) if per_concept else float("nan")
    threshold = 0.5 + eps
    passed = bool(recovery_auc <= threshold)

    fp = eraser.fingerprint()
    concept = concept_id or eraser.concept_id
    value = ErasureCertificate(
        recovery_auc=float(recovery_auc),
        per_concept_auc=[float(a) for a in per_concept],
        threshold=float(threshold),
        eps=float(eps),
        passed=passed,
        n_holdout=int(n),
        n_probe_train=int(train_idx.size),
        n_probe_eval=int(eval_idx.size),
        eraser_fingerprint=fp,
        concept_id=concept,
        fit_data_id=eraser.fit_data_id,
    )
    # The certificate is the calibration: a passing recovery on held-out planted structure is
    # exactly the answer-key check gate 1 asks for. A failing certificate confers none.
    calibration = (
        CalibrationRef(
            scorecard_entry=fp,
            organism_family=str(concept) if concept is not None else "erasure-holdout",
            regime_match="held-out probe recovery",
            operating_point={"recovery_auc": float(recovery_auc), "threshold": float(threshold)},
        )
        if passed
        else None
    )
    subject = SubjectRef(
        interventions=(fp,),
        extra={"certifies": "eraser", "concept_id": concept, "fit_data_id": eraser.fit_data_id},
    )
    return make_evidence(
        observable="interventions.certify_erasure",
        observable_version="1",
        subject=subject,
        value=value,
        uncertainty=Uncertainty(n=int(eval_idx.size), method="held-out-probe"),
        gauge=GaugeStatus.INVARIANT,
        calibration=calibration,
        provenance=provenance,
    )


def eraser_evidence(
    eraser: "Eraser",
    certificate: Evidence[ErasureCertificate] | None = None,
    *,
    provenance: Provenance | None = None,
) -> Evidence[dict]:
    """Evidence describing an eraser, EXPLORATORY unless a passing certificate is attached.

    This makes the gate-1 rule concrete: an eraser on its own is an uncalibrated artifact,
    so the Evidence describing it is EXPLORATORY; hand it a passing :func:`certify_erasure` result
    and the eraser inherits that certificate's calibration and rises to CALIBRATED. A failing or
    absent certificate leaves it EXPLORATORY. The value is a small provenance record (fingerprint,
    rank, fit-data id); the point of this function is the trust level, not the payload.
    """
    calibration = None
    if certificate is not None and getattr(certificate.value, "passed", False):
        calibration = certificate.calibration
    value = {
        "eraser_fingerprint": eraser.fingerprint(),
        "rank": int(eraser.rank),
        "dim": int(eraser.dim),
        "fit_data_id": eraser.fit_data_id,
        "concept_id": eraser.concept_id,
        "certificate": certificate.id if certificate is not None else None,
    }
    # The certificate is the parent Evidence this record derives its trust from (I5); parents live
    # on the Provenance, so thread the certificate id through there rather than as a bare argument.
    parents = (certificate.id,) if certificate is not None else ()
    prov = provenance or capture_provenance(parents=parents)
    return make_evidence(
        observable="interventions.eraser",
        observable_version="1",
        subject=SubjectRef(
            interventions=(eraser.fingerprint(),), extra={"fit_data_id": eraser.fit_data_id}
        ),
        value=value,
        gauge=GaugeStatus.INVARIANT,
        calibration=calibration,
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# Robustness certificate (SENSITIVE arm: touches the hack generator, RK8)
# ---------------------------------------------------------------------------


@register_payload
@dataclass
class RobustnessCertificate:
    """The payload of :func:`certify_robustness`. SENSITIVE arm.

    ``attack_family`` names the perturbation set searched; ``budgets`` and ``recovered_auc`` are the
    per-budget probe recovery after attack; ``budget_to_rebreak`` is the smallest budget at which
    recovery reaches ``rebreak_auc``, or None if no tested budget did. ``skipped`` is True when the
    dual-use attack generator was unavailable, in which case ``budget_to_rebreak`` is None and
    ``reason`` says why; the certificate never invents a budget. ``sensitivity`` marks the artifact
    dual-use so the store and cards exclude it from public exports (RK8).
    """

    attack_family: str
    budgets: list[float]
    recovered_auc: list[float]
    budget_to_rebreak: float | None
    rebreak_auc: float
    eraser_fingerprint: str
    skipped: bool = False
    reason: str = ""
    sensitivity: str = _SENSITIVE


def _load_attack() -> Callable[..., Any] | None:
    """Lazily resolve the dual-use ``gradient_ascent_probe`` by name, or None if unavailable (RK8).

    Kept as a seam so importing this module never imports the hack-generation surface, and so the
    honest skip path (attack generator absent) is exercisable: a caller or test supplies its own
    loader returning None to force the gate.
    """
    try:
        from reward_lens.geometry.hessian import gradient_ascent_probe

        return gradient_ascent_probe
    except ImportError:  # pragma: no cover - exercised via an injected loader in tests
        return None


def certify_robustness(
    eraser: "Eraser",
    X_holdout: np.ndarray,
    z_holdout: np.ndarray,
    concept_direction: np.ndarray,
    *,
    budgets: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0),
    rebreak_auc: float = 0.6,
    probe_train_frac: float = 0.5,
    steps: int = 2,
    l2: float = 1e-3,
    seed: int = 0,
    attack_loader: Callable[[], Callable[..., Any] | None] | None = None,
    provenance: Provenance | None = None,
) -> Evidence[RobustnessCertificate]:
    """Certify how much attack budget rebreaks an erasure, as Evidence. SENSITIVE.

    The stated attack family is a per-sample L2 perturbation of norm at most ``B`` that pushes each
    erased feature toward its concept class along ``concept_direction``. For each budget in
    ``budgets`` the search uses the dual-use ``geometry.hessian.gradient_ascent_probe`` (imported
    lazily, never re-exported) to move each sample under the norm-ball constraint, then measures the
    probe recovery AUC on the perturbed features. ``budget_to_rebreak`` is the smallest budget whose
    recovery reaches ``rebreak_auc``.

    This arm is defensive: it quantifies the eraser's resistance, which is what a defender needs to
    know. It is nonetheless marked SENSITIVE because it exercises the hack generator. If that
    generator is unavailable (``attack_loader`` returns None, or the import fails), the certificate
    is gated honestly: it returns a skipped Evidence with ``budget_to_rebreak = None`` and never
    fabricates a budget. The certificate is INVARIANT, so no frame is required.
    """
    loader = attack_loader or _load_attack
    attack = loader()
    fp = eraser.fingerprint()
    attack_family = "per_sample_l2_toward_concept"

    if attack is None:
        value = RobustnessCertificate(
            attack_family=attack_family,
            budgets=[float(b) for b in budgets],
            recovered_auc=[],
            budget_to_rebreak=None,
            rebreak_auc=float(rebreak_auc),
            eraser_fingerprint=fp,
            skipped=True,
            reason="dual-use attack generator (geometry.hessian.gradient_ascent_probe) unavailable",
        )
        return make_evidence(
            observable="interventions.certify_robustness",
            observable_version="1",
            subject=SubjectRef(interventions=(fp,), extra={"certifies": "robustness"}),
            value=value,
            gauge=GaugeStatus.INVARIANT,
            provenance=provenance or Provenance(extra={"sensitivity": _SENSITIVE}),
        )

    X_holdout = np.asarray(X_holdout, dtype=np.float64)
    labels = (np.asarray(z_holdout, dtype=np.float64).ravel() > 0.5).astype(np.float64)
    d_hat = np.asarray(concept_direction, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(d_hat))
    if norm > 1e-12:
        d_hat = d_hat / norm
    signs = 2.0 * labels - 1.0

    X_er = eraser.apply(X_holdout)
    n = X_er.shape[0]
    train_idx, eval_idx = _split(n, probe_train_frac, seed)

    recovered: list[float] = []
    for b in budgets:
        if b <= 0.0:
            perturbed = X_er
        else:
            perturbed = X_er.copy()
            for i in range(n):
                # grad of the per-sample reward s_i * (e . d_hat) w.r.t. e is the constant s_i d_hat;
                # gradient_ascent_probe ascends it and clamps the move to the norm ball of radius b.
                s_i = float(signs[i])

                def grad_fn(e: np.ndarray, _s: float = s_i) -> np.ndarray:
                    return _s * d_hat

                trace = attack(
                    grad_fn, X_er[i], int(max(1, steps)), constraint=float(b), lr=float(b)
                )
                perturbed[i] = np.asarray(trace.value.final_embeddings, dtype=np.float64).ravel()
        auc = probe_recovery_auc(
            perturbed[train_idx], labels[train_idx], perturbed[eval_idx], labels[eval_idx], l2=l2
        )
        recovered.append(float(auc))

    budget_to_rebreak: float | None = None
    for b, auc in zip(budgets, recovered):
        if auc >= rebreak_auc:
            budget_to_rebreak = float(b)
            break

    value = RobustnessCertificate(
        attack_family=attack_family,
        budgets=[float(b) for b in budgets],
        recovered_auc=recovered,
        budget_to_rebreak=budget_to_rebreak,
        rebreak_auc=float(rebreak_auc),
        eraser_fingerprint=fp,
        skipped=False,
    )
    return make_evidence(
        observable="interventions.certify_robustness",
        observable_version="1",
        subject=SubjectRef(interventions=(fp,), extra={"certifies": "robustness"}),
        value=value,
        gauge=GaugeStatus.INVARIANT,
        provenance=provenance or Provenance(extra={"sensitivity": _SENSITIVE}),
    )


# gradient_ascent_probe is deliberately NOT re-exported here (SENSITIVE / dual-use, RK8).
__all__ = [
    "probe_recovery_auc",
    "ErasureCertificate",
    "certify_erasure",
    "eraser_evidence",
    "RobustnessCertificate",
    "certify_robustness",
]

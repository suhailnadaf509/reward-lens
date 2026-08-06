"""The probe factory: linear concept probes with grouped CV and scorecard binding.

A concept probe is a linear readout of a signal's activations, trained to predict whether a
concept is present. The v1 primitive for a concept direction is the mean difference between the
positive and negative activations (`concepts/vectors.concept_direction`); a probe is its
supervised generalization, a logistic readout whose weight vector is the concept direction and
whose held-out AUC is an honest measure of how decodable the concept is at that site.

Three disciplines separate a trustworthy probe from a leaky one, and this module builds all three
in rather than leaving them to the caller.

- Seed-level cross-validation. Preference data is full of clones: the chosen and rejected sides of
  one pair, and a corruption and its parent, share a lineage seed. Splitting those across folds
  leaks the answer, so the CV here groups by seed id and never puts two examples of one seed in
  different folds. The no-leak property is asserted by construction in the tests.
- Class-balance handling. A concept present in a tenth of the examples must not be learned away as
  "always absent"; the solver takes inverse-frequency sample weights so the minority class carries
  its share of the loss.
- Automatic scorecard binding. If a matching organism family is supplied (a planted-structure
  answer key the probe can be graded against), the factory runs the answer-key ROC through
  `organisms.scorecard.MethodScorecard` and attaches the resulting `CalibrationRef` (gate 1). If
  none is available it attaches ``calibration: None``, which taints downstream Evidence to
  EXPLORATORY and leaves the gap as a visible TODO on the card. It never invents a calibration
  number, so a probe that was never graded cannot masquerade as one that was.

The output is a persisted `Direction`: a named, sited, unit-normalized fp32 vector
that knows its training data and its calibration reference (or the honest absence of one). The
direction is stored as Evidence so it is a first-class, provenance-carrying store citizen, the
same live-object / registered-artifact split the geometry frame uses.

The linear algebra is pure numpy, so the whole factory runs and is proven on CPU. The only place
torch appears is `capture_probe_inputs`, which reads activations off a real signal and is imported
lazily; the fit itself never touches torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from reward_lens.core.evidence import Evidence, Uncertainty, make_evidence, register_payload
from reward_lens.core.gates import CalibrationRef
from reward_lens.core.provenance import capture_provenance
from reward_lens.core.types import (
    DatasetID,
    DirectionID,
    GaugeStatus,
    Site,
    SubjectRef,
    content_hash,
)
from reward_lens.stats.roc import roc_pr

if TYPE_CHECKING:  # torch and the store are only needed on the model-capture / persist paths.
    from reward_lens.core.store import EvidenceStore
    from reward_lens.organisms.spec import AnswerKey
    from reward_lens.signals.base import RewardSignal

_PROBE_VERSION = "1.0"


# ---------------------------------------------------------------------------
# The persisted Direction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Direction:
    """A named, sited concept direction with its calibration reference.

    ``vector`` is unit-normalized fp32 at ``site``. ``method`` records how it was estimated
    (``"probe_lr"`` for a logistic probe, ``"contrast_mean"`` for the mean-difference primitive, and
    so on) so a direction never loses the provenance of how it was made. ``train_data`` is the
    `DatasetID` of the captures it was fit on, because a direction that does not know its training
    data cannot be reused honestly. ``calibration`` is the `CalibrationRef` from the answer-key
    scorecard, or ``None`` when the probe was never graded against a planted structure; a direction
    with ``calibration is None`` can be used but taints downstream Evidence to EXPLORATORY.

    ``meta`` carries the fit's read-only summary (held-out AUC, cross-validated AUC and its spread,
    the site the sweep chose, the class counts). It is descriptive, not part of the direction's
    identity beyond what the id already hashes.
    """

    id: DirectionID
    name: str
    site: Site
    vector: np.ndarray
    method: str
    train_data: DatasetID | None
    calibration: CalibrationRef | None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def d_model(self) -> int:
        """Ambient activation dimension the direction lives in."""
        return int(np.asarray(self.vector).shape[-1])

    @property
    def is_calibrated(self) -> bool:
        """Whether the direction carries an answer-key calibration reference (gate 1)."""
        return self.calibration is not None


@register_payload
@dataclass
class DirectionArtifact:
    """The serializable payload form of a `Direction`.

    A `Direction` holds a live `Site`; this artifact holds its canonical dict plus the fp32 vector,
    so it round-trips exactly through the evidence store's value codec. ``direction_evidence`` wraps
    one of these in an `Evidence` so a fitted direction is persisted with full provenance, and the
    `DirectionID` in ``direction_id`` is the content hash the store dedups on.
    """

    direction_id: str
    name: str
    site: dict[str, Any]
    vector: np.ndarray
    method: str
    train_data: str | None
    calibration: dict[str, Any] | None
    meta: dict[str, Any] = field(default_factory=dict)


def _unit_fp32(vector: np.ndarray) -> np.ndarray:
    """Unit-normalize a vector to fp32, leaving a near-zero vector unscaled rather than blowing up."""
    v = np.asarray(vector, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(v))
    if norm > 1e-12:
        v = v / norm
    return v.astype(np.float32)


def make_direction(
    *,
    name: str,
    site: Site,
    vector: np.ndarray,
    method: str,
    train_data: DatasetID | None,
    calibration: CalibrationRef | None = None,
    meta: dict[str, Any] | None = None,
) -> Direction:
    """Build a `Direction`, unit-normalizing the vector and computing its content-derived id.

    The id hashes the structural content (name, site, method, training-data id, calibration
    reference) together with the fp32 vector, so two identical directions from identical inputs
    share a `DirectionID` and the store dedups them. The vector is normalized here, so a caller
    passing a raw probe weight or a raw mean difference both land on a unit direction.
    """
    v = _unit_fp32(vector)
    cal = calibration.__canonical__() if calibration is not None else None
    material = {
        "name": name,
        "site": site.__canonical__(),
        "method": method,
        "train_data": train_data,
        "calibration": cal,
        "vector": v.tolist(),
    }
    direction_id = DirectionID(content_hash(material, "dir"))
    return Direction(
        id=direction_id,
        name=name,
        site=site,
        vector=v,
        method=method,
        train_data=train_data,
        calibration=calibration,
        meta=dict(meta or {}),
    )


def direction_evidence(
    direction: Direction,
    *,
    signals: tuple[str, ...] = (),
    parents: tuple[str, ...] = (),
    n: int | None = None,
) -> Evidence:
    """Wrap a `Direction` as COVARIANT Evidence so a fitted direction is a store citizen.

    A direction is a covariant quantity (it transforms with the residual-stream basis, gate 2), so
    the Evidence is typed COVARIANT and any cross-signal comparison of two directions will be forced
    through a frame. The Evidence carries the direction's `CalibrationRef` on its ``calibration``
    field, so the trust level falls out of the gate: a graded direction is CALIBRATED, an ungraded
    one EXPLORATORY. ``parents`` names the scorecard Evidence the calibration reference points at, so
    the store's DAG links the direction to the answer-key ROC that certified it.
    """
    artifact = DirectionArtifact(
        direction_id=str(direction.id),
        name=direction.name,
        site=direction.site.__canonical__(),
        vector=np.asarray(direction.vector, dtype=np.float32),
        method=direction.method,
        train_data=None if direction.train_data is None else str(direction.train_data),
        calibration=(
            direction.calibration.__canonical__() if direction.calibration is not None else None
        ),
        meta=dict(direction.meta),
    )
    subject = SubjectRef(
        signals=tuple(signals),
        dataset=direction.train_data,
        readout=direction.name,
        extra={"kind": "concept-direction", "method": direction.method},
    )
    provenance = capture_provenance(
        config={"name": direction.name, "method": direction.method},
        parents=tuple(parents),
    )
    return make_evidence(
        observable=f"ConceptProbe[{direction.name}]",
        observable_version=_PROBE_VERSION,
        subject=subject,
        value=artifact,
        uncertainty=Uncertainty(n=n, method="grouped-cv-roc"),
        gauge=GaugeStatus.COVARIANT,
        calibration=direction.calibration,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Captures: the substrate-free input the fit consumes
# ---------------------------------------------------------------------------


@dataclass
class SiteCaptures:
    """Per-site activation matrices with labels and CV grouping (the probe's substrate-free input).

    ``features`` maps each `Site` to an ``(n, d)`` fp32 activation matrix; every site shares the row
    ordering, so row ``i`` is the same example at every site. ``labels`` is the ``(n,)`` binary
    target (1 = concept present). ``groups`` is the ``(n,)`` seed id per row: rows sharing a seed are
    clones and the cross-validation keeps them in the same fold, which is what stops a probe from
    scoring itself on a copy of its training data.

    This is deliberately numpy-only: the whole factory runs on it without torch, so a planted
    synthetic capture (where the concept is a known direction by construction) proves the probe
    recovers what it should. `capture_probe_inputs` builds one of these from a real signal.
    """

    features: dict[Site, np.ndarray]
    labels: np.ndarray
    groups: np.ndarray
    dataset_id: DatasetID | None = None
    answer_key: "AnswerKey | None" = None
    name: str = "captures"

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels).ravel().astype(np.int64)
        self.groups = np.asarray(self.groups).ravel()
        n = self.labels.shape[0]
        if self.groups.shape[0] != n:
            raise ValueError(f"labels ({n}) and groups ({self.groups.shape[0]}) must align")
        clean: dict[Site, np.ndarray] = {}
        for site, mat in self.features.items():
            arr = np.asarray(mat, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] != n:
                raise ValueError(f"site {site} features must be (n={n}, d); got shape {arr.shape}")
            clean[site] = arr
        self.features = clean

    @property
    def n(self) -> int:
        return int(self.labels.shape[0])

    @property
    def sites(self) -> tuple[Site, ...]:
        return tuple(self.features.keys())


# ---------------------------------------------------------------------------
# Grouped cross-validation (seed-level, never split a clone)
# ---------------------------------------------------------------------------


def group_kfold_indices(
    groups: np.ndarray, n_splits: int, *, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped k-fold split indices that never split a group across folds.

    Every distinct value in ``groups`` is a seed; all rows of one seed go to exactly one test fold.
    Groups are shuffled deterministically (by ``seed``) and dealt to the fold with the fewest rows so
    far, which balances fold sizes without ever separating clones. Returns ``n_splits`` (train_idx,
    test_idx) pairs. This is the split that makes a probe's held-out AUC honest on cloned data.
    """
    groups = np.asarray(groups).ravel()
    unique = np.unique(groups)
    n_splits = int(max(2, min(n_splits, unique.size)))
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(unique.size)
    fold_of_group: dict[Any, int] = {}
    fold_load = np.zeros(n_splits, dtype=np.int64)
    group_sizes = {g: int(np.sum(groups == g)) for g in unique}
    for idx in order:
        g = unique[idx]
        f = int(np.argmin(fold_load))
        fold_of_group[g] = f
        fold_load[f] += group_sizes[g]
    fold_assignment = np.array([fold_of_group[g] for g in groups])
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(groups.size)
    for f in range(n_splits):
        test = all_idx[fold_assignment == f]
        train = all_idx[fold_assignment != f]
        splits.append((train, test))
    return splits


# ---------------------------------------------------------------------------
# The linear probe (numpy IRLS logistic regression, optional sklearn)
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _fit_logreg_numpy(
    x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, l2: float, max_iter: int = 100
) -> tuple[np.ndarray, float]:
    """L2-regularized logistic regression by IRLS (Newton) in pure numpy.

    Fits ``coef`` and ``intercept`` maximizing the weighted Bernoulli log-likelihood minus
    ``0.5 * l2 * ||coef||^2`` (the intercept is not penalized). IRLS is deterministic and, with the
    ridge term keeping the Hessian positive definite, stays finite even when the classes are linearly
    separable, which the mean-difference primitive it generalizes cannot do. Returns
    ``(coef (d,), intercept)``.
    """
    n, d = x.shape
    xa = np.concatenate([x, np.ones((n, 1), dtype=np.float64)], axis=1)
    beta = np.zeros(d + 1, dtype=np.float64)
    reg = np.ones(d + 1, dtype=np.float64) * float(l2)
    reg[-1] = 0.0  # do not penalize the intercept
    w = np.asarray(sample_weight, dtype=np.float64).ravel()
    yv = np.asarray(y, dtype=np.float64).ravel()
    for _ in range(max_iter):
        eta = xa @ beta
        p = _sigmoid(eta)
        s = np.clip(p * (1.0 - p), 1e-8, None) * w
        grad = xa.T @ (w * (p - yv)) + reg * beta
        hess = (xa * s[:, None]).T @ xa + np.diag(reg) + 1e-8 * np.eye(d + 1)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:  # pragma: no cover - ridge normally prevents this
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        beta = beta - step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return beta[:-1].astype(np.float64), float(beta[-1])


def _fit_logreg(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray,
    l2: float,
    solver: str,
) -> tuple[np.ndarray, float]:
    """Fit a logistic probe, preferring sklearn when asked and available, else the numpy IRLS path.

    ``solver="numpy"`` uses the deterministic in-module IRLS (the default, so a proof does not depend
    on a scikit-learn version). ``solver="auto"`` uses ``sklearn.linear_model.LogisticRegression``
    when it imports and falls back to the numpy path otherwise, matching the house convention of
    preferring sklearn with a numpy fallback. Both return ``(coef, intercept)`` in the same units.
    """
    if solver == "auto":
        try:
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(C=1.0 / max(l2, 1e-8), solver="lbfgs", max_iter=1000)
            clf.fit(x, y, sample_weight=sample_weight)
            return clf.coef_.ravel().astype(np.float64), float(clf.intercept_.ravel()[0])
        except Exception:  # noqa: BLE001 - sklearn absent or degenerate: use the numpy path
            pass
    return _fit_logreg_numpy(x, y, sample_weight, l2)


def _class_weights(y: np.ndarray, balance: bool) -> np.ndarray:
    """Per-row sample weights: inverse class frequency when balancing, else all ones."""
    y = np.asarray(y).ravel()
    if not balance:
        return np.ones(y.shape[0], dtype=np.float64)
    w = np.ones(y.shape[0], dtype=np.float64)
    for cls in (0, 1):
        m = y == cls
        c = int(np.sum(m))
        if c > 0:
            w[m] = y.shape[0] / (2.0 * c)
    return w


# ---------------------------------------------------------------------------
# Per-site sweep and the fit result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteProbe:
    """One site's probe in the depth sweep.

    ``coef`` is the full-data logistic weight at ``site`` (the concept direction before
    normalization); ``held_out_auc`` is the pooled out-of-fold AUC (each row scored by a fold that
    did not train on it or on any of its clones); ``cv_auc_mean`` / ``cv_auc_std`` summarize the
    per-fold AUCs. ``n_pos`` / ``n_neg`` are the class counts, surfaced so an imbalanced probe is
    read as such rather than trusted blindly.
    """

    site: Site
    coef: np.ndarray
    intercept: float
    held_out_auc: float
    cv_auc_mean: float
    cv_auc_std: float
    n_pos: int
    n_neg: int


@dataclass(frozen=True)
class ProbeFit:
    """The result of `fit_probe`: the persisted direction plus its fit provenance.

    ``direction`` is the persisted `Direction` (the headline artifact, from the best site).
    ``evidence`` is that direction stored as Evidence. ``per_site`` is the depth curve, one
    `SiteProbe` per swept site. ``calibration`` is the answer-key `CalibrationRef` (gate 1) or
    ``None``; ``scorecard_evidence`` is the answer-key ROC Evidence it points at, or ``None`` when
    the probe was not graded. ``held_out_auc`` and ``best_site`` are passthroughs of the chosen
    site's numbers.
    """

    direction: Direction
    evidence: Evidence
    per_site: tuple[SiteProbe, ...]
    calibration: CalibrationRef | None
    scorecard_evidence: Evidence | None
    held_out_auc: float
    best_site: Site

    @property
    def is_calibrated(self) -> bool:
        return self.calibration is not None


def _oof_scores(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    cv: int,
    l2: float,
    balance: bool,
    solver: str,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold decision scores and per-fold AUCs under grouped CV (no clone leakage).

    Each fold trains on the other folds and scores its own held-out rows; because the split is
    grouped by seed, no row is ever scored by a model that saw a clone of it. Returns the pooled
    out-of-fold score per row (``nan`` for a row whose fold could not be scored) and the list of
    per-fold AUCs.
    """
    n = x.shape[0]
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_aucs: list[float] = []
    for train_idx, test_idx in group_kfold_indices(groups, cv, seed=seed):
        y_tr = y[train_idx]
        if np.unique(y_tr).size < 2:
            continue  # a fold with one class cannot train a discriminative probe; skip honestly
        w_tr = _class_weights(y_tr, balance)
        coef, intercept = _fit_logreg(x[train_idx], y_tr, sample_weight=w_tr, l2=l2, solver=solver)
        scores = x[test_idx] @ coef + intercept
        oof[test_idx] = scores
        fold_aucs.append(float(roc_pr(scores, y[test_idx]).auc))
    return oof, fold_aucs


def _sweep_site(
    site: Site,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    cv: int,
    l2: float,
    balance: bool,
    solver: str,
    seed: int,
) -> tuple[SiteProbe, np.ndarray]:
    """Fit and cross-validate the probe at one site; return the `SiteProbe` and its OOF scores."""
    x = np.asarray(x, dtype=np.float64)
    oof, fold_aucs = _oof_scores(
        x, y, groups, cv=cv, l2=l2, balance=balance, solver=solver, seed=seed
    )
    finite = np.isfinite(oof)
    held_out_auc = float(roc_pr(oof[finite], y[finite]).auc) if np.any(finite) else float("nan")
    w_full = _class_weights(y, balance)
    if np.unique(y).size < 2:
        coef = np.zeros(x.shape[1], dtype=np.float64)
        intercept = 0.0
    else:
        coef, intercept = _fit_logreg(x, y, sample_weight=w_full, l2=l2, solver=solver)
    fold_arr = np.asarray(fold_aucs, dtype=np.float64)
    probe = SiteProbe(
        site=site,
        coef=coef,
        intercept=float(intercept),
        held_out_auc=held_out_auc,
        cv_auc_mean=float(np.nanmean(fold_arr)) if fold_arr.size else float("nan"),
        cv_auc_std=float(np.nanstd(fold_arr)) if fold_arr.size else float("nan"),
        n_pos=int(np.sum(y == 1)),
        n_neg=int(np.sum(y == 0)),
    )
    return probe, oof


def _bind_scorecard(
    name: str,
    oof_scores: np.ndarray,
    labels: np.ndarray,
    answer_key: "AnswerKey",
    *,
    target_tpr: float,
    target_fpr: float,
) -> tuple[CalibrationRef, Evidence]:
    """Grade the probe's held-out scores against the organism answer key (gate 1).

    The out-of-fold scores plus the answer-key labels are exactly a detector readout, so the answer
    key ROC is `MethodScorecard.evaluate` on that readout. When the concept is decodable the AUC is
    high and the returned `CalibrationRef` carries a real operating point; when the labels were
    shuffled the same machinery reports an AUC near chance, which is the honest outcome rather than a
    fabricated number. Returns the reference and the stored scorecard Evidence it points at.
    """
    from reward_lens.organisms.scorecard import MethodScorecard

    finite = np.isfinite(oof_scores)
    entry = MethodScorecard(f"ConceptProbe[{name}]").evaluate(
        (oof_scores[finite], labels[finite]),
        answer_key,
        target_tpr=target_tpr,
        target_fpr=target_fpr,
    )
    return entry.calibration_ref, entry.evidence


def fit_probe(
    signal: "SiteCaptures | RewardSignal",
    view: Any = None,
    target: "Callable[[Any, str], int | None] | None" = None,
    sites: tuple[Site, ...] | None = None,
    *,
    cv: int = 5,
    name: str | None = None,
    method: str = "probe_lr",
    l2: float = 1.0,
    class_balance: bool = True,
    solver: str = "numpy",
    answer_key: "AnswerKey | None" = None,
    store: "EvidenceStore | None" = None,
    seed: int = 0,
    target_tpr: float = 0.90,
    target_fpr: float = 0.05,
) -> ProbeFit:
    """Train a linear concept probe with grouped CV and automatic scorecard binding.

    ``signal`` is either a `SiteCaptures` (the substrate-free path the proofs run on: activations,
    labels, and seed groups already in hand) or a live `RewardSignal`, in which case ``view`` (a
    DataView of pairs) and ``target`` (a ``(item, side) -> label`` callable) are captured into a
    `SiteCaptures` at ``sites`` first. The probe is a logistic readout fit per site with seed-grouped
    cross-validation, so no clone is ever scored by a model that trained on it; the site with the
    highest out-of-fold AUC becomes the returned `Direction`.

    Scorecard binding is automatic and honest. If an ``answer_key`` is supplied (or carried on the
    captures), the out-of-fold scores are graded against it and the resulting `CalibrationRef` is
    attached (the direction becomes CALIBRATED). If none is available the direction carries
    ``calibration: None`` and stays EXPLORATORY: the gap is visible, never papered over.

    Args:
        signal: A `SiteCaptures`, or a `RewardSignal` to capture from (with ``view`` and ``target``).
        view: The DataView of pairs, when capturing from a signal.
        target: A ``(item, side) -> int | None`` labeler, when capturing from a signal.
        sites: Sites to sweep; defaults to every site in the captures.
        cv: Number of seed-grouped CV folds.
        name: The concept name; defaults to the target's or captures' name.
        method: Stored on the direction (``"probe_lr"`` by default).
        l2: Ridge strength on the logistic weight.
        class_balance: Weight the minority class up by inverse frequency.
        solver: ``"numpy"`` (deterministic IRLS) or ``"auto"`` (sklearn, numpy fallback).
        answer_key: The organism `AnswerKey` to grade against for calibration (gate 1).
        store: If given, the direction Evidence (and its scorecard parent) are appended.
        seed: Seed for the CV shuffle.
        target_tpr: The true-positive rate of the operating point the scorecard states its detection at.
        target_fpr: The false-positive rate of that same operating point.

    Returns:
        A `ProbeFit` whose ``direction`` is the persisted `Direction`.
    """
    captures = _as_captures(signal, view, target, sites, name=name)
    sweep_sites = sites or captures.sites
    if not sweep_sites:
        raise ValueError("fit_probe: no sites to sweep; captures carry no features")
    concept_name = name or captures.name

    per_site: list[SiteProbe] = []
    oof_by_site: dict[Site, np.ndarray] = {}
    for site in sweep_sites:
        if site not in captures.features:
            raise KeyError(f"fit_probe: site {site} not present in captures")
        probe, oof = _sweep_site(
            site,
            captures.features[site],
            captures.labels,
            captures.groups,
            cv=cv,
            l2=l2,
            balance=class_balance,
            solver=solver,
            seed=seed,
        )
        per_site.append(probe)
        oof_by_site[site] = oof

    best = max(
        per_site,
        key=lambda p: p.held_out_auc if np.isfinite(p.held_out_auc) else -1.0,
    )

    key = answer_key if answer_key is not None else captures.answer_key
    calibration: CalibrationRef | None = None
    scorecard_ev: Evidence | None = None
    if key is not None:
        calibration, scorecard_ev = _bind_scorecard(
            concept_name,
            oof_by_site[best.site],
            captures.labels,
            key,
            target_tpr=target_tpr,
            target_fpr=target_fpr,
        )

    direction = make_direction(
        name=concept_name,
        site=best.site,
        vector=best.coef,
        method=method,
        train_data=captures.dataset_id,
        calibration=calibration,
        meta={
            "held_out_auc": best.held_out_auc,
            "cv_auc_mean": best.cv_auc_mean,
            "cv_auc_std": best.cv_auc_std,
            "n_pos": best.n_pos,
            "n_neg": best.n_neg,
            "n": captures.n,
            "cv_folds": int(cv),
            "sweep": {str(p.site): round(p.held_out_auc, 6) for p in per_site},
        },
    )
    parents = (str(scorecard_ev.id),) if scorecard_ev is not None else ()
    evidence = direction_evidence(direction, parents=parents, n=captures.n)

    if store is not None:
        if scorecard_ev is not None:
            store.append(scorecard_ev)
        store.append(evidence)

    return ProbeFit(
        direction=direction,
        evidence=evidence,
        per_site=tuple(per_site),
        calibration=calibration,
        scorecard_evidence=scorecard_ev,
        held_out_auc=best.held_out_auc,
        best_site=best.site,
    )


def _as_captures(
    signal: "SiteCaptures | RewardSignal",
    view: Any,
    target: "Callable[[Any, str], int | None] | None",
    sites: tuple[Site, ...] | None,
    *,
    name: str | None,
) -> SiteCaptures:
    """Resolve the first argument into a `SiteCaptures`, capturing from a signal when needed."""
    if isinstance(signal, SiteCaptures):
        return signal
    if view is None or target is None or sites is None:
        raise ValueError(
            "fit_probe: capturing from a signal needs view, target, and sites; "
            "pass a SiteCaptures directly for the pre-captured path"
        )
    return capture_probe_inputs(signal, view, target, sites, name=name)


def probe_scores(direction: Direction, activations: np.ndarray) -> np.ndarray:
    """Project activations onto a direction: the probe's decision score up to its intercept.

    ``activations`` is ``(n, d)`` at the direction's site; the return is ``(n,)``. This is the read
    a downstream detector uses, and the ``featurize`` a concept bank exposes to the indices.
    """
    a = np.asarray(activations, dtype=np.float64)
    v = np.asarray(direction.vector, dtype=np.float64)
    return a @ v


# ---------------------------------------------------------------------------
# Capturing from a real signal (torch-gated)
# ---------------------------------------------------------------------------


def capture_probe_inputs(
    signal: "RewardSignal",
    view: Any,
    target: "Callable[[Any, str], int | None]",
    sites: tuple[Site, ...],
    *,
    name: str | None = None,
    max_pairs: int | None = None,
) -> SiteCaptures:
    """Capture per-example activations and labels from a signal (the production path, torch-gated).

    Each pair in ``view`` contributes two examples, its chosen and rejected sides; the activations at
    ``sites`` are read in fp32 and the label comes from ``target(item, side)`` where ``side`` is
    ``"chosen"`` or ``"rejected"``. A label of ``None`` drops that side (a target that does not apply
    to it). Both sides of a pair share the pair's seed id, so the grouped CV never splits them, which
    is the clone-safety the factory depends on. torch is imported lazily inside this function only.
    """
    rows: list[tuple[Any, str, int]] = []
    chosen_items: list[Any] = []
    rejected_items: list[Any] = []
    groups: list[Any] = []
    items = list(view)
    if max_pairs is not None:
        items = items[:max_pairs]
    for item in items:
        seed = getattr(item, "seed_id", None) or getattr(
            getattr(item, "lineage", None), "seed_id", None
        )
        seed = seed if seed is not None else f"row{len(groups)}"
        for side in ("chosen", "rejected"):
            label = target(item, side)
            if label is None:
                continue
            text = getattr(getattr(item, side), "text")
            prompt = item.prompt_text
            (chosen_items if side == "chosen" else rejected_items).append((prompt, text))
            rows.append((item, side, int(label)))
            groups.append(seed)

    # Capture the two sides in their own forward batches, then interleave back to row order.
    labels = np.array([r[2] for r in rows], dtype=np.int64)
    order_side = [r[1] for r in rows]
    caps_chosen = _capture_matrix(signal, chosen_items, sites) if chosen_items else {}
    caps_rejected = _capture_matrix(signal, rejected_items, sites) if rejected_items else {}

    features: dict[Site, np.ndarray] = {}
    for site in sites:
        d = caps_chosen[site].shape[1] if chosen_items else caps_rejected[site].shape[1]
        mat = np.zeros((len(rows), d), dtype=np.float32)
        ci = ri = 0
        for i, side in enumerate(order_side):
            if side == "chosen":
                mat[i] = caps_chosen[site][ci]
                ci += 1
            else:
                mat[i] = caps_rejected[site][ri]
                ri += 1
        features[site] = mat

    dataset_id = view.checksum() if hasattr(view, "checksum") else None
    return SiteCaptures(
        features=features,
        labels=labels,
        groups=np.array(groups, dtype=object),
        dataset_id=dataset_id,
        name=name or "probe-captures",
    )


def _capture_matrix(
    signal: "RewardSignal", items: list[Any], sites: tuple[Site, ...]
) -> dict[Site, np.ndarray]:
    """Final-token activations at ``sites`` for ``items`` as numpy fp32 (torch-gated)."""
    from reward_lens.measure.battery._common import capture_sites

    tensors = capture_sites(signal, items, sites, dtype="float32")
    return {site: t.detach().to("cpu").numpy().astype(np.float32) for site, t in tensors.items()}


# ---------------------------------------------------------------------------
# Common targets
# ---------------------------------------------------------------------------


def feature_target(feature: str) -> "Callable[[Any, str], int | None]":
    """A target that labels a response by whether it carries a controlled feature marker.

    Reads the organism feature substrate (`organisms._features.extract_features`), so a probe built
    from it predicts a surface concept whose ground truth is exact. This is the target the probe
    proof plants: the concept is a known feature, so recovery is checkable.
    """
    from reward_lens.organisms._features import extract_features

    def label(item: Any, side: str) -> int:
        text = getattr(getattr(item, side), "text")
        return int(extract_features(text)[feature])

    return label


__all__ = [
    "Direction",
    "DirectionArtifact",
    "make_direction",
    "direction_evidence",
    "SiteCaptures",
    "group_kfold_indices",
    "SiteProbe",
    "ProbeFit",
    "fit_probe",
    "probe_scores",
    "capture_probe_inputs",
    "feature_target",
]

"""Preference-difference dictionaries: the exact Bradley-Terry margin decomposition.

This is the corpus's reward-model-native dictionary. Where an SAE is trained on activations, a
preference-difference dictionary is trained on the difference vector ``delta_h = h_chosen -
h_rejected``, because for a reward model the chosen-minus-rejected difference is not a stimulus, it
is the object the Bradley-Terry loss operates on. That choice buys an exact algebraic identity that
no activation dictionary has.

For a linear reward readout ``w_r`` the Bradley-Terry margin of a pair is the reward difference,
``margin = w_r . delta_h``. Write the dictionary reconstruction as ``delta_h = sum_i f_i d_i`` with
``f_i`` the activations and ``d_i`` the atoms. Then the margin decomposes exactly:

    margin = w_r . delta_h = w_r . (sum_i f_i d_i) = sum_i f_i (w_r . d_i).

Every atom carries a fixed contribution ``w_r . d_i`` to the margin, scaled per pair by its
activation ``f_i``. This says precisely which directions of preference difference the reward reads
and by how much, in units of margin. When the reconstruction is exact the identity holds to
numerical precision and is an algebraic fact, not an approximation; when the dictionary reconstructs
delta_h only approximately, the residual is ``w_r . (delta_h - reconstruction)`` and the trainer
reports it honestly rather than hiding it.

The trainer verifies the identity on held-out pairs and stores the verification as Evidence next to
the dictionary, so a dictionary in the store always carries the residual it was certified at. The
whole construction is linear algebra on numpy arrays and runs on CPU; the proof plants a known
dictionary and checks the reconstructed margin against the true margin to ~1e-6 on held-out pairs.

One honesty note carried in the code: the identity is about the linear readout margin ``w_r .
delta_h`` at the readout's input site. Where a production reward applies a final nonlinearity (an
RMSNorm before the score head), ``delta_h`` must be taken at the linear readout's input for the
margin to be ``w_r . delta_h``; the decomposition is not claimed to equal a post-nonlinearity score
difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from reward_lens.core.evidence import Evidence, Uncertainty, make_evidence, register_payload
from reward_lens.core.provenance import capture_provenance
from reward_lens.core.types import DatasetID, GaugeStatus, SubjectRef, content_hash

if TYPE_CHECKING:
    from reward_lens.core.store import EvidenceStore

_DIFFDICT_VERSION = "1.0"
_EXACT_TOL = 1e-6


# ---------------------------------------------------------------------------
# The dictionary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffDictionary:
    """A preference-difference dictionary with its per-atom margin contributions.

    ``atoms`` are the ``(k, d)`` dictionary directions ``d_i`` (orthonormal rows as fit by SVD).
    ``w_r`` is the fp32 reward direction the margin is taken along. ``atom_margins`` is the fixed
    per-atom margin contribution ``w_r . d_i`` (shape ``(k,)``), which is the reward-native content of
    the dictionary: it is how much each atom moves the Bradley-Terry margin. ``train_data`` is the
    `DatasetID` of the delta_h it was trained on, and ``meta`` carries the fit summary.

    Activations for a batch of ``delta_h`` are ``F = delta_h @ atoms.T`` (exact projection
    coefficients because the atoms are orthonormal), and the decomposed margin is ``F @ atom_margins``.
    """

    id: str
    atoms: np.ndarray
    w_r: np.ndarray
    atom_margins: np.ndarray
    method: str
    train_data: DatasetID | None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return int(np.asarray(self.atoms).shape[0])

    @property
    def d_model(self) -> int:
        return int(np.asarray(self.atoms).shape[1])


@register_payload
@dataclass
class DiffDictArtifact:
    """The serializable payload form of a `DiffDictionary`."""

    dict_id: str
    atoms: np.ndarray
    w_r: np.ndarray
    atom_margins: np.ndarray
    method: str
    train_data: str | None
    meta: dict[str, Any] = field(default_factory=dict)


@register_payload
@dataclass
class DiffDictVerification:
    """The stored verification of the exact-decomposition identity on held-out pairs.

    This is the Evidence payload that certifies a dictionary. ``max_abs_residual`` and
    ``mean_abs_residual`` are the held-out gap between the true margin ``w_r . delta_h`` and the
    reconstructed margin ``sum_i f_i (w_r . d_i)``; ``rel_residual`` normalizes by the margin scale.
    ``reconstruction_r2`` is how much of the held-out delta_h the dictionary captures (1.0 when the
    reconstruction is exact). ``exact`` is whether ``max_abs_residual`` is below the numerical
    tolerance, which is the algebraic-identity case. The small ``margin_true`` / ``margin_recon``
    samples let a reader see the identity hold pair by pair without loading the full arrays.
    """

    dict_id: str
    n_heldout: int
    n_atoms: int
    max_abs_residual: float
    mean_abs_residual: float
    rel_residual: float
    reconstruction_r2: float
    exact: bool
    tol: float
    margin_true: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    margin_recon: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    notes: str = ""


# ---------------------------------------------------------------------------
# Core algebra
# ---------------------------------------------------------------------------


def delta_h_from_pairs(h_chosen: np.ndarray, h_rejected: np.ndarray) -> np.ndarray:
    """The preference-difference matrix ``delta_h = h_chosen - h_rejected`` in fp64.

    ``h_chosen`` and ``h_rejected`` are ``(n, d)`` activations of the chosen and rejected sides at the
    readout's input site. This is the object the dictionary is trained on and the object whose margin
    ``w_r . delta_h`` the decomposition explains.
    """
    hc = np.asarray(h_chosen, dtype=np.float64)
    hr = np.asarray(h_rejected, dtype=np.float64)
    if hc.shape != hr.shape:
        raise ValueError(f"h_chosen {hc.shape} and h_rejected {hr.shape} must have the same shape")
    return hc - hr


def true_margin(delta_h: np.ndarray, w_r: np.ndarray) -> np.ndarray:
    """The Bradley-Terry margin per pair, ``w_r . delta_h`` (shape ``(n,)``)."""
    return np.asarray(delta_h, dtype=np.float64) @ np.asarray(w_r, dtype=np.float64).ravel()


def activations(dictionary: DiffDictionary, delta_h: np.ndarray) -> np.ndarray:
    """Dictionary activations ``F = delta_h @ atoms.T`` for a batch of preference differences."""
    d = np.asarray(delta_h, dtype=np.float64)
    return d @ np.asarray(dictionary.atoms, dtype=np.float64).T


def reconstruct(dictionary: DiffDictionary, delta_h: np.ndarray) -> np.ndarray:
    """The dictionary reconstruction ``F @ atoms`` of a batch of preference differences."""
    f = activations(dictionary, delta_h)
    return f @ np.asarray(dictionary.atoms, dtype=np.float64)


def decompose_margin(
    dictionary: DiffDictionary, delta_h: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The decomposed margin and its per-atom contributions.

    Returns ``(margin_recon (n,), contributions (n, k))`` where ``contributions[:, i] = f_i * (w_r .
    d_i)`` and ``margin_recon = contributions.sum(axis=1) = F @ atom_margins``. The per-atom
    contributions are the reward-native readout: which preference-difference directions the reward
    reads, in units of margin, pair by pair.
    """
    f = activations(dictionary, delta_h)
    atom_margins = np.asarray(dictionary.atom_margins, dtype=np.float64).ravel()
    contributions = f * atom_margins[None, :]
    return contributions.sum(axis=1), contributions


def verify_decomposition(
    dictionary: DiffDictionary,
    delta_h_heldout: np.ndarray,
    w_r: np.ndarray | None = None,
    *,
    tol: float = _EXACT_TOL,
    sample: int = 8,
) -> DiffDictVerification:
    """Check the exact-decomposition identity on held-out pairs and bundle the residual.

    Computes the true margin ``w_r . delta_h`` and the reconstructed margin ``sum_i f_i (w_r . d_i)``
    on held-out pairs and reports their gap. When the dictionary reconstructs delta_h exactly the
    residual is at numerical precision and ``exact`` is True (the identity is algebraic); otherwise
    the honest residual is carried. ``w_r`` defaults to the dictionary's own reward direction, so the
    same margin the atoms were scored against is the one verified.
    """
    delta = np.asarray(delta_h_heldout, dtype=np.float64)
    w = np.asarray(dictionary.w_r if w_r is None else w_r, dtype=np.float64).ravel()

    margin_true = delta @ w
    margin_recon, _ = decompose_margin(dictionary, delta)
    residual = margin_true - margin_recon
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    mean_abs = float(np.mean(np.abs(residual))) if residual.size else 0.0
    scale = float(np.mean(np.abs(margin_true))) if margin_true.size else 0.0
    rel = max_abs / scale if scale > 1e-12 else max_abs

    recon = reconstruct(dictionary, delta)
    ss_res = float(np.sum((delta - recon) ** 2))
    ss_tot = float(np.sum((delta - delta.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

    k = int(min(sample, margin_true.size))
    return DiffDictVerification(
        dict_id=dictionary.id,
        n_heldout=int(delta.shape[0]),
        n_atoms=dictionary.n_atoms,
        max_abs_residual=max_abs,
        mean_abs_residual=mean_abs,
        rel_residual=float(rel),
        reconstruction_r2=float(r2),
        exact=bool(max_abs <= tol),
        tol=float(tol),
        margin_true=margin_true[:k].astype(np.float32),
        margin_recon=margin_recon[:k].astype(np.float32),
        notes=(
            "identity margin = sum_i f_i (w_r . d_i) verified on held-out delta_h; "
            "exact to numerical precision when the reconstruction is exact"
        ),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _content_id(atoms: np.ndarray, w_r: np.ndarray, method: str, train_data: Any) -> str:
    material = {
        "atoms": np.asarray(atoms, dtype=np.float32).tolist(),
        "w_r": np.asarray(w_r, dtype=np.float32).tolist(),
        "method": method,
        "train_data": None if train_data is None else str(train_data),
    }
    return content_hash(material, "ddict")


def train_diff_dict(
    delta_h_train: np.ndarray,
    w_r: np.ndarray,
    n_atoms: int,
    *,
    method: str = "svd",
    train_data: DatasetID | None = None,
    meta: dict[str, Any] | None = None,
) -> DiffDictionary:
    """Fit a preference-difference dictionary of ``n_atoms`` atoms from training delta_h.

    The atoms are the top ``n_atoms`` right singular vectors of ``delta_h_train`` (an orthonormal
    basis for its dominant preference-difference subspace), so the reconstruction of any delta_h is
    its orthogonal projection onto that subspace and is exact whenever delta_h lies in it. Each atom's
    margin contribution ``w_r . d_i`` is computed once and stored, which is what makes the margin
    decomposition a lookup rather than a refit. Returns a persisted-ready `DiffDictionary`.

    Choosing ``n_atoms`` at least the rank of the training delta_h makes the identity exact on data
    drawn from the same subspace; choosing fewer truncates the dictionary and the trainer will report
    the honest residual at verification time.
    """
    delta = np.asarray(delta_h_train, dtype=np.float64)
    if delta.ndim != 2:
        raise ValueError(f"delta_h_train must be (n, d); got shape {delta.shape}")
    w = np.asarray(w_r, dtype=np.float64).ravel()
    if w.shape[0] != delta.shape[1]:
        raise ValueError(f"w_r dim {w.shape[0]} does not match delta_h dim {delta.shape[1]}")
    k = int(max(1, min(n_atoms, min(delta.shape))))

    # Right singular vectors span the row space (the preference-difference subspace).
    _u, _s, vt = np.linalg.svd(delta, full_matrices=False)
    atoms = np.asarray(vt[:k], dtype=np.float32)  # (k, d), orthonormal rows
    atom_margins = (atoms.astype(np.float64) @ w).astype(np.float32)

    dict_id = _content_id(atoms, w, method, train_data)
    full_meta = {
        "n_train": int(delta.shape[0]),
        "singular_values": [float(x) for x in _s[:k]],
        "requested_atoms": int(n_atoms),
    }
    full_meta.update(meta or {})
    return DiffDictionary(
        id=dict_id,
        atoms=atoms,
        w_r=w.astype(np.float32),
        atom_margins=atom_margins,
        method=method,
        train_data=train_data,
        meta=full_meta,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def diff_dict_evidence(dictionary: DiffDictionary, *, signals: tuple[str, ...] = ()) -> Evidence:
    """Wrap a `DiffDictionary` as COVARIANT Evidence so the dictionary is a store citizen."""
    artifact = DiffDictArtifact(
        dict_id=dictionary.id,
        atoms=np.asarray(dictionary.atoms, dtype=np.float32),
        w_r=np.asarray(dictionary.w_r, dtype=np.float32),
        atom_margins=np.asarray(dictionary.atom_margins, dtype=np.float32),
        method=dictionary.method,
        train_data=None if dictionary.train_data is None else str(dictionary.train_data),
        meta=dict(dictionary.meta),
    )
    subject = SubjectRef(
        signals=tuple(signals),
        dataset=dictionary.train_data,
        readout="preference-difference-dictionary",
        extra={"kind": "diff-dict", "n_atoms": dictionary.n_atoms, "method": dictionary.method},
    )
    provenance = capture_provenance(config={"method": dictionary.method, "id": dictionary.id})
    return make_evidence(
        observable="PreferenceDiffDictionary",
        observable_version=_DIFFDICT_VERSION,
        subject=subject,
        value=artifact,
        uncertainty=Uncertainty(n=dictionary.meta.get("n_train"), method="svd-subspace"),
        gauge=GaugeStatus.COVARIANT,
        provenance=provenance,
    )


def verification_evidence(
    verification: DiffDictVerification,
    *,
    signals: tuple[str, ...] = (),
    parents: tuple[str, ...] = (),
) -> Evidence:
    """Wrap a `DiffDictVerification` as INVARIANT Evidence (a residual is gauge-invariant).

    ``parents`` names the dictionary Evidence this verification certifies, so the store's DAG links
    the certificate to the artifact it certifies. The residual is a scalar gap between two margins and
    is basis-independent, hence INVARIANT.
    """
    subject = SubjectRef(
        signals=tuple(signals),
        readout="preference-difference-dictionary",
        extra={"kind": "diff-dict-verification", "dict_id": verification.dict_id},
    )
    provenance = capture_provenance(
        config={"dict_id": verification.dict_id}, parents=tuple(parents)
    )
    return make_evidence(
        observable="PreferenceDiffDictionary.verify",
        observable_version=_DIFFDICT_VERSION,
        subject=subject,
        value=verification,
        uncertainty=Uncertainty(n=verification.n_heldout, method="exact-decomposition-residual"),
        gauge=GaugeStatus.INVARIANT,
        provenance=provenance,
    )


@dataclass(frozen=True)
class DiffDictResult:
    """A trained dictionary with its held-out verification and their Evidence.

    ``dictionary`` is the artifact; ``verification`` carries the held-out residual; ``dict_evidence``
    and ``verification_evidence`` are the stored forms, linked in the DAG. ``exact`` and
    ``max_abs_residual`` are passthroughs of the certified identity.
    """

    dictionary: DiffDictionary
    verification: DiffDictVerification
    dict_evidence: Evidence
    verification_evidence: Evidence

    @property
    def exact(self) -> bool:
        return self.verification.exact

    @property
    def max_abs_residual(self) -> float:
        return self.verification.max_abs_residual


def fit_and_verify(
    delta_h_train: np.ndarray,
    delta_h_heldout: np.ndarray,
    w_r: np.ndarray,
    n_atoms: int,
    *,
    method: str = "svd",
    train_data: DatasetID | None = None,
    tol: float = _EXACT_TOL,
    store: "EvidenceStore | None" = None,
    signals: tuple[str, ...] = (),
) -> DiffDictResult:
    """Train a difference dictionary and verify its margin decomposition on held-out pairs.

    Fits the dictionary on ``delta_h_train``, checks the exact-decomposition identity on
    ``delta_h_heldout``, and bundles both as linked Evidence. When ``store`` is given, the dictionary
    Evidence is appended first and the verification Evidence second (naming the dictionary as its
    parent), so the store carries the artifact and its certificate as a DAG. The returned result
    exposes whether the identity held exactly and at what residual.
    """
    dictionary = train_diff_dict(delta_h_train, w_r, n_atoms, method=method, train_data=train_data)
    verification = verify_decomposition(dictionary, delta_h_heldout, w_r, tol=tol)
    dict_ev = diff_dict_evidence(dictionary, signals=signals)
    ver_ev = verification_evidence(verification, signals=signals, parents=(str(dict_ev.id),))
    if store is not None:
        store.append(dict_ev)
        store.append(ver_ev)
    return DiffDictResult(
        dictionary=dictionary,
        verification=verification,
        dict_evidence=dict_ev,
        verification_evidence=ver_ev,
    )


__all__ = [
    "DiffDictionary",
    "DiffDictArtifact",
    "DiffDictVerification",
    "delta_h_from_pairs",
    "true_margin",
    "activations",
    "reconstruct",
    "decompose_margin",
    "verify_decomposition",
    "train_diff_dict",
    "diff_dict_evidence",
    "verification_evidence",
    "DiffDictResult",
    "fit_and_verify",
]

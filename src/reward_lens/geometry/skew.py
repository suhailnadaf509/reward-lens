"""The skew-symmetric preference operator: cyclic structure the scalar head cannot express.

A scalar reward induces preferences through ``P(x > y) = sigma(r(x) - r(y))``, which is always
transitive: it is a total order by ``r``. Real preference data is not always transitive (the
rock-paper-scissors / Condorcet-cycle phenomenon), and a scalar head provably cannot represent a
cycle (theorem T8). The object that can is a skew-symmetric bilinear form on the activations,

    s(x, y) = phi(x)^T A phi(y),   A^T = -A,

whose antisymmetry ``s(x, y) = -s(y, x)`` is exactly what a preference relation needs, and whose
rank measures how much intransitive structure the representation supports. A rank-``2k`` skew
operator captures ``k`` independent cyclic "planes"; the scalar head is the degenerate rank-0 case.

`PreferenceRankTest` fits a rank-``k`` skew operator on frozen penultimate activations and tests
held-out cyclic-preference prediction against the best transitive (scalar) model. It returns the
effective preference rank and the cyclic-recovery margin as Evidence. Consumed by paradigm
physiology (S12) and preference topology (S6), which Hodge-decomposes the operator's predictions.

Pure numpy on frozen activations; no model, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from reward_lens.core.errors import NumericsError
from reward_lens.core.evidence import Evidence, make_evidence, register_payload
from reward_lens.core.provenance import Provenance
from reward_lens.core.types import GaugeStatus, SubjectRef

_FAITHFUL_TO = "T8 scalar-bottleneck / intransitive preference"


@register_payload
@dataclass
class PreferenceRankResult:
    """The payload of `PreferenceRankTest`.

    ``effective_rank`` is the algebraic rank of the fitted skew operator (even: skew singular values
    come in pairs); ``effective_rank_pairs`` is that count of independent cyclic planes.
    ``transitive_acc`` and ``skew_acc`` are held-out prediction accuracies of the best scalar model
    and the rank-``k`` skew model; ``cyclic_recovery`` is their difference, the intransitive
    structure the scalar head cannot express but the skew operator recovers. ``singular_values`` is
    the fitted skew spectrum.
    """

    effective_rank: int
    effective_rank_pairs: int
    transitive_acc: float
    skew_acc: float
    cyclic_recovery: float
    singular_values: np.ndarray
    rank_k: int
    n_pairs: int
    n_test: int
    faithful_to: str = _FAITHFUL_TO


def _ridge_transitive(phi: np.ndarray, pairs: np.ndarray, reg: float) -> np.ndarray:
    """Best scalar reward direction: ridge fit of ``w . (phi_i - phi_j) = +1`` over winner-loser pairs.

    Every training pair contributes the constraint that the winner outscores the loser by a unit
    margin. With cyclic data no single ``w`` satisfies all constraints, so the fit is the least-
    squares compromise, which is the strongest transitive model there is.
    """
    x = phi[pairs[:, 0]] - phi[pairs[:, 1]]
    d = phi.shape[1]
    gram = x.T @ x + reg * np.eye(d)
    return np.linalg.solve(gram, x.T @ np.ones(x.shape[0]))


def _fit_skew(phi: np.ndarray, pairs: np.ndarray, rank_k: int, reg: float) -> np.ndarray:
    """Fit a rank-``2k`` skew-symmetric operator ``A`` so ``phi_i^T A phi_j > 0`` on winner-loser pairs.

    Builds an antisymmetric training set (each winner-loser pair and its negation) and solves for
    ``vec(A)`` by ridge least squares, then projects onto the skew-symmetric cone ``(A - A^T)/2`` and
    truncates to rank ``2k`` by its SVD. The convex-fit-then-project route is stable at the moderate
    activation dimension penultimate features are used at.
    """
    d = phi.shape[1]
    pi = phi[pairs[:, 0]]
    pj = phi[pairs[:, 1]]
    # Design rows vec(phi_i phi_j^T) = kron(phi_j, phi_i); antisymmetrize by adding the negation.
    feats_pos = np.einsum("pi,pj->pij", pi, pj).reshape(pairs.shape[0], d * d)
    feats_neg = np.einsum("pi,pj->pij", pj, pi).reshape(pairs.shape[0], d * d)
    x = np.vstack([feats_pos, feats_neg])
    y = np.concatenate([np.ones(pairs.shape[0]), -np.ones(pairs.shape[0])])
    gram = x.T @ x + reg * np.eye(d * d)
    vec_a = np.linalg.solve(gram, x.T @ y)
    a = vec_a.reshape(d, d)
    a = 0.5 * (a - a.T)  # project onto skew-symmetric

    u, s, vh = np.linalg.svd(a)
    r = min(2 * rank_k, s.size)
    a_trunc = (u[:, :r] * s[:r]) @ vh[:r]
    return 0.5 * (a_trunc - a_trunc.T)  # re-skew after truncation


def _accuracy_scalar(phi: np.ndarray, w: np.ndarray, pairs: np.ndarray) -> float:
    """Fraction of held-out pairs the scalar model orders correctly (winner outscores loser)."""
    margin = (phi[pairs[:, 0]] - phi[pairs[:, 1]]) @ w
    return float(np.mean(margin > 0))


def _accuracy_skew(phi: np.ndarray, a: np.ndarray, pairs: np.ndarray) -> float:
    """Fraction of held-out pairs the skew model orders correctly (``phi_i^T A phi_j > 0``)."""
    s = np.einsum("pi,ij,pj->p", phi[pairs[:, 0]], a, phi[pairs[:, 1]])
    return float(np.mean(s > 0))


class PreferenceRankTest:
    """Fit a rank-``k`` skew preference operator and test cyclic recovery (T8).

    ``activations`` is the ``n x d`` matrix of frozen penultimate features (one row per response
    item); ``pairs`` is a ``(P, 2)`` integer array of ``(winner_index, loser_index)`` preferences;
    ``rank_k`` is the skew rank to fit. Optionally project activations to ``proj_dim`` principal
    components first, which keeps the ``d^2`` skew fit tractable at large ``d``.

    ``run`` splits the pairs into train and held-out, fits the transitive (scalar) baseline and the
    rank-``k`` skew operator on the train pairs, and reports their held-out accuracies. The gap is
    the cyclic structure the scalar head cannot express (theorem T8). The result is INVARIANT: rank
    and predictive accuracy do not depend on the activation coordinate basis, so this is a
    within-model structural claim that needs no frame.
    """

    def __init__(
        self,
        activations: Any,
        pairs: Any,
        rank_k: int,
        *,
        proj_dim: int | None = None,
        reg: float = 1e-3,
    ):
        phi = np.asarray(activations, dtype=np.float64)
        if phi.ndim != 2:
            raise NumericsError(f"activations must be 2-D (n x d); got shape {phi.shape}")
        self.pairs = np.asarray(pairs, dtype=np.int64)
        if self.pairs.ndim != 2 or self.pairs.shape[1] != 2:
            raise NumericsError("pairs must be a (P, 2) array of (winner, loser) indices")
        if rank_k < 1:
            raise NumericsError(f"rank_k must be >= 1; got {rank_k}")
        self.rank_k = int(rank_k)
        self.reg = float(reg)

        phi = phi - phi.mean(axis=0)
        if proj_dim is not None and proj_dim < phi.shape[1]:
            u, s, vh = np.linalg.svd(phi, full_matrices=False)
            phi = phi @ vh[:proj_dim].T
        self.phi = phi

    def run(
        self,
        *,
        test_frac: float = 0.3,
        seed: int = 0,
        sig_ratio: float = 0.05,
        subject: SubjectRef | None = None,
        provenance: Provenance | None = None,
    ) -> Evidence[PreferenceRankResult]:
        """Fit both models on a train split and score held-out cyclic prediction."""
        rng = np.random.default_rng(seed)
        p = self.pairs.shape[0]
        if p < 4:
            raise NumericsError(f"need at least 4 pairs to split; got {p}")
        perm = rng.permutation(p)
        n_test = max(1, int(round(test_frac * p)))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        train = self.pairs[train_idx]
        test = self.pairs[test_idx]

        w = _ridge_transitive(self.phi, train, self.reg)
        a = _fit_skew(self.phi, train, self.rank_k, self.reg)

        transitive_acc = _accuracy_scalar(self.phi, w, test)
        skew_acc = _accuracy_skew(self.phi, a, test)

        sv = np.linalg.svd(a, compute_uv=False)
        thresh = sig_ratio * float(sv[0]) if sv.size and sv[0] > 0 else 0.0
        n_sig = int(np.sum(sv > thresh))
        effective_rank = n_sig
        effective_rank_pairs = (n_sig + 1) // 2

        value = PreferenceRankResult(
            effective_rank=effective_rank,
            effective_rank_pairs=effective_rank_pairs,
            transitive_acc=transitive_acc,
            skew_acc=skew_acc,
            cyclic_recovery=float(skew_acc - transitive_acc),
            singular_values=sv.astype(np.float32),
            rank_k=self.rank_k,
            n_pairs=int(p),
            n_test=int(n_test),
        )
        subj = subject or SubjectRef(extra={"rank_k": self.rank_k, "n_pairs": int(p)})
        return make_evidence(
            observable="geometry.preference_rank_test",
            observable_version="1",
            subject=subj,
            value=value,
            gauge=GaugeStatus.INVARIANT,
            provenance=provenance,
        )


__all__ = [
    "PreferenceRankTest",
    "PreferenceRankResult",
]

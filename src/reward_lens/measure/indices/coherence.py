"""A9 Coherence and Contamination: cross-criterion geometry.

Formal definition, A9. For a reward with criterion directions ``{v_k}`` (ArmoRM's nineteen
objectives, a rubric's criteria):

  - Coherence ``μ_jk = v_j · v_k`` on unit-normalized directions is the Gram matrix of the criteria.
  - The Welch floor ``max_{j≠k} |μ_jk| ≥ √((K − d) / (d(K − 1)))`` holds once ``K > d_eff``: you cannot
    pack more nearly-orthogonal criteria than the effective dimension allows, so beyond that dimension
    the criteria must start to overlap (faithful_to Welch 1974).
  - Contamination ``C_jk`` is the change in criterion ``k``'s contribution when criterion ``j``'s
    direction is steered; to first order under a linear head this is exactly ``μ_jk``.
  - ``d_eff`` is the participation ratio of the reward-relevant subspace, the effective number of
    independent criteria.

This is the capacity theory of bias (S5): when a reward tries to carry more criteria than its effective
dimension supports, the criteria contaminate one another and the surplus reward leaks into dark
channels (A10). Deviations from A9: contamination is the first-order linear leakage ``μ_jk``; the
measured causal contamination from steering is the production path. ``d_eff`` here is the participation
ratio of the criteria's own Gram spectrum (the dimension the criteria actually span); passing a
reward-Hessian ``SpectrumResult`` substitutes the Hessian-subspace ``d_eff`` A9 names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.types import (
    Access,
    AccessMatrix,
    Capability,
    Component,
    GaugeStatus,
    Substrate,
)
from reward_lens.geometry import participation_ratio
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.indices._support import GRADER_STUDY_PHASES, MEASURED_BY

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def _unit_rows(directions: np.ndarray) -> np.ndarray:
    v = np.asarray(directions, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError(f"criterion directions must be (K, d); got shape {v.shape}")
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def coherence_matrix(directions: np.ndarray) -> np.ndarray:
    """The coherence (Gram) matrix ``μ_jk = v_j · v_k`` of unit-normalized criterion directions (A9).

    ``directions`` is ``(K, d)``. Returns the ``(K, K)`` symmetric matrix with ones on the diagonal;
    orthogonal criteria give the identity, and duplicated criteria give off-diagonal ones. This is the
    contamination matrix to first order under a linear head, since steering along ``v_j`` moves
    criterion ``k``'s readout by ``v_j · v_k``.
    """
    u = _unit_rows(directions)
    return u @ u.T


def welch_bound(k: int, d: int) -> float:
    """The Welch lower bound on maximum coherence for ``K`` unit vectors in ``d`` dimensions (A9).

    ``√((K − d) / (d(K − 1)))``. It is a real floor only when ``K > d``: below that ``K`` orthonormal
    vectors exist and the maximum coherence can be zero, so this returns ``0.0`` for ``K ≤ d``. An
    equiangular tight frame meets the bound with equality, which is the closed form the test checks.
    """
    if k <= d or k <= 1:
        return 0.0
    return float(np.sqrt((k - d) / (d * (k - 1))))


def max_offdiagonal_coherence(mu: np.ndarray) -> float:
    """The largest off-diagonal magnitude of a coherence matrix (the quantity the Welch floor bounds)."""
    m = np.asarray(mu, dtype=np.float64)
    k = m.shape[0]
    if k < 2:
        return 0.0
    off = m.copy()
    np.fill_diagonal(off, 0.0)
    return float(np.max(np.abs(off)))


def effective_dimension(directions: np.ndarray, spectrum: Any = None) -> float:
    """``d_eff``: the participation ratio of the criteria's spanned subspace (A9).

    With no explicit spectrum, uses the eigenvalues of the criteria's Gram matrix ``V Vᵀ`` (equivalently
    the squared singular values of ``V``), so ``K`` orthonormal criteria give ``d_eff = K`` and ``K``
    identical criteria give ``d_eff = 1``, the closed forms the test checks. Passing a reward-Hessian
    ``SpectrumResult`` (or any eigenvalue array) routes ``participation_ratio`` at the Hessian subspace
    A9 names instead.
    """
    if spectrum is not None:
        return participation_ratio(spectrum)
    u = _unit_rows(directions)
    gram = u @ u.T
    eigs = np.linalg.eigvalsh(gram)
    return participation_ratio(eigs)


def coherence_report(directions: np.ndarray, spectrum: Any = None) -> dict[str, Any]:
    """Bundle the A9 quantities: coherence matrix, Welch floor, contamination, and ``d_eff``.

    Returns the Gram matrix ``μ``, the maximum off-diagonal coherence, the Welch bound for ``(K, d)``,
    whether the criteria are in the over-packed regime ``K > d_eff``, the mean absolute contamination,
    and ``d_eff``. The over-packed flag is A9's capacity signal: past it the criteria must overlap and
    the reward starts leaking into dark channels.
    """
    v = np.asarray(directions, dtype=np.float64)
    k, d = v.shape
    mu = coherence_matrix(v)
    max_off = max_offdiagonal_coherence(mu)
    bound = welch_bound(k, d)
    d_eff = effective_dimension(v, spectrum)
    off = mu.copy()
    np.fill_diagonal(off, 0.0)
    mean_contam = float(np.mean(np.abs(off))) if k > 1 else 0.0
    return {
        "coherence_matrix": mu,
        "contamination_matrix": mu,  # first-order linear contamination equals the coherence
        "max_offdiagonal_coherence": max_off,
        "welch_bound": bound,
        "meets_welch_floor": bool(max_off >= bound - 1e-9) if k > d else None,
        "d_eff": d_eff,
        "overpacked": bool(k > d_eff + 1e-9),
        "mean_contamination": mean_contam,
        "n_criteria": int(k),
        "d_model": int(d),
    }


class Coherence(BaseObservable):
    """A9 cross-criterion coherence, contamination, Welch floor, and ``d_eff``.

    Requires a multi-readout signal (a reward exposing several criterion directions). The criterion
    directions are taken from the signal's ``criterion:k`` readouts (ArmoRM's objectives are first-class,
    never a row mean), or injected directly for the synthetic test. Reports the coherence matrix, the
    Welch bound and whether the criteria are over-packed past ``d_eff``, and the mean contamination, with
    the mean coherence read against a random-direction null so "the criteria are more aligned than
    chance" beats the high-dimensional noise floor. Gauge is INVARIANT: the Gram of one signal's own
    criteria is invariant under a shared orthogonal transform of the representation; a cross-signal
    criterion comparison would be COVARIANT and frame-gated, which is noted as a deviation.

    What it cannot do. The contamination matrix reported here *is* the coherence matrix: to first
    order under a linear head the two coincide, so nothing in this payload is causal evidence that
    steering criterion j moves criterion k, only that their directions overlap. The Welch floor is a
    statement about how many nearly-orthogonal directions fit in ``d`` ambient dimensions, and it
    only binds once ``K > d``; on a wide head with a handful of criteria it returns 0.0 and the
    ``meets_welch_floor`` field is None rather than a pass. ``d_eff`` is the participation ratio of
    the criteria's own Gram unless a Hessian spectrum is supplied, which measures the dimension the
    criteria span and not the dimension the reward has available to spend.
    """

    name = "Coherence"
    version = "1.0"
    capabilities = Capability.MULTI_READOUT
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A9"
    deviations = (
        "contamination is the first-order linear leakage mu_jk; measured causal contamination from "
        "steering is the production path",
        "within one signal's criterion basis the Gram is rotation-invariant (INVARIANT); a "
        "cross-signal criterion comparison would be COVARIANT and frame-gated",
    )

    # -- the observable declarations ---------------------------------------
    quantity = "grader.criterion_coherence"
    requires: AccessMatrix = {Component.GRADER: Access.FORWARD}
    #: NEURAL_SCALAR only. The criterion directions are rows of a multi-objective head; a GenRM has
    #: no per-criterion weight vector and a PROCEDURAL rubric ensemble has criteria with no
    #: geometry, so neither has a Gram matrix to read.
    substrates = frozenset({Substrate.NEURAL_SCALAR})
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: The Gram of unit-normalised rows is unchanged when the same orthogonal map acts on every row,
    #: and `repr.basis` is exactly that map.
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = ("baseline.random_direction_pair", "baseline.welch_bound")
    rung = 0

    def __init__(
        self,
        directions: np.ndarray | None = None,
        *,
        spectrum: Any = None,
        null_draws: int = 5000,
        seed: int = 0,
    ) -> None:
        self.directions = directions
        self.spectrum = spectrum
        self.null_draws = int(null_draws)
        self.seed = int(seed)

    def _criterion_directions(self, ctx: Context) -> np.ndarray:
        if self.directions is not None:
            return np.asarray(self.directions, dtype=np.float64)
        vecs = []
        for r in ctx.signal.readouts():
            if r.name.startswith("criterion:") and r.vector is not None:
                v = r.vector
                v = v.detach().to("cpu").numpy() if hasattr(v, "detach") else np.asarray(v)
                vecs.append(np.asarray(v, dtype=np.float64).ravel())
        if not vecs:
            raise ValueError("no criterion directions found on the signal and none injected")
        return np.stack(vecs, axis=0)

    def measure(self, ctx: Context) -> "Evidence":
        directions = self._criterion_directions(ctx)
        report = coherence_report(directions, self.spectrum)

        from reward_lens.stats.nulls import random_direction_null

        null = random_direction_null(
            report["max_offdiagonal_coherence"],
            d=report["d_model"],
            n=self.null_draws,
            seed=self.seed,
        )

        payload = {
            "coherence_matrix": report["coherence_matrix"],
            "contamination_matrix": report["contamination_matrix"],
            "max_offdiagonal_coherence": report["max_offdiagonal_coherence"],
            "welch_bound": report["welch_bound"],
            "meets_welch_floor": report["meets_welch_floor"],
            "d_eff": report["d_eff"],
            "overpacked": report["overpacked"],
            "mean_contamination": report["mean_contamination"],
            "coherence_null_p95": null["null_p95"],
            "coherence_null_mean": null["null_mean"],
            "n_criteria": report["n_criteria"],
            "d_model": report["d_model"],
        }
        return ctx.emit(payload, uncertainty=Uncertainty(n=report["n_criteria"], method="none"))


__all__ = [
    "coherence_matrix",
    "welch_bound",
    "max_offdiagonal_coherence",
    "effective_dimension",
    "coherence_report",
    "Coherence",
]

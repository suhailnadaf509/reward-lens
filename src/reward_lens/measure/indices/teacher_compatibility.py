"""A3 TeacherCompatibility: the induced reward variance ``w_rᵀ Σ_π w_r``.

Formal definition, A3. ``TC(rm, π) = Var_{y∼π}(w_rᵀ h(y)) = w_rᵀ Σ_π w_r``, the variance of
the reward projection over the on-policy activation distribution, decomposable by layer and by
spectral mode. This is Razin's teacher-induced variance (faithful_to Razin 2503.15477): a reward
model whose scores barely move across a policy's samples is a poor teacher for that policy no matter
how accurate its ranking, because RLHF's first-order learning signal is proportional to the reward
variance the policy actually sees. It is also L1's zeroth-order susceptibility and equals the ``f = r``
diagonal of the χ response identity (A12): ``TC = Cov_0(r, r) = Var(r)``.

Deviations from A3: none in the scalar. The layer decomposition reads each captured residual site's
reward projection variance, and the spectral decomposition splits ``w_rᵀ Σ w_r`` over the eigenbasis
of ``Σ`` (an exact, basis-free split of the same total), which A3 names "decomposable by
layer/feature" without fixing the feature basis; the eigenbasis is the canonical choice and is noted
here as the concrete reading.
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
    Site,
)
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.indices._support import (
    GRADER_STUDY_PHASES,
    MEASURED_BY,
    NEURAL_SUBSTRATES,
    final_activations,
    reward_vector,
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def teacher_compatibility(w_r: np.ndarray, activations: np.ndarray) -> float:
    """The induced reward variance ``TC = w_rᵀ Σ_π w_r`` (A3).

    Equivalently ``Var_y(w_rᵀ h(y))``: project every activation onto the reward direction and take the
    population variance of the resulting scores. The two forms agree exactly because the variance of a
    linear functional is that functional's quadratic form against the covariance, which is the identity
    the test asserts. ``activations`` is ``(n, d)`` and ``w_r`` is ``(d,)``.
    """
    a = np.asarray(activations, dtype=np.float64)
    w = np.asarray(w_r, dtype=np.float64).ravel()
    proj = a @ w
    return float(np.var(proj, ddof=0))


def teacher_compatibility_spectral(
    w_r: np.ndarray, activations: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Split ``w_rᵀ Σ w_r`` over the eigenbasis of ``Σ`` (A3, the feature decomposition).

    Writes the total as ``Σ_k λ_k (w_r · u_k)²`` for eigenpairs ``(λ_k, u_k)`` of the on-policy
    covariance ``Σ``. Each term is the contribution of one principal direction of the activation
    distribution to the reward variance, and they sum to ``teacher_compatibility`` exactly. For an
    independent-feature (diagonal ``Σ``) distribution the eigenbasis is the coordinate basis and the
    contributions reduce to ``σ_ii · w_i²``, which is the closed form the test checks. Returns
    ``(total, contributions_desc, eigenvalues_desc)`` sorted by descending contribution.
    """
    a = np.asarray(activations, dtype=np.float64)
    w = np.asarray(w_r, dtype=np.float64).ravel()
    cov = np.cov(a, rowvar=False, bias=True)
    cov = np.atleast_2d(cov)
    evals, evecs = np.linalg.eigh(cov)
    loadings = evecs.T @ w
    contribs = evals * loadings**2
    order = np.argsort(contribs)[::-1]
    return float(contribs.sum()), contribs[order], evals[order]


def teacher_compatibility_by_layer(
    w_r: np.ndarray, activations_by_site: dict[Any, np.ndarray]
) -> dict[str, float]:
    """Per-site induced variance (A3, the layer decomposition).

    Applies ``teacher_compatibility`` at each captured residual site with the same reward direction,
    tracing where across depth the policy's samples spread the reward. The keys are the string forms of
    the sites so the payload is JSON-clean.
    """
    return {
        str(site): teacher_compatibility(w_r, acts) for site, acts in activations_by_site.items()
    }


class TeacherCompatibility(BaseObservable):
    """A3 induced reward variance ``w_rᵀ Σ_π w_r``, layer- and spectrum-decomposable.

    Requires activation capture and a linear readout. Reads ``w_r`` off the readout, captures the
    on-policy final-token activations at the readout site (and at every residual layer when the signal
    reports its depth, for the layer decomposition), and reports the total with its spectral split.
    Gauge is INVARIANT: it is a single-signal functional, not a cross-signal comparison. It carries
    reward-scale² units, so a cross-model magnitude comparison first needs the two rewards on a common
    scale; that caveat is stated as a deviation rather than silently ignored.

    What it cannot do. This is a variance over whatever samples the view contains, so it is a
    statement about the reward model paired with *that* distribution and not about the reward model:
    move to a policy whose samples spread differently and the number changes with nothing about the
    grader having changed. It carries reward-scale-squared units, so two models' totals are not
    comparable until their rewards are on a common scale, and no part of this payload puts them
    there. A high induced variance says the teacher has signal to give, not that the signal points
    anywhere useful; a well-calibrated reward and a noisy one can report the same total.
    """

    name = "TeacherCompatibility"
    version = "1.0"
    capabilities = Capability.ACTIVATIONS | Capability.LINEAR_READOUT
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A3"
    deviations = (
        "carries reward-scale-squared units; cross-model magnitude comparison requires the two "
        "rewards on a common scale (INVARIANT is with respect to representation rotation, not "
        "reward rescaling)",
        "the feature decomposition is over the eigenbasis of the on-policy covariance, the "
        "canonical basis-free reading of A3's 'decomposable by feature'",
    )

    # -- the observable declarations ---------------------------------------
    quantity = "grader.induced_variance"
    requires: AccessMatrix = {Component.GRADER: Access.FORWARD}
    substrates = NEURAL_SUBSTRATES
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset(
            {RegimeCondition.STATIONARY_GRADER, RegimeCondition.GROUP_NONDEGENERATE}
        ),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: Two groups act on this reading and it transforms two different ways, which is exactly the
    #: case a single `invariance_relation` could not express.
    #:
    #: Under `repr.basis` it is invariant: ``w' Sigma w`` is unchanged when the same orthogonal map
    #: acts on the activations and on the readout.
    #:
    #: Under `reward.affine` (`r -> a*r + b`) it is **covariant at weight 2**. The readout scales to
    #: `a*w_r`, the offset does not reach a variance, and `TC = Var(w_r . h)` therefore scales by
    #: `a^2`. Measured, not asserted: over 2,000 draws of random `(n, d)` activations, random
    #: readouts, and scales log-uniform on [0.1, 10] at both signs, `TC(a*w) == a**2 * TC(w)` to a
    #: maximum relative error of 6.9e-16, and the weight is pinned rather than merely consistent,
    #: since weight 1 is off by a median relative 2.0 and weight 0 by 8.0 at `a = 3`.
    #:
    #: It is still not declared, for the reason `battery/lens.py` records at the same point. The
    #: `reward.affine` group action replaces `InvariancePayload.scores` and leaves `readouts` alone,
    #: and this reading is `w_r' Sigma w_r` off the readout, so the action never reaches it. The
    #: generated test then reports the reading unchanged where weight 2 predicts `a^2`: at
    #: `a = 6.699` it measures 5.34275 against an expected 239.778 and fails. Declaring invariant
    #: instead would turn it green by claiming the thing that is false, which is the substitution
    #: standing rule 4 exists to forbid. What would close it is an affine action that also scales
    #: the readout; until then the true relation stays here, where it is at least written down.
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = ("baseline.random_direction", "baseline.score_variance")
    rung = 0
    #: A white-box reading owes an `IncrementalValidity` and this instrument cannot produce
    #: one. The id is checkable and the prose is the argument.
    incremental_exemption = (
        "NO_PER_ITEM_VERDICT",
        "the reading is one variance of the reward projection over the view, plus its decomposition "
        "by layer and by spectral mode. A variance is a property of the item set rather than of any "
        "item, so there is no per-item verdict to be right or wrong about, and the score variance a "
        "black-box method could compute is already carried as a declared baseline rather than as a "
        "competitor with an error vector.",
    )

    def measure(self, ctx: Context) -> "Evidence":
        signal = ctx.signal
        w_r = reward_vector(signal, ctx.readout)
        site = None  # the readout's own site
        acts = final_activations(signal, ctx.view, site, readout=ctx.readout)

        total = teacher_compatibility(w_r, acts)
        _, contribs, evals = teacher_compatibility_spectral(w_r, acts)

        by_layer: dict[str, float] = {}
        n_layers = getattr(signal.meta, "n_layers", None)
        if n_layers:
            sites = {Site(layer, "resid_post"): None for layer in range(int(n_layers))}
            captured: dict[Any, np.ndarray] = {}
            for s in sites:
                try:
                    captured[s] = final_activations(signal, ctx.view, s, readout=ctx.readout)
                except Exception:  # noqa: BLE001 - a site the runtime cannot read is simply skipped
                    continue
            if captured:
                by_layer = teacher_compatibility_by_layer(w_r, captured)

        payload = {
            "teacher_compatibility": total,
            "spectral_contributions": contribs.tolist(),
            "spectral_eigenvalues": evals.tolist(),
            "by_layer": by_layer,
            "d_model": int(w_r.size),
            "n_items": int(acts.shape[0]),
        }
        return ctx.emit(payload, uncertainty=Uncertainty(n=int(acts.shape[0]), method="none"))


__all__ = [
    "teacher_compatibility",
    "teacher_compatibility_spectral",
    "teacher_compatibility_by_layer",
    "TeacherCompatibility",
]

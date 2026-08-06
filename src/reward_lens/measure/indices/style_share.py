"""A6 Style Share: the style complement of the verification score.

Formal definition, A6. ``StyleShare =`` the fraction of the correctness-``Δr`` removed by
projecting the twin activation difference ``Δh`` onto the style subspace. Where the verification score
(``verification_score.py``) measures how much of the clean-vs-corrupted reward gap lives at the error
span, the style share measures how much of it the reward reads off style directions instead. ``VS`` and
``StyleShare`` need not sum to one; the residual is reward change explained by neither, and A6 keeps it
unexplained rather than forcing a partition.

Deviation from A6: the pure function computes the linear reward fraction carried by the style-subspace
projection of ``Δh`` under the reward direction ``w_r``; the production path supplies the style subspace
from the concept layer's style dictionary. The synthetic test drives it with a planted style component
of a known reward fraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.evidence import Uncertainty
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.types import Access, AccessMatrix, Capability, Component, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context, PreflightResult
from reward_lens.measure.indices._support import (
    GRADER_STUDY_PHASES,
    MEASURED_BY,
    NEURAL_SUBSTRATES,
    measured_without_input,
    missing_injection,
    reward_vector,
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def _orthonormalize(basis: np.ndarray) -> np.ndarray:
    """Orthonormalize the rows of a style basis (``m, d``) so the projector is idempotent."""
    b = np.asarray(basis, dtype=np.float64)
    if b.ndim == 1:
        b = b[None, :]
    q, _ = np.linalg.qr(b.T)
    return q.T  # (rank, d) orthonormal rows


def style_share(delta_h: np.ndarray, style_basis: np.ndarray, w_r: np.ndarray) -> float:
    """The style share ``= (w_r · P_style Δh) / (w_r · Δh)`` (A6).

    Projects the clean-vs-corrupted activation difference ``Δh`` onto the (orthonormalized) style
    subspace and reports the fraction of the reward change ``w_r · Δh`` that the projection carries. A
    reward that responds to the corruption purely through style directions has a style share near one; a
    reward that responds through the error content has a style share near zero. ``Δh`` is ``(d,)`` (or an
    ``(n, d)`` batch, averaged); ``style_basis`` is ``(m, d)``.
    """
    dh = np.asarray(delta_h, dtype=np.float64)
    if dh.ndim == 2:
        dh = dh.mean(axis=0)
    w = np.asarray(w_r, dtype=np.float64).ravel()
    q = _orthonormalize(style_basis)
    projected = q.T @ (q @ dh)  # P_style delta_h
    total = float(w @ dh)
    if total == 0:
        return float("nan")
    return float((w @ projected) / total)


class StyleShare(BaseObservable):
    """A6 fraction of the correctness reward gap the reward reads off style directions.

    Requires activations and a linear readout on the production path (``Δh`` from clean/corrupted twins,
    the style subspace from the concept layer's style dictionary). Here ``Δh`` and the style basis are
    injected so the projection arithmetic is exercised directly. Gauge is INVARIANT: the style share is a
    reward fraction.

    What it cannot do. The style subspace is supplied, so this measures the share carried by
    whatever directions the caller called style; a subspace that quietly contains the error content
    reports a high style share for a reward that is reading the error correctly. The denominator is
    ``w_r . delta_h``, which goes through zero, and the function returns NaN there rather than a
    bound, so a reward whose twins barely differ produces no reading at all. Verification score and
    style share do not sum to one and must not be presented as a partition; the remainder is reward
    change that neither explains.
    """

    name = "StyleShare"
    version = "1.0"
    capabilities = Capability.ACTIVATIONS | Capability.LINEAR_READOUT
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A6"
    deviations = (
        "consumes an injected delta_h and style subspace; the twin activation difference and the "
        "style dictionary are the production path (interventions + concepts)",
    )

    # -- the observable declarations ---------------------------------------
    quantity = "grader.style_share"
    #: The reward direction is read off the head; the twin activation difference and the style
    #: basis are recorded from an earlier corruption experiment and from the concept layer.
    requires: AccessMatrix = {
        Component.GRADER: Access.FORWARD,
        Component.RECORD: Access.RECORD,
    }
    substrates = NEURAL_SUBSTRATES
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.STATIONARY_GRADER, RegimeCondition.ABOVE_LOD}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: The share is a ratio of two inner products against the same readout, with the projector
    #: rotating along with the basis, so a shared orthogonal change of basis leaves it unchanged.
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = ("baseline.random_subspace", "baseline.full_space")
    rung = 0
    #: A white-box reading owes an `IncrementalValidity` and this instrument cannot produce
    #: one. The id is checkable and the prose is the argument.
    incremental_exemption = (
        "NO_PER_ITEM_VERDICT",
        "the reading is a single ratio of two inner products, computed from an injected activation "
        "difference and an injected style basis. A batch of delta_h is averaged to one vector before "
        "the share is taken, so even the per-item structure the caller passes in is gone by the time "
        "a number exists, and no black-box method produces a rival style share.",
    )

    def __init__(
        self,
        delta_h: np.ndarray | None = None,
        style_basis: np.ndarray | None = None,
    ) -> None:
        self.delta_h = delta_h
        self.style_basis = style_basis

    def preflight(self, ctx: Context) -> PreflightResult:
        """The activation difference and the style subspace, or a refusal.

        The injected input is absent, which makes this a `Refusal` rather than an Evidence
        carrying a note. Nothing has to be computed to know it, so the question belongs
        here: `estimate` returns this refusal before `measure` is reached, and the
        capability report gets it with no work at all.
        """
        if self.delta_h is None or self.style_basis is None:
            return missing_injection(
                self,
                needs={
                    "delta_h": "the clean-minus-corrupted activation difference at the readout site",
                    "style_basis": "a (k, d) set of directions spanning the style subspace",
                },
                have="neither was injected",
                remedy=(
                    "Construct `StyleShare(delta_h=..., style_basis=...)`. `delta_h` is the activation "
                    "difference between the clean and corrupted runs at the readout site; `style_basis` is "
                    "the set of directions you are calling style, one row each, in the same `d`. They are "
                    "orthonormalized here, so they have to be independent but need not be orthogonal."
                ),
            )
        return super().preflight(ctx)

    def measure(self, ctx: Context) -> "Evidence":
        if self.delta_h is None or self.style_basis is None:
            raise measured_without_input(self)
        w_r = reward_vector(ctx.signal, ctx.readout)
        share = style_share(self.delta_h, self.style_basis, w_r)
        return ctx.emit(
            {"style_share": share, "style_dim": int(_orthonormalize(self.style_basis).shape[0])},
            uncertainty=Uncertainty(method="none"),
        )


__all__ = ["style_share", "StyleShare"]

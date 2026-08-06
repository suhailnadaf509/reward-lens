"""A11 Interpretability coverage: reward through interpretable features vs error nodes.

Formal definition, A11. The fraction of reward routed through interpretable features versus
reconstruction/error nodes, read off attribution graphs. It is the honest single-number successor to
E04's indictment: instead of asserting that a reward is or is not interpretable, it reports what
fraction of the reward the interpretable features actually carry, with the reconstruction/error nodes
(the part of the residual the feature dictionary fails to reconstruct) as the explicit complement.

Deviation from A11: the pure function is the accounting over supplied node contributions; the
production path supplies those contributions from an attribution graph with a scalar reward sink
(the ``attribution`` subsystem). The synthetic test drives it with planted feature and error-node
contributions of a known split.
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
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def interp_coverage(feature_contributions: np.ndarray, error_contributions: np.ndarray) -> float:
    """Fraction of reward magnitude carried by interpretable features vs error nodes (A11).

    ``= Σ|feature| / (Σ|feature| + Σ|error|)`` over the attribution node contributions. All reward
    through named features gives ``1`` (fully covered); all through reconstruction/error nodes gives
    ``0`` (E04's worst case). Magnitudes are summed so cancelling signed contributions do not inflate
    the coverage. Both inputs are contribution arrays (any shape); they are flattened.
    """
    f = float(np.sum(np.abs(np.asarray(feature_contributions, dtype=np.float64))))
    e = float(np.sum(np.abs(np.asarray(error_contributions, dtype=np.float64))))
    denom = f + e
    if denom == 0:
        return float("nan")
    return f / denom


class InterpCoverage(BaseObservable):
    """A11 fraction of reward routed through interpretable features rather than error nodes.

    Requires activations on the production path (the attribution graph is built from them). The feature
    and error-node contributions are injected here; the production path supplies them from an attribution
    graph with a scalar reward sink. Gauge is INVARIANT: a coverage fraction is scale-free.

    What it cannot do. The split between feature nodes and error nodes is supplied, so this
    instrument reports an accounting it did not perform and cannot check. A dictionary that
    reconstructs the residual well produces a high coverage whether or not the features it uses mean
    anything, so the number measures reconstruction and is read as interpretability; that gap is the
    whole reason the error-node complement is reported beside it rather than folded in.
    """

    name = "InterpCoverage"
    version = "1.0"
    capabilities = Capability.ACTIVATIONS
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "A11"
    deviations = (
        "consumes injected node contributions; the attribution graph with a scalar reward sink is "
        "the production path (attribution subsystem)",
    )

    # -- the observable declarations ---------------------------------------
    quantity = "instrument.interp_coverage"
    #: Both contributions come from an attribution graph built earlier. Nothing here touches the
    #: grader.
    requires: AccessMatrix = {Component.RECORD: Access.RECORD}
    substrates = NEURAL_SUBSTRATES
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.STATIONARY_GRADER}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: A ratio of contribution masses that were themselves computed upstream. No invariance
    #: group acts on it once the graph is fixed, so `trivial` is the answer rather than an
    #: omission, and its generated test passes vacuously.
    invariance = "trivial"
    invariance_relation = INVARIANT
    baselines = ("baseline.random_feature_split", "baseline.error_nodes_only")
    rung = 0
    #: A white-box reading owes an `IncrementalValidity` and this instrument cannot produce
    #: one. The id is checkable and the prose is the argument.
    incremental_exemption = (
        "NO_PER_ITEM_VERDICT",
        "the reading is a single coverage fraction: the share of attribution mass carried by named "
        "feature nodes against reconstruction and error nodes. One number, computed from "
        "contributions supplied by an earlier attribution pass, with no items, no labels, and no "
        "black-box method that produces a rival share to be incremental over.",
    )

    def __init__(
        self,
        feature_contributions: np.ndarray | None = None,
        error_contributions: np.ndarray | None = None,
    ) -> None:
        self.feature_contributions = feature_contributions
        self.error_contributions = error_contributions

    def preflight(self, ctx: Context) -> PreflightResult:
        """Both halves of the attribution split, or a refusal. One half makes the coverage 1 by construction.

        The injected input is absent, which makes this a `Refusal` rather than an Evidence
        carrying a note. Nothing has to be computed to know it, so the question belongs
        here: `estimate` returns this refusal before `measure` is reached, and the
        capability report gets it with no work at all.
        """
        if self.feature_contributions is None or self.error_contributions is None:
            return missing_injection(
                self,
                needs={
                    "feature_contributions": "the attribution contributions of the named features",
                    "error_contributions": "the attribution contributions of the reconstruction and error nodes",
                },
                have="neither was injected",
                remedy=(
                    "Construct `InterpCoverage(feature_contributions=..., error_contributions=...)` from one "
                    "attribution pass over the same samples. The reading is the share of total reward "
                    "magnitude the named features carry, so it needs both halves of the split: passing only "
                    "the features would make the coverage 1 by construction rather than by measurement."
                ),
            )
        return super().preflight(ctx)

    def measure(self, ctx: Context) -> "Evidence":
        if self.feature_contributions is None or self.error_contributions is None:
            raise measured_without_input(self)
        coverage = interp_coverage(self.feature_contributions, self.error_contributions)
        return ctx.emit(
            {"interp_coverage": coverage, "error_share": 1.0 - coverage},
            uncertainty=Uncertainty(method="none"),
        )


__all__ = ["interp_coverage", "InterpCoverage"]

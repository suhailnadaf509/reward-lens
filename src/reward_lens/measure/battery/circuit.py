"""``CircuitJaccard`` (E05): how much two models' reward circuits overlap.

If two reward models attend to the same components to form a preference, their circuits overlap; if
they route the preference through different components, they do not, even when they agree on the
answer. The overlap is a Jaccard index over the top attributed components of each model, which is a
cross-model comparison and so a first-class question about kinship and monoculture.

v1 had no circuit-overlap primitive; its ``ModelComparator`` correlated preference-formation curves
instead. This Observable builds the missing piece from the attribution the battery already computes:
run :class:`~reward_lens.measure.battery.dla.DirectLinearAttribution` on each model, take each model's
top-k components by mean absolute differential contribution, and report the Jaccard of the two sets.
The components are matched by name (``attn_L{l}`` / ``mlp_L{l}``), which requires the two models to
share an architecture; the overlap is a set membership over named components, not a geometric
quantity, so it needs no gauge frame and is reported INVARIANT.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.errors import CapabilityError
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.types import Access, AccessMatrix, Capability, Component, GaugeStatus
from reward_lens.measure.base import BaseObservable, Context, run
from reward_lens.measure.battery._common import (
    GRADER_STUDY_PHASES,
    MEASURED_BY,
    NEURAL_SUBSTRATES,
)
from reward_lens.measure.battery.dla import DirectLinearAttribution

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def jaccard(a: set, b: set) -> float:
    """The Jaccard index ``|a & b| / |a | b|`` of two sets (1.0 when both are empty)."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class CircuitJaccard(BaseObservable):
    """Top-component circuit overlap between two reward models (E05).

    ``ctx.others[0]`` is the second signal to compare against ``ctx.signal``; both are attributed on
    the same view. ``top_k`` (from ``ctx.regime['top_k']``, default 5) sets the circuit size. Requires
    activation capture and a linear readout. INVARIANT: the overlap is a set membership over named
    components, not a covariant geometric quantity.

    What it cannot do. Components are matched by name, so the two models must share an architecture
    and a layer count; nothing here checks that, and a mismatch produces a small Jaccard that reads
    as a scientific result rather than as a comparison that should not have been made. The overlap
    is a hard top-k cut, so two models whose contributions agree closely but straddle the cut at
    rank k report disagreement, and the payload carries no sensitivity to k. The declaration of
    ``LINEAR_READOUT`` is kept because this instrument runs ``DirectLinearAttribution`` twice and
    that reaches the readout vector; the requirement is transitive, not decorative.
    """

    name = "CircuitJaccard"
    version = "1.0"
    capabilities = Capability.ACTIVATIONS | Capability.LINEAR_READOUT
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "E05 circuit overlap"
    deviations = (
        "overlap is the Jaccard of the top-k components by mean absolute DLA contribution; a v1 "
        "circuit-Jaccard did not exist, so this is a new construction on the same attribution",
    )

    # -- the declarations --------------------------------------------------
    quantity = "grader.circuit_overlap"
    requires: AccessMatrix = {Component.GRADER: Access.FORWARD}
    substrates = NEURAL_SUBSTRATES
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.STATIONARY_GRADER, RegimeCondition.ABOVE_LOD}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: The reading is a set overlap over component *names*, so no change of representation basis
    #: touches it; the attribution it ranks on is itself basis-invariant.
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = ("baseline.random_top_k", "baseline.same_model_split_half")
    rung = 0
    #: An `IncrementalValidity` is required on every white-box reading and this
    #: instrument cannot produce one. The id is checkable and the prose is the argument.
    incremental_exemption = (
        "NO_PER_ITEM_VERDICT",
        "the reading is one Jaccard index over two models' top-k component sets. There is a single "
        "number per model pair and no per-item verdict in it, so there is no error vector to "
        "correlate: an error needs an item the instrument can be right or wrong about, and nothing in "
        "the data says which components two circuits ought to have shared. No black-box method "
        "produces a rival circuit overlap either, so there is no competitor to be incremental over.",
    )

    def measure(self, ctx: Context) -> "Evidence":
        if not ctx.others:
            raise CapabilityError(
                "CircuitJaccard is a cross-model comparison; pass the second signal in Context.others"
            )
        top_k = int(ctx.regime.get("top_k", 5))
        other = ctx.others[0]

        circuit_a, contrib_a = self._circuit(ctx.signal, ctx.view, ctx.readout, top_k)
        circuit_b, contrib_b = self._circuit(other, ctx.view, ctx.readout, top_k)
        overlap = jaccard(set(circuit_a), set(circuit_b))

        payload = {
            "top_k": top_k,
            "circuit_a": circuit_a,
            "circuit_b": circuit_b,
            "jaccard": overlap,
            "shared_components": sorted(set(circuit_a) & set(circuit_b)),
        }
        return ctx.emit(payload, subject_extra={"other": str(other.meta.fingerprint)})

    @staticmethod
    def _circuit(signal, view, readout, top_k) -> tuple[list[str], np.ndarray]:
        """The top-k components of one model by mean absolute DLA differential contribution."""
        ev = run(DirectLinearAttribution(), Context(signal=signal, view=view, readout=readout))
        names = ev.value["component_names"]
        differential = np.array(ev.value["differential"])  # (n_pairs, n_components)
        strength = np.abs(differential).mean(axis=0)
        order = np.argsort(strength)[::-1][:top_k]
        return [names[i] for i in order], strength


__all__ = ["CircuitJaccard", "jaccard"]

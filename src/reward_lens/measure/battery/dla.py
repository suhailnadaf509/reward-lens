"""``DirectLinearAttribution`` (E03/E04): per-component reward decomposition.

Because the residual stream is a sum of component outputs and the reward is a linear read of the final
residual, the reward differential of a preference pair decomposes exactly into signed per-component
contributions: the embedding, and each layer's attention and MLP output, projected onto the reward
direction. Positive means the component pushes the chosen completion's reward above the rejected
one's. This is observational, not causal, which is the distinction E04 makes when it correlates this
ranking against the patching ranking (the faithfulness rho).

The decomposition is computed through the one canonical implementation in
:mod:`reward_lens.attribution.dla`, the same code the v1 primitive now calls, so a v3 attribution and
a v1 attribution are the same numbers by construction. The E-parity suite checks that against v1's
``ComponentAttribution`` on the tiny model to 1e-6, and checks the completeness identity that the
contributions sum to the reward differential.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from reward_lens.attribution.dla import component_reward_contributions
from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.types import Access, AccessMatrix, Capability, Component, GaugeStatus, Site
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.battery._common import (
    GRADER_STUDY_PHASES,
    MEASURED_BY,
    NEURAL_SUBSTRATES,
    capture_sites,
    component_sites,
    pair_sides,
    reward_direction,
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


def _component_names(n_layers: int) -> tuple[list[str], list[str], list[int]]:
    """The component names, kinds, and layer indices in v1's ``ComponentAttribution`` order."""
    names = ["embed"]
    kinds = ["embed"]
    layers = [-1]
    for layer in range(n_layers):
        names.append(f"attn_L{layer}")
        kinds.append("attn")
        layers.append(layer)
        names.append(f"mlp_L{layer}")
        kinds.append("mlp")
        layers.append(layer)
    return names, kinds, layers


class DirectLinearAttribution(BaseObservable):
    """Signed per-component reward attribution for preference pairs (E03/E04).

    Requires activation capture and a linear readout. The contributions are in reward units and sum
    to the reward differential; the invariant scientific summary is their ranking, which E04 compares
    to the patching ranking. Marked INVARIANT because the reward is gauge-fixed by the head and the
    faithfulness statistic is rank based.

    What it cannot do. This is observational. A component can carry a large contribution and not be
    causally necessary, because the components that would compensate for it are still running; that
    is the distinction ``PatchGrid`` exists to measure and the reason E04 correlates the two
    rankings rather than treating either as the answer. The envelope requires ``ABOVE_LOD`` because
    the decomposition is exact by construction: it will always sum to the differential, including
    when that differential is entirely the grader disagreeing with itself.
    """

    name = "DirectLinearAttribution"
    version = "1.0"
    capabilities = Capability.ACTIVATIONS | Capability.LINEAR_READOUT
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "E03/E04 direct linear attribution"
    deviations = (
        "raw contributions are in reward units (reward-scale dependent); the invariant summary is "
        "the component ranking, which is what E04's Spearman faithfulness uses",
    )

    # -- the declarations --------------------------------------------------
    quantity = "grader.component_attribution"
    requires: AccessMatrix = {Component.GRADER: Access.FORWARD}
    substrates = NEURAL_SUBSTRATES
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.STATIONARY_GRADER, RegimeCondition.ABOVE_LOD}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: Each contribution is an inner product of a residual component with the readout, so a shared
    #: orthogonal change of basis leaves every one of them unchanged, and the ranking with them.
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = ("baseline.random_direction", "baseline.uniform_attribution")
    rung = 0
    #: An `IncrementalValidity` is required on every white-box reading and this
    #: instrument cannot produce one. The id is checkable and the prose is the argument.
    incremental_exemption = (
        "NO_SUBJECT_WITH_SIGNAL",
        "the per-component differential is a genuine per-pair quantity and the chosen/rejected label "
        "is real and not produced here, so this instrument can carry a record. What it cannot get is "
        "a subject: the only offline grader in this build is `signals.from_tiny`, a randomly "
        "initialised two-layer LlamaForSequenceClassification. Measured on the 84-pair diagnostic set "
        "unfolded to 168 items, it scores AUROC 0.5568 against 0.6892 for the TF-IDF baseline, so the "
        "white-box side of the comparison has no signal to contribute and the four numbers would "
        "describe the fixture rather than the instrument. This needs a trained reward model on a "
        "preference set whose lineages are declared, at which point the shared task is the unfolded "
        "preference decision and the baselines read the completion text.",
    )

    def measure(self, ctx: Context) -> "Evidence":
        signal = ctx.signal
        n_layers = int(signal.meta.n_layers)
        w_r = reward_direction(signal, ctx.readout)
        sites = component_sites(n_layers)
        names, kinds, layers = _component_names(n_layers)

        chosen, rejected = pair_sides(ctx.view)
        cap_c = capture_sites(signal, chosen, sites)
        cap_r = capture_sites(signal, rejected, sites)

        # One column per component, in the canonical order; each is (chosen - rejected) . w_r.
        contrib_c = np.stack(
            [
                component_reward_contributions(cap_c[Site(*_site(name, layer))], w_r)
                for name, layer in zip(names, layers)
            ],
            axis=1,
        )
        contrib_r = np.stack(
            [
                component_reward_contributions(cap_r[Site(*_site(name, layer))], w_r)
                for name, layer in zip(names, layers)
            ],
            axis=1,
        )
        differential = contrib_c - contrib_r  # (n_pairs, n_components)

        # Completeness: the contributions sum to the reward differential (up to the head bias, which
        # cancels in the difference). This is the w_r identity the E-parity suite asserts.
        reward_diff = differential.sum(axis=1)
        total_abs = np.abs(differential).sum(axis=1)
        # The dominant component per pair (largest signed magnitude), a stable summary of the ranking.
        dominant = [
            names[int(np.argmax(np.abs(differential[i])))] for i in range(differential.shape[0])
        ]

        payload = {
            "component_names": names,
            "component_kinds": kinds,
            "component_layers": layers,
            "differential": differential.tolist(),
            "contributions_chosen": contrib_c.tolist(),
            "contributions_rejected": contrib_r.tolist(),
            "reward_differential": reward_diff.tolist(),
            "dominant_component": dominant,
            "mean_abs_contribution": float(np.mean(total_abs)),
            "n_pairs": int(differential.shape[0]),
        }
        return ctx.emit(payload)


def _site(name: str, layer: int) -> tuple[int, str]:
    """Map a component name to its capture site ``(layer, point)``."""
    if name == "embed":
        return (-1, "embed")
    if name.startswith("attn_"):
        return (layer, "attn_out")
    return (layer, "mlp_out")


__all__ = ["DirectLinearAttribution"]

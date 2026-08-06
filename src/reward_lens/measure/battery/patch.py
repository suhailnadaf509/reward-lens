"""``PatchGrid`` (E15): causal patch effects across components and heads.

Attribution is observational: a component can carry a large contribution yet not be causally necessary
because other components compensate. Activation patching answers the causal question by splicing a
component's activation from the rejected forward into the chosen forward and reading how much the
reward differential shrinks. A large shrink means the component is causally load-bearing for the
preference.

This Observable reproduces v1's noising ``patch_all_components``: for each pair it patches every
attention and MLP output (or, at head granularity, every attention head) with the rejected side's
activation and measures ``original_differential - patched_differential``. The patching itself is an
:class:`~reward_lens.interventions.patch.ComponentPatch` / ``HeadPatch``, an Intervention whose
fingerprint enters the Evidence subject, so a patched number cannot masquerade as a clean one. The
head-granularity path is what E15 uses to find the strongest attention head; that headline is
recorded in ``fixtures/e_parity/golden.json`` and its recompute at 8B is GPU-gated (it needs the 8B
model's ``w_r`` and forwards), so on this hardware the head grid runs on the tiny model as a
correctness check, not as the 8B result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.types import Access, AccessMatrix, Capability, Component, GaugeStatus, Site
from reward_lens.interventions.patch import ComponentPatch, HeadPatch, run_patched_scores
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.battery._common import (
    GRADER_STUDY_PHASES,
    MEASURED_BY,
    NEURAL_SUBSTRATES,
    capture_sites,
    pair_sides,
)

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


class PatchGrid(BaseObservable):
    """Causal patch effects for every component (or head) on preference pairs (E15).

    ``granularity`` is ``"component"`` (attention and MLP sublayers, the default) or ``"head"`` (every
    attention head). Requires activation capture and a linear readout. Patch effects are in reward
    units and are gauge-invariant within a signal.

    What it cannot do. A patch tests necessity and says nothing about sufficiency, and there is no
    rescue arm here: nothing puts the activation back and checks the differential returns, so an
    off-manifold artifact and a real causal effect look identical in this payload. The patched
    activation comes from the rejected side of the same pair, which makes the source distribution
    close but not matched, so a large effect at a component the two sides differ on for unrelated
    reasons is a confound this instrument cannot separate. The envelope requires ``ABOVE_LOD``
    because every cell of the grid is a difference of two rewards and the grid is large: at 32
    components the largest of 32 noise draws looks like a finding.
    """

    name = "PatchGrid"
    version = "1.0"
    capabilities = Capability.ACTIVATIONS | Capability.LINEAR_READOUT
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "E15 head path patching / activation patching"
    deviations = (
        "noising convention (patch the rejected activation into the chosen forward); patched reward "
        "is the fp32 head projection, which matches the native head to head-in-fp32 tolerance",
    )

    # -- the declarations --------------------------------------------------
    quantity = "grader.component_patch_effect"
    requires: AccessMatrix = {Component.GRADER: Access.FORWARD | Access.MUTATE}
    substrates = NEURAL_SUBSTRATES
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.STATIONARY_GRADER, RegimeCondition.ABOVE_LOD}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: The effect is a difference of two reward projections, both inner products against the same
    #: readout, so a shared orthogonal change of basis leaves every cell of the grid unchanged.
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = ("baseline.self_patch", "baseline.random_component_patch")
    rung = 0
    #: An `IncrementalValidity` is required on every white-box reading and this
    #: instrument cannot produce one. The id is checkable and the prose is the argument.
    incremental_exemption = (
        "NO_SUBJECT_WITH_SIGNAL",
        "each cell is a per-pair causal effect, so per-item numbers exist, but a patch effect is a "
        "statement about necessity rather than a verdict about which side wins, and the shared task "
        "would have to be the preference decision the effects are computed against. That task has no "
        "subject here: the only offline grader in this build is `signals.from_tiny`, a randomly "
        "initialised two-layer LlamaForSequenceClassification. Measured on the 84-pair diagnostic set "
        "unfolded to 168 items, it scores AUROC 0.5568 against 0.6892 for the TF-IDF baseline, so the "
        "white-box side of the comparison has no signal to contribute and the four numbers would "
        "describe the fixture rather than the instrument. The grid also costs one forward per "
        "component per pair, so a record on a real grader is a GPU decision rather than a free one.",
    )

    def __init__(self, granularity: str = "component") -> None:
        if granularity not in ("component", "head"):
            raise ValueError(f"granularity must be 'component' or 'head', got {granularity!r}")
        self.granularity = granularity

    def measure(self, ctx: Context) -> "Evidence":
        signal = ctx.signal
        n_layers = int(signal.meta.n_layers)
        n_heads = int(signal.meta.n_heads or 0)
        readout = ctx.readout

        chosen, rejected = pair_sides(ctx.view)
        names = self._component_names(n_layers, n_heads)
        effects = np.zeros((len(chosen), len(names)))
        originals = np.zeros(len(chosen))

        for i, (chosen_item, rejected_item) in enumerate(zip(chosen, rejected)):
            reward_c = self._score(signal, [chosen_item], readout)[0]
            reward_r = self._score(signal, [rejected_item], readout)[0]
            original_diff = reward_c - reward_r
            originals[i] = original_diff
            patches = self._patches(signal, rejected_item, n_layers, n_heads)
            for j, patch in enumerate(patches):
                compiled = patch.compile(signal)
                patched_reward = float(
                    run_patched_scores(signal, compiled, [chosen_item], readout)[0]
                )
                patched_diff = patched_reward - reward_r
                effects[i, j] = original_diff - patched_diff

        mean_effect = effects.mean(axis=0)
        mean_abs = np.abs(effects).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        top = [(names[k], float(mean_abs[k])) for k in order[:10]]

        payload = {
            "granularity": self.granularity,
            "component_names": names,
            "mean_effect": mean_effect.tolist(),
            "mean_abs_effect": mean_abs.tolist(),
            "per_pair_effect": effects.tolist(),
            "original_differential": originals.tolist(),
            "top_components": top,
            "top_component": top[0][0] if top else None,
            "max_abs_effect": float(mean_abs.max()) if len(names) else 0.0,
            "n_pairs": len(chosen),
        }
        return ctx.emit(payload)

    # -- helpers ------------------------------------------------------------

    def _component_names(self, n_layers: int, n_heads: int) -> list[str]:
        if self.granularity == "head":
            return [
                f"head_L{layer}_H{head}" for layer in range(n_layers) for head in range(n_heads)
            ]
        names: list[str] = []
        for layer in range(n_layers):
            names.append(f"attn_L{layer}")
            names.append(f"mlp_L{layer}")
        return names

    def _patches(self, signal: Any, rejected_item: Any, n_layers: int, n_heads: int) -> list[Any]:
        """Build one patch per component/head, sourced from the rejected side's activation."""
        if self.granularity == "head":
            sites = tuple(
                Site(layer, "head_out", head)
                for layer in range(n_layers)
                for head in range(n_heads)
            )
            source = capture_sites(signal, [rejected_item], sites, full_sequence=True)
            return [
                HeadPatch(site=site, source_head=source[site], n_heads=n_heads) for site in sites
            ]
        sites: list[Site] = []
        for layer in range(n_layers):
            sites.append(Site(layer, "attn_out"))
            sites.append(Site(layer, "mlp_out"))
        source = capture_sites(signal, [rejected_item], tuple(sites), full_sequence=True)
        return [ComponentPatch(site=site, source=source[site], mode="replace") for site in sites]

    @staticmethod
    def _score(signal: Any, items: list[Any], readout: str) -> np.ndarray:
        return signal.score(items, readout).value.values


__all__ = ["PatchGrid"]

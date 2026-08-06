"""``PathEffect`` (E15): two-hop head-level path patching.

Direct patching tells you a component matters; path patching tells you whether its effect flows
through a particular downstream path. The sender is an attention head, the receiver a downstream
layer, and the path effect is the change in reward when only the sender to receiver path carries the
source-side activation while every other path stays clean. This is Goldowsky-Dill et al.'s
construction at head granularity, the resolution v1 settled on because sublayer-level path patching is
uninformative.

This is a working port of v1's ``PathPatcher``. It computes the sender head's residual contribution on
both the source (rejected) and target (chosen) sides through the ``o_proj`` weight slice, then adds the
difference at the receiver's residual input as a :class:`~reward_lens.interventions.patch.ResidualAddPatch`
and reads the reward change. The full E15 head-effect leaderboard at 8B is GPU/``w_r``-gated (it needs
the 8B model's reward head and forwards); here the mechanism runs on the tiny model as a correctness
check. The sender and receiver default to the first head into the last layer and can be set through
``ctx.regime['sender']`` / ``ctx.regime['receiver']``.

**The head projection is read through a protocol.** This instrument used to reach
``signal.runtime.adapter.get_attn_o_proj(signal.runtime.adapter.get_layers(signal.runtime.model)[layer])``,
four attribute hops past the last protocol call, through an architecture-adapter ABC an instrument has
no business knowing about and into the module tree the runtime exists to hide. It worked and it meant
the instrument could only ever run against one backend. It now calls
:func:`reward_lens.policy.base.site_weights`, which returns the runtime itself when the runtime
implements ``weight_at`` and otherwise wraps it in the single module-tree adapter this library has. A
backend that answers natively, which ``policy.hf.HFPolicyRuntime`` does, makes this instrument portable
to it without the instrument changing, and that is what lets the same path-patching mechanism read a
policy's attention head and a grader's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from reward_lens.core.envelope import EnvelopeSpec, RegimeCondition
from reward_lens.core.invariance import INVARIANT
from reward_lens.core.types import Access, AccessMatrix, Capability, Component, GaugeStatus, Site
from reward_lens.interventions.patch import ResidualAddPatch, run_patched_scores
from reward_lens.measure.base import BaseObservable, Context
from reward_lens.measure.battery._common import (
    GRADER_STUDY_PHASES,
    MEASURED_BY,
    NEURAL_SUBSTRATES,
    capture_sites,
    pair_sides,
)
from reward_lens.policy.base import site_weights

if TYPE_CHECKING:
    from reward_lens.core.evidence import Evidence


class PathEffect(BaseObservable):
    """Two-hop head-to-receiver path effect on reward for preference pairs (E15).

    Requires activation capture and a linear readout. The path effect is in reward units and
    gauge-invariant within a signal.

    What it cannot do. One sender and one receiver per call, chosen by the caller, so this measures
    a path somebody already suspected rather than finding one. The splice truncates to the shorter
    of the two sequences, so on pairs whose sides differ in length the tail of the longer side is
    never patched and the effect is an underestimate by an amount this payload does not report. The
    envelope requires ``ABOVE_LOD``: a two-hop effect is a difference of differences and is
    routinely smaller than the grader's disagreement with itself.
    """

    name = "PathEffect"
    version = "1.0"
    capabilities = Capability.ACTIVATIONS | Capability.LINEAR_READOUT
    gauge_status = GaugeStatus.INVARIANT
    faithful_to = "E15 head path patching"
    deviations = (
        "single sender head to single receiver, 2-hop, noising; the sender residual contribution is "
        "spliced at the receiver's resid_pre exactly as v1's PathPatcher did",
    )

    # -- the declarations --------------------------------------------------
    quantity = "grader.path_effect"
    requires: AccessMatrix = {Component.GRADER: Access.FORWARD | Access.MUTATE}
    substrates = NEURAL_SUBSTRATES
    phases = GRADER_STUDY_PHASES
    envelope = EnvelopeSpec(
        requires=frozenset({RegimeCondition.STATIONARY_GRADER, RegimeCondition.ABOVE_LOD}),
        measured_by=MEASURED_BY,
        on_violation="refuse",
    )
    #: The effect is a difference of reward projections and the splice is a linear map of the
    #: sender's residual, so a shared orthogonal change of basis leaves it unchanged.
    invariance = "repr.basis"
    invariance_relation = INVARIANT
    baselines = ("baseline.random_sender_head", "baseline.direct_patch")
    rung = 0
    #: An `IncrementalValidity` is required on every white-box reading and this
    #: instrument cannot produce one. The id is checkable and the prose is the argument.
    incremental_exemption = (
        "NO_SUBJECT_WITH_SIGNAL",
        "the path effect is per-pair and could be scored against the preference decision, but it is a "
        "difference of differences and routinely smaller than the grader's disagreement with itself, "
        "so the comparison needs both a trained grader and a measured limit of detection. Neither is "
        "here: the only offline grader in this build is `signals.from_tiny`, a randomly initialised "
        "two-layer LlamaForSequenceClassification. Measured on the 84-pair diagnostic set unfolded to "
        "168 items, it scores AUROC 0.5568 against 0.6892 for the TF-IDF baseline, so the white-box "
        "side of the comparison has no signal to contribute and the four numbers would describe the "
        "fixture rather than the instrument.",
    )

    def measure(self, ctx: Context) -> "Evidence":
        signal = ctx.signal
        n_layers = int(signal.meta.n_layers)
        readout = ctx.readout

        sender = ctx.regime.get("sender", (0, 0))
        receiver_layer = ctx.regime.get("receiver", n_layers - 1)
        s_layer, s_head = int(sender[0]), int(sender[1])
        sender_site = Site(s_layer, "head_out", s_head)
        receiver_site = Site(receiver_layer, "resid_pre")

        import torch

        # (d_model, d_head): the sender head's own columns of the attention output projection, read
        # through the SiteWeights protocol rather than by walking the architecture adapter.
        w_h = site_weights(signal.runtime).weight_at(sender_site)

        chosen, rejected = pair_sides(ctx.view)
        effects = np.zeros(len(chosen))
        for i, (chosen_item, rejected_item) in enumerate(zip(chosen, rejected)):
            reward_c = float(signal.score([chosen_item], readout).value.values[0])
            reward_r = float(signal.score([rejected_item], readout).value.values[0])
            original_diff = reward_c - reward_r

            src = capture_sites(signal, [rejected_item], (sender_site,), full_sequence=True)[
                sender_site
            ].to(torch.float32)
            tgt = capture_sites(signal, [chosen_item], (sender_site,), full_sequence=True)[
                sender_site
            ].to(torch.float32)
            src_contrib = src @ w_h.T  # (1, T_src, d_model)
            tgt_contrib = tgt @ w_h.T  # (1, T_tgt, d_model)
            min_len = min(src_contrib.shape[1], tgt_contrib.shape[1])
            delta = src_contrib[:, :min_len, :] - tgt_contrib[:, :min_len, :]

            patch = ResidualAddPatch(site=receiver_site, delta=delta)
            patched = float(
                run_patched_scores(signal, patch.compile(signal), [chosen_item], readout)[0]
            )
            patched_diff = patched - reward_r
            effects[i] = original_diff - patched_diff

        payload = {
            "sender": [s_layer, s_head],
            "receiver_layer": receiver_layer,
            "mean_path_effect": float(np.mean(effects)),
            "per_pair_path_effect": effects.tolist(),
            "max_abs_path_effect": float(np.max(np.abs(effects))) if len(effects) else 0.0,
            "n_pairs": len(chosen),
        }
        return ctx.emit(payload)


__all__ = ["PathEffect"]

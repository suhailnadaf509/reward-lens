"""Weight-space projection edit of the reward head as an Intervention.

Steering and ablation act on activations; an edit acts on the weights. The canonical case, v1's E17
and the weight-space analog science S12 uses, projects the reward head's own direction to remove its
component along a concept direction: ``w_r' = w_r - alpha (w_r . u) u`` for a unit direction ``u``.
The reward is the projection of the head-input hidden state onto ``w_r``; after the edit it is the
projection onto ``w_r'``, which has no component along ``u``, so the reward becomes insensitive to
that concept: moving the activation along ``u`` no longer changes the score. That is the whole point
of the edit, and it is exact linear algebra, provable to floating-point tolerance.

Realizing an edit is where this intervention differs mechanically from the activation ones. A steer or
an ablation is an activation mount, so it rides the frozen runtime's single hook path through
``patch.py``'s ``run_patched_scores``. An edit changes the readout vector the score is projected onto,
which is applied outside the model forward (in the signal's scoring code), so it has no activation
mount to install. Following the precedent ``patch.py`` set for exactly this kind of contract mismatch
with the frozen runtime, the edit compiles to a mount-free
:class:`~reward_lens.interventions.base.CompiledIntervention` that carries the edited readout vector in
its ``meta``, and :func:`run_edited_scores` is the thin sibling runner that projects the head input
onto that edited vector. When the M6 interventions contract lands and readout edits get a first-class
seam, this runner is the one place that changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from reward_lens.core.types import Site, content_hash, hash_bytes
from reward_lens.interventions.base import CompiledIntervention, MountHook
from reward_lens.interventions.steer import unit_direction

if TYPE_CHECKING:
    from reward_lens.signals.base import RewardSignal


@dataclass
class EditIntervention:
    """Project a concept direction out of the reward head weight; an ``Intervention``.

    ``direction`` is the array-like ``(d_model,)`` concept direction ``u`` to remove, in the space the
    reward head reads (the head-input hidden state), unit-normalized on use. ``readout`` names which
    readout's vector to edit (``"reward"`` by default). ``strength`` is the coefficient ``alpha`` in
    ``w_r' = w_r - alpha (w_r . u) u``: ``alpha = 1`` removes the component entirely (the E17 edit),
    ``alpha`` between 0 and 1 attenuates it, ``alpha > 1`` over-corrects past orthogonality.

    Because the edit lives on the readout vector rather than on an activation, a compiled edit carries
    no mounts; it carries the edited vector in ``meta`` and is applied by :func:`run_edited_scores`.
    """

    direction: Any
    readout: str = "reward"
    strength: float = 1.0
    id: str = "edit"

    def _direction_id(self) -> str:
        return hash_bytes(unit_direction(self.direction).tobytes(), "dir")

    def fingerprint(self) -> str:
        """A stable cache/provenance key over the readout, direction content, and strength.

        The fingerprint changes with which readout is edited, which direction is removed (the content
        hash, not just the shape), and how strongly, so an edited-head Evidence carries an identity
        distinct from the base head and from any other edit.
        """
        return content_hash(
            {
                "kind": "weight_edit",
                "readout": self.readout,
                "direction": self._direction_id(),
                "strength": float(self.strength),
            },
            "iv",
        )

    def edited_vector(self, signal: "RewardSignal", dtype: Any = np.float64) -> np.ndarray:
        """The edited readout vector ``w_r' = w_r - alpha (w_r . u) u`` for ``signal``.

        Reads the readout's current vector off the signal and removes the ``direction`` component in
        ``dtype`` (float64 by default, so the orthogonality is exact to double precision for the
        algebraic proofs; the runner requests float32 to match the head-projection dtype). Returns a
        numpy vector ``(d_model,)``.
        """
        w = np.asarray(signal.readout(self.readout).vector.detach().cpu().numpy(), dtype=dtype)
        u = unit_direction(self.direction).astype(dtype)
        return w - dtype(self.strength) * dtype(np.dot(w, u)) * u

    def compile(self, signal: "RewardSignal") -> CompiledIntervention:
        """Resolve into a mount-free compiled intervention carrying the edited readout vector.

        The edit is a weight-space operation with no activation mount, so ``mounts`` is empty; the
        realized ``w_r'`` (float32, ready for the head projection) rides in ``meta`` and is consumed by
        :func:`run_edited_scores`. The fingerprint is stamped exactly as the activation interventions
        stamp theirs, so an edited-head number carries its provenance the same way.
        """
        edited = self.edited_vector(signal, dtype=np.float32)
        return CompiledIntervention(
            fingerprint=self.fingerprint(),
            mounts={},
            meta={
                "kind": "weight_edit",
                "readout": self.readout,
                "strength": float(self.strength),
                "edited_vector": edited,
            },
        )


def run_edited_scores(
    signal: "RewardSignal",
    compiled: CompiledIntervention,
    view: Any,
    readout: str = "reward",
) -> np.ndarray:
    """Score ``view`` under a weight-space edit, returning the fp32 reward per item.

    The sibling of ``patch.py``'s ``run_patched_scores`` for the edit contract: an edit changes the
    readout vector, not an activation, so this runner forwards the batch (installing any activation
    mounts the compiled intervention also carries, which a pure edit has none of, through the same
    ``resolve_module`` path captures use) and then projects the head-input hidden state onto the edited
    vector in ``compiled.meta`` rather than the signal's original readout. The projection is
    ``signal.policy.head_project``, matching ``ClassifierRM.score`` exactly, so an edited score and a
    clean score are computed the same way apart from the substituted direction. Handles are always
    removed.
    """
    import torch

    from reward_lens.runtime.hooks import resolve_module

    edited = compiled.meta.get("edited_vector")
    if edited is None:
        raise ValueError("compiled intervention carries no 'edited_vector'; not a weight edit")
    weight = torch.as_tensor(np.asarray(edited), dtype=torch.float32)
    bias = float(signal.readout(readout).meta.get("bias", 0.0))

    runtime = signal.runtime
    model = runtime.model
    site_map = runtime.site_map

    tokenized = [signal.tokenize(it) for it in view]
    batch = runtime.collate(tokenized)

    handles: list[Any] = []

    def make_forward_hook(site: Site, hook: MountHook) -> Any:
        def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            new_hidden = hook(hidden, {"site": site})
            if isinstance(output, tuple):
                return (new_hidden,) + tuple(output[1:])
            return new_hidden

        return _hook

    try:
        for site, hook in compiled.mounts.items():
            module = resolve_module(model, site_map.resolve(site))
            handles.append(module.register_forward_hook(make_forward_hook(site, hook)))
        raw = runtime.forward(batch)
    finally:
        for handle in handles:
            handle.remove()

    head_input = raw.extra["head_input"]
    final_pos = raw.extra["final_pos"]
    idx = torch.arange(head_input.shape[0], device=head_input.device)
    pooled = head_input[idx, final_pos]
    values = signal.policy.head_project(pooled, weight, bias)
    return values.detach().to("cpu", dtype=torch.float32).numpy()


__all__ = ["EditIntervention", "run_edited_scores"]

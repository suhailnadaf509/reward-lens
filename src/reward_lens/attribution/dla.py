"""Canonical Direct Linear Attribution for reward models.

The residual stream is a sum of component outputs and the reward is a linear read of the final
residual, so the reward decomposes exactly into per-component signed contributions:

    r = w_r . h_embed + sum_l (w_r . attn^(l) + w_r . mlp^(l)) + b_r

and an attention layer's contribution splits further into per-head terms because ``o_proj`` is
linear: head h contributes ``head_out_h @ W_o[:, h*d_head:(h+1)*d_head].T``, whose reward
contribution is that vector's projection onto ``w_r``.

This module is the single canonical implementation of head-level reward attribution. There used to
be three of it, written separately: an einsum over a reshaped ``o_proj`` weight, an explicit
per-head slice plus matmul, and the inline ``o_proj`` slicing inside the path patcher, which
recomputes the same slice for a different purpose. All three agreed on the mathematics and differed
in dtype handling, device placement, detach discipline, and tensor layout, which is exactly the
operationalization-drift liability the kernel exists to remove. Everything head-level now routes
through :func:`head_reward_contributions`, so there is one place the head decomposition is defined
and one place it can be wrong.

The functions here are deliberately substrate-free: they take tensors and a reward direction, not a
``RewardModel`` or a ``RewardSignal``, so any caller that has activations and a reward direction
runs the same code on the same numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def project_onto_reward(activation: "torch.Tensor", w_r: "torch.Tensor") -> "torch.Tensor":
    """Project a component output onto the reward direction in fp32 (``act . w_r``).

    ``activation`` is ``(..., d_model)`` in whatever dtype the trunk produced; ``w_r`` is
    ``(d_model,)``. Both are upcast to fp32 before the contraction, matching the head-in-fp32 policy
    so a contribution computed here equals the corresponding slice of the fp32 reward. Returns
    ``(...)``.
    """
    import torch

    act = activation.to(torch.float32)
    weight = w_r.to(dtype=torch.float32, device=act.device)
    return act @ weight


def head_reward_contributions(
    head_out: "torch.Tensor",
    o_proj_weight: "torch.Tensor",
    w_r: "torch.Tensor",
    n_heads: int,
) -> np.ndarray:
    """Signed per-head reward contributions (the one canonical head decomposition).

    ``head_out`` is the per-head input to ``o_proj`` at the position being attributed, shaped
    ``(batch, n_heads, d_head)`` or ``(n_heads, d_head)`` for a single item. ``o_proj_weight`` is the
    output projection ``W_o`` of shape ``(d_model, n_heads * d_head)``, whose columns are grouped by
    head. ``w_r`` is the reward direction ``(d_model,)``.

    Head h contributes ``head_out_h @ W_h.T`` to the residual stream, where ``W_h`` is the
    ``(d_model, d_head)`` column block ``W_o[:, h*d_head:(h+1)*d_head]``; its reward contribution is
    that vector dotted with ``w_r``. Precomputing the per-head reward projector
    ``projector[h] = W_h.T @ w_r`` and contracting once is algebraically identical to slicing and
    matmul-ing each head, and this function is the sole definition of that quantity. Returns
    ``(batch, n_heads)`` (or ``(n_heads,)`` when a single item was passed), as a detached fp32 numpy
    array.
    """
    import torch

    acts = head_out.to(torch.float32)
    single = acts.ndim == 2
    if single:
        acts = acts.unsqueeze(0)
    batch, heads, d_head = acts.shape
    if heads != n_heads:  # defend against a mis-declared head count rather than silently misgroup
        raise ValueError(
            f"head_reward_contributions: head_out has {heads} heads but n_heads={n_heads}"
        )
    weight = o_proj_weight.to(dtype=torch.float32, device=acts.device)
    if weight.shape[1] != n_heads * d_head:
        raise ValueError(
            f"o_proj weight second dim {weight.shape[1]} != n_heads*d_head "
            f"{n_heads * d_head}; head grouping would be wrong"
        )
    direction = w_r.to(dtype=torch.float32, device=acts.device)
    weight_heads = weight.reshape(weight.shape[0], n_heads, d_head)  # (d_model, n_heads, d_head)
    projector = torch.einsum("d,dhk->hk", direction, weight_heads)  # (n_heads, d_head)
    contrib = torch.einsum("bhk,hk->bh", acts, projector)  # (batch, n_heads)
    out = contrib.detach().cpu().numpy()
    return out[0] if single else out


def component_reward_contributions(activation: "torch.Tensor", w_r: "torch.Tensor") -> np.ndarray:
    """Signed reward contribution of a component output, as a detached fp32 numpy array.

    ``activation`` is ``(batch, d_model)`` (or ``(d_model,)``); ``w_r`` is ``(d_model,)``. This is
    the component-granularity analogue of :func:`head_reward_contributions`, used for the embedding,
    per-layer attention, and per-layer MLP terms. Returns ``(batch,)`` (or a scalar array).
    """
    return project_onto_reward(activation, w_r).detach().cpu().numpy()


__all__ = [
    "project_onto_reward",
    "head_reward_contributions",
    "component_reward_contributions",
]

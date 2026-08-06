"""Canonical concept-direction extraction and reward alignment (E08).

A concept direction is the mean of per-example difference vectors ``h_pos - h_neg`` at a chosen
site, unit-normalized. This is the same mean-difference estimator the v1 ``ConceptExtractor`` used;
it is pulled out here as a substrate-free function of captured activations so both the v1 primitive
and the v3 ``ConceptDoseResponse`` Observable compute a concept direction the same way, and so the
dose-response probe and the reward-alignment read share one definition of "the concept direction".

The reward alignment of a concept is the cosine between its direction and the reward direction. That
cosine is a raw-coordinate quantity: it depends on the residual-stream basis and carries no shared
frame, so it is honest as an internal, single-signal geometry and must not be compared across signals
without a frame (gate 2). The Observable that consumes these functions declares that gauge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def concept_direction(pos: "torch.Tensor", neg: "torch.Tensor") -> np.ndarray:
    """Unit-normalized mean-difference concept direction from paired activations.

    ``pos`` and ``neg`` are ``(n, d_model)`` activations at a site for the positive and negative side
    of ``n`` concept pairs. The direction is ``mean(pos - neg)`` over the pairs, normalized to unit
    length; a degenerate (near-zero) mean returns the raw mean rather than dividing by a vanishing
    norm. Returns an fp32 numpy vector ``(d_model,)``.
    """
    import torch

    delta = (pos.to(torch.float32) - neg.to(torch.float32)).mean(dim=0)
    norm = torch.linalg.vector_norm(delta)
    if float(norm) > 1e-12:
        delta = delta / norm
    return delta.detach().cpu().numpy()


def reward_alignment(direction: np.ndarray, w_r: "torch.Tensor") -> float:
    """Cosine between a concept direction and the reward direction (raw coordinates).

    Both vectors are normalized before the dot product, so the result is a cosine in ``[-1, 1]``. A
    positive value means the concept pushes reward up; a negative value means it pushes reward down.
    This is a RAW_ONLY quantity (it depends on the residual-stream basis) and is only meaningful
    within a single signal.
    """
    import torch

    d = np.asarray(direction, dtype=np.float64)
    w = w_r.to(torch.float32).detach().cpu().numpy().astype(np.float64)
    dn = np.linalg.norm(d)
    wn = np.linalg.norm(w)
    if dn < 1e-12 or wn < 1e-12:
        return 0.0
    return float(np.dot(d, w) / (dn * wn))


def dose_response_slope(doses: np.ndarray, rewards: np.ndarray) -> float:
    """Ordinary-least-squares slope of reward against concept dose.

    ``doses`` are the intervention strengths applied along a concept direction and ``rewards`` the
    resulting scalar rewards. The slope is ``d reward / d dose`` from a least-squares line, which is
    the dose-response summary the E08 study reports. A constant dose vector (no variation) returns
    ``0.0`` rather than dividing by a zero variance.
    """
    x = np.asarray(doses, dtype=np.float64)
    y = np.asarray(rewards, dtype=np.float64)
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(xc, y - y.mean()) / denom)


__all__ = ["concept_direction", "reward_alignment", "dose_response_slope"]

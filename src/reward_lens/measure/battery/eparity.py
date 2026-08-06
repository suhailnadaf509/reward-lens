"""E-parity recompute wiring from v1 activation caches.

The trust anchor for v3 is that it reproduces v1's verified-clean headline numbers from the cached
activations before it is trusted to produce new ones. The v1 campaign left its final-token activations
on disk (``read_v1_cache`` loads one shard into a :class:`~reward_lens.runtime.store.V1Cache`), and the
targets are recorded in ``fixtures/e_parity/golden.json``. This module is the wiring that turns a cache
plus a reward direction into the recompute, so the path ``cache + w_r + Observable = golden number`` is
a single importable function rather than a description.

The unavoidable gate is stated plainly here and enforced by the caller. The reward direction ``w_r`` is
the 8B model's score-head weight. It is a model weight, not a cached activation, and it is not uniquely
recoverable from the cache (360 samples in 4096 dimensions is underdetermined; a ridge fit overfits to
a false direction). Reproducing the real 8B E02 / E04 / E15 numbers therefore needs the 8B model's
score head, which is GPU/download-gated on this machine. What the cache alone supports without ``w_r``
is the reward margin, because the per-sample scalar reward was cached directly: :func:`reward_margins`
needs no ``w_r``. Everything that needs ``w_r`` across intermediate layers is gated and must never be
fabricated; :func:`population_lens` takes ``w_r`` as an explicit argument so a caller cannot forget it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

    from reward_lens.runtime.store import V1Cache


def population_lens(cache: "V1Cache", w_r: "torch.Tensor") -> dict[int, np.ndarray]:
    """Project every cached final-token residual onto ``w_r``, per layer (the reward lens).

    Returns ``{layer: (N,) projection}`` for each layer present in the cache's residual streams. This
    is the population reward lens: the reward the model would assign at each layer for each cached
    sample. It requires ``w_r`` (the 8B score head), which is the gated input; the cache supplies the
    residuals but not the direction. On a small model where ``w_r`` is available this reproduces the v1
    lens exactly, which is what the E-parity test proves before the number is trusted.
    """
    import torch

    weight = w_r.to(torch.float32)
    out: dict[int, np.ndarray] = {}
    for layer, resid in cache.residual_streams.items():
        out[int(layer)] = (resid.to(torch.float32) @ weight).cpu().numpy()
    return out


def reward_margins(
    cache: "V1Cache", chosen_idx: np.ndarray, rejected_idx: np.ndarray
) -> np.ndarray:
    """Per-pair reward margins from the cached scalar rewards, with no ``w_r`` needed.

    The v1 cache stored the per-sample scalar reward directly, so a pair's reward margin is just
    ``rewards[chosen] - rewards[rejected]``. This is the honest, ``w_r``-free anchor: the final
    differential-lens value equals this margin by the definition of the head, and it is computable from
    the cache alone. ``chosen_idx`` and ``rejected_idx`` index the cached population.
    """
    if cache.rewards is None:
        raise ValueError("cache has no rewards; cannot compute reward margins")
    rewards = cache.rewards.to("cpu").numpy().astype(np.float64)
    return rewards[np.asarray(chosen_idx)] - rewards[np.asarray(rejected_idx)]


def w_r_available_in_cache(cache: "V1Cache") -> bool:
    """Whether the reward direction is present in the cache (it is not; recompute is w_r-gated).

    The cache holds residual streams, attention/MLP outputs, per-head outputs, the scalar rewards, and
    the final-token positions. It does not hold the score-head weight. This helper exists so a test can
    assert the gate honestly: the reward direction is absent, so an intermediate-layer recompute needs
    the 8B model's score head (GPU/download-gated) and must not be faked.
    """
    return getattr(cache, "reward_direction", None) is not None


__all__ = ["population_lens", "reward_margins", "w_r_available_in_cache"]

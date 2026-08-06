"""Additive steering along a direction as an Intervention.

Steering is the simplest causal probe of a reward: push the residual stream a fixed amount along a
chosen direction and read how the scalar reward responds. In v1 this was a bespoke hook that added a
vector during scoring and could not compose with capture or carry its own identity. Here a steer is
an :class:`~reward_lens.interventions.base.Intervention`: it compiles against a signal into a single
site-addressed mount hook that adds ``strength * unit(direction)`` to the activation, and it carries a
fingerprint derived from the site, the direction content, and the strength into the Evidence subject,
so a steered-run number can never be mistaken for a clean-run number (the same guarantee ``patch.py``
gives).

The direction is unit-normalized before it is used, so ``strength`` is the actual displacement in
residual-norm units and an alpha grid is a clean dose axis: this is exactly what the standard
dose-response Observable consumes. Because activation steering rides the frozen runtime's single mount
path, a compiled steer runs through the same ``run_patched_scores`` helper ``patch.py`` exposes; this
module is the intervention object and its hook, not a runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from reward_lens.core.types import Site, content_hash, hash_bytes
from reward_lens.interventions.base import CompiledIntervention, MountHook

if TYPE_CHECKING:
    import torch

    from reward_lens.signals.base import RewardSignal


def unit_direction(direction: Any) -> np.ndarray:
    """Unit-normalize a direction to an fp32 vector; a near-zero direction is an error.

    ``direction`` is any array-like ``(d_model,)``. It is flattened, cast to fp32, and divided by its
    Euclidean norm. A direction whose norm is below ``1e-12`` has no orientation to steer along, so
    this raises rather than dividing by a vanishing norm and returning a garbage direction (the E09
    NaN lesson, applied at construction time instead of after a table of NaNs).
    """
    d = np.asarray(direction, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(d))
    if norm < 1e-12:
        raise ValueError(
            "steering direction is (near) zero; there is no orientation to steer along"
        )
    return d / norm


@dataclass
class SteeringIntervention:
    """Add ``strength * unit(direction)`` to a site's activation; an ``Intervention``.

    ``direction`` is the array-like ``(d_model,)`` steering direction (unit-normalized on use, so its
    magnitude is irrelevant and only its orientation matters). ``site`` names where to add it, in the
    residual stream or a sublayer output. ``strength`` is the signed displacement in unit-direction
    units, the dose. ``positions``, if given, restricts the addition to those (padded) sequence
    columns; the default ``None`` steers every position, which is what a final-token readout needs and
    what the dose-response Observable uses.
    """

    direction: Any
    site: Site
    strength: float
    positions: tuple[int, ...] | None = None
    id: str = "steer"

    def _direction_id(self) -> str:
        """A content hash of the unit direction, so the fingerprint tracks direction content."""
        return hash_bytes(unit_direction(self.direction).tobytes(), "dir")

    def fingerprint(self) -> str:
        """A stable cache/provenance key derived from site, direction content, and strength.

        Two steers differ in fingerprint whenever they differ in where they push, which way they push
        (the direction content hash, not just its shape), or how hard they push. A strength of zero is
        a distinct fingerprint from any nonzero strength even though it is a numerical no-op, because a
        provenance record should say a steer with strength zero was applied, not that nothing was.
        """
        return content_hash(
            {
                "kind": "steer",
                "site": str(self.site),
                "direction": self._direction_id(),
                "strength": float(self.strength),
                "positions": list(self.positions) if self.positions is not None else None,
            },
            "iv",
        )

    def _hook(self) -> MountHook:
        unit = unit_direction(self.direction)
        strength = float(self.strength)
        positions = self.positions

        def apply(hidden: "torch.Tensor", _ctx: dict) -> "torch.Tensor":
            import torch

            # Strength zero is a bit-exact no-op: return the activation untouched rather than adding a
            # zero tensor, so a steer at strength zero reproduces the clean forward to the bit.
            if strength == 0.0:
                return hidden
            delta = strength * torch.as_tensor(unit, device=hidden.device, dtype=hidden.dtype)
            if positions is None:
                return hidden + delta
            out = hidden.clone()
            length = out.shape[1]
            for p in positions:
                if -length <= p < length:
                    out[:, p, :] = out[:, p, :] + delta
            return out

        return apply

    def compile(self, signal: "RewardSignal") -> CompiledIntervention:
        """Resolve into a concrete mount hook at ``site`` (the ``Intervention`` protocol).

        The site is architecture-resolved by the runner's SiteMap at mount time, exactly as in
        ``patch.py``, so compilation only needs to bind the hook and stamp the fingerprint.
        """
        del signal
        return CompiledIntervention(
            fingerprint=self.fingerprint(),
            mounts={self.site: self._hook()},
            meta={"kind": "steer", "site": str(self.site), "strength": float(self.strength)},
        )


__all__ = ["SteeringIntervention", "unit_direction"]

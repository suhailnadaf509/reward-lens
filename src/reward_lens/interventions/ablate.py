"""Directional, mean, and head ablation as Interventions.

Ablation asks the causal question steering does not: not "what happens if I add this direction" but
"what happens if I remove it". Three removals live here, all as
:class:`~reward_lens.interventions.base.Intervention` objects that mount through the frozen runtime's
single hook path and carry their fingerprint into provenance, exactly as ``patch.py`` and ``steer.py``
do.

:class:`AblationIntervention` with ``mode="directional"`` projects a direction out of the residual:
``h -> h - (h.u) u`` for a unit direction ``u``, so the activation keeps no component along ``u``.
With ``mode="mean"`` it does not zero that component but replaces it with a supplied dataset-mean
projection ``m``: ``h -> h - (h.u) u + m u``, which sets the along-direction coordinate to the value
the direction takes on average over a reference set (the counterfactual "what if this example were
average on this axis", not "what if this axis did not exist"). With ``mode="head"`` it zeros one
attention head's contribution by clearing that head's slice of the ``o_proj`` input, the same
per-head surface ``patch.py``'s :class:`~reward_lens.interventions.patch.HeadPatch` edits; a
head-ablation therefore composes cleanly with the head machinery instead of half-reimplementing it.

Directional and mean ablation are exact linear-algebraic operations on the activation, so their
correctness is provable to floating-point tolerance on synthetic tensors: after a directional
ablation the residual's dot with ``u`` is zero, the operation is idempotent, and a vector orthogonal
to ``u`` is left untouched. Those are the invariants the test file pins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from reward_lens.core.types import Site, content_hash, hash_bytes
from reward_lens.interventions.base import CompiledIntervention, MountHook
from reward_lens.interventions.steer import unit_direction

if TYPE_CHECKING:
    import torch

    from reward_lens.signals.base import RewardSignal


@dataclass
class AblationIntervention:
    """Remove a direction (or a head) from an activation; an ``Intervention``.

    ``site`` names where the ablation acts. ``mode`` selects the removal:

    - ``"directional"`` projects ``direction`` out of the residual (``h -> h - (h.u) u``);
      ``direction`` is required and ``mean_projection`` is ignored.
    - ``"mean"`` replaces the along-direction coordinate with ``mean_projection`` instead of zeroing
      it (``h -> h - (h.u) u + mean_projection * u``); ``direction`` is required. The supplied scalar
      is the direction's average projection over a reference dataset, computed upstream.
    - ``"head"`` zeros the attention head named by ``site.head`` at ``Site(layer, "head_out", head)``;
      ``n_heads`` is required so the ``o_proj`` input can be split into per-head slices, and
      ``direction`` is ignored.

    The direction, where used, is unit-normalized on use, so only its orientation matters.
    """

    site: Site
    direction: Any = None
    mode: str = "directional"
    mean_projection: float = 0.0
    n_heads: int | None = None
    id: str = "ablate"

    _MODES = ("directional", "mean", "head")

    def __post_init__(self) -> None:
        if self.mode not in self._MODES:
            raise ValueError(f"unknown ablation mode {self.mode!r}; expected one of {self._MODES}")

    def fingerprint(self) -> str:
        """A stable cache/provenance key over the mode, site, and the mode's payload.

        A directional and a mean ablation of the same direction at the same site have different
        fingerprints because the mode and the substituted mean differ; a head ablation keys on the
        head count as well, so the same ``site.head`` on two different backbones does not collide.
        """
        payload: dict[str, Any] = {"kind": "ablate", "mode": self.mode, "site": str(self.site)}
        if self.mode in ("directional", "mean"):
            payload["direction"] = hash_bytes(unit_direction(self.direction).tobytes(), "dir")
        if self.mode == "mean":
            payload["mean_projection"] = float(self.mean_projection)
        if self.mode == "head":
            payload["n_heads"] = int(self.n_heads or 0)
            payload["head"] = self.site.head
        return content_hash(payload, "iv")

    def _projection_hook(self) -> MountHook:
        unit = unit_direction(self.direction)
        substitute = float(self.mean_projection) if self.mode == "mean" else 0.0

        def apply(hidden: "torch.Tensor", _ctx: dict) -> "torch.Tensor":
            import torch

            vec = torch.as_tensor(unit, device=hidden.device, dtype=hidden.dtype)
            # coord = h . u along the last axis; keepdim so it broadcasts back over d_model.
            coord = (hidden * vec).sum(dim=-1, keepdim=True)
            # Replace the along-u component with `substitute` (0 for directional ablation): the new
            # along-u coordinate is exactly `substitute`, every orthogonal coordinate is untouched.
            return hidden + (substitute - coord) * vec

        return apply

    def _head_hook(self) -> MountHook:
        head = self.site.head or 0
        n_heads = int(self.n_heads or 0)

        def apply(x: "torch.Tensor", _ctx: dict) -> "torch.Tensor":
            # x is the o_proj input (B, T, n_heads * d_head); zero one head's slice, keep the rest,
            # which is HeadPatch's mechanic with a zero source (E15's head surface).
            b, t, feat = x.shape
            d_head = feat // n_heads
            view = x.view(b, t, n_heads, d_head).clone()
            view[:, :, head, :] = 0.0
            return view.view(b, t, feat)

        return apply

    def compile(self, signal: "RewardSignal") -> CompiledIntervention:
        """Resolve into a concrete mount hook at ``site`` (the ``Intervention`` protocol).

        Directional and mean ablation bind a projection hook; head ablation binds a per-head zeroing
        hook that the runner mounts as an ``o_proj`` pre-hook (the ``head_out`` surface). The site is
        architecture-resolved at mount time, as in ``patch.py``, so compilation only binds the hook
        and stamps the fingerprint.
        """
        del signal
        if self.mode == "head":
            if self.n_heads is None:
                raise ValueError("head ablation requires n_heads to slice the o_proj input")
            hook = self._head_hook()
        else:
            if self.direction is None:
                raise ValueError(f"{self.mode} ablation requires a direction to project out")
            hook = self._projection_hook()
        return CompiledIntervention(
            fingerprint=self.fingerprint(),
            mounts={self.site: hook},
            meta={"kind": "ablate", "mode": self.mode, "site": str(self.site)},
        )


__all__ = ["AblationIntervention"]

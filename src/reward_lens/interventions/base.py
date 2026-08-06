"""The intervention protocol and the causal algebra.

An Intervention is anything that modifies a forward pass: a patch, a steer, an ablation, an
erasure, a head edit. The design's key move is that interventions and captures share one mounting
path in the runtime, so any Observable can run under any Intervention without either knowing about
the other. In v1, patching and caching were separate code paths, which is why an Observable could
not be measured under an arbitrary intervention; that is designed out here.

Interventions compose: ``signal.with_interventions(erase_verbosity, steer(conf, +1.5))`` returns
a wrapped signal any Observable accepts unchanged. Every intervened Evidence records the
intervention fingerprints in its SubjectRef, so an erased-model card can never masquerade as a
base-model card. This is a frozen interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence, runtime_checkable

from reward_lens.core.types import Site

if TYPE_CHECKING:
    import torch

    from reward_lens.signals.base import RewardSignal

# A mount hook takes the activation tensor at a site (and the forward context) and returns the
# possibly-modified tensor. Both captures and interventions are expressed as mounts, which is the
# whole point of the single hook path.
MountHook = Callable[["torch.Tensor", dict], "torch.Tensor"]


@dataclass
class CompiledIntervention:
    """An Intervention resolved against a specific signal.

    Compilation resolves adapter-specific sites and shapes once, up front, producing the concrete
    mount hooks the runtime installs. ``fingerprint`` is the cache-key component that keeps a
    patched-run cache from ever aliasing a clean-run cache (a latent v1 hazard); it is carried
    into the ActivationStore key and the intervened Evidence's SubjectRef.
    """

    fingerprint: str
    mounts: dict[Site, MountHook] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Intervention(Protocol):
    """A modification of a forward pass.

    ``compile`` resolves the intervention against a signal (turning "steer the verbosity
    direction by +1.5 at the final residual" into concrete site-addressed mount hooks);
    ``fingerprint`` produces the stable cache-key and provenance component. Implementations live
    in ``interventions/`` (patch, steer, ablate, erase, edit, certify) and mount through the
    runtime's single hook path.
    """

    id: str

    def compile(self, signal: "RewardSignal") -> CompiledIntervention: ...

    def fingerprint(self) -> str: ...


def compose(interventions: Sequence[Intervention]) -> "ComposedIntervention":
    """Compose several interventions into one that mounts them together in order."""
    return ComposedIntervention(tuple(interventions))


@dataclass
class ComposedIntervention:
    """Several interventions applied together (the causal algebra's product).

    Compiles each member against the signal and merges their mounts; when two interventions
    target the same site, their hooks chain in declaration order. The composite fingerprint is
    derived from the members' fingerprints, so a composed intervention has a stable identity.
    """

    members: tuple[Intervention, ...]

    @property
    def id(self) -> str:
        return "compose(" + ",".join(m.id for m in self.members) + ")"

    def fingerprint(self) -> str:
        from reward_lens.core.types import content_hash

        return content_hash([m.fingerprint() for m in self.members], "iv")

    def compile(self, signal: "RewardSignal") -> CompiledIntervention:
        merged: dict[Site, list[MountHook]] = {}
        meta: dict[str, Any] = {}
        for member in self.members:
            compiled = member.compile(signal)
            meta[member.id] = compiled.meta
            for site, hook in compiled.mounts.items():
                merged.setdefault(site, []).append(hook)

        def _chain(hooks: list[MountHook]) -> MountHook:
            def _apply(tensor: "torch.Tensor", ctx: dict) -> "torch.Tensor":
                for hook in hooks:
                    tensor = hook(tensor, ctx)
                return tensor

            return _apply

        return CompiledIntervention(
            fingerprint=self.fingerprint(),
            mounts={site: _chain(hooks) for site, hooks in merged.items()},
            meta=meta,
        )


__all__ = [
    "MountHook",
    "CompiledIntervention",
    "Intervention",
    "ComposedIntervention",
    "compose",
]

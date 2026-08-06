"""Record what an ablation removed, and put it back (C6, and the mechanics C5 and C7 share).

Interpretability's standard causal claim is a knockout with no rescue: remove X, observe a change,
conclude X mattered. Genetics stopped accepting that decades ago, because a knockout confounds the
loss of X with every downstream consequence of the perturbation itself, and the fix is to restore X
and check that the phenotype comes back. It costs one extra forward pass over the ablation you have
already run, and as far as the field scan behind this catalogue established, essentially nobody in
interpretability does it.

**What is here and what is deliberately not.** `AblationIntervention` already removes a direction and
it is not reimplemented. Two small interventions are added beside it:

`RecordRemoved` is a pure observer. It reads the coordinate `h·u` at a site, stores it, and returns
the activation untouched. Composed *before* an ablation at the same site it captures exactly what
the ablation is about to remove, because `ComposedIntervention` chains hooks at one site in
declaration order.

`Reinject` adds a recorded coordinate back, at the same site or at a later one, along the direction
it came from or along a substitute. That substitute is the control, and it is norm-matched by
construction rather than by arithmetic: re-injecting `c·v` for any unit `v` has the same magnitude
as re-injecting `c·u`, so a random re-injection differs from the real one only in direction.

**Same site or a different one, and why the answer is both.** Restoring at the site the ablation
acted on is close to a no-op and is the sanity check: it should recover the behaviour almost
exactly, and if it does not, something in the apparatus is wrong rather than something in the model.
Restoring at a *later* site is the informative version, because it asks whether the direction is
carrying the behaviour or merely correlated with a pathway that is. Both are supported and the
`RescueSpec` records which was run, because reporting the first as though it were the second is the
way this control becomes decorative.

**What it cannot do.** The coordinate is recorded and replayed within one forward pass, so this
rescues an ablation applied during that pass and cannot rescue a weight edit. And re-injecting at a
later layer puts the coordinate into a residual stream that the intervening layers have already
written to under the ablated condition, so a rescue fraction below 1 confounds "the direction was
not sufficient" with "the intervening computation had already gone somewhere else". That is the
honest limit of a within-pass rescue and it is why the number is a fraction rather than a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from reward_lens.core.types import Site, content_hash, hash_bytes
from reward_lens.interventions.base import CompiledIntervention, MountHook
from reward_lens.interventions.steer import unit_direction

if TYPE_CHECKING:
    import torch

    from reward_lens.signals.base import RewardSignal


class RescueError(RuntimeError):
    """A rescue whose recorded coordinate does not match the site it is being replayed into.

    Raised rather than returned. A shape mismatch between the recording site and the re-injection
    site is a programming error in how the pass was assembled, not an anticipated condition of the
    subject, and turning it into a `Refusal` would hide a wiring bug behind a remedy string.
    """


@dataclass
class Mounted:
    """One site's hook behind the shape `runtime/hooks.py` actually mounts.

    **There are two intervention contracts in this library and they do not meet.**
    `interventions/base.py` defines `Intervention.compile(signal) -> CompiledIntervention`, whose
    `mounts` is a `{Site: hook}` mapping, and that is what the causal algebra composes over.
    `runtime/hooks.py::mounted_interventions` mounts objects exposing `site: Site` and
    `apply(hidden) -> hidden`, one site each, and its own docstring says
    "``interventions.base.CompiledIntervention`` will satisfy it in M6", which it does not yet. So a
    `ComposedIntervention` has no `.site` and cannot be mounted at all: it raises
    `AttributeError: 'CompiledIntervention' object has no attribute 'site'` inside the runtime.

    This adapter is the seam until that is reconciled. It is deliberately tiny and it does not
    change either contract: it takes the `{Site: hook}` a compiled intervention already produces and
    returns one mountable object per site, in a stable order, so an arm built out of the algebra can
    be handed to the shipped runtime. The reconciliation itself is a request in this package's
    report rather than an edit, because both files are outside its path set.
    """

    site: Site
    hook: MountHook
    label: str = ""

    def apply(self, hidden: "torch.Tensor") -> "torch.Tensor":
        return self.hook(hidden, {})


def mountable(intervention: Any, *, signal: Any = None) -> list[Mounted]:
    """Compile an `Intervention` and return one mountable object per site it touches.

    Ordering is by the site's layer then its point, so a recorder at layer 3 and a re-injection at
    layer 7 mount in the order the forward pass reaches them. Within one site the compiled hooks are
    already chained by `ComposedIntervention` in declaration order, so a recorder composed before an
    ablation still sees the activation the ablation is about to change.
    """
    compiled = intervention.compile(signal)
    ordered = sorted(compiled.mounts.items(), key=lambda kv: (kv[0].layer, str(kv[0].point)))
    return [
        Mounted(site=site, hook=hook, label=str(compiled.meta.get("kind", "")))
        for site, hook in ordered
    ]


@dataclass
class RemovedCoordinate:
    """The scalar coordinate an ablation took out, per batch row and position.

    A mutable box shared by the recorder and the re-injector, because they are two hooks in one
    forward pass and the value has to travel between them. One box per rescue: reusing a box across
    two passes replays the first pass's coordinates into the second, which is a silent wrong answer
    rather than a crash, so `clear` is called by the recorder on every entry.
    """

    #: `(batch, positions, 1)`, in the activation's own dtype and device. None before the recorder
    #: has run, which is what `Reinject` refuses on.
    value: Any = None
    site: Site | None = None
    n_calls: int = 0
    label: str = ""

    def clear(self) -> None:
        self.value = None
        self.n_calls = 0

    @property
    def recorded(self) -> bool:
        return self.value is not None

    def magnitude(self) -> float:
        """The mean absolute coordinate, for the record. NaN before anything was recorded."""
        if self.value is None:
            return float("nan")
        return float(self.value.abs().mean().item())


@dataclass
class RecordRemoved:
    """Read `h·u` at a site into a box and pass the activation through unchanged.

    A pure observer with an `Intervention`'s shape, so it mounts through the same hook path as
    everything else and composes with an ablation at the same site rather than needing a second
    capture pass. Its fingerprint is distinct from the ablation's, so a recorded run and a plain
    ablated run do not share a cache key even though they produce identical activations.
    """

    site: Site
    direction: Any
    into: RemovedCoordinate = field(default_factory=RemovedCoordinate)
    id: str = "record_removed"

    def fingerprint(self) -> str:
        return content_hash(
            {
                "kind": "record_removed",
                "site": str(self.site),
                "direction": hash_bytes(unit_direction(self.direction).tobytes(), "dir"),
            },
            "iv",
        )

    def _hook(self) -> MountHook:
        unit = unit_direction(self.direction)
        box = self.into
        site = self.site

        def apply(hidden: "torch.Tensor", _ctx: dict) -> "torch.Tensor":
            import torch

            vec = torch.as_tensor(unit, device=hidden.device, dtype=hidden.dtype)
            coord = (hidden * vec).sum(dim=-1, keepdim=True)
            box.value = coord.detach().clone()
            box.site = site
            box.n_calls += 1
            return hidden

        return apply

    def compile(self, signal: "RewardSignal | None" = None) -> CompiledIntervention:
        del signal  # reading a coordinate is signal-independent; the site is resolved at mount
        self.into.clear()
        return CompiledIntervention(
            fingerprint=self.fingerprint(),
            mounts={self.site: self._hook()},
            meta={"kind": "record_removed", "site": str(self.site)},
        )


@dataclass
class Reinject:
    """Add a recorded coordinate back into the residual, along `u` or along a substitute.

    ``direction`` is the direction the coordinate is replayed along. Passing the direction it was
    recorded from is the rescue; passing any other unit vector is the norm-matched control, and it
    is matched by construction because the magnitude comes from the recorded coordinate rather than
    from the direction.

    ``scale`` exists for the dose sweep rather than for tuning. Re-injecting a fraction of what was
    removed turns a rescue into a dose-response curve on the same axis C4 sweeps, which is how a
    partial rescue becomes a number instead of an anecdote.
    """

    site: Site
    direction: Any
    source: RemovedCoordinate
    scale: float = 1.0
    id: str = "reinject"

    def fingerprint(self) -> str:
        return content_hash(
            {
                "kind": "reinject",
                "site": str(self.site),
                "direction": hash_bytes(unit_direction(self.direction).tobytes(), "dir"),
                "scale": float(self.scale),
                "from": str(self.source.site) if self.source.site else None,
            },
            "iv",
        )

    def _hook(self) -> MountHook:
        unit = unit_direction(self.direction)
        box = self.source
        scale = float(self.scale)
        site = self.site

        def apply(hidden: "torch.Tensor", _ctx: dict) -> "torch.Tensor":
            import torch

            if not box.recorded:
                raise RescueError(
                    f"nothing was recorded before the re-injection at {site}. A `RecordRemoved` "
                    f"has to be mounted at a site the forward pass reaches *before* this one, and "
                    f"the two have to be composed into one intervention so they run in the same "
                    f"pass. Re-injecting a coordinate captured in an earlier pass would replay the "
                    f"wrong items."
                )
            coord = box.value
            if coord.shape[:-1] != hidden.shape[:-1]:
                raise RescueError(
                    f"the coordinate recorded at {box.site} has shape {tuple(coord.shape)} and the "
                    f"activation at {site} has shape {tuple(hidden.shape)}. The two sites see "
                    f"different batch or position axes, so there is no row-for-row correspondence "
                    f"to replay along."
                )
            vec = torch.as_tensor(unit, device=hidden.device, dtype=hidden.dtype)
            return hidden + scale * coord.to(hidden.dtype).to(hidden.device) * vec

        return apply

    def compile(self, signal: "RewardSignal | None" = None) -> CompiledIntervention:
        del signal
        return CompiledIntervention(
            fingerprint=self.fingerprint(),
            mounts={self.site: self._hook()},
            meta={
                "kind": "reinject",
                "site": str(self.site),
                "scale": float(self.scale),
                "from": str(self.source.site) if self.source.site else None,
            },
        )


@dataclass(frozen=True)
class RescueSpec:
    """Which rescue was run: the ablated site, the restoring site, and along what.

    Carried onto the reading because "restored at the same site" and "restored three layers later"
    are different experiments and the first is close to a no-op. A rescue fraction reported without
    saying which one it was is the way this control stops being one.
    """

    ablate_at: Site
    restore_at: Site
    direction_id: str
    substitute_id: str | None = None
    scale: float = 1.0

    @property
    def is_same_site(self) -> bool:
        """The sanity check rather than the informative version."""
        return self.ablate_at == self.restore_at

    @property
    def is_control(self) -> bool:
        return self.substitute_id is not None

    def render(self) -> str:
        where = (
            f"restored at the ablated site {self.ablate_at}"
            if self.is_same_site
            else f"ablated at {self.ablate_at}, restored at {self.restore_at}"
        )
        along = (
            f"along the substitute {self.substitute_id}"
            if self.substitute_id
            else f"along {self.direction_id}"
        )
        scale = "" if self.scale == 1.0 else f" at {self.scale:g}x"
        return f"{where}, {along}{scale}"


def knockout_and_rescue(
    *,
    ablate_at: Site,
    direction: Any,
    restore_at: Site | None = None,
    substitute: Any = None,
    scale: float = 1.0,
    direction_id: str = "direction",
    substitute_id: str | None = None,
) -> tuple[list[Mounted], list[Mounted], RescueSpec]:
    """Build the ablated arm and the rescued arm, each as a list of mountable single-site objects.

    Returns `(ablated, rescued, spec)`. The ablated arm is a plain `AblationIntervention` with a
    recorder in front of it; the rescued arm is that same pair plus a `Reinject`. The recorder is
    shared between them by construction: the rescued arm records and replays inside one forward
    pass, which is the only way the coordinate replayed is the one that was removed.

    ``substitute`` is the norm-matched control direction. Absent, the coordinate is replayed along
    the direction it came from, which is the rescue itself.

    Each arm is a list because that is what the shipped runtime mounts. Pass it straight to
    `subject.with_interventions(*arm)` or to `policy.selection.behaviour_under(..., intervention=arm)`.
    """
    from reward_lens.interventions.ablate import AblationIntervention
    from reward_lens.interventions.base import compose

    restore = restore_at if restore_at is not None else ablate_at
    box = RemovedCoordinate(label=direction_id)
    recorder = RecordRemoved(site=ablate_at, direction=direction, into=box)
    ablation = AblationIntervention(site=ablate_at, direction=direction, mode="directional")
    along = substitute if substitute is not None else direction
    reinject = Reinject(site=restore, direction=along, source=box, scale=scale)

    # Returned as mountable single-site objects rather than as `ComposedIntervention`s, because the
    # shipped runtime mounts `site` + `apply(hidden)` and cannot mount a composed one. See
    # `Mounted` for why the two contracts differ and what the fix is.
    ablated = mountable(compose([recorder, ablation]))
    rescued = mountable(compose([recorder, ablation, reinject]))
    spec = RescueSpec(
        ablate_at=ablate_at,
        restore_at=restore,
        direction_id=direction_id,
        substitute_id=substitute_id if substitute is not None else None,
        scale=float(scale),
    )
    return ablated, rescued, spec


def norm_matched_random(direction: Any, *, seed: int = 0) -> np.ndarray:
    """A random unit direction of the same dimension, for the rescue control.

    Unit rather than norm-matched to the input, because the magnitude replayed comes from the
    recorded coordinate and not from this vector: `c·v` and `c·u` have the same norm for any two
    unit vectors. Making that structural rather than arithmetic is what stops the control being
    accidentally weaker than the thing it controls for.
    """
    u = unit_direction(direction)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(u.shape[0])
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else u


__all__ = [
    "Mounted",
    "RecordRemoved",
    "Reinject",
    "RemovedCoordinate",
    "RescueError",
    "RescueSpec",
    "knockout_and_rescue",
    "mountable",
    "norm_matched_random",
]

"""Hook site addressing and mounting.

This is the port of v1's hook mechanics (forward hooks per layer for resid/attn/mlp; o_proj
pre-hooks for per-head capture; handles always removed in ``finally``) with the one structural
change that matters: **captures and interventions share a single mounting path**. In v1,
patching and caching were separate code paths, which is why an Observable could not be measured
under an arbitrary Intervention. Here both a capture and an intervention are just a hook on a module
resolved from the ``SiteMap``, installed by the same machinery and torn down in the same ``finally``.

A capture reads the activation at a ``Site`` (optionally gathered at resolved positions to keep it
small, v1-style); an intervention replaces the activation at a ``Site``. The point vocabulary is
the one in ``core.types.Site``: ``resid_pre``/``resid_post`` on the decoder block, ``attn_out`` on
the attention sublayer, ``mlp_out`` on the MLP, ``head_out`` per attention head via the o_proj
input, and ``embed`` on the token embedding.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence, cast

from reward_lens.core.types import Site

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

    from reward_lens.runtime.backend import SiteMap
    from reward_lens.signals.adapters import GraderAdapter

    # Structurally identical to ``interventions.base.MountHook``. Spelled again here rather than
    # imported so the runtime layer keeps no import edge into ``interventions/``; the algebra
    # depends on the runtime and not the other way round.
    MountHookT = Callable[["torch.Tensor", dict], "torch.Tensor"]


def resolve_module(model: "nn.Module", path: str) -> "nn.Module":
    """Walk a dotted module path (numeric segments index into a ``ModuleList``)."""
    module: Any = model
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _extract_hidden(adapter: "GraderAdapter", site: Site, output: Any) -> "torch.Tensor":
    """Pull the hidden-state tensor out of a module's forward output for a given site point."""
    if site.point == "attn_out":
        return adapter.extract_attn_output(output)
    if site.point == "mlp_out":
        return adapter.extract_mlp_output(output)
    if site.point == "embed":
        return output[0] if isinstance(output, tuple) else output
    # resid_post (decoder block output) and the generic case.
    return adapter.extract_layer_output(output)


def _rewrap(output: Any, new_hidden: "torch.Tensor") -> Any:
    """Put ``new_hidden`` back into the same container shape the module returned."""
    if isinstance(output, tuple):
        return (new_hidden,) + tuple(output[1:])
    return new_hidden


class CaptureMount:
    """Install capture hooks for a set of sites and collect their activations.

    Used as a context manager around a single forward pass::

        mount = CaptureMount(model, adapter, site_map, sites, positions=pos)
        with mount:
            model(**inputs)
        acts = mount.tensors     # dict[Site, Tensor]

    When ``positions`` (a ``(B,)`` tensor of per-row token indices) is given, each hook gathers just
    that position and stores ``(B, d)`` (or ``(B, d_head)`` for a head site), which is the memory
    -light v1 behaviour for the common final-token case. With ``full_sequence=True`` it stores the
    whole ``(B, T, d)``. Head sites (``head_out``) reshape the o_proj input to ``(B, T, H, d_head)``
    and slice the requested head. All handles are removed on exit, always.
    """

    def __init__(
        self,
        model: "nn.Module",
        adapter: "GraderAdapter",
        site_map: "SiteMap",
        sites: Sequence[Site],
        positions: "torch.Tensor | None" = None,
        full_sequence: bool = False,
        dtype: str | None = None,
    ):
        self.model = model
        self.adapter = adapter
        self.site_map = site_map
        self.sites = list(sites)
        self.positions = positions
        self.full_sequence = full_sequence
        self.dtype = dtype
        self.tensors: dict[Site, "torch.Tensor"] = {}
        self._handles: list[Any] = []

    def _store(self, site: Site, hidden: "torch.Tensor") -> None:
        import torch

        if self.full_sequence or self.positions is None:
            value = hidden
        else:
            batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
            pos = self.positions.to(hidden.device).clamp_(0, hidden.shape[1] - 1)
            value = hidden[batch_idx, pos]
        value = value.detach()
        if self.dtype is not None:
            value = value.to(dtype=getattr(torch, self.dtype))
        self.tensors[site] = value

    def _make_forward_hook(self, site: Site) -> Callable:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            self._store(site, _extract_hidden(self.adapter, site, output))

        return hook

    def _make_pre_hook(self, site: Site) -> Callable:
        def hook(_module: Any, args: Any) -> None:
            hidden = args[0] if isinstance(args, tuple) else args
            self._store(site, hidden)

        return hook

    def _make_head_hook(self, layer: int, heads: list[int]) -> Callable:
        import torch

        n_heads = self.site_map.n_heads

        def hook(_module: Any, args: Any) -> None:
            x = args[0] if isinstance(args, tuple) else args
            b, t, feat = x.shape
            d_head = feat // n_heads
            reshaped = x.view(b, t, n_heads, d_head)
            for head in heads:
                per_head = reshaped[:, :, head, :]  # (B, T, d_head)
                if self.full_sequence or self.positions is None:
                    value = per_head
                else:
                    batch_idx = torch.arange(b, device=x.device)
                    pos = self.positions.to(x.device).clamp_(0, t - 1)
                    value = per_head[batch_idx, pos]
                value = value.detach()
                if self.dtype is not None:
                    value = value.to(dtype=getattr(torch, self.dtype))
                self.tensors[Site(layer, "head_out", head)] = value

        return hook

    def __enter__(self) -> "CaptureMount":
        head_sites: dict[int, list[int]] = {}
        for site in self.sites:
            if site.point == "head_out":
                head_sites.setdefault(site.layer, []).append(site.head or 0)
                continue
            path = self.site_map.resolve(site)
            module = resolve_module(self.model, path)
            if site.point == "resid_pre":
                self._handles.append(module.register_forward_pre_hook(self._make_pre_hook(site)))
            else:
                self._handles.append(module.register_forward_hook(self._make_forward_hook(site)))
        for layer, heads in head_sites.items():
            path = self.site_map.resolve(Site(layer, "head_out", None))
            module = resolve_module(self.model, path)
            self._handles.append(
                module.register_forward_pre_hook(self._make_head_hook(layer, heads))
            )
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class LeafCutMount:
    """Substitute a differentiable leaf at a site so grad/hvp can differentiate w.r.t. it.

    A forward hook at the site replaces the module's hidden output with ``hidden.detach().clone()
    .requires_grad_(True)`` and stashes the leaf. The rest of the network then runs as a function of
    the leaf, so ``autograd.grad(scalar, leaf)`` gives the reward gradient at that site and a second
    ``create_graph=True`` pass gives Hessian-vector products. This is the mechanism behind the
    runtime's ``grad`` and ``hvp``. The leaf is available as ``mount.leaf`` after the
    forward.
    """

    def __init__(
        self,
        model: "nn.Module",
        adapter: "GraderAdapter",
        site_map: "SiteMap",
        site: Site,
    ):
        self.model = model
        self.adapter = adapter
        self.site_map = site_map
        self.site = site
        self.leaf: "torch.Tensor | None" = None
        self._handle: Any = None

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        hidden = _extract_hidden(self.adapter, self.site, output)
        leaf = hidden.detach().clone().requires_grad_(True)
        self.leaf = leaf
        return _rewrap(output, leaf)

    def _pre_hook(self, _module: Any, args: Any) -> Any:
        hidden = args[0] if isinstance(args, tuple) else args
        leaf = hidden.detach().clone().requires_grad_(True)
        self.leaf = leaf
        rest = tuple(args[1:]) if isinstance(args, tuple) else ()
        return (leaf,) + rest

    def __enter__(self) -> "LeafCutMount":
        path = self.site_map.resolve(self.site)
        module = resolve_module(self.model, path)
        if self.site.point == "resid_pre":
            self._handle = module.register_forward_pre_hook(self._pre_hook)
        else:
            self._handle = module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def _resolve_key(site: Site) -> Site:
    """The site the ``SiteMap`` is keyed by, which is not always the site being written.

    Head sites all live on the one head-agnostic ``o_proj`` module and the map keys it with
    ``head=None``, so a mount at ``Site(3, "head_out", 5)`` resolves through
    ``Site(3, "head_out", None)`` while the hook itself edits head 5's slice of the input.
    """
    return Site(site.layer, "head_out", None) if site.point == "head_out" else site


def _is_input_side(site: Site) -> bool:
    """Whether the mount edits the module's input rather than its output.

    ``resid_pre`` is the block's input by definition. ``head_out`` is the ``o_proj`` *input*,
    because a head's contribution is only separable before the projection mixes the heads back
    together; after ``o_proj`` there is no per-head slice left to edit. Both are pre-hooks.
    """
    return site.point in ("resid_pre", "head_out")


def mount_points(intervention: Any, signal: Any = None) -> list[tuple[Site, "MountHookT"]]:
    """Normalise anything the causal algebra produces into ``(site, hook)`` pairs.

    Three shapes reach the runtime and all three are accepted here:

    - a ``CompiledIntervention`` (anything carrying a ``mounts`` mapping), used as-is;
    - an ``Intervention`` (anything with ``compile``), compiled against ``signal`` first, which is
      what makes ``compose(steer, ablate)`` mountable: a composed intervention has no single site,
      it has a ``{site: chained hook}`` mapping, and that is the only shape a multi-site
      intervention can have;
    - a single-site object exposing ``site`` and ``apply(hidden) -> hidden``, which is the older
      minimal contract the head-rescue adapter and the test doubles use.

    Pairs come back ordered by layer, then by point, then by head, so a multi-site intervention
    mounts in the order the forward pass reaches its sites. Within one site the hooks are already
    chained in declaration order by ``ComposedIntervention``, so a recorder composed before an
    ablation still sees the activation the ablation is about to change.
    """
    mounts = getattr(intervention, "mounts", None)
    if mounts is None:
        compile_fn = getattr(intervention, "compile", None)
        if callable(compile_fn):
            mounts = compile_fn(signal).mounts
    if mounts is not None:
        ordered = sorted(
            mounts.items(),
            key=lambda kv: (
                kv[0].layer,
                str(kv[0].point),
                -1 if kv[0].head is None else kv[0].head,
            ),
        )
        return [(site, hook) for site, hook in ordered]

    site = getattr(intervention, "site", None)
    apply_fn = getattr(intervention, "apply", None)
    if site is not None and callable(apply_fn):

        def single_site(hidden: "torch.Tensor", _ctx: dict) -> "torch.Tensor":
            return cast("torch.Tensor", apply_fn(hidden))

        return [(site, single_site)]

    raise TypeError(
        f"{type(intervention).__name__} is not mountable: it carries neither a `mounts` mapping "
        f"nor a `compile` method, and it does not expose `site` + `apply(hidden)`. If it needs a "
        f"signal to compile (a weight edit does), call `.compile(signal)` yourself and pass the "
        f"resulting CompiledIntervention."
    )


@contextlib.contextmanager
def mounted_interventions(
    model: "nn.Module",
    adapter: "GraderAdapter",
    site_map: "SiteMap",
    interventions: Sequence[Any],
    signal: Any = None,
) -> Iterator[None]:
    """Mount interventions on the same hook path captures use.

    Each element is an ``Intervention``, an already-compiled ``CompiledIntervention``, or a
    single-site object with ``site`` and ``apply(hidden)``; see :func:`mount_points`. A pre-hook or
    a forward hook at each site replaces the module's activation with the intervention's output, so
    any Observable measured inside this context runs under the intervention with no change to the
    Observable. Handles are removed on exit, always. This is the shared mounting path: the same
    ``resolve_module`` + register/remove machinery serves both directions.

    ``signal`` is only consulted for interventions that still need compiling and whose compilation
    is signal-dependent. Every activation intervention in ``interventions/`` resolves its site
    through the ``SiteMap`` at mount time and discards the signal, so the default of None is
    correct for all of them. A weight edit is the exception and it has no activation mount at all.
    """
    handles: list[Any] = []

    def forward_hook(site: Site, hook: "MountHookT") -> Callable:
        def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = _extract_hidden(adapter, site, output)
            return _rewrap(output, hook(hidden, {"site": site}))

        return _hook

    def pre_hook(site: Site, hook: "MountHookT") -> Callable:
        def _hook(_module: Any, args: Any) -> Any:
            hidden = args[0] if isinstance(args, tuple) else args
            rest = tuple(args[1:]) if isinstance(args, tuple) else ()
            return (hook(hidden, {"site": site}),) + rest

        return _hook

    try:
        for iv in interventions:
            for site, hook in mount_points(iv, signal):
                module = resolve_module(model, site_map.resolve(_resolve_key(site)))
                if _is_input_side(site):
                    handles.append(module.register_forward_pre_hook(pre_hook(site, hook)))
                else:
                    handles.append(module.register_forward_hook(forward_hook(site, hook)))
        yield
    finally:
        for handle in handles:
            handle.remove()


__all__ = [
    "resolve_module",
    "CaptureMount",
    "LeafCutMount",
    "mount_points",
    "mounted_interventions",
]
